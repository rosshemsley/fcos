import pathlib
import os
import logging

import cv2
import math
import shapecheck
from typing import List
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader

from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    RandomErasing,
    Resize,
    ToTensor,
    RandomAffine,
    Compose,
    ColorJitter,
)

from fcos.datasets import tensor_to_image, collate_fn, CityscapesData, Split
from fcos.inference import (
    compute_detections_for_tensor,
    render_detections_to_image,
    detections_from_net,
    detections_from_network_output,
)
from fcos.models import FCOS, normalize_batch
from fcos.metrics import compute_metrics

from .targets import generate_targets

logger = logging.getLogger(__name__)


def train(cityscapes_dir: pathlib.Path, writer: SummaryWriter):
    val_loader = DataLoader(
        CityscapesData(Split.VALIDATE, cityscapes_dir, image_transforms=[Resize(512)]),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        CityscapesData(Split.TRAIN, cityscapes_dir, image_transforms=[Resize(512)]),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    if torch.cuda.is_available():
        logger.info("Using Cuda")
        device = torch.device("cuda")
    else:
        logger.warning("Cuda not available, falling back to cpu")
        device = torch.device("cpu")

    model = FCOS()
    model.to(device)
    checkpoint = 0

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, 10000):
        logger.info(f"Starting epoch {epoch}")

        for batch_index, (x, class_labels, box_labels) in enumerate(train_loader, 0):
            model.train()
            optimizer.zero_grad()

            x = x.to(device)
            batch = normalize_batch(x)
            classes, centernesses, boxes = model(batch)

            class_targets, centerness_targets, box_targets = generate_targets(
                x.shape, class_labels, box_labels, model.strides
            )

            loss = _compute_loss(
                model.strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
            )
            logging.info(f"Epoch: {epoch}, batch: {batch_index}/{len(train_loader)}, loss: {loss.item()}")

            writer.add_scalar("Loss/train", loss.item(), batch_index)
            loss.backward()
            optimizer.step()

        logger.info("Running validation...")

        with torch.no_grad():
            _test_model(checkpoint, writer, model, val_loader, device)

        path = os.path.join(writer.log_dir, f"fcos_{checkpoint}.chkpt")
        logger.info(f"Saving checkpoint to '{path}'")
        state = dict(
            model=model.state_dict(),
            epoch=epoch,
            batch_index=batch_index,
            optimizer_state=optimizer.state_dict(),
        )
        torch.save(state, path)
        checkpoint += 1


def _compute_loss(
    strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
) -> torch.Tensor:
    batch_size = classes[0].shape[0]
    num_classes = classes[0].shape[-1]

    class_loss = torch.nn.CrossEntropyLoss()
    box_loss = torch.nn.L1Loss()
    centerness_loss = torch.nn.BCELoss()

    losses = []
    device = classes[0].device

    for feature_idx in range(len(classes)):
        cls_target = class_targets[feature_idx].to(device).view(batch_size, -1)
        centerness_target = centerness_targets[feature_idx].to(device).view(batch_size, -1)
        box_target = box_targets[feature_idx].to(device).view(batch_size, -1, 4)

        cls_view = classes[feature_idx].view(batch_size, -1, num_classes)
        box_view = boxes[feature_idx].view(batch_size, -1, 4)
        centerness_view = centernesses[feature_idx].view(batch_size, -1)

        losses.append(centerness_loss(centerness_view, centerness_target))
        ls = class_loss(cls_view.view(-1, num_classes), cls_target.view(-1))
        losses.append(ls)

        for batch_idx in range(batch_size):
            mask = cls_target[batch_idx] > 0

            if mask.nonzero().sum() > 0:
                l = box_loss(box_view[batch_idx][mask], box_target[batch_idx][mask]) * strides[feature_idx]
                losses.append(l)

    return torch.stack(losses).mean()


def _test_model(checkpoint, writer, model, loader, device):
    model.eval()

    all_detections = []
    all_box_labels = []
    all_class_labels = []

    images = []

    for i, (x, class_labels, box_labels) in enumerate(loader, 0):
        logging.info(f"Validation for {i}")
        img = tensor_to_image(x[0])

        x = x.to(device)
        x = normalize_batch(x)

        classes, centernesses, boxes = model(x)
        img_height, img_width = x.shape[2:4]
        detections = detections_from_network_output(
            img_height, img_width, classes, centernesses, boxes, model.scales, model.strides
        )
        render_detections_to_image(img, detections[0])
        _render_targets_to_image(img, box_labels[0])

        class_targets, centerness_targets, box_targets = generate_targets(
            x.shape, class_labels, box_labels, model.strides
        )
        if i == 0:
            for j in range(len(classes)):
                writer.add_image(f"class {i} feat {j}", classes[j][0][:, :, 1], checkpoint, dataformats="HW")
                writer.add_image(
                    f"class target {i} feat {j}", class_targets[j][0], checkpoint, dataformats="HW"
                )
                writer.add_image(f"centerness {i} feat {j}", centernesses[j][0], checkpoint, dataformats="HW")
                writer.add_image(
                    f"centerness target {i} feat {j}", centerness_targets[j][0], checkpoint, dataformats="HW"
                )

        loss = _compute_loss(
            model.strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
        )
        logging.info(f"Validation loss: {loss.item()}")

        writer.add_scalar("Loss/val", loss.item(), checkpoint)

        images.append(img)
        all_detections.extend(detections)
        all_box_labels.extend(box_labels)
        all_class_labels.extend(class_labels)

    grid = _image_grid(images[0:24], 3, 2048)
    writer.add_image(f"fcos test {i}", grid, checkpoint, dataformats="HWC")

    metrics = compute_metrics(all_detections, all_class_labels, all_box_labels)
    logging.info(
        f"Pascal voc metrics: TP: {metrics.true_positive_count}, FP: {metrics.false_positive_count}, mAP: {metrics.mean_average_precision}, total gt: {metrics.total_ground_truth_detections}"
    )
    writer.add_scalar("Metrics/mAP", metrics.mean_average_precision, checkpoint)

    writer.flush()


def _image_grid(images: List[np.ndarray], images_per_row: int, image_width: int) -> np.ndarray:
    max_image_width = int(image_width / images_per_row)
    images_per_col = int(math.ceil(len(images) / images_per_row))

    rescale = min(max_image_width / image.shape[1] for image in images)

    max_height = max(int(image.shape[0] * rescale) for image in images)

    image_height = images_per_col * max_height
    result = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        resized_image = cv2.resize(img, (int(img.shape[1] * rescale), int(img.shape[0] * rescale)))

        r, c = divmod(i, images_per_row)
        c *= max_image_width
        r *= max_height

        h = resized_image.shape[0]
        w = resized_image.shape[1]

        result[r : r + h, c : c + w] = resized_image

    return result


@shapecheck.check_args(box_labels=("N", ("min_x", "min_y", "max_x", "max_y")))
def _render_targets_to_image(img: np.ndarray, box_labels: torch.Tensor):
    for i in range(box_labels.shape[0]):
        start_point = (int(box_labels[i][0].item()), int(box_labels[i][1].item()))
        end_point = (int(box_labels[i][2].item()), int(box_labels[i][3].item()))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

    return img
