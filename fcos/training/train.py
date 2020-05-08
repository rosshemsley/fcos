import pathlib
import os

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader

from fcos.datasets import tensor_to_image, collate_fn, CityscapesData, Split
from fcos.inference import (
    compute_detections_for_tensor,
    render_detections_to_image,
    detections_from_net,
    detections_from_network_output,
)
from fcos.models import FCOS, normalize_batch

from .targets import generate_targets

logger = logging.getLogger(__name__)


def train(cityscapes_dir: pathlib.Path, writer: SummaryWriter):
    val_loader = DataLoader(
        CityscapesData(Split.VALIDATE, cityscapes_dir),
        batch_size=3,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        CityscapesData(Split.TRAIN, cityscapes_dir),
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

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    logger.info("Freezing backbone network")
    model.freeze_backbone()

    for epoch in range(100):
        logger.info(f"Starting epoch {epoch}")

        model.train()

        if epoch == 3:
            logger.info("Unfreezing backbone network")
            model.unfreeze_backbone()

        for train_idx, (x, class_labels, box_labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()

            x = x.to(device)
            batch = normalize_batch(x)
            classes, centernesses, boxes = model(batch)

            class_targets, centerness_targets, box_targets = generate_targets(
                x.shape, class_labels, box_labels, model.strides
            )

            loss = _compute_loss(classes, centernesses, boxes, class_targets, centerness_targets, box_targets)

            logging.info(
                f"Epoch: {epoch} batch {train_idx} of {len(train_loader)}, loss: {loss.item()}"
            )

            writer.add_scalar("Loss/train", loss.item(), train_idx)
            loss.backward()
            optimizer.step()

            if train_idx % 100 == 0:
                logger.info("Running validation...")

                with torch.no_grad():
                    _test_model(checkpoint, writer, model, val_loader, device)

                path = os.path.join("checkpoints", f"{checkpoint}.chkpt")
                logger.info(f"Saving checkpoint to '{path}'")
                torch.save(model.state_dict(), path)
                checkpoint += 1

        scheduler.step()


def _compute_loss(
    classes, centernesses, boxes, class_targets, centerness_targets, box_targets
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

        for batch_idx in range(batch_size):
            losses.append(class_loss(cls_view[batch_idx], cls_target[batch_idx]))
            losses.append(centerness_loss(centerness_view, centerness_target))

            mask = cls_target[batch_idx] > 0
            if mask.nonzero().shape[0] > 0:
                losses.append(box_loss(box_view[batch_idx][mask], box_target[batch_idx][mask]) / 50.0)

    return torch.stack(losses).mean()


def _test_model(checkpoint, writer, model, loader, device):
    model.eval()

    images = []
    for i, (x, class_labels, box_labels) in enumerate(loader, 0):
        if i == 10:
            break

        logging.info(f"Validation for {i}")
        img = tensor_to_image(x[0])

        x = x.to(device)
        x = normalize_batch(x)

        classes, centernesses, boxes = model(x)
        detections = detections_from_network_output(classes, centernesses, boxes)
        render_detections_to_image(img, detections[0])

        class_targets, centerness_targets, box_targets = generate_targets(
            x.shape, class_labels, box_labels, model.strides
        )

        loss = _compute_loss(classes, centernesses, boxes, class_targets, centerness_targets, box_targets)
        logging.info(f"Validation loss: {loss.item()}")

        writer.add_scalar("Loss/val", loss.item(), checkpoint)
        writer.add_image(f"fcos test {i}", img, checkpoint, dataformats="HWC")

    writer.flush()
