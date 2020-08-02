import os

import click
import torch
import cv2
import time

from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from fcos.models import FCOS, normalize_batch
from fcos.inference import detections_from_network_output, render_detections_to_image
from fcos.datasets import CityscapesData, Split, collate_fn, tensor_to_image
from fcos.metrics import compute_metrics


@click.command()
@click.option(
    "--model-checkpoint",
    required=True,
    help="path to fcos model",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--cityscapes-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
)
def test(cityscapes_dir, model_checkpoint, output):
    """
    Test the fcos model from a given checkpoint on the cityscapes test set.
    Writes detections to the given output directory.
    """

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        print("using cpu")
        device = torch.device("cpu")

    state = torch.load(model_checkpoint)

    model = FCOS()
    model.to(device)
    model.load_state_dict(state["model"])

    loader = DataLoader(
        CityscapesData(Split.VALIDATE, cityscapes_dir),
        # CityscapesData(Split.VALIDATE, cityscapes_dir),
        # CityscapesData(Split.TEST, cityscapes_dir),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    model.eval()

    all_detections = []
    all_box_labels = []
    all_class_labels = []
    total_time_elapsed = 0.0

    for i, (x, class_labels, box_labels) in enumerate(loader, 0):
        print(f"Running detection for {i}/{len(loader)}")
        img = tensor_to_image(x[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        x = x.to(device)
        x = normalize_batch(x)

        with torch.no_grad():
            t0 = time.time()
            classes, centernesses, boxes = model(x)
            total_time_elapsed += time.time() - t0

        img_height, img_width = x.shape[2:4]

        detections = detections_from_network_output(
            img_height, img_width, classes, centernesses, boxes, model.scales, model.strides
        )
        render_detections_to_image(img, detections[0])

        path = os.path.join(output, f"img_{i}.png")
        cv2.imwrite(path, img)

        all_detections.extend(detections)
        all_box_labels.extend(box_labels)
        all_class_labels.extend(class_labels)
    print(f"Average inference time per image {total_time_elapsed / i}s")

    # Note(Ross): For some reason I have no ground truth annotations for the Cityscapes test set.
    # Therefore, this code throws an exception in that case. It works for the val set.
    metrics = compute_metrics(all_detections, all_class_labels, all_box_labels)
    print(
        f"""\
Pascal voc metrics:
    total ground truth detections: {metrics.total_ground_truth_detections}
    TP: {metrics.true_positive_count}
    FP: {metrics.false_positive_count}
    mAP: {metrics.mean_average_precision}"""
    )
