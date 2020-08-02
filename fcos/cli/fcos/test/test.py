import os

import click
import torch
import cv2
from torch.utils.data import DataLoader

from torchvision.transforms import Resize

from fcos.models import FCOS, normalize_batch
from fcos.inference import detections_from_network_output, render_detections_to_image
from fcos.datasets import CityscapesData, Split, collate_fn, tensor_to_image


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
        # CityscapesData(Split.TEST, cityscapes_dir, image_transforms=[Resize(512)]),
        CityscapesData(Split.TEST, cityscapes_dir),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    model.eval()

    for i, (x, class_labels, box_labels) in enumerate(loader, 0):
        print(f"Running detection for {i}/{len(loader)}")
        img = tensor_to_image(x[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        x = x.to(device)
        x = normalize_batch(x)

        with torch.no_grad():
            classes, centernesses, boxes = model(x)

        img_height, img_width = x.shape[2:4]

        detections = detections_from_network_output(
            img_height, img_width, classes, centernesses, boxes, model.scales, model.strides
        )
        render_detections_to_image(img, detections[0])

        path = os.path.join(output, f"img_{i}.png")
        cv2.imwrite(path, img)
