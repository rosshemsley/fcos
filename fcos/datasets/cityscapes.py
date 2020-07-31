import pathlib
import logging

from torch import nn
import numpy as np
import torch
import torch.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
import cv2
from torchvision.transforms import ToPILImage

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

logger = logging.getLogger(__name__)


from enum import Enum


class Split(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATE = 3


class CityscapesData(Dataset):
    def __init__(self, split: Split, cityscapes_dir: pathlib.Path, image_transforms=None):
        v = _get_split(split)
        logger.info(f"Loading Cityscapes '{v}' dataset from '{cityscapes_dir}'")

        t = image_transforms if image_transforms is not None else []

        self.dataset = Cityscapes(
            # TODO(Ross): make this an argument
            cityscapes_dir,
            split=v,
            mode="fine",
            target_type=["polygon"],
            transform=Compose([*t, ToTensor()]),
        )

    def __len__(self) -> int:
        # return min(len(self.dataset), 10)
        return len(self.dataset)

    def __getitem__(self, idx):
        img, poly = self.dataset[idx]
        class_labels, box_labels = _poly_to_labels(img, poly)
        return img, class_labels, box_labels


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch], dim=0),
        [b[1] for b in batch],
        [b[2] for b in batch],
    )


def tensor_to_image(t) -> np.ndarray:
    """
    Return a PIL image (RGB)
    """
    img = Compose([ToPILImage(),])(t)
    return np.array(img)


def _poly_to_labels(image_tensor, poly):
    _, img_height, img_width = image_tensor.shape

    # TODO(Ross): fix this.
    h = poly["imgHeight"]
    w = poly["imgWidth"]

    scaling = img_height / h

    box_labels = []
    class_labels = []

    for obj in poly["objects"]:
        if obj["label"] == "car":
            polygon = obj["polygon"]
            min_x = min(x for x, _ in polygon) * scaling
            max_x = max(x for x, _ in polygon) * scaling
            max_y = max(y for _, y in polygon) * scaling
            min_y = min(y for _, y in polygon) * scaling

            box_labels.append(torch.FloatTensor([min_x, min_y, max_x, max_y]))
            class_labels.append(torch.IntTensor([1]))

    if len(class_labels) == 0:
        return torch.zeros((0, 1)), torch.zeros(0, 4)

    return torch.stack(class_labels), torch.stack(box_labels)


def _get_split(split_name: str) -> Split:
    if split_name is Split.TEST:
        return "test"
    elif split_name is Split.VALIDATE:
        return "val"
    elif split_name is Split.TRAIN:
        return "train"
    else:
        raise ValueError(f"unknown split kind {split_name}")
