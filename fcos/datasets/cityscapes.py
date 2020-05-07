from torch import nn
import numpy as np
import torch
import torch.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
import cv2

import importlib
import PIL.Image
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


dataset = Cityscapes(
    "/home/ross/datasets/CS",
    split="train",
    mode="fine",
    target_type=["polygon"],
    transform=Compose([Resize(256), ToTensor(),]),
)


def tensor_to_image(t) -> np.ndarray:
    """
    Return an opencv convention image (BGR)
    """
    img = Compose([ToPILImage(),])(t)

    return np.array(img)


class CityscapesData(Dataset):
    def __init__(self, dataset=None):
        if dataset is None:
            self.dataset = Cityscapes(
                # TODO(Ross): make this an argument
                "/home/ross/datasets/CS",
                split="train",
                mode="fine",
                target_type=["polygon"],
                transform=Compose([Resize(512), ToTensor(),]),
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, poly = self.dataset[idx]
        class_labels, box_labels = _poly_to_labels(img, poly)
        return img, class_labels, box_labels


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


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch], dim=0),
        [b[1] for b in batch],
        [b[2] for b in batch],
    )
