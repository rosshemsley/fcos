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
from torchvision.transforms import (RandomResizedCrop, RandomHorizontalFlip, Normalize, RandomErasing, Resize, ToTensor, RandomAffine, Compose, ColorJitter)


dataset = Cityscapes("/home/ross/datasets/CS", 
    split="train",
    mode="fine",
    target_type=["polygon"],
    transform=Compose([
        Resize(256),
        ToTensor(),
    ])
)

def tensor_to_image(t) -> np.ndarray:
    """
    Return an opencv convention image (BGR)
    """
    img = Compose([
        ToPILImage(),
    ])(t)

    arr = np.array(img)
    result = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return result


def image_to_tensor(t):
    ...
    # img = Compose([
    #     ToPILImage(),
    # ])(t)


# def tensor_to_image(t, labels=None):
#     img = Compose([
#         ToPILImage(),
#     ])(t)

#     # return img
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#     if labels:
#         h = labels["imgHeight"]
#         w = labels["imgWidth"]

#         scaling = min(img.shape[:2])/min(h,w)

#         for obj in labels["objects"]:
#             if obj["label"] == "car":
#                 polygon = obj["polygon"]
#                 min_x = int(min(x for x, _ in polygon) * scaling)
#                 max_x = int(max(x for x, _ in polygon) * scaling)
#                 max_y = int(max(y for _, y in polygon) * scaling)
#                 min_y = int(min(y for _, y in polygon) * scaling)
#                 img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0,0,0),1 )

#     # open_cv_image = np.array(img)
#     # open_cv_image = open_cv_image[:, :, ::-1].copy() 
#     img = img[:,:,[2,1,0]]
#     return PIL.Image.fromarray(img)

#     # return open_cv_image


class CityscapesData(Dataset):
    def __init__(self, dataset=None):
        if dataset is None:
            self.dataset = Cityscapes("/home/ross/datasets/CS", 
                split="train",
                mode="fine",
                target_type=["polygon"],
                transform=Compose([
                    Resize(512),
                    ToTensor(),
                ])
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, poly = self.dataset[idx]
        box_labels, class_labels = _poly_to_labels(img, poly)
        return img, box_labels, class_labels


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
        return torch.zeros(0, 4), torch.zeros(0, 1)
    return torch.stack(box_labels), torch.stack(class_labels)


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch], dim=0), [b[1] for b in batch], [b[2] for b in batch])
