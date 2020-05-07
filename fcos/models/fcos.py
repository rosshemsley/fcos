import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision.transforms import Normalize
from typing import List, Tuple

from .backbone import Backbone

CLASSES = [
    "BACKGROUND",
    "CAR",
]


class FCOS(nn.Module):
    classes = CLASSES
    strides = [8, 16, 32, 64, 128]

    def __init__(self):
        super(FCOS, self).__init__()

        self.backbone = Backbone()
        self.scales = nn.Parameter(torch.FloatTensor([8, 16, 32, 64, 128]))

        # Feature Pyramid Network
        self.layer_1_to_p3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.layer_2_to_p4 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.layer_3_to_p5 = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.p5_to_p6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.p6_to_p7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        self.class_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(CLASSES), kernel_size=3, padding=1),
        )

        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.FloatTensor) -> List[Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Takes as input a batch of images encoded as BCHW, which have been normalized using
        the function `fcos.models.normalize_batch`.

        Returns a list of tuples, where each entry in the list represents one of the levels of the five
        feature maps. The tuple contains respectively, 
        - A float tensor indexed as BHWC, where C represents the 1-hot encoded class labels.
        - A float tensor indexed as BHW[x_min, y_min, x_max, y_max], where the values correspond
          directly with the input tensor, x
        """
        _, _, img_height, img_width = x.shape

        layer_1, layer_2, layer_3 = self.backbone(x)

        p5 = self.layer_3_to_p5(layer_3)
        p4 = self.layer_2_to_p4(layer_2) + _upsample(p5)
        p3 = self.layer_1_to_p3(layer_1) + _upsample(p4)
        p6 = self.p5_to_p6(p5)
        p7 = self.p6_to_p7(p6)

        feature_pyramid = [p3, p4, p5, p6, p7]

        classes_by_feature = []
        boxes_by_feature = []

        for scale, stride, feature in zip(self.scales, self.strides, feature_pyramid):
            classes = self.class_head(feature)
            reg = self.regression_head(feature)

            # B[C]HW  -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            reg = reg.permute(0, 2, 3, 1).contiguous()

            boxes = _boxes_from_regression(reg, img_height, img_width, scale, stride)
            boxes_by_feature.append(boxes)
            classes_by_feature.append(classes)

        return classes_by_feature, boxes_by_feature


def normalize_batch(x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Given a tensor representing a batch of unnormalized B[RGB]HW images,
    where RGB are floating point values in the range 0 to 255, prepare the tensor for the
    FCOS network. This has been defined to match the backbone resnet50 network.
    See https://pytorch.org/docs/master/torchvision/models.html for more details.
    """
    f = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for b in range(x.shape[0]):
        f(x[b])

    return x


def _boxes_from_regression(reg, img_height, img_width, scale, stride):
    """
    Returns B[x_min, y_min, x_max, y_max], in image space, given regression
    values, which represent offests (left, top, right, bottom).
    """
    # Note(Ross): we square to ensure all values are positive
    reg = torch.pow(reg * scale, 2)
    half_stride = stride / 2.0
    _, rows, cols, _ = reg.shape

    y = torch.linspace(half_stride, img_height - half_stride, rows).to(reg.device)
    x = torch.linspace(half_stride, img_width - half_stride, cols).to(reg.device)

    center_y, center_x = torch.meshgrid(y, x)
    center_y = center_y.squeeze(0)
    center_x = center_x.squeeze(0)

    x_min = center_x - reg[:, :, :, 0]
    y_min = center_y - reg[:, :, :, 1]
    x_max = center_x + reg[:, :, :, 2]
    y_max = center_y + reg[:, :, :, 3]

    return torch.stack([x_min, y_min, x_max, y_max], dim=3)


def _upsample(x):
    return F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear", align_corners=True)
