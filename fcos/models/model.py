import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from .backbone import Backbone

DEFAULT_MAX_DETECTIONS = 3000

CLASSES = [
    "BACKGROUND",
    "CAR",
]

class FCOS(nn.Module):
    classes = CLASSES


    def __init__(self):
        super(FCOS, self).__init__()
        self.max_detections = DEFAULT_MAX_DETECTIONS

        # Backbone uses a pretrained Resnet50
        self.backbone = Backbone()

        # Size of feature map compared to image size.
        # the image is scales[i] times bigger than feature map i.
        self.strides = [8, 16, 32, 64, 128]
        self.scales = nn.Parameter(torch.FloatTensor([8, 16, 32, 64, 128]))

        # FPN
        self.conv_layer_1_to_p3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_layer_2_to_p4 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv_layer_3_to_p5 = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.conv_p5_to_p6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.conv_p6_to_p7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        # Class assigment head
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

        # Bbox regression head
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

    def forward(self, x, box_labels=None, class_labels=None):
        """
        forward takes as input an image encoded as BCHW
        Optionally takes labels of the form
        - box_labels B[x_min, y_min, x_max, y_max] (float)
        - class_labels B[class] (int)
        If labels are provided, returns loss averaged over all batches.

        If no labels are provided, returns
        - BHW[c] class index of each box
        - BHW[p] class probability of each box
        - BHW[x_min, y_min, x_max, y_max] (box dimensions)
        """
        x = x / 255

        n_batches, _, image_height, image_width = x.shape

        # Backbone
        layer_1, layer_2, layer_3 = self.backbone(x)

        # FPN
        p5 = self.conv_layer_3_to_p5(layer_3)
        p4 = self.conv_layer_2_to_p4(layer_2) + _upsample(p5)
        p3 = self.conv_layer_1_to_p3(layer_1) + _upsample(p4)
        p6 = self.conv_p5_to_p6(p5)
        p7 = self.conv_p6_to_p7(p6)

        feature_pyramid = [p3, p4, p5, p6, p7]

        # Detection at each level of the FPN
        boxes_by_feature = []
        classes_by_feature = []

        for i, feature in enumerate(feature_pyramid):
            regression_i = self.regression_head(feature)
            classes_i = self.class_head(feature)

            # B[ltrb]HW  -> BHW[ltrb]
            regression_i = regression_i.permute(0,2,3,1).contiguous()

            # B[classes]HW  -> BHW[classes]
            classes_i = classes_i.permute(0,2,3,1).contiguous()

            boxes_i = _extract_boxes(
                regression_i,
                image_height,
                image_width,
                self.scales[i],
                self.strides[i],
            )

            boxes_by_feature.append(boxes_i)
            classes_by_feature.append(classes_i)

        if box_labels is None or class_labels is None:
            # box_labels = box_labels.to(x.device)
            # class_labels = class_labels.to(x.device)
            # boxes: B[ltrb]
            # classes: B[classes]
            boxes = torch.cat([t.view(n_batches, -1, 4) for t in boxes_by_feature], dim=1)
            classes = torch.cat([t.view(n_batches, -1, len(CLASSES)) for t in classes_by_feature], dim=1)
            scores, classes, boxes = _gather_detections(boxes, classes, self.max_detections)
            return scores, classes, boxes

        box_target, class_target = _targets(
            image_height,
            image_width,
            box_labels,
            class_labels,
            len(self.classes),
        )
        box_target = box_target.to(x.device)
        class_target = class_target.to(x.device)

        losses = []

        # Compute loss for each feature map
        for i, feature in enumerate(feature_pyramid):
            boxes_i = boxes_by_feature[i]
            classes_i = classes_by_feature[i]

            l = _box_loss(boxes_i, self.strides[i], box_target, class_target) / 100.0
            lc = _class_loss(classes_i, self.strides[i], class_target) 

            losses.append(l)
            losses.append(lc)

        loss = torch.stack(losses)
        return loss.mean()


def _box_loss(boxes, stride, box_target, class_target):
    target_view = box_target[:, ::stride, ::stride, :]
    mask = class_target[:, ::stride, ::stride] > 0
    loss = nn.L1Loss()
    v = loss(boxes[mask], target_view[mask])
    return v


def _class_loss(classes, stride, class_target):
    target_view = class_target[:, ::stride, ::stride]

    loss = nn.CrossEntropyLoss()

    inval = classes.reshape(-1, 2)
    tar = target_view.reshape(-1)


    return loss(inval, tar)

def _targets(image_height, image_width, box_labels, class_labels, num_classes):
    """
    labels: B[x_min, y_min, x_max, y_max, class]
    """
    batches = len(class_labels)

    box_targets = torch.zeros(batches, image_height, image_width, 4)
    box_classes = torch.zeros(batches, image_height, image_width, dtype=torch.int64)

    for batch in range(batches):
        widths = box_labels[batch][:,2] -  box_labels[batch][:,0]
        heights = box_labels[batch][:,3] -  box_labels[batch][:,1]
        areas = torch.mul(widths, heights)

        indices = torch.argsort(areas, dim=0, descending=True)
        for index in indices:
            bbox = box_labels[batch][index]
            box_x_min = int(bbox[0])
            box_y_min = int(bbox[1])
            box_x_max = int(bbox[2]) + 1
            box_y_max = int(bbox[3]) + 1
            box_targets[batch, box_y_min:box_y_max, box_x_min:box_x_max] = bbox
            box_classes[batch, box_y_min:box_y_max, box_x_min:box_x_max] = 1

    return box_targets, box_classes


def _feature_loss_targets(labels, features, image_height, image_width, stride):
    """
    Takes a feature map for a single layer of the FPN
    """

def _gather_detections(boxes, classes, max_detections):
    # classes: BHW[c] class index of each box
    class_scores, class_indices = torch.max(classes.sigmoid(), dim=2)

    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    # TODO(Ross): there must be a nicer way
    n_batches = boxes.shape[0]
    for i in range(n_batches):
        non_background_points = class_indices[i] > 0

        class_scores_i = class_scores[i][non_background_points]
        boxes_i = boxes[i][non_background_points]
        class_indices_i = class_indices[i][non_background_points]

        num_detections = min(class_scores_i.shape[0], max_detections)
        _, top_detection_indices = torch.topk(class_scores_i, num_detections, dim=0)

        top_boxes_i = torch.index_select(boxes_i, 0, top_detection_indices)
        top_classes_i = torch.index_select(class_indices_i, 0, top_detection_indices)
        top_scores_i = torch.index_select(class_scores_i, 0, top_detection_indices)
        
        boxes_to_keep = torchvision.ops.nms(top_boxes_i, top_scores_i, 0.6)

        top_boxes_i = top_boxes_i[boxes_to_keep]
        top_classes_i = top_classes_i[boxes_to_keep]
        top_scores_i = top_scores_i[boxes_to_keep]

        boxes_by_batch.append(top_boxes_i)
        classes_by_batch.append(top_classes_i)
        scores_by_batch.append(top_scores_i)


    top_boxes = torch.stack(boxes_by_batch, dim=0)
    top_classes = torch.stack(classes_by_batch, dim=0)
    top_scores = torch.stack(scores_by_batch, dim=0)

    return top_scores, top_classes, top_boxes


def _extract_boxes(pred_box, image_height, image_width, scale, stride):
    """
    Returns B[x_min, y_min, x_max, y_max], in image space
    """
    # force all values to be positive
    pred_box = torch.pow(pred_box, 2) * scale
    half_stride = stride / 2.0
    batches, rows, cols, _ = pred_box.shape

    y = torch.linspace(half_stride, image_height - half_stride, rows).to(pred_box.device)
    x = torch.linspace(half_stride, image_width - half_stride, cols).to(pred_box.device)

    center_y, center_x = torch.meshgrid(y, x)
    center_y = center_y.squeeze(0)
    center_x = center_x.squeeze(0)

    pred_box = pred_box.clamp(0, (image_height *8)/float(stride))

    x_min = center_x - pred_box[:, :, :, 0]
    y_min = center_y - pred_box[:, :, :, 1]
    x_max = center_x + pred_box[:, :, :, 2]
    y_max = center_y + pred_box[:, :, :, 3]


    result = torch.stack([x_min, y_min, x_max, y_max], dim=3)

    return result


def _upsample(x):
    return F.interpolate(
        x,
        size=(x.shape[2] * 2, x.shape[3] * 2),
        mode="bilinear",
        align_corners=True
    )
