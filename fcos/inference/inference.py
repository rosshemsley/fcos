import math

import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torchvision

from fcos.models import FCOS, normalize_batch

MIN_SCORE = 0.05
DEFAULT_MAX_DETECTIONS = 3000


@dataclass
class Detection:
    score: float
    object_class: int
    bbox: np.ndarray  # (min_x, min_y, max_x, max_y)


def render_detections_to_image(img: np.ndarray, detections: List[Detection]):
    for detection in detections:
        if detection.score > 0.3:
            start_point = (detection.bbox[0], detection.bbox[1])
            end_point = (detection.bbox[2], detection.bbox[3])
            img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)

    return img


def compute_detections(model: FCOS, img: np.ndarray, device) -> List[Detection]:
    """
    Take an image using opencv conventions and return a list of detections.
    """
    tensor = img
    return compute_detections_for_tensor(model, tensor, device)


def compute_detections_for_tensor(model: FCOS, x, device) -> List[Detection]:
    with torch.no_grad():
        x = x.to(device)

        batch_size = x.shape[0]
        batch = normalize_batch(x)
        classes_by_feature, centerness_by_feature, boxes_by_feature = model(batch)


def detections_from_network_output(
    img_height, img_width, classes, centernesses, boxes, scales, strides
) -> List[List[Detection]]:
    all_classes = []
    all_centernesses = []
    all_boxes = []

    n_classes = classes[0].shape[-1]
    batch_size = classes[0].shape[0]

    for feat_classes, feat_centernesses, feat_boxes, scale, stride in zip(
        classes, centernesses, boxes, scales, strides
    ):
        boxes = _boxes_from_regression(feat_boxes, img_height, img_width, scale, stride)

        all_classes.append(feat_classes.view(batch_size, -1, n_classes))
        all_centernesses.append(feat_centernesses.view(batch_size, -1))
        all_boxes.append(boxes.view(batch_size, -1, 4))

    classes_ = torch.cat(all_classes, dim=1)
    centernesses_ = torch.cat(all_centernesses, dim=1)
    boxes_ = torch.cat(all_boxes, dim=1)

    gathered_boxes, gathered_classes, gathered_scores = _gather_detections(classes_, centernesses_, boxes_)
    return detections_from_net(gathered_boxes, gathered_classes, gathered_scores)


def _boxes_from_regression(reg, img_height, img_width, scale, stride):
    """
    Returns B[x_min, y_min, x_max, y_max], in image space, given regression
    values, which represent offests (left, top, right, bottom).
    """
    half_stride = stride // 2
    _, rows, cols, _ = reg.shape

    y = torch.linspace(0, img_height - stride, rows).to(reg.device)
    x = torch.linspace(0, img_width - stride, cols).to(reg.device)

    center_y, center_x = torch.meshgrid(y, x)
    center_y = center_y.squeeze(0)
    center_x = center_x.squeeze(0)

    x_min = center_x - reg[:, :, :, 0] * stride
    y_min = center_y - reg[:, :, :, 1] * stride
    x_max = center_x + reg[:, :, :, 2] * stride
    y_max = center_y + reg[:, :, :, 3] * stride

    return torch.stack([x_min, y_min, x_max, y_max], dim=3)


def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
    """
    - BHW[c] class index of each box (int)
    - BHW[p] class probability of each box (float)
    - BHW[min_x, y_min, x_min, y_max, x_max] (box dimensions, floats)
    """
    result = []

    for batch in range(len(classes_by_batch)):
        scores = scores_by_batch[batch] if scores_by_batch is not None else None
        classes = classes_by_batch[batch]
        boxes = boxes_by_batch[batch]

        result.append(
            [
                Detection(
                    score=scores[i].item() if scores is not None else 1.0,
                    object_class=classes[i].item(),
                    bbox=boxes[i].cpu().numpy().astype(int),
                )
                for i in range(boxes.shape[0])
                if classes[i] != 0
            ]
        )

    return result


def _gather_detections(classes, centernesses, boxes, max_detections=DEFAULT_MAX_DETECTIONS):
    # classes: BHW[c] class index of each box
    class_scores, class_indices = torch.max(classes, dim=2)

    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    # TODO(Ross): there must be a nicer way
    n_batches = boxes.shape[0]
    for i in range(n_batches):
        non_background_points = class_indices[i] > 0

        class_scores_i = class_scores[i][non_background_points]
        boxes_i = boxes[i][non_background_points]
        centerness_i = centernesses[i][non_background_points]
        class_indices_i = class_indices[i][non_background_points]

        class_scores_i = class_scores_i.mul(centerness_i)

        non_minimal_points = class_scores_i > 0.05

        class_scores_i = class_scores_i[non_minimal_points]
        boxes_i = boxes_i[non_minimal_points]
        centerness_i = centerness_i[non_minimal_points]
        class_indices_i = class_indices_i[non_minimal_points]

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

    return boxes_by_batch, classes_by_batch, scores_by_batch
