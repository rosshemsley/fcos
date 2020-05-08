from typing import List, Optional, Set

import numpy as np
from enum import Enum


class MAPConvention:
    AUC = 1
    COCO_MAP = 2
    PASCAL_VOC = 3


def intersection_over_union(box_1: np.ndarray, box_2: np.ndarray) -> float:
    """
    Takes bounding boxes using the convention of [x_min, y_min, x_max, y_max]
    """
    x_min = max(box_1[0], box_2[0])
    y_min = max(box_1[1], box_2[1])
    x_max = min(box_1[2], box_2[2])
    y_max = min(box_1[3], box_2[3])

    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection_area = (x_max - x_min) * (y_max - y_min)

    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    return intersection_area / float(area_1 + area_2 - intersection_area)


def mean_average_precision(
    iou_threshold: float,
    ground_truth_boxes: List[np.ndarray],
    predicted_scores: List[float],
    predicted_boxes: List[np.ndarray],
    convention: MAPConvention = MAPConvention.AUC,
) -> float:

    if convention is not MAPConvention.AUC:
        raise NotImplementedError("only AUC convention mAP is implemented currently.")

    matched_box_indices = set(range(len(ground_truth_boxes)))

    true_positives = 0
    previous_precision = 0.0
    current_precision = 0.0

    total = 0.0
    for i, (score, box) in enumerate(sorted(zip(predicted_scores, predicted_boxes), reverse=True)):
        gt_index = _find_matching_ground_truth_box(
            iou_threshold, ground_truth_boxes, box, matched_box_indices
        )
        if gt_index is not None:
            true_positives += 1
            matched_box_indices.add(gt_index)

        precision = true_positives / float(i)

        if precision > previous_precision:
            current_precision = precision

        total += current_precision
        previous_precision = precision

    return total / len(ground_truth_boxes)


def _find_matching_ground_truth_box(
    iou_threshold: float, ground_truth_boxes: List[np.ndarray], box: np.ndarray, matched_box_indices: Set[int]
) -> Optional[int]:

    for i, gt_box in enumerate(ground_truth_boxes):
        if i in matched_box_indices:
            continue

        if intersection_over_union(gt_box, box) >= iou_threshold:
            return i

    return None
