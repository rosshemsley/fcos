from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from fcos.vendor.pascal_voc_tools.evaluator import voc_eval

@dataclass
class PascalVOCMetrics:
    true_positive_count: int
    false_positive_count: int
    recall: float
    precision: float
    mean_average_precision: float


def compute_pasal_voc_metrics(
    ground_truth_boxes_by_image: List[List[np.ndarray]],
    predicted_boxes_by_image: List[List[np.ndarray]],
    predicted_scores_by_image: List[List[float]],
    iou_threshold=0.5,
) -> Dict[str, float]:
    if not len(ground_truth_boxes_by_image) == len(predicted_boxes_by_image) == len(predicted_scores_by_image):
        raise ValueError("expect same number of entries for each list")

    image_index_to_gt_boxes: Dict[int, np.ndarray] = {}

    for i, gt_boxes_for_image in enumerate(ground_truth_boxes_by_image):
        all_boxes = np.zeros((len(gt_boxes_for_image), 4))
        for j, box in enumerate(gt_boxes_for_image):
            all_boxes[j,:] = box
        image_index_to_gt_boxes[i] = dict(bbox=all_boxes)

    all_image_indices = []
    all_boxes = []
    all_scores = []

    for i, (boxes, scores) in enumerate(zip(predicted_boxes_by_image, predicted_scores_by_image)):
        for score, box in zip(scores, boxes):
            all_image_indices.append(i)
            all_boxes.append(box)
            all_scores.append(score)

    if len(all_boxes) == 0:
        # TODO(Ross): This is technically not correct, if there are also no ground truth labels,
        # then the mAP should be 1.0
        # We should also consider the case where there are no ground truth boxes.
        return PascalVOCMetrics(
            true_positive_count=0,
            false_positive_count=0,
            recall=[],
            precision=[],
            mean_average_precision=0.0,
        )

    detections = dict(
        image_ids=np.stack(all_image_indices),
        bbox=np.stack(all_boxes),
        confidence=np.stack(all_scores),
    )

    dct = voc_eval(image_index_to_gt_boxes, detections, iou_threshold)
    return PascalVOCMetrics(
        true_positive_count=dct["true_positive_number"],
        false_positive_count=dct["false_positive_number"],
        recall=dct["recall"],
        precision=dct["precision"],
        mean_average_precision=dct["average_precision"],
    )