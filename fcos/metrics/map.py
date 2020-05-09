from typing import List, Optional, Set

import numpy as np
from enum import Enum


class MAPConvention:
    AUC = 1
    COCO_MAP = 2
    PASCAL_VOC = 3


def compute_iou(box_1: np.ndarray, box_2: np.ndarray) -> float:
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


# def compute_map(
#     iou_threshold: float,
#     ground_truth_boxes: List[np.ndarray],
#     predicted_scores: List[float],
#     predicted_boxes: List[np.ndarray],
#     convention: MAPConvention = MAPConvention.AUC,
# ) -> float:

#     if convention is not MAPConvention.AUC:
#         raise NotImplementedError("only AUC convention mAP is implemented currently.")

#     classified_detections = []
#     matched_box_indices = set(range(len(ground_truth_boxes)))

#     for score, box in sorted(zip(predicted_scores, predicted_boxes), reverse=True)):
#         gt_index = _find_matching_box(
#             iou_threshold, ground_truth_boxes, box, matched_box_indices
#         )
#         if gt_index is not None:
#             classified_detections.append(True)
#             matched_box_indices.add(gt_index)
#         else:
#             classified_detections.append(False)

#     return compute_auc(len(ground_truth_boxes), classified_detections)


# def _find_matching_box(
#     iou_threshold: float, ground_truth_boxes: List[np.ndarray], box: np.ndarray, matched_box_indices: Set[int]
# ) -> Optional[int]:

#     best_match = None
#     best_iou = 0.0

#     for i, gt_box in enumerate(ground_truth_boxes):
#         if i in matched_box_indices:
#             continue

#         iou = compute_iou(gt_box, box) 
#         if iou >= iou_threshold:
#             if best_match is None or iou > best_iou:
#                 best_match = i
#                 best_iou = iou

#     return best_match


# def compute_overlaps(boxes, one_box):
#     """
#     iou = compute_overlaps(boxes, one_box)
#     compute intersection over union of ndarray.
#     The format of one_box is [xmin, ymin, xmax, ymax].
#     Args:
#         boxes: the (n, 4) shape ndarray, ground truth boundboxes;
#         bb: the (4,) shape ndarray, detected boundboxes;
#     Returns:
#         a (n, ) shape ndarray.
#     """
#     # compute overlaps
#     # intersection
#     ixmin = np.maximum(boxes[:, 0], one_box[0])
#     iymin = np.maximum(boxes[:, 1], one_box[1])
#     ixmax = np.minimum(boxes[:, 2], one_box[2])
#     iymax = np.minimum(boxes[:, 3], one_box[3])
#     iw = np.maximum(ixmax - ixmin + 1., 0.)
#     ih = np.maximum(iymax - iymin + 1., 0.)
#     inters = iw * ih

#     # union
#     boxes_area = (boxes[:, 2] - boxes[:, 0] + 1.) * (boxes[:, 3] -
#                                                      boxes[:, 1] + 1.)
#     one_box_area = (one_box[2] - one_box[0] + 1.) * (one_box[3] - one_box[1] +
#                                                      1.)
#     iou = inters / (one_box_area + boxes_area - inters)

#     return iou