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


def compute_map(
    iou_threshold: float,
    ground_truth_boxes: List[np.ndarray],
    predicted_scores: List[float],
    predicted_boxes: List[np.ndarray],
    convention: MAPConvention = MAPConvention.AUC,
) -> float:

    if convention is not MAPConvention.AUC:
        raise NotImplementedError("only AUC convention mAP is implemented currently.")

    classified_detections = []
    matched_box_indices = set(range(len(ground_truth_boxes)))

    for score, box in sorted(zip(predicted_scores, predicted_boxes), reverse=True)):
        gt_index = _find_matching_box(
            iou_threshold, ground_truth_boxes, box, matched_box_indices
        )
        if gt_index is not None:
            classified_detections.append(True)
            matched_box_indices.add(gt_index)
        else:
            classified_detections.append(False)

    return compute_auc(len(ground_truth_boxes), classified_detections)


def _find_matching_box(
    iou_threshold: float, ground_truth_boxes: List[np.ndarray], box: np.ndarray, matched_box_indices: Set[int]
) -> Optional[int]:

    best_match = None
    best_iou = 0.0

    for i, gt_box in enumerate(ground_truth_boxes):
        if i in matched_box_indices:
            continue

        iou = compute_iou(gt_box, box) 
        if iou >= iou_threshold:
            if best_match is None or iou > best_iou:
                best_match = i
                best_iou = iou

    return best_match


def voc_ap(recall, precision, use_07_metric=False):
    """
    Note(Ross): Vendored from official repo

    ap = voc_ap(recall, precision, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses  the
    VOC 07 11 point method (default: False).
    Please make shure that recall and precison are sorted by scores.
    Args:
        recall: the shape of (n,) ndarray;
        precision: the shape of (n,) ndarray;
        use_07_metric: if true, the 11 points method will be used.
    Returns:
        the float number result of average precision.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_overlaps(boxes, one_box):
    """
    iou = compute_overlaps(boxes, one_box)
    compute intersection over union of ndarray.
    The format of one_box is [xmin, ymin, xmax, ymax].
    Args:
        boxes: the (n, 4) shape ndarray, ground truth boundboxes;
        bb: the (4,) shape ndarray, detected boundboxes;
    Returns:
        a (n, ) shape ndarray.
    """
    # compute overlaps
    # intersection
    ixmin = np.maximum(boxes[:, 0], one_box[0])
    iymin = np.maximum(boxes[:, 1], one_box[1])
    ixmax = np.minimum(boxes[:, 2], one_box[2])
    iymax = np.minimum(boxes[:, 3], one_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1.) * (boxes[:, 3] -
                                                     boxes[:, 1] + 1.)
    one_box_area = (one_box[2] - one_box[0] + 1.) * (one_box[3] - one_box[1] +
                                                     1.)
    iou = inters / (one_box_area + boxes_area - inters)

    return iou

def new_voc_eval(ground_truth_boxes: List[List[np.ndarray]], boxes: List[List[np.ndarray]], iou_threshold):
    ...

def voc_eval(class_recs: dict,
             detect: dict,
             iou_thresh: float = 0.5,
             use_07_metric: bool = False):
    """
    recall, precision, ap = voc_eval(class_recs, detection,
                                [iou_thresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    Please make sure that the class_recs only have one class annotations.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    Args:
        class_recalls: recalls dict of a class
            class_recs[image_name]={'bbox': []}.
        detection: Path to annotations
            detection={'image_ids':[], bbox': [], 'confidence':[]}.
        [iou_thresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
    Returns:
        a dict of result including true_positive_number, false_positive_number,
        recall, precision and average_precision.
    Raises:
        TypeError: the data format is not np.ndarray.
    """
    # format data
    # class_rec data load
    npos = 0
    for imagename in class_recs.keys():
        if not isinstance(class_recs[imagename]['bbox'], np.ndarray):
            raise TypeError
        detected_num = class_recs[imagename]['bbox'].shape[0]
        npos += detected_num
        class_recs[imagename]['det'] = [False] * detected_num

    # detections data load
    image_ids = detect['image_ids']
    confidence = detect['confidence']
    BB = detect['bbox']
    if not isinstance(confidence, np.ndarray):
        raise TypeError
    if not isinstance(BB, np.ndarray):
        raise TypeError

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        iou_max = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            overlaps = compute_overlaps(BBGT, bb)
            iou_max = np.max(overlaps)
            iou_max_index = np.argmax(overlaps)

        if iou_max > iou_thresh:
            if not R['det'][iou_max_index]:
                tp[d] = 1.
                R['det'][iou_max_index] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    true_positive_number = tp[-1]
    false_positive_number = fp[-1]

    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    average_precision = voc_ap(recall, precision, use_07_metric)

    result = {}
    result['true_positive_number'] = true_positive_number
    result['false_positive_number'] = false_positive_number
    result['recall'] = recall
    result['precision'] = precision
    result['average_precision'] = average_precision
    return result

