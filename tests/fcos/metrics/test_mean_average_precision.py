import pytest
import numpy as np

from fcos.metrics import intersection_over_union, compute_auc

def test_iou():
    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([1.0, 1.0, 2.0, 2.0])    
    assert intersection_over_union(box_1, box_2) == pytest.approx(0.0)

    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([0.0, 0.0, 1.0, 1.0])
    assert intersection_over_union(box_1, box_2) == pytest.approx(1.0)

    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([0.0, 0.5, 1.0, 1.5])
    assert intersection_over_union(box_1, box_2) == pytest.approx(1/3.0)


def test_auc():
    """
    ground truth = 2

    True     1/1     1.0
    False    1/2     1.0
    True     2/3     0.666
    False    2/4     0.666
    False    2/5     0.666
    True     3/6     0.5
    False    3/7     0.5

    So AUC mAP is  (1 + 1 + 0.666 + 0.666 + 0.666 + 0.5 + 0.5)/2 = 5/7 = 2
    """
    values = [
        True,
        False,
        True,
        True,
    ]

    total_gt_box = 2

    compute_auc()