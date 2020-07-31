import pytest
import numpy as np

from fcos.metrics import compute_iou


def test_iou():
    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([1.0, 1.0, 2.0, 2.0])
    assert compute_iou(box_1, box_2) == pytest.approx(0.0)

    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_iou(box_1, box_2) == pytest.approx(1.0)

    box_1 = np.array([0.0, 0.0, 1.0, 1.0])
    box_2 = np.array([0.0, 0.5, 1.0, 1.5])
    assert compute_iou(box_1, box_2) == pytest.approx(1 / 3.0)
