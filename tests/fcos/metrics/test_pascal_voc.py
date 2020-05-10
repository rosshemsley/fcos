import pytest

import numpy as np

from fcos.metrics import compute_pascal_voc_metrics

def test_pascal_voc_1():
    ground_truth_boxes = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 3.0, 3.0]),
    ]
    predicted_boxes = ground_truth_boxes
    metrics = compute_pascal_voc_metrics([ground_truth_boxes], [predicted_boxes], [[1.0, 1.0]])

    assert metrics.true_positive_count == 2
    assert metrics.false_positive_count == 0
    assert metrics.mean_average_precision == pytest.approx(1.0)


def test_pascal_voc_2():
    ground_truth_boxes = [
        np.array([0.0, 0.0, 1.0, 1.0]),
    ]
    predicted_boxes = [
        np.array([2.0, 2.0, 3.0, 3.0]),
    ]

    metrics = compute_pascal_voc_metrics([ground_truth_boxes], [predicted_boxes], [[1.0, 0.5]])

    assert metrics.true_positive_count == 0
    assert metrics.false_positive_count == 1
    assert metrics.mean_average_precision == pytest.approx(0.0)


def test_pascal_voc_3():
    ground_truth_boxes = [
        np.array([0.0, 0.0, 1.0, 1.0]),
    ]
    predicted_boxes = []

    metrics = compute_pascal_voc_metrics([ground_truth_boxes], [predicted_boxes], [[1.0]])

    assert metrics.true_positive_count == 0
    assert metrics.false_positive_count == 0
    assert metrics.mean_average_precision == pytest.approx(0.0)