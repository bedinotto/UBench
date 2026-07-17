"""Metric authority tests (UB-11, T2.2).

One definition of IoU and Dice, shared by trainer and benchmark, on **argmax**
predictions. The hand-computed numeric checks are the real guarantee that the
confusion-matrix derivation is correct; the module-identity check guarantees
both consumers actually route through this single authority (R5).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import codes.benchmark_models as benchmark_models
import codes.unified_training as unified_training
from codes.metrics import SegmentationMetrics, compute_segmentation_metrics


def _logits_from_argmax(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build logits whose argmax over dim=1 is exactly ``pred`` (N, H, W)."""
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


def test_hard_dice_and_iou_match_hand_computed() -> None:
    """Confusion-matrix IoU/Dice equal values worked out by hand.

    pred  = [[0, 1], [1, 1]]        target = [[0, 1], [2, 1]]   (num_classes=3)
    cm (rows=target, cols=pred) = [[1,0,0],[0,2,0],[0,1,0]]
      class0: TP=1 FP=0 FN=0 -> IoU=1.0    Dice=1.0
      class1: TP=2 FP=1 FN=0 -> IoU=2/3    Dice=0.8
      class2: TP=0 FP=0 FN=1 -> IoU=0.0    Dice=0.0
    macro over present non-background {1,2}: IoU=1/3, Dice=0.4
    """
    pred = torch.tensor([[[0, 1], [1, 1]]])
    target = torch.tensor([[[0, 1], [2, 1]]])
    m = compute_segmentation_metrics(_logits_from_argmax(pred, 3), target, 3)

    assert m["per_class_iou"] == pytest.approx([1.0, 2 / 3, 0.0])
    assert m["per_class_dice"] == pytest.approx([1.0, 0.8, 0.0])
    assert m["mean_iou"] == pytest.approx(1 / 3)
    assert m["mean_dice"] == pytest.approx(0.4)
    assert m["present_classes"] == [0, 1, 2]


def test_absent_class_excluded_from_macro() -> None:
    """A class absent from the target is not counted in the macro average.

    target contains only {0, 1}; the model predicts class 2 at one pixel.
    Class 2 must be excluded from ``present_classes`` and from the macro, which
    therefore reduces to class 1 alone.
    """
    pred = torch.tensor([[[0, 2], [1, 1]]])
    target = torch.tensor([[[0, 1], [1, 1]]])
    m = compute_segmentation_metrics(_logits_from_argmax(pred, 3), target, 3)

    # class1: TP=2 FP=0 FN=1 -> IoU=2/3, Dice=0.8
    assert 2 not in m["present_classes"]
    assert m["present_classes"] == [0, 1]
    assert m["mean_iou"] == pytest.approx(2 / 3)   # class 1 only, not (2/3 + 0)/2
    assert m["mean_dice"] == pytest.approx(0.8)


def test_background_reported_separately() -> None:
    """Background (index 0) is split out and never folded into the macro."""
    pred = torch.tensor([[[0, 1], [1, 1]]])
    target = torch.tensor([[[0, 1], [2, 1]]])
    m = compute_segmentation_metrics(_logits_from_argmax(pred, 3), target, 3)

    assert "background_iou" in m and "background_dice" in m
    assert m["background_iou"] == pytest.approx(1.0)
    assert m["background_dice"] == pytest.approx(1.0)
    # Background is perfect (1.0) but the macro is 1/3 — proof it is excluded.
    assert m["mean_iou"] != pytest.approx(m["background_iou"])


def test_trainer_and_benchmark_share_one_authority() -> None:
    """Both modules resolve the same metric class (no divergent local copies)."""
    assert unified_training.SegmentationMetrics is SegmentationMetrics
    assert benchmark_models.SegmentationMetrics is SegmentationMetrics


def test_accumulation_is_batch_invariant() -> None:
    """Trainer- and benchmark-style batched accumulation give identical values.

    The confusion matrix is additive, so evaluating N samples in one batch must
    equal evaluating them across several ``update`` calls — the property the old
    per-batch-then-average Dice lacked.
    """
    pred = torch.tensor([[[0, 1], [1, 1]], [[2, 2], [0, 0]]])
    target = torch.tensor([[[0, 1], [2, 1]], [[2, 1], [0, 0]]])
    logits = _logits_from_argmax(pred, 3)

    one_shot = compute_segmentation_metrics(logits, target, 3)

    accumulated = SegmentationMetrics(3)
    accumulated.update(logits[:1], target[:1])
    accumulated.update(logits[1:], target[1:])

    assert accumulated.compute() == one_shot
