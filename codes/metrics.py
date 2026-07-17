"""Segmentation metric authority (UB-11, T2.2).

One definition per metric, shared by the trainer and the benchmark (R5), so
cross-tool numbers are actually comparable. Before this module the trainer used
a *soft* Dice (softmax-weighted, background included) while the benchmark used a
different soft Dice and a hard IoU — the two toolchains disagreed on what "Dice"
even meant.

Definitions
-----------
Metrics are computed on **argmax** (hard) predictions, accumulated across
batches through a single ``torchmetrics.ConfusionMatrix`` — the accumulation
authority. Per-class IoU and Dice are derived from that matrix::

    IoU_c  =     TP_c / (TP_c + FP_c + FN_c)
    Dice_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)      # == per-class F1

**Why derive Dice from the confusion matrix** rather than a ``torchmetrics``
Dice class: the pinned torchmetrics (1.9.0) ships neither ``torchmetrics.Dice``
nor ``torchmetrics.segmentation.DiceScore`` (both verified absent). The
confusion-matrix derivation is exact, version-stable, and unit-checked against
hand-computed values (see ``codes/tests/test_metrics.py``).

Aggregation (M2)
----------------
* The macro average **excludes classes absent from the target** — a class the
  ground truth never contains does not count toward the mean (it is not a
  legitimate 0, it is simply not evaluable on this data).
* The **background** class (index 0 by default) is reported *separately* and is
  never folded into the macro.

ConfusionMatrix convention (verified on torchmetrics 1.9.0): ``cm[i, j]`` is the
number of pixels with true label ``i`` predicted as ``j``. Hence **row** sums
are ground-truth (target) support and **column** sums are predicted support.
"""

from __future__ import annotations

from typing import Dict

import torch
import torchmetrics


class SegmentationMetrics:
    """Accumulate a confusion matrix and derive hard IoU / Dice from it.

    Usage mirrors a ``torchmetrics`` metric: ``reset()`` → ``update(...)`` per
    batch → ``compute()`` once at the end. Both the trainer and the benchmark
    drive it identically, which is the whole point (R5).
    """

    def __init__(self, num_classes: int, device: object = "cpu",
                 background_index: int = 0) -> None:
        self.num_classes = num_classes
        self.background_index = background_index
        self.confmat = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes,
        ).to(device)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.confmat.reset()

    def update(self, logits_or_preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            logits_or_preds: either raw logits ``(N, C, H, W)`` (argmax is taken
                over dim=1) or already-argmaxed predictions ``(N, H, W)``.
            target: integer label map ``(N, H, W)``.
        """
        if logits_or_preds.dim() == target.dim() + 1:
            preds = logits_or_preds.argmax(dim=1)
        else:
            preds = logits_or_preds
        self.confmat.update(preds, target)

    def compute(self) -> Dict:
        """Return hard IoU/Dice with macro (excl. absent) and background split.

        Returns:
            Dict with keys ``mean_iou``, ``mean_dice`` (macro over present,
            non-background classes), ``background_iou``, ``background_dice``,
            ``per_class_iou``, ``per_class_dice`` (length ``num_classes``; NaN
            for a class absent from both prediction and target), and
            ``present_classes`` (classes appearing in the target).
        """
        cm = self.confmat.compute().to(torch.float64)
        tp = torch.diagonal(cm)
        target_count = cm.sum(dim=1)   # row sums  -> ground-truth support
        pred_count = cm.sum(dim=0)     # column sums -> predicted support
        fp = pred_count - tp
        fn = target_count - tp

        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        nan = torch.full_like(tp, float("nan"))
        iou = torch.where(denom_iou > 0, tp / denom_iou, nan)
        dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, nan)

        present = target_count > 0     # class appears in the target at least once
        classes = torch.arange(self.num_classes, device=cm.device)
        macro_mask = present & (classes != self.background_index)

        def _macro(values: torch.Tensor) -> float:
            sel = values[macro_mask]
            return float(sel.mean().item()) if sel.numel() > 0 else 0.0

        bg = self.background_index
        return {
            "mean_iou": _macro(iou),
            "mean_dice": _macro(dice),
            "background_iou": float(iou[bg].item()),
            "background_dice": float(dice[bg].item()),
            "per_class_iou": [float(v) for v in iou.tolist()],
            "per_class_dice": [float(v) for v in dice.tolist()],
            "present_classes": [int(c) for c in classes[present].tolist()],
        }


def compute_segmentation_metrics(logits_or_preds: torch.Tensor, target: torch.Tensor,
                                 num_classes: int, background_index: int = 0) -> Dict:
    """One-shot metrics for a single ``(logits, target)`` pair.

    Convenience wrapper around :class:`SegmentationMetrics` for callers that
    have all predictions in a single tensor. Because the confusion matrix is
    additive, this is identical to accumulating the same data in any number of
    batches (a property the tests assert).
    """
    metric = SegmentationMetrics(num_classes, device=logits_or_preds.device,
                                 background_index=background_index)
    metric.update(logits_or_preds, target)
    return metric.compute()
