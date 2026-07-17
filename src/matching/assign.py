"""
assign.py — pairing predictions to ground truth.

One rule governs this whole module: matching is class-blind.

Pairing is decided on geometry alone, and only afterwards are the two classes
read off and compared. If matching were class-aware, a Person labelled
Forklift would fail to pair with its own ground truth and be counted as a miss
plus a ghost — the misclassification, the very thing being measured, would
vanish from the report.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt_boxes, pred_boxes):
    """
    IoU between every GT box and every predicted box.

    Boxes are [x1, y1, x2, y2] in pixels. Returns (n_gt, n_pred).
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pred_boxes, dtype=float)

    x1 = np.maximum(g[:, None, 0], p[None, :, 0])
    y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2])
    y2 = np.minimum(g[:, None, 3], p[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    union = area_g[:, None] + area_p[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def distance_matrix(gt_locs, pred_locs):
    """Euclidean distance in world space, in meters. Returns (n_gt, n_pred)."""
    if len(gt_locs) == 0 or len(pred_locs) == 0:
        return np.zeros((len(gt_locs), len(pred_locs)))
    g = np.asarray(gt_locs, dtype=float)
    p = np.asarray(pred_locs, dtype=float)
    return np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)


def match(cost, threshold, mode):
    """
    Optimal one-to-one assignment, then drop pairs that fail the threshold.

    Hungarian, not greedy. Greedy takes the best pair first and lets it block
    a better global arrangement; with objects standing close together — which
    is most of a warehouse — that produces pairings that are locally sensible
    and globally wrong, and the resulting confusion cells are fiction.

    mode 'iou'  : higher is better, keep pairs >= threshold
    mode 'dist' : lower is better,  keep pairs <= threshold

    Returns (pairs, unmatched_gt, unmatched_pred) as index lists.
    """
    n_gt, n_pred = cost.shape
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    if mode == "iou":
        rows, cols = linear_sum_assignment(-cost)
        keep = [(r, c) for r, c in zip(rows, cols) if cost[r, c] >= threshold]
    elif mode == "dist":
        rows, cols = linear_sum_assignment(cost)
        keep = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= threshold]
    else:
        raise ValueError(f"unknown mode: {mode}")

    matched_gt = {r for r, _ in keep}
    matched_pred = {c for _, c in keep}
    return (keep,
            [i for i in range(n_gt) if i not in matched_gt],
            [j for j in range(n_pred) if j not in matched_pred])