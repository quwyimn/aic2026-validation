"""
layer_2d.py — does the model recognise the object at all?

Compares File B (raw per-camera detections, before any 3D lifting, fusion or
tracking) against the 2D boxes already present in the ground truth.

This layer sees the detector and nothing else. That isolation is the point:
whatever it reports cannot be blamed on fusion, tracking, or a remap, because
none of those have run yet. Layer 3D measures the same objects after all of
that has happened, and the difference between the two layers is the diagnosis.
"""

from collections import defaultdict

from src.io.loaders import stream_gt_2d, read_file_b, box_area
from src.matching.assign import iou_matrix, match
from src.metrics.confusion import Confusion


def run(gt_path, file_b_path, cfg, max_frames=None, progress=None):
    """
    Returns (Confusion, stats dict).

    File B is loaded whole via pandas and indexed by (camera, frame); the GT is
    streamed. Matching then happens per camera per frame — a detection in one
    camera has nothing to say about a detection in another, so pairing across
    them would be meaningless.
    """
    class_ids = set(cfg.classes.values())
    df = read_file_b(file_b_path, class_ids=class_ids, max_frames=max_frames)
    n_raw = int(len(df))

    # Predictions are filtered by the same area rule as the GT. Filtering only
    # one side would invent a mismatch that isn't in the data.
    areas = ((df.x2 - df.x1).clip(lower=0) * (df.y2 - df.y1).clip(lower=0))
    keep = areas >= cfg.min_box_area
    n_pred_degenerate = int((~keep).sum())
    df = df[keep]

    # (camera, frame) -> (boxes, class_ids)
    index = {}
    for (cam, fid), grp in df.groupby(["camera_id", "frame_id"], sort=False):
        index[(int(cam), int(fid))] = (
            grp[["x1", "y1", "x2", "y2"]].to_numpy(),
            grp["class_id"].to_numpy(),
        )

    cm = Confusion(cfg.class_names, cfg.classes)
    stats = {
        "frames": 0,
        "cameras": set(),
        "gt_boxes": 0,
        "pred_boxes": int(len(df)),
        "pred_boxes_raw": n_raw,
        "pred_degenerate": n_pred_degenerate,
        "gt_degenerate": 0,
        "gt_malformed": 0,
        "min_box_area_px": cfg.min_box_area,
        "matched": 0,
        "iou_sum": 0.0,
    }
    seen_keys = set()

    for fid, per_cam in stream_gt_2d(gt_path, cfg.classes, max_frames,
                                     min_area=cfg.min_box_area, stats=stats):
        stats["frames"] += 1
        if progress and stats["frames"] % 1000 == 0:
            progress(stats["frames"])

        for cam, rows in per_cam.items():
            stats["cameras"].add(cam)
            gt_boxes = [r[2] for r in rows]
            gt_cls = [r[1] for r in rows]
            stats["gt_boxes"] += len(rows)

            key = (cam, fid)
            seen_keys.add(key)
            pred_boxes, pred_cls = index.get(key, ([], []))

            cost = iou_matrix(gt_boxes, pred_boxes)
            pairs, un_gt, un_pred = match(cost, cfg.iou_threshold, "iou")

            for gi, pi in pairs:
                cm.add_pair(int(gt_cls[gi]), int(pred_cls[pi]))
                stats["matched"] += 1
                stats["iou_sum"] += float(cost[gi, pi])
            for gi in un_gt:
                cm.add_miss(int(gt_cls[gi]))
            for pi in un_pred:
                cm.add_ghost(int(pred_cls[pi]))

    # Detections in File B for (camera, frame) combinations the GT never
    # mentions are ghosts too — they'd otherwise be silently dropped.
    for key, (boxes, cls) in index.items():
        if key not in seen_keys:
            for c in cls:
                cm.add_ghost(int(c))

    stats["cameras"] = sorted(stats["cameras"])
    stats["mean_iou"] = (stats["iou_sum"] / stats["matched"]
                         if stats["matched"] else float("nan"))
    del stats["iou_sum"]
    return cm, stats