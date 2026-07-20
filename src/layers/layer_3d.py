"""
layer_3d.py — after fuse, track and remap, is the class still right?

Compares File A (track1.txt — the 3D tracking output the pipeline actually
submits) against the 3D locations in the ground truth. Matching is by
Euclidean distance in world space, class-blind, Hungarian — the same
discipline as layer 2D, in meters instead of pixels.

This layer sees the whole pipeline at once. On its own it cannot say which
stage broke a class. That is exactly why layer 2D exists alongside it: layer
2D saw the detector in isolation, and the gap between the two layers is the
diagnosis. A fault that layer 2D reported as clean but this layer reports as
broken was introduced after the detector — in fusion, tracking, or the remap.
"""

from collections import Counter, defaultdict

from src.io.loaders import stream_gt_3d, read_file_a
from src.matching.assign import distance_matrix, match
from src.metrics.confusion import Confusion


def resolve_track_classes(df, method="majority"):
    """
    Collapse each track's per-frame class labels into one label per track.

    File A carries a class on every row, but a track is one physical object and
    must resolve to a single class. The pipeline decides this upstream; the
    validator only needs to read whatever it decided. Two schemes are supported
    so the validator can match the pipeline's own choice:

      majority — the class the track wears on the most frames
      first    — the class on the track's earliest frame

    If the pipeline is already consistent per track, every scheme agrees and
    the choice is moot. The value of resolving here is catching the case where
    it is NOT consistent — a track that flickers between classes is itself a
    defect worth surfacing.

    Returns (class_by_object, flicker) where flicker maps object_id to the
    number of distinct classes it wore, for ids that wore more than one.
    """
    class_by_object = {}
    flicker = {}
    for oid, grp in df.groupby("object_id", sort=False):
        counts = Counter(grp["class_id"].tolist())
        if len(counts) > 1:
            flicker[int(oid)] = len(counts)
        if method == "first":
            first_frame = grp["frame_id"].idxmin()
            class_by_object[int(oid)] = int(grp.loc[first_frame, "class_id"])
        else:  # majority
            class_by_object[int(oid)] = int(counts.most_common(1)[0][0])
    return class_by_object, flicker


def run(gt_path, file_a_path, cfg, max_frames=None, track_class="majority",
        progress=None):
    """
    Returns (Confusion, stats dict).

    File A is loaded whole via pandas and indexed by frame; the GT is streamed.
    Matching happens per frame, in world coordinates — unlike layer 2D there is
    no camera axis here, because File A is already fused into a single world.
    """
    class_ids = set(cfg.classes.values())
    df = read_file_a(file_a_path, class_ids=class_ids, max_frames=max_frames)

    # One class per track, matching whatever scheme the pipeline used.
    track_cls, flicker = resolve_track_classes(df, track_class)

    # frame_id -> (locations array, object_ids array)
    index = {}
    for fid, grp in df.groupby("frame_id", sort=False):
        index[int(fid)] = (grp[["x", "y", "z"]].to_numpy(),
                            grp["object_id"].to_numpy())

    cm = Confusion(cfg.class_names, cfg.classes)
    stats = {
        "frames": 0,
        "gt_objects": 0,
        "pred_objects": int(len(df)),
        "matched": 0,
        "dist_sum": 0.0,
        "gt_track_count": 0,
        "pred_track_count": int(df["object_id"].nunique()),
        "flicker_tracks": len(flicker),
    }
    gt_ids = set()
    seen_frames = set()

    for fid, rows in stream_gt_3d(gt_path, cfg.classes, max_frames):
        stats["frames"] += 1
        if progress and stats["frames"] % 1000 == 0:
            progress(stats["frames"])

        gt_locs = [r[2] for r in rows]
        gt_cls = [r[1] for r in rows]
        for r in rows:
            gt_ids.add(r[0])
        stats["gt_objects"] += len(rows)

        seen_frames.add(fid)
        pred_locs, pred_ids = index.get(fid, ([], []))

        cost = distance_matrix(gt_locs, pred_locs)
        pairs, un_gt, un_pred = match(cost, cfg.dist_threshold, "dist")

        for gi, pi in pairs:
            pred_class = track_cls.get(int(pred_ids[pi]), -1)
            cm.add_pair(int(gt_cls[gi]), pred_class)
            stats["matched"] += 1
            stats["dist_sum"] += float(cost[gi, pi])
        for gi in un_gt:
            cm.add_miss(int(gt_cls[gi]))
        for pi in un_pred:
            cm.add_ghost(track_cls.get(int(pred_ids[pi]), -1))

    # Predictions in frames the GT never mentions are ghosts too.
    for fid, (locs, ids) in index.items():
        if fid not in seen_frames:
            for oid in ids:
                cm.add_ghost(track_cls.get(int(oid), -1))

    stats["gt_track_count"] = len(gt_ids)
    stats["mean_dist_m"] = (stats["dist_sum"] / stats["matched"]
                            if stats["matched"] else float("nan"))
    del stats["dist_sum"]
    # Fragmentation shows here: many more predicted tracks than GT tracks, with
    # classes still correct. The ratio is the tell.
    stats["track_ratio"] = (stats["pred_track_count"] / stats["gt_track_count"]
                            if stats["gt_track_count"] else float("nan"))
    return cm, stats