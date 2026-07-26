#!/usr/bin/env python3
"""
gt_to_txt.py — convert ground_truth.json to the official 11-column text format.

This is the last shared-assumption surface between the validator and the
official scorer. Everything else in Step 6 is NVIDIA's code: it parses, it
range-checks class ids, it matches by 3D IoU, it computes HOTA. The only thing
we hand it that we wrote is this file's output. So this converter is written to
be as thin and checkable as possible — it reads the GT with the validator's own
streaming loader (no second JSON parser to disagree), maps class by *name*
against config.yaml (which already carries the official ids, so there is no
internal→official remap to get wrong), and writes exactly the 11 columns the
scorer's README specifies:

    scene_id class_id object_id frame_id x y z w l h yaw

Two facts pinned from the data, not assumed:
  - yaw is Euler component index 2 (rotation is [0, 0, yaw]; vertical axis is z,
    contradicting the submission spec's "about the y-axis"). The data is the
    source of truth. Verified: GT row 0 rotation [0.0, 0.0, 2.1712...].
  - the GT already uses meters and radians, 0-indexed frames — no unit change.

The converter is testable independently of HOTA: for the `clean` fixture the
prediction file was generated from this same GT, so the two files must agree on
(scene, class, object, frame). That diff is the converter's own unit test —
see the --self-check flag.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loaders import Config, stream_gt_3d
from src.io import loaders


def scene_id_of(scene_name):
    """Warehouse_020 -> 20. Matches make_synthetic.scene_id_of."""
    digits = "".join(ch for ch in scene_name if ch.isdigit())
    if not digits:
        sys.exit(f"cannot derive scene_id from {scene_name!r}")
    return int(digits)


def _stream_gt_full(gt_path, class_table, max_frames):
    """Yield (frame_id, object_id, class_id, x, y, z, w, l, h, yaw).

    stream_gt_3d only returns location, not scale/rotation, so we read the GT
    directly here with the same ijson streaming discipline and the same schema
    resolver from loaders — one parser, one schema definition, shared with the
    validator."""
    import ijson
    schema = None
    with open(gt_path, "rb") as f:
        for frame_key, objects in ijson.kvitems(f, "", use_float=True):
            try:
                fid = int(frame_key)
            except ValueError:
                continue
            if max_frames is not None and fid >= max_frames:
                continue
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if schema is None:
                    schema = loaders.resolve_schema(obj)
                cls = obj.get(schema["class"])
                if cls not in class_table:
                    continue
                oid = obj.get(schema["object_id"])
                loc = obj.get(schema["loc3d"]) or [0, 0, 0]
                scale = obj.get(schema["bbox3d"]) or [1, 1, 1]
                rot = obj.get(schema.get("rot3d")) or [0, 0, 0]
                x, y, z = (float(v) for v in (list(loc) + [0, 0, 0])[:3])
                w, l, h = (float(v) for v in (list(scale) + [1, 1, 1])[:3])
                # yaw = Euler index 2 (z-axis). See module docstring.
                yaw = float(rot[2]) if len(rot) >= 3 else 0.0
                yield fid, oid, class_table[cls], x, y, z, w, l, h, yaw


def convert(gt_path, scene_name, class_table, out_path, max_frames):
    sid = scene_id_of(scene_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fo:
        for fid, oid, cid, x, y, z, w, l, h, yaw in _stream_gt_full(
                gt_path, class_table, max_frames):
            fo.write(f"{sid} {cid} {oid} {fid} "
                     f"{x:.4f} {y:.4f} {z:.4f} "
                     f"{w:.4f} {l:.4f} {h:.4f} {yaw:.4f}\n")
            n += 1
    return n


def self_check(out_path, pred_path):
    """The converter's unit test. For `clean`, GT-txt and the prediction share
    the same (scene, class, object, frame) keys because the fixture was built
    from this GT. Compares the key columns only — location may differ by the
    generator's noise, but identity must match exactly."""
    def keys(path):
        s = set()
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if len(p) == 11:
                    s.add((p[0], p[1], p[2], p[3]))
        return s
    gt = keys(out_path)
    pred = keys(pred_path)
    only_gt = gt - pred
    only_pred = pred - gt
    print(f"  GT rows (keys):   {len(gt)}")
    print(f"  pred rows (keys): {len(pred)}")
    print(f"  in GT not pred:   {len(only_gt)}")
    print(f"  in pred not GT:   {len(only_pred)}")
    ok = not only_gt and not only_pred
    print("  MATCH" if ok else "  MISMATCH — converter or generator disagree")
    for k in list(only_gt)[:3]:
        print(f"    only GT:   {k}")
    for k in list(only_pred)[:3]:
        print(f"    only pred: {k}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--scene", required=True, help="e.g. Warehouse_020")
    ap.add_argument("--out", default=None,
                    help="output txt (default data/gt_txt/<scene>/ground_truth.txt)")
    ap.add_argument("--frames", type=int, default=None,
                    help="truncate to first N frames (match --num_frames_to_eval)")
    ap.add_argument("--self-check", default=None,
                    help="a clean track1.txt to diff keys against")
    args = ap.parse_args()

    cfg = Config(args.config)
    gt_path = cfg.find_gt(args.scene)
    out = Path(args.out) if args.out else (
        Path("data/gt_txt") / args.scene / "ground_truth.txt")

    print(f"converting {gt_path}\n        -> {out}")
    n = convert(gt_path, args.scene, cfg.classes, out, args.frames)
    print(f"wrote {n} rows")

    if args.self_check:
        print("\nself-check against", args.self_check)
        ok = self_check(out, args.self_check)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()