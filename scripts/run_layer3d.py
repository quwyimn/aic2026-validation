#!/usr/bin/env python3
"""
run_layer3d.py — run the 3D layer and write a report.

Usage:
    python3 scripts/run_layer3d.py --scene Warehouse_020 --set clean
    python3 scripts/run_layer3d.py --scene Warehouse_020 --set mapping_shift
    python3 scripts/run_layer3d.py --scene Warehouse_020 \
        --file-a data/input/v1/Warehouse_020/track1.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loaders import Config
from src.layers import layer_3d


def main():
    ap = argparse.ArgumentParser(description="Layer 3D — tracking output vs 3D ground truth.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--set", dest="error_set", default=None)
    ap.add_argument("--file-a", default=None,
                    help="Path to track1.txt (overrides --set)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--track-class", choices=["majority", "first"],
                    default="majority",
                    help="How a track's single class is resolved from its "
                         "per-frame labels. Match the pipeline's own scheme.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(args.config)

    if args.file_a:
        file_a = Path(args.file_a)
        source = str(file_a)
    elif args.error_set:
        file_a = cfg.synthetic_root / args.scene / args.error_set / "track1.txt"
        source = f"synthetic/{args.scene}/{args.error_set}"
    else:
        sys.exit("Give either --set or --file-a")

    gt_path = cfg.find_gt(args.scene)
    block = cfg.block_of(args.scene)

    print(f"scene      {args.scene}  [{block}]")
    print(f"gt         {gt_path}")
    print(f"file A     {file_a}")
    print(f"threshold  distance <= {cfg.dist_threshold} m, hungarian, class-blind")
    print(f"track cls  {args.track_class}")
    print("running    ", end="", flush=True)

    t0 = time.time()
    cm, stats = layer_3d.run(gt_path, file_a, cfg, args.frames,
                             track_class=args.track_class,
                             progress=lambda n: print(".", end="", flush=True))
    dt = time.time() - t0
    print(f" {dt:.1f}s")

    print(f"\n{stats['frames']} frames, {stats['gt_objects']} GT objects, "
          f"{stats['pred_objects']} predicted, {stats['matched']} matched "
          f"(mean dist {stats['mean_dist_m']:.3f} m)")
    print(f"tracks     GT {stats['gt_track_count']}, "
          f"predicted {stats['pred_track_count']} "
          f"(ratio {stats['track_ratio']:.2f})")

    if stats["flicker_tracks"]:
        print(f"!! {stats['flicker_tracks']} tracks wear more than one class "
              f"across their frames — an object's class should be stable, so "
              f"this is a defect in File A, not just a metric.")

    if stats["track_ratio"] > 1.15:
        print(f"!! predicted tracks outnumber GT tracks {stats['track_ratio']:.1f}x "
              f"— tracks may be fragmented. Classes may still read correct; "
              f"the association is what's broken.")

    print(cm.render(title=f"LAYER 3D — {args.scene} [{block}]",
                    thin_threshold=cfg.thin_threshold))

    sig = cm.mapping_signature()
    if sig:
        print("\n!! SYSTEMATIC MISLABELLING in File A:")
        for gt_name, (pred_name, frac) in sig.items():
            print(f"   {gt_name} -> {pred_name} on {frac*100:.1f}% of instances")
        print("   If layer 2D was clean for these same classes, the detector is "
              "fine\n   and the fault is in the internal -> official remap. That "
              "is the mapping\n   bug this whole two-layer design exists to catch.")

    out = Path(args.out) if args.out else (
        cfg.reports_root / args.scene / (args.error_set or "input") / "layer_3d.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "layer": "3d",
        "scene": args.scene,
        "block": block,
        "source": source,
        "track_class_method": args.track_class,
        "frames": stats["frames"],
        "gt_objects": stats["gt_objects"],
        "pred_objects": stats["pred_objects"],
        "matched": stats["matched"],
        "mean_dist_m": stats["mean_dist_m"],
        "dist_threshold_m": cfg.dist_threshold,
        "gt_track_count": stats["gt_track_count"],
        "pred_track_count": stats["pred_track_count"],
        "track_ratio": stats["track_ratio"],
        "flicker_tracks": stats["flicker_tracks"],
        "runtime_s": round(dt, 1),
        **cm.to_dict(),
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()