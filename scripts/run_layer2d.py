#!/usr/bin/env python3
"""
run_layer2d.py — run the 2D layer and write a report.

Usage:
    # against a synthetic set (Step 3 development / Step 5 verification)
    python3 scripts/run_layer2d.py --scene Warehouse_020 --set clean

    # against real files from the training stage (Step 8)
    python3 scripts/run_layer2d.py --scene Warehouse_020 \
        --file-b data/input/v1/Warehouse_020/detections_2d.txt

    # quick pass while iterating
    python3 scripts/run_layer2d.py --scene Warehouse_020 --set mixed --frames 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loaders import Config
from src.layers import layer_2d


def main():
    ap = argparse.ArgumentParser(description="Layer 2D — detector vs 2D ground truth.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--set", dest="error_set", default=None,
                    help="Synthetic error set name, e.g. clean, mapping_shift")
    ap.add_argument("--file-b", default=None,
                    help="Path to detections_2d.txt (overrides --set)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="Where to write the JSON report")
    args = ap.parse_args()

    cfg = Config(args.config)

    if args.file_b:
        file_b = Path(args.file_b)
        source = str(file_b)
    elif args.error_set:
        file_b = cfg.synthetic_root / args.scene / args.error_set / "detections_2d.txt"
        source = f"synthetic/{args.scene}/{args.error_set}"
    else:
        sys.exit("Give either --set or --file-b")

    gt_path = cfg.find_gt(args.scene)
    block = cfg.block_of(args.scene)

    print(f"scene      {args.scene}  [{block}]")
    print(f"gt         {gt_path}")
    print(f"file B     {file_b}")
    print(f"threshold  IoU >= {cfg.iou_threshold}, hungarian, class-blind")
    print("running    ", end="", flush=True)

    t0 = time.time()
    cm, stats = layer_2d.run(gt_path, file_b, cfg, args.frames,
                             progress=lambda n: print(".", end="", flush=True))
    dt = time.time() - t0
    print(f" {dt:.1f}s")

    print(f"\n{stats['frames']} frames, {len(stats['cameras'])} cameras, "
          f"{stats['gt_boxes']} GT boxes, {stats['pred_boxes']} predicted, "
          f"{stats['matched']} matched (mean IoU {stats['mean_iou']:.3f})")

    if stats["gt_degenerate"] or stats["pred_degenerate"] or stats["gt_malformed"]:
        print(f"\nexcluded  boxes under {stats['min_box_area_px']} px area — "
              f"not detectable, so not counted against the model:")
        if stats["gt_degenerate"]:
            print(f"          GT   {stats['gt_degenerate']}")
        if stats["pred_degenerate"]:
            print(f"          pred {stats['pred_degenerate']}")
        if stats["gt_malformed"]:
            print(f"          GT malformed (fewer than 4 values) "
                  f"{stats['gt_malformed']}")

    print(cm.render(title=f"LAYER 2D — {args.scene} [{block}]",
                    thin_threshold=cfg.thin_threshold))

    sig = cm.mapping_signature()
    if sig:
        print("\n!! SYSTEMATIC MISLABELLING in File B:")
        for gt_name, (pred_name, frac) in sig.items():
            print(f"   {gt_name} -> {pred_name} on {frac*100:.1f}% of instances")
        print("   Every instance of a class going to one wrong label is a wiring "
              "fault,\n   not perception. Check the internal -> official remap "
              "before the export.")

    out = Path(args.out) if args.out else (
        cfg.reports_root / args.scene / (args.error_set or "input") / "layer_2d.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "layer": "2d",
        "scene": args.scene,
        "block": block,
        "source": source,
        "frames": stats["frames"],
        "cameras": stats["cameras"],
        "gt_boxes": stats["gt_boxes"],
        "pred_boxes": stats["pred_boxes"],
        "pred_boxes_raw": stats["pred_boxes_raw"],
        "excluded": {
            "min_box_area_px": stats["min_box_area_px"],
            "gt_degenerate": stats["gt_degenerate"],
            "pred_degenerate": stats["pred_degenerate"],
            "gt_malformed": stats["gt_malformed"],
        },
        "matched": stats["matched"],
        "mean_iou": stats["mean_iou"],
        "iou_threshold": cfg.iou_threshold,
        "runtime_s": round(dt, 1),
        **cm.to_dict(),
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()