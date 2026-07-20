#!/usr/bin/env python3
"""
verify_validator.py — Step 5. Prove the validator reports exactly what was
injected.

Runs both layers against every synthetic set of a scene and checks each result
against that set's injected_errors.json answer key. Prints a pass/fail table.
A single FAIL means the validator cannot be trusted on real data yet.

Usage:
    python3 scripts/verify_validator.py --scene Warehouse_020
    python3 scripts/verify_validator.py --scene Warehouse_020 --frames 500
    python3 scripts/verify_validator.py                     # all source scenes
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loaders import Config
from src.layers import layer_2d, layer_3d
from src.diagnose import verify


def run_one(cfg, scene, error_set, max_frames):
    """Run both layers on one set, return (key, report_2d, report_3d)."""
    base = cfg.synthetic_root / scene / error_set
    key_path = base / "injected_errors.json"
    if not key_path.exists():
        return None, None, None
    key = json.loads(key_path.read_text())

    gt_path = cfg.find_gt(scene)
    cm2, _ = layer_2d.run(gt_path, base / "detections_2d.txt", cfg, max_frames)
    cm3, s3 = layer_3d.run(gt_path, base / "track1.txt", cfg, max_frames)

    r2 = cm2.to_dict()
    r3 = cm3.to_dict()
    r3["track_ratio"] = s3["track_ratio"]
    r3["flicker_tracks"] = s3["flicker_tracks"]
    return key, r2, r3


def main():
    ap = argparse.ArgumentParser(description="Step 5 — verify the validator.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--scene", default=None,
                    help="One scene; default is every source scene in config")
    ap.add_argument("--frames", type=int, default=None,
                    help="Limit frames for a fast pass (full run is the real test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(args.config)
    scenes = [args.scene] if args.scene else cfg.raw["synthetic"]["source_scenes"]
    sets = list(cfg.raw["synthetic"]["error_sets"].keys())

    all_pass = True
    summary = []

    for scene in scenes:
        print(f"\n{'='*70}\nSCENE {scene}\n{'='*70}")
        for es in sets:
            key, r2, r3 = run_one(cfg, scene, es, args.frames)
            if key is None:
                print(f"\n{es}: no fixture found — skipped")
                continue
            checks = verify.verify_set(es, key, r2, r3)
            passed = all(c.ok for c in checks)
            all_pass = all_pass and passed
            mark = "PASS" if passed else "FAIL"
            print(f"\n{es}  [{mark}]")
            for c in checks:
                print(c)
            summary.append({
                "scene": scene, "error_set": es, "passed": passed,
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail}
                           for c in checks],
            })

    # --- verdict ---------------------------------------------------------
    print(f"\n{'='*70}")
    n_pass = sum(1 for s in summary if s["passed"])
    n_total = len(summary)
    print(f"RESULT: {n_pass}/{n_total} sets passed")
    if all_pass:
        print("The validator reproduces every injected error. It can be trusted\n"
              "to measure a real model — Step 8 is now meaningful.")
    else:
        print("At least one set failed. The validator is not trustworthy yet;\n"
              "the failing checks above point at what to fix. Do NOT run it on a\n"
              "real model and report the numbers until this is green.")
    print(f"{'='*70}")

    out = Path(args.out) if args.out else (cfg.reports_root / "step5_verify.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"passed": all_pass, "n_pass": n_pass, "n_total": n_total,
         "results": summary}, indent=2))
    print(f"report -> {out}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()