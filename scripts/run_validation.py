#!/usr/bin/env python3
"""
run_validation.py — Step 6. Cross-reference the two layers into one verdict.

The real value of the whole system: not "something is wrong" but *which stage*.
The two layers are run, their reports handed to diagnose(), and the verdict —
MODEL / MAPPING / PIPELINE / CLEAN — printed.

Two modes:

  --verify-diagnosis   (default)  Run both layers on the diagnostic fixtures of
                                  every source scene, then assert diagnose()
                                  returns the expected verdict for each. This is
                                  the Step-6 analogue of verify_validator.py:
                                  it proves the cross-reference logic before it
                                  is ever pointed at a real model. Run at full
                                  frames — short runs hide density effects.

  --input <version>               Real run. For each scene, read File A + File B
                                  from data/input/<version>/, run both layers,
                                  diagnose, and print CLEAN and SEEN as separate
                                  blocks (never averaged), macro-average first.

Usage:
    python3 scripts/run_validation.py                                  # verify, full
    python3 scripts/run_validation.py --scene Warehouse_020 --frames 500
    python3 scripts/run_validation.py --input v1_option_d
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loaders import Config
from src.layers import layer_2d, layer_3d
from src.diagnose import verdict as vd


# The verdict each diagnostic fixture must produce. 'mixed' is a compound fault
# with no single correct bucket, so it is reported but not asserted.
EXPECTED = {
    "clean":                 "CLEAN",
    "class_swap":            "MODEL",
    "mapping_shift":         "MAPPING",
    "position_shift":        "CLEAN",      # mild, under the 1m match threshold
    "position_shift_severe": "PIPELINE",
    "deletion":              "MODEL",      # miss shows in 2D -> detector's fault
    "phantom":               "MODEL",      # ghost shows in 2D -> detector's fault
    "fragmentation":         "PIPELINE",
    "mixed":                 None,
}


def thresholds(cfg):
    """Read the diagnosis thresholds from config if present, else fall back to
    the fixture-calibrated defaults. Config stays the single source of truth."""
    d = cfg.raw.get("diagnosis", {})
    return {
        "tau_conf": d.get("confusion_fraction", vd.TAU_CONF),
        "tau_loss": d.get("loss_fraction", vd.TAU_LOSS),
        "tau_ratio": d.get("track_ratio", vd.TAU_RATIO),
    }


def run_layers(cfg, gt_path, file_a, file_b, max_frames):
    """Run both layers, return (r2d, r3d) as diagnose() expects them —
    identical wiring to verify_validator.run_one so the two stay in lockstep."""
    cm2, _ = layer_2d.run(gt_path, file_b, cfg, max_frames)
    cm3, s3 = layer_3d.run(gt_path, file_a, cfg, max_frames)
    r2 = cm2.to_dict()
    r3 = cm3.to_dict()
    r3["track_ratio"] = s3["track_ratio"]
    r3["flicker_tracks"] = s3["flicker_tracks"]
    return r2, r3


# --- mode 1: verify the diagnosis on synthetic fixtures ---------------------

def verify_diagnosis(cfg, scenes, max_frames, tau):
    all_pass = True
    summary = []
    for scene in scenes:
        print(f"\n{'='*70}\nSCENE {scene}\n{'='*70}")
        base_scene = cfg.synthetic_root / scene
        gt_path = cfg.find_gt(scene)
        for es, expect in EXPECTED.items():
            base = base_scene / es
            if not (base / "injected_errors.json").exists():
                print(f"\n{es}: no fixture found — skipped")
                continue
            r2, r3 = run_layers(cfg, gt_path, base / "track1.txt",
                                base / "detections_2d.txt", max_frames)
            v = vd.diagnose(r2, r3, **tau)
            if expect is None:
                mark = "----"
                ok = True
                note = "(compound; reported, not asserted)"
            else:
                ok = v.verdict == expect
                all_pass = all_pass and ok
                mark = "PASS" if ok else "FAIL"
                note = f"expected {expect}"
            print(f"\n{es}  [{mark}]  -> {v.verdict}  {note}")
            print(f"     {v.reason}")
            for o in v.observations:
                print(f"       · {o}")
            summary.append({"scene": scene, "error_set": es,
                            "expected": expect, "got": v.verdict,
                            "passed": ok, "reason": v.reason,
                            # observations carry every signal both layers saw,
                            # not just the one the verdict names. On a compound
                            # fault the verdict under-reports by design, so the
                            # dashboard must show these beside it.
                            "observations": v.observations,
                            "evidence": v.evidence,
                            # Full layer reports, so the dashboard renders the
                            # confusion matrices from the same numbers the
                            # verdict was derived from — never recomputed, never
                            # a second source that can drift out of agreement.
                            "report_2d": r2,
                            "report_3d": r3})

    print(f"\n{'='*70}")
    asserted = [s for s in summary if s["expected"] is not None]
    n_pass = sum(1 for s in asserted if s["passed"])
    print(f"RESULT: {n_pass}/{len(asserted)} diagnostic verdicts correct")
    if all_pass:
        print("The cross-reference maps every injected fault to the right\n"
              "stage — MODEL / MAPPING / PIPELINE. Step 6 can be trusted on a\n"
              "real model; Step 8 is now meaningful.")
    else:
        print("At least one verdict is wrong. The diagnosis logic — not the\n"
              "layers — is at fault; the FAIL rows above point at which\n"
              "signature was misread. Do NOT ship a verdict until this is green.")
    print(f"{'='*70}")
    return all_pass, summary


# --- mode 2: run for real, CLEAN and SEEN as separate blocks ----------------

def _input_root(cfg, version):
    root = getattr(cfg, "input_root", None)
    if root is None:
        root = Path(cfg.reports_root).parent / cfg.raw["paths"]["input_root"]
    return Path(root) / version


def run_real(cfg, version, max_frames, tau):
    in_root = _input_root(cfg, version)
    blocks = {"CLEAN": cfg.raw["scenes"].get("clean", []),
              "SEEN": cfg.raw["scenes"].get("seen", [])}
    report = {"version": version, "blocks": {}}

    for block, entries in blocks.items():
        print(f"\n{'#'*70}\n# {block} BLOCK"
              + ("   — predictive of the test score" if block == "CLEAN"
                 else "   — upper bound only; success here proves nothing")
              + f"\n{'#'*70}")
        rows = []
        for entry in entries:
            scene = entry["name"] if isinstance(entry, dict) else entry
            sdir = in_root / scene
            file_a, file_b = sdir / "track1.txt", sdir / "detections_2d.txt"
            if not file_a.exists() or not file_b.exists():
                print(f"\n{scene}: missing File A/B under {sdir} — skipped")
                continue
            gt_path = cfg.find_gt(scene)
            r2, r3 = run_layers(cfg, gt_path, file_a, file_b, max_frames)
            v = vd.diagnose(r2, r3, **tau)
            m2, m3 = r2["macro"], r3["macro"]
            print(f"\n{scene}   [{v.verdict}]")
            print(f"   2D  macro P {m2['precision']:.4f}  R {m2['recall']:.4f}")
            print(f"   3D  macro P {m3['precision']:.4f}  R {m3['recall']:.4f}"
                  f"   track_ratio {r3.get('track_ratio', 1.0):.2f}")
            print(f"   -> {v.reason}")
            for o in v.observations:
                print(f"        · {o}")
            rows.append({"scene": scene, "verdict": v.to_dict(),
                         "report_2d": r2, "report_3d": r3})
        report["blocks"][block] = rows
    return report


def main():
    ap = argparse.ArgumentParser(description="Step 6 — cross-reference layers.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--verify-diagnosis", action="store_true",
                    help="Assert diagnose() on the fixtures (default action)")
    ap.add_argument("--input", default=None,
                    help="Model version under data/input/ for a real run")
    ap.add_argument("--scene", default=None,
                    help="Limit verify to one source scene")
    ap.add_argument("--frames", type=int, default=None,
                    help="Cap frames for a fast pass; full run is the real test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(args.config)
    tau = thresholds(cfg)

    if args.input:
        report = run_real(cfg, args.input, args.frames, tau)
        out = Path(args.out) if args.out else (
            cfg.reports_root / f"step6_{args.input}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {out}")
        sys.exit(0)

    # default: verify the diagnosis
    scenes = [args.scene] if args.scene else cfg.raw["synthetic"]["source_scenes"]
    all_pass, summary = verify_diagnosis(cfg, scenes, args.frames, tau)
    out = Path(args.out) if args.out else (cfg.reports_root / "step6_verify.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"passed": all_pass, "thresholds": tau, "results": summary}, indent=2))
    print(f"report -> {out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()