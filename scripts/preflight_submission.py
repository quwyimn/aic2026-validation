#!/usr/bin/env python3
"""
preflight_submission.py — will the official scorer accept this file?

Runs the ground-truth-free checks in src/io/preflight.py against one or more
track1.txt files. Needs no ground truth, so it is the only part of the
validator that runs on the test set.

    python3 scripts/preflight_submission.py --input data/input/v1/track1.txt
    python3 scripts/preflight_submission.py \
        --input data/synthetic/Warehouse_020/clean/track1.txt \
        --json reports/preflight_clean.json

    # test set: 5 scenes, and W026/W027 are only 1800 frames long
    python3 scripts/preflight_submission.py --input submission.txt \
        --expect-scenes 23,24,25 --num-frames 9000
    python3 scripts/preflight_submission.py --input submission.txt \
        --expect-scenes 26,27 --num-frames 1800

Exit code 0 when no FATAL is found, 1 otherwise, so it drops into CI beside
steps 5 and 6. --strict also fails on warnings.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.preflight import check_file, format_report, save_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="one or more track1.txt files")
    ap.add_argument("--num-frames", type=int, default=9000,
                    help="evaluation window, exclusive upper bound "
                         "(official default 9000; use 1800 for W026/W027)")
    ap.add_argument("--expect-scenes", default=None,
                    help="comma-separated scene_ids that must be present, "
                         "e.g. 23,24,25 — a missing scene is FATAL")
    ap.add_argument("--json", default=None,
                    help="write the report as JSON (one file only)")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on warnings as well as fatals")
    args = ap.parse_args()

    expect = None
    if args.expect_scenes:
        expect = [int(s) for s in args.expect_scenes.split(",") if s.strip()]

    if args.json and len(args.input) > 1:
        ap.error("--json takes a single --input")

    worst = 0
    for i, path in enumerate(args.input):
        if not Path(path).exists():
            print(f"not found: {path}", file=sys.stderr)
            worst = 1
            continue
        report = check_file(path, num_frames=args.num_frames,
                            expect_scenes=expect,
                            max_examples=args.max_examples)
        if i:
            print("\n" + "=" * 70 + "\n")
        print(format_report(report))
        if args.json:
            out = save_report(report, args.json)
            print(f"\nwrote {out}")
        if report["fatal_count"] or (args.strict and report["warn_count"]):
            worst = 1

    return worst


if __name__ == "__main__":
    sys.exit(main())