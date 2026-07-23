"""
test_verdict.py — prove Step 6's logic against every fixture signature.

The full 9000-frame run happens on the real machine (it needs src/ + GT). Here
we feed diagnose() report dicts shaped like Confusion.to_dict(), using the
REAL magnitudes observed on the full run — not round guesses.

That distinction matters: an earlier version of this test gave `phantom` a 5%
ghost rate and passed, while the real fixture has 894 ghosts (0.05%) plus a
6.70x track inflation and FAILED on the machine. Test numbers that don't match
reality hide the very bug the test exists to catch — the same lesson as the
zero-area boxes in Step 3. The numbers below are transcribed from the machine
run so the test now guards the case that actually broke.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diagnose.verdict import diagnose


def rep(*, gt, matched, off_diag=0, miss=0, ghost=0, macro_p=1.0,
        mapping_signature=None, track_ratio=1.0, flicker=0, present=None):
    """A Confusion.to_dict()-shaped report. Only the fields diagnose() reads
    are populated — that is the whole contract between the layers and Step 6."""
    present = present or ["Person", "Forklift", "PalletTruck"]
    return {
        "cells": {"_matched": matched},
        "totals": {"gt": gt, "miss": miss, "ghost": ghost,
                   "off_diagonal": off_diag},
        "macro": {"precision": macro_p, "recall": macro_p},
        "per_class": {n: {"support": 1} for n in present},
        "mapping_signature": mapping_signature or {},
        "track_ratio": track_ratio,
        "flicker_tracks": flicker,
    }


# Observed on W020, 9000 frames.
GT2 = 1_657_000       # 2D GT instances (from deletion: 63469 miss = 3.83%)
GT3 = 610_776         # 3D GT instances (the survey's detection count)

CASES = [
    # clean — perfect both layers.
    ("clean",
     rep(gt=GT2, matched=GT2),
     rep(gt=GT3, matched=GT3),
     "CLEAN"),

    # class_swap — confusion in BOTH layers. 2D 4.98%, 3D 5.55% (from the run).
    ("class_swap",
     rep(gt=GT2, matched=GT2, off_diag=int(GT2 * 0.0498), macro_p=0.9908),
     rep(gt=GT3, matched=GT3, off_diag=int(GT3 * 0.0555), macro_p=0.966),
     "MODEL"),

    # mapping_shift — 2D clean; 3D every present class relabelled 100%.
    ("mapping_shift",
     rep(gt=GT2, matched=GT2),
     rep(gt=GT3, matched=GT3, off_diag=GT3, macro_p=0.0,
         mapping_signature={
             "Person":      {"predicted_as": "Forklift",   "fraction": 1.0},
             "Forklift":    {"predicted_as": "NovaCarter", "fraction": 1.0},
             "PalletTruck": {"predicted_as": "Person",     "fraction": 1.0},
         }),
     "MAPPING"),

    # position_shift (mild) — under 1m threshold, still matches, classes right.
    ("position_shift_mild",
     rep(gt=GT2, matched=GT2),
     rep(gt=GT3, matched=GT3, off_diag=int(GT3 * 0.0002)),
     "CLEAN"),

    # position_shift_severe — 2D clean (File B untouched); 3D miss+ghost 15.71%.
    ("position_shift_severe",
     rep(gt=GT2, matched=GT2),
     rep(gt=GT3, matched=int(GT3 * 0.85),
         miss=int(GT3 * 0.0785), ghost=int(GT3 * 0.0786), macro_p=0.8914),
     "PIPELINE"),

    # deletion — misses in BOTH layers, ghost 0. 2D miss 3.83%, 3D miss 5.21%.
    ("deletion",
     rep(gt=GT2, matched=int(GT2 * 0.9617), miss=63469),
     rep(gt=GT3, matched=int(GT3 * 0.9479), miss=int(GT3 * 0.0521)),
     "MODEL"),

    # phantom — the case that failed. 2D ghost 894 (0.05%), NOT a fraction
    # signal; 3D ghost present + track inflation 6.70x. Routes MODEL only via
    # ghost-in-both, not via any fraction gate.
    ("phantom",
     rep(gt=GT2, matched=GT2, ghost=894),
     rep(gt=GT3, matched=GT3, ghost=1500, track_ratio=6.70),
     "MODEL"),

    # fragmentation — 2D clean; 3D classes intact, NO ghost, track_ratio 1.20.
    # Distinguished from phantom purely by the absence of ghosts.
    ("fragmentation",
     rep(gt=GT2, matched=GT2),
     rep(gt=GT3, matched=GT3, track_ratio=1.20, flicker=0),
     "PIPELINE"),
]


def main():
    print(f"\n{'set':<24}{'expected':<10}{'got':<10}{'result'}")
    print("-" * 60)
    all_ok = True
    for name, r2d, r3d, expected in CASES:
        v = diagnose(r2d, r3d)
        ok = v.verdict == expected
        all_ok = all_ok and ok
        print(f"{name:<24}{expected:<10}{v.verdict:<10}"
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    -> {v.reason}")
            print(f"    -> evidence: {v.evidence}")
    print("-" * 60)
    n = sum(1 for _, a, b, e in CASES if diagnose(a, b).verdict == e)
    print(f"RESULT: {n}/{len(CASES)} verdicts correct")
    if all_ok:
        print("Every fixture signature maps to the right verdict, at the real\n"
              "magnitudes. Run at full 9000 frames on the real layers to\n"
              "confirm end to end.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()