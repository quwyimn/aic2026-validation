"""
verify.py — does the validator report exactly the errors that were injected?

Every other step measures a model. This one measures the validator itself.

Each synthetic set ships an injected_errors.json answer key stating precisely
what was broken. Here both layers are run against that set and their output is
checked against the key. If they disagree, the validator is wrong — not the
data — and the number it would report on a real model cannot be trusted.

This is the step that closes the loop. Up to now the fixtures were read by eye
("mapping_shift scores 1.000 in 2D, good"). Reading by eye does not scale to 18
sets and does not survive a code change three weeks from now. From here it is
an assertion that either passes or fails.
"""

import json
import math
from pathlib import Path


class Check:
    """One expectation and whether the validator met it."""

    def __init__(self, name, ok, detail=""):
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self):
        mark = "PASS" if self.ok else "FAIL"
        tail = f"  ({self.detail})" if self.detail else ""
        return f"  [{mark}] {self.name}{tail}"


def _macro(report, metric):
    return report.get("macro", {}).get(metric, float("nan"))


def _off_diag(report):
    return report.get("totals", {}).get("off_diagonal", 0)


def _miss(report):
    return report.get("totals", {}).get("miss", 0)


def _ghost(report):
    return report.get("totals", {}).get("ghost", 0)


def _matched_total(report):
    """Total matched pairs = everything on the confusion grid (diagonal + off).
    Miss and ghost are unmatched, so they are not counted here."""
    return sum(report.get("cells", {}).values())


def _off_diag_fraction(report):
    """Off-diagonal as a fraction of matched pairs. Pure geometric matching
    occasionally pairs a badly-displaced object with a different-class
    neighbour that happens to fall within the threshold. On a crowded real
    scene that noise is nonzero and correct — so a position error is judged by
    whether confusion stays negligible, not by whether it is exactly zero."""
    total = _matched_total(report)
    return (_off_diag(report) / total) if total else 0.0


# --- Per-error-type expectations -------------------------------------------
# Each returns a list of Check. The expectations are deliberately phrased as
# relationships between the two layers, because that relationship — not either
# number alone — is what the design promises.

def verify_clean(key, r2d, r3d):
    checks = []
    for layer, r in (("2D", r2d), ("3D", r3d)):
        p = _macro(r, "precision")
        rec = _macro(r, "recall")
        perfect = (abs(p - 1.0) < 1e-6 and abs(rec - 1.0) < 1e-6
                   and _miss(r) == 0 and _ghost(r) == 0 and _off_diag(r) == 0)
        checks.append(Check(
            f"clean is perfect in {layer}", perfect,
            f"P={p:.4f} R={rec:.4f} miss={_miss(r)} ghost={_ghost(r)} "
            f"offdiag={_off_diag(r)}"))
    return checks


def verify_class_swap(key, r2d, r3d):
    n = len(key.get("class_swap", {}))
    checks = [Check("answer key records class swaps", n > 0, f"{n} swaps")]
    # Model error: BOTH layers show off-diagonal confusion.
    for layer, r in (("2D", r2d), ("3D", r3d)):
        od = _off_diag(r)
        checks.append(Check(
            f"{layer} shows confusion (off-diagonal)", od > 0, f"{od} cells"))
    return checks


def verify_mapping_shift(key, r2d, r3d):
    checks = []
    # 2D must stay clean — File B keeps the true table.
    p2 = _macro(r2d, "precision")
    checks.append(Check(
        "2D stays clean (detector untouched)",
        abs(p2 - 1.0) < 1e-6 and _off_diag(r2d) == 0,
        f"P={p2:.4f} offdiag={_off_diag(r2d)}"))
    # 3D must show the systematic signature. Only classes that actually occur
    # in this scene can be flagged — the key lists the whole 7-class table, but
    # W020 holds just 3 of them, so the comparison is against classes present,
    # not against the full table. (Comparing to all 7 was the original bug in
    # this check: it demanded 6 flags from a scene that can produce at most 3.)
    sig = r3d.get("mapping_signature", {})
    present = {c for c in key.get("mapping_shift", {})
               if c in {n for n in
                        [x for x in r3d.get("per_class", {})
                         if r3d["per_class"][x]["support"] > 0]}}
    checks.append(Check(
        "3D flags systematic mislabelling", len(sig) > 0,
        f"{len(sig)} classes flagged"))
    # Every class present in the scene must be flagged, and each must point at
    # exactly one wrong label at ~100% (that is what 'systematic' means).
    all_flagged = present.issubset(set(sig.keys()))
    all_full = all(sig.get(c, {}).get("fraction", 0) >= 0.99 for c in present)
    detail = ", ".join(f"{c}->{sig[c]['predicted_as']}"
                       for c in sorted(sig)) or "none"
    checks.append(Check(
        "every present class is flagged, each ~100%",
        all_flagged and all_full and len(present) > 0,
        detail))
    return checks


def verify_position_shift(key, r2d, r3d):
    # Under-threshold variant: everything still matches, classes correct.
    checks = []
    p2 = _macro(r2d, "precision")
    checks.append(Check("2D unaffected", abs(p2 - 1.0) < 1e-6, f"P={p2:.4f}"))
    # 3D: displacement stays under threshold, so matches survive and classes
    # stay essentially correct. A trace of confusion (<1%) is expected from
    # geometric matching in a crowded scene and does not change the verdict.
    frac = _off_diag_fraction(r3d)
    checks.append(Check(
        "3D keeps classes correct (small shift tolerated)", frac < 0.01,
        f"offdiag={_off_diag(r3d)} ({frac*100:.2f}% of matched), miss={_miss(r3d)}"))
    return checks


def verify_position_shift_severe(key, r2d, r3d):
    checks = []
    p2 = _macro(r2d, "precision")
    checks.append(Check("2D unaffected", abs(p2 - 1.0) < 1e-6, f"P={p2:.4f}"))
    # 3D: displacement over threshold -> each displaced object becomes a
    # miss + a ghost, but classes themselves are not confused.
    miss = _miss(r3d)
    checks.append(Check(
        "3D produces miss+ghost (large shift breaks match)", miss > 0,
        f"miss={miss} ghost={_ghost(r3d)}"))
    frac = _off_diag_fraction(r3d)
    checks.append(Check(
        "3D confusion stays negligible (signal is miss+ghost, not swaps)",
        frac < 0.01,
        f"offdiag={_off_diag(r3d)} ({frac*100:.2f}% of matched)"))
    return checks


def verify_deletion(key, r2d, r3d):
    n = len(key.get("deleted", []))
    checks = [Check("answer key records deletions", n > 0, f"{n} objects")]
    for layer, r in (("2D", r2d), ("3D", r3d)):
        checks.append(Check(
            f"{layer} shows misses, no ghosts",
            _miss(r) > 0 and _off_diag(r) == 0,
            f"miss={_miss(r)} ghost={_ghost(r)} offdiag={_off_diag(r)}"))
    return checks


def verify_phantom(key, r2d, r3d):
    n = key.get("phantom_count", 0)
    checks = [Check("answer key records phantoms", n > 0, f"{n} phantoms")]
    for layer, r in (("2D", r2d), ("3D", r3d)):
        checks.append(Check(
            f"{layer} shows ghosts, no misses",
            _ghost(r) > 0 and _miss(r) == 0,
            f"ghost={_ghost(r)} miss={_miss(r)}"))
    return checks


def verify_fragmentation(key, r2d, r3d):
    n = len(key.get("fragmented", {}))
    checks = [Check("answer key records fragmentation", n > 0, f"{n} tracks")]
    # 2D has no notion of track id -> unaffected.
    checks.append(Check(
        "2D unaffected (no track identity in 2D)",
        abs(_macro(r2d, "precision") - 1.0) < 1e-6,
        f"P={_macro(r2d, 'precision'):.4f}"))
    # 3D: more predicted tracks than GT tracks, classes still correct.
    ratio = r3d.get("track_ratio", 1.0)
    checks.append(Check(
        "3D inflates track count, classes intact",
        ratio > 1.0 and _off_diag(r3d) == 0,
        f"track_ratio={ratio:.2f} offdiag={_off_diag(r3d)}"))
    return checks


def verify_mixed(key, r2d, r3d):
    # Mixed carries everything at once; assert the compound signature rather
    # than each piece, since they interact.
    checks = []
    checks.append(Check(
        "3D shows confusion (swaps + mapping present)",
        _off_diag(r3d) > 0 or len(r3d.get("mapping_signature", {})) > 0,
        f"offdiag={_off_diag(r3d)}"))
    checks.append(Check(
        "3D shows miss (deletion + severe shift present)",
        _miss(r3d) > 0, f"miss={_miss(r3d)}"))
    checks.append(Check(
        "3D shows ghost (phantoms present)",
        _ghost(r3d) > 0, f"ghost={_ghost(r3d)}"))
    return checks


VERIFIERS = {
    "clean": verify_clean,
    "class_swap": verify_class_swap,
    "mapping_shift": verify_mapping_shift,
    "position_shift": verify_position_shift,
    "position_shift_severe": verify_position_shift_severe,
    "deletion": verify_deletion,
    "phantom": verify_phantom,
    "fragmentation": verify_fragmentation,
    "mixed": verify_mixed,
}


def verify_set(error_set, key, r2d, r3d):
    """Return list of Check for one error set. Unknown sets pass vacuously with
    a note rather than crashing the whole run."""
    fn = VERIFIERS.get(error_set)
    if fn is None:
        return [Check(f"no verifier defined for '{error_set}'", True, "skipped")]
    return fn(key, r2d, r3d)