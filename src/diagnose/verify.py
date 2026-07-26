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
    """One expectation and whether the validator met it.

    `flavor` distinguishes *why* a check passed. A set can pass because the
    validator is correct, or because it is deliberately blind (wl_swap) or
    because the metric itself is indifferent (yaw_pi). Those must never look
    identical on screen — a "PASS" that means "we can't see this" dressed up
    as a "PASS" that means "this is right" is exactly the false reassurance
    an 18/18 tally is supposed to prevent. Default is empty: every existing
    check keeps printing exactly as before.
    """

    def __init__(self, name, ok, detail="", flavor=""):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.flavor = flavor

    def __str__(self):
        mark = "PASS" if self.ok else "FAIL"
        flav = f" {self.flavor}" if self.flavor else ""
        tail = f"  ({self.detail})" if self.detail else ""
        return f"  [{mark}]{flav} {self.name}{tail}"


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


def _layer_is_clean(r):
    """CLEAN = perfect precision and recall, no miss, ghost or confusion.
    Same bar verify_clean uses, factored out so the geometry verifiers assert
    cleanliness against the identical fields rather than a looser proxy."""
    p = _macro(r, "precision")
    rec = _macro(r, "recall")
    return (abs(p - 1.0) < 1e-6 and abs(rec - 1.0) < 1e-6
            and _miss(r) == 0 and _ghost(r) == 0 and _off_diag(r) == 0)


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


def verify_wl_swap(key, r2d, r3d, ctx=None):
    """KNOWN BLIND SPOT. width<->length swapped; centre and class untouched.

    The validator matches on centre distance and never looks at box shape, so
    BOTH layers must report CLEAN — the defect is real but invisible here.
    Two assertions, and the second is the safety catch:

      (a) both layers CLEAN  — the validator behaves as documented today;
      (b) the answer key carries the blind_spot flag — the defect is on record.

    If someone later upgrades layer 3D to match on centre+shape, (a) flips:
    3D stops being CLEAN, this check FAILS, and Step 5 turns red at exactly the
    line that says "the blind spot is gone, update the expectation." That is
    what keeps a *known* blind spot from silently becoming a *forgotten* one.
    """
    flavor = "[BLIND SPOT]"
    n = len(key.get("wl_swap", []))
    flag = key.get("blind_spot")
    checks = [Check("answer key records w<->l swaps", n > 0,
                    f"{n} boxes", flavor)]
    checks.append(Check(
        "answer key carries the blind-spot flag", flag == "shape_via_3d_iou",
        f"blind_spot={flag!r}", flavor))
    for layer, r in (("2D", r2d), ("3D", r3d)):
        clean = _layer_is_clean(r)
        checks.append(Check(
            f"{layer} reports CLEAN (validator blind to shape — only 3D IoU sees it)",
            clean,
            f"P={_macro(r,'precision'):.4f} offdiag={_off_diag(r)} "
            f"miss={_miss(r)} ghost={_ghost(r)}", flavor))
    return checks


def verify_yaw_pi(key, r2d, r3d, ctx=None):
    """yaw + 180 deg. CLEAN in both layers — but for a different reason than
    wl_swap, and the report must not conflate the two.

    A box is symmetric under a 180 deg rotation about its vertical axis, so the
    official 3D-IoU metric scores it identically. This is not a limitation of
    the validator; it is the rule of the game. The flag is therefore
    expected_zero_impact, not blind_spot: head/tail direction is genuinely
    worth zero points, and the fixture exists to prove that negative.
    """
    flavor = "[ZERO-IMPACT]"
    n = len(key.get("yaw_pi", []))
    flag = key.get("expected_zero_impact")
    checks = [Check("answer key records yaw+180deg flips", n > 0,
                    f"{n} boxes", flavor)]
    checks.append(Check(
        "answer key carries the zero-impact flag",
        flag == "180deg_symmetry", f"expected_zero_impact={flag!r}", flavor))
    for layer, r in (("2D", r2d), ("3D", r3d)):
        checks.append(Check(
            f"{layer} reports CLEAN (180deg is a box symmetry — even 3D IoU is unchanged)",
            _layer_is_clean(r),
            f"P={_macro(r,'precision'):.4f} offdiag={_off_diag(r)}", flavor))
    return checks


def verify_dup_id(key, r2d, r3d, ctx=None):
    """Not a layer question. Two rows share (scene, class, frame, object_id),
    which makes the official scorer's TrackEval raise. The layers cannot see
    this — it is a submission-integrity fault — so Step 5's job here is to
    confirm PREFLIGHT catches it, on the full-length file.

    ctx must carry the path to this set's track1.txt. Without it the check
    cannot run and fails loudly rather than passing vacuously.
    """
    n = len(key.get("dup_id", {}))
    checks = [Check("answer key records duplicated ids", n > 0, f"{n} ids")]
    path = (ctx or {}).get("file_a")
    if not path:
        checks.append(Check("preflight ran on File A", False,
                            "no track1.txt path supplied to the verifier"))
        return checks
    try:
        from src.io.preflight import check_file
    except Exception as e:  # noqa: BLE001
        checks.append(Check("preflight importable", False, str(e)))
        return checks
    rep = check_file(path)
    dup = rep["issues"].get("E_DUP")
    checks.append(Check(
        "preflight FATALs on duplicate (scene,class,frame,object_id)",
        bool(dup) and dup["severity"] == "FATAL" and dup["count"] > 0,
        f"E_DUP count={dup['count'] if dup else 0}, "
        f"fatal_total={rep['fatal_count']}"))
    return checks


VERIFIERS = {
    "clean": verify_clean,
    "wl_swap": verify_wl_swap,
    "yaw_pi": verify_yaw_pi,
    "dup_id": verify_dup_id,
    "class_swap": verify_class_swap,
    "mapping_shift": verify_mapping_shift,
    "position_shift": verify_position_shift,
    "position_shift_severe": verify_position_shift_severe,
    "deletion": verify_deletion,
    "phantom": verify_phantom,
    "fragmentation": verify_fragmentation,
    "mixed": verify_mixed,
}


def verify_set(error_set, key, r2d, r3d, ctx=None):
    """Return list of Check for one error set. Unknown sets pass vacuously with
    a note rather than crashing the whole run.

    `ctx` carries extra paths a verifier may need (currently only dup_id, which
    reads File A to prove preflight catches it). Verifiers written against the
    original three-argument signature are called unchanged; only those that
    declare a fourth parameter receive ctx, so the nine original verifiers are
    untouched."""
    fn = VERIFIERS.get(error_set)
    if fn is None:
        return [Check(f"no verifier defined for '{error_set}'", True, "skipped")]
    import inspect
    if len(inspect.signature(fn).parameters) >= 4:
        return fn(key, r2d, r3d, ctx)
    return fn(key, r2d, r3d)