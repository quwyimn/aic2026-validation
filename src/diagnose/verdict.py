"""
verdict.py — Step 6. Cross-reference the two layers into one diagnosis.

Layer 2D and Layer 3D each produce a report (Confusion.to_dict(); the 3D report
additionally carries track_ratio and flicker_tracks). Neither number alone says
*where* a fault lives — that is the whole reason two layers exist. The contrast
between them is the diagnosis:

    MODEL     The detector itself is wrong. The defect is already visible in
              Layer 2D (File B), before any 3D lifting, fusion or tracking.
              Sub-types: misclassification, missed detections, phantoms.

    MAPPING   The class lookup table is miswired. Layer 2D is clean, and Layer
              3D relabels every instance of a class the same way — a constant,
              not perception failing here and there. The single biggest risk:
              logs look fine, only the submission score is inexplicably low.

    PIPELINE  Fuse / track / lift corrupts the output. Layer 2D is clean, and
              Layer 3D is wrong in a way that is *scattered*, not systematic:
              displaced matches (miss+ghost), or one track split into many ids.

    CLEAN     Neither layer shows a defect above tolerance.

Two dimensions, judged differently — because they carry noise differently:

  confusion (off-diagonal)  fraction-based (>= tau_conf). Class-blind matching
                            on a dense scene occasionally pairs a displaced
                            object with a wrong-class neighbour; that trace
                            scales with crowding and must be tolerated.

  miss / ghost              NOT fraction-based. Crowd density produces
                            confusion, never miss/ghost (a matched pair is a
                            match regardless of class). So on a clean layer
                            these are ~0, and any miss/ghost is signal. The
                            detector is implicated when the SAME miss/ghost
                            shows in both layers: 894 phantom boxes are only
                            0.05% of 1.66M detections — invisible to a fraction
                            gate, unmistakable as "a fabricated object that was
                            there before any fusion". Presence-in-both, not
                            fraction, is the right instrument.

That distinction is exactly what separates deletion/phantom (MODEL) from
position_shift_severe (PIPELINE): identical-looking miss+ghost, but one shows
in Layer 2D and the other only in Layer 3D.
"""

from dataclasses import dataclass, field
from typing import List


# Defaults — the calibration derived from the synthetic fixtures. Overridable
# from config (see run_validation.py); defined here only as named fallbacks so
# no bare literal ever sits inside a branch.
TAU_CONF = 0.01     # off-diagonal / matched above this = real class confusion
TAU_LOSS = 0.01     # (miss + ghost) / gt above this = a large detection loss
TAU_RATIO = 1.15    # predicted-track / gt-track above this = fragmentation


# --- report accessors -------------------------------------------------------

def _matched_total(r):
    """Every matched pair = the whole confusion grid (diagonal + off).
    Miss and ghost are unmatched and are not counted here."""
    return sum(r.get("cells", {}).values())


def off_diag_fraction(r):
    """Off-diagonal as a fraction of matched pairs. This — not the raw count —
    tells a systematic swap from crowd-noise."""
    total = _matched_total(r)
    return (r.get("totals", {}).get("off_diagonal", 0) / total) if total else 0.0


def _miss(r):
    return r.get("totals", {}).get("miss", 0)


def _ghost(r):
    return r.get("totals", {}).get("ghost", 0)


def loss_fraction(r):
    """(miss + ghost) as a fraction of GT. A magnitude signal for a large loss;
    small-but-real losses are caught by cross-layer presence, not this."""
    gt = r.get("totals", {}).get("gt", 0)
    mg = _miss(r) + _ghost(r)
    return (mg / gt) if gt else 0.0


def present_classes(r):
    """Class names with ground truth actually present in this block. Only these
    can be flagged systematic — an absent class produces no matched pairs."""
    return {name for name, pc in r.get("per_class", {}).items()
            if pc.get("support", 0) > 0}


# --- verdict ----------------------------------------------------------------

@dataclass
class Verdict:
    verdict: str                       # MODEL | MAPPING | PIPELINE | CLEAN
    reason: str
    evidence: dict = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": self.evidence,
            "observations": self.observations,
        }

    def line(self):
        return f"{self.verdict:<9} {self.reason}"


def diagnose(r2d, r3d, *, tau_conf=TAU_CONF, tau_loss=TAU_LOSS,
             tau_ratio=TAU_RATIO):
    """Read both layer reports, return one Verdict.

    r2d: Layer 2D report (Confusion.to_dict()).
    r3d: Layer 3D report (Confusion.to_dict() + track_ratio + flicker_tracks).
    """
    conf2d, conf3d = off_diag_fraction(r2d), off_diag_fraction(r3d)
    loss2d, loss3d = loss_fraction(r2d), loss_fraction(r3d)
    miss2, ghost2 = _miss(r2d), _ghost(r2d)
    miss3, ghost3 = _miss(r3d), _ghost(r3d)
    ratio = r3d.get("track_ratio", 1.0)
    flicker = r3d.get("flicker_tracks", 0)
    sysflags = set(r3d.get("mapping_signature", {}).keys())
    present = present_classes(r3d)

    # Detector-origin miss/ghost: the SAME loss type appears in both layers.
    # Judged by presence-in-both (not fraction) so a small-but-real fabrication
    # is not washed out by a huge GT denominator.
    ghost_both = ghost2 > 0 and ghost3 > 0
    miss_both = miss2 > 0 and miss3 > 0

    evidence = {
        "conf_2d": round(conf2d, 6), "loss_2d": round(loss2d, 6),
        "conf_3d": round(conf3d, 6), "loss_3d": round(loss3d, 6),
        "miss_2d": miss2, "ghost_2d": ghost2,
        "miss_3d": miss3, "ghost_3d": ghost3,
        "track_ratio": round(ratio, 4), "flicker_tracks": flicker,
        "systematic_classes": {
            c: r3d["mapping_signature"][c] for c in sorted(sysflags)
        },
        "present_classes": sorted(present),
        "thresholds": {"conf": tau_conf, "loss": tau_loss, "ratio": tau_ratio},
    }

    # Observations: everything the two layers noticed, listed regardless of the
    # primary verdict. On a compound fault (the 'mixed' set) the verdict names
    # the dominant cause; these keep the rest visible.
    obs = []
    if conf2d >= tau_conf:
        obs.append(f"2D confusion {conf2d*100:.2f}% of matched")
    if loss2d >= tau_loss:
        obs.append(f"2D miss+ghost {loss2d*100:.2f}% of GT "
                   f"(miss={miss2}, ghost={ghost2})")
    elif ghost2 > 0 or miss2 > 0:
        obs.append(f"2D miss={miss2}, ghost={ghost2} (small but present)")
    if sysflags:
        obs.append(f"3D systematic relabel on {sorted(sysflags)}")
    if conf3d >= tau_conf and not sysflags:
        obs.append(f"3D scattered confusion {conf3d*100:.2f}% of matched")
    if loss3d >= tau_loss:
        obs.append(f"3D miss+ghost {loss3d*100:.2f}% of GT "
                   f"(miss={miss3}, ghost={ghost3})")
    if ratio > tau_ratio:
        obs.append(f"3D track inflation {ratio:.2f}x")
    if flicker:
        obs.append(f"3D class flicker on {flicker} tracks")

    # (1) Layer 2D carries the defect -> the detector is at fault. Nothing a
    # later stage does can un-break a wrong detection, so this wins outright.

    # 1a. Misclassification — confusion visible in File B.
    if conf2d >= tau_conf:
        return Verdict(
            "MODEL",
            f"detector misclassifies ({conf2d*100:.2f}% of matched confused)",
            evidence, obs)

    # 1b. Large detection loss already in 2D.
    if loss2d >= tau_loss:
        if miss2 >= ghost2:
            sub = f"detector misses objects (miss {loss2d*100:.2f}% of GT)"
        else:
            sub = f"detector hallucinates ({ghost2} ghost boxes in 2D)"
        return Verdict("MODEL", sub, evidence, obs)

    # 1c. Fabricated objects: ghosts in BOTH layers. Small in fraction, decisive
    # in kind — a false box present before any fusion, still there after. This
    # is what separates a phantom (MODEL) from track fragmentation (PIPELINE):
    # fragmentation re-ids real objects and produces no ghosts at all.
    if ghost_both:
        return Verdict(
            "MODEL",
            f"detector invents objects (phantoms): {ghost2} false boxes in 2D, "
            f"still present in 3D",
            evidence, obs)

    # 1d. Missed objects mirrored in both layers but below the fraction gate.
    if miss_both:
        return Verdict(
            "MODEL",
            f"detector misses objects (miss in both layers, 2D miss={miss2})",
            evidence, obs)

    # (2) Layer 2D is clean -> the detector is fine. Any 3D defect is downstream.

    # Systematic across every present class = a miswired table, not perception.
    if present and present <= sysflags:
        pairs = ", ".join(
            f"{c}->{r3d['mapping_signature'][c]['predicted_as']}"
            for c in sorted(present))
        return Verdict(
            "MAPPING",
            f"every present class relabelled the same way, ~100% ({pairs})",
            evidence, obs)

    # Systematic on some but not all present classes: still a wiring fault.
    if sysflags:
        return Verdict(
            "MAPPING",
            f"partial: {sorted(sysflags)} systematically relabelled while 2D is clean",
            evidence, obs)

    # Scattered class confusion with a clean detector = fuse/track corrupts the
    # label. Not systematic (no class hit the 90% signature), so not mapping.
    if conf3d >= tau_conf:
        return Verdict(
            "PIPELINE",
            f"class corrupted after the detector (scattered 3D confusion "
            f"{conf3d*100:.2f}% of matched, 2D clean)",
            evidence, obs)

    # Classes intact, one object wears many track ids, and no ghosts — real
    # objects re-identified, not fabricated. Association, not perception.
    if ratio > tau_ratio:
        return Verdict(
            "PIPELINE",
            f"broken association: {ratio:.2f}x more tracks than GT, "
            f"classes intact, no ghosts",
            evidence, obs)

    # Classes intact, but objects dropped/displaced only after 2D — the 3D lift
    # or fusion, not the detector.
    if loss3d >= tau_loss:
        return Verdict(
            "PIPELINE",
            f"3D lift/fuse drops or displaces objects "
            f"(miss+ghost {loss3d*100:.2f}% of GT, 2D clean)",
            evidence, obs)

    return Verdict("CLEAN", "no defect above tolerance in either layer",
                   evidence, obs)