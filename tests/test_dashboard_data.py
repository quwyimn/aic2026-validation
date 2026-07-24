"""
test_dashboard_data.py — the dashboard's data layer, checked without a browser.

Streamlit cannot be asserted on, but everything that turns a report into a
number or a table can. These are the functions that would fail loudly in front
of the team — a KeyError on a real report, or a layer painted green beside a
fault the tool had already caught.

The fixture below is built from the actual machine run, not from invented
numbers. That is the same discipline as test_verdict.py, and for the same
reason: the one time round guesses were used there, they hid the exact bug the
test existed to catch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from app import (confusion_frame, fixture_table, per_class_frame,
                 strip_signals, trust_summary)


def rep(*, gt, matched, cells=None, miss=None, ghost=None, off_diag=0,
        macro_p=1.0, per_class=None, track_ratio=None, mapping_signature=None):
    r = {
        "cells": cells if cells is not None else {"Person->Person": matched},
        "miss": miss or {},
        "ghost": ghost or {},
        "per_class": per_class or {
            "Person":      {"support": gt, "correct": matched, "precision": macro_p,
                            "recall": macro_p, "f1": macro_p, "miss": 0, "ghost": 0},
            "AgilityDigit": {"support": 0, "correct": 0, "precision": float("nan"),
                             "recall": float("nan"), "f1": float("nan"),
                             "miss": 0, "ghost": 0},
        },
        "macro": {"precision": macro_p, "recall": macro_p, "f1": macro_p},
        "totals": {"gt": gt, "miss": sum((miss or {}).values()),
                   "ghost": sum((ghost or {}).values()),
                   "off_diagonal": off_diag},
        "absent_class_predictions": {},
        "mapping_signature": mapping_signature or {},
    }
    if track_ratio is not None:
        r["track_ratio"] = track_ratio
        r["flicker_tracks"] = 0
    return r


# Numbers transcribed from the W020 full-length run.
VERIFY = {
    "passed": True,
    "thresholds": {"tau_conf": 0.01, "tau_loss": 0.01, "tau_ratio": 1.15},
    "results": [
        {"scene": "Warehouse_020", "error_set": "clean", "expected": "CLEAN",
         "got": "CLEAN", "passed": True,
         "reason": "no defect above tolerance in either layer",
         "observations": [],
         "report_2d": rep(gt=1657000, matched=1657000),
         "report_3d": rep(gt=610776, matched=610776, track_ratio=1.0)},

        # phantom: 894 ghosts in 2D — 0.05%, far under any fraction gate. The
        # dashboard must still read this layer as dirty, or it would show green
        # next to a MODEL verdict and contradict the tool on screen.
        {"scene": "Warehouse_020", "error_set": "phantom", "expected": "MODEL",
         "got": "MODEL", "passed": True,
         "reason": "detector invents objects (phantoms): 894 false boxes in 2D",
         "observations": ["2D miss=0, ghost=894 (small but present)",
                          "3D track inflation 6.70x"],
         "report_2d": rep(gt=1657000, matched=1657000, ghost={"Person": 894}),
         "report_3d": rep(gt=610776, matched=610776, ghost={"Person": 1500},
                          track_ratio=6.70)},

        {"scene": "Warehouse_020", "error_set": "mapping_shift",
         "expected": "MAPPING", "got": "MAPPING", "passed": True,
         "reason": "every present class relabelled the same way, ~100%",
         "observations": ["3D systematic relabel on ['Forklift', 'PalletTruck', 'Person']"],
         "report_2d": rep(gt=1657000, matched=1657000),
         "report_3d": rep(gt=610776, matched=610776, off_diag=610776, macro_p=0.0,
                          cells={"Person->Forklift": 610776},
                          track_ratio=1.0,
                          mapping_signature={"Person": {"predicted_as": "Forklift",
                                                        "fraction": 1.0}})},

        # 'mixed' has no single correct answer and must not pad the pass rate.
        {"scene": "Warehouse_020", "error_set": "mixed", "expected": None,
         "got": "MODEL", "passed": True,
         "reason": "detector misclassifies (6.40% of matched confused)",
         "observations": ["2D confusion 6.40% of matched"],
         "report_2d": rep(gt=1657000, matched=1657000, off_diag=106048,
                          macro_p=0.936),
         "report_3d": rep(gt=610776, matched=610776, track_ratio=6.75)},
    ],
}


# fragmentation: classes perfect, nothing missing or invented — the only
# signal is that one object wears many track ids (W011 measured 1.22x).
FRAGMENTATION_2D = rep(gt=1657000, matched=1657000)
FRAGMENTATION_3D = rep(gt=610776, matched=610776, track_ratio=1.22)

# position_shift (mild): displacement mostly under the 1m match threshold, so a
# handful of objects fall out and return as miss+ghost pairs. Real, tiny, and
# not what the CLEAN verdict rests on.
MILD_SHIFT_2D = rep(gt=1657000, matched=1657000)
MILD_SHIFT_3D = rep(gt=610776, matched=608000, miss={"Person": 320},
                    ghost={"Person": 320}, track_ratio=1.0)


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ok = True
    print("\nstrip_signals — the strip must never contradict the verdict")
    r = VERIFY["results"]

    l2, v2, l3, v3 = strip_signals(r[0]["report_2d"], r[0]["report_3d"])
    ok &= check("clean: both layers read clean beside a CLEAN verdict",
                (l2, v2, l3, v3) == ("clean", "ok", "clean", "ok"))

    l2, v2, l3, v3 = strip_signals(r[1]["report_2d"], r[1]["report_3d"])
    ok &= check("phantom: 894 ghosts read decisive, not 'below threshold'",
                v2 == "bad" and "894 ghost" in l2,
                f"2D={l2!r} ({v2}) — 0.05% of GT, decisive only because 3D "
                f"has ghosts too")

    l2, v2, l3, v3 = strip_signals(r[2]["report_2d"], r[2]["report_3d"])
    ok &= check("mapping_shift: 2D clean, 3D names the systematic relabel",
                v2 == "ok" and v3 == "bad" and "systematic" in l3,
                f"3D={l3!r}")

    # The two rows that were wrong on screen. Both looked self-contradictory:
    # a CLEAN verdict beside an orange layer, and a PIPELINE verdict beside two
    # green ones.
    l2, v2, l3, v3 = strip_signals(FRAGMENTATION_2D, FRAGMENTATION_3D)
    ok &= check("fragmentation: track inflation is visible, so PIPELINE follows",
                v3 == "bad" and "track inflated" in l3,
                f"3D={l3!r} — no confusion, no miss, no ghost; the whole signal "
                f"is the track count")

    l2, v2, l3, v3 = strip_signals(MILD_SHIFT_2D, MILD_SHIFT_3D)
    ok &= check("position_shift: stray ghosts read as below threshold, not a fault",
                v2 == "ok" and v3 == "note",
                f"3D={l3!r} ({v3}) — shown, but not what a CLEAN verdict rests on")

    l2, v2, l3, v3 = strip_signals(None, None)
    ok &= check("missing reports degrade to n/a, no crash", l2 == "n/a")

    print("\ntrust_summary — unasserted sets must not pad the pass rate")
    s = trust_summary(VERIFY)
    ok &= check("counts only asserted sets", s["total"] == 3, f"total={s['total']}")
    ok &= check("mixed counted apart", s["unasserted"] == 1)
    ok &= check("all_pass true", s["all_pass"] is True)
    ok &= check("no report degrades to None", trust_summary(None) is None)

    print("\nfixture_table — one row per fixture, both layer signals present")
    df = fixture_table(VERIFY)
    ok &= check("row per fixture", len(df) == 4, f"{len(df)} rows")
    ok &= check("phantom row pairs decisive ghosts with MODEL",
                df.loc[df.fixture == "phantom", "sev_2D"].iloc[0] == "bad"
                and df.loc[df.fixture == "phantom", "verdict"].iloc[0] == "MODEL")
    ok &= check("expected None renders as em dash",
                df.loc[df.fixture == "mixed", "expected"].iloc[0] == "—")

    print("\nconfusion_frame — MISS column and GHOST row must both exist")
    cf = confusion_frame(r[1]["report_2d"])
    ok &= check("MISS column present", "MISS" in cf.columns)
    ok &= check("GHOST row present", "GHOST" in cf.index)
    ok &= check("894 ghosts land in the GHOST row",
                int(cf.loc["GHOST", "Person"]) == 894)
    cf2 = confusion_frame(r[2]["report_3d"])
    ok &= check("off-diagonal cell placed at [GT, pred]",
                int(cf2.loc["Person", "Forklift"]) == 610776)
    ok &= check("no report degrades to None", confusion_frame(None) is None)

    # The W011 clean case, exactly as it appeared on screen: seven classes, one
    # of them absent, names long enough that the matrix was clipped and the
    # AgilityDigit diagonal fell off the right edge — a row reading as all
    # zeros beside a per-class precision of 1.0000.
    wide = rep(gt=610776, matched=610776,
               cells={f"{c}->{c}": 90000 for c in
                      ["Person", "Forklift", "NovaCarter", "Transporter",
                       "FourierGR1T2", "AgilityDigit"]},
               per_class={c: {"support": 90000, "correct": 90000,
                              "precision": 1.0, "recall": 1.0, "f1": 1.0,
                              "miss": 0, "ghost": 0}
                          for c in ["Person", "Forklift", "NovaCarter",
                                    "Transporter", "FourierGR1T2",
                                    "AgilityDigit"]}
               | {"PalletTruck": {"support": 0, "correct": 0,
                                  "precision": float("nan"),
                                  "recall": float("nan"), "f1": float("nan"),
                                  "miss": 0, "ghost": 0}})
    cw = confusion_frame(wide)
    ok &= check("class with nothing anywhere is dropped from the matrix",
                not any("PalletTru" in str(c) for c in cw.columns),
                f"columns={list(cw.columns)}")
    ok &= check("every remaining class keeps its diagonal on screen",
                all(int(cw.iloc[i, i]) == 90000 for i in range(6)),
                "AgilityDigit included — the row that read as all zeros")
    ok &= check("long names shortened once the matrix is wide",
                all(len(str(c)) <= 10 for c in cw.columns))

    print("\nper_class_frame — absent classes kept and marked, never dropped")
    pc = per_class_frame(r[0]["report_2d"])
    ok &= check("absent class still listed",
                "AgilityDigit" in list(pc["class"]))
    ok &= check("absent class marked, not shown as zero",
                pc.loc[pc["class"] == "AgilityDigit", "note"].iloc[0] == "no GT here")
    ok &= check("NaN metrics render as em dash, not nan",
                pc.loc[pc["class"] == "AgilityDigit", "precision"].iloc[0] == "—")

    print("\n" + ("RESULT: dashboard data layer OK"
                  if ok else "RESULT: FAILURES ABOVE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()