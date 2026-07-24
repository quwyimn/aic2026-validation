"""
app.py — Step 7. The validation dashboard.

Reads reports/ and computes nothing. Every number on screen was produced by the
layers and is shown exactly as recorded — a dashboard that recalculates is a
second source of truth, and two sources drift.

Two things this shows, in this order:

  1. Can the tool be trusted?   The fixture results. Errors whose answer is
     known, and whether the tool named the right stage for each. Until a real
     model arrives this is the only claim on screen that has been proven, and
     it is also the one a reader should demand first.

  2. What did the model do?     The real run, once File A and File B arrive
     from the training stage. CLEAN and SEEN stay separate blocks.

Run:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Verdict colours are ordered by position in the pipeline, so the palette
# itself carries information: MODEL sits at the detector, PIPELINE in the
# middle at fuse/track/lift, MAPPING last at the class remap. CLEAN is the
# absence of all three. A reader learns the order once, then reads it off the
# colour without consulting a legend.
STAGE = {
    "CLEAN":    {"c": "#0F7A5F", "bg": "#E4F2ED", "where": "no defect"},
    "MODEL":    {"c": "#B4531B", "bg": "#FBEBE0", "where": "at the detector"},
    "PIPELINE": {"c": "#2B5FBF", "bg": "#E6EDFA", "where": "at fuse / track / lift"},
    "MAPPING":  {"c": "#7A3B8F", "bg": "#F1E8F5", "where": "at the class remap"},
}
FAIL = "#B3261E"


# =============================================================================
# Data — pure functions, no Streamlit. Exercised by tests/test_dashboard_data.py
# =============================================================================

def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def list_real_runs(reports_dir=REPORTS):
    """Every real-model report present, excluding the fixture report."""
    d = Path(reports_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("step6_*.json") if p.name != "step6_verify.json")


def strip_signals(r2d, r3d, thresholds=None):
    """Describe what each layer actually measured, as the evidence behind the
    verdict rather than a second copy of it.

    Both layers are read together, because two of the rules that decide a
    verdict cannot be evaluated one layer at a time:

      · track inflation lives only in the 3D report, and produces no confusion,
        no miss and no ghost — a strip that watched only those three would call
        a fragmented run clean and then show PIPELINE beside it;
      · ghosts implicate the detector when they appear in BOTH layers, however
        small the fraction. 894 phantom boxes are 0.05% of 1.66M detections;
        judged per-layer they look like noise, judged as a pair they are the
        whole signal.

    Returns (label_2d, severity_2d, label_3d, severity_3d) where severity is
    'ok' (nothing seen), 'note' (seen, below the threshold, did not decide the
    verdict) or 'bad' (this is what the verdict rests on).
    """
    tau = thresholds or {}
    tc = tau.get("conf", 0.01)
    tl = tau.get("loss", 0.01)
    tr = tau.get("ratio", 1.15)

    def stats(rep):
        if rep is None:
            return None
        tot = rep.get("totals", {})
        matched = sum(rep.get("cells", {}).values())
        gt = tot.get("gt", 0)
        miss, ghost = tot.get("miss", 0), tot.get("ghost", 0)
        return {
            "conf": (tot.get("off_diagonal", 0) / matched) if matched else 0.0,
            "loss": ((miss + ghost) / gt) if gt else 0.0,
            "miss": miss, "ghost": ghost,
            "sys": rep.get("mapping_signature", {}) or {},
            "ratio": rep.get("track_ratio", 1.0),
        }

    s2, s3 = stats(r2d), stats(r3d)
    ghost_both = bool(s2 and s3 and s2["ghost"] > 0 and s3["ghost"] > 0)
    miss_both = bool(s2 and s3 and s2["miss"] > 0 and s3["miss"] > 0)

    def counts(s):
        parts = []
        if s["miss"]:
            parts.append(f"{s['miss']:,} miss")
        if s["ghost"]:
            parts.append(f"{s['ghost']:,} ghost")
        return " + ".join(parts)

    def describe(s):
        if s is None:
            return "n/a", "note"
        # Decisive, in the order diagnose() consults them.
        if s["sys"]:
            n = len(s["sys"])
            return f"systematic relabel · {n} class{'es' if n != 1 else ''}", "bad"
        if s["conf"] >= tc:
            return f"confusion {s['conf']*100:.2f}%", "bad"
        decisive_loss = s["loss"] >= tl \
            or (ghost_both and s["ghost"] > 0) or (miss_both and s["miss"] > 0)
        if decisive_loss and counts(s):
            return counts(s), "bad"
        if s["ratio"] > tr:
            return f"track inflated {s['ratio']:.2f}×", "bad"
        # Seen, but did not move the verdict. Shown rather than hidden: a
        # number the reader can see and dismiss beats a blank they cannot.
        if counts(s):
            return f"{counts(s)} · below threshold", "note"
        return "clean", "ok"

    l2, v2 = describe(s2)
    l3, v3 = describe(s3)
    return l2, v2, l3, v3


def confusion_frame(rep):
    """Confusion matrix as a DataFrame: GT rows, prediction columns, plus a
    MISS column and a GHOST row. Those two bands are why the matrix is not
    square — 'never saw it' and 'saw it, named it wrong' are different failures
    with different fixes, and must never share a cell.
    """
    if not rep:
        return None
    per_class = rep.get("per_class", {})
    cells = rep.get("cells", {})

    # Axis names come from the union of every source, not from per_class alone.
    # A misclassification can name a class that has no ground truth in this
    # scene — on W020 the mapping shift sends Forklift to NovaCarter, and
    # NovaCarter has no GT there. Those land as matched pairs, not ghosts, so
    # keying the axes off per_class would drop the entire column and hide the
    # fault. The same trap cost Step 3 a bug once already.
    names = set()
    for key in cells:
        if "->" in key:
            g, p = key.split("->", 1)
            names.add(g)
            names.add(p)
    names |= {n for n, v in rep.get("miss", {}).items() if v}
    names |= {n for n, v in rep.get("ghost", {}).items() if v}
    names |= {n for n, m in per_class.items() if m.get("support", 0) > 0}

    # Stable order: registry order first, then anything the reports introduced.
    ordered = [n for n in per_class if n in names]
    ordered += sorted(n for n in names if n not in per_class)
    names = ordered
    if not names:
        return None

    grid = {p: {g: 0 for g in names} for p in names}
    for key, v in rep.get("cells", {}).items():
        if "->" not in key:
            continue
        g, p = key.split("->", 1)
        if p in grid and g in grid[p]:
            grid[p][g] = v

    miss_of = rep.get("miss", {})
    ghost_of = rep.get("ghost", {})

    # A class with nothing anywhere — no GT, no prediction, no miss, no ghost —
    # contributes a row and a column of zeros and pushes real classes past the
    # visible width. Dropping it is safe: it still appears in the per-class
    # table below, marked as having no ground truth. The alternative is a
    # clipped row of zeros sitting next to a per-class precision of 1.0000,
    # which reads as a contradiction and costs the reader their trust in the
    # page — the opposite of what this dashboard is for.
    def empty(n):
        return (all(grid[p][n] == 0 for p in names)
                and all(grid[n][g] == 0 for g in names)
                and not miss_of.get(n) and not ghost_of.get(n))

    names = [n for n in names if not empty(n)] or names

    # Long names force wide columns, and past five classes the two panels sit
    # side by side with no room left. Nine characters still separate all seven
    # classes in the registry.
    def short(n):
        return n if len(n) <= 10 else n[:9] + "·"

    lab = {n: short(n) for n in names} if len(names) > 5 else {n: n for n in names}

    df = pd.DataFrame({lab[p]: [grid[p][g] for g in names] for p in names},
                      index=[lab[g] for g in names])
    df["MISS"] = [miss_of.get(g, 0) for g in names]
    df.loc["GHOST"] = [ghost_of.get(p, 0) for p in names] + [0]
    df.index.name = "GT \\ pred"
    return df


def per_class_frame(rep):
    """Per-class metrics. Classes with no GT here are kept and marked, not
    dropped: an absent class can still collect predictions, and every one of
    those is a false positive the macro average cannot see.
    """
    if not rep:
        return None

    # Every column is rendered as text. A column holding both a number and an
    # em dash is a mixed-type column, and Arrow — which Streamlit serialises
    # through — refuses it, so the table fails to draw at all. Formatting here
    # keeps the em dash meaningful without breaking the render.
    def num(v, sup):
        if not sup or not isinstance(v, (int, float)) or v != v:
            return "—"
        return f"{v:.4f}"

    rows = []
    for name, m in rep.get("per_class", {}).items():
        sup = m.get("support", 0)
        rows.append({
            "class": name,
            "support": f"{sup:,}" if sup else "—",
            "precision": num(m.get("precision"), sup),
            "recall": num(m.get("recall"), sup),
            "F1": num(m.get("f1"), sup),
            "miss": f"{m.get('miss', 0):,}",
            "ghost": f"{m.get('ghost', 0):,}",
            "note": "" if sup else "no GT here",
        })
    return pd.DataFrame(rows)


def fixture_table(verify):
    """One row per fixture: the two layer states side by side, and the verdict
    that pair produces. The pair is the point — the verdict is a function of
    the contrast, not of either column alone.
    """
    if not verify:
        return None
    tau = verify.get("thresholds", {})
    thresholds = {"conf": tau.get("tau_conf", 0.01),
                  "loss": tau.get("tau_loss", 0.01),
                  "ratio": tau.get("tau_ratio", 1.15)}
    rows = []
    for r in verify.get("results", []):
        l2, v2, l3, v3 = strip_signals(r.get("report_2d"), r.get("report_3d"),
                                       thresholds)
        rows.append({
            "scene": r.get("scene", "?"),
            "fixture": r.get("error_set", "?"),
            "2D": l2, "sev_2D": v2,
            "3D": l3, "sev_3D": v3,
            "verdict": r.get("got", "?"),
            "expected": r.get("expected") or "—",
            "ok": bool(r.get("passed")),
            "reason": r.get("reason", ""),
        })
    return pd.DataFrame(rows)


def trust_summary(verify):
    """How many verdicts the tool got right on errors whose answer was known.

    Sets with no single correct answer — the compound 'mixed' — are counted
    apart rather than folded in. A pass rate padded with rows that could not
    fail overstates what was actually proven.
    """
    if not verify:
        return None
    res = verify.get("results", [])
    asserted = [r for r in res if r.get("expected")]
    return {
        "passed": sum(1 for r in asserted if r.get("passed")),
        "total": len(asserted),
        "unasserted": len(res) - len(asserted),
        "scenes": sorted({r.get("scene", "?") for r in res}),
        "all_pass": bool(asserted) and all(r.get("passed") for r in asserted),
    }


# =============================================================================
# Render
# =============================================================================

CSS = """
<style>
  .stApp { background: #F4F6F8; }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }

  .eyebrow { font: 600 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
             letter-spacing: .14em; text-transform: uppercase; color: #6B7688; }
  .h1 { font: 700 30px/1.15 ui-sans-serif, system-ui, -apple-system, sans-serif;
        letter-spacing: -.02em; margin: .15rem 0 .35rem; color:#10151F; }
  .sub { color: #56616F; font-size: 14.5px; line-height: 1.55; max-width: 70ch; }

  .card { background:#fff; border:1px solid #E1E5EB; border-radius:10px;
          padding:16px 18px; }
  .metric { font: 700 34px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing:-.02em; color:#10151F; }
  .metric-label { font-size:12.5px; color:#6B7688; margin-top:5px;
                  line-height:1.4; }

  /* signature element: the two layer states, then the verdict they produce */
  .strip { display:flex; align-items:stretch; margin:2px 0 8px;
           border:1px solid #E1E5EB; border-radius:8px; overflow:hidden; }
  .cell { flex:1; padding:9px 13px; background:#fff;
          border-right:1px solid #E1E5EB; }
  .cell .k { font:600 10px/1.3 ui-monospace,monospace; letter-spacing:.12em;
             color:#8A94A3; text-transform:uppercase; }
  .cell .v { font:600 14px/1.4 ui-monospace,monospace; margin-top:3px; }
  .v.ok { color:#0F7A5F; }
  .v.note { color:#6B7688; }
  .v.bad { color:#B4531B; }
  .arrow { display:flex; align-items:center; padding:0 12px; color:#B7BFCA;
           background:#fff; border-right:1px solid #E1E5EB; font-size:15px; }
  .verd { flex:0 0 230px; padding:9px 14px; }
  .verd .k { font:600 10px/1.3 ui-monospace,monospace; letter-spacing:.12em;
             text-transform:uppercase; opacity:.8; }
  .verd .v { font:700 14.5px/1.4 ui-sans-serif,system-ui,sans-serif;
             margin-top:3px; }

  .chip { display:inline-block; padding:3px 10px; border-radius:20px;
          font:700 11.5px/1.6 ui-monospace,monospace; }
  .obs { font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;
         color:#3D4756; }
  .empty { background:#fff; border:1px dashed #C8CFD9; border-radius:10px;
           padding:26px 24px; color:#3D4756; line-height:1.7; }
  .empty code { background:#EDF0F4; padding:2px 6px; border-radius:4px;
                font-size:12.5px; }
  hr { border:0; border-top:1px solid #E1E5EB; margin:1.7rem 0; }
</style>
"""


def show_df(df, **kw):
    """st.dataframe across Streamlit versions. The full-width argument was
    renamed (use_container_width -> width='stretch'), and this project is run
    on more than one machine, so the version present decides which name to
    send rather than a pinned guess."""
    import inspect
    params = inspect.signature(st.dataframe).parameters
    if "width" in params:
        return st.dataframe(df, width="stretch", **kw)
    return st.dataframe(df, use_container_width=True, **kw)


def verdict_chip(v):
    s = STAGE.get(v, {"c": "#6B7688", "bg": "#ECEFF3"})
    return f'<span class="chip" style="background:{s["bg"]};color:{s["c"]}">{v}</span>'


def render_strip(label2d, sev2d, label3d, sev3d, verdict):
    s = STAGE.get(verdict, {"c": "#6B7688", "bg": "#ECEFF3", "where": ""})
    return f"""
    <div class="strip">
      <div class="cell"><div class="k">Layer 2D · detector</div>
        <div class="v {sev2d}">{label2d}</div></div>
      <div class="arrow">→</div>
      <div class="cell"><div class="k">Layer 3D · after fuse / track / remap</div>
        <div class="v {sev3d}">{label3d}</div></div>
      <div class="arrow">=</div>
      <div class="verd" style="background:{s['bg']};color:{s['c']}">
        <div class="k">fault lies</div>
        <div class="v">{verdict} · {s['where']}</div></div>
    </div>"""


def render_layer_panel(title, rep, note=""):
    head = f"**{title}**"
    if note:
        head += f" <span class='obs'>· {note}</span>"
    st.markdown(head, unsafe_allow_html=True)
    if not rep:
        st.caption("no report recorded")
        return

    m = rep.get("macro", {})
    t = rep.get("totals", {})
    a, b, c, d = st.columns(4)
    a.metric("macro P", f"{m.get('precision', float('nan')):.4f}")
    b.metric("macro R", f"{m.get('recall', float('nan')):.4f}")
    c.metric("miss", f"{t.get('miss', 0):,}")
    d.metric("ghost", f"{t.get('ghost', 0):,}")

    cf = confusion_frame(rep)
    if cf is not None:
        st.caption("Rows are ground truth, columns are prediction. Off-diagonal "
                   "means the object was matched but named wrong.")
        show_df(cf)

    pc = per_class_frame(rep)
    if pc is not None:
        show_df(pc, hide_index=True)

    absent = rep.get("absent_class_predictions", {})
    if absent:
        n = sum(absent.values())
        st.warning(f"{n:,} predictions name a class with no ground truth here. "
                   f"Every one is a false positive, and none of them reach the "
                   f"macro average above — {dict(absent)}")


# --- pages ------------------------------------------------------------------

def page_trust(verify):
    if not verify:
        st.markdown(
            '<div class="empty"><b>No fixture report yet.</b><br><br>'
            'Build the report this page reads:<br><br>'
            '<code>python3 scripts/run_validation.py</code><br><br>'
            'It runs both layers against every synthetic fixture on W020 and '
            'W011 at full length and writes '
            '<code>reports/step6_verify.json</code>.</div>',
            unsafe_allow_html=True)
        return

    s = trust_summary(verify)
    st.markdown('<div class="eyebrow">The tool measuring itself</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="h1">Does this tool name the right stage?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Every fixture below is ground truth with one known '
        'fault injected on purpose, so the correct answer is known in advance '
        'and the tool can be graded against it. Until real model output '
        'arrives, this is the only claim here that has been proven — and it is '
        'the one to demand first, because numbers from an unchecked validator '
        'are merely confident.</div>',
        unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    colour = STAGE["CLEAN"]["c"] if s["all_pass"] else FAIL
    c1.markdown(f'<div class="card"><div class="metric" style="color:{colour}">'
                f'{s["passed"]}/{s["total"]}</div>'
                f'<div class="metric-label">verdicts correct on faults whose '
                f'answer was known</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="metric">{len(s["scenes"])}</div>'
                f'<div class="metric-label">source scenes, one per warehouse '
                f'family · '
                f'{", ".join(s["scenes"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="metric">9,000</div>'
                f'<div class="metric-label">frames per fixture — full length, '
                f'where density effects appear</div></div>',
                unsafe_allow_html=True)

    if not s["all_pass"]:
        st.error("At least one verdict is wrong. Numbers from this tool on a "
                 "real model cannot be trusted until every row below passes.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Read left to right</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">A verdict comes from the <i>pair</i> of layer states, '
        'never from one. A miss visible in 2D belongs to the detector; the same '
        'miss visible only in 3D belongs to a later stage. Producing that '
        'contrast is the entire reason there are two layers.</div>',
        unsafe_allow_html=True)
    st.write("")

    df = fixture_table(verify)
    scene = st.selectbox("Scene", sorted(df["scene"].unique()))
    sub = df[df["scene"] == scene]

    for _, r in sub.iterrows():
        mark = "" if r["ok"] else (f' <span style="color:{FAIL};font-weight:700">'
                                   f'✗ expected {r["expected"]}</span>')
        st.markdown(f'<div class="eyebrow" style="margin-top:15px">'
                    f'{r["fixture"]}{mark}</div>', unsafe_allow_html=True)
        st.markdown(render_strip(r["2D"], r["sev_2D"], r["3D"], r["sev_3D"],
                                 r["verdict"]), unsafe_allow_html=True)
        st.markdown(f'<div class="obs">{r["reason"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Inspect one fixture</div>',
                unsafe_allow_html=True)
    pick = st.selectbox("Fixture", list(sub["fixture"]))
    rec = next(r for r in verify["results"]
               if r.get("scene") == scene and r.get("error_set") == pick)

    st.markdown(verdict_chip(rec.get("got", "?")) + f' &nbsp; {rec.get("reason","")}',
                unsafe_allow_html=True)
    obs = rec.get("observations") or []
    if obs:
        st.caption("Everything both layers saw. The verdict names only the "
                   "dominant cause, so on a compound fault the rest of the "
                   "story is here, not there.")
        for o in obs:
            st.markdown(f'<div class="obs">· {o}</div>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    with left:
        render_layer_panel("Layer 2D — does the model see the object?",
                           rec.get("report_2d"), "File B, before any 3D work")
    with right:
        r3 = rec.get("report_3d")
        note = "File A, after fuse / track / remap"
        if r3 and r3.get("track_ratio", 1.0) > 1.0:
            note += f" · track ratio {r3['track_ratio']:.2f}×"
        render_layer_panel("Layer 3D — is the class still right?", r3, note)


def page_model(runs):
    st.markdown('<div class="eyebrow">The model under test</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="h1">How did the model do?</div>',
                unsafe_allow_html=True)

    if not runs:
        st.markdown(
            '<div class="empty"><b>No model output yet.</b><br><br>'
            'This page fills in as soon as the training stage hands over its '
            'two files. Put them here, one folder per model version:<br><br>'
            '<code>data/input/&lt;version&gt;/&lt;scene&gt;/track1.txt</code><br>'
            '<code>data/input/&lt;version&gt;/&lt;scene&gt;/detections_2d.txt'
            '</code><br><br>then run:<br><br>'
            '<code>python3 scripts/run_validation.py --input &lt;version&gt;'
            '</code><br><br>'
            'Results land in <code>reports/</code> and appear here. Nothing '
            'else needs changing — the tool is already verified against known '
            'faults on the first tab.</div>',
            unsafe_allow_html=True)
        return

    names = [p.stem.replace("step6_", "") for p in runs]
    pick = st.selectbox("Model version", names)
    data = load_json(runs[names.index(pick)])
    if not data:
        st.error("That report could not be read. Re-run "
                 "`scripts/run_validation.py --input " + pick + "`.")
        return

    for block in ("CLEAN", "SEEN"):
        rows = data.get("blocks", {}).get(block)
        if rows is None:
            continue
        if block == "CLEAN":
            st.markdown('<div class="eyebrow" style="margin-top:20px">CLEAN — '
                        'scenes the model never trained on</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sub">These are the numbers that predict '
                        'the test score.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="eyebrow" style="margin-top:20px">SEEN — '
                        'scenes the model trained on</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sub">An upper bound, not a prediction. '
                        'Included only because two of the seven classes have no '
                        'ground truth anywhere else. Never averaged into the '
                        'block above.</div>', unsafe_allow_html=True)
        if not rows:
            st.caption("no scenes in this block")
            continue
        for row in rows:
            v = row.get("verdict", {})
            st.markdown(f'<div class="eyebrow" style="margin-top:16px">'
                        f'{row.get("scene","?")}</div>', unsafe_allow_html=True)
            st.markdown(verdict_chip(v.get("verdict", "?"))
                        + f' &nbsp; {v.get("reason","")}', unsafe_allow_html=True)
            for o in v.get("observations", []):
                st.markdown(f'<div class="obs">· {o}</div>',
                            unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                render_layer_panel("Layer 2D", row.get("report_2d"))
            with b:
                render_layer_panel("Layer 3D", row.get("report_3d"))


def main():
    st.set_page_config(page_title="Track 1 — Validation", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    verify = load_json(REPORTS / "step6_verify.json")
    runs = list_real_runs()

    tab1, tab2 = st.tabs(["Can the tool be trusted?", "How did the model do?"])
    with tab1:
        page_trust(verify)
    with tab2:
        page_model(runs)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Every figure is read from reports/ exactly as the layers "
               "recorded it. This page computes nothing — a dashboard that "
               "recalculates becomes a second source of truth, and two sources "
               "drift apart.")


if __name__ == "__main__":
    main()