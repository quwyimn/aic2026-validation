# Validation — AI City Challenge 2026, Track 1

Quality-control stage for the Track 1 model. Takes the model's output files, compares them against ground truth, reports whether object classes are correct — and if not, which pipeline stage is at fault.

Scope: **validation only**. This project does not train and does not modify the model.

> **Is this validator actually correct?** Yes — and not on our own say-so: the official NVIDIA HOTA scorer, run independently on our own fixtures, agrees with the diagnosis. The full evidence is in **[PROOF.md](docs/PROOF.md)**.

📋 **[Full plan → docs/roadmap_validate.md](docs/roadmap_validate.md)**  ·  📈 **[Progress → docs/PROGRESS.md](docs/PROGRESS.md)**  ·  ✅ **[Proof → PROOF.md](docs/PROOF.md)**

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Step 1 — survey every GT scene, write config/class_registry.json
python3 scripts/inspect_gt.py --gt-root data/gt

# Step 2 — build the synthetic fixtures with their answer key
python3 scripts/make_synthetic.py --config config/config.yaml

# Step 3 — the 2D layer. Run `clean` first: it must score a perfect 1.000.
python3 scripts/run_layer2d.py --scene Warehouse_020 --set clean

# Step 5 — prove the layers reproduce every injected error
python3 scripts/verify_validator.py

# Step 6 — prove the diagnosis names the right stage for each
python3 scripts/run_validation.py

# Step 7 — read the results on screen
streamlit run dashboard/app.py

# Preflight — will the official scorer even accept a submission file?
#   (no ground truth needed; the only check that runs on the test set)
python3 scripts/preflight_submission.py --input data/input/v1/track1.txt

# External check — confirm the diagnosis against the official HOTA scorer.
#   Not a pipeline step; an independent cross-check. See PROOF.md for results
#   and the environment the scorer needs.
python3 scripts/gt_to_txt.py --scene Warehouse_020 \
    --self-check data/synthetic/Warehouse_020/clean/track1.txt
```

The GT is not in this repo — download the val and train splits from the
organizers into `data/gt/<split>/<scene>/ground_truth.json` first. See
[step 1 of the roadmap](docs/roadmap_validate.md) for which files are needed.

Steps 5 and 6 exit 0 on all-pass and 1 otherwise, so both drop into CI as they
are. Run them at full length: every bug this project has caught so far was
invisible at 500 frames.

---

## Structure

```
validation/
├── README.md
├── PROOF.md                     # the evidence the validator is correct
├── requirements.txt
├── scene_map.json               # scene_id -> name, for the official scorer
├── config/
│   ├── config.yaml
│   └── class_registry.json      # written by inspect_gt.py
├── docs/
│   ├── roadmap_validate.md
│   └── PROGRESS.md
├── data/
│   ├── gt/                       # ground_truth.json per scene (from organizers)
│   ├── gt_txt/                   # GT as 11-column txt (for the official scorer)
│   ├── input/                    # File A + File B from the training stage
│   └── synthetic/                # fixtures with injected errors + answer keys
├── src/
│   ├── io/
│   │   ├── loaders.py            # GT / File A / File B parsing
│   │   └── preflight.py          # submission-format gate, no GT needed
│   ├── matching/
│   │   └── assign.py             # IoU, 3D distance, Hungarian (class-blind)
│   ├── metrics/
│   │   └── confusion.py          # confusion matrix, precision/recall, signatures
│   ├── layers/
│   │   ├── layer_2d.py           # File B vs 2D GT
│   │   └── layer_3d.py           # File A vs 3D GT
│   └── diagnose/
│       ├── verify.py             # check a layer result against the answer key
│       └── verdict.py            # cross-reference both layers into the verdict
├── scripts/
│   ├── inspect_gt.py             # Step 1  — survey the GT
│   ├── make_synthetic.py         # Step 2  — build fixtures + answer keys
│   ├── run_layer2d.py            # Step 3  — run the 2D layer on one fixture
│   ├── run_layer3d.py            # Step 4  — run the 3D layer on one fixture
│   ├── verify_validator.py       # Step 5  — assert layers reproduce injected errors
│   ├── run_validation.py         # Step 6  — cross-reference into a verdict
│   ├── preflight_submission.py   # preflight — will the scorer accept this file?
│   └── gt_to_txt.py              # external check — GT -> txt for the HOTA scorer
├── reports/                      # step5_verify.json, step6_verify.json, step6_<version>.json
├── dashboard/
│   └── app.py                    # Step 7  — Streamlit dashboard
└── tests/
    ├── test_verdict.py
    ├── test_dashboard_data.py
    └── test_preflight.py
```

> Regenerate this listing any time to confirm it still matches disk:
> `find . -path ./venv -prune -o -name '*.py' -print -o -name '*.md' -print | sort`

---

## Folders

### `config/`
Single source of truth for every tunable value: paths, IoU threshold, 3D distance threshold, diagnosis thresholds, the official class_id table. Nothing hardcoded anywhere else — one file to change, one place to look when a number seems wrong. `class_registry.json` is generated by `inspect_gt.py`, not edited by hand.

### `docs/`
Planning and reference documents. [`roadmap_validate.md`](docs/roadmap_validate.md) is the one to read first: it carries the handoff format agreed with the training stage, the findings from the GT survey, and the eight-step plan. [`PROGRESS.md`](docs/PROGRESS.md) records what each finished step proved, with the numbers.

### `data/`
All input data. Nothing here is generated by `src/` except `synthetic/`.

- **`gt/`** — ground truth downloaded from the organizers. The answer key. Read-only; never edited.
- **`input/`** — `track1.txt` (File A) and `detections_2d.txt` (File B) received from the training stage. One subfolder per model version so results stay comparable across runs.
- **`synthetic/`** — fake File A / File B generated from the GT with deliberately injected errors, each set shipping an `injected_errors.json` answer key. Used to prove the validator itself works.

### `src/`
Library code. No entry points here — everything is importable, nothing runs on import.

- **`io/`** — reading and parsing. GT loader with schema auto-detection (the abbreviated variant `type` / `loc3d` / `bbox3d` / `rot3d` never appears in the 23 scenes surveyed, but detection is kept as insurance for the test set), File A parser, File B parser. Format validation lives here: wrong column count, out-of-range class_id, and unit anomalies are caught at load time, not three stages later. `preflight.py` answers a narrower question with no ground truth: *will the official scorer even accept this file?* — every FATAL rule traces to a line of the official scorer's own code (see PROOF.md), and it is the only check that runs on the test set, where GT is withheld.
- **`matching/`** — pairing predictions to GT. 2D IoU, 3D Euclidean distance, Hungarian assignment. **All matching here is class-blind by design** — matching on geometry alone is what makes misclassification visible instead of hiding it as a miss + ghost pair.
- **`metrics/`** — turning matched pairs into numbers. Confusion matrix (with miss/ghost bands), precision, recall, per-class support, and the systematic-mapping signature that separates a miswired lookup from ordinary confusion.
- **`layers/`** — the two validation layers. `layer_2d.py` measures File B against 2D GT (does the model recognize the object?). `layer_3d.py` measures File A against 3D GT (after fuse/track/remap, is the class still right?). Both are thin: they wire `io` → `matching` → `metrics` together.
- **`diagnose/`** — `verify.py` checks each layer result against the injected-error answer key; `verdict.py` cross-references the two layers into the verdict: MODEL / MAPPING / PIPELINE / CLEAN. This is the part that answers *where* the problem is, not just *that* there is one.

> **Note on synthetic generation.** Error injection lives in `scripts/make_synthetic.py`, not under `src/`. It is the one place where a script carries real logic rather than wrapping it, and it stays there deliberately: that code underpins every fixture the validator has been proven against, and reorganising it to match the pattern would risk the fixtures to gain nothing but symmetry.

### `scripts/`
Command-line entry points. Thin wrappers over `src/` — argument parsing and orchestration only — with the one exception noted above.

- `inspect_gt.py` — print the class list, per-class object counts, and detected schema of a GT file. First thing to run after downloading data.
- `make_synthetic.py` — build the synthetic dataset with injected errors, and its answer key.
- `run_layer2d.py` / `run_layer3d.py` — run one layer against one fixture. Useful while iterating; not the acceptance test.
- `verify_validator.py` — Step 5. Runs both layers against every synthetic set and asserts each result matches the injected-error key.
- `run_validation.py` — Step 6. Cross-references both layers into a verdict. Asserts the verdict on the fixtures by default; `--input <version>` runs it against real model output instead.
- `preflight_submission.py` — run `preflight` on a `track1.txt`. Exit 0 clean, 1 on any FATAL, so it drops into CI. The submission-format gate that runs even where there is no ground truth.
- `gt_to_txt.py` — convert `ground_truth.json` to the 11-column txt the official HOTA scorer needs, using the validator's own loader so there is no second parser to disagree. `--self-check` asserts the output matches a fixture key-for-key.

### `reports/`
Run output as JSON — machine-readable, for the dashboard and for diffing across versions. Two kinds live here, and they answer different questions:

| File | Answers | Lifetime |
|---|---|---|
| `step5_verify.json`, `step6_verify.json` | is the **tool** sound? | regenerated on demand; only the latest matters |
| `step6_<version>.json` | is the **model** sound? | kept forever — comparing version N against N-1 is the point |

Steps 1–4 write no report. They print to the terminal or write `config/class_registry.json` directly, and a report exists only where someone will read it back.

### `dashboard/`
Streamlit app. Reads `reports/` and computes nothing itself — a dashboard that recalculates becomes a second source of truth, and two sources drift apart.

Two tabs. The first shows the fixture results: each fault type, the state of both layers, and the verdict that pair produces. The second shows the real model run once File A and File B arrive; until then it says what to put where. Verdict colours are ordered by position in the pipeline, so the palette itself carries the diagnosis.

```bash
streamlit run dashboard/app.py
```

### `tests/`
Assertions that run without ground truth, so they stay fast and work on any machine:

- `test_verdict.py` — feeds `diagnose()` report structures at the magnitudes measured on the real fixtures and asserts the verdict for each.
- `test_dashboard_data.py` — the dashboard's data layer: layer states, confusion frames, per-class tables, and the empty-report cases.
- `test_preflight.py` — 33 cases, and the reason it is trustworthy is that every rule has both a case it must catch and a case it must stay silent on. A checker that flags everything is as useless as one that flags nothing.

The heavier verification — running the real layers against `data/synthetic/` and asserting the reported errors match the injected ones — is `scripts/verify_validator.py` and `scripts/run_validation.py`, because both need the GT on disk.

---

## External verification

Everything above proves the validator is *self-consistent*: fixtures the team built, checked by a tool the team built. Necessary, but it cannot by itself rule out a shared mistake — if the fixture generator and the validator misread the same thing, both agree and both are wrong.

The one thing that breaks that loop is running the **official NVIDIA HOTA scorer** (`spatialai-data-utils`) on the team's own fixtures: an independent parser, an independent 3D-IoU matcher, an independent metric. On `clean` it returns HOTA = 100.00; on `mapping_shift` it returns ≈ 0 — both predicted in advance, and the `clean = 100` in particular means the file was accepted and scored as perfect by the organizers' own code.

The full evidence, what each layer does and does *not* establish, and how a sceptic reproduces every number, is in **[PROOF.md](docs/PROOF.md)**.

---

## Flow

```
data/gt/  ──┬──────────────────────────────► src/layers ──► reports/ ──► dashboard/
            │                                    ▲
            └──► make_synthetic ──► data/synthetic┤
                                                  │
data/input/ (from training stage) ──┬─────────────┘
                                     └──► preflight ──► ACCEPT / REJECT (no GT needed)

data/gt/ ──► gt_to_txt ──► data/gt_txt/ ──┐
                                          ├──► official HOTA scorer ──► matches the verdict
data/synthetic/ ──────────────────────────┘
```

Ground truth feeds both the real run and the synthetic run. The synthetic path proves the layers are correct before the real files arrive; the HOTA path proves that proof isn't circular.