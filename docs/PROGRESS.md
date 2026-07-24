# Progress — Validation, AI City Challenge 2026 Track 1

One entry per step. Each records what was completed and the evidence for it —
numbers from the synthetic fixtures, which anyone on the team can reproduce.
Click a step to expand the commands and detail.

Full plan: [docs/roadmap_validate.md](roadmap_validate.md)

| Step | What | Status |
|---|---|---|
| 1 | Survey the ground truth | ✅ done |
| 2 | Build synthetic fixtures | ✅ done |
| 3 | Layer 2D — detector check | ✅ done |
| 4 | Layer 3D — pipeline check | ✅ done |
| 5 | Verify the validator | ✅ done |
| 6 | Cross-reference the two layers | ✅ done |
| 7 | Dashboard | ◻ next |
| 8 | Run on real model output | ◻ |

---

## Step 1 — Survey the ground truth ✅

Scanned all 23 GT scenes and wrote `config/class_registry.json`. This settles
the facts everything else depends on, rather than assuming them.

**Evidence.** The class table is closed at 7 classes with none unknown.
PalletTruck — absent from the organizers' published table — is present in the
GT (97 objects across 13 scenes), so it is kept as class id 6. The class
coverage per split is the finding that reshaped the plan:

| | Person | Forklift | PalletTruck | NovaCarter | Transporter | AgilityDigit | FourierGR1T2 |
|---|---|---|---|---|---|---|---|
| in val | ✅ | ✅ | ✅ | 2 obj | 2 obj | ❌ | ❌ |

Two classes have no ground truth in val at all — so validating on val alone is
blind to them.

<details>
<summary>Detail & commands</summary>

```bash
python3 scripts/inspect_gt.py --gt-root data/gt
```

- All 23 scenes use the full schema (`object type`, `3d location`, …); the
  abbreviated variant never appears. Auto-detection is kept only as insurance
  for the test set.
- Rotation is Euler `[0, 0, yaw]` — the vertical axis is z, not the y-axis the
  organizers' spec text describes. The data is the source of truth here.
- The dataset splits into two warehouse families: family A (~100 m, Person /
  Forklift / PalletTruck) covers W000–007, W017–022; family B (~16 m, the
  robot classes) covers W008–016. Val lies in family A.
- Consequence: **W011** (family B, 6 of 7 classes) is needed for class
  coverage. Its metrics go in a separate **SEEN** block, because the model
  trained on it — see the CLEAN/SEEN rule in the roadmap.
- Scene pairs W000/001, W002/003, W004/005, W006/007, W020/021 are byte
  duplicates. Not independent samples.

</details>

---

## Step 2 — Build synthetic fixtures ✅

`make_synthetic.py` turns the GT into fake model output with deliberately
injected errors, plus an `injected_errors.json` answer key stating exactly what
was broken. This is what makes the validator itself testable in Step 5 — you
cannot check a measuring tool against data whose true answer you don't know.

**Evidence.** 18 fixture sets (9 error types × W020 and W011). The `clean` set
is a faithful copy of the GT — 610,776 rows for W020, matching the detection
count reported by the survey exactly. Each error type is designed to leave a
different fingerprint across the two layers, which is what lets Step 6 tell
them apart:

| Error set | Layer 2D | Layer 3D |
|---|---|---|
| clean | perfect | perfect |
| class_swap | wrong | wrong |
| mapping_shift | **clean** | wrong, systematic |
| position_shift_severe | **clean** | miss + ghost |
| fragmentation | **clean** | id count inflated |
| deletion | miss | miss |
| phantom | ghost | ghost |

<details>
<summary>Detail & commands</summary>

```bash
python3 scripts/make_synthetic.py --config config/config.yaml
```

- Victims are planned up front from the known object list, so the answer key
  states exact counts (e.g. "4 class swaps", "450 phantoms") rather than
  probabilistic ones. A fixture whose contents depend on luck is not a fixture.
- Seeded (`seed: 42`) — the same config reproduces the same fixture byte for
  byte.
- `mapping_shift` is the important one: it writes File B (raw 2D) with the
  correct class table but File A (3D) with a shifted one. That asymmetry is the
  signature of a miswired remap, and it is invisible unless both layers are
  measured.
- Defaults to full length (9000 frames). Override with `--frames` while
  iterating.

</details>

---

## Step 3 — Layer 2D: does the model see the object? ✅

Compares File B (raw per-camera detections, before any 3D lifting or tracking)
against the 2D boxes in the GT. Class-blind IoU + Hungarian matching, confusion
matrix with miss/ghost bands, macro-averaged metrics. Because it sits right
after the detector, anything it reports belongs to the detector alone.

**Evidence.** Verified against the fixtures on both warehouse families.

| Fixture (W020, 9000 frames) | Macro P | Macro R | Reads as |
|---|---|---|---|
| clean | 1.0000 | 1.0000 | perfect — the acceptance test |
| mapping_shift | **1.0000** | **1.0000** | blind, correctly — fault is downstream |
| class_swap | 0.9908 | 0.9641 | model error, visible here |
| deletion | 1.0000 | 0.9700 | 63,469 misses |
| phantom | 0.9996 | 1.0000 | 894 ghosts |

`mapping_shift` scoring a perfect 1.000 is the point, not a bug: the detector
did its job, so this layer stays silent and leaves the fault for Layer 3D to
catch.

<details>
<summary>Detail, the two bugs this step caught, & commands</summary>

```bash
python3 scripts/run_layer2d.py --scene Warehouse_020 --set clean
python3 scripts/run_layer2d.py --scene Warehouse_020 --set mapping_shift
```

Two real bugs surfaced because `clean` refused to score a perfect 1.000:

1. **Zero-area GT boxes.** The GT carries boxes with no area — objects occluded
   to nothing or clipped at the frame edge (3391 per 500 frames on W020). They
   match nothing, so `clean` scored 0.948. Both sides now filter by
   `min_box_area_px`, and the excluded count is reported rather than dropped
   silently — charging the model with a miss for an object that occupies no
   pixels would be wrong.

2. **False positives on absent classes.** On W020 only 3 of 7 classes are
   present, so a misclassification can land on a class with no GT — and the
   macro average, which skips classes with no GT, never saw it. `class_swap`
   was hiding 73,549 such false positives behind a precision of 1.000. They are
   now counted and reported in their own block.

Both were caught by the fixture, not by luck — which is the entire reason the
`clean` set exists.

</details>

---

## Step 4 — Layer 3D: after fuse/track/remap, is the class still right? ✅

Compares File A (`track1.txt`, the submitted 3D output) against the GT 3D
locations. Euclidean distance matching in world space, class-blind, Hungarian —
the same discipline as Layer 2D, in meters instead of pixels. Each track is
resolved to a single class (majority vote), matching how the pipeline assigns
one class per object.

**Evidence.** The three fixtures that Layer 2D was blind to must all become
visible here — that inversion is the whole reason two layers exist.

| Fixture (W020, 9000 frames) | Layer 2D | Layer 3D | Diagnosis |
|---|---|---|---|
| clean | 1.0000 | 1.0000 | — |
| mapping_shift | 1.0000 | **0.0000**, `Person→Forklift 100%` | mapping error |
| position_shift_severe | 1.0000 | **0.8914**, 47,979 miss + 47,979 ghost | pipeline error |
| fragmentation | 1.0000 | class 1.000, **track ratio 1.20** | broken association |
| class_swap | 0.964 | **0.966** | model error |

The contrast between the two layers is the diagnosis: `mapping_shift` clean in
2D but 100%-wrong in 3D can only be the remap; `class_swap` wrong in both is the
model itself.

<details>
<summary>Detail & commands</summary>

```bash
python3 scripts/run_layer3d.py --scene Warehouse_020 --set clean
python3 scripts/run_layer3d.py --scene Warehouse_020 --set mapping_shift
```

- `mapping_shift` reports the systematic pattern explicitly: every Person →
  Forklift, every Forklift → NovaCarter, at 100%. A model that confused objects
  would err here and there; erring on every single instance in one fixed
  direction is a constant, i.e. a wiring fault.
- `position_shift_severe` produces scattered miss+ghost rather than a clean
  100% shift — that difference is what separates a pipeline error from a
  mapping error at diagnosis time. Each displaced object yields exactly one
  miss **and** one ghost: 47,979 of each on W020, 43,984 on W011.
- `fragmentation` keeps classes perfect while the predicted-track count exceeds
  the GT-track count. The coarse ratio warning fires above 1.15; the exact
  per-track evidence is checked against the answer key in Step 5.
- A track that wears more than one class across its frames is flagged as
  `flicker` — an object's class should be stable, so that is itself a defect.

</details>

---

## Step 5 — Verify the validator ✅

Runs both layers against every synthetic set and checks each result against
that set's `injected_errors.json` answer key. This is the step that measures
the validator instead of a model: if it cannot reproduce errors whose answer
is known, its numbers on a real model mean nothing. Reading 18 sets by eye
does not scale and does not survive a code change — from here it is an
assertion that passes or fails.

**Evidence.** All 18 sets pass on both scenes at full length (9000 frames).

```
RESULT: 18/18 sets passed
The validator reproduces every injected error. It can be trusted
to measure a real model.
```

Each expectation is phrased as a relationship between the two layers, because
that relationship — not either number alone — is what the design promises. For
example `mapping_shift` requires 2D clean **and** every present class flagged
at ~100% in 3D; `class_swap` requires confusion in **both** layers.

<details>
<summary>Detail, the two check bugs full-scale runs caught, & commands</summary>

```bash
python3 scripts/verify_validator.py            # all source scenes, full length
python3 scripts/verify_validator.py --scene Warehouse_020 --frames 500  # fast
```

Exit code is 0 on all-pass, 1 otherwise, so it drops into CI directly.

Two bugs surfaced — both in the *checks*, not the validator, and both hidden by
short smoke-test runs:

1. **mapping_shift check compared against the whole 7-class table.** W020 holds
   only 3 classes, so it can flag at most 3, and the check demanded 6 → false
   FAIL. Fixed to compare against classes actually present in the scene. The
   validator was right the whole time; the check was wrong, and only a
   side-by-side number comparison caught it.

2. **position_shift check demanded exactly zero class confusion.** At 9000
   frames on a crowded scene, a badly-displaced object occasionally lands
   within 1m of a different-class neighbour, so class-blind matching pairs
   them — 56 out of 48,000 (0.01%). That is correct behaviour, not a defect.
   The check now allows negligible confusion (<1% of matched) and only fails
   when confusion becomes the signal. Measured noise: 0.01–0.07% across both
   scenes.

The lesson both times: verify at full length before declaring a step done.
Small runs don't reach the density where these effects appear — the same thing
happened with zero-area boxes in Step 3.

</details>

---

## Step 6 — Cross-reference the two layers ✅

Reads both layer reports and names the stage at fault — MODEL / MAPPING /
PIPELINE / CLEAN — automatically. Up to Step 5 the two layers produced numbers
that a human read side by side; from here the contrast between them is a
verdict the tool states itself.

**Evidence.** 16/16 diagnostic verdicts correct on both warehouse families at
full length (9000 frames). Each verdict is derived from the *relationship*
between the layers, never from either number alone:

| Fixture (W020) | Layer 2D | Layer 3D | Verdict |
|---|---|---|---|
| clean | clean | clean | CLEAN |
| class_swap | conf 4.98% | conf 5.55% | **MODEL** |
| mapping_shift | clean | all present classes ~100% | **MAPPING** |
| position_shift | clean | clean (under 1m) | CLEAN |
| position_shift_severe | clean | miss+ghost 15.71% | **PIPELINE** |
| deletion | miss 3.83% | miss 5.21% | **MODEL** |
| phantom | ghost 894 | ghost + track 6.70x | **MODEL** |
| fragmentation | clean | track 1.20x, no ghost | **PIPELINE** |

The decisive question is **"is Layer 2D clean"**, not "is there a miss/ghost".
`deletion` and `position_shift_severe` both produce large miss+ghost and are
indistinguishable by magnitude; one shows in 2D (the detector's fault) and the
other only in 3D (a later stage's). That single distinction is what the whole
two-layer design buys.

On W011 the same logic holds against a 6-class table: `mapping_shift` flags the
full offset-1 cycle — Person→Forklift→NovaCarter→Transporter→FourierGR1T2→
AgilityDigit→PalletTruck — proving the rule does not depend on how many classes
a scene happens to hold.

<details>
<summary>Detail, the bug a full-length run caught, & commands</summary>

```bash
python3 tests/test_verdict.py            # logic only, no GT needed
python3 scripts/run_validation.py        # both scenes, full length
python3 scripts/run_validation.py --input v1_option_d   # real model run
```

Exit code 0 on all-pass, 1 otherwise — drops into CI beside Step 5.

**One real bug, caught only at full length.** The first version gated *every*
dimension by a fraction of GT. `phantom` injects 894 false boxes against 1.66M
2D detections — 0.05%, far under the 1% gate — so Layer 2D read as clean, the
fault fell through to the 3D branch, hit the 6.70x track inflation, and was
diagnosed **PIPELINE instead of MODEL**. A fabricated object was blamed on the
tracker.

The fix was to stop treating the two dimensions alike:

- **Confusion** stays fraction-based. Class-blind matching on a dense scene
  genuinely does pair a displaced object with a wrong-class neighbour; that
  trace scales with crowding and must be tolerated (measured 0.01–0.07%).
- **Miss / ghost** is not. Density produces confusion, never a fabricated or
  dropped match — so on a clean layer these sit at zero and any nonzero value
  is signal. What implicates the detector is the same loss appearing in **both**
  layers, regardless of how small the fraction is.

`phantom` now routes MODEL on the presence of ghosts in both layers, and
`fragmentation` stays PIPELINE because it re-ids real objects and produces no
ghosts at all. Confirmed on two scenes with unrelated magnitudes — 894 ghosts /
6.70x on W020, 909 / 9.18x on W011.

The unit test that had passed this case was itself at fault: it gave `phantom`
a 5% ghost rate, a round guess that no fixture produces. Test numbers that
don't match reality hide the exact bug the test exists to catch. It now carries
the magnitudes transcribed from the machine run.

**Known limit — compound faults.** `mixed` reports MODEL, and that is the
correct dominant cause (2D confusion 6.40%), but the same run also carries a
systematic relabel on 2 classes and 6.75x track inflation. A single verdict
names one stage; the rest lives in the `observations` list attached to every
result. The Step 7 dashboard must surface observations alongside the verdict,
or a real compound fault will hide behind one reassuring word.

**Thresholds** live in `config.yaml` under `diagnosis:` — nothing hardcoded in
`src/`. They mirror the tolerances already proven in Step 5.

</details>

---

## Step 7 — Dashboard ✅

Puts the results on screen for the team. Reads reports/ and computes nothing: a dashboard that recalculates becomes a second source of truth, and two sources drift apart.

Two tabs, in this order. Can the tool be trusted? shows the fixture results — known faults, and whether the tool named the right stage for each. How did the model do? stays empty until the training stage hands over File A and File B, and says what to put where in the meantime.

Evidence. The data layer is asserted without a browser (tests/test_dashboard_data.py), and the app itself is rendered head­lessly through Streamlit's AppTest in all three states it can be opened in: fixture report present, nothing at all, and a real model run present. No exception in any of them.

The centrepiece is one row per fixture — the two layer states, then the verdict they produce:

fragmentation   clean  →  track inflated 1.22×   =  PIPELINE · at fuse / track / lift
deletion        61,718 miss  →  27,000 miss      =  MODEL · at the detector

Reading the rows side by side is the argument: deletion and position_shift_severe both lose objects, and the only thing separating them is which layer saw it. Verdict colours are ordered by position in the pipeline — MODEL at the detector, PIPELINE at fuse/track, MAPPING at the remap — so the palette carries the diagnosis rather than decorating it.

<details> <summary>Detail, the three display bugs this step caught, & commands</summary>
bash
python3 tests/test_dashboard_data.py     # no GT needed, runs anywhere
streamlit run dashboard/app.py

scripts/run_validation.py now persists the full layer reports in verify mode. Without them the dashboard has verdicts but no confusion matrices, so reports/step6_verify.json must be regenerated after that change.

Three bugs, none of which touched a verdict — the diagnosis was right the whole time, the screen described it wrongly. On a page whose entire job is to earn trust, that is not a lesser class of bug:

The confusion matrix silently dropped columns. Axis names came from per_class, which holds only classes with GT. But mapping_shift on W020 sends Forklift → NovaCarter, and NovaCarter has no GT there; those land as matched pairs, not ghosts, so the whole column vanished. This is the same trap as the absent-class false positives in Step 3. Axes now come from the union of cells, miss, ghost and per_class.
The per-class table would not render. A column holding both a number and an em dash is mixed-type, and Arrow — which Streamlit serialises through — refuses it. Every cell is now formatted as text.
The layer strip contradicted the verdict on two rows. It watched confusion, miss and ghost per layer, while the verdict also weighs track inflation and the ghost-in-both-layers rule. So fragmentation showed two clean layers beside a PIPELINE verdict, and position_shift showed an orange 3D layer beside a CLEAN one. The strip now reads both reports together and states the evidence at three levels: clean, seen-but-below- threshold, and decisive. verdict.py gained the 3D counterpart of the "small but present" observation so the CLI and the dashboard describe the same run identically.

Bug 3 is the one worth remembering. The rule it broke is that a reader must never be asked to reconcile two parts of the same screen — and the fix was not to soften the strip toward the verdict, but to make it report what the verdict actually rests on.

Known limit — compound faults. mixed displays MODEL while its 3D cell reads systematic relabel · 5 classes. Both are true: 2D is dirty, so MODEL wins outright, because a detector that already mislabels cannot be rescued downstream. The rest of the story is in the observations list under each fixture. Say this out loud when presenting — a single verdict naming one stage is a design decision, not an oversight.

What the fixture numbers confirm on their own. Two counts that must hold by construction, and do: the 3D row count divided by 9000 frames comes out to whole objects (W011: 20 Person, 3 Forklift, 3 NovaCarter, 9 Transporter, 10 FourierGR1T2, 10 AgilityDigit = 55, against 494,981 rows where 55 × 9000 = 495,000), and 2D exceeds 3D by ~2.3×, the mean number of cameras that see an object. Neither depends on the validator being correct, which is what makes them worth watching — and both work on the test set, where there is no ground truth to check against at all.

</details>

---

## Steps 8 — not started

- **Step 8** runs the whole thing on real output from the training stage. By
  then it is just pointing the finished tool at their files.