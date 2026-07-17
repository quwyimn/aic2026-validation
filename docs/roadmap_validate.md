# Validation Roadmap — AI City Challenge 2026, Track 1

## 0. Objective

Validation is the quality-control stage that checks the model's output before submission. It answers three questions:

1. Does the model assign the correct class to each object? (Is a Person ever labeled as a PalletTruck?)
2. If something is wrong — which stage of the pipeline is it wrong in?
3. How wrong, on which class, and where in the warehouse?

Validation does **not** modify the model and does **not** retrain. It takes the output files, compares them against ground truth, and produces a report and a dashboard.

Validation and training are connected by **files**, not by code. This is what allows the two stages to be built in parallel: the model can change backbone, change architecture, change anything — as long as the exported files follow the agreed format, validation runs unchanged.

---

## 1. Handoff Format

This is the most important section of the document. If it isn't nailed down, columns get misaligned, units get mixed up, frame indices get off by one — and a week disappears into debugging.

### 1.1 Official class_id table

| class_id | Class |
|---|---|
| 0 | Person |
| 1 | Forklift |
| 2 | NovaCarter |
| 3 | Transporter |
| 4 | FourierGR1T2 |
| 5 | AgilityDigit |
| **6** | **PalletTruck** |

> **Note:** the table published by the organizers only lists 0–5 and omits PalletTruck. The team assigns PalletTruck = 6. PalletTruck must **never** be dropped from the output — this was previously a silent bug that cost points.
>
> **Use these official class_ids in both files.** Do not use the model's internal IDs. The internal → official remap must be completed before the files are exported. One less place to break.

### 1.2 Findings from the GT scan (Step 1, completed)

All 23 GT files were scanned with `scripts/inspect_gt.py`. Results below are settled facts; they are not assumptions.

**Class table is closed.** All 7 classes exist in the GT, with no unknown classes present. PalletTruck is confirmed (97 unique objects across 13 scenes) — class_id = 6 stands.

**Schema is uniform.** All 23 scenes use the full field names (`object type`, `3d location`, ...). The abbreviated variant does not appear anywhere in train or val. Auto-detection is retained only as insurance for the test set.

**Class coverage per split** — this is the finding that shapes the whole plan:

| Class | In val (W020–022)? | Unique objects in val |
|---|---|---|
| Person | ✅ | 144 |
| Forklift | ✅ | 19 |
| PalletTruck | ✅ | 20 |
| NovaCarter | ✅ (thin) | 2 |
| Transporter | ✅ (thin) | 2 |
| **AgilityDigit** | ❌ **absent** | 0 |
| **FourierGR1T2** | ❌ **absent** | 0 |

The dataset splits into two warehouse families:

| Family | Scenes | Classes present | Extent |
|---|---|---|---|
| A | W000–007, W017–019, W020–022 | Person, Forklift, PalletTruck (+ a few robots in W022) | ~100m, x ≈ −99…0 |
| B | W008–016 | Person, AgilityDigit, FourierGR1T2, NovaCarter, Transporter, (Forklift) | ~16m, x ≈ −10…9 |

Val lies almost entirely in family A. **Two classes have no ground truth in val at all**, and two more have only 2 objects each — statistically meaningless on their own. Validating on val alone leaves the team blind to any error affecting AgilityDigit or FourierGR1T2, including a miswired mapping.

**Scene selection.** W011 covers 6 of 7 classes in a single scene (all but PalletTruck). W011 + W020 together cover all 7. Scenes to use:

| Purpose | Scenes |
|---|---|
| Scoring on data the model has not seen | W020, W021, W022 |
| Coverage of the two classes val lacks | W011 (add W010, W013 … if inference budget allows) |

**Reporting rule.** Scenes W000–W019 were used for training. Metrics from them are still valid, but they mean something different and must be reported in a separate block, never merged into a single figure:

| Block | Scenes | Interpretation |
|---|---|---|
| **CLEAN** | W020–022 | Predictive — the test score should land near this |
| **SEEN** | W011 etc. | Upper bound — reality is no better than this. Failure here is conclusive; success here proves nothing about the test set |

A mapping error is systematic and shows up regardless of whether the model trained on the scene, so the SEEN block is fully valid for the mapping check — its only limitation is as evidence of accuracy.

**Other notes.**
- Scene pairs W000/001, W002/003, W004/005, W006/007, and W020/021 are duplicates (identical detection counts and bounds). They are not independent samples; do not treat them as double the evidence.
- W022 is irregular: 9270 frames instead of 9000, only 4 cameras. Do not hardcode a frame count.
- Severe class imbalance: Person has 5.9M detections vs NovaCarter's 253K. Reports must lead with macro-averaged metrics — a high overall accuracy can mean nothing more than "always guess Person".

### 1.3 File A — `track1.txt`

The official submission format. One line = one object at one frame. Space-separated.

```
<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
```

| Field | Type | Description |
|---|---|---|
| scene_id | int | Unique identifier for each multi-camera sequence |
| class_id | int | Per table 1.1 |
| object_id | int | Positive, unique **per scene and per class**. Remains constant across all cameras within the same scene |
| frame_id | int | Frame index, **zero-based**, within that scene |
| x, y, z | float | 3D coordinates of the bounding-box centroid in the world coordinate system, in **meters** |
| width, length, height | float | Box dimensions along the x (width), y (length), and z (height) axes of the object-centered coordinate system, origin at the centroid. In **meters** |
| yaw | float | Euler angle in **radians** about the y-axis of the object-centered coordinate system, defining the box's heading in the world coordinate system. Pitch and roll are assumed zero |

### 1.4 File B — `detections_2d.txt`

A team-internal format, not required by the organizers. This is the model's **raw** output: one image from one camera in, a list of boxes out — before 3D lifting, before fusion, before tracking.

```
<camera_id> <frame_id> <class_id> <x1> <y1> <x2> <y2> <conf>
```

| Field | Type | Description |
|---|---|---|
| camera_id | int | Camera number, e.g. `Camera_0003` → `3` |
| frame_id | int | Zero-based, consistent with File A |
| class_id | int | Per table 1.1 |
| x1, y1, x2, y2 | float | Top-left and bottom-right corners, in **pixels**, on the original 1920×1080 image, origin at top-left |
| conf | float | Confidence, 0–1 |

**Why File B is needed:** it is the evidence that separates model errors from pipeline errors. If File B says Person and File A says Forklift, the fault lies in the fuse/track/remap stage and the model is clean. Without File B, you only know that something is wrong — not where.

### 1.5 Convention summary

| Item | Agreed value |
|---|---|
| class_id | Official, Person=0 … PalletTruck=6. Both files. |
| frame_id | Zero-based |
| 3D coordinates | Meters |
| 2D boxes | Pixels, 1920×1080 image, origin at top-left |
| yaw | Radians |
| Separator | Space |
| Encoding | UTF-8, `\n` line endings |

---

## 2. Two Validation Layers

The class label is touched at three points in the pipeline:

```
[1] 2D Detector  →  [2] 3D Lift + Fuse + Track  →  [3] Remap + Write file
     (File B)                                           (File A)
```

Measuring only at File A tells you something is wrong but not which stage caused it — and these three stages are fixed in completely different ways. So both layers must be measured:

| Layer | Measured on | Question it answers |
|---|---|---|
| **2D** | File B vs 2D GT | Does the model recognize the object correctly? |
| **3D** | File A vs 3D GT | After fuse/track/remap, is the class still correct? |

### 2.1 Three error types to distinguish

| Type | Signature | Meaning |
|---|---|---|
| Model error | 2D wrong + 3D wrong | The model genuinely fails to recognize the object |
| Pipeline error | 2D correct + 3D wrong, scattered | Fusion or tracking corrupts the class |
| **Mapping error** | 2D correct + 3D wrong **systematically** (every Person → Forklift) | The lookup table is wired incorrectly |

**Mapping errors are the biggest risk** — the model runs fine, the logs look clean, and the only symptom is a submission score that's inexplicably low. Same family as the earlier PalletTruck-dropped bug. With only one layer measured, all three error types look identical.

---

## 3. Execution Roadmap

### Step 1 — Get the data and survey it ✅ done

Download the ground truth for **every** scene that has one: train (W000–W019) and val (W020–W022). Both splits are needed — val alone does not cover all 7 classes (see section 1.2).

Per scene, three small files; the two large folders are skipped:

| Item | Take? | Used for |
|---|---|---|
| `ground_truth.json` | ✅ | The answer key. Without it, nothing can be validated |
| `calibration.json` | ✅ | Cheap; needed for any 2D↔3D projection |
| `map.png` | ✅ | Background for the BEV view in the dashboard |
| `depth_maps/` | ❌ | ~30 GB. Only if validation runs inference itself |
| `videos/` | ❌ | ~12 GB. Same |

Files are Git LFS pointers — a plain clone yields a few-hundred-byte stub instead of real JSON. Download through the web UI, or `git lfs pull --include="*/ground_truth.json"`.

Then run the survey:

```bash
python3 -m venv venv && source venv/bin/activate
pip install ijson
python3 scripts/inspect_gt.py --gt-root data/gt --limit-frames 50   # smoke test
python3 scripts/inspect_gt.py --gt-root data/gt                     # full scan
```

`inspect_gt.py` streams each GT with `ijson` — the files are 200–500 MB each, so nothing is loaded whole. It writes `config/class_registry.json` and prints the schema per scene, the class registry, the class×scene presence matrix, and the coordinate bounds.

Run the full scan, not just a sample. The 50-frame smoke test misses classes that only appear later in a sequence — W022's robots, for instance, are invisible in the first 50 frames.

Results are recorded in section 1.2.

### Step 2 — Build synthetic data

Take the GT of W020 (family A) and W011 (family B) — together they cover all 7 classes — inject **deliberate, known** errors, and export a fake File A and File B: a stand-in for a bad model's output.

| Injected error | Simulates |
|---|---|
| Swap the class on a few objects | Model misclassification |
| Shift positions by a few tens of cm | 3D lift error |
| Delete a few objects | Model misses |
| Add phantom objects | False positives |
| Shift the entire class table | Mapping error |
| Split one track into several IDs | Fragmentation |

This is an exam with a known answer key. Without it, you finish building and have no way to tell whether validation itself is broken.

### Step 3 — Build the 2D layer

For each camera, each frame:
1. Read the 2D GT boxes from `ground_truth.json` — field `2d bounding box visible`, a dict of the form `{Camera_0000: [x1,y1,x2,y2], ...}`. Already there; nothing to generate.
2. Match against the boxes in File B using **IoU**, threshold 0.5
3. Match with **Hungarian** assignment (optimal 1-1), not greedy

> **Critical rule:** matching must **ignore the class**. Match on geometry alone, then read the classes off the matched pairs and compare. If matching is class-aware, a Person mislabeled as Forklift will fail to match the Person GT and turn into a "miss + ghost" instead of showing up in the correct confusion cell. The error gets hidden.

Output:
- Confusion matrix (rows = GT, columns = prediction)
- Extra row/column for **miss** (GT present, model saw nothing) and **ghost** (model saw something, no GT)
- Precision / recall per class
- GT count per class

### Step 4 — Build the 3D layer

Same as Step 3, but in world space:
1. Read `3d location` from the GT
2. Match against x, y, z in File A using **Euclidean distance**, threshold ~1m
3. Hungarian, **class-blind**

Output: the same metrics as Step 3.

**Known complication:** in the 3D layer an object carries **one class for the entire track**, but the detector predicts a class per frame per camera, and those can disagree. This needs to be settled with the training stage: how is a track's final class decided — majority vote, or highest-confidence frame? Validation has to know in order to measure correctly.

### Step 5 — Verify the validator

Run validation (Steps 3 + 4) against the synthetic data from Step 2. It must surface **exactly** the injected errors — no more, no less.

If it doesn't → **the validator is broken, stop and fix it**. A validator that has never been verified produces numbers that are just made up.

### Step 6 — Cross-reference the two layers

A script reads both layers' results and prints the verdict table from section 2.1: model error / pipeline error / mapping error. This is the real value of the whole system — not just reporting *that* something is wrong, but *where*.

Results are split into the CLEAN and SEEN blocks defined in section 1.2 and reported separately. Metrics from the two blocks are never averaged into a single number: an overall figure carried by scenes the model trained on would mislead everyone reading it. Every table leads with macro-averaged metrics, given the 20:1 imbalance between Person and the rarest classes.

### Step 7 — Dashboard

Streamlit, showing:
- Both confusion matrices (2D and 3D) side by side
- Per-class table: precision, recall, GT count
- The verdict table from Step 6
- BEV map: plot the positions of misclassified objects, to see whether errors cluster in one corner of the warehouse

Runs locally; no deployment needed.

### Step 8 — Run for real

Receive File A + File B from the training stage → run → deliver the report. Repeat for every new model version.

Inference is the expensive part — 9000 frames × 10–20 cameras per scene — so the request to the training stage is tiered:

| Tier | Scenes | Frames | When |
|---|---|---|---|
| Mapping check | W011 + W020 (covers all 7 classes) | every 30th frame | Every new model version. A miswiring is systematic, so a sample catches it |
| Scoring | W020, W021, W022 | full | Before submission |

Additional family-B scenes (W010, W013, …) can be added to the mapping tier at no cost to validation if inference budget allows. They must not enter the scoring tier.

---

## 4. Order and dependencies

```
Step 1 (download + survey GT)  ✅ done
   └─→ Step 2 (synthetic data)
          └─→ Steps 3, 4 (build both layers)   ← build and test interleaved
                 └─→ Step 5 (verify the validator)
                        └─→ Step 6 (cross-reference)
                               └─→ Step 7 (dashboard)
                                      └─→ Step 8 (run for real)  ← the only step that needs files from training
```

**Steps 1–7 can be built immediately and depend on nothing from the training stage.** Only Step 8 needs the handoff files. The format in section 1.5 therefore needs to be agreed **up front**, in parallel with the build.

**Step 1 is complete** — the GT is downloaded, the class table is closed, and the scene selection is settled (section 1.2). Next is Step 2.

**Open question for the organizers:** does the test set contain AgilityDigit and FourierGR1T2? If it does, the team is validating two classes with no clean ground truth behind them, and everyone should know that.