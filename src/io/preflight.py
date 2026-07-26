"""
preflight.py — submission checks that need no ground truth.

Everything here answers one question: will the official scorer accept this
file at all? Not "is the model good" — that needs GT and lives in the two
layers. These are the failures that lose the whole submission rather than
some points, which is why they run first and run everywhere, including on
the test set where there is no ground truth to check against.

Every FATAL rule below is traced to the line of official code that enforces
it. The references are to two repositories:

  [EVAL] NVIDIA-AI-Blueprints/video-search-and-summarization @ develop
         libs/analytics/spatialai-data-utils/
           spatialai_data_utils/eval/tracking/aicity_mtmc_eval.py
           spatialai_data_utils/eval/tracking/hota/datasets/_base_dataset.py
  [P3D]  facebookresearch/pytorch3d @ main
           pytorch3d/ops/iou_box3d.py

Anyone who doubts a threshold can open those files and read the line. That
is the point of citing them: a number nobody can trace becomes folklore.
"""

import json
import math
from array import array
from collections import Counter
from pathlib import Path

# --- Official constants ------------------------------------------------------

# [EVAL] spatialai_data_utils/datasets/aicity26/spec.py — CLASS_ID_TO_NAME.
# The 2025 table (0-5) plus PalletTruck at 6. A prediction row carrying any
# other class id is rejected outright:
#   aicity_mtmc_eval.py:346-352  ->  raise ValueError(...)
CLASS_ID_TO_NAME = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}

# [EVAL] aicity_mtmc_eval.py:304-313 — the row is split on a single space and
# must yield exactly this many fields, or the whole run raises.
NUM_FIELDS = 11

# [P3D] pytorch3d/ops/iou_box3d.py:73-89, function _check_nonzero, called from
# box3d_overlap at line 163-164 with eps=1e-4. It computes the area of every
# face of the box and raises ValueError("Planes have zero areas") if any face
# is below eps. A box face is one of w*l, w*h, l*h.
#
# Note this is NOT "a dimension is zero". A box of 0.01 x 0.005 x 2.0 has all
# three dimensions positive and still crashes, because 0.01*0.005 = 5e-5.
MIN_FACE_AREA_M2 = 1e-4

# Bit budget for the duplicate key, see _pack(). Rows outside these ranges
# fall back to a plain set, which is slower and heavier but always correct.
_BITS = {"scene": 8, "cls": 3, "frame": 21, "obj": 31}

FATAL = "FATAL"
WARN = "WARN"

# Rule codes, kept short so they can be grepped out of CI logs.
CODES = {
    "E_FIELDS": "row does not split into exactly 11 space-separated fields",
    "E_CR": "carriage return in line (file is CRLF, must be LF)",
    "E_INT": "scene_id / class_id / object_id / frame_id is not an integer",
    "E_FLOAT": "a numeric field is not a finite float",
    "E_CLASS": "class_id outside the official 0-6 table",
    "E_FRAME": "frame_id is negative",
    "E_OBJ": "object_id is not positive",
    "E_DEGEN": "degenerate box: a face area is below 1e-4 m2",
    "E_DUP": "same object_id appears twice in one (scene, class, frame)",
    "E_SCENE_MISSING": "an expected scene has no rows at all",
    "W_XCLASS": "object_id reused across two classes in the same scene",
    "W_FRAME_HI": "frame_id at or beyond the evaluation window",
    "W_YAW": "yaw outside [-pi, pi]",
    "W_DIM_SMALL": "a box dimension below 0.05 m",
    "W_BLANK": "blank line",
    "W_NO_EOL": "file does not end with a newline",
    "W_Z_VS_H": "centroid z far from height/2 on many rows",
    "W_FRAME_GAPS": "frames in the scene are not contiguous",
}


def _pack(scene, cls, obj, frame):
    """Fold one row identity into a single int64, or return None if it will
    not fit. Packing keeps the duplicate check at 8 bytes per row instead of
    a ~70 byte Python tuple; on a 1.5M row submission that is 12 MB rather
    than a few hundred."""
    if not (0 <= scene < (1 << _BITS["scene"])):
        return None
    if not (0 <= cls < (1 << _BITS["cls"])):
        return None
    if not (0 <= frame < (1 << _BITS["frame"])):
        return None
    if not (0 <= obj < (1 << _BITS["obj"])):
        return None
    return (((scene << _BITS["cls"] | cls) << _BITS["frame"] | frame)
            << _BITS["obj"]) | obj


def _unpack(key):
    obj = key & ((1 << _BITS["obj"]) - 1)
    key >>= _BITS["obj"]
    frame = key & ((1 << _BITS["frame"]) - 1)
    key >>= _BITS["frame"]
    cls = key & ((1 << _BITS["cls"]) - 1)
    scene = key >> _BITS["cls"]
    return scene, cls, obj, frame


class _SceneStats:
    def __init__(self):
        self.rows = 0
        self.frames = set()
        self.tracks = set()          # (class_id, object_id)
        self.class_rows = Counter()
        self.obj_to_classes = {}     # object_id -> set of class_id
        self.lo = [math.inf] * 3
        self.hi = [-math.inf] * 3
        self.dim_lo = [math.inf] * 3
        self.dim_hi = [-math.inf] * 3
        self.zh_abs_sum = 0.0
        self.zh_abs_max = 0.0
        self.zh_over = 0             # rows with |z - h/2| > 0.25 m
        self.yaw_bad = 0

    def to_dict(self, num_frames):
        n_frames = len(self.frames)
        fmin = min(self.frames) if self.frames else None
        fmax = max(self.frames) if self.frames else None
        span = (fmax - fmin + 1) if self.frames else 0
        return {
            "rows": self.rows,
            "frames_covered": n_frames,
            "frame_min": fmin,
            "frame_max": fmax,
            "frame_gaps": span - n_frames if span else 0,
            "tracks": len(self.tracks),
            "rows_per_frame": round(self.rows / n_frames, 4) if n_frames else None,
            "rows_per_track": round(self.rows / len(self.tracks), 2) if self.tracks else None,
            "class_rows": {CLASS_ID_TO_NAME.get(c, str(c)): n
                           for c, n in sorted(self.class_rows.items())},
            "class_tracks": {CLASS_ID_TO_NAME.get(c, str(c)): n for c, n in
                             sorted(Counter(c for c, _ in self.tracks).items())},
            "bounds_xyz": [[round(self.lo[i], 3), round(self.hi[i], 3)]
                           for i in range(3)] if self.rows else None,
            "bounds_wlh": [[round(self.dim_lo[i], 3), round(self.dim_hi[i], 3)]
                           for i in range(3)] if self.rows else None,
            "z_vs_half_h_mean_abs": round(self.zh_abs_sum / self.rows, 4) if self.rows else None,
            "z_vs_half_h_max_abs": round(self.zh_abs_max, 4) if self.rows else None,
            "z_vs_half_h_over_25cm": self.zh_over,
            "yaw_out_of_range": self.yaw_bad,
        }


def check_file(path, num_frames=9000, expect_scenes=None, max_examples=5):
    """
    Read one track1.txt and report every rule it breaks.

    :param path: the submission file.
    :param num_frames: the evaluation window, 0-indexed exclusive upper bound.
        The official default is 9000; use 1800 for Warehouse_026 / _027.
    :param expect_scenes: optional list of scene_ids that must be present.
        A scene with ground truth but no prediction rows scores a hard zero
        weighted by its full GT row count -- aicity_mtmc_eval.py:514-530 --
        so an omitted scene is a fatal, unlike an omitted class.
    :param max_examples: how many offending line numbers to keep per rule.
    :return: a report dict; report["fatal_count"] == 0 means the file will at
        least be accepted. It says nothing about whether it scores well.
    """
    path = Path(path)
    issues = {}
    counts = Counter()

    def record(code, severity, line_no, detail=""):
        counts[code] += 1
        bucket = issues.setdefault(code, {"severity": severity, "count": 0,
                                          "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < max_examples:
            bucket["examples"].append({"line": line_no, "detail": detail})

    keys = array("q")
    overflow_seen = set()
    overflow_dups = set()
    scenes = {}
    st_yaw_bad = {}
    total_lines = 0
    ok_rows = 0
    last_char = "\n"

    with open(path, "r", encoding="utf-8", newline="") as fp:
        for line_no, raw in enumerate(fp, start=1):
            total_lines += 1
            if raw:
                last_char = raw[-1]
            if "\r" in raw:
                record("E_CR", FATAL, line_no, repr(raw[-12:]))
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                record("W_BLANK", WARN, line_no)
                continue

            # Mirror the official splitter exactly: a plain split on one space.
            # This is what turns a double space or a trailing space into a
            # field-count error -- aicity_mtmc_eval.py:302-313.
            parts = line.split(" ")
            if len(parts) != NUM_FIELDS:
                record("E_FIELDS", FATAL, line_no,
                       f"{len(parts)} fields: {line[:70]!r}")
                continue

            try:
                scene_id = int(parts[0])
                class_id = int(parts[1])
                object_id = int(parts[2])
                frame_id = int(parts[3])
            except ValueError:
                record("E_INT", FATAL, line_no, line[:70])
                continue

            try:
                nums = [float(v) for v in parts[4:11]]
            except ValueError:
                record("E_FLOAT", FATAL, line_no, line[:70])
                continue
            if not all(math.isfinite(v) for v in nums):
                record("E_FLOAT", FATAL, line_no, line[:70])
                continue
            x, y, z, w, l, h, yaw = nums

            bad_row = False
            if class_id not in CLASS_ID_TO_NAME:
                record("E_CLASS", FATAL, line_no, f"class_id={class_id}")
                bad_row = True
            if frame_id < 0:
                record("E_FRAME", FATAL, line_no, f"frame_id={frame_id}")
                bad_row = True
            if object_id <= 0:
                record("E_OBJ", FATAL, line_no, f"object_id={object_id}")
                bad_row = True

            min_face = min(abs(w * l), abs(w * h), abs(l * h))
            if min_face < MIN_FACE_AREA_M2 or w <= 0 or l <= 0 or h <= 0:
                record("E_DEGEN", FATAL, line_no,
                       f"w={w:g} l={l:g} h={h:g} min_face={min_face:.3g}")
                bad_row = True
            elif min(w, l, h) < 0.05:
                record("W_DIM_SMALL", WARN, line_no, f"w={w:g} l={l:g} h={h:g}")

            if bad_row:
                continue

            if frame_id >= num_frames:
                record("W_FRAME_HI", WARN, line_no, f"frame_id={frame_id}")
            # Tolerance is 1e-3, not float epsilon: a file rounded to four
            # decimals prints pi as 3.1416, which is genuinely above pi. The
            # rule is meant to catch a systematic convention error -- a whole
            # file written in [0, 2pi) -- not rounding noise.
            if not (-math.pi - 1e-3 <= yaw <= math.pi + 1e-3):
                record("W_YAW", WARN, line_no, f"yaw={yaw:g}")
                st_yaw_bad[scene_id] = st_yaw_bad.get(scene_id, 0) + 1

            key = _pack(scene_id, class_id, object_id, frame_id)
            if key is None:
                # Ids too large to fold into 63 bits. Rare, but they must not
                # skip the duplicate check just because the fast path cannot
                # hold them.
                tup = (scene_id, class_id, object_id, frame_id)
                if tup in overflow_seen:
                    overflow_dups.add(tup)
                else:
                    overflow_seen.add(tup)
            else:
                keys.append(key)

            st = scenes.setdefault(scene_id, _SceneStats())
            st.rows += 1
            st.frames.add(frame_id)
            st.tracks.add((class_id, object_id))
            st.class_rows[class_id] += 1
            st.obj_to_classes.setdefault(object_id, set()).add(class_id)
            for i, v in enumerate((x, y, z)):
                st.lo[i] = min(st.lo[i], v)
                st.hi[i] = max(st.hi[i], v)
            for i, v in enumerate((w, l, h)):
                st.dim_lo[i] = min(st.dim_lo[i], v)
                st.dim_hi[i] = max(st.dim_hi[i], v)
            dev = abs(z - h / 2.0)
            st.zh_abs_sum += dev
            st.zh_abs_max = max(st.zh_abs_max, dev)
            if dev > 0.25:
                st.zh_over += 1
            ok_rows += 1

    if last_char != "\n" and total_lines:
        record("W_NO_EOL", WARN, total_lines)

    # --- Duplicate identities -----------------------------------------------
    # TrackEval raises as soon as one tracker id appears twice in a single
    # timestep -- _base_dataset.py, _check_unique_ids. Files are split per
    # (scene, class) first, so the collision scope is (scene, class, frame).
    dups = [_unpack(k) for k in _find_duplicates(keys)]
    dups += sorted(overflow_dups)
    shown = 0
    for scn, c, o, f in dups:
        if shown < max_examples * 4:
            record("E_DUP", FATAL, None,
                   f"scene={scn} class={CLASS_ID_TO_NAME.get(c, c)} "
                   f"object_id={o} frame={f}")
            shown += 1
        else:
            counts["E_DUP"] += 1
            issues["E_DUP"]["count"] += 1

    # --- object_id shared across classes (warning, not fatal) ---------------
    # The spec says object_id is unique per scene AND per class, so this is a
    # spec violation. It is not fatal to the scorer: the splitter writes one
    # file per (scene, class), so a reused id never collides there.
    for scene_id, st in scenes.items():
        shared = [oid for oid, cs in st.obj_to_classes.items() if len(cs) > 1]
        for oid in shared[:max_examples]:
            record("W_XCLASS", WARN, None,
                   f"scene={scene_id} object_id={oid} classes="
                   f"{sorted(CLASS_ID_TO_NAME.get(c, c) for c in st.obj_to_classes[oid])}")
        if len(shared) > max_examples:
            counts["W_XCLASS"] += len(shared) - max_examples
            issues["W_XCLASS"]["count"] += len(shared) - max_examples

    # --- Per-scene shape warnings -------------------------------------------
    for scene_id, st in scenes.items():
        st.yaw_bad = st_yaw_bad.get(scene_id, 0)
        if st.rows and st.zh_over > 0.10 * st.rows:
            record("W_Z_VS_H", WARN, None,
                   f"scene={scene_id}: {st.zh_over}/{st.rows} rows have "
                   f"|z - h/2| > 0.25 m")
        if st.frames:
            span = max(st.frames) - min(st.frames) + 1
            if span - len(st.frames) > 0:
                record("W_FRAME_GAPS", WARN, None,
                       f"scene={scene_id}: {span - len(st.frames)} frames "
                       f"in [{min(st.frames)}, {max(st.frames)}] have no rows")

    # --- Missing scenes ------------------------------------------------------
    if expect_scenes:
        for scene_id in expect_scenes:
            if scenes.get(int(scene_id)) is None:
                record("E_SCENE_MISSING", FATAL, None,
                       f"scene_id={scene_id} has no rows; a scene absent from "
                       f"the submission scores 0 weighted by its full GT size")

    fatal_count = sum(v["count"] for v in issues.values()
                      if v["severity"] == FATAL)
    warn_count = sum(v["count"] for v in issues.values()
                     if v["severity"] == WARN)

    return {
        "file": str(path),
        "num_frames_to_eval": num_frames,
        "total_lines": total_lines,
        "valid_rows": ok_rows,
        "fatal_count": fatal_count,
        "warn_count": warn_count,
        "issues": {k: {"severity": v["severity"], "count": v["count"],
                       "message": CODES.get(k, k), "examples": v["examples"]}
                   for k, v in sorted(issues.items())},
        "scenes": {str(sid): st.to_dict(num_frames)
                   for sid, st in sorted(scenes.items())},
    }


def _find_duplicates(keys):
    """Return the set of packed keys that occur more than once."""
    if len(keys) < 2:
        return set()
    try:
        import numpy as np
        arr = np.frombuffer(memoryview(keys), dtype=np.int64)
        uniq, cnt = np.unique(arr, return_counts=True)
        return set(int(k) for k in uniq[cnt > 1])
    except ImportError:
        seen, dup = set(), set()
        for k in keys:
            if k in seen:
                dup.add(k)
            else:
                seen.add(k)
        return dup


# --- Presentation ------------------------------------------------------------

def format_report(report):
    out = []
    out.append(f"file          : {report['file']}")
    out.append(f"lines / rows  : {report['total_lines']} / {report['valid_rows']}")
    out.append(f"eval window   : [0, {report['num_frames_to_eval']})")
    out.append("")

    if not report["issues"]:
        out.append("No issues.")
    for code, info in report["issues"].items():
        out.append(f"[{info['severity']:5}] {code:16} x{info['count']:<9} "
                   f"{info['message']}")
        for ex in info["examples"]:
            where = f"line {ex['line']}" if ex["line"] else "     "
            out.append(f"          {where:>12}  {ex['detail']}")
    out.append("")

    for sid, s in report["scenes"].items():
        out.append(f"--- scene {sid} " + "-" * 52)
        out.append(f"  rows {s['rows']}   frames {s['frames_covered']} "
                   f"[{s['frame_min']}..{s['frame_max']}] gaps {s['frame_gaps']}"
                   f"   tracks {s['tracks']}")
        out.append(f"  rows/frame {s['rows_per_frame']}   "
                   f"rows/track {s['rows_per_track']}")
        out.append(f"  class rows   {s['class_rows']}")
        out.append(f"  class tracks {s['class_tracks']}")
        out.append(f"  xyz bounds   {s['bounds_xyz']}")
        out.append(f"  wlh bounds   {s['bounds_wlh']}")
        out.append(f"  |z - h/2|    mean {s['z_vs_half_h_mean_abs']}  "
                   f"max {s['z_vs_half_h_max_abs']}  "
                   f"over 25cm {s['z_vs_half_h_over_25cm']}")
        out.append(f"  yaw outside [-pi, pi]  {s['yaw_out_of_range']}")
    out.append("")
    verdict = "REJECT" if report["fatal_count"] else "ACCEPT"
    out.append(f"{verdict}  —  {report['fatal_count']} fatal, "
               f"{report['warn_count']} warnings")
    if not report["fatal_count"]:
        out.append("The scorer will accept this file. It says nothing about "
                   "the score.")
    return "\n".join(out)


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path