"""
loaders.py — reading everything the validator consumes.

All format assumptions live here. If a file is malformed, it fails at load
time with a clear message rather than three stages later as a mysterious
metric.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    import ijson
except ImportError:
    sys.exit("Missing dependency. Install with: pip install ijson")


# --- Config -----------------------------------------------------------------

class Config:
    """Thin wrapper over config.yaml plus the registry written by inspect_gt."""

    def __init__(self, path="config/config.yaml"):
        self.path = Path(path)
        if not self.path.exists():
            sys.exit(f"Config not found: {path}")
        self.raw = yaml.safe_load(self.path.read_text())

        self.classes = self.raw["classes"]                  # name -> official id
        self.class_by_id = {v: k for k, v in self.classes.items()}
        self.class_names = list(self.classes.keys())

        self.iou_threshold = self.raw["matching"]["layer_2d"]["iou_threshold"]
        self.min_box_area = self.raw["matching"]["layer_2d"].get("min_box_area_px", 1.0)
        self.dist_threshold = self.raw["matching"]["layer_3d"]["distance_threshold_m"]
        self.thin_threshold = self.raw["reporting"]["thin_class_threshold"]

        self.gt_root = Path(self.raw["paths"]["gt_root"])
        self.synthetic_root = Path(self.raw["paths"]["synthetic_root"])
        self.input_root = Path(self.raw["paths"]["input_root"])
        self.reports_root = Path(self.raw["paths"]["reports_root"])

        reg_path = Path(self.raw["paths"]["class_registry"])
        self.registry = None
        if reg_path.exists():
            self.registry = json.loads(reg_path.read_text())

        self.clean_scenes = [s["name"] for s in self.raw["scenes"]["clean"]]
        self.seen_scenes = [s["name"] for s in self.raw["scenes"]["seen"]]

    def block_of(self, scene):
        """CLEAN = model never trained on it, so the number predicts the test
        score. SEEN = model trained on it, so the number is an upper bound.
        The two are never averaged together."""
        if scene in self.clean_scenes:
            return "CLEAN"
        if scene in self.seen_scenes:
            return "SEEN"
        return "SEEN"

    def find_gt(self, scene):
        hits = list(self.gt_root.glob(f"*/{scene}/ground_truth.json"))
        if not hits:
            sys.exit(f"ground_truth.json for {scene} not found under {self.gt_root}")
        return hits[0]


# --- Ground truth -----------------------------------------------------------

FIELD_ALIASES = {
    "object_id": ["object id", "id", "object_id"],
    "class":     ["object type", "type", "object_type", "class"],
    "loc3d":     ["3d location", "loc3d", "location", "3d_location"],
    "bbox3d":    ["3d bounding box scale", "bbox3d", "scale"],
    "rot3d":     ["3d bounding box rotation", "rot3d", "rotation"],
    "bbox2d":    ["2d bounding box visible", "bbox2d", "2d_bounding_box_visible"],
}


def resolve_schema(obj):
    out = {}
    for canonical, candidates in FIELD_ALIASES.items():
        for key in candidates:
            if key in obj:
                out[canonical] = key
                break
    return out


def camera_id_of(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def stream_gt_2d(gt_path, class_table, max_frames=None, min_area=1.0, stats=None):
    """
    Yield (frame_id, {camera_id: [(object_id, class_id, [x1,y1,x2,y2]), ...]}).

    Streamed with ijson — GT files run 200-500 MB and must never be loaded
    whole. The 2D boxes are already in the GT under '2d bounding box visible';
    nothing is projected or computed here.

    Boxes smaller than `min_area` are dropped and counted in `stats`. The GT
    holds a real population of zero-area boxes: objects occluded to nothing or
    clipped at the frame border. Keeping them would charge the model with
    misses for objects that occupy no pixels. The count is surfaced in the
    report so the exclusion stays visible rather than becoming a silent
    convenience.
    """
    schema = None
    with open(gt_path, "rb") as f:
        for frame_key, objects in ijson.kvitems(f, "", use_float=True):
            try:
                fid = int(frame_key)
            except ValueError:
                continue
            if max_frames is not None and fid >= max_frames:
                continue
            if not isinstance(objects, list):
                continue

            per_cam = {}
            for obj in objects:
                if schema is None:
                    schema = resolve_schema(obj)
                cls = obj.get(schema["class"])
                if cls not in class_table:
                    continue
                oid = obj.get(schema["object_id"])
                cid = class_table[cls]
                boxes = obj.get(schema.get("bbox2d")) or {}
                for cam, box in boxes.items():
                    if not box or len(box) < 4:
                        if stats is not None:
                            stats["gt_malformed"] = stats.get("gt_malformed", 0) + 1
                        continue
                    b = [float(v) for v in box[:4]]
                    if box_area(b) < min_area:
                        if stats is not None:
                            stats["gt_degenerate"] = stats.get("gt_degenerate", 0) + 1
                        continue
                    per_cam.setdefault(camera_id_of(cam), []).append((oid, cid, b))
            yield fid, per_cam


def stream_gt_3d(gt_path, class_table, max_frames=None):
    """Yield (frame_id, [(object_id, class_id, [x,y,z]), ...]) for layer 3D."""
    schema = None
    with open(gt_path, "rb") as f:
        for frame_key, objects in ijson.kvitems(f, "", use_float=True):
            try:
                fid = int(frame_key)
            except ValueError:
                continue
            if max_frames is not None and fid >= max_frames:
                continue
            if not isinstance(objects, list):
                continue
            rows = []
            for obj in objects:
                if schema is None:
                    schema = resolve_schema(obj)
                cls = obj.get(schema["class"])
                if cls not in class_table:
                    continue
                loc = obj.get(schema["loc3d"]) or [0, 0, 0]
                rows.append((obj.get(schema["object_id"]), class_table[cls],
                             [float(v) for v in loc[:3]]))
            yield fid, rows


# --- Prediction files -------------------------------------------------------

FILE_A_COLS = ["scene_id", "class_id", "object_id", "frame_id",
               "x", "y", "z", "width", "length", "height", "yaw"]
FILE_B_COLS = ["camera_id", "frame_id", "class_id", "x1", "y1", "x2", "y2", "conf"]


def _read_table(path, cols, name):
    path = Path(path)
    if not path.exists():
        sys.exit(f"{name} not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols,
                     engine="c", dtype=None)
    if df.shape[1] != len(cols):
        sys.exit(f"{name} has {df.shape[1]} columns, expected {len(cols)}: {path}")
    return df


def read_file_a(path, class_ids=None, max_frames=None):
    """File A — track1.txt. Validated on load: column count and class range."""
    df = _read_table(path, FILE_A_COLS, "File A")
    if max_frames is not None:
        df = df[df.frame_id < max_frames]
    if class_ids is not None:
        bad = set(df.class_id.unique()) - set(class_ids)
        if bad:
            sys.exit(f"File A contains class ids outside the official table: "
                     f"{sorted(bad)}\n  {path}")
    return df


def read_file_b(path, class_ids=None, max_frames=None):
    """File B — detections_2d.txt."""
    df = _read_table(path, FILE_B_COLS, "File B")
    if max_frames is not None:
        df = df[df.frame_id < max_frames]
    if class_ids is not None:
        bad = set(df.class_id.unique()) - set(class_ids)
        if bad:
            sys.exit(f"File B contains class ids outside the official table: "
                     f"{sorted(bad)}\n  {path}")
    return df