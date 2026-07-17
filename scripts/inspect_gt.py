#!/usr/bin/env python3
"""
inspect_gt.py — Scan every ground_truth.json and build the class registry.

Step 1 of the validation roadmap. Answers:
  - Which schema does each scene use (full vs abbreviated)?
  - What is the complete class list across all scenes?
  - Which scenes contain which classes, and how many objects of each?
  - What are the coordinate ranges, frame counts, camera counts?

GT files are 200-500 MB each, so parsing is streamed with ijson.
Nothing is loaded whole into memory.

Usage:
    python3 scripts/inspect_gt.py --gt-root data/gt --out config/class_registry.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import ijson
except ImportError:
    sys.exit("Missing dependency. Install with: pip install ijson")


# --- Schema handling -------------------------------------------------------
# Two known variants. Values are candidate key names, tried in order.
FIELD_ALIASES = {
    "object_id":   ["object id", "id", "object_id"],
    "class":       ["object type", "type", "object_type", "class"],
    "loc3d":       ["3d location", "loc3d", "location", "3d_location"],
    "bbox3d":      ["3d bounding box scale", "bbox3d", "scale", "3d_bounding_box_scale"],
    "rot3d":       ["3d bounding box rotation", "rot3d", "rotation"],
    "bbox2d":      ["2d bounding box visible", "bbox2d", "2d_bounding_box_visible"],
}


def resolve_schema(sample_obj):
    """Map canonical field names -> actual key names present in this file."""
    resolved = {}
    for canonical, candidates in FIELD_ALIASES.items():
        for key in candidates:
            if key in sample_obj:
                resolved[canonical] = key
                break
    return resolved


def schema_label(resolved):
    """Human-readable name for the detected schema variant."""
    cls_key = resolved.get("class")
    if cls_key == "object type":
        return "full"
    if cls_key == "type":
        return "abbreviated"
    return f"unknown({cls_key})"


# --- Scanning --------------------------------------------------------------

class SceneStats:
    def __init__(self, scene, split):
        self.scene = scene
        self.split = split
        self.split_name = split
        self.schema = None
        self.resolved = {}
        self.class_counts = Counter()        # detections per class
        self.class_track_ids = defaultdict(set)  # unique object ids per class
        self.frames = 0
        self.cameras = set()
        self.bounds = {"x": [None, None], "y": [None, None], "z": [None, None]}
        self.missing_fields = set()
        self.detections = 0

    def note_bounds(self, loc):
        if not loc or len(loc) < 3:
            return
        for axis, v in zip("xyz", loc[:3]):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return
            lo, hi = self.bounds[axis]
            self.bounds[axis] = [v if lo is None else min(lo, v),
                                 v if hi is None else max(hi, v)]

    def to_dict(self):
        return {
            "scene": self.scene,
            "split": self.split_name,
            "schema": self.schema,
            "field_map": self.resolved,
            "frames": self.frames,
            "detections": self.detections,
            "cameras": sorted(self.cameras),
            "camera_count": len(self.cameras),
            "classes": {
                c: {"detections": n, "unique_objects": len(self.class_track_ids[c])}
                for c, n in sorted(self.class_counts.items())
            },
            "bounds": {k: v for k, v in self.bounds.items()},
            "missing_fields": sorted(self.missing_fields),
        }


def scan_scene(path, scene, split, limit_frames=None):
    """Stream one ground_truth.json. Returns SceneStats."""
    st = SceneStats(scene, split)

    with open(path, "rb") as f:
        # GT is a top-level object: {frame_id: [obj, obj, ...], ...}
        for frame_id, objects in ijson.kvitems(f, "", use_float=True):
            st.frames += 1
            if not isinstance(objects, list):
                continue

            for obj in objects:
                if st.schema is None:
                    st.resolved = resolve_schema(obj)
                    st.schema = schema_label(st.resolved)
                    for canonical in FIELD_ALIASES:
                        if canonical not in st.resolved:
                            st.missing_fields.add(canonical)

                st.detections += 1

                cls_key = st.resolved.get("class")
                cls = obj.get(cls_key) if cls_key else None
                if cls is not None:
                    st.class_counts[cls] += 1
                    oid_key = st.resolved.get("object_id")
                    if oid_key and oid_key in obj:
                        st.class_track_ids[cls].add(obj[oid_key])

                loc_key = st.resolved.get("loc3d")
                if loc_key:
                    st.note_bounds(obj.get(loc_key))

                bb_key = st.resolved.get("bbox2d")
                if bb_key and isinstance(obj.get(bb_key), dict):
                    st.cameras.update(obj[bb_key].keys())

            if limit_frames and st.frames >= limit_frames:
                break

    return st


# --- Reporting -------------------------------------------------------------

def print_report(scenes, registry):
    W = 78
    print("=" * W)
    print("GROUND TRUTH INSPECTION")
    print("=" * W)

    # Per-scene summary
    print(f"\n{'Scene':<18}{'Split':<7}{'Schema':<14}{'Frames':>8}{'Cams':>6}{'Objects':>10}")
    print("-" * W)
    for s in scenes:
        total_obj = sum(len(v) for v in s.class_track_ids.values())
        print(f"{s.scene:<18}{s.split_name:<7}{s.schema or '-':<14}"
              f"{s.frames:>8}{len(s.cameras):>6}{total_obj:>10}")

    # Class registry
    print("\n" + "=" * W)
    print("CLASS REGISTRY — complete list across all scenes")
    print("=" * W)
    print(f"\n{'Class':<20}{'Scenes':>8}{'Objects':>10}{'Detections':>14}")
    print("-" * W)
    for cls, info in registry["classes"].items():
        print(f"{cls:<20}{len(info['scenes']):>8}{info['unique_objects']:>10}"
              f"{info['detections']:>14}")

    # Presence matrix
    print("\n" + "=" * W)
    print("CLASS x SCENE PRESENCE  (number of unique objects, '.' = absent)")
    print("=" * W)
    class_names = list(registry["classes"].keys())
    header = f"\n{'Scene':<18}" + "".join(f"{c[:9]:>10}" for c in class_names)
    print(header)
    print("-" * len(header))
    for s in scenes:
        row = f"{s.scene:<18}"
        for c in class_names:
            n = len(s.class_track_ids.get(c, ()))
            row += f"{n if n else '.':>10}"
        print(row)

    # Coordinate bounds
    print("\n" + "=" * W)
    print("COORDINATE BOUNDS (meters)")
    print("=" * W)
    print(f"\n{'Scene':<18}{'x min':>10}{'x max':>10}{'y min':>10}{'y max':>10}"
          f"{'z min':>9}{'z max':>9}")
    print("-" * W)
    for s in scenes:
        b = s.bounds
        def fmt(v):
            return f"{v:.1f}" if isinstance(v, float) else "-"
        print(f"{s.scene:<18}{fmt(b['x'][0]):>10}{fmt(b['x'][1]):>10}"
              f"{fmt(b['y'][0]):>10}{fmt(b['y'][1]):>10}"
              f"{fmt(b['z'][0]):>9}{fmt(b['z'][1]):>9}")


def check_against_official(registry, official):
    W = 78
    print("\n" + "=" * W)
    print("CHECK AGAINST OFFICIAL CLASS TABLE")
    print("=" * W)
    found = set(registry["classes"].keys())
    expected = set(official.keys())

    print()
    for cls, cid in sorted(official.items(), key=lambda kv: kv[1]):
        if cls in found:
            n = registry["classes"][cls]["unique_objects"]
            print(f"  [OK]      id={cid:<3} {cls:<18} {n} objects")
        else:
            print(f"  [MISSING] id={cid:<3} {cls:<18} not present in any GT scene")

    unknown = found - expected
    if unknown:
        print()
        for cls in sorted(unknown):
            print(f"  [UNKNOWN] {cls:<18} in GT but not in the official table")

    print()
    if "PalletTruck" in found:
        print("  => PalletTruck exists in GT. Keeping class_id = 6 is correct.")
    else:
        print("  => PalletTruck NOT found in any GT scene. Confirm with the team")
        print("     before relying on class_id = 6.")
    if unknown:
        print("  => Unknown classes present. The class table must be updated")
        print("     before building either validation layer.")


# --- Main ------------------------------------------------------------------

OFFICIAL = {
    "Person": 0, "Forklift": 1, "NovaCarter": 2, "Transporter": 3,
    "FourierGR1T2": 4, "AgilityDigit": 5, "PalletTruck": 6,
}


def main():
    ap = argparse.ArgumentParser(description="Scan GT files and build the class registry.")
    ap.add_argument("--gt-root", default="data/gt",
                    help="Root containing train/ and val/ subfolders")
    ap.add_argument("--out", default="config/class_registry.json",
                    help="Where to write the registry")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Only read the first N frames per scene (quick smoke test)")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    paths = sorted(gt_root.glob("*/*/ground_truth.json"))
    if not paths:
        sys.exit(f"No ground_truth.json found under {gt_root}/<split>/<scene>/")

    scenes = []
    for p in paths:
        scene = p.parent.name
        split = p.parent.parent.name
        print(f"scanning {split}/{scene} ...", end=" ", flush=True)
        try:
            st = scan_scene(p, scene, split, args.limit_frames)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(f"{st.frames} frames, {st.detections} detections, schema={st.schema}")
        scenes.append(st)

    if not scenes:
        sys.exit("Nothing scanned.")

    # Aggregate registry
    classes = defaultdict(lambda: {"scenes": [], "unique_objects": 0, "detections": 0})
    for s in scenes:
        for cls, n in s.class_counts.items():
            classes[cls]["scenes"].append(s.scene)
            classes[cls]["detections"] += n
            classes[cls]["unique_objects"] += len(s.class_track_ids[cls])

    registry = {
        "classes": {k: classes[k] for k in sorted(classes)},
        "official_class_ids": OFFICIAL,
        "scenes": [s.to_dict() for s in scenes],
    }

    print()
    print_report(scenes, registry)
    check_against_official(registry, OFFICIAL)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2, default=str))
    print(f"\nRegistry written to: {out}")


if __name__ == "__main__":
    main()