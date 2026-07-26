#!/usr/bin/env python3
"""
make_synthetic.py — Generate fake model output from ground truth, with
deliberately injected errors.

Step 2 of the validation roadmap. For each source scene and each error set it
writes three files:

    track1.txt          File A — fake 3D tracking output
    detections_2d.txt   File B — fake raw 2D detections
    injected_errors.json  the answer key: exactly what was broken, and how

Step 5 runs the validator against these and checks it reports precisely the
errors listed in the answer key. Anything else means the validator is broken.

Injection is designed so each error type produces a distinct signature across
the two layers — which is what makes the Step 6 diagnosis testable:

    error type      File B (2D)   File A (3D)   diagnosed as
    ------------------------------------------------------------------
    class_swap      wrong         wrong         model error
    mapping_shift   correct       wrong         mapping error (systematic)
    position_shift  correct       displaced     pipeline error
    fragmentation   correct       ids split     pipeline error
    deletion        missing       missing       model miss
    phantom         extra         extra         false positive
    wl_swap         correct       correct*       nothing in the validator — a
                                                 KNOWN BLIND SPOT. File A keeps
                                                 the centre and class, only w
                                                 and l are swapped, so the 1m
                                                 centre-distance match still
                                                 pairs it and the validator
                                                 reports CLEAN. Only 3D IoU
                                                 (the official metric) sees it.
    yaw_pi          correct       correct        nothing, correctly — a box is
                                                 symmetric under a 180 deg yaw,
                                                 so even 3D IoU is unchanged.
                                                 Injected to PROVE the negative.
    dup_id          correct       (fatal)        not a metric error: two rows
                                                 share (scene,class,frame,id).
                                                 The scorer's TrackEval raises.
                                                 Caught by preflight, not layers.
    clean           correct       correct        nothing — must score perfect

Usage:
    python3 scripts/make_synthetic.py --config config/config.yaml
    python3 scripts/make_synthetic.py --frames 200 --error-sets clean class_swap
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

try:
    import ijson
    import yaml
except ImportError as e:
    sys.exit(f"Missing dependency ({e.name}). Install with: pip install ijson pyyaml")


# --- GT field resolution ---------------------------------------------------

FIELD_ALIASES = {
    "object_id": ["object id", "id", "object_id"],
    "class":     ["object type", "type", "object_type", "class"],
    "loc3d":     ["3d location", "loc3d", "location", "3d_location"],
    "bbox3d":    ["3d bounding box scale", "bbox3d", "scale", "3d_bounding_box_scale"],
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


def extract_yaw(rot, yaw_index):
    """
    Convert the GT rotation field to a single yaw angle in radians.

    The rotation field's format is auto-detected by length:
      4 values -> quaternion (x, y, z, w), yaw taken about the vertical axis
      3 values -> Euler angles, yaw taken at `yaw_index`

    NOTE: the organizers' spec describes yaw as rotation about the y-axis of
    the object-centered frame, but the GT world bounds show z as the vertical
    axis. The script prints a sample of the raw rotation values on the first
    scene so this can be verified against the data rather than assumed.
    """
    if rot is None:
        return 0.0
    if len(rot) == 4:
        x, y, z, w = (float(v) for v in rot)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if len(rot) == 3:
        return float(rot[yaw_index])
    return 0.0



# --- Pre-pass ---------------------------------------------------------------

def collect_objects(gt_path, max_frames, class_table):
    """
    One streaming pass to learn which objects exist and which frames they span.

    Costs an extra read of the GT, but it buys exact planning: instead of
    rolling a die per object and hoping the rate lands, the injector picks a
    precise, logged set of victims. A test fixture whose contents depend on
    luck is not a fixture.
    """
    schema = None
    objects = {}          # oid -> class
    frames = set()
    with open(gt_path, "rb") as f:
        for frame_key, objs in ijson.kvitems(f, "", use_float=True):
            try:
                fid = int(frame_key)
            except ValueError:
                continue
            if fid >= max_frames or not isinstance(objs, list):
                continue
            frames.add(fid)
            for o in objs:
                if schema is None:
                    schema = resolve_schema(o)
                oid = o.get(schema["object_id"])
                cls = o.get(schema["class"])
                if oid is not None and cls in class_table:
                    objects[oid] = cls
    return objects, sorted(frames), schema


# --- Error injection --------------------------------------------------------

class Injector:
    """
    Applies one error set to a scene.

    Every victim is chosen up front from the known object list, so the answer
    key states exact counts rather than expected ones. Selection is seeded, so
    the same config reproduces the same fixture byte for byte.
    """

    def __init__(self, name, params, cfg, scene_info, objects, frames):
        self.name = name
        self.p = params or {}
        self.seed = cfg["synthetic"]["seed"]
        self.classes = cfg["classes"]
        self.class_names = list(cfg["classes"].keys())
        self.scene_info = scene_info
        self.objects = objects            # oid -> true class
        self.frames = frames
        self.total_frames = max(frames) + 1 if frames else 1
        self.rng = random.Random(f"{self.seed}:{name}:plan")

        self.log = {
            "error_set": name,
            "params": dict(self.p),
            "seed": self.seed,
            "objects_in_scene": len(objects),
            "class_swap": {},
            "mapping_shift": {},
            "position_shift": [],
            "deleted": [],
            "fragmented": {},
            "phantom_count": 0,
            "phantom_frames_sample": [],
            "phantom_ids_sample": [],
        }

        self._plan()

    # -- planning ----------------------------------------------------------
    def _active(self, kind):
        """An error set may be a named variant of a kind — position_shift and
        position_shift_severe are the same injection at different magnitudes,
        one below the match threshold and one above it. Both must be active
        for the kind they extend."""
        return (self.name == kind
                or self.name.startswith(kind + "_")
                or self.name == "mixed")

    def _pick(self, kind, pool=None):
        """Choose a precise number of victims: round(rate x N), at least 1 if
        the error type is active at all. An error set that silently injects
        nothing would pass Step 5 for the wrong reason."""
        if not self._active(kind):
            return []
        rate = self.p.get("rate", 0)
        pool = list(self.objects.keys()) if pool is None else list(pool)
        if not pool or rate <= 0:
            return []
        n = max(1, round(rate * len(pool)))
        return self.rng.sample(pool, min(n, len(pool)))

    def _plan(self):
        # deletion
        self.deleted = set(self._pick("deletion"))
        self.log["deleted"] = sorted(self.deleted)

        # class swap — victims must not also be deleted, or the swap is invisible
        remaining = [o for o in self.objects if o not in self.deleted]
        self.swap_map = {}
        for oid in self._pick("class_swap", remaining):
            true_cls = self.objects[oid]
            others = [c for c in self.class_names if c != true_cls]
            if not others:
                continue
            self.swap_map[oid] = self.rng.choice(others)
            self.log["class_swap"][str(oid)] = [true_cls, self.swap_map[oid]]

        # position shift
        self.shifted = set(self._pick("position_shift", remaining))
        self.log["position_shift"] = sorted(self.shifted)

        # fragmentation
        self.frag = set(self._pick("fragmentation", remaining))

        # phantoms — per frame, not per object
        self.phantom_frames = set()
        if self._active("phantom") and self.frames:
            rate = self.p.get("rate", 0)
            n = max(1, round(rate * len(self.frames)))
            self.phantom_frames = set(self.rng.sample(self.frames, min(n, len(self.frames))))
            # Count the real total, and keep only a sample of the frame list.
            # These must be separate fields: Step 5 asserts against the count,
            # and a truncated list silently understates it.
            self.log["phantom_count"] = len(self.phantom_frames)
            self.log["phantom_frames_sample"] = sorted(self.phantom_frames)[:50]

        # mapping shift — systematic, affects every object of every class
        self.map_shift = None
        if self._active("mapping_shift"):
            off = self.p.get("offset", 1)
            n = len(self.classes)
            self.map_shift = {c: (i + off) % n for c, i in self.classes.items()}
            for c, i in self.classes.items():
                self.log["mapping_shift"][c] = [i, self.map_shift[c]]
            self.log["params"]["offset"] = off

        # wl_swap / yaw_pi — geometry-only corruptions of File A. Centre and
        # class are untouched, so the validator's centre-distance match (both
        # layers) stays perfect. Their whole purpose is to expose that layer
        # 3D never looks at box shape or heading, while the official 3D-IoU
        # metric does. See docs/PROGRESS.md, step 6.5.
        self.wl_swap = set(self._pick("wl_swap", remaining))
        self.log["wl_swap"] = sorted(self.wl_swap)
        self.yaw_pi = set(self._pick("yaw_pi", remaining))
        self.log["yaw_pi"] = sorted(self.yaw_pi)
        # Self-describing markers verify.py reads. Two different kinds of
        # CLEAN: wl_swap is CLEAN because the validator is blind (only 3D IoU
        # would catch it); yaw_pi is CLEAN because the official metric itself
        # assigns zero penalty (a box is symmetric under 180deg yaw). They
        # must never look alike in a report.
        if self.wl_swap:
            self.log["blind_spot"] = "shape_via_3d_iou"
        if self.yaw_pi:
            self.log["expected_zero_impact"] = "180deg_symmetry"

        # dup_id — pick victims and a per-victim collision target, another
        # object OF THE SAME CLASS. On every shared frame the victim is
        # re-emitted under the target's id, so (scene,class,frame,id) collides.
        # This is the full-length positive case for the preflight duplicate
        # check; mapping_shift cannot produce it because the remap is a
        # bijection. See lượt-3 handoff.
        self.dup_target = {}
        if self._active("dup_id"):
            by_class = {}
            for oid, cls in self.objects.items():
                if oid not in self.deleted:
                    by_class.setdefault(cls, []).append(oid)
            for oid in self._pick("dup_id", remaining):
                pool = [o for o in by_class.get(self.objects[oid], []) if o != oid]
                if pool:
                    self.dup_target[oid] = self.rng.choice(pool)
            self.log["dup_id"] = {str(k): v for k, v in self.dup_target.items()}

    # -- per-object application --------------------------------------------
    def is_deleted(self, oid):
        return oid in self.deleted

    def swapped_class(self, oid, cls):
        """The class the model believes it sees. Wrong in BOTH layers — a
        genuine perception error, not a wiring error."""
        return self.swap_map.get(oid, cls)

    def shifted_location(self, oid, loc, rng):
        """Displace the 3D position only. Layer 2D stays correct: the detector
        saw the object fine, the lift put it somewhere else."""
        if oid not in self.shifted:
            return loc
        sigma = self.p.get("sigma_m", 0.3)
        return [loc[0] + rng.gauss(0, sigma),
                loc[1] + rng.gauss(0, sigma),
                loc[2] + rng.gauss(0, sigma) * 0.2]

    def fragmented_id(self, oid, frame_id):
        """Cut one track into several ids. Detection is perfect every frame;
        only the association breaks."""
        if oid not in self.frag:
            return oid
        splits = self.p.get("splits", 3)
        seg = min(int(frame_id / max(self.total_frames, 1) * splits), splits - 1)
        new_id = oid if seg == 0 else 800000 + int(oid) * 10 + seg
        ids = self.log["fragmented"].setdefault(str(oid), [])
        if new_id not in ids:
            ids.append(new_id)
        return new_id

    def swapped_scale(self, oid, w, l):
        """wl_swap: exchange width and length. A rectangle's footprint IoU
        against its w/l-swapped self is well under the first HOTA alpha for
        any elongated box, so it tanks the official metric — but the centre
        never moves, so the validator's match is unaffected."""
        if oid in self.wl_swap:
            return l, w
        return w, l

    def flipped_yaw(self, oid, yaw):
        """yaw_pi: rotate 180 deg. A box is symmetric under this, so IoU is
        unchanged: the fixture must score CLEAN even under the real metric.
        It exists to prove head/tail direction is worth zero points."""
        if oid in self.yaw_pi:
            a = yaw + math.pi
            while a > math.pi:
                a -= 2 * math.pi
            return a
        return yaw

    def duplicated_id(self, oid):
        """dup_id: re-label the victim with another same-class object's id,
        creating a colliding (scene,class,frame,id) on every shared frame."""
        return self.dup_target.get(oid, oid)

    def official_id(self, cls):
        """Class id as written to File A. mapping_shift diverts here and only
        here — File B keeps the true table. That asymmetry is the signature
        the validator has to detect."""
        if self.map_shift is not None:
            return self.map_shift[cls]
        return self.classes[cls]

    def phantoms(self, frame_id, rng):
        """Objects that never existed. Appear in both layers."""
        if frame_id not in self.phantom_frames:
            return []
        b = self.scene_info["bounds"]
        cams = self.scene_info["cameras"]
        cls = rng.choice(self.class_names)
        oid = 900000 + frame_id
        if len(self.log["phantom_ids_sample"]) < 50:
            self.log["phantom_ids_sample"].append(oid)
        loc = [rng.uniform(b["x"][0], b["x"][1]),
               rng.uniform(b["y"][0], b["y"][1]),
               rng.uniform(b["z"][0], b["z"][1])]
        pick = rng.sample(cams, min(len(cams), rng.randint(1, 3))) if cams else []
        boxes = {}
        for c in pick:
            x1, y1 = rng.uniform(0, 1700), rng.uniform(0, 900)
            boxes[c] = [x1, y1, x1 + rng.uniform(40, 200), y1 + rng.uniform(60, 180)]
        return [{"oid": oid, "cls": cls, "loc": loc, "scale": [0.7, 0.7, 1.7],
                 "yaw": rng.uniform(-math.pi, math.pi), "boxes": boxes}]


# --- Generation -------------------------------------------------------------

def conf_for(seed, sid, frame_id, oid):
    """Detection confidence for a File B row.

    Keyed only by (seed, scene, frame, object) — NOT by the error set — so
    File B is byte-identical across every set that does not itself change a
    File B row. That invariant is checkable in one second with md5sum, which
    is worth more than a comment explaining why the confidences drift. The
    original code drew conf from the same rng stream as phantoms, so the
    stream forked per set and File B differed for no real reason.
    """
    h = random.Random(f"{seed}:conf:{sid}:{frame_id}:{oid}")
    return 0.55 + h.random() * 0.44


def scene_id_of(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def camera_id_of(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def generate(gt_path, scene, cfg, registry_scene, error_set, params, out_dir,
             max_frames, yaw_index, objects, frames, verbose_sample=False):
    scene_info = {
        "bounds": registry_scene["bounds"],
        "cameras": registry_scene["cameras"],
    }
    inj = Injector(error_set, params, cfg, scene_info, objects, frames)
    rng = random.Random(f"{cfg['synthetic']['seed']}:{scene}:{error_set}")

    sid = scene_id_of(scene)
    stride = cfg["synthetic"].get("frame_stride", 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fa = open(out_dir / "track1.txt", "w")
    fb = open(out_dir / "detections_2d.txt", "w")

    schema = None
    rows_a = rows_b = 0
    frames_done = 0
    sample_printed = False

    with open(gt_path, "rb") as f:
        for frame_key, objects in ijson.kvitems(f, "", use_float=True):
            try:
                frame_id = int(frame_key)
            except ValueError:
                continue
            if frame_id >= max_frames:
                continue
            if stride > 1 and frame_id % stride:
                continue
            if not isinstance(objects, list):
                continue

            frames_done += 1

            for obj in objects:
                if schema is None:
                    schema = resolve_schema(obj)
                    if verbose_sample and not sample_printed:
                        raw = obj.get(schema.get("rot3d"))
                        print(f"    rotation sample: {raw}  "
                              f"({len(raw) if raw else 0} values -> "
                              f"{'quaternion' if raw and len(raw) == 4 else 'euler'})")
                        sample_printed = True

                oid = obj.get(schema["object_id"])
                true_cls = obj.get(schema["class"])
                if oid is None or true_cls not in cfg["classes"]:
                    continue

                if inj.is_deleted(oid):
                    continue

                loc = obj.get(schema["loc3d"]) or [0, 0, 0]
                scale = obj.get(schema["bbox3d"]) or [1, 1, 1]
                rot = obj.get(schema.get("rot3d"))
                boxes = obj.get(schema.get("bbox2d")) or {}

                # What the model believes it saw. Affects both layers.
                seen_cls = inj.swapped_class(oid, true_cls)

                # --- File B: raw 2D detections ---------------------------
                # Written with the true class table. mapping_shift does not
                # touch this file — that asymmetry is the whole point.
                b_cls = cfg["classes"][seen_cls]
                for cam, box in boxes.items():
                    if not box or len(box) < 4:
                        continue
                    conf = conf_for(cfg["synthetic"]["seed"], sid, frame_id, oid)
                    fb.write(f"{camera_id_of(cam)} {frame_id} {b_cls} "
                             f"{float(box[0]):.2f} {float(box[1]):.2f} "
                             f"{float(box[2]):.2f} {float(box[3]):.2f} {conf:.3f}\n")
                    rows_b += 1

                # --- File A: 3D tracking output --------------------------
                loc_out = inj.shifted_location(oid, [float(v) for v in loc[:3]], rng)
                out_id = inj.fragmented_id(oid, frame_id)
                out_id = inj.duplicated_id(out_id)   # dup_id, after frag
                a_cls = inj.official_id(seen_cls)
                yaw = inj.flipped_yaw(oid, extract_yaw(rot, yaw_index))
                w, l, h = (float(v) for v in (list(scale) + [1, 1, 1])[:3])
                w, l = inj.swapped_scale(oid, w, l)  # wl_swap: shape only
                fa.write(f"{sid} {a_cls} {out_id} {frame_id} "
                         f"{loc_out[0]:.4f} {loc_out[1]:.4f} {loc_out[2]:.4f} "
                         f"{w:.4f} {l:.4f} {h:.4f} {yaw:.4f}\n")
                rows_a += 1

            # --- phantoms -----------------------------------------------
            for ph in inj.phantoms(frame_id, rng):
                a_cls = inj.official_id(ph["cls"])
                b_cls = cfg["classes"][ph["cls"]]
                fa.write(f"{sid} {a_cls} {ph['oid']} {frame_id} "
                         f"{ph['loc'][0]:.4f} {ph['loc'][1]:.4f} {ph['loc'][2]:.4f} "
                         f"{ph['scale'][0]:.4f} {ph['scale'][1]:.4f} "
                         f"{ph['scale'][2]:.4f} {ph['yaw']:.4f}\n")
                rows_a += 1
                for cam, box in ph["boxes"].items():
                    fb.write(f"{camera_id_of(cam)} {frame_id} {b_cls} "
                             f"{box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} "
                             f"{conf_for(cfg['synthetic']['seed'], sid, frame_id, ph['oid']):.3f}\n")
                    rows_b += 1

    fa.close()
    fb.close()

    inj.log.update({
        "scene": scene,
        "scene_id": sid,
        "frames_written": frames_done,
        "rows_file_a": rows_a,
        "rows_file_b": rows_b,
        "expectation": EXPECTATIONS[error_set],
    })
    (out_dir / "injected_errors.json").write_text(json.dumps(inj.log, indent=2))
    return inj.log


# What the validator must report for each set. Step 5 asserts against this.
EXPECTATIONS = {
    "clean": "Both layers perfect. Any reported error means the validator "
             "itself is broken.",
    "class_swap": "Layer 2D and layer 3D both show the same off-diagonal "
                  "cells. Diagnosis: model error.",
    "mapping_shift": "Layer 2D clean, layer 3D wrong for every object of "
                     "every class, shifted by a constant. Diagnosis: mapping "
                     "error.",
    "position_shift": "Layer 2D clean. Displacement stays under the 1m match "
                      "threshold, so objects must still match and classes must "
                      "stay correct — this set proves the threshold tolerates "
                      "small error rather than punishing it.",
    "position_shift_severe": "Layer 2D clean. Displacement exceeds the 1m "
                             "threshold, so layer 3D must show a miss and a "
                             "ghost for each displaced object. Diagnosis: "
                             "pipeline error.",
    "deletion": "Both layers show misses on the same objects, no ghosts, no "
                "off-diagonal cells.",
    "phantom": "Both layers show ghosts, no misses, no off-diagonal cells.",
    "fragmentation": "Both layers clean on class. Object id count in File A "
                     "far exceeds the GT track count.",
    "wl_swap": "KNOWN BLIND SPOT. Both layers report CLEAN: centre and class "
               "are untouched, so the centre-distance match is perfect. The "
               "defect is real — width and length are swapped — but only 3D "
               "IoU sees it. verify.py expects CLEAN here AND asserts the "
               "blind-spot note is attached, so a future centre+shape match "
               "would flip this to a caught error and fail Step 5 loudly.",
    "yaw_pi": "Both layers CLEAN, and correctly so: a box is symmetric under "
              "a 180 deg yaw, so even the official 3D-IoU metric is unchanged. "
              "Proves head/tail direction costs no points.",
    "dup_id": "Layers are not the check here. preflight must FATAL on a "
              "duplicate (scene,class,frame,object_id); this is its "
              "full-length positive case.",
    "mixed": "All of the above at once.",
}


# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic model output from GT.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--frames", type=int, default=None,
                    help="Override synthetic.frames from config")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Override synthetic.source_scenes")
    ap.add_argument("--error-sets", nargs="*", default=None,
                    help="Only generate these sets")
    ap.add_argument("--yaw-index", type=int, default=2,
                    help="Which Euler component is yaw, if rotation is Euler")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    reg_path = Path(cfg["paths"]["class_registry"])
    if not reg_path.exists():
        sys.exit(f"{reg_path} not found. Run scripts/inspect_gt.py first.")
    registry = json.loads(reg_path.read_text())
    reg_by_scene = {s["scene"]: s for s in registry["scenes"]}

    scenes = args.scenes or cfg["synthetic"]["source_scenes"]
    sets = args.error_sets or list(cfg["synthetic"]["error_sets"].keys())
    max_frames = args.frames or cfg["synthetic"]["frames"]
    out_root = Path(cfg["paths"]["synthetic_root"])

    gt_root = Path(cfg["paths"]["gt_root"])
    summary = []

    for scene in scenes:
        if scene not in reg_by_scene:
            print(f"!! {scene} not in class_registry.json — skipped")
            continue
        matches = list(gt_root.glob(f"*/{scene}/ground_truth.json"))
        if not matches:
            print(f"!! ground_truth.json for {scene} not found — skipped")
            continue
        gt_path = matches[0]

        avail = reg_by_scene[scene]["frames"]
        print(f"\n{scene}  ({avail} frames available, using {min(max_frames, avail)})")
        print("  pre-pass ...", end=" ", flush=True)
        objects, frames, _ = collect_objects(gt_path, max_frames, cfg["classes"])
        print(f"{len(objects)} objects, {len(frames)} frames")

        for i, es in enumerate(sets):
            params = cfg["synthetic"]["error_sets"].get(es, {})
            out_dir = out_root / scene / es
            print(f"  {es:<16}", end=" ", flush=True)
            log = generate(gt_path, scene, cfg, reg_by_scene[scene], es, params,
                           out_dir, max_frames, args.yaw_index, objects, frames,
                           verbose_sample=(i == 0))
            print(f"A={log['rows_file_a']:>9}  B={log['rows_file_b']:>10}  -> {out_dir}")
            summary.append(log)

    # --- report ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("INJECTED ERRORS — the answer key for Step 5")
    print("=" * 76)
    for log in summary:
        if log["error_set"] == "clean":
            continue
        bits = []
        if log["class_swap"]:
            bits.append(f"{len(log['class_swap'])} class swaps")
        if log["mapping_shift"]:
            bits.append(f"class table shifted by {log['params'].get('offset')}")
        if log["position_shift"]:
            bits.append(f"{len(log['position_shift'])} objects displaced")
        if log["deleted"]:
            bits.append(f"{len(log['deleted'])} objects deleted")
        if log["fragmented"]:
            bits.append(f"{len(log['fragmented'])} tracks fragmented")
        if log["phantom_count"]:
            bits.append(f"{log['phantom_count']} phantoms")
        if log.get("wl_swap"):
            bits.append(f"{len(log['wl_swap'])} boxes w<->l swapped "
                        f"[blind spot: {log.get('blind_spot')}]")
        if log.get("yaw_pi"):
            bits.append(f"{len(log['yaw_pi'])} boxes yaw+180deg "
                        f"[zero-impact: {log.get('expected_zero_impact')}]")
        if log.get("dup_id"):
            bits.append(f"{len(log['dup_id'])} ids duplicated onto a "
                        f"same-class object (preflight E_DUP)")
        print(f"\n{log['scene']} / {log['error_set']}")
        print(f"  injected: {', '.join(bits) if bits else 'nothing'}")
        print(f"  expected: {log['expectation']}")

    print(f"\n\nGenerated {len(summary)} sets under {out_root}/")
    print("Each set's injected_errors.json is the ground truth for the validator "
          "itself.\nIf Step 5 can't reproduce these exactly, the validator is wrong "
          "— not the data.")


if __name__ == "__main__":
    main()