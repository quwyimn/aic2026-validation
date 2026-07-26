#!/usr/bin/env python3
"""
test_preflight.py — assertions for the ground-truth-free submission checks.

Runs anywhere: no ground truth, no fixtures, no network. Each case writes a
tiny file with exactly one defect and asserts the rule that must fire.

The point of the negative cases is the one that matters. A checker that
flags everything is as useless as one that flags nothing, so every rule is
also given a file it must stay silent on.

    python3 tests/test_preflight.py
"""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.preflight import check_file, CLASS_ID_TO_NAME, MIN_FACE_AREA_M2


def row(scene=20, cls=0, oid=1, frame=0, x=-10.0, y=-20.0, z=0.85,
        w=0.7, l=0.7, h=1.7, yaw=0.5):
    return (f"{scene} {cls} {oid} {frame} {x} {y} {z} {w} {l} {h} {yaw}")


def write(lines, newline="\n", trailing=True):
    fd = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     newline="")
    body = newline.join(lines)
    if trailing:
        body += newline
    fd.write(body)
    fd.close()
    return fd.name


def codes(report):
    return set(report["issues"].keys())


RESULTS = []


def case(name, lines, must_have=(), must_not_have=(), **kw):
    path = write(lines) if isinstance(lines, list) else lines
    rep = check_file(path, **kw)
    got = codes(rep)
    missing = [c for c in must_have if c not in got]
    spurious = [c for c in must_not_have if c in got]
    ok = not missing and not spurious
    RESULTS.append((name, ok, missing, spurious, rep))
    return rep


# --- The file that must stay silent -----------------------------------------

CLEAN = [row(oid=1, frame=f) for f in range(5)] + \
        [row(oid=2, cls=6, frame=f, w=0.82, l=1.88, h=1.96, z=0.98)
         for f in range(5)]

case("clean file is silent", CLEAN,
     must_not_have=["E_FIELDS", "E_CR", "E_INT", "E_FLOAT", "E_CLASS",
                    "E_FRAME", "E_OBJ", "E_DEGEN", "E_DUP", "W_XCLASS",
                    "W_YAW", "W_DIM_SMALL", "W_Z_VS_H", "W_FRAME_GAPS"])

# --- Fatals ------------------------------------------------------------------

case("double space breaks the field count",
     [row(), row().replace("20 0", "20  0", 1)],
     must_have=["E_FIELDS"])

case("trailing space breaks the field count",
     [row(), row() + " "],
     must_have=["E_FIELDS"])

case("tab separator breaks the field count",
     [row(), row().replace(" ", "\t")],
     must_have=["E_FIELDS"])

case("header line breaks the field count",
     ["scene_id class_id object_id frame_id x y z w l h yaw", row()],
     must_have=["E_INT"])

case("CRLF is caught", write([row(), row(frame=1)], newline="\r\n"),
     must_have=["E_CR"])

case("class_id 7 is rejected", [row(cls=7)], must_have=["E_CLASS"])
case("class_id -1 is rejected", [row(cls=-1)], must_have=["E_CLASS"])

case("negative frame_id", [row(frame=-1)], must_have=["E_FRAME"])
case("object_id zero", [row(oid=0)], must_have=["E_OBJ"])
case("object_id negative", [row(oid=-5)], must_have=["E_OBJ"])

case("non-numeric float field", [row().replace("0.7 0.7", "0.7 NaN", 1)],
     must_have=["E_FLOAT"])
case("inf coordinate", [row(x=float("inf"))], must_have=["E_FLOAT"])

case("zero height", [row(h=0.0)], must_have=["E_DEGEN"])
case("negative width", [row(w=-0.7)], must_have=["E_DEGEN"])
# All three dimensions positive, still below the pytorch3d face-area floor:
# 0.01 * 0.005 = 5e-5 < 1e-4.
case("thin but positive box still fatal", [row(w=0.01, l=0.005, h=2.0)],
     must_have=["E_DEGEN"])
# Just above the floor: 0.02 * 0.02 = 4e-4 > 1e-4. Must be a warning only.
case("small but legal box is a warning", [row(w=0.02, l=0.02, h=2.0)],
     must_have=["W_DIM_SMALL"], must_not_have=["E_DEGEN"])

case("duplicate identity in one frame",
     [row(oid=7, cls=1, frame=3), row(oid=7, cls=1, frame=3, x=1.0)],
     must_have=["E_DUP"])
case("same id in different frames is fine",
     [row(oid=7, cls=1, frame=3), row(oid=7, cls=1, frame=4)],
     must_not_have=["E_DUP"])
case("same id in different classes is not fatal",
     [row(oid=7, cls=1, frame=3), row(oid=7, cls=2, frame=3)],
     must_have=["W_XCLASS"], must_not_have=["E_DUP"])
case("same id in different scenes is fine",
     [row(oid=7, scene=20, frame=3), row(oid=7, scene=21, frame=3)],
     must_not_have=["E_DUP", "W_XCLASS"])

case("missing expected scene is fatal", [row(scene=23)],
     must_have=["E_SCENE_MISSING"], expect_scenes=[23, 24, 25])
case("all expected scenes present",
     [row(scene=23), row(scene=24), row(scene=25)],
     must_not_have=["E_SCENE_MISSING"], expect_scenes=[23, 24, 25])

# --- Warnings ----------------------------------------------------------------

case("frame beyond the window", [row(frame=9000)],
     must_have=["W_FRAME_HI"], num_frames=9000)
case("last valid frame is silent", [row(frame=8999)],
     must_not_have=["W_FRAME_HI"], num_frames=9000)
case("1800-frame window for W026/W027", [row(scene=26, frame=1800)],
     must_have=["W_FRAME_HI"], num_frames=1800)

case("yaw out of range", [row(yaw=4.0)], must_have=["W_YAW"])
case("yaw at pi is silent", [row(yaw=math.pi)], must_not_have=["W_YAW"])

# z should sit near h/2 for anything standing on the floor. A lift that
# writes the ground contact point instead of the centroid shows up here and
# nowhere else without ground truth.
case("centroid at floor level", [row(z=0.0, h=1.7) for _ in range(10)],
     must_have=["W_Z_VS_H"])
case("centroid at half height is silent",
     [row(z=0.85, h=1.7, frame=f) for f in range(10)],
     must_not_have=["W_Z_VS_H"])

case("frame gaps", [row(frame=0), row(frame=5)], must_have=["W_FRAME_GAPS"])

case("blank line", [row(), "", row(frame=1)], must_have=["W_BLANK"])
case("no trailing newline", write([row()], trailing=False),
     must_have=["W_NO_EOL"])

# --- Exit --------------------------------------------------------------------

def main():
    width = max(len(n) for n, *_ in RESULTS)
    failed = 0
    for name, ok, missing, spurious, _ in RESULTS:
        status = "pass" if ok else "FAIL"
        line = f"  {status}  {name:<{width}}"
        if not ok:
            bits = []
            if missing:
                bits.append(f"missing {missing}")
            if spurious:
                bits.append(f"spurious {spurious}")
            line += "   " + "; ".join(bits)
            failed += 1
        print(line)
    print()
    print(f"RESULT: {len(RESULTS) - failed}/{len(RESULTS)} cases passed")
    if failed:
        print("The checker is wrong. Fix it before trusting a real run.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())