#!/usr/bin/env python3
"""The ORACLE for the tower rig's two cameras — MuJoCo's world poses, printed.

    pixi run python tools/tasks/check_tower_cameras.py

Loads the composed `scenes/so101_tower.xml`, and for two arm configurations —
the rest pose and a pose with every joint moved (so the WRIST camera's
parent-body composition is non-vacuous) — prints `cam_xpos` and the three
columns of `cam_xmat` for `robot_wrist_cam` and `tower_overhead_cam`.

`tests/tasks/test_so101_tower_cameras.mojo` pins OUR parser + FK + camera
composition against these numbers. Two routes to one pose: MuJoCo compiles
each attached asset and attaches the RESULT; we splice text and compose
`mj_camlight`'s parent-body rule ourselves
(`physics3d/kinematics/camera_frame.mojo`). If they agree at BOTH poses, the
camera the tracer renders from is where the rig's camera is — to the
accuracy of the CAD-derived pose, which calibration then refines.

It also prints the two sanity numbers `docs/camera-rig.md` §5 asks for on the
real rig: the overhead camera's height above the desk and its tilt from
vertical.
"""
import math
import os
import sys

import mujoco
import numpy as np

SCENE = "noeira/tasks/scenes/so101_tower.xml"
POSES = {
    "rest": [0.0] * 6,
    # every arm joint off zero, inside its range; wrist_roll far from zero
    "moved": [0.35, -1.10, 0.90, 0.60, 2.20, 0.45],
}


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    assert names == ["robot_wrist_cam", "tower_overhead_cam"], names
    for c in range(m.ncam):
        b = m.cam_bodyid[c]
        print("camera", c, names[c], "on body", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
              "fovy %.4f" % m.cam_fovy[c])
    for pname, q in POSES.items():
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        print("pose", pname, "qpos[:6] =", q)
        for c in range(m.ncam):
            R = d.cam_xmat[c].reshape(3, 3)
            print("  %-19s pos  %s" % (names[c], " ".join("%.9f" % v for v in d.cam_xpos[c])))
            for k, ax in enumerate("xyz"):
                print("  %-19s %s    %s" % ("", ax, " ".join("%.9f" % v for v in R[:, k])))
    # the rig sanity numbers (rest pose is irrelevant: the tower is static)
    c = names.index("tower_overhead_cam")
    R = d.cam_xmat[c].reshape(3, 3)
    axis = -R[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -axis[2]))))
    print("overhead_cam: height above desk %.3f m, %.1f deg from vertical,"
          " axis meets the desk at x=%.3f y=%.3f (robot frame)"
          % (d.cam_xpos[c][2], tilt,
             d.cam_xpos[c][0] + axis[0] * d.cam_xpos[c][2] / -axis[2],
             d.cam_xpos[c][1] + axis[1] * d.cam_xpos[c][2] / -axis[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
