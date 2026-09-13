#!/usr/bin/env python
"""Do OUR goals fire on LIBERO's own successful demonstrations? — the L3 gate
that is not of my own making.

    pixi run libero-demo-success                     # dump, then run the Mojo leg
    pixi run libero-demo-success --suite libero_goal --demos 50

`tests/tasks/test_libero_goal_eval.mojo` evaluates the ten goals on ELEVEN
STATES I BUILT. That is the blind shape this tree keeps paying for
(`_a_gate_that_shares_its_reference_implementation_is_blind`): I constructed
each state to satisfy the predicate I had just written, so the two agree by
construction and a wrong threshold, an inverted argument or a mis-anchored
box would survive.

LIBERO's demonstrations are ground truth. Every one is a SUCCESSFUL human
teleoperation of exactly the task whose `.task` file we generated, so the
state it ends in is a state the benchmark calls solved. If our goal does not
fire there, our goal is wrong — no interpretation needed.

This half reads the HDF5 and writes one dump per task: for each demo, the
final `states` row converted into OUR joint order BY NAME, and that demo's
own fixture placement (LIBERO redraws it every episode; see the assessment's
§6e). `examples/tasks/libero_demo_success.mojo` reads the dumps, drives our
engine to that state, runs collision, evaluates the bound goal, and prints a
table. This file then has nothing left to check — the verdict is the Mojo
leg's, so there is no second implementation of a predicate here to drift.

⚠⚠ A WINDOW OF TRAILING STATES, NOT THE FINAL ONE — AND THAT WAS A REAL
CORRECTION. The first version of this gate asked only whether the LAST
recorded state satisfies our goal, and scored 498 of 500: one demo of
`put_the_bowl_on_the_plate` and one of
`put_the_wine_bottle_on_top_of_the_cabinet` came out False. The predicates
were right and the QUESTION was wrong. LIBERO's protocol
(`lifelong/metric.py`) is `done = _check_success()` EVERY step with success
at ANY step, and recording keeps going afterwards — the stove demos satisfy
their goal about eleven steps before they end. So a human can nudge the bowl
off the plate after the benchmark has already scored the episode, and the
final frame is legitimately unsolved.

`--window K` dumps the last K states of each demo and the Mojo leg asks
whether the goal fires ANYWHERE in that window, which is the benchmark's own
question. A demo that does not fire within K is reported with its window
size so the number can be raised rather than guessed at.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import h5py
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEMOS = os.path.join(ROOT, "references", "libero_demos")
SCENES = os.path.join(ROOT, "mojo_rl", "tasks", "scenes")
RS14 = os.environ.get(
    "ROBOSUITE_140", os.path.join(ROOT, "references", "robosuite-1.4.0", "robosuite")
)
PACK = os.path.join(ROOT, "mojo_rl", "tasks", "libero", "assets")


def rewrite(xml):
    xml = xml.replace("/Users/yifengz/workspace/robosuite-master/robosuite", RS14)
    xml = xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets", PACK)
    return xml.replace(
        '<compiler angle="radian" meshdir="meshes/" autolimits="true"/>',
        '<compiler angle="radian" meshdir="meshes/" autolimits="true" inertiagrouprange="0 0"/>',
        1,
    )


def ours(name):
    for p in ("robot0_", "gripper0_"):
        if name.startswith(p):
            return "robot_" + name[len(p):]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--window", type=int, default=40,
                    help="how many trailing states per demo to dump (see the header)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(DEMOS, "_dumps", a.suite)
    os.makedirs(out_dir, exist_ok=True)

    scene = os.path.join(SCENES, a.suite + ".xml")
    mo = mujoco.MjModel.from_xml_path(scene)
    our_j = [mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mo.njnt)]
    our_b = [mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mo.nbody)]

    files = sorted(glob.glob(os.path.join(DEMOS, a.suite, "*_demo.hdf5")))
    if not files:
        sys.exit(f"no demo files under {os.path.join(DEMOS, a.suite)} — see the header")
    print(f"{a.suite}: {len(files)} demo files, scene nq {mo.nq} nv {mo.nv}")
    written = []
    for path in files:
        stem = os.path.basename(path)[: -len("_demo.hdf5")]
        f = h5py.File(path, "r")
        data = f["data"]
        names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))[: a.demos]
        lines = []
        n_ok = 0
        for nm in names:
            g = data[nm]
            states = np.array(g["states"])
            xml = rewrite(g.attrs["model_file"])
            tmp = os.path.join(out_dir, "_m.xml")
            with open(tmp, "w") as fh:
                fh.write(xml)
            mt = mujoco.MjModel.from_xml_path(tmp)
            their_j = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mt.njnt)]
            their_b = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mt.nbody)]
            if states.shape[1] != 1 + mt.nq + mt.nv or mt.nq != mo.nq or mt.nv != mo.nv:
                sys.exit(f"{stem}/{nm}: shape mismatch {states.shape} vs nq {mt.nq} nv {mt.nv} / ours {mo.nq} {mo.nv}")
            # the joint remap is the same for every state of this demo
            remap = []
            for j, jn in enumerate(their_j):
                oj = our_j.index(ours(jn))
                nqj = (int(mo.jnt_qposadr[oj + 1]) if oj + 1 < mo.njnt else mo.nq) - int(mo.jnt_qposadr[oj])
                nvj = (int(mo.jnt_dofadr[oj + 1]) if oj + 1 < mo.njnt else mo.nv) - int(mo.jnt_dofadr[oj])
                remap.append((int(mt.jnt_qposadr[j]), int(mo.jnt_qposadr[oj]), nqj,
                              int(mt.jnt_dofadr[j]), int(mo.jnt_dofadr[oj]), nvj))
            win = states[-a.window:] if a.window < len(states) else states
            lines.append(f"DEMO {nm} {len(win)} {len(states)}")
            for row in win:
                qt = row[1:1 + mt.nq]
                vt = row[1 + mt.nq:]
                qo = np.zeros(mo.nq)
                vo = np.zeros(mo.nv)
                for ta, oa, nqj, tv, ov, nvj in remap:
                    qo[oa:oa + nqj] = qt[ta:ta + nqj]
                    vo[ov:ov + nvj] = vt[tv:tv + nvj]
                lines.append("QPOS " + " ".join(repr(float(x)) for x in qo))
                lines.append("QVEL " + " ".join(repr(float(x)) for x in vo))
            for i, bn in enumerate(their_b):
                if not bn.endswith("_main") or mt.body_parentid[i] != 0 or mt.body_jntnum[i] != 0:
                    continue
                ob = bn[:-5] + "_object"
                if ob not in our_b:
                    sys.exit(f"{stem}: no body {ob} in {scene}")
                lines.append(
                    "FIX " + ob + " "
                    + " ".join(repr(float(x)) for x in mt.body_pos[i])
                    + " " + " ".join(repr(float(x)) for x in mt.body_quat[i])
                )
            n_ok += 1
            os.remove(tmp)
        dump = os.path.join(out_dir, stem + ".dump")
        with open(dump, "w") as fh:
            fh.write(f"{n_ok}\n" + "\n".join(lines) + "\n")
        written.append((stem, n_ok))
        print(f"  {stem:<52} {n_ok} demos x <={a.window} trailing states")
        f.close()
    index = os.path.join(out_dir, "index.txt")
    with open(index, "w") as fh:
        for stem, n in written:
            fh.write(f"{a.suite}__{stem} {os.path.join(out_dir, stem + '.dump')}\n")
    print(f"wrote {index}")
    print()
    print("now:  pixi run mojo run -I . examples/tasks/libero_demo_success.mojo "
          + index)


if __name__ == "__main__":
    main()
