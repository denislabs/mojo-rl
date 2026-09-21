#!/usr/bin/env python
"""Do OUR goals fire on LIBERO's own successful demonstrations? — the L3 gate
that is not of my own making.

    pixi run libero-demo-success                     # dump, then run the Mojo leg
    pixi run libero-demo-success --suite libero_goal --demos 50

`tests/libero/test_libero_goal_eval.mojo` evaluates the ten goals on ELEVEN
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
§6e). `examples/libero/libero_demo_success.mojo` reads the dumps, drives our
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_demo_common import (  # noqa: E402
    DEMOS,
    SCENES,
    body_names,
    build_remap,
    convert_state,
    fixture_poses,
    rewrite,
)


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
    our_b = body_names(mo)

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
            if states.shape[1] != 1 + mt.nq + mt.nv or mt.nq != mo.nq or mt.nv != mo.nv:
                sys.exit(f"{stem}/{nm}: shape mismatch {states.shape} vs nq {mt.nq} nv {mt.nv} / ours {mo.nq} {mo.nv}")
            # the joint remap is the same for every state of this demo
            remap = build_remap(mt, mo)
            win = states[-a.window:] if a.window < len(states) else states
            lines.append(f"DEMO {nm} {len(win)} {len(states)}")
            for row in win:
                qo, vo = convert_state(row, remap, mt, mo)
                lines.append("QPOS " + " ".join(repr(float(x)) for x in qo))
                lines.append("QVEL " + " ".join(repr(float(x)) for x in vo))
            for ob, bp, bq in fixture_poses(mt, our_b):
                lines.append(
                    "FIX " + ob + " "
                    + " ".join(repr(float(x)) for x in bp)
                    + " " + " ".join(repr(float(x)) for x in bq)
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
            fh.write(f"{a.suite}__{stem} {stem + '.dump'}\n")
    print(f"wrote {index}")
    print()
    print("now:  pixi run mojo run -I . examples/libero/libero_demo_success.mojo "
          + index)


if __name__ == "__main__":
    main()
