#!/usr/bin/env python3
"""Every prop's STARTING HEIGHT, read out of LIBERO's own frozen states.

    pixi run libero-init-z --suite libero_object
    pixi run libero-init-z --suite libero_object --check

Writes `mojo_rl/tasks/libero/init_z_<suite>.kv`: one line per prop, the z its
free joint holds in `libero/libero/init_files/<suite>/*.pruned_init`.

## ⚠⚠ WHY THIS EXISTS — THE SAMPLER HAD A CENTIMETRE MISSING

`sampler.sample_placements` placed a prop at `workspace_site_z - bottom_z`,
transcribed from `SiteRegionRandomSampler`. That is the FIXTURE sampler.
LIBERO routes a table or floor region to `TableRegionSampler`
(`envs/regions/__init__.py`'s `REGION_SAMPLERS`), whose own signature carries
`z_offset=0.01` and to which `bddl_base_domain` passes none — so every prop on
a table or a floor starts a centimetre higher than this tree put it. On
`libero_object`, whose props lie on the ground plane rather than on a table,
that centimetre was the difference between resting clear and starting 2.5 cm
inside the floor.

Reading it out of the FROZEN STATES rather than out of the code is the point:
the states are what the benchmark actually restores at reset, and they are one
number per prop with no draw in them (`object_z = z_offset + base_offset[2] -
bottom_offset[-1]`, no sampled term).

## ⚠ THE COLUMN ORDER IS ASSUMED AND THEN CHECKED

A `.pruned_init` row is `[time, qpos, qvel]` with the robot's nine joints
first and then one 7-word free joint per object, in the `.bddl`'s `:objects`
order. That order is an assumption about `objects_dict`, so it is VERIFIED
rather than trusted: each suite's props appear in different subsets across its
files, and a wrong column order would give a prop a different z in different
tasks. Requiring one z per prop across the whole suite is what makes the
assumption falsifiable, and it is why this refuses instead of writing the
first value it saw.

⚠ NEEDS NEITHER torch NOR mujoco NOR the LIBERO package — a `.pruned_init` is
a zip with one STORED pickle of an ndarray (`libero_init_table.read_pruned_init`
records why).
"""
import argparse
import glob
import os
import pickle
import re
import sys
import zipfile

BDDL_ROOT = "references/LIBERO-master/libero/libero/bddl_files"
INIT_ROOT = "references/LIBERO-master/libero/libero/init_files"
OUT_DIR = "mojo_rl/tasks/libero"
N_ROBOT_QPOS = 9
"""Panda's seven arm joints plus the gripper's two."""
FREE_QPOS = 7
FREE_QVEL = 6
Z_TOL = 1e-12
"""Two rows of one prop must agree to this. They are the same float in
practice; the tolerance is here so a disagreement is REPORTED with its size
rather than hidden by an exact comparison that happens to hold."""


def read_pruned_init(path):
    """The frozen rows, without torch — see `libero_init_table`."""
    with zipfile.ZipFile(path) as z:
        blob = z.read("archive/data.pkl")
    a = pickle.loads(blob)
    if not hasattr(a, "ndim") or a.ndim != 2:
        sys.exit(f"{path}: expected a 2-D ndarray, got {type(a).__name__}")
    return a


def bddl_objects(path):
    """The `:objects` instance names, in file order.

    ⚠ NOT `name - category` TRIPLES. One line may declare several instances of
    one category (`akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl`
    in every `libero_spatial` file), so the names are what precedes each `-`.
    """
    m = re.search(r"\(:objects(.*?)\n  \)", open(path).read(), re.S)
    if m is None:
        sys.exit(f"{path}: no (:objects ...) block")
    out, pending, expect_category = [], [], False
    for tok in m.group(1).split():
        if expect_category:
            expect_category = False
            continue
        if tok == "-":
            out.extend(pending)
            pending = []
            expect_category = True
            continue
        pending.append(tok)
    if pending or expect_category is False and not out:
        sys.exit(f"{path}: (:objects) does not end on a category")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--check", action="store_true",
                    help="fail if the written file would change")
    a = ap.parse_args()

    bdir = os.path.join(BDDL_ROOT, a.suite)
    idir = os.path.join(INIT_ROOT, a.suite)
    for d in (bdir, idir):
        if not os.path.isdir(d):
            sys.exit(f"no {d} — this needs the LIBERO corpus")

    seen = {}          # prop -> (z, the task that first gave it)
    per_task = []
    for bddl in sorted(glob.glob(os.path.join(bdir, "*.bddl"))):
        stem = os.path.basename(bddl)[: -len(".bddl")]
        init = os.path.join(idir, stem + ".pruned_init")
        if not os.path.exists(init):
            sys.exit(f"{stem}: no .pruned_init beside the .bddl")
        objs = bddl_objects(bddl)
        rows = read_pruned_init(init)
        want = 1 + N_ROBOT_QPOS + FREE_QPOS * len(objs) \
            + N_ROBOT_QPOS + FREE_QVEL * len(objs)
        if rows.shape[1] != want:
            sys.exit(
                f"{stem}: a row is {rows.shape[1]} wide; {len(objs)} objects"
                f" and a nine-joint robot make {want} ([time, qpos, qvel])."
                "\n  ⚠ A SUITE WHOSE SCENE HAS A JOINTED FIXTURE DOES NOT FIT"
                " THIS READER. `libero_goal`'s cabinet contributes three"
                " slides and its stove a hinge, and where those four words sit"
                " among the object blocks is exactly the question"
                " `tools/tasks/libero_init_table.py` answers — by driving both"
                " models and comparing body poses. Use that for such a suite;"
                " this one is for a scene of free props only."
            )
        for k, nm in enumerate(objs):
            base = 1 + N_ROBOT_QPOS + k * FREE_QPOS
            col = rows[:, base + 2]
            lo, hi = float(col.min()), float(col.max())
            if hi - lo > Z_TOL:
                sys.exit(
                    f"{stem}: '{nm}' has z spread {hi - lo:.3e} over"
                    f" {len(rows)} rows. A prop's start height has no draw in"
                    " it (object_z = z_offset + base_offset[2] -"
                    " bottom_offset[-1]) — the column order is wrong."
                )
            if nm in seen and abs(seen[nm][0] - lo) > Z_TOL:
                sys.exit(
                    f"'{nm}' starts at {seen[nm][0]!r} in {seen[nm][1]} and"
                    f" {lo!r} in {stem}. One prop, two heights: the assumed"
                    " `:objects` column order does not hold. See the header."
                )
            if nm not in seen:
                seen[nm] = (lo, stem)
        per_task.append((stem, len(objs), len(rows)))

    lines = [
        f"# GENERATED by tools/tasks/libero_init_z.py --suite {a.suite}",
        f"# from {idir} ({len(per_task)} files) — the z each prop's free",
        "# joint holds in LIBERO's own frozen initial states. Do not edit.",
        "schema_version=1",
        f"suite={a.suite}",
    ]
    for nm in sorted(seen):
        lines.append(f"z={nm}:{seen[nm][0]!r}")
    text = "\n".join(lines) + "\n"

    out = os.path.join(OUT_DIR, f"init_z_{a.suite}.kv")
    old = None
    if os.path.exists(out):
        old = open(out).read()
    for stem, nobj, nrow in per_task:
        print(f"  {stem:<52} {nobj:>2} props  {nrow:>3} rows")
    print()
    print(f"  {len(seen)} props, each with ONE start height across the suite:")
    for nm in sorted(seen):
        print(f"    {nm:<24} z = {seen[nm][0]:+.4f}")
    if old == text:
        print(f"  unchanged {out}")
        return 0
    if a.check:
        print(f"  STALE {out}")
        return 1
    with open(out, "w") as fh:
        fh.write(text)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
