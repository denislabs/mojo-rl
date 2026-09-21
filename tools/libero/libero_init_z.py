#!/usr/bin/env python3
"""Every prop's STARTING HEIGHT, read out of LIBERO's own frozen states.

    pixi run libero-init-z libero_object
    pixi run libero-init-z libero_10 libero_90 --by-scene
    pixi run libero-init-z libero_object --check

Writes `noeira/tasks/libero/init_z_<family>.kv`: one line per prop, the z its
free joint holds in `libero/libero/init_files/<suite>/*.pruned_init`. The
grouping mirrors `tools/tasks/gen_libero_family.mojo` — one file per FAMILY,
which is the suite for the first three and the SCENE for LIBERO-10/90.

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

## ⚠ THE COLUMN ORDER IS ASSUMED AND THEN CHECKED, TWO WAYS

A `.pruned_init` row is `[time, qpos, qvel]`: the robot's nine joints, then one
7-word free joint per object in the `.bddl`'s `:objects` order, then the
JOINTED FIXTURES' own dofs. Measured on `KITCHEN_SCENE5`, whose cabinet
contributes three slides — the objects come FIRST and the fixtures after, which
is the opposite of the order the first version of this file assumed.

Both halves of that are verified rather than trusted:

1. ⚠⚠ **THE QUATERNION MUST BE A QUATERNION.** Words 3-6 of a free joint are
   `(w, x, y, z)` and their norm is 1. Read at the wrong offset they are a
   slide value and three position words, and the norm is not 1 — measured
   1.226 / 1.206 / 0.520 for `KITCHEN_SCENE5` under the fixtures-first
   assumption. This is a STRUCTURAL check: it needs nothing from our side of
   the port, so it cannot agree with a mistake we also made.
2. **one z per (prop, REGION) across the group.** A prop appears in different
   subsets across a family's files; a column order that is wrong in a way the
   norm does not catch would give it different heights in different tasks.

   ⚠ PER REGION, NOT PER PROP, and `libero_spatial` is why: its
   `akita_black_bowl_1` starts at z 0.9700 on the table in one task and
   1.15063 INSIDE the cabinet's top drawer in another. robosuite's height is
   `z_offset + base_offset[2] - bottom_offset[-1]` and `base_offset` is the
   REGION's reference, so a prop has one height per region it can start in and
   the first version of this rule refused a correct file.

   The region is keyed by the `.bddl`'s OWN composed name
   (`wooden_cabinet_1_top_side`), never by the family's — a union family
   renames a region whose role moves between rectangles
   (`libero_import.RegionAlias`), and a gate reading this file reads the
   `.bddl` for the same key. Both sides stay in LIBERO's vocabulary.

The fixture dof count `k` is not read from anywhere — it is SOLVED from the
row width, `width = 19 + 2k + 13 * n_objects`, and refused unless that gives a
non-negative whole number. So a scene whose fixtures this reasoning does not
describe is refused rather than mis-read.

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
OUT_DIR = "noeira/tasks/libero"
N_ROBOT_QPOS = 9
"""Panda's seven arm joints plus the gripper's two."""
FREE_QPOS = 7
FREE_QVEL = 6
QUAT_TOL = 1e-9
"""How far a free joint's `(w, x, y, z)` may be from unit norm. They are
written to full precision; this is a structural check, not a tolerance to
tune."""
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


def bddl_inits(path):
    """`prop -> the region or prop its `:init` starts it on`.

    Both `On` and `In`, and the target may be another PROP (a stack). What is
    wanted is a stable key for "where this thing starts", and the target's own
    name is it.

    ⚠⚠ THE `:init` BLOCK ONLY. `(:goal (And (On akita_black_bowl_1 plate_1)))`
    has the same shape as an init term, and scanning the whole file let a
    GOAL overwrite the region a prop starts in — which then read as "one prop,
    two heights" and refused a correct suite. The goal says where it must END
    UP.
    """
    text = open(path).read()
    at = text.find("(:init")
    if at < 0:
        sys.exit(f"{path}: no (:init ...) block")
    # to the matching close paren
    depth = 0
    end = at
    while end < len(text):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                break
        end += 1
    out = {}
    for m in re.finditer(r"\((?:On|In) ([a-z_0-9]+) ([a-z_0-9]+)\)",
                         text[at : end + 1]):
        out[m.group(1)] = m.group(2)
    return out


def scene_family(stem):
    """`KITCHEN_SCENE10_close_...` -> `libero_kitchen_scene10`.

    The same rule as `gen_libero_family._scene_prefix`, in the other language:
    only the filename says which of the twenty scenes a LIBERO-10/90 file is
    on."""
    m = re.match(r"([A-Z_]+_SCENE\d+)_", stem)
    if m is None:
        sys.exit(f"{stem}: no `<SCENE><n>_` prefix, so --by-scene cannot"
                 " group it")
    return "libero_" + m.group(1).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+",
                    help="corpus directories, e.g. libero_object")
    ap.add_argument("--by-scene", action="store_true",
                    help="one file per SCENE across the given suites")
    ap.add_argument("--check", action="store_true",
                    help="fail if a written file would change")
    a = ap.parse_args()

    # group the corpus files exactly as gen_libero_family does
    groups = {}
    for suite in a.suites:
        bdir = os.path.join(BDDL_ROOT, suite)
        idir = os.path.join(INIT_ROOT, suite)
        for d in (bdir, idir):
            if not os.path.isdir(d):
                sys.exit(f"no {d} — this needs the LIBERO corpus")
        for bddl in sorted(glob.glob(os.path.join(bdir, "*.bddl"))):
            stem = os.path.basename(bddl)[: -len(".bddl")]
            key = scene_family(stem) if a.by_scene else suite
            groups.setdefault(key, []).append((bddl, idir, stem))

    rc = 0
    for name in sorted(groups):
        rc |= build_group(name, groups[name], a.check)
    return rc


def build_group(name, files, check):
    seen = {}          # prop -> (z, the task that first gave it)
    per_task = []
    for bddl, idir, stem in files:
        init = os.path.join(idir, stem + ".pruned_init")
        if not os.path.exists(init):
            sys.exit(f"{stem}: no .pruned_init beside the .bddl")
        objs = bddl_objects(bddl)
        where = bddl_inits(bddl)
        rows = read_pruned_init(init)
        # width = 1 + (9 + 7n + k) + (9 + 6n + k)
        slack = rows.shape[1] - (1 + 2 * N_ROBOT_QPOS
                                 + (FREE_QPOS + FREE_QVEL) * len(objs))
        if slack < 0 or slack % 2 != 0:
            sys.exit(
                f"{stem}: a row is {rows.shape[1]} wide, which leaves {slack}"
                f" words for the jointed fixtures beside {len(objs)} objects"
                " and a nine-joint robot (width = 19 + 2k + 13n). A whole,"
                " non-negative k is the only layout this reader can describe;"
                " `tools/tasks/libero_init_table.py` answers the general case"
                " by driving both models."
            )
        n_fixture_dof = slack // 2
        for k, nm in enumerate(objs):
            base = 1 + N_ROBOT_QPOS + k * FREE_QPOS
            quat = rows[:, base + 3 : base + 7]
            norm = (quat * quat).sum(axis=1) ** 0.5
            off = float(abs(norm - 1.0).max())
            if off > QUAT_TOL:
                sys.exit(
                    f"{stem}: '{nm}' has |quat| off unity by {off:.3e} at"
                    f" column {base + 3}. Words 3-6 of a free joint are a unit"
                    " quaternion, so this is not a free joint — the column"
                    " layout is wrong. See the header."
                )
            col = rows[:, base + 2]
            lo, hi = float(col.min()), float(col.max())
            if hi - lo > Z_TOL:
                sys.exit(
                    f"{stem}: '{nm}' has z spread {hi - lo:.3e} over"
                    f" {len(rows)} rows. A prop's start height has no draw in"
                    " it (object_z = z_offset + base_offset[2] -"
                    " bottom_offset[-1]) — the column order is wrong."
                )
            if nm not in where:
                sys.exit(
                    f"{stem}: '{nm}' is in (:objects) and no `:init` places"
                    " it. Its start height belongs to no region."
                )
            key = (nm, where[nm])
            if key in seen and abs(seen[key][0] - lo) > Z_TOL:
                sys.exit(
                    f"'{nm}' on '{where[nm]}' starts at {seen[key][0]!r} in"
                    f" {seen[key][1]} and {lo!r} in {stem}. One (prop,"
                    " region), two heights: the assumed `:objects` column"
                    " order does not hold. See the header."
                )
            if key not in seen:
                seen[key] = (lo, stem)
        per_task.append((stem, len(objs), len(rows), n_fixture_dof))

    lines = [
        f"# GENERATED by tools/tasks/libero_init_z.py — the z each prop's free",
        f"# joint holds in LIBERO's own frozen initial states ({len(per_task)}",
        "# .pruned_init files). Do not edit; re-run the tool.",
        "schema_version=1",
        f"family={name}",
    ]
    for key in sorted(seen):
        lines.append(f"z={key[0]}@{key[1]}:{seen[key][0]!r}")
    text = "\n".join(lines) + "\n"

    out = os.path.join(OUT_DIR, f"init_z_{name}.kv")
    old = None
    if os.path.exists(out):
        old = open(out).read()
    for stem, nobj, nrow, nfx in per_task:
        print(f"  {stem:<52} {nobj:>2} props  {nrow:>3} rows"
              f"  {nfx} fixture dof")
    print()
    print(f"  {name}: {len(seen)} (prop, region) pairs, each with ONE"
          " start height:")
    for key in sorted(seen):
        print(f"    {key[0]:<24} on {key[1]:<34} z = {seen[key][0]:+.4f}")
    if old == text:
        print(f"  unchanged {out}")
        return 0
    if check:
        print(f"  STALE {out}")
        return 1
    with open(out, "w") as fh:
        fh.write(text)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
