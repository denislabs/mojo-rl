#!/usr/bin/env python
"""LIBERO's `.pruned_init` -> our joint order, verified against MuJoCo — G12,
the first half.

    pixi run libero-init-dump                        # dump, then run the Mojo leg
    pixi run libero-init-dump --suite libero_goal --mutate

A success rate over states the run sampled for itself is not comparable with
anything (`TASK_LAYER_PLAN.md` §6.2). LIBERO froze fifty per task and every
number in the papers is measured on them, so adopting the file is the whole
difference between "our policy scores 0.62" and "our policy scores 0.62 on the
benchmark's own inits".

## ⚠⚠ WHY THIS IS NOT `np.load` AND A HEADER

`TASK_LAYER_IMPLEMENTATION.md` §5.1 says a `.pruned_init` converts "by writing
a header and nothing else". That is true only if the joint ORDER matches, and
it does not. robosuite merges the robot first and then the objects in
`:objects` order; our composer orders by SLOT. Measured on `libero_goal`:

    theirs   arm 0..8   objects 9..36            fixtures 37..40
    ours     arm 0..8   fixtures 9..12           objects 13..40

Same `nq` (41), same `nv` (37), every address different. A straight copy puts
the bowl's free-joint pose into the cabinet's three drawer slides and the
scene still loads.

## ⚠ WHAT THIS HALF DOES, AND WHAT IT DELIBERATELY DOES NOT

It reads the pickle, builds the remap BY NAME (`libero_demo_common.build_remap`,
shared with the two demo gates so the rule is written once), converts the rows,
verifies them through MuJoCo, and writes a text dump. It does NOT write the
init table: the active mask, the family key and the store format are rules that
live in Mojo (`tasks/active.mojo`, `tasks/init_table.mojo`) and a second copy
here is the shape `_a_rule_written_inline_twice_drifts` names. The Mojo leg
`examples/tasks/libero_init_freeze.mojo` reads these dumps and freezes them.

## ⚠⚠ THE VERIFICATION IS NOT THE REMAP RUN BACKWARDS

Checking a remap by inverting it proves nothing
(`_a_gate_that_shares_its_reference_implementation_is_blind`). Instead both
models are DRIVEN: their raw row into their model, our converted row into ours,
`mj_forward` on each, and then body POSES are compared — a quantity neither
`build_remap` nor `convert_state` computes. A swapped pair of joints moves a
link and the residual says so.

Poses are compared in a frame chosen per body so that the comparison is about
the JOINTS and not about placement:

  * a top-level body with a FREE joint (every movable object) — compared in the
    WORLD frame, because its seven qpos numbers ARE its world pose;
  * anything else — compared relative to its top-level ancestor, which makes
    the check immune to the two models disagreeing about where the robot base
    or a fixture sits. They do disagree, and §"the jitter" below is why.

`--mutate` swaps one adjacent pair of joints in the remap and requires the
residual to EXPLODE. Without it, "0 mismatches" is also what a comparison of
two constants prints.

## ⚠⚠ THE JITTER: WHAT A FROZEN INIT CANNOT CARRY, IN LIBERO EITHER

`bddl_base_domain._reset_internal` re-draws every FIXTURE on every reset and
writes the draw into `model.body_pos` — while `set_init_state` restores only
`(time, qpos, qvel)`. So a `.pruned_init` row is NOT paired with a fixture
placement, in LIBERO's own eval loop either: the bowl is restored to where it
was, and the cabinet it was standing in has moved.

MEASURED over the fifty recorded draws of `libero_goal` (the per-episode
`model_file`), the band is about a centimetre:

    wooden_cabinet_1   x 0.0268..0.0359   y -0.2419..-0.2357
    flat_stove_1       x -0.4200..-0.4147 y  0.2007..0.2189
    wine_rack_1        x -0.2689..-0.2516 y -0.2700..-0.2544

and yaw is a fixed point in all fifty. Our family's single static pose
(0.030, -0.240 / -0.410, 0.210 / -0.260, -0.260) is the CENTRE of each band, so
restoring a frozen row against it is exactly as valid as LIBERO's own reset and
strictly more reproducible. The tool prints our offset against the measured
band per fixture and REFUSES if we are outside it — that is the number that
says the two protocols agree, and it is checked rather than assumed.

⚠ This is also why the eval driver's five settle steps are not decoration:
LIBERO steps five zero-action frames after `set_init_state` for exactly this
reason, and a centimetre of drop is what they absorb.
"""

import argparse
import glob
import os
import pickle
import sys
import zipfile

import h5py
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_demo_common import (  # noqa: E402
    DEMOS,
    ROOT,
    SCENES,
    body_names,
    build_remap,
    convert_state,
    joint_names,
    ours,
    rewrite,
)

REMAP_DIR = os.path.join(ROOT, "mojo_rl", "tasks", "libero")

INIT_FILES = os.path.join(
    ROOT, "references", "LIBERO-master", "libero", "libero", "init_files"
)

POSE_TOL = 1e-6
"""The band between two MEASURED edges, both of them printed every run.

⚠⚠ THE FLOOR IS NOT FLOAT NOISE, IT IS THE RECORDED XML'S SIX DECIMALS.
`model_file` is MuJoCo's own XML WRITER output, and its writer rounds:

    theirs   quat="0.923785 0 0 -0.382911"          (robot0_right_hand)
    ours     0.923785244892942 0 0 -0.38291098354328656

1.0e-7 in the quaternion, and 2.2e-7 m by the time it reaches the hand and the
two fingers hanging off it. Nothing about the conversion can remove that, so a
tolerance at 1e-9 refuses every correct table — the first version of this tool
did, and the residual it printed (2.179e-07 m at `robot0_right_hand`) is what
identified the writer.

⚠ THE CEILING IS FOUR ORDERS AWAY. `--mutate` swaps one joint pair and measures
what a real remap error costs: metres. `model_floor` re-measures the lower edge
on every run and REFUSES if it has moved, so this constant is a checked claim
and not a number someone once found convenient."""

MODEL_FLOOR_MAX = 1e-6
"""What the two models may disagree about in their own `body_pos`/`body_quat`,
before any state is applied. At the measured 1.0e-7 this has an order of
headroom; a re-vendored Panda that moved a link would trip it here, with the
body named, instead of surfacing as an unexplained pose residual."""

MUTATE_FLOOR = 1e-3
"""What a single swapped joint pair must move a body by for the check to have
teeth. A metre-scale scene whose worst body moves less than a millimetre under
a permuted remap is a comparison that is not looking at the state."""


def read_pruned_init(path):
    """The fifty frozen rows, without torch.

    ⚠ `torch.save` OF A NUMPY ARRAY IS A ZIP WITH ONE STORED PICKLE. There is
    no tensor and no `torch._utils` rebuild in it — `archive/data.pkl` unpickles
    to an `ndarray` through `numpy.core.multiarray._reconstruct` alone. So this
    needs neither torch nor an inflate; asserting the type on the way out is
    what keeps that from being a silent assumption.
    """
    with zipfile.ZipFile(path) as z:
        blob = z.read("archive/data.pkl")
    a = pickle.loads(blob)
    if not isinstance(a, np.ndarray) or a.ndim != 2 or a.dtype != np.float64:
        sys.exit(
            f"{path}: expected a 2-D float64 ndarray, got "
            f"{type(a).__name__} {getattr(a, 'shape', '?')} "
            f"{getattr(a, 'dtype', '?')} — see read_pruned_init"
        )
    return a


def their_body(name):
    """Their body name -> ours, or None when this tree has no counterpart.

    ⚠ THE MOUNT IS NOT A PREFIX RENAME. robosuite calls the pedestal's root
    `mount0_base` and our composer calls it `robot_mount`; `mount0_base` ->
    `robot_base` would map it onto the ARM's root and compare two different
    bodies, which is worse than not comparing it.
    """
    if name == "mount0_base":
        return "robot_mount"
    for p in ("robot0_", "gripper0_", "mount0_"):
        if name.startswith(p):
            return "robot_" + name[len(p):]
    if name.endswith("_main"):
        return name[: -len("_main")] + "_object"
    return None


def top_level(m, b):
    """The ancestor of body `b` whose parent is the world."""
    while m.body_parentid[b] != 0:
        b = int(m.body_parentid[b])
    return b


def has_free_joint(m, b):
    for j in range(int(m.body_jntnum[b])):
        if m.jnt_type[int(m.body_jntadr[b]) + j] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False


def frames(m):
    """`[(body, anchor)]` — the frame each body is compared in. See the header.

    ⚠ A TOP-LEVEL BODY WITHOUT A FREE JOINT ANCHORS TO ITSELF, which makes its
    own comparison vacuous — and `body_pairs` drops it for exactly that reason.
    It is not world-anchored: a static fixture's world pose is the PLACEMENT
    LIBERO re-draws every reset, so comparing it absolutely reports that jitter
    (7.6 mm, measured) as if it were a conversion error. Its children still
    carry the drawer and knob angles and are compared against it.
    """
    out = []
    for b in range(m.nbody):
        t = top_level(m, b)
        out.append((b, 0 if has_free_joint(m, t) else t))
    return out


def rel_pose(d, b, anchor):
    """`b`'s pose expressed in `anchor`'s frame, as a 12-vector."""
    if anchor == 0:
        return np.concatenate([d.xpos[b], d.xmat[b]])
    R = d.xmat[anchor].reshape(3, 3)
    return np.concatenate(
        [R.T @ (d.xpos[b] - d.xpos[anchor]), (R.T @ d.xmat[b].reshape(3, 3)).ravel()]
    )


def model_floor(mt, mo, pairs):
    """How far apart the two models are BEFORE any state is applied.

    The largest `body_pos` / `body_quat` disagreement over the compared bodies.
    See `POSE_TOL`: this is the lower edge of the tolerance band, and measuring
    it each run is what keeps the tolerance honest.

    ⚠ WORLD-ANCHORED BODIES ARE SKIPPED, AND THEY ARE THE FREE OBJECTS. Their
    `body_pos` is a PARK slot — ours sits at 50 m (`park=` in the `.family`) and
    is overwritten by the free joint the instant a state is applied. Including
    them measures the parking convention, not a model difference: 5.0e+01.
    """
    worst, where = 0.0, ""
    for tb, ob, ta, _ in pairs:
        if ta == 0:
            continue
        r = max(
            float(np.max(np.abs(mt.body_pos[tb] - mo.body_pos[ob]))),
            float(np.max(np.abs(mt.body_quat[tb] - mo.body_quat[ob]))),
        )
        if r > worst:
            worst, where = r, mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_BODY, tb)
    return worst, where


def pose_residual(mt, dt, mo, do, pairs):
    """Worst body-pose disagreement over `pairs`, and where."""
    worst, where = 0.0, ""
    for tb, ob, ta, oa in pairs:
        r = float(np.max(np.abs(rel_pose(dt, tb, ta) - rel_pose(do, ob, oa))))
        if r > worst:
            worst, where = r, mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_BODY, tb)
    return worst, where


def body_pairs(mt, mo):
    """`[(their_body, our_body, their_anchor, our_anchor)]` for every mapped body.

    ⚠ REFUSES ON A DISAGREEING FRAME CHOICE. If a body is world-anchored in one
    model and ancestor-anchored in the other, the two `rel_pose` calls are not
    comparable and a large residual would read as a remap bug.

    ⚠ SELF-ANCHORED BODIES ARE DROPPED, NOT COMPARED. See `frames`: they would
    contribute an exact zero to every row and inflate the "bodies compared"
    count with checks that cannot fail.
    """
    tf, of = dict(frames(mt)), dict(frames(mo))
    ob_names = body_names(mo)
    out = []
    for tb, tn in enumerate(body_names(mt)):
        want = their_body(tn) if tn else None
        if want is None or want not in ob_names:
            continue
        ob = ob_names.index(want)
        ta, oa = tf[tb], of[ob]
        if ta == tb or oa == ob:
            continue
        if (ta == 0) != (oa == 0):
            sys.exit(
                f"body {tn!r}/{want!r}: world-anchored in one model and "
                "ancestor-anchored in the other — see body_pairs"
            )
        out.append((tb, ob, ta, oa))
    return out


def verify(mt, mo, rows, remap, pairs):
    """Drive both models from every row; return the worst pose residual."""
    dt, do = mujoco.MjData(mt), mujoco.MjData(mo)
    worst, where, at = 0.0, "", -1
    for i, row in enumerate(rows):
        dt.qpos[:] = row[1 : 1 + mt.nq]
        dt.qvel[:] = row[1 + mt.nq :]
        mujoco.mj_forward(mt, dt)
        qo, vo = convert_state(row, remap, mt, mo)
        do.qpos[:], do.qvel[:] = qo, vo
        mujoco.mj_forward(mo, do)
        r, w = pose_residual(mt, dt, mo, do, pairs)
        if r > worst:
            worst, where, at = r, w, i
    return worst, where, at


def fixture_band(files, our_model):
    """Their fifty draws per fixture body vs our one static pose.

    Returns `[(body, lo, hi, ours, inside)]` with `lo`/`hi` the per-axis extent
    of the recorded draws. See the header's "the jitter".
    """
    ob = body_names(our_model)
    draws = {}
    for path, names in files:
        f = h5py.File(path, "r")
        for nm in names:
            mt = mujoco.MjModel.from_xml_string(rewrite(f["data"][nm].attrs["model_file"]))
            for i, bn in enumerate(body_names(mt)):
                if not bn or not bn.endswith("_main"):
                    continue
                if mt.body_parentid[i] != 0 or mt.body_jntnum[i] != 0:
                    continue
                draws.setdefault(their_body(bn), []).append(np.array(mt.body_pos[i]))
        f.close()
    out = []
    for name in sorted(draws):
        a = np.array(draws[name])
        lo, hi = a.min(axis=0), a.max(axis=0)
        mine = np.array(our_model.body_pos[ob.index(name)]) if name in ob else None
        inside = mine is not None and bool(
            np.all(mine >= lo - 1e-12) and np.all(mine <= hi + 1e-12)
        )
        out.append((name, lo, hi, mine, inside, len(a)))
    return out


def write_remap_kv(suite, remap, their_joints, nq, nv, n_tasks):
    """The verified remap, as checked-in data — `state_remap_<suite>.kv`.

    ⚠⚠ WHY THIS IS A FILE AND NOT A RETURN VALUE. The remap is the ONE thing in
    this conversion that needs a MuJoCo compile of the recorded `model_file`,
    and the recorded models live in the ~6 GB of gitignored demonstrations. Two
    other consumers want it and neither should need them: the demo importer
    (G13, `mojo_rl/data/libero_demos.mojo`, which reads their HDF5 natively and
    has no MuJoCo and no attribute bindings) and anyone rewriting a recorded
    state by hand.

    ⚠ IT IS WRITTEN ONLY AFTER THE POSE CHECK HAS PASSED ON EVERY TASK, and it
    is asserted identical across them (`main`). A per-task remap would be a
    finding about the scene, not a row in this file.

    Same shape as `categories.kv`: `key=value` lines, `#` comments, and every
    number quoted from somewhere rather than chosen.
    """
    out = os.path.join(REMAP_DIR, f"state_remap_{suite}.kv")
    lines = [
        f"# LIBERO's recorded joint order -> ours, for {suite}. GENERATED by",
        "# tools/tasks/libero_init_table.py — do not edit; re-run the tool.",
        "#",
        "# robosuite merges the robot first and then the objects in `:objects`",
        "# order; our composer orders by SLOT. Same nq, same nv, different",
        "# addresses — so a recorded `states` row copied straight across puts an",
        "# object's free-joint pose into a fixture's slide joints and the scene",
        "# still loads.",
        "#",
        "# ⚠ VERIFIED, NOT DERIVED HERE. Every row below was checked by driving",
        "# both models through `mj_forward` from all fifty frozen inits of all",
        f"# {n_tasks} tasks and comparing BODY POSES — a quantity the remap does not",
        "# compute. Worst residual 2.18e-07 m, which is the recorded XML's own",
        "# six-decimal `quat` on robot0_right_hand and not the conversion.",
        "#",
        "# joint=<theirs> <ours> <their_qadr> <our_qadr> <nq_j> <their_vadr> <our_vadr> <nv_j>",
        "schema_version=1",
        f"family={suite}",
        f"nq={nq}",
        f"nv={nv}",
        f"tasks={n_tasks}",
    ]
    for k, (ta, oa, nqj, tv, ov, nvj) in enumerate(remap):
        tn = their_joints[k]
        lines.append(f"joint={tn} {ours(tn)} {ta} {oa} {nqj} {tv} {ov} {nvj}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mutate", action="store_true",
                    help="also prove the verification catches a permuted remap")
    a = ap.parse_args()

    out_dir = a.out or os.path.join(DEMOS, "_init", a.suite)
    os.makedirs(out_dir, exist_ok=True)

    scene = os.path.join(SCENES, a.suite + ".xml")
    if not os.path.exists(scene):
        sys.exit(f"no composed scene at {scene} — run `pixi run libero-family` first")
    mo = mujoco.MjModel.from_xml_path(scene)

    demo_files = sorted(glob.glob(os.path.join(DEMOS, a.suite, "*_demo.hdf5")))
    if not demo_files:
        sys.exit(
            f"no demo files under {os.path.join(DEMOS, a.suite)}. THEIR joint order"
            " comes from the recorded `model_file`, and neither robosuite nor the"
            " LIBERO package is importable in this environment, so there is no"
            " second source for it. Fetch the demonstrations."
        )

    print(f"{a.suite}: our scene nq {mo.nq} nv {mo.nv}, {len(demo_files)} demo files")
    print()

    written, first_remap, first_task = [], None, ""
    total_rows = 0
    worst_all, worst_where, worst_task = 0.0, "", ""
    worst_floor = 0.0
    for path in demo_files:
        stem = os.path.basename(path)[: -len("_demo.hdf5")]
        pruned = os.path.join(INIT_FILES, a.suite, stem + ".pruned_init")
        if not os.path.exists(pruned):
            sys.exit(f"{stem}: no {pruned}")
        rows = read_pruned_init(pruned)

        f = h5py.File(path, "r")
        mt = mujoco.MjModel.from_xml_string(rewrite(f["data"]["demo_0"].attrs["model_file"]))
        f.close()
        if rows.shape[1] != 1 + mt.nq + mt.nv:
            sys.exit(
                f"{stem}: {pruned} rows are {rows.shape[1]} wide but their model is"
                f" 1 + nq {mt.nq} + nv {mt.nv} = {1 + mt.nq + mt.nv}"
            )
        if mt.nq != mo.nq or mt.nv != mo.nv:
            sys.exit(
                f"{stem}: their nq/nv is {mt.nq}/{mt.nv} and ours is {mo.nq}/{mo.nv}."
                " The remap is by name and cannot bridge a different joint SET."
            )
        remap = build_remap(mt, mo)

        # ⚠ ONE REMAP FOR THE FAMILY, ASSERTED PER TASK. Every task of a suite
        # shares the scene, so a task whose `:objects` order differs would be a
        # real finding — and would silently produce a table half of whose rows
        # are in a different order.
        if first_remap is None:
            first_remap, first_task = remap, stem
        elif remap != first_remap:
            sys.exit(
                f"{stem}: its joint remap differs from {first_task}'s. Every task of"
                " a suite composes the same scene, so this is a scene difference,"
                " not a conversion detail."
            )

        pairs = body_pairs(mt, mo)
        floor, floor_at = model_floor(mt, mo, pairs)
        if floor > MODEL_FLOOR_MAX:
            sys.exit(
                f"{stem}: the two models already disagree by {floor:.3e} at"
                f" {floor_at!r} with no state applied, over the"
                f" {MODEL_FLOOR_MAX:.0e} this tolerance assumes. See POSE_TOL —"
                " the vendored robot or an asset has moved and the pose check"
                " can no longer separate that from a remap error."
            )
        worst_floor = max(worst_floor, floor)
        worst, where, at = verify(mt, mo, rows, remap, pairs)
        if worst > worst_all:
            worst_all, worst_where, worst_task = worst, where, stem
        if worst > POSE_TOL:
            sys.exit(
                f"{stem}: worst body-pose residual {worst:.3e} m at {where!r}"
                f" (row {at}) exceeds {POSE_TOL:.0e}. The remap put a joint value"
                " at the wrong address."
            )

        lines = [f"{len(rows)} {1 + mo.nq + mo.nv}"]
        for row in rows:
            qo, vo = convert_state(row, remap, mt, mo)
            lines.append(
                "ROW " + repr(float(row[0])) + " "
                + " ".join(repr(float(x)) for x in qo) + " "
                + " ".join(repr(float(x)) for x in vo)
            )
        dump = os.path.join(out_dir, stem + ".dump")
        with open(dump, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        written.append((stem, len(rows), dump))
        total_rows += len(rows)
        print(f"  {stem:<52} {len(rows):>3} rows  {len(pairs):>3} bodies compared"
              f"  worst {worst:.2e} m")

    print()
    print(f"  remap ({len(first_remap)} joints, identical across all"
          f" {len(written)} tasks):")
    tj, oj = None, joint_names(mo)
    f = h5py.File(demo_files[0], "r")
    mt0 = mujoco.MjModel.from_xml_string(rewrite(f["data"]["demo_0"].attrs["model_file"]))
    f.close()
    tj = joint_names(mt0)
    moved = 0
    for k, (ta, oa, nqj, tv, ov, nvj) in enumerate(first_remap):
        if ta != oa:
            moved += 1
    for k, (ta, oa, nqj, tv, ov, nvj) in enumerate(first_remap):
        flag = "  <- moved" if ta != oa else ""
        print(f"    {tj[k]:<32} q {ta:>3} -> {oa:>3} ({nqj})   {ours(tj[k]):<28}{flag}")
    print(f"    {moved} of {len(first_remap)} joints change address; a straight"
          " copy would be wrong for every one of them")

    if a.mutate:
        # ⚠ THE ANTI-VACUITY LEG. Swap the destinations of one adjacent pair and
        # require the pose check to notice. It is run on the LAST task's models,
        # which are still bound above.
        print()
        best = 0.0
        for k in range(len(first_remap) - 1):
            mut = list(first_remap)
            x, y = mut[k], mut[k + 1]
            mut[k] = (x[0], y[1], y[2], x[3], y[4], y[5])
            mut[k + 1] = (y[0], x[1], x[2], y[3], x[4], x[5])
            if mut[k][2] != x[2] or mut[k + 1][2] != y[2]:
                continue  # a free/hinge pair: the widths differ, convert_state would raise
            w, _, _ = verify(mt, mo, rows[:4], mut, pairs)
            best = max(best, w)
        if best < MUTATE_FLOOR:
            sys.exit(
                f"--mutate: the worst permuted remap moved nothing by more than"
                f" {best:.3e} m. The pose check is not looking at the state."
            )
        print(f"  --mutate: a swapped joint pair moves a body by up to {best:.4f} m,"
              f" against a {POSE_TOL:.0e} tolerance — the check has teeth")

    print()
    print("  fixture jitter LIBERO re-draws every reset and never restores"
          " (see the header):")
    band = fixture_band([(p, ["demo_%d" % i for i in range(50)]) for p in demo_files[:1]], mo)
    bad = []
    for name, lo, hi, mine, inside, n in band:
        if mine is None:
            continue
        span = np.round((hi - lo) * 1000, 2)
        print(f"    {name:<28} span mm {span.tolist()}   ours"
              f" {np.round(mine, 4).tolist()}   {'inside' if inside else 'OUTSIDE'}"
              f" ({n} draws)")
        if not inside:
            bad.append(name)
    if bad:
        sys.exit(
            "our static fixture pose is outside the band LIBERO itself draws from"
            f" for {bad}. A row frozen against their draw would be restored against"
            " a fixture this benchmark never places there."
        )

    kv = write_remap_kv(a.suite, first_remap, tj, mo.nq, mo.nv, len(written))
    print(f"  wrote {kv}")

    index = os.path.join(out_dir, "index.txt")
    with open(index, "w") as fh:
        for stem, n, dump in written:
            fh.write(f"{a.suite}__{stem} {n} {dump}\n")
    print()
    print(f"  {total_rows} rows over {len(written)} tasks, worst residual"
          f" {worst_all:.2e} m at {worst_where!r} ({worst_task})")
    print(f"  the two models' own floor, no state applied: {worst_floor:.2e}"
          f" (the recorded XML's six-decimal quat — see POSE_TOL)")
    print(f"wrote {index}")
    print()
    print("now:  pixi run mojo run -I . examples/tasks/libero_init_freeze.mojo "
          + index)


if __name__ == "__main__":
    main()
