"""Layer-1 gate for the SO-ARM ports: our XML vs the reference, both via MuJoCo.

    pixi run python tests/robots/so_arm_ref.py

The pattern is `docs/DM_CONTROL_PORT.md`'s standing invariant, and the same one
`tests/dm_control/mjmodel_diff.py` implements for the suite:

    compile OUR ported XML with MuJoCo, compile the REFERENCE's XML with
    MuJoCo, and diff every table.

Both sides are MuJoCo, so neither our parser nor our engine is in the loop: a
mismatch isolates the XML TEXT. Tolerance is **0.0**, not an epsilon — the same
compiler ran on both.

⚠⚠ THIS IS THE GATE THAT MAKES THE THREE DEVIATIONS SAFE. `so_arm_bake.py`
re-spells `inheritrange`, `dampratio` and `fullinertia` into forms our parser
implements. Each is a derived value, so a wrong substitution is invisible to
every downstream gate — a bad `kv` reads as a badly tuned servo, a bad `iquat`
leaves total mass and every scalar moment correct while rotating the inertia
frame. Only a constant-by-constant diff against the reference can see them.

⚠ WHAT IT COMPARES IS `SO_ARM10x_ROBOT_XML`, NOT `SO_ARM10x_XML`. The reach
target is a body the reference does not have, so the task fragment is excluded
by construction rather than by a skip list. The env's model is
`merge_mjcf(ROBOT, TASK)`; if that merge ever stops being additive this gate
will not notice, which is what `test_so_arm100_vs_mujoco.mojo`'s index pins are
for.

⚠ `references/` IS GITIGNORED. Both reference trees are local-only:

    references/mujoco_menagerie-main/trs_so_arm100/
    references/SO-ARM100-main/Simulation/SO101/   (from
        https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)

Absent either one, the corresponding arm SKIPS with a message rather than
passing quietly — a gate that silently checks nothing is worse than a red one.

⚠ MESH PATHS ARE REPO-ROOT-RELATIVE (`mojo_rl/envs/robots/assets/...`), so this
must run FROM THE REPO ROOT. It chdir's there itself so a stray cwd cannot turn
into a mesh-not-found that reads as a model difference.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tests", "dm_control"))
sys.path.insert(0, HERE)

import mjmodel_diff  # noqa: E402
import so_arm_bake  # noqa: E402


ARMS = {
    "so_arm100": os.path.join(so_arm_bake.SO100_REF, "scene.xml"),
    "so_arm101": os.path.join(so_arm_bake.SO101_REF, "scene.xml"),
}


def reference(which):
    import mujoco

    return mujoco.MjModel.from_xml_path(ARMS[which])


def ours(which):
    """Compile the string AS CHECKED IN — not the baker's output.

    ⚠ The distinction is the whole point. Compiling `so_arm_bake.bake_*()` here
    would gate the generator against the reference and say nothing about the
    `.mojo` that actually ships; the two drift the moment someone edits the
    module by hand or forgets `--inject`.
    """
    import mujoco

    return mujoco.MjModel.from_xml_string(so_arm_bake.extract(which))


def check(which):
    """Returns a list of mismatch strings; empty means identical."""
    return mjmodel_diff.diff_models(reference(which), ours(which))


def check_full_model(which):
    """The env's model must be the robot PLUS the target body, and nothing else.

    ⚠⚠ THIS EXISTS BECAUSE THE TWO STRINGS ARE GENERATED SEPARATELY. The bake
    inlines the task body itself rather than calling `merge_mjcf`, which
    silently relocates SO-101's nested `<default>` into `<asset>` and makes the
    model uncompilable. Emitting two independent strings removes that failure
    mode and introduces a new one — they could drift — so this asserts the
    exact relationship: same counts everywhere except one body, one geom and
    one mocap, and every robot-side table identical over its original range.
    """
    import mujoco
    import numpy as np

    r = ours(which)
    f = mujoco.MjModel.from_xml_string(so_arm_bake.extract_full(which))
    bad = []
    if f.nbody != r.nbody + 1:
        bad.append("nbody: robot {} -> full {} (expected +1)".format(
            r.nbody, f.nbody))
    if f.ngeom != r.ngeom + 1:
        bad.append("ngeom: robot {} -> full {} (expected +1)".format(
            r.ngeom, f.ngeom))
    if f.nmocap != 1:
        bad.append("nmocap {} (expected 1 — the target)".format(f.nmocap))
    for n in ("nq", "nv", "nu", "njnt", "nexclude"):
        if getattr(f, n) != getattr(r, n):
            bad.append("{}: robot {} != full {} — the task fragment must not"
                       " touch the robot".format(n, getattr(r, n),
                                                 getattr(f, n)))
    # Robot-side rows must be untouched by the append.
    for n in ("body_mass", "body_inertia", "body_iquat", "body_pos",
              "jnt_range", "dof_armature", "actuator_gainprm",
              "actuator_biasprm", "actuator_ctrlrange"):
        a = np.asarray(getattr(r, n), dtype=np.float64)
        b = np.asarray(getattr(f, n), dtype=np.float64)[: len(a)]
        if a.shape != b.shape or np.abs(a - b).max() > 0.0:
            bad.append("{}: robot rows changed when the task body was added"
                       .format(n))
    return bad


def _inertia_detail(which):
    """Per-body `body_iquat` / `body_inertia`, printed for the SO-101 bake.

    Redundant with `diff_models` — `body_iquat` and `body_inertia` are both in
    its table list — and kept anyway, because the `fullinertia` substitution is
    the one deviation whose failure mode is a *valid but different* frame. When
    it breaks, "body_iquat[3][2]: ref x != ours y" is a worse message than a
    per-body table.
    """
    import numpy as np

    r, g = reference(which), ours(which)
    import mujoco

    rows = []
    for b in range(r.nbody):
        name = mujoco.mj_id2name(r, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
        dq = float(np.abs(r.body_iquat[b] - g.body_iquat[b]).max())
        di = float(np.abs(r.body_inertia[b] - g.body_inertia[b]).max())
        dm = float(abs(r.body_mass[b] - g.body_mass[b]))
        rows.append((name, dq, di, dm))
    return rows


def main():
    os.chdir(REPO)
    rc = 0
    ran = 0
    for which, ref_path in ARMS.items():
        print("=" * 70)
        print("  {}   ({} tables, {} checks)".format(
            which, mjmodel_diff.n_tables(), mjmodel_diff.n_checks()))
        print("=" * 70)
        if not os.path.exists(ref_path):
            print("  SKIP — reference absent: {}".format(ref_path))
            print("  (references/ is gitignored; fetch it locally)")
            continue
        ran += 1
        bad = check(which)
        if which == "so_arm101":
            print("  per-body inertia (the fullinertia bake):")
            for name, dq, di, dm in _inertia_detail(which):
                print("    {:24s} |dquat| {:.3e}  |dinertia| {:.3e}"
                      "  |dmass| {:.3e}".format(name, dq, di, dm))
        bad += ["full-model: " + b for b in check_full_model(which)]
        if bad:
            rc = 1
            print("  FAIL — {} mismatch(es):".format(len(bad)))
            for b in bad:
                print("    " + b)
        else:
            print("  PASS — every table, count and <option> field identical"
                  " at tolerance 0.0, and the full model is the robot + one"
                  " mocap target")
    if ran == 0:
        print("\nNOTHING RAN — both reference trees are missing.")
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
