#!/usr/bin/env python3
"""Does the transform gizmo sit on the part, and does its edit reach the file?

    pixi run python scripts/check_gizmo_vs_mujoco.py

`tests/physics3d/test_gizmo_math.mojo` gates the frame algebra against our own
forward kinematics and our own parser. That leaves two questions it CANNOT
answer, and both are the ones a user notices:

  A. IS THE GIZMO WHERE THE PART IS? The studio composes a world frame from
     the parser's records. MuJoCo composes `geom_xpos` / `geom_xmat` from its
     own. If those disagree the handle floats beside the shape — and every
     in-tree check would still pass, because both sides of it would be ours.

  B. IS THE EDITED DOCUMENT STILL A MODEL? Writing `quat=` means REMOVING the
     `euler=` / `axisangle=` / `xyaxes=` / `zaxis=` / `fromto=` the file used
     — MuJoCo refuses a tag carrying two orientation specifiers. A studio
     that got that wrong would show the rotation happily and write a file
     that will not load.

⚠⚠ THE SURVEY ARM SKIPS MESH GEOMS, AND SAYS HOW MANY. MuJoCo BAKES a mesh's
recentering (`mesh_pos`/`mesh_quat`) into `geom_pos`/`geom_quat` at compile
time, so its `geom_xpos` is not the frame the document states and a
comparison would be measuring that bake rather than the gizmo. Skipping is
correct; skipping SILENTLY would let a model contribute zero comparisons and
still read as a pass, which is why the count is printed and why a model that
compares nothing at all is reported as a failure.

⚠ THE EDITED FILE IS WRITTEN BESIDE THE ORIGINAL and removed afterwards.
MJCF asset paths are relative to the document, so a copy in /tmp cannot be
loaded without moving every mesh with it.
"""

import os
import subprocess
import sys

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    sys.exit("mujoco is not importable — run this under `pixi run python`")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPER = "tests/physics3d/dump_gizmo_frames.mojo"

# ⚠ THE SET IS CHOSEN FOR WHAT IT SPELLS, not for coverage of environments.
# ant and swimmer carry `fromto` capsules; cheetah and walker2d state
# orientations as `euler`; so_arm101 is meshes on rotated bodies (the model
# the outline bug was found on); humanoid mixes all three.
MODELS = [
    "mojo_rl/envs/ant/assets/ant.xml",
    "mojo_rl/envs/swimmer/assets/swimmer.xml",
    "mojo_rl/envs/half_cheetah/assets/half_cheetah.xml",
    "mojo_rl/envs/walker2d/assets/walker2d.xml",
    "mojo_rl/envs/hopper/assets/hopper.xml",
    "mojo_rl/envs/humanoid/assets/humanoid.xml",
    "mojo_rl/envs/reacher/assets/reacher.xml",
    "mojo_rl/envs/robots/assets/so_arm101.xml",
    "mojo_rl/envs/dm_control/assets/cheetah.xml",
    "mojo_rl/envs/dm_control/assets/finger.xml",
]

POS_TOL = 2e-6   # the gizmo matrix crosses the FFI as float32
QUAT_TOL = 2e-6

MESH = 5
PLANE = 0


def quat_gap(a, b):
    """Distance between two ROTATIONS. `q` and `-q` are the same one."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(min(np.abs(a - b).sum(), np.abs(a + b).sum()))


def mat_to_quat(m9):
    """MuJoCo's row-major 3x3 as (w, x, y, z)."""
    m = np.asarray(m9, dtype=float).reshape(3, 3)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, m.reshape(9))
    return q


def parse_dump(text):
    geoms, gtypes, bodies, fk = {}, {}, {}, {}
    qpos = None
    nmocap = 0
    edit = want = got = None
    wrote = None
    nedits = None
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "NMOCAP":
            nmocap = int(f[1])
        elif f[0] == "QPOS":
            qpos = np.array([float(x) for x in f[2:]])
        elif f[0] == "GEOM":
            geoms[int(f[1])] = (
                np.array([float(x) for x in f[3:6]]),
                np.array([float(x) for x in f[6:10]]),
            )
        elif f[0] == "GTYPE":
            gtypes[int(f[1])] = int(f[2])
        elif f[0] == "BODY":
            bodies[int(f[1])] = (
                np.array([float(x) for x in f[3:6]]),
                np.array([float(x) for x in f[6:10]]),
            )
        elif f[0] == "FK":
            fk[int(f[1])] = (
                np.array([float(x) for x in f[3:6]]),
                np.array([float(x) for x in f[6:10]]),
            )
        elif f[0] == "EDIT" and f[1] != "none":
            edit = (int(f[1]), f[2])
        elif f[0] == "WANT":
            want = (
                np.array([float(x) for x in f[1:4]]),
                np.array([float(x) for x in f[4:8]]),
            )
        elif f[0] == "GOT":
            got = (
                np.array([float(x) for x in f[1:4]]),
                np.array([float(x) for x in f[4:8]]),
            )
        elif f[0] == "NEDITS":
            nedits = (int(f[1]), int(f[2]))
        elif f[0] == "WROTE" and f[1] != "none":
            wrote = f[1]
    return (geoms, gtypes, bodies, fk, edit, want, got, nedits, wrote,
            qpos, nmocap)


def main():
    os.chdir(ROOT)

    exe = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "mrl_dump_gizmo_frames"
    )
    print("building the dumper once ...")
    build = subprocess.run(
        ["pixi", "run", "mojo", "build", "-I", ".", DUMPER, "-o", exe],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        print(build.stderr[-4000:])
        sys.exit("the dumper did not build")

    total_geoms = 0
    total_bodies = 0
    total_drifted = 0
    total_mocap = 0
    failures = 0
    edited_models = 0

    for path in MODELS:
        if not os.path.exists(path):
            print(f"\n{path}\n  SKIP — not in this tree")
            continue
        out_path = path + ".gizmocheck.xml"
        print(f"\n{path}")
        run = subprocess.run([exe, path, out_path], capture_output=True,
                             text=True)
        if run.returncode != 0:
            print(run.stderr[-2000:])
            print("  FAIL — the dumper raised")
            failures += 1
            continue
        (geoms, gtypes, bodies, fk, edit, want, got, nedits, wrote,
         qpos, nmocap) = parse_dump(run.stdout)
        # ⚠⚠ A MOCAP BODY IS THE CASE `forward_kinematics` SKIPS. Its world
        # pose is an external input, so a tool that is not an env leaves it
        # at (0, 0, 0) unless something seeds it — so_arm101's `target`
        # sphere read 0.25 m off before `reset_mocap_from_model` existed, and
        # the survey arm below is what says so. Counted so a run in which no
        # model had one cannot pass as coverage of it.
        total_mocap += nmocap
        if nmocap:
            print(f"  {nmocap} mocap body(ies) — seeded from the XML frame")

        try:
            m = mujoco.MjModel.from_xml_path(path)
        except Exception as exc:
            print(f"  SKIP — MuJoCo will not load the ORIGINAL: {exc}")
            if wrote and os.path.exists(wrote):
                os.remove(wrote)
            continue
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        # ⚠⚠ MuJoCo IS PUT INTO **OUR** POSE, NOT ITS OWN. `qpos0` is one
        # thing the two sides deliberately disagree about: this tree honours
        # `<custom><numeric name="init_qpos">` as the reference pose
        # (Gymnasium's convention) and MuJoCo derives `qpos0` from the body
        # frames instead — ant's torso is at z=0.55 here and z=0.75 there.
        # Comparing at each side's own reference would measure THAT, and
        # report a 0.2 m error in a gizmo that is exactly right.
        if qpos is not None and len(qpos) == m.nq:
            d.qpos[:] = qpos
        elif qpos is not None:
            print(f"  FAIL — nq disagrees: ours {len(qpos)}, MuJoCo {m.nq}")
            failures += 1
        mujoco.mj_forward(m, d)

        # ── arm A: the gizmo sits where MuJoCo puts the part ──────────────
        cmp_g = skip_mesh = 0
        worst_gp = worst_gq = 0.0
        for gi, (p, q) in sorted(geoms.items()):
            if gtypes.get(gi) in (MESH, PLANE):
                skip_mesh += 1
                continue
            if gi >= m.ngeom:
                continue
            cmp_g += 1
            dp = float(np.abs(d.geom_xpos[gi] - p).max())
            dq = quat_gap(mat_to_quat(d.geom_xmat[gi]), q)
            worst_gp = max(worst_gp, dp)
            worst_gq = max(worst_gq, dq)
        # ⚠⚠ BODIES ARE COMPARED AGAINST **FK**, NOT AGAINST THE EDIT FRAME.
        # The edit frame is the body's `pos=`/`quat=` composed onto its
        # parent — the frame the gizmo edits — and a body whose joint has
        # moved is NOT there. ant's `init_qpos` parks every hinge at ±1 rad,
        # so its edit frames sit 0.2 m and 0.8 quaternion-distance from where
        # the bodies are, which is `frame_drift` doing its job. Checking the
        # edit frame here would report that as a gizmo error.
        cmp_b = 0
        worst_bp = worst_bq = 0.0
        # ...and where our own FK says the body IS at its edit frame — no
        # joint has moved it — MuJoCo must agree with the edit frame too.
        # That is the arm that ties the gizmo's placement on a BODY to an
        # external reference rather than only to ours.
        cmp_still = 0
        worst_sp = worst_sq = 0.0
        for bi, (p, q) in sorted(bodies.items()):
            if bi >= m.nbody or bi not in fk:
                continue
            fp, fq = fk[bi]
            cmp_b += 1
            worst_bp = max(worst_bp, float(np.abs(d.xpos[bi] - fp).max()))
            worst_bq = max(worst_bq, quat_gap(d.xquat[bi], fq))
            if float(np.abs(fp - p).max()) < 1e-12 and quat_gap(fq, q) < 1e-12:
                cmp_still += 1
                worst_sp = max(worst_sp,
                               float(np.abs(d.xpos[bi] - p).max()))
                worst_sq = max(worst_sq, quat_gap(d.xquat[bi], q))
        total_geoms += cmp_g
        total_bodies += cmp_b

        # ⚠ VACUITY IS THE DEFAULT FAILURE. "0 mismatches" and "nothing
        # compared" print the same way unless the count is beside it.
        moved_bodies = cmp_b - cmp_still
        total_drifted += moved_bodies
        print(f"  survey: {cmp_g} geoms ({skip_mesh} mesh/plane skipped),"
              f" {cmp_b} bodies ({cmp_still} at their edit frame,"
              f" {moved_bodies} moved by a joint)")
        if cmp_g == 0 and cmp_b == 0:
            print("  FAIL — this model compared NOTHING")
            failures += 1
        else:
            print(f"    geom       worst dpos {worst_gp:.3e}"
                  f"  dquat {worst_gq:.3e}")
            print(f"    body (FK)  worst dpos {worst_bp:.3e}"
                  f"  dquat {worst_bq:.3e}")
            print(f"    body (giz) worst dpos {worst_sp:.3e}"
                  f"  dquat {worst_sq:.3e}   over {cmp_still} bodies")
            if max(worst_gp, worst_bp, worst_sp) > 1e-9 \
                    or max(worst_gq, worst_bq, worst_sq) > 1e-9:
                print("  FAIL — the gizmo would not sit on the part")
                failures += 1

        # ── arm B: the edit reaches the file, and the file still loads ────
        if edit is None or wrote is None:
            print("  (no editable geom in this model — arm B skipped)")
            continue
        edited_models += 1
        gi, gname = edit
        print(f"  edit:   geom {gi} '{gname}'"
              f"  ({nedits[0]} pos + {nedits[1]} quat edits)")
        try:
            m2 = mujoco.MjModel.from_xml_path(wrote)
        except Exception as exc:
            print(f"  FAIL — MuJoCo REFUSES the edited document: {exc}")
            failures += 1
            os.remove(wrote)
            continue
        d2 = mujoco.MjData(m2)
        mujoco.mj_resetData(m2, d2)
        if qpos is not None and len(qpos) == m2.nq:
            d2.qpos[:] = qpos
        mujoco.mj_forward(m2, d2)
        # ⚠ AGAINST `WANT`, NOT AGAINST `GOT`. `GOT` is the studio's own
        # answer to the same question; comparing the two would be the studio
        # agreeing with itself. `WANT` is the pose the gizmo was DRAGGED to.
        dp = float(np.abs(d2.geom_xpos[gi] - want[0]).max())
        dq = quat_gap(mat_to_quat(d2.geom_xmat[gi]), want[1])
        print(f"    after the edit: dpos {dp:.3e}  dquat {dq:.3e}")
        if dp > POS_TOL or dq > QUAT_TOL:
            print("  FAIL — MuJoCo does not put the geom where the gizmo"
                  " left it")
            failures += 1
        # ⚠ AND THE EDIT MUST HAVE MOVED IT. A document write that did
        # nothing at all would land at the ORIGINAL pose, and the arm above
        # would only catch that because `want` differs from it — so say so
        # explicitly rather than relying on the deltas being large.
        moved = float(np.abs(d2.geom_xpos[gi] - d.geom_xpos[gi]).max())
        if moved < 1e-4:
            print(f"  FAIL — the geom did not move at all ({moved:.3e})"
                  " — the document write is a no-op")
            failures += 1
        os.remove(wrote)

    print("\n" + "=" * 70)
    print(f"compared {total_geoms} geom frames and {total_bodies} body frames"
          f" against MuJoCo")
    print(f"{edited_models} models had a gizmo edit written and reloaded")
    # ⚠ NON-VACUITY FOR THE DRIFT SPLIT. If no body in the whole run had been
    # moved off its edit frame by a joint, the two body arms would be the
    # same arm and the distinction they exist to police would be untested.
    print(f"{total_drifted} bodies were displaced by a joint (the case that"
          f" makes the two body arms different)")
    if total_drifted == 0:
        print("FAIL — no model exercised a body away from its edit frame")
        sys.exit(1)
    print(f"{total_mocap} mocap bodies were seeded (the case forward"
          f" kinematics skips)")
    if total_mocap == 0:
        print("FAIL — no model in the set has a mocap body, so the one case"
              " FK deliberately skips went untested")
        sys.exit(1)
    if total_geoms == 0 or edited_models == 0:
        print("FAIL — the run compared nothing; this is not a pass")
        sys.exit(1)
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
