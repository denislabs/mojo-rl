"""Can `reach_site_features` go through the COMPTIME model path at all?

This is a de-risking probe before any env logic, and it answers a question
nothing else has: every Phase 7 gate drove Jaco through the RUNTIME path
(`parse_xml_full` + `build_model_fields_from_flat`), which carries no
actuators. `Phyics3dEnv` is built on `ModelDefFromXML`, so stepping this task
requires the comptime route — and that route has its own element counter
(`parse_xml`), its own comptime capacity limits, and a history of compile
cliffs.

WHAT IS CHECKED, in the order that localises a failure:

  1. it COMPILES and instantiates — the comptime capacity question, which for
     this tree is a real one and cannot be answered by reading;
  2. every dimension `parse_xml` counted matches MuJoCo's, element for
     element. ⚠ `parse_xml` is NOT the parser the Phase 7 fixes landed in:
     `init_fields` uses `parse_xml_full`, `parse_xml` only counts. A fix in
     one is not a fix in the other, so the counts are asserted rather than
     assumed — see `feedback_physics3d_two_parser_paths`;
  3. forward kinematics through the comptime model reproduces MuJoCo's, which
     is what proves the model was BUILT correctly and not merely counted
     correctly;
  4. the actuators' control ranges — and this one FAILS, which is the most
     useful thing the probe found.

⚠⚠ TWO BLOCKERS FOUND, NEITHER FIXED HERE. Do not read a pass as "this model
can be stepped".

  * PER-ACTUATOR `ctrlrange` DOES NOT EXIST on the comptime path. It keeps a
    single model-wide `(CTRL_MIN, CTRL_MAX)`, read only from a root
    `<default><motor ctrlrange>` — and only from `<motor>`, while Jaco
    actuates with `<velocity>`. Measured: ours (-1, 1) against MuJoCo's
    ±0.6283 (3 joints), ±0.8378 (3 joints) and ±5.0 (3 fingers). All 9
    disagree. The arm would be clamped LOOSER than its real limit and the
    fingers to a FIFTH of theirs. Task #52, and the wider relative of #48.
  * ELLIPTIC CONE + `noslip_iterations=5` is this task's option block, and our
    `mj_solNoSlip` is pyramidal-only. The model def carries
    `allow_missing_noslip=True` to build at all, which is honest but means a
    rollout will not match MuJoCo under sliding friction. Task #53.

⚠ WHAT THIS DOES NOT CHECK: that a STEP matches MuJoCo — with both of the
above open it could not. That gate comes after they are closed.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulation_reach_def.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)

comptime DTYPE = DType.float64
comptime M = ReachSiteFeaturesModel
comptime NMESH_VERTS: Int = 60000
comptime FK_TOL: Float64 = 1e-9


def test_manipulation_reach_def_matches_mujoco() raises:
    print("=== reach_site_features through the COMPTIME model path ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("manipulation_ref")

    var mm = refmod.model("reach_site_features")
    var dat = mujoco.MjData(mm)

    # ── 2. the comptime counts, element for element ──────────────────────
    print("  counts   ours / MuJoCo")
    print("    nbody ", M.NBODY, "/", Int(py=mm.nbody))
    print("    njoint", M.NJOINT, "/", Int(py=mm.njnt))
    print("    nq    ", M.NQ, "/", Int(py=mm.nq))
    print("    nv    ", M.NV, "/", Int(py=mm.nv))
    print("    ngeom ", M.NGEOM, "/", Int(py=mm.ngeom))
    print("    nsite ", M.NSITE, "/", Int(py=mm.nsite))
    print("    nact  ", M.ACTION_DIM, "/", Int(py=mm.nu))
    assert_true(M.NBODY == Int(py=mm.nbody), "nbody disagrees with MuJoCo")
    assert_true(M.NJOINT == Int(py=mm.njnt), "njoint disagrees with MuJoCo")
    assert_true(M.NQ == Int(py=mm.nq), "nq disagrees with MuJoCo")
    assert_true(M.NV == Int(py=mm.nv), "nv disagrees with MuJoCo")
    assert_true(M.NGEOM == Int(py=mm.ngeom), "ngeom disagrees with MuJoCo")
    assert_true(M.NSITE == Int(py=mm.nsite), "nsite disagrees with MuJoCo")
    assert_true(
        M.ACTION_DIM == Int(py=mm.nu),
        "actuator count disagrees with MuJoCo — the comptime path exists to"
        " carry actuators, so getting this wrong defeats the purpose of it",
    )

    var mj_ts = Float64(py=mm.opt.timestep)
    print("    timestep", M.TIMESTEP, "/", mj_ts)
    assert_true(
        abs(M.TIMESTEP - mj_ts) < 1e-12,
        "timestep disagrees with MuJoCo — every step gate downstream would be"
        " comparing different amounts of elapsed time",
    )

    # ── 4. the actuator control ranges ──────────────────────────────────
    print("  CTRL_MIN / CTRL_MAX (model-wide):", M.CTRL_MIN, M.CTRL_MAX)
    var n_range_mismatch = 0
    for a in range(Int(py=mm.nu)):
        var mlo = Float64(py=mm.actuator_ctrlrange[a][0])
        var mhi = Float64(py=mm.actuator_ctrlrange[a][1])
        if abs(mlo - M.CTRL_MIN) > 1e-9 or abs(mhi - M.CTRL_MAX) > 1e-9:
            n_range_mismatch += 1
        if a < 9:
            print("    act", a, " MuJoCo [", mlo, ",", mhi, "]")
    print("  actuators whose range differs from the model-wide pair:",
          n_range_mismatch, "of", Int(py=mm.nu))
    print("")
    print("  ⚠⚠ KNOWN GAP, NOT GATED HERE: the comptime path stores ONE")
    print("     model-wide (CTRL_MIN, CTRL_MAX) read from <default><motor")
    print("     ctrlrange>. Jaco declares THREE different ranges on")
    print("     <velocity> actuators, so all 9 disagree. Task #52 — this")
    print("     must be fixed before any step or policy work.")
    print("")
    # ⚠ ASSERTED AS A KNOWN GAP, so that FIXING it fails this test and forces
    # the note above to be revisited. A test that merely printed the problem
    # would go stale silently the moment someone repaired the parser.
    assert_true(
        n_range_mismatch == Int(py=mm.nu) and abs(M.CTRL_MIN + 1.0) < 1e-12,
        "the model-wide ctrlrange is no longer (-1, 1) with every actuator"
        " disagreeing — per-actuator ranges may have landed. Re-check the"
        " banner above and turn this into a real per-actuator comparison",
    )

    # ── 1/3. it builds, and the built model reproduces MuJoCo's FK ───────
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, NMESH_VERTS, M.NPAIR,
    ]()
    M.init_fields[DTYPE, NMESH_VERTS](ctx, mf)
    var d = Data[
        DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]()

    for i in range(M.NQ):
        var qv = 0.11 * Float64(i + 1) - 0.4
        dat.qpos[i] = qv
        d.qpos.data[i] = Scalar[DTYPE](qv)
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](d, mf)

    var worst_fk = 0.0
    for b in range(M.NBODY):
        for k in range(3):
            var e = abs(
                Float64(d.xpos.data[b * 3 + k]) - Float64(py=dat.xpos[b][k])
            )
            if e > worst_fk:
                worst_fk = e
    print("  body FK: worst |d(xpos)| over", M.NBODY, "bodies:", worst_fk)
    assert_true(
        worst_fk <= FK_TOL,
        "the COMPTIME model builds a different kinematic tree from the one"
        " the runtime path builds and MuJoCo agrees with. Counting the"
        " elements correctly is not the same as placing them correctly",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
