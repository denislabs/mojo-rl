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

⚠⚠ A CORRECTION LIVES HERE, because this file is where the mistake was made.
The first version of this probe reported that per-actuator `ctrlrange` did not
exist on the comptime path. THAT WAS WRONG. It measured
`ModelDefFromXML.CTRL_MIN/CTRL_MAX` — a model-wide summary read from a ROOT
`<default><motor ctrlrange>` — and generalised from it. The per-actuator
arrays (`_acd.motor_ctrl_min/max`) were always there, always resolved through
element -> `class=` -> root default, always handled `<velocity>`, and
`apply_actions` always clamped with them. Section 4 below now measures THOSE,
and they match MuJoCo exactly. Measure the quantity that is used, not the one
that is easy to reach.

What was really wrong was one layer up — the env advertised a single scalar
action bound — and that is fixed and gated in
`test_per_actuator_action_bounds.mojo`.

⚠ ONE BLOCKER REMAINS, NOT FIXED HERE. Do not read a pass as "this model can
be stepped": ELLIPTIC CONE + `noslip_iterations=5` is this task's option
block, and our `mj_solNoSlip` is pyramidal-only. The model def carries
`allow_missing_noslip=True` to build at all, which is honest but means a
rollout will not match MuJoCo under sliding friction. Task #53.

⚠ WHAT THIS DOES NOT CHECK: that a STEP matches MuJoCo — with #53 open it
could not. That gate comes after it is closed.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulation_reach_def.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
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

    # ── 4. the actuator control ranges, PER ACTUATOR ────────────────────
    # ⚠ `M.default_ctrl_range()` is NOT what the clamp uses. It is a
    # single model-wide pair from `_xml_default_motor_ctrlrange`, which reads
    # only a ROOT `<default><motor ctrlrange>`; `apply_actions` clamps with
    # `_acd.motor_ctrl_min[i]` / `[i]`, resolved three ways (element, then
    # `class=`, then the root default) and for `<velocity>` too. Measuring the
    # model-wide pair and concluding the per-actuator ranges were missing is
    # exactly the mistake this comment exists to stop.
    var scalar_range = M.default_ctrl_range()
    print("  model-wide CTRL_MIN/CTRL_MAX (NOT the clamp):",
          scalar_range[0], scalar_range[1])
    var sf = M.make_spec_fields[DType.float64]()
    var cmin = actuator_column(sf, ACT_IDX_CTRL_MIN, M.nact)
    var cmax = actuator_column(sf, ACT_IDX_CTRL_MAX, M.nact)
    var worst_ctrl = 0.0
    for a in range(Int(py=mm.nu)):
        var mlo = Float64(py=mm.actuator_ctrlrange[a][0])
        var mhi = Float64(py=mm.actuator_ctrlrange[a][1])
        var e0 = abs(cmin[a] - mlo)
        var e1 = abs(cmax[a] - mhi)
        if e0 > worst_ctrl:
            worst_ctrl = e0
        if e1 > worst_ctrl:
            worst_ctrl = e1
        print("    act", a, " ours [", cmin[a], ",", cmax[a],
              "]  MuJoCo [", mlo, ",", mhi, "]")
    print("  worst |d ctrlrange| over", Int(py=mm.nu), "actuators:", worst_ctrl)
    assert_true(
        worst_ctrl < 1e-12,
        "per-actuator ctrlrange disagrees with MuJoCo. `apply_actions` clamps"
        " with these, so a wrong range silently rescales what a policy's"
        " action means",
    )

    # ── 1/3. it builds, and the built model reproduces MuJoCo's FK ───────
    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=M.NPAIR]]()
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
