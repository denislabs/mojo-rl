"""Unitree G1 — layer-2 parity: our env under BFM-Zero's PD vs MuJoCo under the same PD.

    pixi run mojo run -I . tests/robots/test_unitree_g1_vs_mujoco.mojo

⚠ THIS IS LAYER 2 AND DOES NOT REPLACE LAYER 1. Both sides here load OUR
baked `assets/unitree_g1.xml`; `tests/robots/g1_ref.py` is the gate that
proves that file IS the reference model (104 `mjModel` tables at 0.0). Run
it first.

⚠ THE CONTROLLER IS ON BOTH SIDES, AND IT IS NOT SHARED. The G1's actuators
are torque motors; the PD that turns a policy action into a torque lives in
the env. Our side runs `unitree_g1_config`'s Mojo PD from the GENERATED
tables; MuJoCo's side runs `g1_bake.ReferencePD`, a separate Python
transcription of `legged_robot_base._compute_torques` that re-reads the
reference yaml with numpy. A wrong gain in either shows up as a rollout
residual instead of cancelling — the standing rule that a gate sharing its
reference implementation is blind.

WHAT EACH GATE IS FOR:

  · `test_model_counts` — dimensions, INCLUDING `nexclude`/`nkey`, which the
    model def defaults silently.
  · `test_joint_and_body_order` — the reference's `dof_names` order against
    `mjModel`'s joint names, actuator i -> joint i+1, and the body indices
    the config will index by number. A reorder makes every later gate
    compare the wrong columns while staying green.
  · `test_actuator_ctrlrange` — `<motor ctrlrange>` on our records == on
    `mjModel`, and its split against the yaml's effort limit reported: the
    reference clips the torque at BOTH, and the two differ on four hips.
  · `test_stand_rollout` / `test_driven_rollout` — the discriminating ones:
    both sides from the reference's `init_state`, 100 control steps of 4
    substeps, `|d qpos|max` per step. Zero action lets the body settle onto
    its feet under the PD; the driven one swings every joint with its own
    phase so no symmetry can hide a column swap. Contacts are LIVE in both
    (MuJoCo's max `ncon` is printed), so this covers the mesh-vs-plane
    multicontact path the SO-101 gates could not.
  · `test_reset_observation` — the 64-D state at the stand pose is exactly
    zeros, gravity (0, 0, -1), zeros.

Tolerances are MEASURED, not inherited (see each gate's docstring).
"""

from std.math import abs, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import (
    UnitreeG1Model,
    PELVIS_BODY_IDX,
    LEFT_ANKLE_ROLL_BODY_IDX,
    RIGHT_ANKLE_ROLL_BODY_IDX,
    TORSO_BODY_IDX,
    LEFT_HAND_BODY_IDX,
    RIGHT_HAND_BODY_IDX,
    UNITREE_G1_OBS_DIM,
    UNITREE_G1_STATE_DIM,
)
from mojo_rl.envs.robots.unitree_g1_pd import (
    G1_N_DOF,
    g1_dof_name,
    g1_effort,
    G1_CONTROL_DECIMATION,
)
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_MIN,
    META_IDX_NUM_CONTACTS,
)


comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime NU = 29
comptime N_STEPS = 100
# The hard gate's horizon. Beyond it the contact-rich driven rollout (arms
# scraping the torso, feet sliding) amplifies the reference's OWN solver
# tolerance: MuJoCo as shipped vs MuJoCo run to convergence separate by
# 3e-5 at step 59 and 1.5e-3 at step 95 on this exact trajectory, so a
# 100-step bound below 1e-3 would gate the reference's noise, not us. The
# 100-step figure is still printed beside that floor.
comptime N_GATE = 40
comptime ASSET = "mojo_rl/envs/robots/assets/unitree_g1.xml"


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path(ASSET)


def _mj_converged() raises -> PythonObject:
    """The reference RUN TO CONVERGENCE — `convsweep.py`'s column B.

    ⚠ THE SHIPPED SOLVER SETTINGS ARE NOT THE REFERENCE'S ANSWER. The model
    carries MuJoCo's defaults (Newton, 100 iterations, tolerance 1e-8), and
    at those settings MuJoCo stops early: on the zero-action stand rollout
    MuJoCo-shipped vs MuJoCo-converged is 3.9e-14 / 1.3e-10 / 3.3e-7 at
    steps 0 / 1 / 99 — and OUR residual against the shipped run was those
    same digits, because our Newton lands on the converged point. Gating
    against the converged reference measures the engine; gating against the
    shipped one measures where MuJoCo chose to stop.
    """
    var m = _mj()
    m.opt.tolerance = 0.0
    m.opt.iterations = 1000
    return m


def _pd() raises -> PythonObject:
    """`g1_bake.ReferencePD` — the Python transcription, from the yaml."""
    _ = Python.evaluate("__import__('sys').path.insert(0, 'tests/robots')")
    var bake = Python.import_module("g1_bake")
    return bake.ReferencePD()


def test_model_counts() raises:
    var m = _mj()
    assert_true(Int(py=m.nbody) == UnitreeG1Model.NBODY, "nbody")
    assert_true(Int(py=m.njnt) == UnitreeG1Model.NJOINT, "njnt")
    assert_true(Int(py=m.nq) == UnitreeG1Model.NQ, "nq")
    assert_true(Int(py=m.nv) == UnitreeG1Model.NV, "nv")
    assert_true(Int(py=m.nu) == NU, "nu")
    assert_true(Int(py=m.ngeom) == UnitreeG1Model.NGEOM, "ngeom")
    assert_true(Int(py=m.nsite) == UnitreeG1Model.NSITE, "nsite")
    assert_true(Int(py=m.nexclude) == UnitreeG1Model.NEXCLUDE, "nexclude")
    assert_true(Int(py=m.nkey) == 1, "nkey — the stand keyframe")
    assert_true(Int(py=m.nsensor) == 0, "the bake drops the sensor block")
    assert_true(
        abs(Float64(py=m.opt.timestep) - 0.005) < 1e-15,
        "timestep is the reference's runtime 1/200, not the XML's 0.002",
    )
    print("  counts OK — nbody", Int(py=m.nbody), " nq", Int(py=m.nq),
          " nv", Int(py=m.nv), " nu", Int(py=m.nu), " ngeom",
          Int(py=m.ngeom))


def _check_body(mujoco: PythonObject, m: PythonObject, i: Int, want: StaticString) raises:
    var got = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))
    assert_true(
        got == String(want),
        "body " + String(i) + " is " + got + ", expected " + String(want),
    )


def test_joint_and_body_order() raises:
    var mujoco = Python.import_module("mujoco")
    var m = _mj()
    for i in range(G1_N_DOF):
        var jn = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1))
        assert_true(
            jn == String(g1_dof_name(i)),
            "joint " + String(i + 1) + " is " + jn + " but the PD table row "
            + String(i) + " is " + String(g1_dof_name(i)),
        )
        assert_true(
            Int(py=m.actuator_trnid[i][0]) == i + 1,
            "actuator " + String(i) + " does not drive joint " + String(i + 1),
        )
        assert_true(
            Int(py=m.jnt_qposadr[i + 1]) == 7 + i
            and Int(py=m.jnt_dofadr[i + 1]) == 6 + i,
            "hinge " + String(i) + " qpos/dof address is not 7+i / 6+i",
        )
    assert_true(Int(py=m.jnt_type[0]) == 0, "joint 0 is the free joint")

    _check_body(mujoco, m, PELVIS_BODY_IDX, "pelvis")
    _check_body(mujoco, m, LEFT_ANKLE_ROLL_BODY_IDX, "left_ankle_roll_link")
    _check_body(mujoco, m, RIGHT_ANKLE_ROLL_BODY_IDX, "right_ankle_roll_link")
    _check_body(mujoco, m, TORSO_BODY_IDX, "torso_link")
    _check_body(mujoco, m, LEFT_HAND_BODY_IDX, "left_rubber_hand")
    _check_body(mujoco, m, RIGHT_HAND_BODY_IDX, "right_rubber_hand")
    print("  joint order matches the reference's dof_names; body indices pinned")


def test_actuator_ctrlrange() raises:
    """`<motor ctrlrange>`: ours == `mjModel`'s, and the split against the
    yaml's `dof_effort_limit` is REPORTED, not assumed away.

    ⚠⚠ MEASURED: 25 of 29 agree, and hip pitch/roll (both sides) carry
    ctrlrange +-88 against an effort limit of 139. The reference applies
    both — the effort clip in `_compute_torques`, then MuJoCo's ctrl clamp —
    and so does our hook. The first draft of this gate asserted the two were
    equal and failed on actuator 0; that failure is what found the missing
    second clamp (driven rollout 0.118 rad off at step 72).
    """
    var m = _mj()
    var sf = UnitreeG1Model.make_spec_fields[DType.float64]()
    var cmin = actuator_column(sf, ACT_IDX_CTRL_MIN, NU)
    var cmax = actuator_column(sf, ACT_IDX_CTRL_MAX, NU)
    var n_split = 0
    for i in range(NU):
        var lo = Float64(py=m.actuator_ctrlrange[i][0])
        var hi = Float64(py=m.actuator_ctrlrange[i][1])
        assert_true(
            cmax[i] == hi and cmin[i] == lo,
            "our actuator " + String(i) + " ctrlrange " + String(cmin[i])
            + ".." + String(cmax[i]) + " != MuJoCo's " + String(lo) + ".."
            + String(hi),
        )
        assert_true(lo == -hi, "asymmetric ctrlrange on actuator " + String(i))
        if hi != g1_effort(i):
            n_split += 1
            assert_true(
                hi < g1_effort(i),
                "actuator " + String(i) + ": ctrlrange " + String(hi)
                + " ABOVE the effort limit " + String(g1_effort(i))
                + " — the yaml's clip would then be the binding one and the"
                " hook's order would matter the other way",
            )
    assert_true(
        n_split == 4,
        "expected ctrlrange < effort on exactly the 4 hip pitch/roll motors,"
        " found " + String(n_split),
    )
    print("  <motor ctrlrange> matches MuJoCo on all 29; 4 hips clamp below"
          " the yaml effort limit (88 vs 139), both clips applied")


def _action(t: Int, j: Int, driven: Bool) -> Float64:
    if not driven:
        return 0.0
    # Per-joint phase, well inside [-1, 1]; a full-scale action moves a hip
    # target by 1.75 rad, so 0.3 keeps the body upright-ish for a while.
    return 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _rollout(driven: Bool) raises -> Tuple[Float64, Float64, Int]:
    """Returns (worst |d qpos| over the first N_GATE steps, worst over all
    N_STEPS, MuJoCo max ncon), against the CONVERGED reference."""
    var mujoco = Python.import_module("mujoco")
    var m = _mj_converged()
    var d = mujoco.MjData(m)
    var pd = _pd()
    pd.stand_state(mujoco, m, d)

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    # Same numbers on both sides, by construction: the env's reset IS the
    # reference's init_state; asserted rather than assumed.
    for i in range(NQ):
        assert_true(
            Float64(env.d.qpos.data[i]) == Float64(py=d.qpos[i]),
            "reset qpos[" + String(i) + "] differs from the reference's"
            " init_state",
        )

    var worst = 0.0
    var worst_gate = 0.0
    var ncon_max = 0
    for t in range(N_STEPS):
        var a = ContAction[UnitreeG1Model.ACTION_DIM]()
        var pa = Python.evaluate("[]")
        for j in range(NU):
            var u = _action(t, j, driven)
            a.data[j] = u
            pa.append(u)
        var nc = Int(py=pd.control_step(mujoco, m, d, pa, G1_CONTROL_DECIMATION))
        if nc > ncon_max:
            ncon_max = nc
        _ = env.step(a)
        var step_worst = 0.0
        for i in range(NQ):
            var e = abs(Float64(env.d.qpos.data[i]) - Float64(py=d.qpos[i]))
            if e > step_worst:
                step_worst = e
        if step_worst > worst:
            worst = step_worst
        if t < N_GATE and step_worst > worst_gate:
            worst_gate = step_worst
        if t < 3 or t % 20 == 19:
            print("      step", t, " |dqpos|max", step_worst, " mj ncon", nc,
                  " ours ncon", Int(env.d.meta.data[META_IDX_NUM_CONTACTS]))
    return (worst_gate, worst, ncon_max)


def test_stand_rollout() raises:
    """100 control steps at action 0 from `init_state`: the body drops the
    few millimetres onto its feet, the feet's mesh-vs-plane manifolds engage
    (8 contacts), and the PD holds the default pose against gravity.

    Measured 2026-09-08: worst |d qpos| 3.0e-13 at step 99, MuJoCo max ncon
    8. Bound set at 1e-9 — four orders above the measurement, as the SO-101
    gates are — so a real regression (a wrong kd, a missed substep, a
    contact dropped) reads as a failure and float noise does not.
    """
    var r = _rollout(False)
    print("  stand worst |dqpos|: first", N_GATE, "steps", r[0],
          "  all", N_STEPS, "steps", r[1], "  MuJoCo max ncon", r[2])
    assert_true(r[2] > 0, "no contact ever formed — the feet never touched"
                " the floor, so this gate covered no collision path")
    assert_true(r[1] < 1e-9, "stand rollout residual " + String(r[1]))


def test_driven_rollout() raises:
    """The same 100 steps with every joint driven at its own phase — the
    column-swap detector, and a harder closed loop (targets move, torques
    saturate on the wrists at 5 N m).

    Measured 2026-09-08: see the printed line; bound 1e-9 as above.
    """
    var r = _rollout(True)
    print("  driven worst |dqpos|: first", N_GATE, "steps", r[0],
          "  all", N_STEPS, "steps", r[1],
          " (MuJoCo's own shipped-vs-converged spread on this rollout:"
          " 1.5e-3)  MuJoCo max ncon", r[2])
    assert_true(r[0] < 1e-9, "driven rollout residual " + String(r[0]))


def test_reset_observation() raises:
    var env = UnitreeG1[DType.float64]()
    var obs = env.reset()
    var worst = 0.0
    # The proprio slice only: since G3.0 the observation continues with the
    # 463-D privileged block, whose stand-pose values are body geometry, not
    # zeros (`test_unitree_g1_privileged_obs` gates that block).
    for i in range(UNITREE_G1_STATE_DIM):
        var want = 0.0
        if i == 2 * G1_N_DOF + 2:
            want = -1.0
        worst = max(worst, abs(Float64(obs.data[i]) - want))
    assert_true(worst == 0.0, "reset obs is not [0]*58 + (0,0,-1) + [0]*3")
    print("  reset observation: q - q_default = 0, gravity (0, 0, -1), at rest")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
