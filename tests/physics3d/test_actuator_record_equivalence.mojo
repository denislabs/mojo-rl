"""Phase 1a — the `build_spec_fields` transcription gate.

    pixi run mojo run -I . tests/physics3d/test_actuator_record_equivalence.mojo

WHAT THIS IS NOW, AND WHAT IT WAS. It began as the DIFFERENTIAL gate that
licensed phase 1a: runtime `FlatModelDef` against the comptime
`ComptimeActData` (`_acd`), field by field, while both existed. That diff is
gone with `_acd`. What survives is its second leg, and the second leg was
never the redundant one:

    leg 1  record vs the old oracle     — retired with `_acd`
    leg 2  TENSOR vs record             — this file

⚠ LEG 2 IS NOT IMPLIED BY LEG 1. "The record agrees" says nothing about
whether `build_spec_fields` put it in the right COLUMN. 1a.2's negative
control is the proof: writing `kv` into `ACT_IDX_KP` left every record correct
and failed 4 of 5 models here. Packed layouts fail by transcription, and a
transcription error is invisible to a record diff.

⚠⚠ THE MODEL LIST IS THE GATE. Every field group in this file needed a model
ADDED before it discriminated — `gear` agreed on cartpole/quadruped/dog only
because 1.0 is the struct default, `force_*` because none of them declares a
`forcerange`, the tendon springs because fish is the only model in the tree
with one. **In a differential gate "0 mismatches" is the default outcome and
means nothing until a model that exercises the field is present.** The prints
below say which rows are vacuous rather than letting a 0 read as coverage.

Two fixtures are synthetic because NO model in the tree can distinguish the
case, and both were written after the case cost a real defect:

  · `ctrllimited_matrix` — `ctrlrange="0 0"` is MuJoCo's UNDEFINED marker, so
    such an actuator is UNLIMITED. `_fill_actuators` read it as a clamp and
    delivered ZERO FORCE. Every model here either omits `ctrlrange` or gives a
    real one, so the gate reported 0 on every run until this fixture existed.
  · `keyframe_stride` — `FlatModelDef.key_qvel` strides by NQ and the tensor
    by NV. so_arm100 is the only model with a `<keyframe>` and has
    nq == nv == 6, so the two expressions are the SAME NUMBER; and one key is
    not enough either, since the strides differ by `k * (nq - nv)`, zero at
    k = 0. Free joint + hinge (nq 8, nv 7) with TWO keys.
"""

from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import (
    parse_xml_full,
    parse_xml,
    ModelDefFromXML,
)
from mojo_rl.physics3d.parser.fields_build import build_spec_fields
from mojo_rl.physics3d.fields import SpecFields, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    TENDON_MAX_WRAPS,
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_KIND,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_LIMITED,
    ACT_IDX_FORCE_MIN,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_LIMITED,
    ACT_IDX_KP,
    ACT_IDX_KV,
    ACT_IDX_DYN_TAU,
    ACT_IDX_ACT_ADR,
    ACT_IDX_TRN_N,
    ACT_IDX_DOF_ADR,
    ACT_IDX_TRN_QADR_0,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_COEF_0,
    MODEL_ACT_TENDON_SIZE,
    ACTTEN_IDX_STIFFNESS,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_TRN_N,
    ACTTEN_IDX_TRN_QADR_0,
    ACTTEN_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_COEF_0,
    POSE_IDX_QPOS0_NQ,
    POSE_IDX_FREE_JOINT_QPOS_ADR,
    KEY_META_SIZE,
    KEY_IDX_TIME,
    KEY_IDX_NQPOS,
    KEY_IDX_NQVEL,
    KEY_IDX_NCTRL,
    JLIM_SIZE,
    JLIM_IDX_LIMITED,
    JLIM_IDX_RANGE_MIN,
    JLIM_IDX_RANGE_MAX,
)

from mojo_rl.envs.dm_control.cartpole.cartpole_xml import DMCartpole1Model
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.fish.fish_xml import DMFishSwimModel
from mojo_rl.envs.dm_control.finger.finger_xml import DMFingerSpinModel
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.robots.so_arm100_xml import SoArm100Model


# ═══ the `ctrlrange="0 0"` fixture — added AFTER it cost a bug ═══════════════
#
# Four actuators, three of them unlimited for three DIFFERENT reasons (no
# range at all, an explicit `ctrllimited="false"` over a real range, and the
# degenerate `"0 0"`), because they take different branches of the resolver.
# Mirrors `test_ctrllimited_vs_mujoco`'s `ctrllimited_matrix`, trimmed;
# duplicated rather than imported because a test module is not an importable
# package here. If one changes, the other must.
comptime CTRLLIM_XML = String(
    """<mujoco model="ctrllimited_matrix_gate">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="b0" pos="0 0 0"><joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b1" pos="0 1 0"><joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b2" pos="0 2 0"><joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b3" pos="0 3 0"><joint name="j3" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0"/>
    <motor name="a1" joint="j1" ctrlrange="-2 3"/>
    <motor name="a2" joint="j2" ctrlrange="-1 1" ctrllimited="false"/>
    <motor name="a3" joint="j3" ctrlrange="0 0"/>
  </actuator>
</mujoco>"""
)

comptime _clpm = parse_xml(CTRLLIM_XML)
comptime CtrlLimModel = ModelDefFromXML[
    xml=CTRLLIM_XML,
    nbody=_clpm.NBODY, njoint=_clpm.NJOINT, nq=_clpm.NQ, nv=_clpm.NV,
    ngeom=_clpm.NGEOM, nact=_clpm.NACT, ntex=_clpm.NTEX, nmat=_clpm.NMAT,
    nlight=_clpm.NLIGHT, ncam=_clpm.NCAM, nsite=_clpm.NSITE, neq=_clpm.NEQ,
    nexclude=_clpm.NEXCLUDE, npair=_clpm.NPAIR, max_tendon=_clpm.NTENDON,
    max_condim=_clpm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_clpm.TIMESTEP, noslip_iter=_clpm.NOSLIP_ITER,
]


# ═══ the `nq != nv` keyframe fixture — the STRIDE the tree cannot test ═══════
comptime KEYSTRIDE_XML = String(
    """<mujoco model="keyframe_stride">
  <option timestep="0.002"/>
  <worldbody>
    <body name="root" pos="0 0 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass="1"/>
      <body name="arm" pos=".1 0 0">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .1 0 0" size=".02" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="a" joint="j" gear="2" ctrlrange="-1 1"/></actuator>
  <keyframe>
    <key name="k0" time="0.25"
         qpos="0.1 0.2 0.3 0.7071068 0 0.7071068 0 0.4"
         qvel="1 2 3 4 5 6 7" ctrl="0.5"/>
    <key name="k1" time="0.75"
         qpos="-0.1 -0.2 -0.3 1 0 0 0 -0.4"
         qvel="11 12 13 14 15 16 17" ctrl="-0.5"/>
  </keyframe>
</mujoco>"""
)

comptime _kspm = parse_xml(KEYSTRIDE_XML)
comptime KeyStrideModel = ModelDefFromXML[
    xml=KEYSTRIDE_XML,
    nbody=_kspm.NBODY, njoint=_kspm.NJOINT, nq=_kspm.NQ, nv=_kspm.NV,
    ngeom=_kspm.NGEOM, nact=_kspm.NACT, ntex=_kspm.NTEX, nmat=_kspm.NMAT,
    nlight=_kspm.NLIGHT, ncam=_kspm.NCAM, nsite=_kspm.NSITE, neq=_kspm.NEQ,
    nexclude=_kspm.NEXCLUDE, npair=_kspm.NPAIR, max_tendon=_kspm.NTENDON,
    max_condim=_kspm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_kspm.TIMESTEP, noslip_iter=_kspm.NOSLIP_ITER,
    nkey = 2,
]


def _diff[
    NACT: Int, NTEN: Int, NQ: Int, NV: Int, NKEY: Int, NJOINT: Int
](name: String, xml: String) raises:
    """Every `SpecFields` slot against the `FlatModelDef` field it was built
    from, plus a non-vacuity report for each group."""
    var fmd = parse_xml_full(xml)
    var sf = SpecFields[DType.float64, Dims[nact=NACT, nten=NTEN, nq=NQ, nv=NV, nkey=NKEY, njoint=NJOINT]]()
    build_spec_fields[DType.float64](
        fmd, sf
    )
    print("--- ", name, " ---")

    # ── actuators ────────────────────────────────────────────────────────
    var n_act = 0
    var n_wrap = 0
    var first = String("")
    for i in range(NACT):
        var a = fmd.actuators[i]
        var o = i * MODEL_ACTUATOR_SIZE
        var slots = List[Int]()
        var wants = List[Float64]()
        var names = List[String]()

        @parameter
        def col(s: Int, w: Float64, n: String):
            slots.append(s)
            wants.append(w)
            names.append(n)

        col(ACT_IDX_KIND, Float64(a.kind), "kind")
        col(ACT_IDX_GEAR, a.gear, "gear")
        col(ACT_IDX_CTRL_MIN, a.ctrl_min, "ctrl_min")
        col(ACT_IDX_CTRL_MAX, a.ctrl_max, "ctrl_max")
        col(ACT_IDX_CTRL_LIMITED,
            1.0 if a.is_ctrl_limited else 0.0, "ctrl_limited")
        col(ACT_IDX_FORCE_MIN, a.force_min, "force_min")
        col(ACT_IDX_FORCE_MAX, a.force_max, "force_max")
        col(ACT_IDX_FORCE_LIMITED,
            1.0 if a.force_limited else 0.0, "force_limited")
        col(ACT_IDX_KP, a.kp, "kp")
        col(ACT_IDX_KV, a.kv, "kv")
        col(ACT_IDX_DYN_TAU, a.dyn_tau, "dyn_tau")
        col(ACT_IDX_ACT_ADR, Float64(a.act_adr), "act_adr")
        col(ACT_IDX_TRN_N, Float64(a.trn_n), "trn_n")
        col(ACT_IDX_DOF_ADR, Float64(a.dof_adr), "dof_adr")
        for s in range(len(slots)):
            if sf.actuators.data[o + slots[s]] != Scalar[DType.float64](
                wants[s]
            ):
                n_act += 1
                if first.byte_length() == 0:
                    first = String(
                        names[s], "[", i, "] tensor ",
                        Float64(sf.actuators.data[o + slots[s]]),
                        " vs record ", wants[s],
                    )
        # ⚠ THE WRAP STRIDES DIFFER. `FlatModelDef.motor_trn_*` is one flat
        # array strided by TENDON_MAX_WRAPS; the tensor puts the triples
        # INSIDE each actuator's record at `ACT_IDX_TRN_*_0 + k`.
        for k in range(a.trn_n):
            var fb = i * TENDON_MAX_WRAPS + k
            if (
                sf.actuators.data[o + ACT_IDX_TRN_QADR_0 + k]
                != Scalar[DType.float64](Float64(fmd.motor_trn_qadr[fb]))
                or sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k]
                != Scalar[DType.float64](Float64(fmd.motor_trn_dadr[fb]))
                or sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
                != Scalar[DType.float64](fmd.motor_trn_coef[fb])
            ):
                n_wrap += 1

    # ── act_tendons ──────────────────────────────────────────────────────
    var n_ten = 0
    var nt = len(fmd.tendons)
    var spring_seen = False
    for t in range(nt):
        var td = fmd.tendons[t]
        var o = t * MODEL_ACT_TENDON_SIZE
        if td.stiffness != 0.0:
            spring_seen = True
        var n = td.num_joints
        if n > TENDON_MAX_WRAPS:
            n = TENDON_MAX_WRAPS
        if (
            sf.act_tendons.data[o + ACTTEN_IDX_STIFFNESS]
            != Scalar[DType.float64](td.stiffness)
            or sf.act_tendons.data[o + ACTTEN_IDX_SPRING_LO]
            != Scalar[DType.float64](td.spring_lo)
            or sf.act_tendons.data[o + ACTTEN_IDX_SPRING_HI]
            != Scalar[DType.float64](td.spring_hi)
            or sf.act_tendons.data[o + ACTTEN_IDX_TRN_N]
            != Scalar[DType.float64](Float64(n))
        ):
            n_ten += 1
        for k in range(n):
            if sf.act_tendons.data[
                o + ACTTEN_IDX_TRN_COEF_0 + k
            ] != Scalar[DType.float64](td.coefs[k]):
                n_ten += 1
            # ⚠ qadr/dadr are a CUMULATIVE SUM over the joint list, which
            # `TendonData` does not carry — it stores joint IDS. Recompute it
            # the way `build_spec_fields` does, or this compares the tensor
            # against nothing.
            var jid = td.joint_ids[k]
            if jid < 0 or jid >= len(fmd.joints):
                continue
            var q = 0
            var dd = 0
            for j in range(jid):
                q += fmd.joints[j].nq
                dd += fmd.joints[j].nv
            if (
                sf.act_tendons.data[o + ACTTEN_IDX_TRN_QADR_0 + k]
                != Scalar[DType.float64](Float64(q))
                or sf.act_tendons.data[o + ACTTEN_IDX_TRN_DADR_0 + k]
                != Scalar[DType.float64](Float64(dd))
            ):
                n_ten += 1

    # ── reference pose + keyframes ───────────────────────────────────────
    var n_pose = 0
    if (
        Int(sf.pose_meta.data[POSE_IDX_QPOS0_NQ]) != fmd.qpos0_nq
        or Int(sf.pose_meta.data[POSE_IDX_FREE_JOINT_QPOS_ADR])
        != fmd.free_joint_qpos_adr
    ):
        n_pose += 1
    var q_nonzero = False
    for i in range(fmd.qpos0_nq):
        if i < NQ and i < len(fmd.qpos0):
            if sf.qpos0.data[i] != Scalar[DType.float64](fmd.qpos0[i]):
                n_pose += 1
            if fmd.qpos0[i] != 0.0:
                q_nonzero = True
    var n_key = 0
    for k in range(fmd.nkey):
        var o = k * KEY_META_SIZE
        if (
            sf.key_meta.data[o + KEY_IDX_TIME]
            != Scalar[DType.float64](fmd.key_time[k])
            or Int(sf.key_meta.data[o + KEY_IDX_NQPOS]) != fmd.key_nqpos[k]
            or Int(sf.key_meta.data[o + KEY_IDX_NQVEL]) != fmd.key_nqvel[k]
            or Int(sf.key_meta.data[o + KEY_IDX_NCTRL]) != fmd.key_nctrl[k]
        ):
            n_key += 1
        for i in range(NQ):
            if sf.key_qpos.data[k * NQ + i] != Scalar[DType.float64](
                fmd.key_qpos[k * NQ + i]
            ):
                n_key += 1
        # ⚠⚠ `FlatModelDef.key_qvel` STRIDES BY NQ — one allocation shape for
        # both key arrays — while the tensor is honestly [NKEY, NV]. On every
        # model in the tree with a `<keyframe>` those are the same number; see
        # `keyframe_stride`, which is here because of it.
        for i in range(NV):
            if sf.key_qvel.data[k * NV + i] != Scalar[DType.float64](
                fmd.key_qvel[k * NQ + i]
            ):
                n_key += 1
        for i in range(NACT):
            if sf.key_ctrl.data[k * NACT + i] != Scalar[DType.float64](
                fmd.key_ctrl[k * NACT + i]
            ):
                n_key += 1

    # ── joint limits ─────────────────────────────────────────────────────
    var n_jl = 0
    var n_limited = 0
    for j in range(NJOINT):
        if j >= len(fmd.joints):
            break
        var jd = fmd.joints[j]
        var o = j * JLIM_SIZE
        if jd.is_limited:
            n_limited += 1
        if (
            (sf.joint_limits.data[o + JLIM_IDX_LIMITED] != 0) != jd.is_limited
            or sf.joint_limits.data[o + JLIM_IDX_RANGE_MIN]
            != Scalar[DType.float64](jd.range_min)
            or sf.joint_limits.data[o + JLIM_IDX_RANGE_MAX]
            != Scalar[DType.float64](jd.range_max)
        ):
            n_jl += 1

    # ── the non-vacuity report ───────────────────────────────────────────
    #
    # ⚠ A ZERO ABOVE AND "NOTHING WAS TESTED" LOOK IDENTICAL. Say which.
    var n_wrapped = 0
    var n_limit_flags = 0
    var n_dyn = 0
    var n_flim = 0
    var g_flat = True
    var g0 = Float64(0)
    if NACT > 0:
        g0 = fmd.actuators[0].gear
    for i in range(NACT):
        var a = fmd.actuators[i]
        if a.trn_n > 0:
            n_wrapped += 1
        if a.is_ctrl_limited:
            n_limit_flags += 1
        if a.dyn_tau != 0.0:
            n_dyn += 1
        if a.force_limited:
            n_flim += 1
        if a.gear != g0:
            g_flat = False
    print("    mismatches: actuators", n_act, " wraps", n_wrap,
          " tendons", n_ten, " pose", n_pose, " keys", n_key,
          " joint_limits", n_jl)
    print("    exercised:  nact", NACT, "(", n_wrapped, "transmitted,",
          n_limit_flags, "ctrl-limited,", n_dyn, "dyntype,", n_flim,
          "force-limited)  tendons", nt, "(spring:", spring_seen, ")",
          "  nkey", fmd.nkey, "  limited joints", n_limited, "of", NJOINT)
    if NACT > 1 and g_flat:
        print("  ⚠ all", NACT, "gears are", g0, "— `gear` is VACUOUS here")
    if n_dyn == 0:
        print("  ⚠ no dyntype — `dyn_tau`/`act_adr` are VACUOUS here")
    if n_flim == 0:
        print("  ⚠ no forcerange — `force_*` is VACUOUS here")
    if nt == 0:
        print("  ⚠ no tendons — `act_tendons` is VACUOUS here")
    elif not spring_seen:
        print("  ⚠ no tendon spring — the spring rows are VACUOUS here")
    if fmd.nkey == 0:
        print("  ⚠ no <keyframe> — the key rows are VACUOUS here")
    if not q_nonzero:
        print("  ⚠ qpos0 is ALL ZERO — that row is VACUOUS here")
    if n_limited == 0:
        print("  ⚠ no joint is limited — `joint_limits` is VACUOUS here")

    assert_true(n_act == 0,
        String(name, ": actuator slots disagree with the record in ", n_act,
               " — first: ", first))
    assert_true(n_wrap == 0,
        String(name, ": transmission wraps disagree in ", n_wrap))
    assert_true(n_ten == 0,
        String(name, ": act_tendon rows disagree in ", n_ten))
    assert_true(n_pose == 0,
        String(name, ": qpos0/pose_meta disagree in ", n_pose))
    assert_true(n_key == 0,
        String(name, ": keyframe rows disagree in ", n_key))
    assert_true(n_jl == 0,
        String(name, ": joint_limits disagree in ", n_jl))


def test_cartpole() raises:
    comptime M = DMCartpole1Model
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "cartpole (1 actuator, no <default> class)", M.xml_text()
    )


def test_ctrllimited_matrix() raises:
    """The `ctrlrange="0 0"` case the gate was blind to — see the fixture."""
    comptime M = CtrlLimModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        'ctrllimited matrix (the ctrlrange="0 0" case)', M.xml_text()
    )


def test_quadruped() raises:
    comptime M = DMQuadrupedWalkModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "quadruped (3 classes / 12 actuators, tendons + dyntype)",
        M.xml_text(),
    )


def test_dog() raises:
    comptime M = DMDogStandWalkModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "dog stand/walk (24 classes / 38 actuators — the sharp case)",
        M.xml_text(),
    )


def test_reach_forcerange() raises:
    """jaco — the only model with distinct `forcerange`s."""
    comptime M = ReachSiteFeaturesModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "jaco reach (3 distinct forceranges)", M.xml_text()
    )


def test_fish() raises:
    """fish — the only model in the tree with a tendon SPRING."""
    comptime M = DMFishSwimModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "fish swim (the ONLY model with tendon springs)", M.xml_text()
    )


def test_pose_finger() raises:
    """finger — joint `ref="-90"`, the deg->rad case."""
    comptime M = DMFingerSpinModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "finger spin (joint ref=-90 -> -pi/2)", M.xml_text()
    )


def test_pose_key_stride() raises:
    """The `nq != nv` + two-keys case, for the key_qvel STRIDE."""
    comptime M = KeyStrideModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "keyframe stride (free joint: nq 8 != nv 7, two keys)", M.xml_text()
    )


def test_pose_ant() raises:
    """ant — `<custom><numeric init_qpos>` overrides the joint refs."""
    comptime M = AntModel
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "ant (<custom> init_qpos)", M.xml_text()
    )


def test_pose_so_arm100() raises:
    """so_arm100 — the only model in the tree with a `<keyframe>`."""
    comptime M = SoArm100Model
    _diff[M.NACT, M.NTEN_F, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "so_arm100 (the ONLY model with a <keyframe>)", M.xml_text()
    )


def main() raises:
    print("Phase 1a — `build_spec_fields` transcription gate")
    print("see docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md §10.17")
    TestSuite.discover_tests[__functions_in_module()]().run()
