"""dm_control `fish` — MODEL + DYNAMICS parity against the reference XML.

fish is the first domain in the port whose actuators are POSITION SERVOS
rather than torque motors, and the first with a tendon in the force path. It
exercises three pieces of engine work that landed with it:

  * `<position>` actuators — `force = kp*(ctrl - length) - kv*velocity`,
    recomputed EVERY PHYSICS SUBSTEP because `length` reads `qpos`.
    `test_fish_servo_force_changes_within_a_control_step` is the gate on that
    last part: it fails if the force is ever hoisted back out of the loop.
  * a TENDON transmission — `fins_flap` drives a fixed tendon, so one
    actuator's force lands on two DOFs weighted by the tendon coefficients.
  * a TENDON SPRING — `<fixed name="fins_sym" stiffness="1e-4">`, MuJoCo's
    deadband spring on `tendon_lengthspring`.

BUG 25, found here: `merge_mjcf` did not accumulate `<tendon>` at all, so the
whole section was dropped on the way in and both tendon features silently did
not exist. It was latent because the only other merged model with tendons is
point_mass, which deliberately rewrites its two identity-coefficient tendons as
joint motors. `test_fish_tendons_match_mujoco` is the gate.

Run:
    pixi run mojo run -I . tests/dm_control/test_fish_vs_dm_control.mojo
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.python import Python, PythonObject
from std.math import abs, sin, sqrt
from max.gpu.host import DeviceContext

from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_POS_Y,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_GRAVITY_Z,
)
from mojo_rl.physics3d.parser.flat_model import ACT_KIND_POSITION

from mojo_rl.envs.dm_control.rewards import tolerance
from mojo_rl.envs.dm_control.fish.fish_config import (
    DMFishUprightConfig,
    DMFishSwimConfig,
    SWIM_RADII,
)
from mojo_rl.envs.dm_control.fish.fish_xml import (
    DMFishUprightModel,
    DMFishSwimModel,
    TORSO_BODY_IDX,
    TARGET_BODY_IDX,
    MOUTH_GEOM_IDX,
    TARGET_GEOM_IDX,
    N_ROOT_QPOS,
)

# The comptime tendon tables are strided by this, not by a literal.
from mojo_rl.physics3d.parser.xml_parser import MAX_COMPTIME_TENDON_WRAPS

comptime DTYPE = DType.float64
comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/fish.xml"
)

comptime M = DMFishSwimModel  # same model as upright; they differ only in obs
comptime MODEL_TOL: Float64 = 1e-13

comptime FRAME_SKIP_F: Int = 10
comptime NQ_F: Int = 14
comptime NV_F: Int = 13
comptime NACT_F: Int = 5

comptime STATE_TOL: Float64 = 1e-8
comptime OBS_TOL: Float64 = 1e-8
comptime REWARD_TOL: Float64 = 1e-10

# Every check below resolves the reference side BY NAME rather than by index.
#
# That was written when our geom order was XML TEXT order and the reference's
# was body order; as of the element-order fix (2026-08-03) `full_parser` groups
# by body id and the two agree, so the names are no longer papering over a
# divergence. Kept anyway: name resolution is the stronger form — it survives a
# model gaining a geom, which an index list does not — and it is why this file
# needed no change when the order moved, while three sibling files did.
def _geom_names() -> List[String]:
    return [
        String("ground"), String("eye"), String("mouth"),
        String("lower_mouth"), String("torso"), String("back_fin"),
        String("torso_massive"), String("tail1"), String("tail2"),
        String("finright"), String("finleft"), String("target"),
    ]


def _ref() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path(String(REF_XML))


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= MODEL_TOL * (1.0 + abs(b))


# ── model ────────────────────────────────────────────────────────────────────


def test_fish_counts() raises:
    """Counts match, with the mocap target the single accounted-for extra."""
    var mj = _ref()
    print(
        "  ours  NBODY", M.NBODY, " NJOINT", M.NJOINT, " NQ", M.NQ,
        " NV", M.NV, " NGEOM", M.NGEOM, " NSITE", M.NSITE,
        " NACT", M.ACTION_DIM,
    )
    print(
        "  mjcf  nbody", Int(py=mj.nbody), " njnt", Int(py=mj.njnt),
        " nq", Int(py=mj.nq), " nv", Int(py=mj.nv),
        " ngeom", Int(py=mj.ngeom), " nsite", Int(py=mj.nsite),
        " nu", Int(py=mj.nu),
    )
    # The target moves from a static worldbody geom onto a mocap BODY: one
    # extra body, the same geom count.
    assert_equal(
        M.NBODY, Int(py=mj.nbody) + 1,
        "NBODY should be the reference's + 1 (the mocap target body)",
    )
    assert_equal(M.NJOINT, Int(py=mj.njnt), "joint count")
    assert_equal(M.NQ, Int(py=mj.nq), "nq — the mocap body must add no DOF")
    assert_equal(M.NV, Int(py=mj.nv), "nv — the mocap body must add no DOF")
    assert_equal(M.NGEOM, Int(py=mj.ngeom), "geom count")
    assert_equal(M.NSITE, Int(py=mj.nsite), "site count")
    assert_equal(M.ACTION_DIM, Int(py=mj.nu), "actuator count")
    assert_equal(DMFishUprightModel.OBS_DIM, 21, "upright observation width")
    assert_equal(DMFishSwimModel.OBS_DIM, 24, "swim observation width")
    # `physics.velocity()` is the WHOLE qvel, free root included — 13 of the
    # 24 swim entries. Getting this wrong would still produce a plausible obs.
    assert_equal(
        DMFishSwimModel.OBS_DIM - DMFishUprightModel.OBS_DIM, 3,
        "swim adds exactly mouth_to_target (3) over upright",
    )


def test_fish_actuators_are_position_servos() raises:
    """THE G3 GATE: five `<position>` servos, one of them tendon-driven.

    Compared against MuJoCo's compiled `gainprm`/`biasprm` rather than against
    the `kp=` literals, so the XML, the parser and the compiler all have to
    agree. `<position>` compiles to gaintype=fixed + biastype=affine with
    `gainprm = [kp, 0, 0]` and `biasprm = [0, -kp, -kv]`, which is what makes
    `force = kp*(ctrl - length) - kv*velocity` the right law.

    ⚠ `M._acd` is a COMPTIME struct, and indexing its `InlineArray` fields with
    a RUNTIME loop variable materializes garbage here (an `assert` on
    `M._acd.motor_kp[i]` read 6.4e-314 while a `print` of the same expression
    read 0.0005). Every field is copied out through a `comptime for` — a
    compile-time index — before any of it is compared. `apply_actions` is not
    affected: inside the struct, `Self._acd` is a parameter access rather than
    a materialization, which the exact rollout parity confirms.
    """
    var mj = _ref()
    var gainprm = mj.actuator_gainprm.tolist()
    var biasprm = mj.actuator_biasprm.tolist()
    var gear = mj.actuator_gear.tolist()
    var cr = mj.actuator_ctrlrange.tolist()
    var trntype = mj.actuator_trntype.tolist()

    var kind = List[Int]()
    var kp = List[Float64]()
    var kv = List[Float64]()
    var gears = List[Float64]()
    var cmin = List[Float64]()
    var cmax = List[Float64]()
    var trn_n = List[Int]()

    comptime for a in range(M.ACTION_DIM):
        kind.append(M._acd.motor_kind[a])
        kp.append(M._acd.motor_kp[a])
        kv.append(M._acd.motor_kv[a])
        gears.append(M._acd.motor_gears[a])
        cmin.append(M._acd.motor_ctrl_min[a])
        cmax.append(M._acd.motor_ctrl_max[a])
        trn_n.append(M._acd.motor_trn_n[a])

    var saw_tendon = False
    for i in range(M.ACTION_DIM):
        var kp_mj = Float64(py=gainprm[i][0])
        var kv_mj = -Float64(py=biasprm[i][2])
        print(
            "   act", i, " kind", kind[i],
            " kp", kp[i], "(mj", kp_mj, ")",
            " kv", kv[i], "(mj", kv_mj, ")",
            " n", trn_n[i], " trntype(mj)", Int(py=trntype[i]),
        )
        assert_equal(
            kind[i], ACT_KIND_POSITION,
            "every fish actuator is a <position> servo",
        )
        assert_true(_close(kp[i], kp_mj), "actuator kp (gainprm)")
        assert_true(_close(kv[i], kv_mj), "actuator kv (biasprm)")
        # biasprm[1] is -kp: the term that makes the bias affine in `length`.
        assert_true(
            _close(-Float64(py=biasprm[i][1]), kp_mj),
            "biasprm[1] != -kp — this is not a position servo's gain pair",
        )
        assert_true(
            _close(gears[i], Float64(py=gear[i][0])), "actuator gear"
        )
        assert_true(
            _close(cmin[i], Float64(py=cr[i][0]))
            and _close(cmax[i], Float64(py=cr[i][1])),
            "actuator ctrlrange — `<general ctrllimited=\"true\"/>` in the"
            " default applies to <position> too",
        )
        # mjTRN_JOINT == 0, mjTRN_TENDON == 3.
        assert_true(trn_n[i] > 0, "actuator transmission did not resolve")
        if Int(py=trntype[i]) == 3:
            saw_tendon = True
            assert_equal(
                trn_n[i], 2,
                "the tendon-driven actuator must reach BOTH fin roll DOFs —"
                " one is what a joint transmission would give",
            )
        else:
            assert_equal(trn_n[i], 1, "a joint transmission is a single DOF")

    assert_true(
        saw_tendon,
        "no tendon-transmission actuator left — this gate no longer covers"
        " the multi-DOF path (bug 25)",
    )


def test_fish_tendons_match_mujoco() raises:
    """THE BUG 25 GATE: `merge_mjcf` used to drop `<tendon>` entirely.

    Both of fish's tendons vanished on the way in — the `fins_flap`
    transmission (which then failed the G3 guard, loudly) and the `fins_sym`
    spring (which would NOT have failed anything: a missing passive force is
    just a slightly different fish). Checked against MuJoCo's own
    `tendon_stiffness` / `tendon_lengthspring` / `wrap_*` arrays.
    """
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")
    assert_equal(
        M._acd.ntendon, Int(py=mj.ntendon),
        "tendon count — 0 here means `merge_mjcf` dropped the section",
    )

    var stiff = mj.tendon_stiffness.tolist()
    var lspring = mj.tendon_lengthspring.tolist()
    var tadr = mj.tendon_adr.tolist()
    var tnum = mj.tendon_num.tolist()
    var wobj = mj.wrap_objid.tolist()
    var wprm = mj.wrap_prm.tolist()
    var jdof = mj.jnt_dofadr.tolist()
    var jqpos = mj.jnt_qposadr.tolist()

    # Same comptime-index hoist as the actuator gate — see its docstring.
    var t_k = List[Float64]()
    var t_lo = List[Float64]()
    var t_hi = List[Float64]()
    var t_n = List[Int]()
    var t_qadr = List[Int]()
    var t_dadr = List[Int]()
    var t_coef = List[Float64]()
    comptime for a in range(8):
        t_k.append(M._acd.tendon_stiffness[a])
        t_lo.append(M._acd.tendon_spring_lo[a])
        t_hi.append(M._acd.tendon_spring_hi[a])
        t_n.append(M._acd.tendon_trn_n[a])
    # ⚠ 8 tendons * the WRAP STRIDE, not a literal 32. The stride moved
    # 4 -> 16 with defect 17; copying 32 entries then took two tendons'
    # worth instead of eight, and the `t * 4 + k` reads below compounded it.
    comptime for a in range(8 * MAX_COMPTIME_TENDON_WRAPS):
        t_qadr.append(M._acd.tendon_trn_qadr[a])
        t_dadr.append(M._acd.tendon_trn_dadr[a])
        t_coef.append(M._acd.tendon_trn_coef[a])

    var saw_spring = False
    for t in range(Int(py=mj.ntendon)):
        var n_mj = Int(py=tnum[t])
        print(
            "   tendon", t, " stiffness ours", M._acd.tendon_stiffness[t],
            " mj", Float64(py=stiff[t]),
            " band [", t_lo[t], ",", t_hi[t], "] mj [",
            Float64(py=lspring[t][0]), ",", Float64(py=lspring[t][1]), "]",
            " n", t_n[t], "/", n_mj,
        )
        assert_true(
            _close(t_k[t], Float64(py=stiff[t])),
            "tendon_stiffness",
        )
        # The band defaults to the tendon's length at qpos0, NOT to zero.
        assert_true(
            _close(t_lo[t], Float64(py=lspring[t][0]))
            and _close(t_hi[t], Float64(py=lspring[t][1])),
            "tendon_lengthspring",
        )
        assert_equal(t_n[t], n_mj, "tendon joint count")
        for k in range(n_mj):
            var w = Int(py=tadr[t]) + k
            var jnt = Int(py=wobj[w])
            assert_equal(
                t_dadr[t * MAX_COMPTIME_TENDON_WRAPS + k], Int(py=jdof[jnt]),
                "tendon joint dof address",
            )
            assert_equal(
                t_qadr[t * MAX_COMPTIME_TENDON_WRAPS + k], Int(py=jqpos[jnt]),
                "tendon joint qpos address",
            )
            assert_true(
                _close(t_coef[t * MAX_COMPTIME_TENDON_WRAPS + k], Float64(py=wprm[w])),
                "tendon coefficient (wrap_prm) — the sign is what makes"
                " fins_flap antisymmetric and fins_sym symmetric",
            )
        if Float64(py=stiff[t]) != 0.0:
            saw_spring = True

    assert_true(
        saw_spring,
        "no tendon carries a stiffness any more — the spring half of this"
        " gate is vacuous",
    )


def test_fish_geom_frames_match_mujoco() raises:
    """Geom local pos AND quaternion, per geom, in our text order.

    The quaternions are the point: `mouth` is a `fromto` capsule, so MuJoCo's
    compiler DERIVES its frame (z along the segment) and the MJCF never states
    it. `mouth_to_target` reads exactly that frame, so a different fromto->quat
    convention would rotate a third of the swim observation with nothing else
    going wrong.
    """
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var names = _geom_names()
    var gq = mj.geom_quat.tolist()
    var gp = mj.geom_pos.tolist()
    var worst = Float64(0)
    var saw_derived = False
    for gi in range(M.NGEOM):
        var go = gi * MODEL_GEOM_SIZE
        var name = names[gi]
        var mj_id = Int(
            py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, name)
        )
        assert_true(mj_id >= 0, String("no reference geom named ") + name)
        # The target is OURS-only in placement: the reference's is a static
        # world geom at pos="0 .4 .1", ours sits at the origin of a mocap body
        # carrying that pose. Its local pos is therefore 0 by construction, and
        # comparing it against the reference's world pos would be comparing two
        # different quantities. Its ORIENTATION still has to match.
        var skip_pos = gi == TARGET_GEOM_IDX
        # MuJoCo stores (w, x, y, z); ours (x, y, z, w).
        var pairs = [
            (Float64(mf.geoms.data[go + GEOM_IDX_QUAT_W]), Float64(py=gq[mj_id][0])),
            (Float64(mf.geoms.data[go + GEOM_IDX_QUAT_X]), Float64(py=gq[mj_id][1])),
            (Float64(mf.geoms.data[go + GEOM_IDX_QUAT_Y]), Float64(py=gq[mj_id][2])),
            (Float64(mf.geoms.data[go + GEOM_IDX_QUAT_Z]), Float64(py=gq[mj_id][3])),
            (Float64(mf.geoms.data[go + GEOM_IDX_POS_X]), Float64(py=gp[mj_id][0])),
            (Float64(mf.geoms.data[go + GEOM_IDX_POS_Y]), Float64(py=gp[mj_id][1])),
            (Float64(mf.geoms.data[go + GEOM_IDX_POS_Z]), Float64(py=gp[mj_id][2])),
        ]
        for k in range(len(pairs)):
            if skip_pos and k >= 4:
                continue
            var e = abs(pairs[k][0] - pairs[k][1])
            if e > worst:
                worst = e
            assert_true(
                _close(pairs[k][0], pairs[k][1]),
                String("geom ") + name + " frame component " + String(k),
            )
        # Non-vacuity: at least one geom's quaternion must be non-identity,
        # or this test only proves that identity equals identity.
        if abs(Float64(py=gq[mj_id][0]) - 1.0) > 1e-9:
            saw_derived = True
    print("  worst geom frame abs err =", worst)
    assert_true(
        saw_derived,
        "every geom quaternion is identity — the fromto-derived frame this"
        " gate exists for is gone",
    )

    # The skipped comparison, asserted as the substitution it is: our target
    # geom sits at its mocap body's origin, and the body's rest pose is the
    # reference geom's world position.
    var tgo = TARGET_GEOM_IDX * MODEL_GEOM_SIZE
    var tgt_mj = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )
    assert_true(
        abs(Float64(mf.geoms.data[tgo + GEOM_IDX_POS_X])) <= 1e-15
        and abs(Float64(mf.geoms.data[tgo + GEOM_IDX_POS_Y])) <= 1e-15
        and abs(Float64(mf.geoms.data[tgo + GEOM_IDX_POS_Z])) <= 1e-15,
        "the target geom is not at its mocap body's origin, so"
        " `geom_xpos['target'] == d.xpos[TARGET_BODY_IDX]` no longer holds"
        " and the swim observation reads the wrong point",
    )
    assert_true(
        _close(
            Float64(mf.bodies.data[TARGET_BODY_IDX * MODEL_BODY_SIZE + BODY_IDX_POS_Y]),
            Float64(py=gp[tgt_mj][1]),
        ),
        "the mocap target body's rest pose does not match the reference"
        " geom's world position",
    )

    # And the two indices the observation reads by index on our side.
    assert_equal(
        Int(mf.geoms.data[MOUTH_GEOM_IDX * MODEL_GEOM_SIZE + GEOM_IDX_BODY]),
        TORSO_BODY_IDX,
        "MOUTH_GEOM_IDX does not point at a torso geom",
    )
    assert_equal(
        Int(mf.geoms.data[TARGET_GEOM_IDX * MODEL_GEOM_SIZE + GEOM_IDX_BODY]),
        TARGET_BODY_IDX,
        "TARGET_GEOM_IDX does not point at the mocap target body",
    )


def test_fish_bodies_and_options_match_mujoco() raises:
    """Mass/inertia per body, plus the two `<option>` facts fish rests on.

    `torso_massive` is `group="4"` and is the ONLY torso geom with mass, so a
    narrower `inertiagrouprange` than MuJoCo's "0 5" default would leave the
    torso massless and every force on it infinite.
    """
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var dens = Float64(mf.meta.data[MODEL_META_IDX_DENSITY])
    var gz = Float64(mf.meta.data[MODEL_META_IDX_GRAVITY_Z])
    print("  density", dens, " gravity_z", gz)
    assert_true(_close(dens, Float64(py=mj.opt.density)), "option/density")
    assert_true(
        dens > 0.0,
        "density parsed as 0 — the fluid path early-outs and fish, which has"
        " no contacts and no gravity, has no forces left at all",
    )
    assert_true(
        abs(gz) <= 1e-15,
        "`<flag gravity=\"disable\"/>` did not zero the gravity vector",
    )

    var mm = mj.body_mass.tolist()
    var mi = mj.body_inertia.tolist()
    var worst = Float64(0)
    for b in range(Int(py=mj.nbody)):
        var bo = b * MODEL_BODY_SIZE
        var pairs = [
            (Float64(mf.bodies.data[bo + BODY_IDX_MASS]), Float64(py=mm[b])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IXX]), Float64(py=mi[b][0])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IYY]), Float64(py=mi[b][1])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IZZ]), Float64(py=mi[b][2])),
        ]
        for k in range(len(pairs)):
            var e = abs(pairs[k][0] - pairs[k][1])
            if e > worst:
                worst = e
            assert_true(
                _close(pairs[k][0], pairs[k][1]),
                String("body ") + String(b) + " mass/inertia " + String(k),
            )
    print("  worst body mass/inertia abs err =", worst)
    assert_true(
        Float64(py=mm[TORSO_BODY_IDX]) > 0.0,
        "the torso is massless — `torso_massive` (group 4) fell outside the"
        " inertiagrouprange",
    )


def test_fish_constraint_disable_is_reproduced() raises:
    """`<flag constraint="disable"/>`: no contacts, and every joint unlimited.

    Every fish joint carries a `range` while `class="fish"` sets
    `limited="false"`, so the range is decoration — MuJoCo agrees, and a
    parser that inferred "has a range therefore limited" would invent limit
    rows for eight joints in a model whose solver is switched off entirely.
    """
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var jlim = mj.jnt_limited.tolist()
    var mj_damp = mj.dof_damping.tolist()
    var mj_stiff = mj.jnt_stiffness.tolist()
    var mj_dadr = mj.jnt_dofadr.tolist()
    var saw_stiffness = False
    for j in range(M.NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var lo = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MIN])
        var hi = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MAX])
        assert_equal(
            Int(py=jlim[j]), 0,
            String("joint ") + String(j) + " is limited in MuJoCo, which"
            " contradicts this model",
        )
        assert_true(
            lo < -1e9 and hi > 1e9,
            String("joint ") + String(j) + " does not carry our unlimited"
            " sentinel, so the limit builder would invent a row for it",
        )
        var d_adr = Int(py=mj_dadr[j])
        assert_true(
            _close(
                Float64(mf.joints.data[jo + JOINT_IDX_DAMPING]),
                Float64(py=mj_damp[d_adr]),
            ),
            "dof_damping",
        )
        assert_true(
            _close(
                Float64(mf.joints.data[jo + JOINT_IDX_STIFFNESS]),
                Float64(py=mj_stiff[j]),
            ),
            "jnt_stiffness — tail2 carries one (8e-5)",
        )
        if Float64(py=mj_stiff[j]) != 0.0:
            saw_stiffness = True

    assert_true(saw_stiffness, "no joint spring left in the model")

    var names = _geom_names()
    for g in range(M.NGEOM):
        var go = g * MODEL_GEOM_SIZE
        assert_true(
            Int(mf.geoms.data[go + GEOM_IDX_CONTYPE]) == 0
            and Int(mf.geoms.data[go + GEOM_IDX_CONAFFINITY]) == 0,
            String("geom ") + names[g] + " can still collide, but the"
            " model disables the constraint solver",
        )


def test_fish_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0` against MuJoCo's own arrays.

    Run for every newly ported model since finger (bug 20). Vacuous for the
    dynamics HERE — fish has no constraints at all — but these are also the
    first quantities to move if the mass distribution is wrong, which for a
    model whose only massive geom is in group 4 is worth a direct check.
    """
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var worst = Float64(0)
    for b in range(Int(py=mj.nbody)):
        for k in range(2):
            var ours = Float64(mf.body_invweight0.data[2 * b + k])
            var mref = Float64(py=biw[b][k])
            var rel = abs(ours - mref) / (1e-15 + abs(mref))
            if rel > worst:
                worst = rel
            assert_true(
                rel <= 1e-9,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(Int(py=mj.nv)):
        var o = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(o - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(rel <= 1e-9, "dof_invweight0 mismatch")
    print("  worst invweight0 rel err =", worst)


# ── dynamics + observation + reward ──────────────────────────────────────────


def _seed_qpos() -> List[Float64]:
    """A deterministic tilted pose: root quaternion off-axis (so `upright` is
    neither 1 nor 0) and every internal joint bent."""
    var q0 = List[Float64]()
    for _ in range(NQ_F):
        q0.append(0.0)
    q0[2] = 0.1
    var qw = 0.9
    var qx = 0.2
    var qy = -0.3
    var qz = 0.15
    var n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    q0[3] = qw / n
    q0[4] = qx / n
    q0[5] = qy / n
    q0[6] = qz / n
    for i in range(N_ROOT_QPOS, NQ_F):
        q0[i] = 0.15 * sin(1.3 * Float64(i))
    return q0^


def _swim_rollout(n_steps: Int) raises -> List[Float64]:
    """Lockstep against the reference. Returns
    [worst_state, worst_obs, worst_reward, upright_min, upright_max,
     reward_min, reward_max]."""
    comptime EnvT = Phyics3dEnv[
        DMFishSwimModel, DMFishSwimConfig, DType.float64, False
    ]
    var mujoco = Python.import_module("mujoco")
    var mj = _ref()
    var dat = mujoco.MjData(mj)

    var tgt_gid = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )
    var mouth_gid = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "mouth")
    )
    var torso_bid = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "torso")
    )

    var q0 = _seed_qpos()
    var tx = 0.2
    var ty = 0.3
    var tz = 0.15

    mujoco.mj_resetData(mj, dat)
    mj.geom_pos[tgt_gid][0] = tx
    mj.geom_pos[tgt_gid][1] = ty
    mj.geom_pos[tgt_gid][2] = tz
    for i in range(NQ_F):
        dat.qpos[i] = q0[i]
    mujoco.mj_forward(mj, dat)

    var env = EnvT()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(NQ_F):
        qs.append(q0[i])
    for _ in range(NV_F):
        vs.append(0.0)
    env.set_state(qs, vs)
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = tx
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = ty
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = tz

    var worst_state = Float64(0)
    var worst_obs = Float64(0)
    var worst_rew = Float64(0)
    var u_min = Float64(1e9)
    var u_max = Float64(-1e9)
    var r_min = Float64(1e9)
    var r_max = Float64(-1e9)

    for step in range(n_steps):
        var act = EnvT.ActionType()
        for k in range(NACT_F):
            var a = 0.9 * sin(0.21 * Float64(step) + 1.1 * Float64(k))
            dat.ctrl[k] = a
            act.data[k] = a
        for _ in range(FRAME_SKIP_F):
            mujoco.mj_step(mj, dat)
        mujoco.mj_forward(mj, dat)
        var out = env.step(act)
        var obs = out[0]

        var ds = Float64(0)
        for i in range(NQ_F):
            var e = abs(Float64(py=dat.qpos[i]) - Float64(env.d.qpos.data[i]))
            if e > ds:
                ds = e
        for i in range(NV_F):
            var e = abs(Float64(py=dat.qvel[i]) - Float64(env.d.qvel.data[i]))
            if e > ds:
                ds = e
        if ds > worst_state:
            worst_state = ds

        # Reference observation, as `Swim.get_observation` builds it.
        var ref_obs = List[Float64]()
        for i in range(N_ROOT_QPOS, NQ_F):  # physics.joint_angles()
            ref_obs.append(Float64(py=dat.qpos[i]))
        var xm = dat.xmat.tolist()
        var upright = Float64(py=xm[torso_bid][8])  # xmat['torso', 'zz']
        ref_obs.append(upright)
        var gx = dat.geom_xpos.tolist()
        var gm = dat.geom_xmat.tolist()
        var dv = List[Float64]()
        for c in range(3):
            dv.append(
                Float64(py=gx[tgt_gid][c]) - Float64(py=gx[mouth_gid][c])
            )
        var m2t = List[Float64]()
        for c in range(3):  # mouth_to_target.dot(geom_xmat['mouth'])
            var acc = Float64(0)
            for r in range(3):
                acc += dv[r] * Float64(py=gm[mouth_gid][r * 3 + c])
            m2t.append(acc)
            ref_obs.append(acc)
        for i in range(NV_F):  # physics.velocity()
            ref_obs.append(Float64(py=dat.qvel[i]))

        var do_ = Float64(0)
        for i in range(len(ref_obs)):
            var e = abs(ref_obs[i] - Float64(obs.data[i]))
            if e > do_:
                do_ = e
        if do_ > worst_obs:
            worst_obs = do_

        var dist = sqrt(
            m2t[0] * m2t[0] + m2t[1] * m2t[1] + m2t[2] * m2t[2]
        )
        var in_target = tolerance(dist, 0.0, SWIM_RADII, 2.0 * SWIM_RADII)
        var ref_r = (7.0 * in_target + 0.5 * (upright + 1.0)) / 8.0
        var dr = abs(ref_r - Float64(out[1]))
        if dr > worst_rew:
            worst_rew = dr
        if upright < u_min:
            u_min = upright
        if upright > u_max:
            u_max = upright
        if ref_r < r_min:
            r_min = ref_r
        if ref_r > r_max:
            r_max = ref_r

    return [
        worst_state, worst_obs, worst_rew, u_min, u_max, r_min, r_max,
    ]


def test_fish_swim_dynamics_obs_and_reward_match_mujoco() raises:
    """The real gate: position servos, the tendon transmission and spring,
    fluid drag, and all 24 swim observation entries, over 60 control steps
    (600 integrator steps)."""
    var r = _swim_rollout(60)
    print(
        "  worst state", r[0], " obs", r[1], " reward", r[2],
        "\n  upright in [", r[3], ",", r[4], "]  reward in [", r[5], ",",
        r[6], "]",
    )
    assert_true(r[0] <= STATE_TOL, "qpos/qvel diverge from MuJoCo")
    assert_true(r[1] <= OBS_TOL, "observation diverges from MuJoCo")
    assert_true(r[2] <= REWARD_TOL, "reward diverges from MuJoCo")

    # Non-vacuity: the pose has to actually move, and `upright` has to be off
    # its saturation point, or the reward is a constant matching a constant.
    assert_true(
        r[4] - r[3] > 1e-4,
        "`upright` never moved — the rollout says nothing about the"
        " orientation half of the reward",
    )
    assert_true(
        r[6] - r[5] > 1e-5, "the reward never moved"
    )


def test_fish_upright_obs_and_reward_match_mujoco() raises:
    """The `upright` task: the same physics, a different observation (no
    target) and the degenerate-interval `tolerance(u, (1, 1), margin=1)`."""
    comptime EnvT = Phyics3dEnv[
        DMFishUprightModel, DMFishUprightConfig, DType.float64, False
    ]
    var mujoco = Python.import_module("mujoco")
    var mj = _ref()
    var dat = mujoco.MjData(mj)
    var torso_bid = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "torso")
    )

    var q0 = _seed_qpos()
    mujoco.mj_resetData(mj, dat)
    for i in range(NQ_F):
        dat.qpos[i] = q0[i]
    mujoco.mj_forward(mj, dat)

    var env = EnvT()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(NQ_F):
        qs.append(q0[i])
    for _ in range(NV_F):
        vs.append(0.0)
    env.set_state(qs, vs)

    var worst_obs = Float64(0)
    var worst_rew = Float64(0)
    var r_min = Float64(1e9)
    var r_max = Float64(-1e9)
    for step in range(40):
        var act = EnvT.ActionType()
        for k in range(NACT_F):
            var a = 0.7 * sin(0.17 * Float64(step) + 0.9 * Float64(k))
            dat.ctrl[k] = a
            act.data[k] = a
        for _ in range(FRAME_SKIP_F):
            mujoco.mj_step(mj, dat)
        mujoco.mj_forward(mj, dat)
        var out = env.step(act)
        var obs = out[0]

        var ref_obs = List[Float64]()
        for i in range(N_ROOT_QPOS, NQ_F):
            ref_obs.append(Float64(py=dat.qpos[i]))
        var xm = dat.xmat.tolist()
        var upright = Float64(py=xm[torso_bid][8])
        ref_obs.append(upright)
        for i in range(NV_F):
            ref_obs.append(Float64(py=dat.qvel[i]))

        for i in range(len(ref_obs)):
            var e = abs(ref_obs[i] - Float64(obs.data[i]))
            if e > worst_obs:
                worst_obs = e

        var ref_r = tolerance(upright, 1.0, 1.0, 1.0)
        var dr = abs(ref_r - Float64(out[1]))
        if dr > worst_rew:
            worst_rew = dr
        if ref_r < r_min:
            r_min = ref_r
        if ref_r > r_max:
            r_max = ref_r

    print(
        "  worst obs", worst_obs, " reward", worst_rew,
        " reward in [", r_min, ",", r_max, "]",
    )
    assert_true(worst_obs <= OBS_TOL, "observation diverges from MuJoCo")
    assert_true(worst_rew <= REWARD_TOL, "reward diverges from MuJoCo")
    assert_true(r_max - r_min > 1e-5, "the reward never moved")


def test_fish_servo_force_changes_within_a_control_step() raises:
    """THE PER-SUBSTEP GATE.

    A `<position>` servo's force is `kp*(ctrl - length)`, and `length` is a
    `qpos` read — so hoisting the force computation out of the substep loop
    (which is where it lived for every motor-only model before fish) freezes
    the spring at its start-of-step value for all `FRAME_SKIP` substeps.

    `test_fish_swim_dynamics_*` already fails if the force is hoisted. This
    test exists so the FAILURE IS LEGIBLE rather than a mystery drift: it
    measures how much the servo term actually moves across one control step
    and asserts that motion is far larger than the parity tolerance. If this
    passes and the parity test fails, the hoist is the reason.
    """
    comptime EnvT = Phyics3dEnv[
        DMFishSwimModel, DMFishSwimConfig, DType.float64, False
    ]
    var env = EnvT()
    _ = env.reset()
    var q0 = _seed_qpos()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(NQ_F):
        qs.append(q0[i])
    for _ in range(NV_F):
        vs.append(0.0)
    env.set_state(qs, vs)

    # Actuator 0 drives `tail1` (qpos 7) with kp = 5e-4.
    var qadr = M._acd.motor_trn_qadr[0]
    var kp = M._acd.motor_kp[0]
    var ctrl = 0.9

    var worst_drift = Float64(0)
    for step in range(20):
        var before = Float64(env.d.qpos.data[qadr])
        var act = EnvT.ActionType()
        for k in range(NACT_F):
            act.data[k] = ctrl
        _ = env.step(act)
        var after = Float64(env.d.qpos.data[qadr])
        # How much the servo's own force term moved across the control step.
        var drift = abs(kp * (before - after))
        if drift > worst_drift:
            worst_drift = drift

    print(
        "  kp", kp, " max |kp * d(length)| across one control step",
        worst_drift, " vs STATE_TOL", STATE_TOL,
    )
    assert_true(
        worst_drift > 100.0 * STATE_TOL,
        "the servo's transmission length barely moves across a control step,"
        " so this model cannot distinguish a per-substep force from a hoisted"
        " one — the gate is vacuous and the parity test is not covering the"
        " per-substep behaviour",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
