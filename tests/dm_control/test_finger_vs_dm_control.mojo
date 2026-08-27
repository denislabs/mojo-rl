"""dm_control `finger` — MODEL parity against the reference XML.

First gate of the finger port (spin, turn_easy, turn_hard). It checks only the
compiled MODEL, because every later gate is meaningless if the constants differ:
counts, masses/inertias, joint ranges and `ref`, damping/frictionloss, gears,
and the body/site indices the observation wiring will hard-code.

Deliberate deviations from the reference, asserted here rather than glossed:
  * the target site is wrapped in a MOCAP BODY (gap G4) — so NBODY is one more
    than the reference's, and the extra body is inert (no joint, no geom, and
    it must not add a DOF);
  * geom ORDER is ours (XML text order), not MuJoCo's (sorted by body id), so
    geoms are matched BY NAME, never by index — `mj_name2id` on a geom is a
    known trap in this port.

⚠ `joint proximal` carries `ref="-90"`, the one joint in the suite so far whose
qpos0 is non-zero, and bug 18 showed a wrong reference pose corrupts every
constraint inverse weight silently. So this gate pins `qpos0`/`dof_invweight0`
against MuJoCo explicitly.

Run: pixi run mojo run -I . tests/dm_control/test_finger_vs_dm_control.mojo
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.python import Python, PythonObject
from std.math import abs, sin, cos, sqrt, log1p, pi
from max.gpu.host import DeviceContext

from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.dm_control.finger.finger_config import DMFingerTurnConfig
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_DAMPING,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_QPOS0,
)
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.envs.dm_control.finger.finger_xml import (
    DMFingerSpinModel,
    DMFingerTurnModel,
    PROXIMAL_BODY_IDX,
    DISTAL_BODY_IDX,
    SPINNER_BODY_IDX,
    TARGET_BODY_IDX,
    SPINNER_RADIUS,
    TARGET_Z,
)
comptime MD_3 = ModelDims[DMFingerTurnModel]
comptime MD_2 = ModelDims[DMFingerSpinModel]

comptime DTYPE = DType.float64
comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/finger.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime M = DMFingerTurnModel  # verbatim reference; spin differs by design
comptime MD = ModelDims[M]
comptime MODEL_TOL: Float64 = 1e-14

# Rollout gate parameters. `_CONTROL_TIMESTEP` .02 over a .01 physics step.
comptime NQ_F: Int = 3
comptime NV_F: Int = 3
comptime NACT_F: Int = 2
comptime FRAME_SKIP_F: Int = 2
comptime N_STEPS_F: Int = 80
comptime AMP_F: Float64 = 0.35

comptime STATE_TOL_F: Float64 = 1e-9
comptime OBS_TOL_F: Float64 = 1e-9

# Early-contact residual. The contact phase IS gated — 8.84e-9 measured
# 2026-08-01 over 56 contact steps with peak forces of 116 N — since
# `constraints/scalar_rows.mojo` made joint limits and dry friction rows of the
# same system as the contacts (see the test docstring).
#
# ⚠ This comment used to read "NOT a parity tolerance — records a KNOWN,
# UNFIXED contact-phase gap" and the bound was 0.08, ~7 orders of magnitude
# loose. That was already stale when the fix landed on 2026-07-30, and it cost
# real time on 2026-08-01: it was read as evidence that finger still had an
# open contact defect, which made it look like the common cause behind
# manipulator's grasp error and MetaWorld sawyer's NaN. It is not — finger's
# elliptic contacts are exact. **A stale comment on a loose bound is a false
# lead with a long half-life; retire the bound WITH the fix.**
#
# Tightened to a MuJoCo-anchored number with an order of magnitude of headroom.
comptime CONTACT_STATE_BOUND: Float64 = 1e-7
comptime EARLY_CONTACT_STEPS: Int = 10

comptime BIG_TARGET: Float64 = 0.07  # _EASY_TARGET_SIZE
comptime SMALL_TARGET: Float64 = 0.03  # _HARD_TARGET_SIZE


def _ref() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path(String(REF_XML))


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= MODEL_TOL * (1.0 + abs(b))


def test_finger_counts() raises:
    """Counts match, with the mocap target the single accounted-for extra."""
    var mj = _ref()
    print(
        "  ours  NBODY", M.NBODY, " NJOINT", M.NJOINT, " NQ", M.NQ,
        " NV", M.NV, " NGEOM", M.NGEOM, " NSITE", M.NSITE,
    )
    print(
        "  mjcf  nbody", Int(py=mj.nbody), " njnt", Int(py=mj.njnt),
        " nq", Int(py=mj.nq), " nv", Int(py=mj.nv),
        " ngeom", Int(py=mj.ngeom), " nsite", Int(py=mj.nsite),
    )
    # The mocap target body is ours alone; everything else must agree.
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


def test_finger_joints_match() raises:
    """Joint ranges, `ref`, damping, frictionloss and armature, per joint.

    `proximal` has ref="-90" (degrees in this file, so -pi/2 after the
    compiler's angle conversion) — the value that seeds qpos0.
    """
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var jr = mj.jnt_range.tolist()
    var mj_qpos0 = mj.qpos0.tolist()
    var mj_damp = mj.dof_damping.tolist()
    var mj_fric = mj.dof_frictionloss.tolist()
    var mj_arm = mj.dof_armature.tolist()
    var mj_qadr = mj.jnt_qposadr.tolist()
    var mj_dadr = mj.jnt_dofadr.tolist()
    var mj_limited = mj.jnt_limited.tolist()

    var worst = Float64(0)
    for j in range(M.NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var lo = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MIN])
        var hi = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MAX])
        var q0 = Float64(mf.joints.data[jo + JOINT_IDX_QPOS0])
        var dp = Float64(mf.joints.data[jo + JOINT_IDX_DAMPING])
        var fr = Float64(mf.joints.data[jo + JOINT_IDX_FRICTIONLOSS])
        var ar = Float64(mf.joints.data[jo + JOINT_IDX_ARMATURE])

        var d_adr = Int(py=mj_dadr[j])
        var q_adr = Int(py=mj_qadr[j])
        print(
            "   joint", j, ": range [", lo, ",", hi, "] ref", q0,
            " damping", dp, " friction", fr, " armature", ar,
        )
        # Only LIMITED joints have a meaningful range to compare: MuJoCo
        # stores (0, 0) for an unlimited joint, where we store a +-1e10
        # sentinel. `hinge` is unlimited here, so comparing it would be
        # comparing two different conventions, not two values.
        if Int(py=mj_limited[j]) != 0:
            assert_true(_close(lo, Float64(py=jr[j][0])), "jnt_range min")
            assert_true(_close(hi, Float64(py=jr[j][1])), "jnt_range max")
        else:
            assert_true(
                lo < -1e9 and hi > 1e9,
                "an UNLIMITED joint must carry our unlimited sentinel, else"
                " the limit builder will invent a constraint row for it",
            )
        assert_true(
            _close(q0, Float64(py=mj_qpos0[q_adr])),
            "joint ref / qpos0 — a wrong reference pose silently corrupts"
            " every constraint inverse weight (bug 18)",
        )
        assert_true(_close(dp, Float64(py=mj_damp[d_adr])), "dof_damping")
        assert_true(
            _close(fr, Float64(py=mj_fric[d_adr])), "dof_frictionloss"
        )
        assert_true(_close(ar, Float64(py=mj_arm[d_adr])), "dof_armature")
        var e = abs(q0 - Float64(py=mj_qpos0[q_adr]))
        if e > worst:
            worst = e
    print("  worst |d qpos0| =", worst)

    # Non-vacuity: `proximal`'s ref is the whole reason this test pins qpos0.
    var any_nonzero = False
    for i in range(M.NQ):
        if abs(Float64(py=mj_qpos0[i])) > 1e-9:
            any_nonzero = True
    assert_true(
        any_nonzero,
        "no non-zero qpos0 — the ref=-90 case this gate exists for is gone",
    )


def test_finger_bodies_match() raises:
    """Masses and diagonal inertias, and the mocap target being inert."""
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var mj_mass = mj.body_mass.tolist()
    var mj_inertia = mj.body_inertia.tolist()
    # Reference body ids for the three real bodies (ours share these indices).
    var pairs = [
        (PROXIMAL_BODY_IDX, "proximal"),
        (DISTAL_BODY_IDX, "distal"),
        (SPINNER_BODY_IDX, "spinner"),
    ]
    var mujoco = Python.import_module("mujoco")
    var worst = Float64(0)
    for p in range(len(pairs)):
        var idx = pairs[p][0]
        var name = pairs[p][1]
        var ref_id = Int(
            py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, name)
        )
        assert_equal(
            idx, ref_id, String("body index for ") + String(name)
        )
        var bo = idx * MODEL_BODY_SIZE
        var m_ours = Float64(mf.bodies.data[bo + BODY_IDX_MASS])
        var m_ref = Float64(py=mj_mass[ref_id])
        print("   body", name, " mass ours", m_ours, " ref", m_ref)
        assert_true(_close(m_ours, m_ref), String("mass ") + String(name))
        var e = abs(m_ours - m_ref)
        if e > worst:
            worst = e
        var ii = [BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ]
        for k in range(3):
            var i_ours = Float64(mf.bodies.data[bo + ii[k]])
            var i_ref = Float64(py=mj_inertia[ref_id][k])
            assert_true(
                _close(i_ours, i_ref),
                String("inertia ") + String(name),
            )
    print("  worst |d mass| =", worst)

    # The mocap target must be physically INERT. Note it does NOT have zero
    # mass: it carries no geom, and our parser's default for a geomless body
    # is mass 1.0 (MuJoCo has no counterpart here at all — in the reference the
    # target is a bare site, not a body). That fabricated mass is harmless
    # *because the body carries no joint*: with no DOF it never enters M, so
    # its inverse weights are identically zero. Assert that, rather than the
    # mass, since the zero invweight is the property the physics depends on.
    var to = TARGET_BODY_IDX * MODEL_BODY_SIZE
    print(
        "  mocap target: mass =", Float64(mf.bodies.data[to + BODY_IDX_MASS]),
        " invweight0 =", Float64(mf.body_invweight0.data[2 * TARGET_BODY_IDX]),
        Float64(mf.body_invweight0.data[2 * TARGET_BODY_IDX + 1]),
    )
    assert_true(
        abs(Float64(mf.body_invweight0.data[2 * TARGET_BODY_IDX])) <= 1e-12
        and abs(Float64(mf.body_invweight0.data[2 * TARGET_BODY_IDX + 1]))
        <= 1e-12,
        "the mocap target has a non-zero inverse weight — it is reachable by"
        " some DOF and is therefore NOT inert",
    )


def test_finger_geoms_match_by_name() raises:
    """Geom sizes matched BY NAME — our geom ORDER differs from MuJoCo's."""
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")
    var mj_size = mj.geom_size.tolist()
    # `cap1`'s size drives Turn's target placement radius (.04 + .09 = .13).
    var cap1 = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "cap1")
    )
    var r = Float64(py=mj_size[cap1][0])
    var h = Float64(py=mj_size[cap1][1])
    print("  cap1 size =", r, h, " sum =", r + h)
    assert_true(
        _close(r + h, 0.13),
        "SPINNER_RADIUS no longer equals geom_size['cap1'].sum()",
    )


def test_spin_model_differs_only_in_hinge_damping() raises:
    """`spin` compiles from its own XML — assert the substitution took.

    `Spin.initialize_episode` writes `dof_damping['hinge'] = .03` (XML: .5),
    a dynamics change our shared unbatched model cannot make per episode, so
    spin gets its own compiled model. A silent no-op in that string
    substitution would leave spin running the TURN dynamics and nothing else
    would notice, so pin it: hinge damping .03 in spin, .5 in turn, and every
    other joint identical.
    """
    var ctx = DeviceContext()
    var spin = Model[DTYPE, MD_2]()
    DMFingerSpinModel.init_fields[DTYPE](ctx, spin)
    var turn = Model[DTYPE, MD_3]()
    DMFingerTurnModel.init_fields[DTYPE](ctx, turn)

    # Joint 2 is `hinge` (XML order: proximal, distal, hinge).
    var hs = Float64(spin.joints.data[2 * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING])
    var ht = Float64(turn.joints.data[2 * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING])
    print("  hinge damping: spin", hs, " turn", ht)
    assert_true(
        _close(hs, 0.03),
        "spin's hinge damping is not .03 — the XML substitution silently"
        " did nothing and spin is running the turn dynamics",
    )
    assert_true(_close(ht, 0.5), "turn's hinge damping should be the XML's .5")
    for j in range(DMFingerTurnModel.NJOINT):
        if j == 2:
            continue
        assert_true(
            _close(
                Float64(spin.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING]),
                Float64(turn.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING]),
            ),
            "the spin substitution changed a joint other than `hinge`",
        )


def test_touch_site_sphere_approximation_is_exact() raises:
    """The touch zones are ELLIPSOIDS that we silently model as SPHERES.

    `childclass="finger"` gives `touchtop`/`touchbottom`
    `type="ellipsoid" size=".025 .03 .025"`, but `_geom_type_from_str`
    (full_parser.mojo) has no `ellipsoid` case and falls through to
    `return _GEOM_SPHERE` — a SILENT substitution, not an error, so the touch
    sensor never raises and instead measures a sphere of radius size[0].

    That is exact HERE and only here, for two reasons worth pinning rather
    than trusting: the ellipsoid's x and z semi-axes both equal size[0], and
    the finger is planar in x-z (every joint has `axis="0 -1 0"`), so contacts
    occur at y ~ 0 where the only differing semi-axis is irrelevant. If either
    fact stops holding, the approximation silently starts lying — hence this
    gate rather than a comment.
    """
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")
    var ss = mj.site_size.tolist()
    var st = mj.site_type.tolist()
    for name in ["touchtop", "touchbottom"]:
        var sid = Int(
            py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, name)
        )
        var sx = Float64(py=ss[sid][0])
        var sy = Float64(py=ss[sid][1])
        var sz = Float64(py=ss[sid][2])
        print(
            "  site", name, " type", Int(py=st[sid]),
            " semi-axes", sx, sy, sz,
        )
        assert_true(
            abs(sx - sz) <= 1e-12,
            String(name)
            + ": x and z semi-axes differ, so a sphere of radius size[0] is"
            " no longer the same zone in the contact plane — the touch sensor"
            " needs real ellipsoid support (sensors/touch.mojo)",
        )
        assert_true(
            sy >= sx - 1e-12,
            String(name)
            + ": the out-of-plane semi-axis shrank below the in-plane one,"
            " so the sphere approximation now OVER-reports the zone",
        )


def test_finger_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0` against MuJoCo's own arrays.

    These multiply EVERY constraint force — `mj_diagApprox` reads
    `dof_invweight0` for joint limits and `body_invweight0` for contacts — so
    an error here is a silent multiplicative error on all of them, with no
    symptom until something touches something else.

    finger is why this check now belongs in every domain gate. Its spinner is
    a symmetric wheel whose CoM sits on its own hinge axis, so MuJoCo gives it
    a translational weight of exactly 0 (it cannot be translated). We were
    substituting the ROTATIONAL weight whenever the translational one came out
    ~0 — under a comment claiming that was "MuJoCo behavior", which it is not
    — making the spinner's 26.24 instead of 0, the contact diagApprox 64x too
    large, and every fingertip/spinner contact 64x too soft.

    It hid because every model gated before finger is FREE-ROOTED, where no
    body has a zero translational weight, and because finger is the first
    model whose contacts are body-vs-BODY rather than body-vs-world-plane.
    """
    var mj = _ref()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var worst = Float64(0)
    var saw_zero_tran = False
    for b in range(Int(py=mj.nbody)):
        var pairs = [
            (Float64(mf.body_invweight0.data[2 * b]), Float64(py=biw[b][0])),
            (
                Float64(mf.body_invweight0.data[2 * b + 1]),
                Float64(py=biw[b][1]),
            ),
        ]
        for k in range(2):
            var ours = pairs[k][0]
            var mref = pairs[k][1]
            var rel = abs(ours - mref) / (1e-15 + abs(mref))
            if rel > worst:
                worst = rel
            assert_true(
                rel <= 1e-9,
                String("body_invweight0 mismatch on body ") + String(b),
            )
        if abs(Float64(py=biw[b][0])) <= 1e-12 and b > 0:
            saw_zero_tran = True
    for i in range(Int(py=mj.nv)):
        var o = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(o - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(rel <= 1e-9, "dof_invweight0 mismatch")
    print("  worst invweight0 rel err =", worst)

    # Non-vacuity: the defect this gate exists for only shows on a body whose
    # translational weight is genuinely zero. The spinner (and the mocap
    # target) are those bodies; if the model ever loses them this test stops
    # testing the thing it was written for.
    assert_true(
        saw_zero_tran,
        "no body has a zero translational invweight0 any more — this gate no"
        " longer covers the case it was written for",
    )


# ── dynamics + observation + reward ──────────────────────────────────────────


def _rollout[
    TSIZE: Float64
](target_angle: Float64, q0: List[Float64]) raises -> List[Float64]:
    """One turn rollout. Returns
    [prefix_len, pre_state, pre_obs, pre_reward, post_state, post_obs,
     touch_seen, r_min, r_max, contact_steps].

    `pre_*` are maxima over the CONTACT-FREE prefix, `post_*` over the whole
    run. The split exists because the two regimes have very different error:
    see the docstrings of the two tests below.
    """
    comptime EnvT = Phyics3dEnv[
        DMFingerTurnModel, DMFingerTurnConfig[TSIZE], DType.float64, False
    ]

    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))
    var dat = mujoco.MjData(m)

    var sadr = m.sensor_adr.tolist()
    var adr = List[Int]()
    var names = [
        "proximal", "distal", "proximal_velocity", "distal_velocity",
        "hinge_velocity", "tip", "target", "spinner", "touchtop",
        "touchbottom",
    ]
    for nm in names:
        adr.append(
            Int(py=sadr[Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nm))])
        )

    var tgt_site = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
    )
    var tx = 0.2 + SPINNER_RADIUS * sin(target_angle)
    var tz = TARGET_Z + SPINNER_RADIUS * cos(target_angle)

    # Reference: the model writes `Turn.initialize_episode` performs.
    mujoco.mj_resetData(m, dat)
    m.site_pos[tgt_site][0] = tx
    m.site_pos[tgt_site][2] = tz
    m.site_size[tgt_site][0] = TSIZE
    for i in range(NQ_F):
        dat.qpos[i] = q0[i]
    mujoco.mj_forward(m, dat)

    # Ours: the same target through the per-env mocap path (gap G4).
    var env = EnvT()
    _ = env.reset()
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = tx
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = 0.0
    env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = tz
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(NQ_F):
        qs.append(q0[i])
        vs.append(0.0)
    env.set_state(qs, vs)

    var prefix = -1
    var pre_state = 0.0
    var pre_obs = 0.0
    var pre_reward = 0.0
    var post_state = 0.0
    var post_obs = 0.0
    var touch_seen = 0.0
    var r_min = 1e9
    var r_max = -1e9
    var contact_steps = 0
    var early_state = 0.0

    for step in range(N_STEPS_F):
        var act = EnvT.ActionType()
        for k in range(NACT_F):
            var a = AMP_F * sin(0.07 * Float64(step) + 2.1 * Float64(k))
            dat.ctrl[k] = a
            act.data[k] = a
        for _ in range(FRAME_SKIP_F):
            mujoco.mj_step(m, dat)
        mujoco.mj_forward(m, dat)
        var out = env.step(act)
        var obs = out[0]

        if Int(py=dat.ncon) > 0 or Int(
            env.d.meta.data[META_IDX_NUM_CONTACTS]
        ) > 0:
            contact_steps += 1
            if prefix < 0:
                prefix = step

        var ds = 0.0
        for i in range(NQ_F):
            var e = abs(Float64(py=dat.qpos[i]) - Float64(env.d.qpos.data[i]))
            if e > ds:
                ds = e
        for i in range(NV_F):
            var e = abs(Float64(py=dat.qvel[i]) - Float64(env.d.qvel.data[i]))
            if e > ds:
                ds = e

        # Reference observation, straight from sensordata as finger.py reads it.
        var sd = dat.sensordata
        var sx = Float64(py=sd[adr[7] + 0])
        var sz = Float64(py=sd[adr[7] + 2])
        var tipx = Float64(py=sd[adr[5] + 0]) - sx
        var tipz = Float64(py=sd[adr[5] + 2]) - sz
        var tgx = Float64(py=sd[adr[6] + 0]) - sx
        var tgz = Float64(py=sd[adr[6] + 2]) - sz
        var ddx = tgx - tipx
        var ddz = tgz - tipz
        var dist = sqrt(ddx * ddx + ddz * ddz) - TSIZE
        var top = Float64(py=sd[adr[8]])
        var bot = Float64(py=sd[adr[9]])
        if top > touch_seen:
            touch_seen = top
        if bot > touch_seen:
            touch_seen = bot
        # ⚠⚠ THE TOUCH TERMS COME FROM NUMPY, NOT FROM MOJO. They used to be
        # `log1p(top), log1p(bot)` — the same `std.math.log1p` the config under
        # test called — so the two sides shared any error and it cancelled
        # exactly. That made this leg structurally blind to the arithmetic:
        # `std.math.log1p` carries up to 1.01e-06 RELATIVE error on
        # x in [0.05, 0.42] (libm: 1e-16), and nothing here could have seen it.
        # `finger.py` calls `np.log1p`, so that is the reference.
        var np_ = Python.import_module("numpy")
        var ref_obs = [
            Float64(py=sd[adr[0]]), Float64(py=sd[adr[1]]), tipx, tipz,
            Float64(py=sd[adr[2]]), Float64(py=sd[adr[3]]),
            Float64(py=sd[adr[4]]),
            Float64(py=np_.log1p(top)), Float64(py=np_.log1p(bot)),
            tgx, tgz, dist,
        ]
        var do_ = 0.0
        for i in range(12):
            var e = abs(ref_obs[i] - Float64(obs.data[i]))
            if e > do_:
                do_ = e

        var ref_r = 1.0 if dist <= 0.0 else 0.0
        var dr = abs(ref_r - Float64(out[1]))
        if ref_r < r_min:
            r_min = ref_r
        if ref_r > r_max:
            r_max = ref_r

        if ds > post_state:
            post_state = ds
        if do_ > post_obs:
            post_obs = do_
        # The first few contact steps measure the DEFECT; later ones measure
        # how fast a freely-spinning wheel amplifies it, which is a property
        # of the system, not of the port.
        if prefix >= 0 and step < prefix + EARLY_CONTACT_STEPS and ds > early_state:
            early_state = ds
        if prefix < 0:
            if ds > pre_state:
                pre_state = ds
            if do_ > pre_obs:
                pre_obs = do_
            if dr > pre_reward:
                pre_reward = dr

    var plen = Float64(N_STEPS_F) if prefix < 0 else Float64(prefix)
    return [
        plen, pre_state, pre_obs, pre_reward, post_state, post_obs,
        touch_seen, r_min, r_max, Float64(contact_steps), early_state,
    ]


def test_finger_contact_free_dynamics_and_obs_match_mujoco() raises:
    """Physics, all 12 observation entries and the reward, before any contact.

    This is the real gate on the port's wiring: integrator, joint limits,
    `framepos` reads through `site_xpos`/`xpos`, the mocap target, and the
    `dist_to_target` reward all have to be exact for these numbers to hold.
    Contacts are excluded here and measured separately below — see that test
    for why.
    """
    var far: List[Float64] = [-1.7, -1.0, 0.0]  # folded away from the spinner
    var r = _rollout[BIG_TARGET](0.0, far)
    print("finger turn_easy (contact-free) vs MuJoCo,", N_STEPS_F, "steps:")
    print("  contact-free prefix =", r[0], " contact steps =", r[9])
    print("  max |d(state)| =", r[1], " |d(obs)| =", r[2], " |d(reward)| =", r[3])

    # Non-vacuity: a rollout that ends instantly would pass everything.
    assert_true(
        r[0] >= 60.0,
        "contact-free prefix too short to gate anything — retune the init",
    )
    assert_true(r[1] <= STATE_TOL_F, "contact-free physics deviated")
    assert_true(r[2] <= OBS_TOL_F, "observation deviated")
    assert_true(r[3] <= 0.0, "reward deviated")


def test_finger_reward_gates_both_branches() raises:
    """`float(dist_to_target <= 0)` is a HARD indicator, so a run that only
    ever sees one side of the threshold gates nothing. Two target angles put
    the tip inside and outside the disc."""
    var far: List[Float64] = [-1.7, -1.0, 0.0]
    var inside = _rollout[BIG_TARGET](0.0, far)
    var outside = _rollout[BIG_TARGET](pi, far)
    print(
        "  reward range: target at 0 rad", inside[7], "..", inside[8],
        " at pi rad", outside[7], "..", outside[8],
    )
    assert_true(
        inside[8] == 1.0 and outside[7] == 0.0,
        "the two target angles no longer straddle the reward threshold —"
        " the indicator is being gated on one branch only",
    )
    assert_true(inside[3] <= 0.0 and outside[3] <= 0.0, "reward deviated")


def test_finger_contact_phase_residual_is_bounded() raises:
    """⚠ The CONTACT phase does NOT reach the parity the rest of the port does,
    and this test records how far off it is rather than pretending otherwise.

    Two engine defects were found here and BOTH are now fixed:

      * `body_invweight0`'s translational half was substituted with the
        rotational half whenever it was ~0 (`invweight.mojo`). The finger
        spinner is a symmetric wheel whose CoM sits on its own hinge axis, so
        its true translational weight IS 0; the substitution made it 26.24,
        the contact `diagApprox` 64x too large and every fingertip/spinner
        contact 64x too soft. Static-pose normal force 29.1 N -> 1860.17 N
        against MuJoCo's 1860.17 N.
      * joint `frictionloss` was an explicit Coulomb force that could not
        arrest motion (period-2 limit cycle); it is now a constraint row, see
        `constraints/friction_dof.mojo`. Isolated, the hinge now tracks
        MuJoCo's decay to ~1e-9 over fifteen orders of magnitude where it used
        to lock at +-0.0329 rad/s forever.

    ⚠ NEITHER fully closed this residual, and the frictionloss attribution
    that used to be written here was WRONG: fixing it moved the early-contact
    number only 0.0475 -> 0.0390.

    LOCALIZED 2026-07-30 to a single scalar — the SOLVED CONTACT FORCE — by a
    substep probe that forks our engine off MuJoCo's state, takes ONE Euler
    substep on each, and compares. What is now RULED OUT, by measurement and
    not by argument:

      * Narrow phase. Detected at an IDENTICAL state (FK + detect, no stepping
        in between), dist / pos / normal agree with MuJoCo to ~5e-11. The
        "~4e-4 at a static pose" written here before was measured at the END of
        a step, after the states had already diverged — a consequence read as a
        cause. A contact whose closest feature is a near-tangent capsule pair
        slides far under a tiny rotation, which is why it looked so big.
      * Contact Jacobian, mass matrix, smooth forces, the damping-implicit
        Euler update, and the saturated friction-dof row. Predicting the
        one-substep d(qvel) from the FORCE DIFFERENCE ALONE, through MuJoCo's
        own J and M+dt*D, reproduces the observed d(qvel) to 2e-10
        (ratio 1.00000003). If any of those differed, it could not.
      * The constraint problem itself. imp, R, aref and K/B were recomputed by
        hand from the model and match MuJoCo's efc rows exactly — including
        the solimp clamp (imp = 0.86403303 with dmin clamped to 1e-4, vs
        0.86402903 unclamped; MuJoCo reports the clamped value).

    CAUSE FOUND 2026-07-30, and it is NOT the solver. MuJoCo puts every
    constraint row in ONE system: here nefc = 4, nf = 1 — three elliptic
    contact rows AND a frictionloss row on the spinner dof, solved together.
    We solve the three contact rows in the Newton core and then apply the
    friction row as a SEPARATE PGS pass afterwards (`_friction_env`, and the
    same holds for `_limits_env` / `_equality_env` / the tendon rows). The
    contact solve therefore never sees the friction force, and the spinner dof
    that carries it is one the contact rows also act on.

    Proved by solving MuJoCo's own primal problem, from MuJoCo's own efc data
    (J, R, aref, qacc_smooth, mj_fullM), two ways at four contact substeps:

      A) all 4 rows jointly  -> reproduces MuJoCo's force to ~1e-14
      B) the 3 contact rows  -> reproduces OUR force to ~1e-7, i.e. to the
                                harness's own convergence level

    So our ELLIPTIC NEWTON SOLVE IS CORRECT — it returns the exact optimum of
    the problem it is handed. The problem it is handed is missing a row. The
    "KKT residual 0.517" recorded here before was measured against the FULL
    4-row system, which our force is not a solution of and was never asked to
    be; and the stale-Hessian LEAD was wrong twice over — H IS rebuilt and
    refactorized whenever a cone state changes (newton_solve.mojo, the
    `state_changed` branch), and a stale metric cannot bias a converged
    gradient anyway.

    FIXED 2026-07-30 by `physics3d/constraints/scalar_rows.mojo`: joint limits
    and dry-friction dofs are now ROWS of the same system, built once and
    solved with the contacts. J = sign*e_dof (one nonzero), so they cost O(rows)
    of local storage rather than O(rows*NV) — the elliptic Newton core sits near
    the Metal local-memory ceiling. Wired into `newton_solve` (both cones) and
    `cg_solve`; their `_limits_env` / `_friction_env` post-passes are gone.

      contact force vs MuJoCo   0.216 off  ->  2e-7
      substep |d qvel|          7.5e-3     ->  6.2e-9
      the residual gated below  0.0390     ->  8.8e-9

    Still SEQUENTIAL, deliberately: equality and tendon rows (they need a dense
    Jacobian, so they need different storage), and the PGS / island-PGS solvers.

    TIGHTENED 2026-08-01 from 0.08 to 1e-7, against a fresh measurement of
    8.84e-9 over 56 contact steps. It had been left ~7 orders loose with a
    stale "KNOWN, UNFIXED" comment on it, which read as an open contact defect
    on the only other ELLIPTIC model and sent a manipulator investigation down
    a blind alley. The domain is gated; the bound now says so.

    Superseded: (a) the claim that our elliptic solve returns a non-KKT point;
    (b) the claim that a one-step SEPARATION TIMING difference is a co-cause —
    it is downstream of the same force error.

    Never relax the bound. If it trips, the contact phase regressed.
    """
    var near: List[Float64] = [-0.9, 0.6, -2.0]
    var r = _rollout[BIG_TARGET](1.6, near)
    print("finger turn_easy (through contact) vs MuJoCo:")
    print("  contact steps =", r[9], " of", N_STEPS_F)
    print(
        "  first", EARLY_CONTACT_STEPS, "contact steps: max |d(state)| =", r[10]
    )
    print("  whole rollout (chaotic): max |d(state)| =", r[4], " |d(obs)| =", r[5])
    print("  peak MuJoCo touch force =", r[6])

    assert_true(r[9] >= 5.0, "no contact steps — this test gates nothing")
    assert_true(
        r[6] > 0.0,
        "MuJoCo's touch sensors never loaded, so the touch observation"
        " entries are still ungated",
    )
    assert_true(
        r[10] <= CONTACT_STATE_BOUND,
        "the early-contact residual GREW — a regression on top of the known"
        " frictionloss defect",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
