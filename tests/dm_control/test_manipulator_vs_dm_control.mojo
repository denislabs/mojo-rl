"""dm_control `manipulator-bring_ball` parity: our model vs MuJoCo's.

The reference is built by `manipulator_ref.py`, not `from_xml_path` — no task
uses `manipulator.xml` as written (see that file's docstring).

WHAT THIS DOMAIN EXERCISES THAT NO EARLIER ONE DID
--------------------------------------------------
  - ORIENTED SITES. `thumb_touch` / `finger_touch` inherit `euler="0 15 0"`
    from `class="hand"`. Every earlier ported site was axis-aligned or a
    sphere, so the site record's missing quaternion had never mattered.
  - BOX touch zones — all five `<touch>` sensors.
  - `<inertial>` as a CHILD ELEMENT (`pinch site`, mass 1e-6, no geom).
  - ELLIPTIC cone WITH a fixed-tendon equality (`coupling`).
  - A `<motor>` on a TENDON transmission (`grasp`).

THIS FILE FOUND TWO ENGINE BUGS, both silent and both now fixed:

  * the plane narrow phase ignored plane ORIENTATION, reducing every plane to
    a horizontal floor at its own z. manipulator's VERTICAL `background` plane
    (`pos="0 .2 .5" zaxis="0 -1 0"`) was read as a floor at z = 0.5 and
    invented a contact with `upper_arm` in every pose. Caught by the
    `our ncon == MuJoCo ncon` assertion below, which is why that assertion is
    there: bucketing by MuJoCo's live rows means nothing if our own row set
    differs.
  * the narrow phase's contact DIRECTION invariant. Twelve reversed-order
    branches negated the primitive's normal AND swapped body_a/body_b — a
    double flip that left them emitting `normal = body_b -> body_a` where the
    ten canonical-order branches emit `body_a -> body_b`. `aref` is built from
    the penetration DEPTH and does not flip with the normal, so a flipped
    normal desynchronises `jar = aref + J*qacc`. The grasp buckets were 5.21
    and 1.22 (a 100%+ qacc error) and are now 4.05e-9 and 1.01e-9.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulator_vs_dm_control.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulator import (
    DMManipulatorBringBallModel as M,
    DMManipulatorBringBallConfig,
    TARGET_BODY_IDX,
)
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_QPOS0,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    GEOM_IDX_BODY,
    GEOM_IDX_TYPE,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    SITE_IDX_SIZE_0,
    SITE_IDX_SIZE_1,
    SITE_IDX_SIZE_2,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
    TENDON_IDX_INVWEIGHT0,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_ELLIPSOID,
)


comptime DTYPE = DType.float64

comptime NQ: Int = M.NQ  # 11
comptime NV: Int = M.NV  # 11
comptime NBODY: Int = M.NBODY  # 12
comptime NJOINT: Int = M.NJOINT  # 11
comptime NGEOM: Int = M.NGEOM  # 21
comptime NSITE: Int = M.NSITE  # 9
comptime NACT: Int = M.nact  # 5
comptime NTEN: Int = M.MAX_TENDON  # 2

# Model constants are exact rational arithmetic on both sides up to the
# inertia integrals, so anything above ~1e-12 is a real disagreement.
comptime TOL_MODEL: Float64 = 1e-9

# qacc gates, set from the measured worst case per bucket with roughly a
# decade of headroom. Measured 2026-08-01, all ten poses:
#   A equality only               9.94e-10
#   B equality + limit            6.02e-10
#   C equality + contact          4.05e-9
#   D equality + limit + contact  1.01e-9
# The C/D buckets were 5.21 and 1.22 — a 100%+ error — until the narrow
# phase's contact DIRECTION invariant was fixed; see the block comment above
# the physics layer. Split so a regression in a coupled bucket cannot hide
# behind the uncoupled one.
comptime TOL_QACC_UNCOUPLED: Float64 = 1e-8
comptime TOL_QACC_COUPLED: Float64 = 1e-8
# Observation and reward are pure readbacks of state the physics layer already
# gates, so they sit at the FK floor. Measured 2026-08-01: arm_pos/arm_vel/
# object_vel/target_pos exact, hand_pos and object_pos 5.0e-11, reward 1.8e-11.
comptime TOL_OBS: Float64 = 1e-9
# TOUCH reads POST-SOLVE contact forces, so it could in principle inherit the
# contact solve's floor (the qacc buckets sit at ~4e-9) rather than FK's.
#
# ⚠⚠ THAT IS WHAT THIS COMMENT USED TO CLAIM, AND IT WAS WRONG. It read:
# "Measured 1.36e-9 through `log1p` of 2-4 N" and attributed the residual to
# the solve. The residual was OUR ARITHMETIC: the touch observable was
# transcribed as `log(1.0 + f)`, which carries up to 1.02e-09 ABSOLUTE error
# against `np.log1p` on real touch forces, and 28% of this task's non-zero
# readings land in [0.05, 0.42] where it is worst. Switching to
# `dtype_math.log1p_dt` (the `2*atanh(f/(2+f))` identity) dropped the measured
# worst touch deviation from 1.36e-9 to 3.84e-14 — 35000x — with the contact
# solve untouched. The solve was never the floor here.
#
# Tightened to 1e-11 accordingly: at 3.84e-14 that is still 260x of headroom,
# and a tolerance three orders above the measurement cannot catch a
# regression. See `dm_control/dtype_math.log1p_dt` for the measurements.
comptime TOL_TOUCH: Float64 = 1e-11

# OUR site order IS MuJoCo's, as of the element-order fix (2026-08-03).
#
# It used to diverge here: `palm_touch` is declared AFTER the `pinch site`
# body but belongs to `hand`, so MuJoCo's body sort pulls it ahead of `pinch`
# while our XML-text walk left it behind. This file carried a permutation,
# `_our_site_to_mj`, to paper over that — treating the divergence as a
# property to record rather than a bug to fix.
#
# It was a bug. The same text-vs-body ordering permutes JOINTS, and
# `fields_build` derives `qpos_adr`/`dof_adr` as running counters over the
# joint array, so the whole `qpos` layout went with it — which is how
# dm_control's dog exposed it. `full_parser` now groups joints, geoms and
# sites by body id (`_stable_group_by_body_*`), gated by
# `tests/physics3d/test_element_order_vs_mujoco.mojo`.
#
# The identity below is kept rather than deleted so the call sites still read
# "our index -> MuJoCo's index", and so that a future divergence has one
# obvious place to be expressed.
def _our_site_to_mj(ours: Int) -> Int:
    """MuJoCo's site index for our site `ours` — now the identity."""
    return ours


def _mj_geom_type(ours: Int) -> Int:
    """Our `GEOM_*` code -> MuJoCo's `mjtGeom`.

    The two enums are NOT the same numbering — MuJoCo interleaves HFIELD at 1
    and orders ellipsoid/cylinder/box differently — so a direct `==` between
    `GEOM_IDX_TYPE` and `m.geom_type` compares nothing. Site types share the
    `mjtGeom` enum, so this maps both.
    """
    if ours == GEOM_PLANE:
        return 0  # mjGEOM_PLANE
    if ours == GEOM_SPHERE:
        return 2  # mjGEOM_SPHERE
    if ours == GEOM_CAPSULE:
        return 3  # mjGEOM_CAPSULE
    if ours == GEOM_ELLIPSOID:
        return 4  # mjGEOM_ELLIPSOID
    if ours == GEOM_CYLINDER:
        return 5  # mjGEOM_CYLINDER
    if ours == GEOM_BOX:
        return 6  # mjGEOM_BOX
    if ours == GEOM_MESH:
        return 7  # mjGEOM_MESH
    return -1


def _ref() raises -> PythonObject:
    """The compiled reference `mjModel` for `bring_ball`."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/dm_control")
    var builder = Python.import_module("manipulator_ref")
    return builder.model(False, False)


def _build() raises -> Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=M.MAX_EQUALITY, ntendon=NTEN, nsite=NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=M.MAX_EQUALITY, ntendon=NTEN, nsite=NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_manipulator_dims_match_mujoco() raises:
    """Element counts. Cheap, and the first thing a dropped `<body>` breaks."""
    var mj = _ref()
    assert_true(Int(py=mj.nq) == NQ, "nq mismatch")
    assert_true(Int(py=mj.nv) == NV, "nv mismatch")
    assert_true(Int(py=mj.nbody) == NBODY, "nbody mismatch")
    assert_true(Int(py=mj.njnt) == NJOINT, "njnt mismatch")
    assert_true(Int(py=mj.ngeom) == NGEOM, "ngeom mismatch")
    assert_true(Int(py=mj.nsite) == NSITE, "nsite mismatch")
    assert_true(Int(py=mj.nu) == NACT, "nu mismatch")
    assert_true(Int(py=mj.ntendon) == NTEN, "ntendon mismatch")
    assert_true(Int(py=mj.neq) == 1, "neq mismatch (the coupling equality)")
    assert_true(Int(py=mj.nsensor) == 5, "nsensor mismatch (5 touch sensors)")
    # cone=1 is mjCONE_ELLIPTIC. The parser does not read the attribute — the
    # cone is a `ModelDefFromXML` parameter — so pin that the reference still
    # asks for the cone we hardcoded.
    assert_true(Int(py=mj.opt.cone) == 1, "reference cone is no longer elliptic")
    assert_true(
        abs(Float64(py=mj.opt.timestep) - M.TIMESTEP) < 1e-15,
        "timestep mismatch",
    )


def test_manipulator_ordering_matches_mujoco() raises:
    """Body / joint / geom / site ORDER, pinned by name on the MuJoCo side.

    Ours is XML text order; MuJoCo's is sorted by body id. They coincide here
    because the four arena geoms and `arm_root` all precede the first body,
    but point_mass proved that assumption can fail silently — so pin it rather
    than trust it.
    """
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")

    var body_names = [
        "world", "upper_arm", "middle_arm", "lower_arm", "hand",
        "pinch site", "thumb", "thumbtip", "finger", "fingertip",
        "ball", "target_ball",
    ]
    for i in range(len(body_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body_names[i]))
            == i,
            String("MuJoCo body order moved at ") + body_names[i],
        )

    var joint_names = [
        "arm_root", "arm_shoulder", "arm_elbow", "arm_wrist",
        "thumb", "thumbtip", "finger", "fingertip",
        "ball_x", "ball_z", "ball_y",
    ]
    for i in range(len(joint_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i]))
            == i,
            String("MuJoCo joint order moved at ") + joint_names[i],
        )

    var geom_names = [
        "floor", "wall1", "wall2", "background", "arm_root",
        "upper_arm", "middle_arm", "lower_arm", "hand", "palm1", "palm2",
        "thumb1", "thumb2", "thumbtip1", "thumbtip2",
        "finger1", "finger2", "fingertip1", "fingertip2",
        "ball", "target_ball",
    ]
    for i in range(len(geom_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, geom_names[i]))
            == i,
            String("MuJoCo geom order moved at ") + geom_names[i],
        )

    # Sites USED to be where the two orders diverge — `palm_touch` is declared
    # AFTER the `pinch site` body but belongs to `hand`, so MuJoCo's body sort
    # pulls it ahead of `pinch` and our text walk did not. The parser now
    # groups by body, so this list is BOTH orders and `_our_site_to_mj` is the
    # identity. Note the swap relative to the XML text: `palm_touch` second.
    var site_names = [
        "grasp", "palm_touch", "pinch",
        "thumb_touch", "thumbtip_touch",
        "finger_touch", "fingertip_touch",
        "ball", "target_ball",
    ]
    for i in range(len(site_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, site_names[i]))
            == _our_site_to_mj(i),
            String("site order moved at ") + site_names[i]
            + " — _our_site_to_mj is stale",
        )

    # And now OUR order, against the same reference indices.
    var mf = _build()
    var bref = mj.body_parentid.tolist()
    for b in range(NBODY):
        assert_true(
            Int(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            == Int(py=bref[b]),
            String("body_parentid mismatch on body ") + String(b),
        )
    var gref = mj.geom_bodyid.tolist()
    for g in range(NGEOM):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
            == Int(py=gref[g]),
            String("geom_bodyid mismatch on geom ") + String(g),
        )
    var sref = mj.site_bodyid.tolist()
    for s in range(NSITE):
        assert_true(
            Int(mf.sites.data[s * MODEL_SITE_SIZE + SITE_IDX_BODY])
            == Int(py=sref[_our_site_to_mj(s)]),
            String("site_bodyid mismatch on site ") + String(s),
        )
    var jref = mj.jnt_bodyid.tolist()
    for j in range(NJOINT):
        assert_true(
            Int(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_BODY_ID])
            == Int(py=jref[j]),
            String("jnt_bodyid mismatch on joint ") + String(j),
        )


def test_manipulator_masses_match_mujoco() raises:
    """`body_mass`, including the `<inertial>`-only `pinch site` body.

    `pinch site` has NO geom, so its mass comes entirely from a CHILD
    `<inertial>` element — which neither parser read before this port. Without
    it the body's mass is whatever the default is rather than 1e-6, shifting
    the hand's composite inertia by ~6e-5 relative. That is invisible to the
    eye and nowhere near this file's tolerance, which is the point.
    """
    var mj = _ref()
    var mf = _build()
    var mref = mj.body_mass.tolist()
    var worst = Float64(0)
    for b in range(NBODY):
        var ours = Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var r = Float64(py=mref[b])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(
            rel <= TOL_MODEL,
            String("body_mass mismatch on body ") + String(b),
        )
    print("  worst body_mass rel err =", worst)


def test_manipulator_joints_match_mujoco() raises:
    """Joint type / limits / damping / stiffness, and `qpos0`.

    `ball_x` and `ball_z` carry `ref=".4"`, so `qpos0` is NOT all-zeros. Per
    bug 18 a mis-scaled `ref` skews every constraint inverse weight, since
    those are built at qpos0 — and `ref` on a SLIDE joint (a length, not an
    angle, so no degree conversion) had never been exercised before.
    """
    var mj = _ref()
    var mf = _build()

    var q0 = mj.qpos0.tolist()
    var jt = mj.jnt_type.tolist()
    var jlim = mj.jnt_limited.tolist()
    var jrange = mj.jnt_range.tolist()
    var jstiff = mj.jnt_stiffness.tolist()
    var jqposadr = mj.jnt_qposadr.tolist()
    var jdofadr = mj.jnt_dofadr.tolist()
    var ddamp = mj.dof_damping.tolist()
    for j in range(NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        # MuJoCo jnt_type: 0 free, 1 ball, 2 slide, 3 hinge.
        assert_true(
            Int(mf.joints.data[jo + JOINT_IDX_TYPE]) == Int(py=jt[j]),
            String("jnt_type mismatch on joint ") + String(j),
        )
        assert_true(
            Int(mf.joints.data[jo + JOINT_IDX_QPOS_ADR])
            == Int(py=jqposadr[j]),
            String("jnt_qposadr mismatch on joint ") + String(j),
        )
        # We store `ref` per JOINT (JOINT_IDX_QPOS0), MuJoCo per QPOS slot.
        assert_true(
            abs(
                Float64(mf.joints.data[jo + JOINT_IDX_QPOS0])
                - Float64(py=q0[Int(py=jqposadr[j])])
            )
            <= 1e-12,
            String("qpos0 mismatch on joint ") + String(j),
        )
        # We carry no `limited` column: an unlimited joint gets a range wider
        # than +-1e9, which is what every limit builder tests.
        var rmin = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MIN])
        var rmax = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MAX])
        var ours_limited = rmin >= -1e9 and rmax <= 1e9
        assert_true(
            ours_limited == (Int(py=jlim[j]) != 0),
            String("jnt_limited mismatch on joint ") + String(j),
        )
        if Int(py=jlim[j]) != 0:
            assert_true(
                abs(rmin - Float64(py=jrange[j][0])) <= 1e-12
                and abs(rmax - Float64(py=jrange[j][1])) <= 1e-12,
                String("jnt_range mismatch on joint ") + String(j),
            )
        assert_true(
            abs(
                Float64(mf.joints.data[jo + JOINT_IDX_STIFFNESS])
                - Float64(py=jstiff[j])
            )
            <= 1e-12,
            String("jnt_stiffness mismatch on joint ") + String(j),
        )
        assert_true(
            abs(
                Float64(mf.joints.data[jo + JOINT_IDX_DAMPING])
                - Float64(py=ddamp[Int(py=jdofadr[j])])
            )
            <= 1e-12,
            String("dof_damping mismatch on joint ") + String(j),
        )


def test_manipulator_sites_match_mujoco() raises:
    """Site body / type / pos / size — and, for the two `euler="0 15 0"`
    zones, ORIENTATION.

    The orientation half is the reason this domain needed a site quaternion at
    all. `thumb_touch` and `finger_touch` come out of `class="hand"` with
    quat [.99144, 0, .13052, 0]; every earlier ported site was axis-aligned or
    a sphere, so substituting the body quat (which `sensors/frame_vel.mojo`
    and `sensors/site_acc.mojo` still do) happened to be exact. Here it is
    wrong by 15 degrees on the two zones that decide whether a grasp is
    detected.

    Both `pos` and `euler` arrive via DEFAULTS, not the site tag, which is a
    second thing no earlier model needed: `full_parser` read site defaults for
    `type` and `size` only.
    """
    var mj = _ref()
    var mf = _build()

    var stype = mj.site_type.tolist()
    var spos = mj.site_pos.tolist()
    var ssize = mj.site_size.tolist()
    var squat = mj.site_quat.tolist()
    var worst_pos = Float64(0)
    var worst_size = Float64(0)
    var worst_quat = Float64(0)
    for s in range(NSITE):
        var so = s * MODEL_SITE_SIZE
        var r = _our_site_to_mj(s)
        assert_true(
            _mj_geom_type(Int(mf.sites.data[so + SITE_IDX_TYPE]))
            == Int(py=stype[r]),
            String("site_type mismatch on site ") + String(s),
        )
        var dx = abs(
            Float64(mf.sites.data[so + SITE_IDX_POS_X]) - Float64(py=spos[r][0])
        )
        var dy = abs(
            Float64(mf.sites.data[so + SITE_IDX_POS_Y]) - Float64(py=spos[r][1])
        )
        var dz = abs(
            Float64(mf.sites.data[so + SITE_IDX_POS_Z]) - Float64(py=spos[r][2])
        )
        var dp = max(dx, max(dy, dz))
        if dp > worst_pos:
            worst_pos = dp
        assert_true(
            dp <= 1e-12, String("site_pos mismatch on site ") + String(s)
        )

        var d0 = abs(
            Float64(mf.sites.data[so + SITE_IDX_SIZE_0])
            - Float64(py=ssize[r][0])
        )
        var d1 = abs(
            Float64(mf.sites.data[so + SITE_IDX_SIZE_1])
            - Float64(py=ssize[r][1])
        )
        var d2 = abs(
            Float64(mf.sites.data[so + SITE_IDX_SIZE_2])
            - Float64(py=ssize[r][2])
        )
        var ds = max(d0, max(d1, d2))
        if ds > worst_size:
            worst_size = ds
        assert_true(
            ds <= 1e-12, String("site_size mismatch on site ") + String(s)
        )

        # Ours is (x, y, z, w); MuJoCo's `site_quat` is (w, x, y, z).
        # Compared up to SIGN — q and -q are the same rotation, and neither
        # side promises a hemisphere.
        var qw = Float64(mf.sites.data[so + SITE_IDX_QUAT_W])
        var qx = Float64(mf.sites.data[so + SITE_IDX_QUAT_X])
        var qy = Float64(mf.sites.data[so + SITE_IDX_QUAT_Y])
        var qz = Float64(mf.sites.data[so + SITE_IDX_QUAT_Z])
        var rw = Float64(py=squat[r][0])
        var rx = Float64(py=squat[r][1])
        var ry = Float64(py=squat[r][2])
        var rz = Float64(py=squat[r][3])
        var sgn = Float64(1.0) if (qw * rw + qx * rx + qy * ry + qz * rz) >= 0 else Float64(-1.0)
        var dq = max(
            max(abs(sgn * qw - rw), abs(sgn * qx - rx)),
            max(abs(sgn * qy - ry), abs(sgn * qz - rz)),
        )
        if dq > worst_quat:
            worst_quat = dq
        assert_true(
            dq <= 1e-12, String("site_quat mismatch on site ") + String(s)
        )
    print(
        "  worst site pos err =", worst_pos,
        " size err =", worst_size,
        " quat err =", worst_quat,
    )


def test_manipulator_geoms_match_mujoco() raises:
    """Geom type + collision masks.

    The two `zaxis`-rotated wall planes and the `background` plane are all
    `contype=1`, so they COLLIDE — a non-vertical plane is not a decoration
    here. `arm_root` (a cylinder) and `target_ball` are the only two geoms
    with collision disabled.
    """
    var mj = _ref()
    var mf = _build()
    var gtype = mj.geom_type.tolist()
    var gct = mj.geom_contype.tolist()
    var gca = mj.geom_conaffinity.tolist()
    for g in range(NGEOM):
        var go = g * MODEL_GEOM_SIZE
        assert_true(
            _mj_geom_type(Int(mf.geoms.data[go + GEOM_IDX_TYPE]))
            == Int(py=gtype[g]),
            String("geom_type mismatch on geom ") + String(g),
        )
        assert_true(
            Int(mf.geoms.data[go + GEOM_IDX_CONTYPE]) == Int(py=gct[g]),
            String("geom_contype mismatch on geom ") + String(g),
        )
        assert_true(
            Int(mf.geoms.data[go + GEOM_IDX_CONAFFINITY]) == Int(py=gca[g]),
            String("geom_conaffinity mismatch on geom ") + String(g),
        )


def test_manipulator_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0` / `tendon_invweight0`.

    Run for every newly ported model since bug 20, and for good reason: both
    bug 20 and bug 26 were silent multipliers living exactly here. This model
    puts all three in play at once — contacts read `body_invweight0`, the
    eight joint limits read `dof_invweight0`, and the `coupling` equality
    reads `tendon_invweight0` (bug 29's slot).
    """
    var mj = _ref()
    var mf = _build()

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var tiw = mj.tendon_invweight0.tolist()
    var worst = Float64(0)
    for b in range(NBODY):
        for k in range(2):
            var ours = Float64(mf.body_invweight0.data[2 * b + k])
            var r = Float64(py=biw[b][k])
            var rel = abs(ours - r) / (1e-15 + abs(r))
            if rel > worst:
                worst = rel
            assert_true(
                rel <= TOL_MODEL,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(NV):
        var ours = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(
            rel <= TOL_MODEL,
            String("dof_invweight0 mismatch on dof ") + String(i),
        )
    for t in range(NTEN):
        var ours = Float64(
            mf.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_INVWEIGHT0]
        )
        var r = Float64(py=tiw[t])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(
            rel <= TOL_MODEL,
            String("tendon_invweight0 mismatch on tendon ") + String(t),
        )
    print("  worst invweight0 rel err =", worst)


# ── physics: qacc vs MuJoCo, split by which constraint rows are live ────────
#
# This is the layer the domain was scoped around. Our ELLIPTIC path solves
# joint limits, equality constraints and fixed tendons SEQUENTIALLY, after the
# Newton contact core; the PYRAMIDAL path moved those rows INSIDE the Newton
# system (task #22) because the split was costing standing quadruped 45% of
# its qacc. manipulator is the first ported model that is elliptic AND carries
# a load-bearing `<equality><tendon>` — the `coupling` row that keeps `finger`
# and `thumb` symmetric under the single `grasp` actuator.
#
# An AGGREGATE error would not answer the question. A sequential split is
# exact when the row sets do not share dofs and degrades only when they
# couple, so the measurement has to be bucketed by what MuJoCo has live:
#
#   A  equality only          — the coupling row alone, nothing to couple with
#   B  equality + limit       — shared dofs, no contacts
#   C  equality + contact     — the grasp, which is the task
#   D  equality + limit + contact
#
# If A is exact and C is not, the split is the cause and the fix is the one
# task #22 applied to the pyramidal path. If A is already off, the equality
# row itself is wrong and the split is a red herring. Reported per bucket for
# exactly that reason.
#
# The MuJoCo side is built from OUR xml string, not `manipulator_ref`. That is
# deliberate: the model constants are already gated against the real reference
# above, so feeding both engines the identical model isolates the SOLVER.

comptime MAXC: Int = M.MAX_CONTACTS
comptime NEQ: Int = M.MAX_EQUALITY
comptime NEXCL: Int = M.NEXCLUDE
comptime NA: Int = M.NA

comptime Integ = EulerIntegrator[
    DTYPE, NQ, NV, NBODY, NJOINT, MAXC, NGEOM, NEQ, NTEN, NSITE,
    NEXCL, 0, M.CONE_TYPE, 1, SOLVER="newton",
]
comptime Dat = Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1]
comptime Mod = Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]

# Ball world position when the arm sits at the `C`/`D` pose — the `grasp`
# site's (x, z). `ball_x`/`ball_z` carry `ref=".4"` matching the body's own
# `pos`, so the joint value IS the world coordinate. Pinned rather than
# recomputed: `test_manipulator_grasp_site_matches_mujoco` proves both engines
# put the site here.
comptime BALL_GRASP_X: Float64 = -0.0023764341177963649
comptime BALL_GRASP_Z: Float64 = 0.90261612280698533
# Somewhere the hand cannot reach, for the contact-free buckets.
comptime BALL_FAR_X: Float64 = -0.3
comptime BALL_FAR_Z: Float64 = 0.55


def _pose_state(
    arm0: Float64, arm1: Float64, arm2: Float64, arm3: Float64,
    thumb: Float64, tip: Float64,
    ball_x: Float64, ball_z: Float64,
    moving: Bool,
) -> List[Float64]:
    """One (qpos, qvel) state, as a flat NQ+NV list.

    The hand is SYMMETRIC (`finger` = `thumb`, `fingertip` = `thumbtip`),
    which is both what `Bring.initialize_episode` sets and what the `coupling`
    equality holds. Starting off-symmetry would put the equality row far from
    its setpoint and turn the gate into a measurement of how two solvers cope
    with a stiff row rather than of the split under test.
    """
    var s = List[Float64]()
    for _ in range(NQ + NV):
        s.append(0.0)
    s[0] = arm0
    s[1] = arm1
    s[2] = arm2
    s[3] = arm3
    s[4] = thumb  # thumb
    s[5] = tip  # thumbtip
    s[6] = thumb  # finger
    s[7] = tip  # fingertip
    s[8] = ball_x
    s[9] = ball_z
    if moving:
        s[NQ + 0] = 0.5
        s[NQ + 1] = -0.3
        s[NQ + 2] = 0.7
        s[NQ + 3] = 0.2
        s[NQ + 4] = 0.1
        s[NQ + 6] = 0.1
        s[NQ + 8] = 0.2
        s[NQ + 9] = -0.1
        s[NQ + 10] = 0.3
    return s^


def _our_qacc(
    state: List[Float64], ctrl: List[Float64]
) raises -> List[Float64]:
    """`qacc` after one constrained solve at `state`, our engine.

    Returns NV accelerations followed by OUR contact count, which the caller
    diffs against MuJoCo's `ncon`. Bucketing by MuJoCo's live rows while our
    own row set differs would classify a pose as "equality only" and then
    measure a phantom contact — the classification has to be checked, not
    assumed.
    """
    var sf = M.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    M.init_fields[DTYPE, 0](ctx, mf)
    M.reset_data(sf, d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ + i])
        d.qfrc.data[i] = Scalar[DTYPE](0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(NA if NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    var integ = Integ()
    M.apply_actions(sf, d, ctrl, act)
    integ.step["cpu"](d, mf)
    var out = List[Float64]()
    for i in range(NV):
        out.append(Float64(integ.scratch.qacc_constrained.data[i]))
    out.append(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    return out^


def _mj_at(
    state: List[Float64], ctrl: List[Float64]
) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ):
        dat.qpos[i] = state[i]
    for i in range(NV):
        dat.qvel[i] = state[NQ + i]
    for i in range(NACT):
        dat.ctrl[i] = ctrl[i]
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def _zero_ctrl() -> List[Float64]:
    var c = List[Float64]()
    for _ in range(NACT):
        c.append(0.0)
    return c^


def test_manipulator_grasp_site_matches_mujoco() raises:
    """The `grasp` site lands where `BALL_GRASP_*` says, in both engines.

    The physics buckets below place the ball by hardcoded coordinates; this is
    what makes that legitimate rather than a magic number.
    """
    var sf = M.make_spec_fields[DTYPE]()
    var state = _pose_state(
        0.0, 0.3, -0.6, 0.2, 0.45, 0.0, BALL_FAR_X, BALL_FAR_Z, False
    )
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    M.init_fields[DTYPE, 0](ctx, mf)
    M.reset_data(sf, d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    var act = List[Scalar[DTYPE]]()
    for _ in range(NA if NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    var integ = Integ()
    M.apply_actions(sf, d, _zero_ctrl(), act)
    integ.step["cpu"](d, mf)

    var mj = _mj_at(state, _zero_ctrl())
    var dat = mj[2]
    # OUR grasp site is index 0 and MuJoCo's is too (`_our_site_to_mj(0) == 0`).
    var ex = abs(Float64(d.site_xpos.data[0]) - Float64(py=dat.site_xpos[0][0]))
    var ez = abs(Float64(d.site_xpos.data[2]) - Float64(py=dat.site_xpos[0][2]))
    print("  grasp site err x =", ex, " z =", ez)
    assert_true(ex < 1e-9 and ez < 1e-9, "grasp site diverges from MuJoCo")
    assert_true(
        abs(Float64(py=dat.site_xpos[0][0]) - BALL_GRASP_X) < 1e-12
        and abs(Float64(py=dat.site_xpos[0][2]) - BALL_GRASP_Z) < 1e-12,
        "BALL_GRASP_* is stale — the grasp site moved",
    )


def test_manipulator_qacc_by_constraint_bucket() raises:
    """`qacc` vs MuJoCo, bucketed by which constraint rows MuJoCo has live.

    See the block comment above for why the split, not the aggregate, is the
    measurement.
    """
    print("--- manipulator: qacc vs MuJoCo, by live constraint rows ---")

    var tags = List[String]()
    var states = List[List[Float64]]()
    var ctrls = List[List[Float64]]()

    # A — equality only.
    tags.append(String("A1 rest, ball far"))
    states.append(_pose_state(0, 0, 0, 0, 0.0, 0.0, BALL_FAR_X, BALL_FAR_Z, False))
    ctrls.append(_zero_ctrl())
    tags.append(String("A2 bent, ball far"))
    states.append(
        _pose_state(0.3, 0.5, -0.9, 0.4, 0.2, 0.0, BALL_FAR_X, BALL_FAR_Z, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("A3 bent, moving, driven"))
    states.append(
        _pose_state(0.3, 0.5, -0.9, 0.4, 0.2, 0.0, BALL_FAR_X, BALL_FAR_Z, True)
    )
    ctrls.append([0.4, -0.3, 0.6, -0.2, 0.5])

    # B — equality + joint limit (the fingertip's `range="-40 20"`), no contact.
    tags.append(String("B1 fingertip limit"))
    states.append(_pose_state(0, 0, 0, 0, 0.0, 0.36, BALL_FAR_X, BALL_FAR_Z, False))
    ctrls.append(_zero_ctrl())
    tags.append(String("B2 fingertip limit, deeper"))
    states.append(_pose_state(0, 0, 0, 0, 0.0, 0.45, BALL_FAR_X, BALL_FAR_Z, False))
    ctrls.append(_zero_ctrl())

    # C — equality + contact: the ball held between the finger and the thumb.
    tags.append(String("C1 grasp, light contact"))
    states.append(
        _pose_state(0, 0.3, -0.6, 0.2, 0.45, 0.0, BALL_GRASP_X, BALL_GRASP_Z, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("C2 grasp"))
    states.append(
        _pose_state(0, 0.3, -0.6, 0.2, 0.48, 0.0, BALL_GRASP_X, BALL_GRASP_Z, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("C3 grasp, driven"))
    states.append(
        _pose_state(0, 0.3, -0.6, 0.2, 0.50, 0.0, BALL_GRASP_X, BALL_GRASP_Z, False)
    )
    ctrls.append([0.2, 0.1, -0.3, 0.1, 0.8])

    # D — all three.
    tags.append(String("D1 closed hand, self-contact + limit"))
    states.append(_pose_state(0, 0, 0, 0, 1.05, 0.0, BALL_FAR_X, BALL_FAR_Z, False))
    ctrls.append(_zero_ctrl())
    tags.append(String("D2 grasp + limit, driven"))
    states.append(
        _pose_state(0, 0.3, -0.6, 0.2, 0.60, 0.36, BALL_GRASP_X, BALL_GRASP_Z, False)
    )
    ctrls.append([0.0, 0.0, 0.0, 0.0, 0.6])

    var worst_by_bucket = List[Float64]()
    var count_by_bucket = List[Int]()
    for _ in range(4):
        worst_by_bucket.append(0.0)
        count_by_bucket.append(0)

    for p in range(len(tags)):
        var mj = _mj_at(states[p], ctrls[p])
        var dat = mj[2]

        # Classify by MuJoCo's own row types: 0 EQUALITY, 3 LIMIT_JOINT,
        # 7 CONTACT_ELLIPTIC.
        var n_eq = 0
        var n_lim = 0
        var n_con = 0
        var nefc = Int(py=dat.nefc)
        for i in range(nefc):
            var t = Int(py=dat.efc_type[i])
            if t == 0:
                n_eq += 1
            elif t == 3:
                n_lim += 1
            elif t == 7:
                n_con += 1
        assert_true(
            n_eq == 1,
            String("pose ") + tags[p] + " lost the coupling equality row",
        )
        var bucket = 0
        if n_lim > 0 and n_con > 0:
            bucket = 3
        elif n_con > 0:
            bucket = 2
        elif n_lim > 0:
            bucket = 1

        var ours = _our_qacc(states[p], ctrls[p])
        var our_ncon = Int(ours[NV])
        var mj_ncon = Int(py=dat.ncon)
        var scale = Float64(1.0)
        for i in range(NV):
            var a = abs(Float64(py=dat.qacc[i]))
            if a > scale:
                scale = a
        var worst = Float64(0)
        var worst_dof = 0
        for i in range(NV):
            var e = abs(ours[i] - Float64(py=dat.qacc[i]))
            if e > worst:
                worst = e
                worst_dof = i
        var rel = worst / scale
        if rel > worst_by_bucket[bucket]:
            worst_by_bucket[bucket] = rel
        count_by_bucket[bucket] += 1
        print(
            "  ", tags[p],
            " | eq", n_eq, "lim", n_lim, "con", n_con,
            "| ncon mj", mj_ncon, "ours", our_ncon,
            "| |qacc|max", scale, " worst dof", worst_dof, " rel", rel,
        )
        assert_true(
            our_ncon == mj_ncon,
            String("pose ") + tags[p] + ": we found " + String(our_ncon)
            + " contacts, MuJoCo " + String(mj_ncon)
            + " — the bucket label is a fiction and the qacc comparison is"
            " measuring a different constraint problem",
        )

    var names = [
        String("A equality only"),
        String("B equality+limit"),
        String("C equality+contact"),
        String("D equality+limit+contact"),
    ]
    print("  --- worst relative qacc error per bucket ---")
    for b in range(4):
        print("   ", names[b], " n =", count_by_bucket[b],
              " worst rel =", worst_by_bucket[b])
        assert_true(
            count_by_bucket[b] > 0,
            String("bucket ") + names[b] + " is EMPTY — the pose table no"
            " longer exercises it, so this file stopped measuring the split",
        )

    # Gates set from the measured values, per bucket, so a regression in the
    # COUPLED buckets cannot hide behind the uncoupled ones.
    assert_true(
        worst_by_bucket[0] <= TOL_QACC_UNCOUPLED,
        "equality-only qacc diverges — the coupling row itself is wrong,"
        " which the sequential split cannot explain",
    )
    assert_true(
        worst_by_bucket[1] <= TOL_QACC_COUPLED, "equality+limit qacc diverges"
    )
    assert_true(
        worst_by_bucket[2] <= TOL_QACC_COUPLED,
        "equality+contact qacc diverges — this is the grasp, i.e. the task",
    )
    assert_true(
        worst_by_bucket[3] <= TOL_QACC_COUPLED,
        "equality+limit+contact qacc diverges",
    )



# ── observation + reward ────────────────────────────────────────────────────

def _set_state_and_fk(
    mut d: Dat, mut mf: Mod, mut integ: Integ,
    state: List[Float64], ctrl: List[Float64],
) raises:
    """Both engines at the same state, ours stepped once so contacts and
    `site_xpos` are live (the touch entries read post-solve contact forces)."""
    var sf = M.make_spec_fields[DTYPE]()
    M.reset_data(sf, d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ + i])
        d.qfrc.data[i] = Scalar[DTYPE](0)
    # The target is a MOCAP body here and a static one in the reference, so
    # pin ours to the reference's XML pose rather than leaving it wherever
    # `reset_data` put it.
    d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](0.4)
    d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](0.001)
    d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](0.4)
    d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
    d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
    d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
    d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)
    # FK SKIPS mocap bodies, so their world pose has to be preset from
    # `mocap_pos`/`mocap_quat`. The env facade does this in
    # `_sync_mocap_to_fields`; this test drives `integ.step` directly and so
    # must do it itself, or the target sits at the origin and every term that
    # reads it silently reports a 40 cm error.
    for k in range(3):
        var pv = d.mocap_pos.data[TARGET_BODY_IDX * 3 + k]
        d.xpos.data[TARGET_BODY_IDX * 3 + k] = pv
        d.xipos.data[TARGET_BODY_IDX * 3 + k] = pv
    for k in range(4):
        d.xquat.data[TARGET_BODY_IDX * 4 + k] = d.mocap_quat.data[
            TARGET_BODY_IDX * 4 + k
        ]
    var act = List[Scalar[DTYPE]]()
    for _ in range(NA if NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    M.apply_actions(sf, d, ctrl, act)
    integ.step["cpu"](d, mf)
    # `integ.step` INTEGRATES. `d.contacts` and `d.site_xpos` were computed at
    # the PRE-integration pose, so restoring qpos/qvel here puts every
    # observation term back on the same state MuJoCo's `mj_forward` saw.
    # Without this the two engines are one Euler step apart and the diff reads
    # as a small, plausible, wrong number rather than an obvious one.
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ + i])


def test_manipulator_observation_matches_mujoco() raises:
    """All 44 observation entries, at a pose that makes every term non-trivial.

    The pose is deliberately ASYMMETRIC in the hand. `_ARM_JOINTS` lists
    finger/fingertip BEFORE thumb/thumbtip while the model declares the thumb
    chain first, so the observation carries a permutation of the model's joint
    order — and a symmetric hand would hide a wrong permutation completely.

    It also holds the ball, so the five `touch` entries are non-zero. Those are
    the first numeric gate on the BOX touch zones: `sensors/touch.mojo` grew a
    `ray_box` branch for this domain and until now nothing compared it to
    MuJoCo's `sensordata`.
    """
    var ctx = DeviceContext()
    var mf = _build()
    var d = Dat()
    var integ = Integ()
    # C-bucket grasp pose, with the two finger chains driven APART so the
    # observation permutation is observable.
    # Grasping AND asymmetric: thumb .60 / finger .66, thumbtip -.30 /
    # fingertip -.18. Chosen by sweeping MuJoCo for a pose that both loads the
    # touch sensors and breaks the hand's symmetry — a symmetric hand would
    # hide a wrong `_ARM_JOINTS` permutation, and a non-touching one would make
    # the five touch entries a 0 == 0 comparison.
    var state = _pose_state(
        0.0, 0.3, -0.6, 0.2, 0.60, -0.30, BALL_GRASP_X, BALL_GRASP_Z, False
    )
    state[6] = 0.66  # finger    != thumb
    state[7] = -0.18  # fingertip != thumbtip
    var ctrl = _zero_ctrl()
    _set_state_and_fk(d, mf, integ, state, ctrl)

    var mj = _mj_at(state, ctrl)
    var mujoco = mj[0]
    var m = mj[1]
    var dat = mj[2]
    var builder = Python.import_module("manipulator_ref")
    var ref_obs = builder.observation(m, dat)

    var obs = List[Scalar[DTYPE]]()
    _ = DMManipulatorBringBallConfig.custom_extract_obs_cpu[
        DTYPE, NQ, NV, NBODY, MAXC, NSITE
    ](
        d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
        List[Scalar[DTYPE]](), obs,
    )
    assert_true(
        len(obs) == 44,
        String("obs dim ") + String(len(obs)) + " != 44",
    )

    var names = [
        String("arm_pos"), String("arm_vel"), String("touch"),
        String("hand_pos"), String("object_pos"), String("object_vel"),
        String("target_pos"),
    ]
    var bounds = [0, 16, 24, 29, 33, 37, 40, 44]
    var worst_all = Float64(0)
    for blk in range(7):
        var worst = Float64(0)
        var wi = 0
        for i in range(bounds[blk], bounds[blk + 1]):
            var e = abs(Float64(obs[i]) - Float64(py=ref_obs[i]))
            if e > worst:
                worst = e
                wi = i
        if worst > worst_all:
            worst_all = worst
        print("   ", names[blk], " worst |d| =", worst, " at", wi)
        var tol = TOL_TOUCH if blk == 2 else TOL_OBS
        assert_true(
            worst <= tol,
            String("observation block ") + names[blk] + " diverges",
        )

    # Non-vacuity: the touch block must actually be loaded, or it gates nothing.
    var touch_sum = Float64(0)
    for i in range(24, 29):
        touch_sum += Float64(py=ref_obs[i])
    print("  MuJoCo touch block sum =", touch_sum)
    assert_true(
        touch_sum > 0.0,
        "MuJoCo's touch sensors read zero at this pose, so the five touch"
        " entries — and the BOX zone code behind them — are still ungated",
    )
    print("  worst |d obs| =", worst_all)


def test_manipulator_reward_matches_mujoco() raises:
    """`_ball_reward` across the whole useful range of the tolerance curve.

    The gaussian sigmoid decays superexponentially — 1.0 inside 1 cm, 5.6e-21
    at 10 cm — so a single distance would gate almost nothing. These four span
    saturated, mid-curve and far.
    """
    var ctx = DeviceContext()
    var mf = _build()
    var integ = Integ()
    var offsets = [0.0, 0.015, 0.03, 0.12]
    var worst = Float64(0)
    for k in range(len(offsets)):
        var d = Dat()
        # Ball parked away from the arm at the target's x, offset in z.
        var state = _pose_state(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4 + offsets[k], False
        )
        _set_state_and_fk(d, mf, integ, state, _zero_ctrl())
        var mj = _mj_at(state, _zero_ctrl())
        var ref_r = Float64(
            py=Python.import_module("manipulator_ref").reward(mj[1], mj[2])
        )
        var got = DMManipulatorBringBallConfig.compute_reward_and_done_cpu[
            DTYPE, NQ, NV, NBODY, MAXC, NSITE
        ](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            Scalar[DTYPE](0), _zero_ctrl(), 0, 1,
        )
        var e = abs(Float64(got[0]) - ref_r)
        if e > worst:
            worst = e
        print("    dz =", offsets[k], " ours =", got[0], " MuJoCo =", ref_r)
        assert_true(
            not got[1], "dm_control tasks never terminate early"
        )
    print("  worst |d reward| =", worst)
    assert_true(worst <= TOL_OBS, "reward diverges from MuJoCo")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
