"""dm_control `stacker` — parity against MuJoCo, both tasks.

    stack_2   2 boxes + target      obs 49
    stack_4   4 boxes + target      obs 63

WHAT IS GATED HERE, AND WHAT IS NOT.

The MODEL, the KINEMATICS, the OBSERVATION and the REWARD are gated exactly, on
both models. So is the CONSTRAINT SOLVE — but only over the poses whose contact
ROW SET our narrow phase can reproduce, which for this domain is a real
restriction rather than a formality:

⚠ THE BOX-FACE RESTRICTION IS GONE, and with it the reason this domain was only
half gateable. Our narrow phase used to emit ONE contact per colliding geom
*pair* where MuJoCo emits up to four for a box on a plane and up to six for two
boxes meeting face to face — and a box supported by a single corner point cannot
be in equilibrium, it pivots, so for stacker that was not a numerical gap, it was
the domain. Every box contact is now byte-identical to MuJoCo — task #42 closed
in four pieces: box/plane, the box/box FACE manifold, the box/box EDGE-EDGE
manifold, and capsule/box's second point — and
`test_stacker_qacc_by_constraint_bucket` gates the stacking poses themselves —
`C9 cube stacked on a cube`, `C10 side by side` — at 2.9e-10.

The manifold gates live in `tests/physics3d/test_box_box_sweep.mojo` (both
box/box axis kinds, 571 points over 217 poses, point for point) and
`tests/physics3d/test_capsule_box_sweep.mojo` (the capsule manifold, including
fixed lying-along-a-face poses).

⚠ `ncon` AGREEING IS NOT THE ROW SET AGREEING, and the converse trap is live
here too: our count can equal MuJoCo's PAIR count while being four times short
of its CONTACT count. Every contact assertion below compares one of those two
explicitly and names which.

THE TWO PERMUTATIONS. `_ARM_JOINTS` lists finger/fingertip before thumb/thumbtip
while the model declares the thumb chain first; and `box_joint_names` is built
`for dim in 'xyz'` while the model declares each box's joints x, z, y. A
symmetric hand hides the first and a box moving along one axis hides the second,
so the observation test drives an asymmetric hand and gives every box three
distinct velocity components.
"""

from std.math import abs, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.stacker import (
    DMStacker2Model as M2,
    DMStacker4Model as M4,
    DMStacker2Config as CFG2,
    DMStacker4Config as CFG4,
    box_body_idx,
    box_site_idx,
    box_vel_qadr,
    target_body_idx,
    target_site_idx,
    stacker_obs_dim,
    BOX_QADR_0,
    BOX_SIZE,
)
from mojo_rl.envs.dm_control.planar_arm import arm_joint_obs_order
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    MODEL_BODY_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_MASS,
    MODEL_JOINT_SIZE,
    JOINT_IDX_QPOS0,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_TYPE,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    MODEL_TENDON_SIZE,
    TENDON_IDX_INVWEIGHT0,
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

# ── stack_2 ─────────────────────────────────────────────────────────────────
comptime NQ2: Int = M2.NQ  # 14
comptime NV2: Int = M2.NV  # 14
comptime NBODY2: Int = M2.NBODY  # 13
comptime NJOINT2: Int = M2.NJOINT  # 14
comptime NGEOM2: Int = M2.NGEOM  # 22
comptime NSITE2: Int = M2.NSITE  # 10
comptime NACT2: Int = M2.nact  # 5
comptime NTEN2: Int = M2.MAX_TENDON  # 2
comptime MAXC2: Int = M2.MAX_CONTACTS
comptime NEQ2: Int = M2.MAX_EQUALITY
comptime NEXCL2: Int = M2.NEXCLUDE
comptime NA2: Int = M2.NA

# ── stack_4 ─────────────────────────────────────────────────────────────────
comptime NQ4: Int = M4.NQ  # 20
comptime NV4: Int = M4.NV  # 20
comptime NBODY4: Int = M4.NBODY  # 15
comptime NJOINT4: Int = M4.NJOINT  # 20
comptime NGEOM4: Int = M4.NGEOM  # 24
comptime NSITE4: Int = M4.NSITE  # 12
comptime NACT4: Int = M4.nact  # 5
comptime NTEN4: Int = M4.MAX_TENDON  # 2
comptime MAXC4: Int = M4.MAX_CONTACTS
comptime NEQ4: Int = M4.MAX_EQUALITY
comptime NEXCL4: Int = M4.NEXCLUDE
comptime NA4: Int = M4.NA

# Model constants are exact rational arithmetic on both sides up to the inertia
# integrals, so anything above ~1e-12 is a real disagreement.
comptime TOL_MODEL: Float64 = 1e-9

# qacc gates, per bucket, so a regression in a COUPLED bucket cannot hide behind
# the uncoupled one.
comptime TOL_QACC_UNCOUPLED: Float64 = 1e-8
comptime TOL_QACC_COUPLED: Float64 = 1e-8
# Observation and reward are pure readbacks of state the physics layer already
# gates, so they sit at the FK floor.
comptime TOL_OBS: Float64 = 1e-9
# TOUCH reads POST-SOLVE contact forces, so it COULD inherit the contact
# solve's floor rather than FK's.
#
# ⚠ IT DID NOT — the floor here was our own arithmetic, as on `manipulator`.
# The observable was transcribed as `log(1.0 + f)`, worth up to 1.02e-09
# ABSOLUTE against `np.log1p`, and 43% of this task's non-zero touch readings
# land in [0.05, 0.42] where that form is worst — the highest share of any
# task in the suite. With `dtype_math.log1p_dt` the measured worst touch
# deviation is 3.38e-14.
comptime TOL_TOUCH: Float64 = 1e-11


# OUR site order IS MuJoCo's, as of the element-order fix (2026-08-03).
#
# It used to diverge in the ARM — the same divergence `manipulator` had:
# `palm_touch` is declared AFTER the `pinch site` body but belongs to `hand`,
# so MuJoCo's body sort pulls it ahead of `pinch` while our XML-text walk left
# it behind. This file carried a permutation to paper over that.
#
# It was a bug, not a property. The same ordering permutes JOINTS, and
# `fields_build` derives `qpos_adr`/`dof_adr` as running counters over the
# joint array, so the whole `qpos` layout went with it — which is how
# dm_control's dog exposed it. `full_parser` now groups joints, geoms and
# sites by body id, gated by
# `tests/physics3d/test_element_order_vs_mujoco.mojo`.
#
# Kept as the identity rather than deleted so the call sites still read
# "our index -> MuJoCo's index", and a future divergence has one place to live.
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


def _ref(n_boxes: Int) raises -> PythonObject:
    """The compiled reference `mjModel` for one stacker task."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/dm_control")
    var builder = Python.import_module("stacker_ref")
    return builder.model(n_boxes)


comptime Mod2 = Model[DTYPE, Dims[nv=NV2, nbody=NBODY2, njoint=NJOINT2, ngeom=NGEOM2, nequality=NEQ2, ntendon=NTEN2, nsite=NSITE2, nexclude=NEXCL2, nmesh_verts=0]]
comptime Dat2 = Data[DTYPE, Dims[nq=NQ2, nv=NV2, nbody=NBODY2, max_contacts=MAXC2, nsite=NSITE2], 1]
comptime Integ2 = EulerIntegrator[
    DTYPE, NQ2, NV2, NBODY2, NJOINT2, MAXC2, NGEOM2, NEQ2, NTEN2, NSITE2,
    NEXCL2, 0, M2.CONE_TYPE, 1, SOLVER="newton",
]

comptime Mod4 = Model[DTYPE, Dims[nv=NV4, nbody=NBODY4, njoint=NJOINT4, ngeom=NGEOM4, nequality=NEQ4, ntendon=NTEN4, nsite=NSITE4, nexclude=NEXCL4, nmesh_verts=0]]
comptime Dat4 = Data[DTYPE, Dims[nq=NQ4, nv=NV4, nbody=NBODY4, max_contacts=MAXC4, nsite=NSITE4], 1]
comptime Integ4 = EulerIntegrator[
    DTYPE, NQ4, NV4, NBODY4, NJOINT4, MAXC4, NGEOM4, NEQ4, NTEN4, NSITE4,
    NEXCL4, 0, M4.CONE_TYPE, 1, SOLVER="newton",
]


def _build2() raises -> Mod2:
    var ctx = DeviceContext()
    var mf = Mod2()
    M2.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def _build4() raises -> Mod4:
    var ctx = DeviceContext()
    var mf = Mod4()
    M4.init_fields[DTYPE, 0](ctx, mf)
    return mf^


# ── model parity ────────────────────────────────────────────────────────────


def test_stacker_dims_match_mujoco() raises:
    """Element counts, both models. Cheap, and the first thing a dropped
    `<body>` breaks.

    The whole reason these are two models rather than one flag: deleting box2
    and box3 moves the target body from 14 to 12, its geom from 23 to 21 and its
    site from 11 to 9.
    """
    var mj2 = _ref(2)
    assert_true(Int(py=mj2.nq) == NQ2, "stack_2 nq mismatch")
    assert_true(Int(py=mj2.nv) == NV2, "stack_2 nv mismatch")
    assert_true(Int(py=mj2.nbody) == NBODY2, "stack_2 nbody mismatch")
    assert_true(Int(py=mj2.njnt) == NJOINT2, "stack_2 njnt mismatch")
    assert_true(Int(py=mj2.ngeom) == NGEOM2, "stack_2 ngeom mismatch")
    assert_true(Int(py=mj2.nsite) == NSITE2, "stack_2 nsite mismatch")
    assert_true(Int(py=mj2.nu) == NACT2, "stack_2 nu mismatch")
    assert_true(Int(py=mj2.ntendon) == NTEN2, "stack_2 ntendon mismatch")
    assert_true(Int(py=mj2.neq) == 1, "stack_2 neq mismatch (coupling equality)")
    assert_true(Int(py=mj2.nsensor) == 5, "stack_2 nsensor mismatch")

    var mj4 = _ref(4)
    assert_true(Int(py=mj4.nq) == NQ4, "stack_4 nq mismatch")
    assert_true(Int(py=mj4.nv) == NV4, "stack_4 nv mismatch")
    assert_true(Int(py=mj4.nbody) == NBODY4, "stack_4 nbody mismatch")
    assert_true(Int(py=mj4.njnt) == NJOINT4, "stack_4 njnt mismatch")
    assert_true(Int(py=mj4.ngeom) == NGEOM4, "stack_4 ngeom mismatch")
    assert_true(Int(py=mj4.nsite) == NSITE4, "stack_4 nsite mismatch")

    # cone=1 is mjCONE_ELLIPTIC. The parser does not read the attribute — the
    # cone is a `ModelDefFromXML` parameter — so pin that the reference still
    # asks for the cone we hardcoded.
    assert_true(
        Int(py=mj2.opt.cone) == 1, "reference cone is no longer elliptic"
    )
    assert_true(
        abs(Float64(py=mj2.opt.timestep) - M2.TIMESTEP) < 1e-15,
        "timestep mismatch",
    )

    # The index helpers, against the layout those counts imply.
    assert_true(target_body_idx(2) == 12, "target_body_idx(2) != 12")
    assert_true(target_body_idx(4) == 14, "target_body_idx(4) != 14")
    assert_true(target_site_idx(2) == 9, "target_site_idx(2) != 9")
    assert_true(target_site_idx(4) == 11, "target_site_idx(4) != 11")
    assert_true(
        stacker_obs_dim(2) == 49 and stacker_obs_dim(4) == 63,
        "stacker_obs_dim disagrees with 16+8+5+4+7n+2",
    )


def test_stacker_ordering_matches_mujoco() raises:
    """Body / joint / geom / site ORDER, pinned by name on the MuJoCo side, and
    then OUR order against the same reference indices.

    The boxes are siblings with no sub-bodies, so our text order and MuJoCo's
    body-id sort agree everywhere except the arm's `palm_touch` / `pinch` swap.
    "Happen to agree" is exactly the kind of claim point_mass proved can fail
    silently, so it is pinned for both models.
    """
    var mujoco = Python.import_module("mujoco")

    var arm_bodies: List[String] = [
        String("world"), String("upper_arm"), String("middle_arm"),
        String("lower_arm"), String("hand"), String("pinch site"),
        String("thumb"), String("thumbtip"), String("finger"),
        String("fingertip"),
    ]
    var arm_joints: List[String] = [
        String("arm_root"), String("arm_shoulder"), String("arm_elbow"),
        String("arm_wrist"), String("thumb"), String("thumbtip"),
        String("finger"), String("fingertip"),
    ]
    var arm_geoms: List[String] = [
        String("floor"), String("wall1"), String("wall2"), String("background"),
        String("arm_root"), String("upper_arm"), String("middle_arm"),
        String("lower_arm"), String("hand"), String("palm1"), String("palm2"),
        String("thumb1"), String("thumb2"), String("thumbtip1"),
        String("thumbtip2"), String("finger1"), String("finger2"),
        String("fingertip1"), String("fingertip2"),
    ]
    # ⚠ `palm_touch` is SECOND here, not third as it appears in the XML text:
    # it belongs to `hand` and MuJoCo groups by body, so it precedes `pinch`
    # (which sits on the nested `pinch site` body). Our parser now agrees.
    var arm_sites: List[String] = [
        String("grasp"), String("palm_touch"), String("pinch"),
        String("thumb_touch"), String("thumbtip_touch"),
        String("finger_touch"), String("fingertip_touch"),
    ]

    for n in [2, 4]:
        var mj = _ref(n)
        var tag = String("stack_") + String(n) + ": "

        var body_names = arm_bodies.copy()
        for b in range(n):
            body_names.append(String("box") + String(b))
        body_names.append(String("target"))
        for i in range(len(body_names)):
            assert_true(
                Int(
                    py=mujoco.mj_name2id(
                        mj, mujoco.mjtObj.mjOBJ_BODY, body_names[i]
                    )
                )
                == i,
                tag + "MuJoCo body order moved at " + body_names[i],
            )

        var joint_names = arm_joints.copy()
        for b in range(n):
            joint_names.append(String("box") + String(b) + "_x")
            joint_names.append(String("box") + String(b) + "_z")
            joint_names.append(String("box") + String(b) + "_y")
        for i in range(len(joint_names)):
            assert_true(
                Int(
                    py=mujoco.mj_name2id(
                        mj, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i]
                    )
                )
                == i,
                tag + "MuJoCo joint order moved at " + joint_names[i],
            )

        var geom_names = arm_geoms.copy()
        for b in range(n):
            geom_names.append(String("box") + String(b))
        geom_names.append(String("target"))
        for i in range(len(geom_names)):
            assert_true(
                Int(
                    py=mujoco.mj_name2id(
                        mj, mujoco.mjtObj.mjOBJ_GEOM, geom_names[i]
                    )
                )
                == i,
                tag + "MuJoCo geom order moved at " + geom_names[i],
            )

        var site_names = arm_sites.copy()
        for b in range(n):
            site_names.append(String("box") + String(b))
        site_names.append(String("target"))
        for i in range(len(site_names)):
            assert_true(
                Int(
                    py=mujoco.mj_name2id(
                        mj, mujoco.mjtObj.mjOBJ_SITE, site_names[i]
                    )
                )
                == _our_site_to_mj(i),
                tag + "site order moved at " + site_names[i]
                + " — _our_site_to_mj is stale",
            )

        # The index helpers the config reads, against those same names. A helper
        # that disagrees with the model is a silently wrong reward.
        for b in range(n):
            assert_true(
                box_body_idx(b) == 10 + b,
                tag + "box_body_idx disagrees with the body order",
            )
            assert_true(
                box_site_idx(b) == 7 + b,
                tag + "box_site_idx disagrees with the site order",
            )
        assert_true(
            target_body_idx(n) == 10 + n and target_site_idx(n) == 7 + n,
            tag + "target index helpers disagree with the order above",
        )

        # `box_vel_qadr` against the joint names it claims to permute: entry 1
        # of the observed triple must be the `_y` HINGE, which the model
        # declares THIRD.
        for b in range(n):
            var base = 8 + 3 * b
            assert_true(
                box_vel_qadr(b, 0) == base + 0
                and box_vel_qadr(b, 1) == base + 2
                and box_vel_qadr(b, 2) == base + 1,
                tag + "box_vel_qadr is not the 'xyz'-over-x,z,y permutation",
            )

    # And now OUR record, against the reference indices, for both models.
    var mj2 = _ref(2)
    var mf2 = _build2()
    var bref2 = mj2.body_parentid.tolist()
    for b in range(NBODY2):
        assert_true(
            Int(mf2.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            == Int(py=bref2[b]),
            String("stack_2 body_parentid mismatch on body ") + String(b),
        )
    var gref2 = mj2.geom_bodyid.tolist()
    for g in range(NGEOM2):
        assert_true(
            Int(mf2.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
            == Int(py=gref2[g]),
            String("stack_2 geom_bodyid mismatch on geom ") + String(g),
        )
    var sref2 = mj2.site_bodyid.tolist()
    for s in range(NSITE2):
        assert_true(
            Int(mf2.sites.data[s * MODEL_SITE_SIZE + SITE_IDX_BODY])
            == Int(py=sref2[_our_site_to_mj(s)]),
            String("stack_2 site_bodyid mismatch on site ") + String(s),
        )

    var mj4 = _ref(4)
    var mf4 = _build4()
    var bref4 = mj4.body_parentid.tolist()
    for b in range(NBODY4):
        assert_true(
            Int(mf4.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            == Int(py=bref4[b]),
            String("stack_4 body_parentid mismatch on body ") + String(b),
        )
    var gref4 = mj4.geom_bodyid.tolist()
    for g in range(NGEOM4):
        assert_true(
            Int(mf4.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
            == Int(py=gref4[g]),
            String("stack_4 geom_bodyid mismatch on geom ") + String(g),
        )
    var sref4 = mj4.site_bodyid.tolist()
    for s in range(NSITE4):
        assert_true(
            Int(mf4.sites.data[s * MODEL_SITE_SIZE + SITE_IDX_BODY])
            == Int(py=sref4[_our_site_to_mj(s)]),
            String("stack_4 site_bodyid mismatch on site ") + String(s),
        )


def test_stacker_model_constants_match_mujoco() raises:
    """`body_mass`, `qpos0`, geom types/masks, site types/pos, and all three
    `invweight0` tables — on stack_4, which contains stack_2's tables entry for
    entry plus two more boxes.

    The FOUR DIFFERENT slide `ref` values (.5, .4, .3, .2 in x, all .4 in z) are
    the reason `qpos0` is checked value by value rather than for shape. Per bug
    18 a mis-scaled `ref` skews every constraint inverse weight, since those are
    built at qpos0, and a copy-paste that left all four boxes at `ref=".5"`
    would move nothing visible in the starting pose.
    """
    var mj = _ref(4)
    var mf = _build4()

    var mref = mj.body_mass.tolist()
    var worst_mass = Float64(0)
    for b in range(NBODY4):
        var ours = Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var r = Float64(py=mref[b])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst_mass:
            worst_mass = rel
        assert_true(
            rel <= TOL_MODEL, String("body_mass mismatch on body ") + String(b)
        )

    var q0 = mj.qpos0.tolist()
    var jqposadr = mj.jnt_qposadr.tolist()
    for j in range(NJOINT4):
        assert_true(
            abs(
                Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS0])
                - Float64(py=q0[Int(py=jqposadr[j])])
            )
            <= 1e-12,
            String("qpos0 mismatch on joint ") + String(j),
        )
    # Non-vacuity: the four boxes' x references must be four DIFFERENT values.
    var xrefs = [0.5, 0.4, 0.3, 0.2]
    for b in range(4):
        assert_true(
            abs(Float64(py=q0[8 + 3 * b]) - xrefs[b]) < 1e-15,
            String("box") + String(b) + "_x qpos0 is not " + String(xrefs[b])
            + " — the four slide `ref`s are no longer distinct, so this file's"
            " claim to exercise them is stale",
        )
        assert_true(
            abs(Float64(py=q0[9 + 3 * b]) - 0.4) < 1e-15,
            String("box") + String(b) + "_z qpos0 is not .4",
        )

    var gtype = mj.geom_type.tolist()
    var gct = mj.geom_contype.tolist()
    var gca = mj.geom_conaffinity.tolist()
    for g in range(NGEOM4):
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
    # The four box geoms collide; the `target` geom is `class="ghost"` and does
    # not. If that ever inverted the boxes would fall through the world and the
    # target would become an obstacle in the middle of the arena.
    for g in range(19, 23):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) != 0,
            String("box geom ") + String(g) + " stopped colliding",
        )
    assert_true(
        Int(mf.geoms.data[23 * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) == 0,
        "the target geom started colliding",
    )

    var stype = mj.site_type.tolist()
    var spos = mj.site_pos.tolist()
    var worst_pos = Float64(0)
    for s in range(NSITE4):
        var so = s * MODEL_SITE_SIZE
        var r = _our_site_to_mj(s)
        assert_true(
            _mj_geom_type(Int(mf.sites.data[so + SITE_IDX_TYPE]))
            == Int(py=stype[r]),
            String("site_type mismatch on site ") + String(s),
        )
        var dp = max(
            abs(
                Float64(mf.sites.data[so + SITE_IDX_POS_X])
                - Float64(py=spos[r][0])
            ),
            max(
                abs(
                    Float64(mf.sites.data[so + SITE_IDX_POS_Y])
                    - Float64(py=spos[r][1])
                ),
                abs(
                    Float64(mf.sites.data[so + SITE_IDX_POS_Z])
                    - Float64(py=spos[r][2])
                ),
            ),
        )
        if dp > worst_pos:
            worst_pos = dp
        assert_true(
            dp <= 1e-12, String("site_pos mismatch on site ") + String(s)
        )

    # The reward reads `box_size` off the TARGET geom, so pin that our
    # `BOX_SIZE` is that number and not merely the boxes'.
    var gsize = mj.geom_size.tolist()
    assert_true(
        abs(Float64(py=gsize[23][0]) - BOX_SIZE) < 1e-15,
        "BOX_SIZE is not geom_size['target', 0] — the reward's margin is wrong",
    )

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var tiw = mj.tendon_invweight0.tolist()
    var worst_iw = Float64(0)
    for b in range(NBODY4):
        for k in range(2):
            var ours = Float64(mf.body_invweight0.data[2 * b + k])
            var r = Float64(py=biw[b][k])
            var rel = abs(ours - r) / (1e-15 + abs(r))
            if rel > worst_iw:
                worst_iw = rel
            assert_true(
                rel <= TOL_MODEL,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(NV4):
        var ours = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst_iw:
            worst_iw = rel
        assert_true(
            rel <= TOL_MODEL,
            String("dof_invweight0 mismatch on dof ") + String(i),
        )
    for t in range(NTEN4):
        var ours = Float64(
            mf.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_INVWEIGHT0]
        )
        var r = Float64(py=tiw[t])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst_iw:
            worst_iw = rel
        assert_true(
            rel <= TOL_MODEL,
            String("tendon_invweight0 mismatch on tendon ") + String(t),
        )
    print(
        "  worst mass rel =", worst_mass,
        " site pos =", worst_pos,
        " invweight0 rel =", worst_iw,
    )


# ── poses ───────────────────────────────────────────────────────────────────
#
# The arm segment is `manipulator`'s verbatim, so the C/D arm pose puts the
# grasp site in exactly the same place. Both constants are pinned against
# MuJoCo by `test_stacker_grasp_geometry_matches_mujoco` rather than trusted.
comptime ARM1: Float64 = 0.3
comptime ARM2: Float64 = -0.6
comptime ARM3: Float64 = 0.2
comptime GRASP_X: Float64 = -0.002376434117796365
comptime GRASP_Z: Float64 = 0.9026161228069853
# With the arm at rest the grasp site is straight above the shoulder.
comptime REST_GRASP_Z: Float64 = 0.915

# ⚠ THE HELD CUBE IS ROTATED. It had to be while a cube meeting a hand capsule
# face-on made MuJoCo emit TWO points for that pair where we emitted one, which
# would have put the qacc buckets below on a different constraint problem
# entirely; these angles were found by sweeping MuJoCo for one-point-per-pair
# grasps. `box_capsule_manifold` emits the second point now, so the constraint
# no longer applies and a face-on grasp would be gateable — the angles stay
# because they are what the numbers below were measured at. Either way the
# bucket test asserts `our ncon == MuJoCo ncon` per pose rather than relying on
# the angle staying lucky.
comptime BOX_HELD_ANGLE_A: Float64 = 0.56
comptime BOX_HELD_ANGLE_B: Float64 = 0.10

# Somewhere airborne and clear of the arm, for the contact-free buckets.
comptime PARK0_X: Float64 = 0.32
comptime PARK0_Z: Float64 = 0.50
comptime PARK1_X: Float64 = 0.45
comptime PARK1_Z: Float64 = 0.62
comptime PARK2_X: Float64 = 0.30
comptime PARK2_Z: Float64 = 0.68
comptime PARK3_X: Float64 = 0.18
comptime PARK3_Z: Float64 = 0.60


def _park(mut s: List[Float64], n_boxes: Int):
    """Put every box in the air, clear of the arm and of each other."""
    var xs = [PARK0_X, PARK1_X, PARK2_X, PARK3_X]
    var zs = [PARK0_Z, PARK1_Z, PARK2_Z, PARK3_Z]
    for b in range(n_boxes):
        s[BOX_QADR_0 + 3 * b + 0] = xs[b]
        s[BOX_QADR_0 + 3 * b + 1] = zs[b]
        s[BOX_QADR_0 + 3 * b + 2] = 0.0


def _state2(
    arm0: Float64, arm1: Float64, arm2: Float64, arm3: Float64,
    thumb: Float64, tip: Float64,
) -> List[Float64]:
    """A stack_2 (qpos, qvel) state with both boxes parked, as a flat list.

    The hand starts SYMMETRIC (`finger` = `thumb`, `fingertip` = `thumbtip`),
    which is both what `Stack.initialize_episode` sets and what the `coupling`
    equality holds. Starting off-symmetry would put the equality row far from
    its setpoint and turn a solver gate into a measurement of how two solvers
    cope with a stiff row.
    """
    var s = List[Float64]()
    for _ in range(NQ2 + NV2):
        s.append(0.0)
    s[0] = arm0
    s[1] = arm1
    s[2] = arm2
    s[3] = arm3
    s[4] = thumb  # thumb
    s[5] = tip  # thumbtip
    s[6] = thumb  # finger
    s[7] = tip  # fingertip
    _park(s, 2)
    return s^


def _state4(
    arm0: Float64, arm1: Float64, arm2: Float64, arm3: Float64,
    thumb: Float64, tip: Float64,
) -> List[Float64]:
    var s = List[Float64]()
    for _ in range(NQ4 + NV4):
        s.append(0.0)
    s[0] = arm0
    s[1] = arm1
    s[2] = arm2
    s[3] = arm3
    s[4] = thumb
    s[5] = tip
    s[6] = thumb
    s[7] = tip
    _park(s, 4)
    return s^


def _box2(mut s: List[Float64], b: Int, x: Float64, z: Float64, a: Float64):
    s[BOX_QADR_0 + 3 * b + 0] = x
    s[BOX_QADR_0 + 3 * b + 1] = z
    s[BOX_QADR_0 + 3 * b + 2] = a


def _zero_ctrl(n: Int) -> List[Float64]:
    var c = List[Float64]()
    for _ in range(n):
        c.append(0.0)
    return c^


def _mj_at2(
    state: List[Float64], ctrl: List[Float64],
    tx: Float64, tz: Float64,
) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    """MuJoCo at the same stack_2 state, built from OUR xml.

    Deliberate: the model constants are gated against the real reference above,
    so feeding both engines the identical model isolates the SOLVER.

    ⚠ The target is a MOCAP body in our XML, and MuJoCo indexes `mocap_pos` by
    MOCAP id, not body id — hence `body_mocapid`.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/stacker_2.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ2):
        dat.qpos[i] = state[i]
    for i in range(NV2):
        dat.qvel[i] = state[NQ2 + i]
    for i in range(NACT2):
        dat.ctrl[i] = ctrl[i]
    var mid = Int(py=m.body_mocapid[target_body_idx(2)])
    dat.mocap_pos[mid][0] = tx
    dat.mocap_pos[mid][1] = 0.001
    dat.mocap_pos[mid][2] = tz
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def _mj_at4(
    state: List[Float64], ctrl: List[Float64],
    tx: Float64, tz: Float64,
) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/stacker_4.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ4):
        dat.qpos[i] = state[i]
    for i in range(NV4):
        dat.qvel[i] = state[NQ4 + i]
    for i in range(NACT4):
        dat.ctrl[i] = ctrl[i]
    var mid = Int(py=m.body_mocapid[target_body_idx(4)])
    dat.mocap_pos[mid][0] = tx
    dat.mocap_pos[mid][1] = 0.001
    dat.mocap_pos[mid][2] = tz
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def _set_state_and_fk2(
    mut d: Dat2, mut mf: Mod2, mut integ: Integ2,
    state: List[Float64], ctrl: List[Float64],
    tx: Float64, tz: Float64,
) raises:
    """Both engines at the same state, ours stepped once so contacts and
    `site_xpos` are live (the touch entries read post-solve contact forces)."""
    var sf = M2.make_spec_fields[DTYPE]()
    M2.reset_data(sf, d)
    for i in range(NQ2):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV2):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ2 + i])
        d.qfrc.data[i] = Scalar[DTYPE](0)

    # FK SKIPS mocap bodies, so the target's world pose has to be preset from
    # `mocap_pos`/`mocap_quat` too — the env facade does that in
    # `_sync_mocap_to_fields`, and this test drives `integ.step` directly.
    # Without it the target sits at the origin with an all-zero quaternion and
    # every term that reads it silently reports a large, plausible error.
    var tb = target_body_idx(2)
    d.mocap_pos.data[tb * 3 + 0] = Scalar[DTYPE](tx)
    d.mocap_pos.data[tb * 3 + 1] = Scalar[DTYPE](0.001)
    d.mocap_pos.data[tb * 3 + 2] = Scalar[DTYPE](tz)
    d.mocap_quat.data[tb * 4 + 0] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 1] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 2] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 3] = Scalar[DTYPE](1)
    for k in range(3):
        var pv = d.mocap_pos.data[tb * 3 + k]
        d.xpos.data[tb * 3 + k] = pv
        d.xipos.data[tb * 3 + k] = pv
    for k in range(4):
        d.xquat.data[tb * 4 + k] = d.mocap_quat.data[tb * 4 + k]

    var act = List[Scalar[DTYPE]]()
    for _ in range(NA2 if NA2 > 0 else 1):
        act.append(Scalar[DTYPE](0))
    M2.apply_actions(sf, d, ctrl, act)
    integ.step["cpu"](d, mf)
    # `integ.step` INTEGRATES. `d.contacts` and `d.site_xpos` were computed at
    # the PRE-integration pose, so restoring qpos/qvel here puts every
    # observation term back on the same state MuJoCo's `mj_forward` saw. Without
    # this the two engines are one Euler step apart and the diff reads as a
    # small, plausible, wrong number rather than an obvious one.
    for i in range(NQ2):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV2):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ2 + i])


def _set_state_and_fk4(
    mut d: Dat4, mut mf: Mod4, mut integ: Integ4,
    state: List[Float64], ctrl: List[Float64],
    tx: Float64, tz: Float64,
) raises:
    var sf = M4.make_spec_fields[DTYPE]()
    M4.reset_data(sf, d)
    for i in range(NQ4):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV4):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ4 + i])
        d.qfrc.data[i] = Scalar[DTYPE](0)

    var tb = target_body_idx(4)
    d.mocap_pos.data[tb * 3 + 0] = Scalar[DTYPE](tx)
    d.mocap_pos.data[tb * 3 + 1] = Scalar[DTYPE](0.001)
    d.mocap_pos.data[tb * 3 + 2] = Scalar[DTYPE](tz)
    d.mocap_quat.data[tb * 4 + 0] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 1] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 2] = Scalar[DTYPE](0)
    d.mocap_quat.data[tb * 4 + 3] = Scalar[DTYPE](1)
    for k in range(3):
        var pv = d.mocap_pos.data[tb * 3 + k]
        d.xpos.data[tb * 3 + k] = pv
        d.xipos.data[tb * 3 + k] = pv
    for k in range(4):
        d.xquat.data[tb * 4 + k] = d.mocap_quat.data[tb * 4 + k]

    var act = List[Scalar[DTYPE]]()
    for _ in range(NA4 if NA4 > 0 else 1):
        act.append(Scalar[DTYPE](0))
    M4.apply_actions(sf, d, ctrl, act)
    integ.step["cpu"](d, mf)
    for i in range(NQ4):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV4):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ4 + i])


def test_stacker_grasp_geometry_matches_mujoco() raises:
    """The hardcoded grasp constants really are where the grasp site lands.

    This is what makes `GRASP_X` / `GRASP_Z` / `REST_GRASP_Z` legitimate rather
    than magic numbers: every contact pose below places a cube at the grasp
    site, and every reward pose measures a distance from it.
    """
    var mf = _build2()
    var integ = Integ2()

    var bent = _state2(0.0, ARM1, ARM2, ARM3, 0.0, 0.0)
    var d = Dat2()
    _set_state_and_fk2(d, mf, integ, bent, _zero_ctrl(NACT2), 0.2, 0.022)
    var mj = _mj_at2(bent, _zero_ctrl(NACT2), 0.2, 0.022)
    var dat = mj[2]
    # `grasp` is our site 0 and MuJoCo's site 0.
    var ex = abs(Float64(d.site_xpos.data[0]) - Float64(py=dat.site_xpos[0][0]))
    var ez = abs(Float64(d.site_xpos.data[2]) - Float64(py=dat.site_xpos[0][2]))
    assert_true(ex < 1e-9 and ez < 1e-9, "grasp site diverges from MuJoCo")
    assert_true(
        abs(Float64(py=dat.site_xpos[0][0]) - GRASP_X) < 1e-12
        and abs(Float64(py=dat.site_xpos[0][2]) - GRASP_Z) < 1e-12,
        "GRASP_X / GRASP_Z are stale — the bent-arm grasp site moved",
    )

    var rest = _state2(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var d2 = Dat2()
    _set_state_and_fk2(d2, mf, integ, rest, _zero_ctrl(NACT2), 0.2, 0.022)
    var mj2 = _mj_at2(rest, _zero_ctrl(NACT2), 0.2, 0.022)
    assert_true(
        abs(Float64(py=mj2[2].site_xpos[0][2]) - REST_GRASP_Z) < 1e-12
        and abs(Float64(py=mj2[2].site_xpos[0][0])) < 1e-15,
        "REST_GRASP_Z is stale — the rest-pose grasp site moved",
    )


def _our_qacc2(
    state: List[Float64], ctrl: List[Float64]
) raises -> List[Float64]:
    """`qacc` after one constrained solve at `state`, our engine.

    Returns NV accelerations followed by OUR contact count, which the caller
    diffs against MuJoCo's. Bucketing by MuJoCo's live rows while our own row
    set differs would classify a pose as "equality only" and then measure a
    phantom contact — the classification has to be checked, not assumed.
    """
    var sf = M2.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Mod2()
    var d = Dat2()
    M2.init_fields[DTYPE, 0](ctx, mf)
    M2.reset_data(sf, d)
    for i in range(NQ2):
        d.qpos.data[i] = Scalar[DTYPE](state[i])
    for i in range(NV2):
        d.qvel.data[i] = Scalar[DTYPE](state[NQ2 + i])
        d.qfrc.data[i] = Scalar[DTYPE](0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(NA2 if NA2 > 0 else 1):
        act.append(Scalar[DTYPE](0))
    var integ = Integ2()
    M2.apply_actions(sf, d, ctrl, act)
    integ.step["cpu"](d, mf)
    var out = List[Float64]()
    for i in range(NV2):
        out.append(Float64(integ.scratch.qacc_constrained.data[i]))
    out.append(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    return out^


def test_stacker_qacc_by_constraint_bucket() raises:
    """`qacc` vs MuJoCo on stack_2, bucketed by which constraint rows MuJoCo has
    live: A equality only, B +limit, C +contact, D all three.

    The question the buckets answer is the one every elliptic-cone domain asks:
    our elliptic path solves limits, equalities and fixed tendons SEQUENTIALLY
    after the Newton contact core, and a sequential split is exact when the row
    sets do not share dofs and degrades only when they couple. An aggregate
    number cannot tell those apart.

    ⚠ WHICH BOX CONTACTS THE BUCKETS CAN HOLD IS SET BY THE NARROW PHASE, and
    it widened three times as task #42 was fixed piece by piece: a cube HELD IN
    THE HAND once #45 landed (capsule/box, sphere/box), a cube resting FLAT ON
    THE FLOOR once box/plane emitted a point per corner, and a cube STACKED ON
    A CUBE once the box/box FACE manifold landed. Each of those poses was
    ungatable before its fix: with one contact where MuJoCo has four, the two
    engines are not solving the same constraint problem and the difference
    shows up as a solver error that is not one.

    The narrow phase no longer restricts what can go here: the box/box EDGE
    manifold and capsule/box's second point closed the rest of #42, so a box
    pair meeting on an EDGE and a capsule LYING ALONG a box face are both
    gateable now. Neither is in the table yet — adding one would extend the
    coverage this bucket has rather than fix anything.

    ⚠ THE HELD CUBE IS STILL ROTATED, and now for a THIRD reason: at these
    angles the hand's capsules meet the cube on a face, which the manifold
    handles, and the per-pose `our ncon == MuJoCo ncon` assertion enforces it
    either way.

    ⚠ BOTH D POSES KEEP THE LIMIT VIOLATION INSIDE ITS IMPEDANCE WIDTH
    (`solimplimit` width = .01 rad; thumb 1.05 is .0028 past the 60 deg stop).
    Driving it past the width hits open defect #41 (task #41), exactly as in
    `manipulator`. This is a DEBT recorded here rather than hidden: when that
    defect is fixed, its pinning test in the bring_peg file fails and this
    bucket should be widened too.
    """
    print("--- stacker: qacc vs MuJoCo, by live constraint rows ---")

    var tags = List[String]()
    var states = List[List[Float64]]()
    var ctrls = List[List[Float64]]()

    # A — equality only. Boxes airborne, hand open.
    tags.append(String("A1 rest, boxes airborne"))
    states.append(_state2(0, 0, 0, 0, 0.0, 0.0))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("A2 bent, boxes airborne"))
    states.append(_state2(0.3, ARM1, ARM2, ARM3, 0.2, 0.0))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("A3 bent, moving, driven"))
    var a3 = _state2(0.3, ARM1, ARM2, ARM3, 0.2, 0.0)
    a3[NQ2 + 0] = 0.5
    a3[NQ2 + 1] = -0.3
    a3[NQ2 + 2] = 0.7
    a3[NQ2 + 3] = 0.2
    a3[NQ2 + 4] = 0.1
    a3[NQ2 + 6] = 0.1
    # All three of a box's dofs at once — the free-flight half of the model.
    a3[NQ2 + 8] = 0.2
    a3[NQ2 + 9] = -0.1
    a3[NQ2 + 10] = 0.3
    states.append(a3^)
    ctrls.append([0.4, -0.3, 0.6, -0.2, 0.5])

    # B — equality + joint limit (the fingertip's `range="-40 20"`), no contact.
    tags.append(String("B1 fingertip limit"))
    states.append(_state2(0, 0, 0, 0, 0.0, 0.36))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("B2 fingertip limit, deeper"))
    states.append(_state2(0, 0, 0, 0, 0.0, 0.45))
    ctrls.append(_zero_ctrl(NACT2))

    # C — equality + contact, no limit: the hand closed on ITSELF. The 60 deg
    # stop is at 1.0472 rad, so anything below that gives contacts with no limit
    # row; the thumb/finger links and the fingertip spheres meet each other from
    # 9 pairs down to 6 as the hand closes further.
    tags.append(String("C1 hand closed .95, 9 self-contacts"))
    states.append(_state2(0, 0, 0, 0, 0.95, 0.0))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("C2 hand closed 1.00, arm bent"))
    states.append(_state2(0, ARM1, ARM2, ARM3, 1.00, 0.0))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("C3 hand closed 1.04, driven"))
    states.append(_state2(0, ARM1, ARM2, ARM3, 1.04, 0.0))
    ctrls.append([0.2, 0.1, -0.3, 0.1, 0.8])

    # C, with a CUBE in the hand — capsule/box and sphere/box. These lived in
    # `test_stacker_capsule_box_contact_record_is_an_open_defect` until task #45
    # was fixed; that test asserted a 0.26-6.6 relative error here.
    var c4 = _state2(0, ARM1, ARM2, ARM3, 0.40, 0.0)
    _box2(c4, 0, GRASP_X, GRASP_Z, BOX_HELD_ANGLE_A)
    tags.append(String("C4 cube held, light, turned .56"))
    states.append(c4^)
    ctrls.append(_zero_ctrl(NACT2))

    var c5 = _state2(0, ARM1, ARM2, ARM3, 0.50, 0.0)
    _box2(c5, 0, GRASP_X, GRASP_Z, BOX_HELD_ANGLE_B)
    tags.append(String("C5 cube held, firmer, turned .10"))
    states.append(c5^)
    ctrls.append(_zero_ctrl(NACT2))

    var c6 = _state2(0, ARM1, ARM2, ARM3, 0.50, 0.0)
    _box2(c6, 0, GRASP_X, GRASP_Z, BOX_HELD_ANGLE_B)
    tags.append(String("C6 cube held, driven"))
    states.append(c6^)
    ctrls.append([0.2, 0.1, -0.3, 0.1, 0.8])

    # C, with a cube RESTING ON THE FLOOR — box/plane, four support points.
    # ⚠ THIS IS THE POSE THE DOMAIN IS ABOUT, and it could not be gated at all
    # until box/plane emitted a point per corner: with one contact the cube is
    # supported at a single corner, so its qacc is a pivot rather than a rest.
    var c7 = _state2(0, 0, 0, 0, 0.0, 0.0)
    _box2(c7, 0, 0.15, 0.0219, 0.0)
    _box2(c7, 1, PARK1_X, PARK1_Z, 0.0)
    tags.append(String("C7 cube resting flat on the floor"))
    states.append(c7^)
    ctrls.append(_zero_ctrl(NACT2))

    # And on an EDGE, where MuJoCo emits two points rather than four — a
    # different branch of the same emit.
    var c8 = _state2(0, 0, 0, 0, 0.0, 0.0)
    _box2(c8, 0, 0.15, 0.0219, 0.5236)
    _box2(c8, 1, PARK1_X, PARK1_Z, 0.0)
    tags.append(String("C8 cube on an edge (30 deg)"))
    states.append(c8^)
    ctrls.append(_zero_ctrl(NACT2))

    # C, box on BOX — THE STACKING POSE, and the reason the task exists. These
    # two lived in `test_stacker_box_box_is_one_point_per_pair`, which asserted
    # we emitted ONE point where MuJoCo emitted four; that test was written to
    # fail once the face manifold landed, and it did, so it is gone and its
    # poses are here.
    var c9 = _state2(0, 0, 0, 0, 0.0, 0.0)
    _box2(c9, 0, 0.15, 0.0219, 0.0)
    _box2(c9, 1, 0.15, 0.0657, 0.0)
    tags.append(String("C9 cube stacked on a cube"))
    states.append(c9^)
    ctrls.append(_zero_ctrl(NACT2))

    var c10 = _state2(0, 0, 0, 0, 0.0, 0.0)
    _box2(c10, 0, 0.15, 0.0219, 0.0)
    _box2(c10, 1, 0.1938, 0.0219, 0.0)
    tags.append(String("C10 cubes side by side on the floor"))
    states.append(c10^)
    ctrls.append(_zero_ctrl(NACT2))

    # D — all three: the same self-contact with the stop just crossed.
    tags.append(String("D1 closed hand, self-contact + limit"))
    states.append(_state2(0, 0, 0, 0, 1.05, 0.0))
    ctrls.append(_zero_ctrl(NACT2))
    tags.append(String("D2 closed hand + limit, bent, driven"))
    states.append(_state2(0, ARM1, ARM2, ARM3, 1.05, 0.0))
    ctrls.append([0.0, 0.0, 0.0, 0.0, 0.6])

    var d3 = _state2(0, ARM1, ARM2, ARM3, 1.05, 0.0)
    _box2(d3, 0, GRASP_X, GRASP_Z, BOX_HELD_ANGLE_B)
    tags.append(String("D3 cube held + limit, driven"))
    states.append(d3^)
    ctrls.append([0.0, 0.0, 0.0, 0.0, 0.6])

    var worst_by_bucket = List[Float64]()
    var count_by_bucket = List[Int]()
    for _ in range(4):
        worst_by_bucket.append(0.0)
        count_by_bucket.append(0)
    var max_ncon = 0

    for p in range(len(tags)):
        var mj = _mj_at2(states[p], ctrls[p], 0.2, 0.022)
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

        var ours = _our_qacc2(states[p], ctrls[p])
        var our_ncon = Int(ours[NV2])
        var mj_ncon = Int(py=dat.ncon)
        if mj_ncon > max_ncon:
            max_ncon = mj_ncon
        var scale = Float64(1.0)
        for i in range(NV2):
            var a = abs(Float64(py=dat.qacc[i]))
            if a > scale:
                scale = a
        var worst = Float64(0)
        var worst_dof = 0
        for i in range(NV2):
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
            " measuring a different constraint problem. Every box narrow phase"
            " is point-for-point with MuJoCo now (tasks #42, #45), so this is"
            " a real regression rather than a known gap; the box/box and"
            " capsule/box sweeps under tests/physics3d/ localise it.",
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

    assert_true(
        worst_by_bucket[0] <= TOL_QACC_UNCOUPLED,
        "equality-only qacc diverges — the coupling row itself is wrong, which"
        " the sequential split cannot explain",
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

    # MAX_CONTACTS headroom. Truncation is SILENT in a rollout — the `ncon`
    # assertion above only covers these poses — so report the margin.
    print("  max MuJoCo ncon over the pose table =", max_ncon,
          " MAX_CONTACTS =", MAXC2)
    assert_true(
        max_ncon < MAXC2,
        "the pose table already reaches MAX_CONTACTS; a rollout would truncate"
        " contacts silently",
    )


# ── observation ─────────────────────────────────────────────────────────────


def _obs_blocks(n_boxes: Int) -> List[Int]:
    """Block boundaries in the flattened observation."""
    var b = List[Int]()
    b.append(0)
    b.append(16)  # arm_pos
    b.append(24)  # arm_vel
    b.append(29)  # touch
    b.append(33)  # hand_pos
    b.append(33 + 4 * n_boxes)  # box_pos
    b.append(33 + 7 * n_boxes)  # box_vel
    b.append(35 + 7 * n_boxes)  # target_pos
    return b^


def _obs_names() -> List[String]:
    return [
        String("arm_pos"), String("arm_vel"), String("touch"),
        String("hand_pos"), String("box_pos"), String("box_vel"),
        String("target_pos"),
    ]


def test_stacker_observation_matches_mujoco() raises:
    """All 49 stack_2 entries at a pose where every term is non-trivial.

    The pose is deliberately ASYMMETRIC in the hand (finger != thumb, fingertip
    != thumbtip), because `_ARM_JOINTS` lists the finger chain before the thumb
    chain while the model declares the thumb chain first, and a symmetric hand
    would hide a wrong permutation entirely.

    It is equally deliberate that every box carries THREE DISTINCT velocity
    components: `box_joint_names` is built `for dim in 'xyz'` over a model that
    declares x, z, y, so entries 1 and 2 of each triple are transposed. A box
    moving along a single axis would pass either way.

    ⚠ THE TOUCH BLOCK IS LOADED BY THE HAND CLOSING ON ITSELF, not by holding a
    cube, and that is forced. `touch` sums POST-SOLVE contact forces inside each
    zone, so it can only be gated where our contact set matches MuJoCo's — and
    with a cube in the hand it does not (task #45). Self-contact is
    capsule/capsule and sphere/capsule, which is exact, so the five touch
    entries and the BOX ZONE code behind them are gated on real forces here.
    """
    var mf = _build2()
    var d = Dat2()
    var integ = Integ2()

    var state = _state2(0.0, ARM1, ARM2, ARM3, 0.95, 0.0)
    state[6] = 1.02  # finger    != thumb
    state[7] = 0.12  # fingertip != thumbtip
    # Three distinct components on each box, so the 'xyz' permutation shows.
    state[NQ2 + 8] = 0.20
    state[NQ2 + 9] = -0.11
    state[NQ2 + 10] = 0.33
    state[NQ2 + 11] = -0.44
    state[NQ2 + 12] = 0.55
    state[NQ2 + 13] = -0.66
    var ctrl = _zero_ctrl(NACT2)
    _set_state_and_fk2(d, mf, integ, state, ctrl, 0.21, 0.066)

    var mj = _mj_at2(state, ctrl, 0.21, 0.066)
    var builder = Python.import_module("stacker_ref")
    var ref_obs = builder.observation(mj[1], mj[2], 2)

    var obs = List[Scalar[DTYPE]]()
    _ = CFG2.custom_extract_obs_cpu[DTYPE, NQ2, NV2, NBODY2, MAXC2, NSITE2](
        d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
        List[Scalar[DTYPE]](), obs,
    )
    assert_true(
        len(obs) == 49, String("stack_2 obs dim ") + String(len(obs)) + " != 49"
    )

    var names = _obs_names()
    var bounds = _obs_blocks(2)
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

    # Non-vacuity. The touch block must be loaded, or the five entries gate
    # nothing; and the box_vel block must have SIX distinct magnitudes, or the
    # 'xyz' permutation is untested.
    var touch_nz = 0
    var touch_zero = 0
    for i in range(24, 29):
        if Float64(py=ref_obs[i]) > 0.0:
            touch_nz += 1
        else:
            touch_zero += 1
    print("  touch zones loaded =", touch_nz, " reading zero =", touch_zero)
    assert_true(
        touch_nz >= 2,
        "fewer than two touch zones are loaded at this pose, so the five touch"
        " entries — and the BOX zone code behind them — are barely gated",
    )
    # ⚠ THE ZEROS ARE THE POINT, not filler. `thumb_touch` and `finger_touch`
    # sit on bodies that ARE in contact here, and MuJoCo still reports 0 for
    # them because the contact points land ~2 mm outside those zones. A contact
    # INSIDE a zone is hit by the sensor ray from either direction, so only an
    # outside contact can tell a correct ray from a reversed one — and a
    # reversed ray is exactly the bug this pose caught (55 N reported where
    # MuJoCo reports nothing). If this count ever drops, the touch block has
    # gone back to gating only the easy case.
    assert_true(
        touch_zero >= 2,
        "no touch zone reads zero at this pose any more, so the sensor's RAY"
        " DIRECTION is no longer gated: every remaining contact is inside its"
        " zone, where the direction cannot change the answer",
    )
    for i in range(41, 47):
        for j in range(i + 1, 47):
            assert_true(
                abs(Float64(py=ref_obs[i]) - Float64(py=ref_obs[j])) > 1e-6,
                "two box_vel entries are equal, so a transposed pair inside the"
                " 'xyz' permutation would read as correct",
            )
    print("  worst |d obs| =", worst_all)


def test_stacker4_observation_matches_mujoco() raises:
    """All 63 stack_4 entries.

    The touch block is deliberately UNLOADED here (the hand is empty), because
    the arm, its sites and the whole sensor half are byte-identical between the
    two models and are gated with live contact forces by the stack_2 test above.
    What stack_4 adds and only stack_4 can gate is the WIDTH: four box poses and
    four box velocity triples in front of a target block that has moved eight
    slots. Reading one box too few would shorten the vector; reading the wrong
    body would leave the target entries where box3's used to be.
    """
    var mf = _build4()
    var d = Dat4()
    var integ = Integ4()

    var state = _state4(0.15, ARM1, ARM2, ARM3, 0.30, 0.10)
    state[6] = 0.42  # finger    != thumb
    state[7] = -0.08  # fingertip != thumbtip
    # Twelve distinct velocity components.
    for b in range(4):
        state[NQ4 + BOX_QADR_0 + 3 * b + 0] = 0.13 + 0.21 * Float64(b)
        state[NQ4 + BOX_QADR_0 + 3 * b + 1] = -0.07 - 0.17 * Float64(b)
        state[NQ4 + BOX_QADR_0 + 3 * b + 2] = 0.31 + 0.29 * Float64(b)
    var ctrl = _zero_ctrl(NACT4)
    _set_state_and_fk4(d, mf, integ, state, ctrl, -0.19, 0.154)

    var mj = _mj_at4(state, ctrl, -0.19, 0.154)
    var builder = Python.import_module("stacker_ref")
    var ref_obs = builder.observation(mj[1], mj[2], 4)

    var obs = List[Scalar[DTYPE]]()
    _ = CFG4.custom_extract_obs_cpu[DTYPE, NQ4, NV4, NBODY4, MAXC4, NSITE4](
        d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
        List[Scalar[DTYPE]](), obs,
    )
    assert_true(
        len(obs) == 63, String("stack_4 obs dim ") + String(len(obs)) + " != 63"
    )

    var names = _obs_names()
    var bounds = _obs_blocks(4)
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
        assert_true(
            worst <= TOL_OBS,
            String("stack_4 observation block ") + names[blk] + " diverges",
        )

    # Non-vacuity: the four box_pos entries must be four DIFFERENT poses, and
    # the target block must carry the pose we actually set.
    for b in range(4):
        var x = Float64(py=ref_obs[37 + 4 * b])
        assert_true(
            abs(x) > 0.01,
            String("box") + String(b) + " reads as ~0 in box_pos, so the block"
            " cannot distinguish four bodies from one unwritten one",
        )
    assert_true(
        abs(Float64(py=ref_obs[61]) + 0.19) < 1e-12
        and abs(Float64(py=ref_obs[62]) - 0.154) < 1e-12,
        "the target_pos block is not the mocap pose this test set — it is"
        " reading the wrong body, or the block is in the wrong place",
    )
    print("  worst |d obs| =", worst_all)


# ── reward ──────────────────────────────────────────────────────────────────


def test_stacker_reward_matches_mujoco() raises:
    """`box_is_close * hand_is_far`, over poses that separate the two factors.

    Three things can independently be wrong — which sites pair with which, the
    `min` over boxes, and the two different tolerance shapes — so the poses
    isolate them rather than sampling the curve evenly:

      * a cube parked ON the target with the arm at rest: `hand_is_far` is
        saturated, so the reward IS `box_is_close` and the four offsets span its
        curve from ~1 to ~0.
      * the target parked just under the grasp site: `box_is_close` is saturated
        and the reward IS `hand_is_far`, which has bounds (.1, inf) and a .01
        margin — a much sharper function than `box_is_close`'s (0, 0) with a
        .044 margin. Reading one shape for the other reads 1.0 here.
      * stack_4 with only the LAST box near the target, and the other three
        further away in ascending order: the `min` has to pick box3. Taking
        box0, or a mean, is a different number at that pose.
    """
    var mf = _build2()
    var integ = Integ2()
    var builder = Python.import_module("stacker_ref")

    var tags = List[String]()
    var states = List[List[Float64]]()
    var txs = List[Float64]()
    var tzs = List[Float64]()

    # box_is_close alone: hand at rest (grasp .915 up), so hand_is_far == 1.
    var offsets = [0.0, 0.005, 0.02, 0.05]
    for k in range(len(offsets)):
        var s = _state2(0, 0, 0, 0, 0.0, 0.0)
        _box2(s, 0, 0.2, 0.022 + offsets[k], 0.0)
        tags.append(String("bring dz=") + String(offsets[k]))
        states.append(s^)
        txs.append(0.2)
        tzs.append(0.022)

    # hand_is_far alone: target just below the rest-pose grasp site, cube ON it.
    var hand_gaps = [0.095, 0.09]
    for k in range(len(hand_gaps)):
        var s = _state2(0, 0, 0, 0, 0.0, 0.0)
        var tz = REST_GRASP_Z - hand_gaps[k]
        _box2(s, 0, 0.0, tz, 0.0)
        tags.append(String("hand near, gap=") + String(hand_gaps[k]))
        states.append(s^)
        txs.append(0.0)
        tzs.append(tz)

    var worst = Float64(0)
    var lo = Float64(2)
    var hi = Float64(-1)
    for k in range(len(tags)):
        var d = Dat2()
        _set_state_and_fk2(d, mf, integ, states[k], _zero_ctrl(NACT2),
                           txs[k], tzs[k])
        var mj = _mj_at2(states[k], _zero_ctrl(NACT2), txs[k], tzs[k])
        var ref_r = Float64(py=builder.reward(mj[1], mj[2], 2))
        var got = CFG2.compute_reward_and_done_cpu[
            DTYPE, NQ2, NV2, NBODY2, MAXC2, NSITE2
        ](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            Scalar[DTYPE](0), _zero_ctrl(NACT2), 0, 1,
        )
        var e = abs(Float64(got[0]) - ref_r)
        if e > worst:
            worst = e
        var v = Float64(got[0])
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        print("   ", tags[k], " ours =", got[0], " MuJoCo =", ref_r)
        assert_true(not got[1], "dm_control tasks never terminate early")
        assert_true(
            e <= TOL_OBS, String("stack_2 reward diverges at ") + tags[k]
        )

    # Non-vacuity: the table has to span the curve, or a constant would pass.
    assert_true(
        lo < 0.15 and hi > 0.9,
        "the reward table no longer spans [~0, ~1], so a constant reward would"
        " pass this test",
    )
    print("  stack_2 worst |d reward| =", worst, " range", lo, "..", hi)

    # stack_4: the `min` over boxes must pick the nearest, which is the LAST.
    var mf4 = _build4()
    var integ4 = Integ4()
    var s4 = _state4(0, 0, 0, 0, 0.0, 0.0)
    _box2(s4, 0, 0.2 + 0.30, 0.30, 0.0)
    _box2(s4, 1, 0.2 + 0.20, 0.25, 0.0)
    _box2(s4, 2, 0.2 + 0.10, 0.20, 0.0)
    _box2(s4, 3, 0.2, 0.022 + 0.005, 0.0)  # the only one near the target
    var d4 = Dat4()
    _set_state_and_fk4(d4, mf4, integ4, s4, _zero_ctrl(NACT4), 0.2, 0.022)
    var mj4 = _mj_at4(s4, _zero_ctrl(NACT4), 0.2, 0.022)
    var ref_r4 = Float64(py=builder.reward(mj4[1], mj4[2], 4))
    var got4 = CFG4.compute_reward_and_done_cpu[
        DTYPE, NQ4, NV4, NBODY4, MAXC4, NSITE4
    ](
        d4, mf4.bodies.data, mf4.joints.data, mf4.geoms.data, mf4.sites.data,
        Scalar[DTYPE](0), _zero_ctrl(NACT4), 0, 1,
    )
    print("   stack_4 min-over-boxes: ours =", got4[0], " MuJoCo =", ref_r4)
    assert_true(
        abs(Float64(got4[0]) - ref_r4) <= TOL_OBS,
        "stack_4 reward diverges — the min over four boxes",
    )
    # And the value must be the NEAR box's, not the far ones'. box3 is .0051
    # from the target (.005 in z, .001 in y); box0 is .43 away, which scores ~0.
    assert_true(
        Float64(got4[0]) > 0.9,
        "the stack_4 reward is not the NEAREST box's score — `min` is reading"
        " the wrong box, or averaging",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
