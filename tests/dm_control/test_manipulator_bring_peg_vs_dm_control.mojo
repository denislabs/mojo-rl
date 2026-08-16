"""dm_control `manipulator-bring_peg` parity: our model vs MuJoCo's.

The reference is `manipulator_ref.model(use_peg=True, insert=False)`, not
`from_xml_path` — no task uses `manipulator.xml` as written (see that file).

WHAT THIS VARIANT ADDS OVER bring_ball
--------------------------------------
The arm, the tendons, the equality, the sensors and the actuators are shared
verbatim (they come out of the same `MANIP_HEAD` / `MANIP_TAIL` segments), so
this file deliberately does NOT re-gate them in depth. What is new is
everything the prop swap brings:

  - A prop with a CHILD BODY. `peg` carries `pommel` as a sub-body, so the
    prop occupies two body slots and the target shifts from 11 to 12. Every
    index after the arm moves, which is exactly why the four tasks are four
    models.
  - THREE colliding prop geoms (two capsules + a sphere) instead of one
    sphere, which is what pushed the measured contact count from 9 to 21.
  - EIGHT box sites (four on the peg, four on the target) where bring_ball had
    two spheres.
  - The four-term `_peg_reward`, which is the only task-level logic in the
    whole domain that `bring_ball` does not already exercise:
        max(bringing, grasping/3)
    Three independent things can be wrong there and each has a distinct
    signature — the pair of means, the `max`, and the `/3` — so the reward
    test drives poses that isolate them (see its docstring).

⚠ THE PEG IS HELD BLADE-OUT. `peg_grasp` sits at the peg origin and
`peg_pinch` .025 further along the peg's LOCAL -z, while the hand's `grasp`
and `pinch` run the other way along hand +z. A correct grasp therefore has the
peg rotated ~pi about y relative to the hand — with `peg_y = 0` the blade
points back through the palm and the wrist, which reads as a plausible
"grasp" pose and silently measures 21 contacts of deep interpenetration
instead. `PEG_HELD_ANGLE` below is that rotation, and
`test_manipulator_peg_grasp_geometry_matches_mujoco` pins it by asserting that
BOTH site pairs coincide.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulator_bring_peg_vs_dm_control.mojo
"""

from std.math import abs, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulator import (
    DMManipulatorBringPegModel as M,
    DMManipulatorBringPegConfig as CFG,
    target_body_idx,
    site_object,
    site_object_pinch,
    site_object_grasp,
    site_object_tip,
    site_target,
    site_target_tip,
    SITE_GRASP,
    SITE_PINCH,
)
from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
    JOINT_IDX_QPOS0,
    GEOM_IDX_BODY,
    GEOM_IDX_TYPE,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
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
comptime NBODY: Int = M.NBODY  # 13
comptime NJOINT: Int = M.NJOINT  # 11
comptime NGEOM: Int = M.NGEOM  # 25
comptime NSITE: Int = M.NSITE  # 15
comptime NACT: Int = M.nact  # 5
comptime NTEN: Int = M.MAX_TENDON  # 2
comptime MAXC: Int = M.MAX_CONTACTS
comptime NEQ: Int = M.MAX_EQUALITY
comptime NEXCL: Int = M.NEXCLUDE
comptime NA: Int = M.NA

comptime USE_PEG: Bool = True
comptime INSERT: Bool = False

# Model constants are exact rational arithmetic on both sides up to the
# inertia integrals, so anything above ~1e-12 is a real disagreement.
comptime TOL_MODEL: Float64 = 1e-9

# qacc gates, per bucket, so a regression in a COUPLED bucket cannot hide
# behind the uncoupled one. Set from the measured worst case with roughly a
# decade of headroom; the printout below reports every pose.
comptime TOL_QACC_UNCOUPLED: Float64 = 1e-8
comptime TOL_QACC_COUPLED: Float64 = 1e-8
# Observation and reward are pure readbacks of state the physics layer already
# gates, so they sit at the FK floor.
comptime TOL_OBS: Float64 = 1e-9
# TOUCH reads POST-SOLVE contact forces, so it inherits the contact solve's
# floor rather than FK's. Split out so the state-readback blocks stay pinned at
# 1e-9 and cannot drift behind it.
comptime TOL_TOUCH: Float64 = 1e-8


# OUR site order IS MuJoCo's, as of the element-order fix (2026-08-03).
#
# It used to diverge here: `palm_touch` is declared AFTER the `pinch site` body
# but belongs to `hand`, so MuJoCo's body sort pulls it ahead of `pinch` while
# our XML-text walk left it behind. This file carried a permutation swapping
# sites 1 and 2 to paper over that — treating the divergence as a property of
# the model to record rather than a bug to fix.
#
# It was a bug. The same text-vs-body ordering permutes JOINTS, and
# `fields_build` derives `qpos_adr`/`dof_adr` as running counters over the
# joint array, so the whole `qpos` layout went with it — which is how
# dm_control's dog exposed it. `full_parser` now groups joints, geoms and sites
# by body id (`_stable_group_by_body_*`), gated by
# `tests/physics3d/test_element_order_vs_mujoco.mojo`.
#
# ⚠ THIS FILE WAS MISSED when the sibling permutations were removed, because
# the sweep grepped for the HELPER in the three files that defined it and not
# for the ASSERTIONS that consume it. It surfaced as `site_bodyid mismatch on
# site 1` with every physics number still green (obs 2.1e-10, reward 1.8e-11).
#
# The identity is kept rather than deleted so the call sites still read
# "our index -> MuJoCo's index", and so a future divergence has one obvious
# place to be expressed.
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
    """The compiled reference `mjModel` for `bring_peg`."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/dm_control")
    var builder = Python.import_module("manipulator_ref")
    return builder.model(True, False)


comptime Mod = Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]
comptime Dat = Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1]
comptime Integ = EulerIntegrator[
    DTYPE, NQ, NV, NBODY, NJOINT, MAXC, NGEOM, NEQ, NTEN, NSITE,
    NEXCL, 0, M.CONE_TYPE, 1, SOLVER="newton",
]


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    M.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_bring_peg_dims_match_mujoco() raises:
    """Element counts. Cheap, and the first thing a dropped `<body>` breaks.

    `pommel` is the reason nbody is 13 and not 12: it is a child body of the
    prop, so deleting the ball costs one body slot and adding the peg buys
    two.
    """
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
    # The whole reason these are four models rather than one: bring_ball puts
    # the target at body 11 and this puts it at 12.
    assert_true(
        target_body_idx(USE_PEG, INSERT) == 12,
        "target_body_idx disagrees with the bring_peg layout",
    )


def test_bring_peg_ordering_matches_mujoco() raises:
    """Body / joint / geom / site ORDER, pinned by name on the MuJoCo side.

    The peg's four sites are declared AFTER its `pommel` child body in the XML,
    so our text order interleaves a sub-body's geom between the parent's geoms
    and its sites. MuJoCo sorts by body id instead. Those two orders happen to
    agree here — `pommel` has no sites and the peg has no geoms after it — but
    "happen to agree" is exactly the kind of claim point_mass proved can fail
    silently, so it is pinned.
    """
    var mj = _ref()
    var mujoco = Python.import_module("mujoco")

    var body_names = [
        "world", "upper_arm", "middle_arm", "lower_arm", "hand",
        "pinch site", "thumb", "thumbtip", "finger", "fingertip",
        "peg", "pommel", "target_peg",
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
        "peg_x", "peg_z", "peg_y",
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
        "blade", "guard", "pommel",
        "target_blade", "target_guard", "target_pommel",
    ]
    for i in range(len(geom_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, geom_names[i]))
            == i,
            String("MuJoCo geom order moved at ") + geom_names[i],
        )

    # ⚠ THIS LIST IS IN MuJoCo's ORDER, which is now also ours: `palm_touch`
    # is at 1 and `pinch` at 2. It was written the other way round, matching
    # the XML text order our parser used to produce, and the swap lived in
    # `_our_site_to_mj`. With that helper now the identity the LIST is what has
    # to move — verified against the compiled reference, not reasoned about:
    #   0 grasp (body 4) · 1 palm_touch (body 4) · 2 pinch (body 5) · ...
    var site_names = [
        "grasp", "palm_touch", "pinch",
        "thumb_touch", "thumbtip_touch",
        "finger_touch", "fingertip_touch",
        "peg", "peg_pinch", "peg_grasp", "peg_tip",
        "target_peg", "target_peg_pinch", "target_peg_grasp", "target_peg_tip",
    ]
    for i in range(len(site_names)):
        assert_true(
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, site_names[i]))
            == _our_site_to_mj(i),
            String("site order moved at ") + site_names[i]
            + " — _our_site_to_mj is stale",
        )

    # The index helpers the config reads, against those same names. A helper
    # that disagrees with the model is a silently wrong reward.
    assert_true(site_object(USE_PEG) == 7, "site_object != peg")
    assert_true(site_object_pinch(USE_PEG) == 8, "site_object_pinch != peg_pinch")
    assert_true(site_object_grasp(USE_PEG) == 9, "site_object_grasp != peg_grasp")
    assert_true(site_object_tip(USE_PEG) == 10, "site_object_tip != peg_tip")
    assert_true(site_target(USE_PEG, INSERT) == 11, "site_target != target_peg")
    assert_true(
        site_target_tip(USE_PEG, INSERT) == 14,
        "site_target_tip != target_peg_tip",
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


def test_bring_peg_model_constants_match_mujoco() raises:
    """`body_mass`, `qpos0`, geom types/masks, site types/pos, and all three
    `invweight0` tables.

    `peg_x` carries `ref="-.4"` — a NEGATIVE slide reference, where bring_ball
    only ever exercised a positive one. Per bug 18 a mis-scaled `ref` skews
    every constraint inverse weight, since those are built at qpos0, and a sign
    error there is invisible in the pose itself.
    """
    var mj = _ref()
    var mf = _build()

    var mref = mj.body_mass.tolist()
    var worst_mass = Float64(0)
    for b in range(NBODY):
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
    for j in range(NJOINT):
        assert_true(
            abs(
                Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS0])
                - Float64(py=q0[Int(py=jqposadr[j])])
            )
            <= 1e-12,
            String("qpos0 mismatch on joint ") + String(j),
        )
    # Non-vacuity: qpos0 must actually be non-trivial, and specifically must
    # carry the NEGATIVE reference this variant introduced.
    assert_true(
        abs(Float64(py=q0[8]) + 0.4) < 1e-15,
        "peg_x qpos0 is not -.4 — the reference model changed and this file's"
        " claim to exercise a negative slide `ref` is stale",
    )

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
    # The three `target_*` geoms are `class="ghost"`, i.e. non-colliding; the
    # three peg geoms are not. If that ever inverted the peg would fall through
    # the world and the target would become an obstacle.
    for g in range(19, 22):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) != 0,
            String("peg geom ") + String(g) + " stopped colliding",
        )
    for g in range(22, 25):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) == 0,
            String("target geom ") + String(g) + " started colliding",
        )

    var stype = mj.site_type.tolist()
    var spos = mj.site_pos.tolist()
    var worst_pos = Float64(0)
    for s in range(NSITE):
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

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var tiw = mj.tendon_invweight0.tolist()
    var worst_iw = Float64(0)
    for b in range(NBODY):
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
    for i in range(NV):
        var ours = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(ours - r) / (1e-15 + abs(r))
        if rel > worst_iw:
            worst_iw = rel
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
# The C/D arm pose and the grasp site it produces are shared with the
# bring_ball file (the arm is the same segment), so the site lands in the same
# place. Pinned by `test_manipulator_peg_grasp_geometry_matches_mujoco` rather
# than trusted.
comptime GRASP_X: Float64 = -0.0023764341177963649
comptime GRASP_Z: Float64 = 0.90261612280698533
# Somewhere the hand cannot reach, for the contact-free buckets.
comptime PEG_FAR_X: Float64 = -0.3
comptime PEG_FAR_Z: Float64 = 0.55

# The C/D arm configuration, and the peg rotation that grasping it requires.
comptime ARM1: Float64 = 0.3
comptime ARM2: Float64 = -0.6
comptime ARM3: Float64 = 0.2
# The hand's world tilt is the sum of the three hinge angles below the root;
# the peg must be ANTI-aligned with it, because `peg_grasp`/`peg_pinch` run
# along the peg's local -z while `grasp`/`pinch` run along the hand's +z.
comptime PEG_HELD_ANGLE: Float64 = pi - (ARM1 + ARM2 + ARM3)

# `target_peg` sits here in the XML; the reward poses park the peg near it.
comptime TARGET_X: Float64 = -0.2
comptime TARGET_Y: Float64 = 0.001
comptime TARGET_Z: Float64 = 0.4


def _pose_state(
    arm0: Float64, arm1: Float64, arm2: Float64, arm3: Float64,
    thumb: Float64, tip: Float64,
    peg_x: Float64, peg_z: Float64, peg_y: Float64,
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
    s[8] = peg_x
    s[9] = peg_z
    s[10] = peg_y
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


def _zero_ctrl() -> List[Float64]:
    var c = List[Float64]()
    for _ in range(NACT):
        c.append(0.0)
    return c^


def _mj_at(
    state: List[Float64], ctrl: List[Float64]
) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    """MuJoCo at the same state, built from OUR xml.

    Deliberate: the model constants are gated against the real reference above,
    so feeding both engines the identical model isolates the SOLVER.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/manipulator_bring_peg.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ):
        dat.qpos[i] = state[i]
    for i in range(NV):
        dat.qvel[i] = state[NQ + i]
    for i in range(NACT):
        dat.ctrl[i] = ctrl[i]
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


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

    # The target is a MOCAP body here and a static one in the reference, so pin
    # ours to the reference's XML pose. FK SKIPS mocap bodies, so the world
    # pose has to be preset from `mocap_pos`/`mocap_quat` too — the env facade
    # does that in `_sync_mocap_to_fields`, and this test drives `integ.step`
    # directly. Without it the target sits at the origin and every term that
    # reads it silently reports a ~45 cm error.
    var tb = target_body_idx(USE_PEG, INSERT)
    d.mocap_pos.data[tb * 3 + 0] = Scalar[DTYPE](TARGET_X)
    d.mocap_pos.data[tb * 3 + 1] = Scalar[DTYPE](TARGET_Y)
    d.mocap_pos.data[tb * 3 + 2] = Scalar[DTYPE](TARGET_Z)
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


def _site_gap(d: Dat, ours_a: Int, ours_b: Int) -> Float64:
    """3-D distance between two of OUR sites, from live `site_xpos`."""
    var s = Float64(0)
    for k in range(3):
        var dd = Float64(d.site_xpos.data[ours_a * 3 + k]) - Float64(
            d.site_xpos.data[ours_b * 3 + k]
        )
        s += dd * dd
    return s ** 0.5


def test_manipulator_peg_grasp_geometry_matches_mujoco() raises:
    """`PEG_HELD_ANGLE` really does put the peg in the hand, in both engines.

    This is what makes the hardcoded pose constants below legitimate rather
    than magic numbers, and it is the assertion that catches the failure mode
    described in the file docstring: at `peg_y = 0` the peg reads as grasped by
    position while its blade passes through the palm, and BOTH site pairs are
    then far apart even though `peg_grasp` is exactly on `grasp`.
    """
    var state = _pose_state(
        0.0, ARM1, ARM2, ARM3, 0.35, 0.0,
        GRASP_X, GRASP_Z, PEG_HELD_ANGLE, False,
    )
    var mf = _build()
    var d = Dat()
    var integ = Integ()
    _set_state_and_fk(d, mf, integ, state, _zero_ctrl())

    var mj = _mj_at(state, _zero_ctrl())
    var dat = mj[2]

    # `grasp` is our site 0 and MuJoCo's site 0.
    var ex = abs(Float64(d.site_xpos.data[0]) - Float64(py=dat.site_xpos[0][0]))
    var ez = abs(Float64(d.site_xpos.data[2]) - Float64(py=dat.site_xpos[0][2]))
    assert_true(ex < 1e-9 and ez < 1e-9, "grasp site diverges from MuJoCo")
    assert_true(
        abs(Float64(py=dat.site_xpos[0][0]) - GRASP_X) < 1e-12
        and abs(Float64(py=dat.site_xpos[0][2]) - GRASP_Z) < 1e-12,
        "GRASP_* is stale — the grasp site moved",
    )

    # Both site pairs coincide: peg_grasp on grasp AND peg_pinch on pinch. The
    # second is the one that fails at the wrong peg angle.
    var g_gap = _site_gap(d, site_object_grasp(USE_PEG), SITE_GRASP)
    var p_gap = _site_gap(d, site_object_pinch(USE_PEG), SITE_PINCH)
    print("  |peg_grasp - grasp| =", g_gap, "  |peg_pinch - pinch| =", p_gap)
    assert_true(
        g_gap < 1e-9,
        "the peg is not at the grasp site — the pose constants are stale",
    )
    assert_true(
        p_gap < 1e-9,
        "peg_grasp is on grasp but peg_pinch is NOT on pinch: the peg is held"
        " backwards, with its blade through the palm. PEG_HELD_ANGLE is wrong.",
    )


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


def test_bring_peg_qacc_by_constraint_bucket() raises:
    """`qacc` vs MuJoCo, bucketed by which constraint rows MuJoCo has live.

    Same four buckets as bring_ball — A equality only, B +limit, C +contact,
    D all three — because the question is the same one: our ELLIPTIC path
    solves limits, equalities and fixed tendons SEQUENTIALLY after the Newton
    contact core, and a sequential split is exact when the row sets do not
    share dofs and degrades only when they couple. An aggregate number cannot
    tell those apart.

    What is new here is the CONTACT half. bring_ball's grasp puts 2 contacts on
    a single sphere; the peg's three geoms reach 6-11 at the same poses, so the
    C and D buckets exercise a much larger elliptic system (3 rows per contact)
    than anything bring_ball built.
    """
    print("--- bring_peg: qacc vs MuJoCo, by live constraint rows ---")

    var tags = List[String]()
    var states = List[List[Float64]]()
    var ctrls = List[List[Float64]]()

    # A — equality only. Peg parked out of reach, hand open.
    tags.append(String("A1 rest, peg far"))
    states.append(
        _pose_state(0, 0, 0, 0, 0.0, 0.0, PEG_FAR_X, PEG_FAR_Z, 0.0, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("A2 bent, peg far"))
    states.append(
        _pose_state(0.3, ARM1, ARM2, ARM3, 0.2, 0.0, PEG_FAR_X, PEG_FAR_Z, 0.0, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("A3 bent, moving, driven"))
    states.append(
        _pose_state(0.3, ARM1, ARM2, ARM3, 0.2, 0.0, PEG_FAR_X, PEG_FAR_Z, 0.0, True)
    )
    ctrls.append([0.4, -0.3, 0.6, -0.2, 0.5])

    # B — equality + joint limit (the fingertip's `range="-40 20"`), no contact.
    tags.append(String("B1 fingertip limit"))
    states.append(
        _pose_state(0, 0, 0, 0, 0.0, 0.36, PEG_FAR_X, PEG_FAR_Z, 0.0, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("B2 fingertip limit, deeper"))
    states.append(
        _pose_state(0, 0, 0, 0, 0.0, 0.45, PEG_FAR_X, PEG_FAR_Z, 0.0, False)
    )
    ctrls.append(_zero_ctrl())

    # C — equality + contact: the peg held blade-out between finger and thumb.
    tags.append(String("C1 grasp, light"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 0.35, 0.0, GRASP_X, GRASP_Z,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("C2 grasp, firmer"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 0.50, 0.0, GRASP_X, GRASP_Z + 0.01,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("C3 grasp, driven"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 0.65, 0.0, GRASP_X, GRASP_Z + 0.02,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append([0.2, 0.1, -0.3, 0.1, 0.8])

    # D — all three.
    #
    # D1/D2 keep the limit violation INSIDE its impedance width (`solimplimit`
    # width = .01 rad; thumb 1.05 is .0028 past the 60 deg stop); D3/D4 drive it
    # WELL PAST the width, so the row switches on at full impedance .99.
    #
    # ⚠ D3/D4 ARE THE POSES THIS BUCKET COULD NOT HOLD UNTIL 2026-08-02, and
    # they are here because the defect that excluded them is fixed. They used
    # to give a rel qacc error of 2.65 / 4.03 and were pinned by an inverted
    # test asserting exactly that. The cause was never the limit: with the hand
    # closed on itself the thumb2/finger2 capsule AXES CROSS, and our
    # capsule/capsule narrow phase resolved that degenerate normal to
    # `centre_B - centre_A` where MuJoCo uses `cross(axis_a, axis_b)`. A wrong
    # contact NORMAL changes the constraint cost, so the solver converged to a
    # different minimiser — at which the limit row correctly reported itself
    # satisfied, which is what made it look like a limit bug. Keep these poses:
    # they are the only ones in the file that exercise a fully-engaged limit
    # against live contacts.
    tags.append(String("D1 closed hand, self-contact + limit"))
    states.append(
        _pose_state(0, 0, 0, 0, 1.05, 0.0, PEG_FAR_X, PEG_FAR_Z, 0.0, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("D2 grasp + limit, driven"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 1.05, 0.0, GRASP_X, GRASP_Z + 0.02,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append([0.0, 0.0, 0.0, 0.0, 0.6])
    tags.append(String("D3 grasp + limit PAST its impedance width"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 1.06, 0.0, GRASP_X, GRASP_Z + 0.02,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append(_zero_ctrl())
    tags.append(String("D4 grasp + limit far past the width"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 1.10, 0.0, GRASP_X, GRASP_Z + 0.02,
                    PEG_HELD_ANGLE, False)
    )
    ctrls.append(_zero_ctrl())

    var worst_by_bucket = List[Float64]()
    var count_by_bucket = List[Int]()
    for _ in range(4):
        worst_by_bucket.append(0.0)
        count_by_bucket.append(0)
    var max_ncon = 0

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
        if mj_ncon > max_ncon:
            max_ncon = mj_ncon
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

    # MAX_CONTACTS headroom. Truncation is SILENT in a rollout — the `ncon`
    # assertion above only covers these ten poses — so report the margin.
    print("  max MuJoCo ncon over the pose table =", max_ncon,
          " MAX_CONTACTS =", MAXC)
    assert_true(
        max_ncon < MAXC,
        "the pose table already reaches MAX_CONTACTS; a rollout would truncate"
        " contacts silently",
    )


def test_bring_peg_observation_matches_mujoco() raises:
    """All 44 observation entries, at a pose that makes every term non-trivial.

    Structurally identical to bring_ball's observation — same 44 slots, same
    blocks — but reading a DIFFERENT prop body, a different target body and a
    different joint triple. That is precisely the kind of change that a
    shared-code port gets wrong silently, since nothing about the shape moves.

    The pose is deliberately ASYMMETRIC in the hand: `_ARM_JOINTS` lists
    finger/fingertip BEFORE thumb/thumbtip while the model declares the thumb
    chain first, and a symmetric hand would hide a wrong permutation.
    """
    var mf = _build()
    var d = Dat()
    var integ = Integ()
    # Grasping AND asymmetric, found by sweeping MuJoCo for a pose that both
    # loads the touch sensors and breaks the hand's symmetry.
    var state = _pose_state(
        0.0, ARM1, ARM2, ARM3, 0.50, 0.0,
        GRASP_X, GRASP_Z + 0.02, PEG_HELD_ANGLE, False,
    )
    state[6] = 0.62  # finger    != thumb
    state[7] = 0.12  # fingertip != thumbtip
    var ctrl = _zero_ctrl()
    _set_state_and_fk(d, mf, integ, state, ctrl)

    var mj = _mj_at(state, ctrl)
    var m = mj[1]
    var dat = mj[2]
    var builder = Python.import_module("manipulator_ref")
    var ref_obs = builder.observation(m, dat, True)

    var obs = List[Scalar[DTYPE]]()
    _ = CFG.custom_extract_obs_cpu[DTYPE](
        d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
        List[Scalar[DTYPE]](), obs,
    )
    assert_true(len(obs) == 44, String("obs dim ") + String(len(obs)) + " != 44")

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

    # Non-vacuity, both halves. The touch block must be loaded, or the five
    # entries gate nothing; and `object_pos` must not be all zeros, which is
    # what reading the WRONG body index would most likely produce.
    var touch_sum = Float64(0)
    var touch_nz = 0
    for i in range(24, 29):
        var v = Float64(py=ref_obs[i])
        touch_sum += v
        if v > 0.0:
            touch_nz += 1
    print("  MuJoCo touch block: sum =", touch_sum, " nonzero zones =", touch_nz)
    assert_true(
        touch_nz >= 2,
        "fewer than two touch zones are loaded at this pose, so the five touch"
        " entries — and the BOX zone code behind them — are barely gated",
    )
    var obj_mag = Float64(0)
    for i in range(33, 37):
        obj_mag += abs(Float64(py=ref_obs[i]))
    assert_true(
        obj_mag > 0.1,
        "the object_pos block is ~zero, so it cannot distinguish the peg body"
        " from an unwritten one",
    )
    print("  worst |d obs| =", worst_all)


def test_bring_peg_reward_matches_mujoco() raises:
    """`_peg_reward` — the one piece of task logic bring_ball cannot gate.

        max(bringing, grasping/3)
          grasping = mean(is_close(peg_grasp, grasp), is_close(peg_pinch, pinch))
          bringing = mean(is_close(peg, target_peg), is_close(target_peg_tip, peg_tip))

    Three things can independently be wrong — which sites pair with which, the
    `max`, and the `/3` — so the poses are chosen to separate them rather than
    to sample the curve evenly:

      * peg AT the target, four offsets: `bringing` alone drives the reward,
        so this is the term-by-term analogue of bring_ball's reward test and
        spans the tolerance curve from saturated (1.0) to ~0.
      * peg HELD, target far: `bringing` is 0 and `grasping` is 1, so the
        reward must be exactly 1/3. Getting the `/3` wrong reads 1.0 or 0.5
        here and nothing else in the file would notice.
      * peg held LOOSELY (both site pairs 2 cm apart): `grasping` is partial,
        so this separates "the /3 is applied" from "the max picked grasping".
    """
    var mf = _build()
    var integ = Integ()
    var builder = Python.import_module("manipulator_ref")

    var tags = List[String]()
    var states = List[List[Float64]]()

    # Bringing: peg parked at the target, offset in z. `peg_y = 0` matches the
    # target's own orientation, so `peg_tip` lines up with `target_peg_tip`.
    var offsets = [0.0, 0.015, 0.03, 0.12]
    for k in range(len(offsets)):
        tags.append(String("bring dz=") + String(offsets[k]))
        states.append(
            _pose_state(0, 0, 0, 0, 0.0, 0.0, TARGET_X, TARGET_Z + offsets[k],
                        0.0, False)
        )

    # Grasping, exact: both site pairs coincident, target far away.
    tags.append(String("grasp exact (expect 1/3)"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 0.35, 0.0, GRASP_X, GRASP_Z,
                    PEG_HELD_ANGLE, False)
    )
    # Grasping, partial.
    tags.append(String("grasp loose"))
    states.append(
        _pose_state(0, ARM1, ARM2, ARM3, 0.50, 0.0, GRASP_X, GRASP_Z + 0.02,
                    PEG_HELD_ANGLE, False)
    )

    var worst = Float64(0)
    var saw_third = False
    for k in range(len(tags)):
        var d = Dat()
        _set_state_and_fk(d, mf, integ, states[k], _zero_ctrl())
        var mj = _mj_at(states[k], _zero_ctrl())
        var ref_r = Float64(py=builder.reward(mj[1], mj[2], True))
        var got = CFG.compute_reward_and_done_cpu[
            DTYPE, NQ, NV, NBODY, MAXC, NSITE
        ](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            Scalar[DTYPE](0), _zero_ctrl(), 0, 1,
        )
        var e = abs(Float64(got[0]) - ref_r)
        if e > worst:
            worst = e
        print("   ", tags[k], " ours =", got[0], " MuJoCo =", ref_r)
        assert_true(not got[1], "dm_control tasks never terminate early")
        assert_true(
            e <= TOL_OBS,
            String("peg reward diverges at ") + tags[k],
        )
        if tags[k] == String("grasp exact (expect 1/3)"):
            saw_third = True
            # The `/3` is the whole point of this pose: assert the VALUE, not
            # just agreement with the reference, because a shared misreading
            # would agree perfectly.
            assert_true(
                abs(Float64(got[0]) - 1.0 / 3.0) < 1e-9,
                "a fully grasped peg with the target far away must score"
                " exactly 1/3; this pose is what pins the `grasping/3` factor",
            )

    assert_true(saw_third, "the 1/3 pose vanished from the table")
    print("  worst |d reward| =", worst)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
