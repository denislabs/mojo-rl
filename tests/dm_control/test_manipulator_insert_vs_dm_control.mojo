"""dm_control `manipulator-insert_ball` / `insert_peg` parity vs MuJoCo.

`make_model(insert=True)` splices a RECEPTACLE — `cup` for the ball, `slot`
for the peg — between the prop and the target, renumbering bodies, geoms and
sites yet again. That is the third and fourth of the domain's four models.

WHAT THESE TWO ADD THAT THE `bring` PAIR DOES NOT
-------------------------------------------------
The observation is byte-identical in shape and the reward is literally the same
function (`get_reward` never mentions the receptacle — inserting is rewarded
only through bringing), so this file does NOT re-gate either. Both are already
pinned against MuJoCo for both props by the two `bring` files, and the config
reads them through the same `(USE_PEG, INSERT)` index helpers, which this file
pins against the model.

What is new is one engine capability: **a COLLIDING MOCAP BODY**.

`Bring.initialize_episode` randomises the receptacle's pose every episode with
`model.body_pos[...]` / `model.body_quat[...]`, and `fields.Model` is a single
shared unbatched tensor set, so — exactly as for the target — the pose has to
live in `d.mocap_pos`/`d.mocap_quat` instead. But every mocap body ported
before these two was INERT: reacher's and finger's targets are `contype=0`
decorations and SawyerReach's weld anchor has no geom at all. `cup` and `slot`
are `class="obstacle"` with default collision masks, and the whole task is the
prop hitting them.

Nothing in the engine special-cases a mocap body's geoms — the narrow phase
derives geom world poses from `xpos`/`xquat`, which `_sync_mocap_to_fields`
presets, and a jointless child of the world contributes no dof either way — so
this is expected to work. It had simply never been executed, which per the
narrow-phase coverage audit is not the same thing. Hence
`test_insert_*_receptacle_collides_at_its_mocap_pose`, which drives the prop
into the receptacle at a NON-DEFAULT mocap pose and requires that our contacts
match MuJoCo's. A receptacle silently left at its XML pose, or silently not
colliding, produces zero contacts and would otherwise look like a working env.

Run with:
    pixi run mojo run -I . tests/dm_control/test_manipulator_insert_vs_dm_control.mojo
"""

from std.math import abs, pi, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulator import (
    DMManipulatorInsertBallModel as MB,
    DMManipulatorInsertPegModel as MP,
    target_body_idx,
    receptacle_body_idx,
    site_object,
    site_target,
    OBJECT_BODY_IDX,
    OBJECT_QADR_X,
    OBJECT_QADR_Z,
    OBJECT_QADR_Y,
)
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_MOCAP,
    BODY_IDX_MASS,
    GEOM_IDX_BODY,
    GEOM_IDX_CONTYPE,
    SITE_IDX_BODY,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_DIST,
)


comptime DTYPE = DType.float64
comptime TOL_MODEL: Float64 = 1e-9

# The receptacle pose the tests drive: deliberately NOT the XML pose, and
# deliberately ROTATED, because a mocap pose that is never written and a mocap
# quaternion that is ignored both look identical at the default pose.
comptime RECEPTACLE_X: Float64 = 0.22
comptime RECEPTACLE_Z: Float64 = 0.31
comptime RECEPTACLE_ANGLE: Float64 = 0.35  # rad about y, |.| < pi/3 per `insert`
comptime TARGET_PARK_X: Float64 = -0.45
comptime TARGET_PARK_Z: Float64 = 0.62


def _ref(use_peg: Bool) raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/dm_control")
    var builder = Python.import_module("manipulator_ref")
    return builder.model(use_peg, True)


# ── insert_ball ─────────────────────────────────────────────────────────────

comptime BNQ: Int = MB.NQ
comptime BNV: Int = MB.NV
comptime BNBODY: Int = MB.NBODY  # 13
comptime BNJOINT: Int = MB.NJOINT
comptime BNGEOM: Int = MB.NGEOM  # 25
comptime BNSITE: Int = MB.NSITE  # 10
comptime BMAXC: Int = MB.MAX_CONTACTS
comptime BModel = Model[
    DTYPE, BNV, BNBODY, BNJOINT, BNGEOM, MB.MAX_EQUALITY, MB.MAX_TENDON,
    BNSITE, MB.NEXCLUDE, 0,
]
comptime BData = Data[DTYPE, BNQ, BNV, BNBODY, BMAXC, BNSITE, 1]
comptime BInteg = EulerIntegrator[
    DTYPE, BNQ, BNV, BNBODY, BNJOINT, BMAXC, BNGEOM, MB.MAX_EQUALITY,
    MB.MAX_TENDON, BNSITE, MB.NEXCLUDE, 0, MB.CONE_TYPE, 1, SOLVER="newton",
]

# ── insert_peg ──────────────────────────────────────────────────────────────

comptime PNQ: Int = MP.NQ
comptime PNV: Int = MP.NV
comptime PNBODY: Int = MP.NBODY  # 14
comptime PNJOINT: Int = MP.NJOINT
comptime PNGEOM: Int = MP.NGEOM  # 28
comptime PNSITE: Int = MP.NSITE  # 17
comptime PMAXC: Int = MP.MAX_CONTACTS
comptime PModel = Model[
    DTYPE, PNV, PNBODY, PNJOINT, PNGEOM, MP.MAX_EQUALITY, MP.MAX_TENDON,
    PNSITE, MP.NEXCLUDE, 0,
]
comptime PData = Data[DTYPE, PNQ, PNV, PNBODY, PMAXC, PNSITE, 1]
comptime PInteg = EulerIntegrator[
    DTYPE, PNQ, PNV, PNBODY, PNJOINT, PMAXC, PNGEOM, MP.MAX_EQUALITY,
    MP.MAX_TENDON, PNSITE, MP.NEXCLUDE, 0, MP.CONE_TYPE, 1, SOLVER="newton",
]


def _check_names(
    mj: PythonObject, kind: PythonObject, names: List[String], swap12: Bool
) raises:
    """Every name resolves to its position in `names`, on the MuJoCo side.

    `swap12` covers the arm's one site-order divergence (`palm_touch` is
    declared after the `pinch site` body but belongs to `hand`, so MuJoCo's
    body sort pulls it ahead of `pinch`), which is identical in all four
    variants because it lives in the shared arm segment.
    """
    var mujoco = Python.import_module("mujoco")
    for i in range(len(names)):
        var want = i
        if swap12:
            if i == 1:
                want = 2
            elif i == 2:
                want = 1
        assert_true(
            Int(py=mujoco.mj_name2id(mj, kind, names[i])) == want,
            String("MuJoCo order moved at ") + names[i],
        )


def test_insert_ball_model_matches_mujoco() raises:
    """Counts, order and the index helpers, for `cup` between ball and target.
    """
    var mj = _ref(False)
    var mujoco = Python.import_module("mujoco")
    assert_true(Int(py=mj.nq) == BNQ, "nq mismatch")
    assert_true(Int(py=mj.nbody) == BNBODY, "nbody mismatch")
    assert_true(Int(py=mj.ngeom) == BNGEOM, "ngeom mismatch")
    assert_true(Int(py=mj.nsite) == BNSITE, "nsite mismatch")
    assert_true(Int(py=mj.nu) == MB.nact, "nu mismatch")
    assert_true(Int(py=mj.neq) == 1, "neq mismatch (the coupling equality)")

    _check_names(mj, mujoco.mjtObj.mjOBJ_BODY, [
        String("world"), String("upper_arm"), String("middle_arm"),
        String("lower_arm"), String("hand"), String("pinch site"),
        String("thumb"), String("thumbtip"), String("finger"),
        String("fingertip"), String("ball"), String("cup"),
        String("target_ball"),
    ], False)
    _check_names(mj, mujoco.mjtObj.mjOBJ_SITE, [
        String("grasp"), String("pinch"), String("palm_touch"),
        String("thumb_touch"), String("thumbtip_touch"),
        String("finger_touch"), String("fingertip_touch"),
        String("ball"), String("cup"), String("target_ball"),
    ], True)

    # The helpers the config reads. `cup` sits BETWEEN the prop and the target,
    # so getting the segment order wrong renumbers both and the reward silently
    # measures the distance to a receptacle.
    assert_true(OBJECT_BODY_IDX == 10, "prop body moved")
    assert_true(receptacle_body_idx(False) == 11, "cup body index wrong")
    assert_true(target_body_idx(False, True) == 12, "target body index wrong")
    assert_true(site_object(False) == 7, "ball site index wrong")
    assert_true(site_target(False, True) == 9, "target_ball site index wrong")

    var ctx = DeviceContext()
    var mf = BModel()
    MB.init_fields[DTYPE, 0](ctx, mf)
    var bref = mj.body_parentid.tolist()
    for b in range(BNBODY):
        assert_true(
            Int(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            == Int(py=bref[b]),
            String("body_parentid mismatch on body ") + String(b),
        )
    var gref = mj.geom_bodyid.tolist()
    for g in range(BNGEOM):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
            == Int(py=gref[g]),
            String("geom_bodyid mismatch on geom ") + String(g),
        )
    var mref = mj.body_mass.tolist()
    for b in range(BNBODY):
        var ours = Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var r = Float64(py=mref[b])
        assert_true(
            abs(ours - r) / (1e-15 + abs(r)) <= TOL_MODEL,
            String("body_mass mismatch on body ") + String(b),
        )

    # The two claims the collision test depends on: the receptacle is a MOCAP
    # body on our side, and its geoms COLLIDE. Either being false turns that
    # test into a study of an object falling through nothing.
    assert_true(
        mf.bodies.data[11 * MODEL_BODY_SIZE + BODY_IDX_MOCAP] != 0,
        "`cup` is not a mocap body — its per-episode pose would be a MODEL"
        " write, shared across the whole batch",
    )
    for g in range(20, 24):  # cup_0 .. cup_3
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) != 0,
            String("cup geom ") + String(g) + " does not collide",
        )


def test_insert_peg_model_matches_mujoco() raises:
    """Counts, order and the index helpers, for `slot` between peg and target.

    This is the largest model in the domain: the peg's `pommel` sub-body and
    the slot's three boxes push it to 14 bodies, 28 geoms and 17 sites.
    """
    var mj = _ref(True)
    var mujoco = Python.import_module("mujoco")
    assert_true(Int(py=mj.nbody) == PNBODY, "nbody mismatch")
    assert_true(Int(py=mj.ngeom) == PNGEOM, "ngeom mismatch")
    assert_true(Int(py=mj.nsite) == PNSITE, "nsite mismatch")

    _check_names(mj, mujoco.mjtObj.mjOBJ_BODY, [
        String("world"), String("upper_arm"), String("middle_arm"),
        String("lower_arm"), String("hand"), String("pinch site"),
        String("thumb"), String("thumbtip"), String("finger"),
        String("fingertip"), String("peg"), String("pommel"),
        String("slot"), String("target_peg"),
    ], False)
    _check_names(mj, mujoco.mjtObj.mjOBJ_SITE, [
        String("grasp"), String("pinch"), String("palm_touch"),
        String("thumb_touch"), String("thumbtip_touch"),
        String("finger_touch"), String("fingertip_touch"),
        String("peg"), String("peg_pinch"), String("peg_grasp"),
        String("peg_tip"), String("slot"), String("slot_end"),
        String("target_peg"), String("target_peg_pinch"),
        String("target_peg_grasp"), String("target_peg_tip"),
    ], True)

    assert_true(receptacle_body_idx(True) == 12, "slot body index wrong")
    assert_true(target_body_idx(True, True) == 13, "target body index wrong")
    assert_true(site_target(True, True) == 13, "target_peg site index wrong")

    var ctx = DeviceContext()
    var mf = PModel()
    MP.init_fields[DTYPE, 0](ctx, mf)
    var gref = mj.geom_bodyid.tolist()
    for g in range(PNGEOM):
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
            == Int(py=gref[g]),
            String("geom_bodyid mismatch on geom ") + String(g),
        )
    # OUR site order IS MuJoCo's, as of the element-order fix (2026-08-03).
    #
    # This loop used to swap sites 1 and 2: `palm_touch` is declared AFTER the
    # `pinch site` body but belongs to `hand`, so MuJoCo's body sort pulled it
    # ahead of `pinch` while our XML-text walk left it behind. `full_parser`
    # now groups joints, geoms and sites by body id
    # (`_stable_group_by_body_*`), so the comparison is elementwise.
    #
    # ⚠ THE SWAP WAS WRITTEN INLINE HERE rather than through the
    # `_our_site_to_mj` helper its sibling files use, which is exactly why the
    # sweep that removed those missed this one — grepping for the helper name
    # cannot find a workaround that was open-coded. It surfaced as
    # `site_bodyid mismatch on site 1` with every physics number still green.
    var sref = mj.site_bodyid.tolist()
    for s in range(PNSITE):
        assert_true(
            Int(mf.sites.data[s * MODEL_SITE_SIZE + SITE_IDX_BODY])
            == Int(py=sref[s]),
            String("site_bodyid mismatch on site ") + String(s),
        )
    assert_true(
        mf.bodies.data[12 * MODEL_BODY_SIZE + BODY_IDX_MOCAP] != 0,
        "`slot` is not a mocap body",
    )
    for g in range(22, 25):  # slot_0 .. slot_2
        assert_true(
            Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONTYPE]) != 0,
            String("slot geom ") + String(g) + " does not collide",
        )


# ── the colliding mocap receptacle ──────────────────────────────────────────


def _sorted_dists(vals: List[Float64]) -> List[Float64]:
    """Ascending copy — contact ORDER is not part of either engine's contract,
    so the two sets are compared as multisets of penetration depth."""
    var out = List[Float64]()
    for i in range(len(vals)):
        out.append(vals[i])
    for i in range(1, len(out)):
        var v = out[i]
        var j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    return out^


def _mj_dists(dat: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for c in range(Int(py=dat.ncon)):
        out.append(Float64(py=dat.contact[c].dist))
    return _sorted_dists(out)


def _mj_insert(
    use_peg: Bool,
    ox: Float64, oz: Float64, oa: Float64,
) raises -> Tuple[PythonObject, PythonObject]:
    """MuJoCo at the same state, from OUR xml, with both mocap bodies posed.

    MuJoCo indexes `mocap_pos` by `body_mocapid`, not by body id, so the two
    mocap bodies have to be looked up rather than assumed adjacent.
    """
    var mujoco = Python.import_module("mujoco")
    var xml = String(
        "mojo_rl/envs/dm_control/assets/manipulator_insert_peg.xml"
    ) if use_peg else String(
        "mojo_rl/envs/dm_control/assets/manipulator_insert_ball.xml"
    )
    var m = mujoco.MjModel.from_xml_path(String(xml))
    var dat = mujoco.MjData(m)
    var rb = receptacle_body_idx(use_peg)
    var tb = target_body_idx(use_peg, True)
    var r_mid = Int(py=m.body_mocapid[rb])
    var t_mid = Int(py=m.body_mocapid[tb])
    assert_true(
        r_mid >= 0 and t_mid >= 0,
        "MuJoCo does not see both bodies as mocap — the XML lost `mocap=true`",
    )
    dat.mocap_pos[r_mid][0] = RECEPTACLE_X
    dat.mocap_pos[r_mid][1] = 0.0
    dat.mocap_pos[r_mid][2] = RECEPTACLE_Z
    dat.mocap_quat[r_mid][0] = cos(RECEPTACLE_ANGLE * 0.5)
    dat.mocap_quat[r_mid][1] = 0.0
    dat.mocap_quat[r_mid][2] = sin(RECEPTACLE_ANGLE * 0.5)
    dat.mocap_quat[r_mid][3] = 0.0
    # The target is parked far away so it cannot contribute anything; it is
    # `contype=0` regardless, but leaving it at the origin would put a ghost
    # on top of the arm and make the printout hard to read.
    dat.mocap_pos[t_mid][0] = TARGET_PARK_X
    dat.mocap_pos[t_mid][1] = 0.001
    dat.mocap_pos[t_mid][2] = TARGET_PARK_Z
    dat.mocap_quat[t_mid][0] = 1.0
    dat.mocap_quat[t_mid][1] = 0.0
    dat.mocap_quat[t_mid][2] = 0.0
    dat.mocap_quat[t_mid][3] = 0.0

    dat.qpos[OBJECT_QADR_X] = ox
    dat.qpos[OBJECT_QADR_Z] = oz
    dat.qpos[OBJECT_QADR_Y] = oa
    mujoco.mj_forward(m, dat)
    return (m, dat)


def test_insert_ball_receptacle_collides_at_its_mocap_pose() raises:
    """Drop the ball into the `cup` at a NON-DEFAULT, ROTATED mocap pose.

    The cup is moved from its XML `pos=".3 0 .4" euler="0 -15 0"` to
    (.22, .31) at +20 deg. Both changes matter: a receptacle whose pose is
    never read stays at the XML pose, and a receptacle whose mocap QUATERNION
    is ignored stays at -15 deg. Either produces a plausible-looking contact
    set at the wrong place, so the test asserts against MuJoCo driven from the
    same mocap values rather than against a contact count.

    The ball's height is SEARCHED downward until MuJoCo finds contacts, so the
    gate cannot go vacuous if the geometry shifts.
    """
    var sf = MB.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = BModel()
    MB.init_fields[DTYPE, 0](ctx, mf)
    var integ = BInteg()

    # Scan the whole band and take the DEEPEST engagement, not the first
    # touch. Stopping at the first contact gives a single grazing row, which
    # would exercise one cup capsule and call the receptacle gated.
    var oz = RECEPTACLE_Z + 0.06
    var best_z = oz
    var mj_ncon = 0
    while oz > RECEPTACLE_Z - 0.06:
        var mj = _mj_insert(False, RECEPTACLE_X, oz, 0.0)
        var n = Int(py=mj[1].ncon)
        if n > mj_ncon:
            mj_ncon = n
            best_z = oz
        oz -= 0.005
    var mjb = _mj_insert(False, RECEPTACLE_X, best_z, 0.0)
    var mj_d = _mj_dists(mjb[1])
    print("  ball z =", best_z, " MuJoCo ncon =", mj_ncon)
    assert_true(
        mj_ncon >= 2,
        "the ball touches the cup in at most one place anywhere in the swept"
        " band, so this test would gate a single capsule — the cup geometry or"
        " its mocap pose moved",
    )
    oz = best_z

    var d = BData()
    MB.reset_data(sf, d)
    d.qpos.data[OBJECT_QADR_X] = Scalar[DTYPE](RECEPTACLE_X)
    d.qpos.data[OBJECT_QADR_Z] = Scalar[DTYPE](oz)
    d.qpos.data[OBJECT_QADR_Y] = Scalar[DTYPE](0)
    for i in range(BNV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    _pose_mocap_ball(d)
    var zero = List[Float64]()
    for _ in range(MB.ACTION_DIM):
        zero.append(0.0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(MB.NA if MB.NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    MB.apply_actions(sf, d, zero, act)
    integ.step["cpu"](d, mf)
    var our_ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var ours = List[Float64]()
    for c in range(our_ncon):
        ours.append(
            Float64(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_DIST])
        )
    var our_d = _sorted_dists(ours)
    print("  our ncon =", our_ncon)
    assert_true(
        our_ncon == mj_ncon,
        String("our ncon ") + String(our_ncon) + " != MuJoCo " + String(mj_ncon)
        + " — the cup either is not at its mocap pose or is not colliding",
    )
    # ncon agreeing is NOT the row set agreeing — that distinction is what the
    # bring_peg limit investigation turned on. Compare the penetration depths.
    var worst = Float64(0)
    for i in range(our_ncon):
        var e = abs(our_d[i] - mj_d[i])
        if e > worst:
            worst = e
    print("  worst |d dist| =", worst, " deepest MuJoCo dist =", mj_d[0])
    assert_true(
        worst <= 1e-9,
        "our contact DEPTHS diverge from MuJoCo's even though the counts"
        " match — the cup is colliding, but not where MuJoCo has it",
    )


def _pose_mocap_ball(mut d: BData):
    """Write both mocap poses and preset the world poses FK skips."""
    _set_mocap[BNQ, BNV, BNBODY, BMAXC, BNSITE](
        d, receptacle_body_idx(False),
        RECEPTACLE_X, 0.0, RECEPTACLE_Z, RECEPTACLE_ANGLE,
    )
    _set_mocap[BNQ, BNV, BNBODY, BMAXC, BNSITE](
        d, target_body_idx(False, True),
        TARGET_PARK_X, 0.001, TARGET_PARK_Z, 0.0,
    )


def _pose_mocap_peg(mut d: PData):
    _set_mocap[PNQ, PNV, PNBODY, PMAXC, PNSITE](
        d, receptacle_body_idx(True),
        RECEPTACLE_X, 0.0, RECEPTACLE_Z, RECEPTACLE_ANGLE,
    )
    _set_mocap[PNQ, PNV, PNBODY, PMAXC, PNSITE](
        d, target_body_idx(True, True),
        TARGET_PARK_X, 0.001, TARGET_PARK_Z, 0.0,
    )


def _set_mocap[
    NQ: Int, NV: Int, NBODY: Int, MAXC: Int, NSITE: Int
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    body: Int,
    x: Float64, y: Float64, z: Float64, angle: Float64,
):
    """`Phyics3dEnvConfig` hook write + the facade's `_sync_mocap_to_fields`.

    FK SKIPS mocap bodies, so `mocap_pos`/`mocap_quat` alone leave the body at
    the origin; the facade presets `xpos`/`xipos`/`xquat` before stepping and
    these tests drive `integ.step` directly, so they must do the same. For an
    INERT mocap body that mistake costs a wrong observation. For a COLLIDING
    one it costs every contact in the model.
    """
    d.mocap_pos.data[body * 3 + 0] = Scalar[DTYPE](x)
    d.mocap_pos.data[body * 3 + 1] = Scalar[DTYPE](y)
    d.mocap_pos.data[body * 3 + 2] = Scalar[DTYPE](z)
    d.mocap_quat.data[body * 4 + 0] = Scalar[DTYPE](0)
    d.mocap_quat.data[body * 4 + 1] = Scalar[DTYPE](sin(angle * 0.5))
    d.mocap_quat.data[body * 4 + 2] = Scalar[DTYPE](0)
    d.mocap_quat.data[body * 4 + 3] = Scalar[DTYPE](cos(angle * 0.5))
    for k in range(3):
        var p = d.mocap_pos.data[body * 3 + k]
        d.xpos.data[body * 3 + k] = p
        d.xipos.data[body * 3 + k] = p
    for k in range(4):
        d.xquat.data[body * 4 + k] = d.mocap_quat.data[body * 4 + k]


def test_insert_peg_receptacle_collides_at_its_mocap_pose() raises:
    """Same, driving the peg into the `slot`.

    Harder than the cup case in one respect: the slot's walls are BOXES and
    the peg's blade is a CAPSULE, so this also exercises a capsule/box pair —
    one of the type pairs the narrow-phase coverage audit found had never
    executed outside its own synthetic gate.
    """
    var sf = MP.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = PModel()
    MP.init_fields[DTYPE, 0](ctx, mf)
    var integ = PInteg()

    # Scan for the pose with the most MuJoCo contacts at a PHYSICAL depth.
    #
    # ⚠ THIS USED TO ALSO REQUIRE ONE POINT PER PAIR, and that restriction is
    # gone with the gap it existed for. Our capsule/box narrow phase emitted
    # one contact per colliding pair where `mjc_CapsuleBox` emits two for a
    # capsule lying along a box face, so poses where MuJoCo doubled up would
    # have failed this gate for a reason it does not measure; they were pinned
    # separately by `test_insert_peg_capsule_box_multipoint_is_an_open_defect`,
    # which is deleted now that `box_capsule_manifold` emits the second point.
    # Dropping the filter is what widens this gate from single-pair poses to
    # the FOUR-pair, seven-contact one — the peg actually lying in the slot,
    # which is the terminal state insert_peg is about.
    #
    # The remaining filter stands on its own: MuJoCo CLAMPS a capsule/box
    # penetration at the box's half-thickness plus the capsule radius (-.015
    # here, and the sweep shows that value appearing verbatim over whole bands
    # of z). Depth parity inside a clamped, 3 cm interpenetration is a
    # different question from contact parity, and not one this file is about.
    var oz = RECEPTACLE_Z + 0.10
    var best_z = oz
    var mj_ncon = 0
    var found = False
    while oz > RECEPTACLE_Z - 0.10:
        var mj = _mj_insert(True, RECEPTACLE_X, oz, RECEPTACLE_ANGLE)
        var n = Int(py=mj[1].ncon)
        if n > 0:
            var deepest = _mj_dists(mj[1])[0]
            if abs(deepest) <= 0.008 and n >= mj_ncon:
                mj_ncon = n
                best_z = oz
                found = True
        oz -= 0.005
    var mjb = _mj_insert(True, RECEPTACLE_X, best_z, RECEPTACLE_ANGLE)
    var mj_d = _mj_dists(mjb[1])
    print("  peg z =", best_z, " MuJoCo ncon =", mj_ncon)
    assert_true(
        found and mj_ncon >= 1,
        "no pose in the swept band has a contact at a physical depth, so this"
        " gate has nothing to stand on — the slot geometry or its mocap pose"
        " moved",
    )
    oz = best_z

    var d = PData()
    MP.reset_data(sf, d)
    d.qpos.data[OBJECT_QADR_X] = Scalar[DTYPE](RECEPTACLE_X)
    d.qpos.data[OBJECT_QADR_Z] = Scalar[DTYPE](oz)
    d.qpos.data[OBJECT_QADR_Y] = Scalar[DTYPE](RECEPTACLE_ANGLE)
    for i in range(PNV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    _pose_mocap_peg(d)
    var zero = List[Float64]()
    for _ in range(MP.ACTION_DIM):
        zero.append(0.0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(MP.NA if MP.NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    MP.apply_actions(sf, d, zero, act)
    integ.step["cpu"](d, mf)
    var our_ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var ours = List[Float64]()
    for c in range(our_ncon):
        ours.append(
            Float64(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_DIST])
        )
    var our_d = _sorted_dists(ours)
    print("  our ncon =", our_ncon)
    assert_true(
        our_ncon == mj_ncon,
        String("our ncon ") + String(our_ncon) + " != MuJoCo " + String(mj_ncon)
        + " — the slot either is not at its mocap pose or is not colliding",
    )
    var worst = Float64(0)
    for i in range(our_ncon):
        var e = abs(our_d[i] - mj_d[i])
        if e > worst:
            worst = e
    print("  worst |d dist| =", worst, " deepest MuJoCo dist =", mj_d[0])
    assert_true(
        worst <= 1e-9,
        "our contact DEPTHS diverge from MuJoCo's even though the counts"
        " match — the slot is colliding, but not where MuJoCo has it",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
