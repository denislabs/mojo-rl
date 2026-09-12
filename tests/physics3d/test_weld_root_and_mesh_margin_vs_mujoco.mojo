"""AUD-35 (mocap bodies are their own weld root; dof-less pairs skip) and
AUD-33 (a {box, mesh} pair with margin takes the perturbation loop) vs 3.12.

    pixi run mojo run -I . tests/physics3d/test_weld_root_and_mesh_margin_vs_mujoco.mojo

AUD-35. 3.12 (commit ed13bf56) made a mocap body its own weld root and added
"both dof-less => skip" to the body-pair filter. Before, a mocap hand
inherited weld id 0 and collided with its own jointed finger, since the
parent-child clause is guarded on both weld ids being non-world. The fixture
overlaps hand/finger (must NOT collide), hand/static box (both dof-less:
must NOT collide) and hand/free ball (must collide): MuJoCo answers exactly
one contact, the old code two.

AUD-33. `maxContacts` returns 1 whenever either object carries a margin
(engine_collision_convex.c:856-859), so a mesh/box or mesh/mesh pair with
`margin > 0` goes through the perturbation loop and reports up to five
points; ours reported one. Same fixtures as `test_mesh_manifold_vs_mujoco`,
with `margin="0.01"` on every geom, at a face-aligned 5 mm overlap.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.mocap import reset_mocap_from_model
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_DIST,
)

comptime DTYPE = DType.float64

comptime WELD_XML = """
<mujoco model="weld root">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="static_box" pos="0 0 0.1">
      <geom name="gstatic" type="box" size="0.3 0.3 0.1"/>
    </body>
    <body name="hand" mocap="true" pos="0 0 0.35">
      <geom name="ghand" type="sphere" size="0.2"/>
      <body name="finger" pos="0.25 0 0">
        <joint name="jf" type="hinge" axis="0 0 1"/>
        <geom name="gfinger" type="sphere" size="0.1"/>
      </body>
    </body>
    <body name="ball" pos="0 0 0.68">
      <joint name="jball" type="free"/>
      <geom name="gball" type="sphere" size="0.15"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime wp = parse_xml(WELD_XML)
comptime WM = ModelDefFromXML[
    xml=WELD_XML,
    nbody=wp.NBODY, njoint=wp.NJOINT, nq=wp.NQ, nv=wp.NV,
    ngeom=wp.NGEOM, nact=wp.NACT, ntex=wp.NTEX, nmat=wp.NMAT,
    nlight=wp.NLIGHT, ncam=wp.NCAM, nsite=wp.NSITE,
    max_tendon=wp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=wp.TIMESTEP,
]
comptime WMD = ModelDims[WM]

comptime MM_XML = """
<mujoco model="mesh margin">
  <option timestep="0.002"/>
  <default>
    <geom margin="0.01"/>
  </default>
  <asset>
    <mesh name="cube" file="tests/physics3d/assets/mc_cube.stl"/>
    <mesh name="hex" file="tests/physics3d/assets/mc_hex.stl"/>
  </asset>
  <worldbody>
    <body name="a0" pos="0 0 0.5">
      <geom name="g0a" type="mesh" mesh="cube"/>
    </body>
    <body name="b0" pos="0 0 0.5">
      <joint name="j0" type="free"/>
      <geom name="g0b" type="box" size=".05 .04 .06"/>
    </body>
    <body name="a1" pos="2 0 0.5">
      <geom name="g1a" type="box" size=".05 .04 .06"/>
    </body>
    <body name="b1" pos="2 0 0.5">
      <joint name="j1" type="free"/>
      <geom name="g1b" type="mesh" mesh="cube"/>
    </body>
    <body name="a2" pos="4 0 0.5">
      <geom name="g2a" type="mesh" mesh="cube"/>
    </body>
    <body name="b2" pos="4 0 0.5">
      <joint name="j2" type="free"/>
      <geom name="g2b" type="mesh" mesh="cube"/>
    </body>
    <body name="a3" pos="6 0 0.5">
      <geom name="g3a" type="mesh" mesh="hex"/>
    </body>
    <body name="b3" pos="6 0 0.5">
      <joint name="j3" type="free"/>
      <geom name="g3b" type="box" size=".05 .04 .06"/>
    </body>
    <body name="a4" pos="8 0 0.5">
      <geom name="g4a" type="mesh" mesh="hex"/>
    </body>
    <body name="b4" pos="8 0 0.5">
      <joint name="j4" type="free"/>
      <geom name="g4b" type="mesh" mesh="cube"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime mm = parse_xml(MM_XML)
comptime MMM = ModelDefFromXML[
    xml=MM_XML,
    nbody=mm.NBODY, njoint=mm.NJOINT, nq=mm.NQ, nv=mm.NV,
    ngeom=mm.NGEOM, nact=mm.NACT, ntex=mm.NTEX, nmat=mm.NMAT,
    nlight=mm.NLIGHT, ncam=mm.NCAM, nsite=mm.NSITE,
    max_tendon=mm.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=mm.TIMESTEP,
]
comptime MMD = ModelDims[MMM, 64]
comptime NGROUP = 5


def _stack_z(g: Int) -> Float64:
    if g == 0:
        return 0.05 + 0.06
    if g == 1:
        return 0.06 + 0.05
    if g == 2:
        return 0.05 + 0.05
    if g == 3:
        return 0.08 + 0.06
    return 0.08 + 0.05


def _mj() raises -> PythonObject:
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    return Python.import_module("mujoco")


def test_mocap_is_its_own_weld_root() raises:
    print("=== AUD-35: mocap weld root and the dof-less pair rule ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(WELD_XML)
    var dat = mujoco.MjData(m)
    mujoco.mj_forward(m, dat)
    var mjn = Int(py=dat.ncon)
    print("  MuJoCo body_weldid =", m.body_weldid, " ncon =", mjn)
    var hand = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"))
    var ball = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball"))
    assert_true(
        Int(py=m.body_weldid[hand]) == hand and mjn == 1,
        "fixture: MuJoCo must weld the mocap body to itself and report the"
        " single hand/ball contact (got ncon " + String(mjn) + ")",
    )
    var g1 = Int(py=dat.contact[0].geom1)
    var g2 = Int(py=dat.contact[0].geom2)
    var mb1 = Int(py=m.geom_bodyid[g1])
    var mb2 = Int(py=m.geom_bodyid[g2])
    assert_true(
        (mb1 == hand and mb2 == ball) or (mb1 == ball and mb2 == hand),
        "fixture: the one MuJoCo contact is not hand/ball",
    )

    var sf = WM.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, WMD]()
    WM.init_fields[DTYPE](ctx, mf)
    for leg in range(2):
        var d = Data[DTYPE, WMD, 1]()
        WM.reset_data[DTYPE](sf, d)
        # FK skips mocap bodies; their pose is seeded from the XML frame the
        # way `mj_resetData` seeds `mocap_pos` (see kinematics/mocap.mojo).
        var nmocap = reset_mocap_from_model(mf, d)
        assert_true(nmocap == 1, "the fixture's hand must be a mocap body")
        forward_kinematics["cpu"](d, mf)
        if leg == 0:
            detect_contacts["cpu"](d, mf)
        else:
            detect_contacts_sap["cpu"](d, mf)
        var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        var label = String("O(N^2)") if leg == 0 else String("SAP")
        print("  ", label, ": ours ncon =", nc)
        for c in range(nc):
            var o = c * CONTACT_SIZE
            print("     contact", c, " bodies", Int(d.contacts.data[o + CONTACT_IDX_BODY_A]),
                  Int(d.contacts.data[o + CONTACT_IDX_BODY_B]),
                  " dist", Float64(d.contacts.data[o + CONTACT_IDX_DIST]))
        assert_true(
            nc == 1,
            label + ": expected the single hand/ball contact, got " + String(nc)
            + " — a mocap hand colliding with its own finger, or with static"
            " geometry, is the pre-3.12 weld rule",
        )
        var o0 = 0
        var ba = Int(d.contacts.data[o0 + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[o0 + CONTACT_IDX_BODY_B])
        assert_true(
            (ba == hand and bb == ball) or (ba == ball and bb == hand),
            label + ": the contact is not hand/ball",
        )


def test_mesh_pairs_with_margin_take_the_perturbation_loop() raises:
    print("=== AUD-33: {box, mesh} pairs with margin -> up to 5 points ===")
    var mujoco = _mj()
    var m = mujoco.MjModel.from_xml_string(MM_XML)
    var dat = mujoco.MjData(m)

    var sf = MMM.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MMD]()
    MMM.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MMD, 1]()
    MMM.reset_data[DTYPE](sf, d)

    # every free body at a face-aligned 5 mm overlap over its partner
    for g in range(NGROUP):
        var z = 0.5 + _stack_z(g) - 0.005
        var x = 2.0 * Float64(g)
        dat.qpos[7 * g + 0] = x
        dat.qpos[7 * g + 1] = 0.0
        dat.qpos[7 * g + 2] = z
        dat.qpos[7 * g + 3] = 1.0
        dat.qpos[7 * g + 4] = 0.0
        dat.qpos[7 * g + 5] = 0.0
        dat.qpos[7 * g + 6] = 0.0
        d.qpos.data[7 * g + 0] = Scalar[DTYPE](x)
        d.qpos.data[7 * g + 1] = Scalar[DTYPE](0)
        d.qpos.data[7 * g + 2] = Scalar[DTYPE](z)
        d.qpos.data[7 * g + 3] = Scalar[DTYPE](1)
        d.qpos.data[7 * g + 4] = Scalar[DTYPE](0)
        d.qpos.data[7 * g + 5] = Scalar[DTYPE](0)
        d.qpos.data[7 * g + 6] = Scalar[DTYPE](0)
    mujoco.mj_forward(m, dat)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)
    var mjn = Int(py=dat.ncon)
    var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    print("  ncon MuJoCo =", mjn, " ours =", nc)

    var total_mj = 0
    for g in range(NGROUP):
        # bodies a_g = 2g+1, b_g = 2g+2
        var ba = 2 * g + 1
        var bb = 2 * g + 2
        var n_mj = 0
        for j in range(mjn):
            var c = dat.contact[j]
            var x1 = Int(py=m.geom_bodyid[Int(py=c.geom1)])
            var x2 = Int(py=m.geom_bodyid[Int(py=c.geom2)])
            if (x1 == ba and x2 == bb) or (x1 == bb and x2 == ba):
                n_mj += 1
        var n_ours = 0
        var worst = Float64(0)
        var worst_dist = Float64(0)
        var n_far = 0
        var far_dz = Float64(0)
        for k in range(nc):
            var o = k * CONTACT_SIZE
            var y1 = Int(d.contacts.data[o + CONTACT_IDX_BODY_A])
            var y2 = Int(d.contacts.data[o + CONTACT_IDX_BODY_B])
            if not ((y1 == ba and y2 == bb) or (y1 == bb and y2 == ba)):
                continue
            n_ours += 1
            var ox = Float64(d.contacts.data[o + CONTACT_IDX_POS_X])
            var oy = Float64(d.contacts.data[o + CONTACT_IDX_POS_Y])
            var oz = Float64(d.contacts.data[o + CONTACT_IDX_POS_Z])
            var od = Float64(d.contacts.data[o + CONTACT_IDX_DIST])
            var best = 1e30
            var bd = 1e30
            for j in range(mjn):
                var c = dat.contact[j]
                var ex = ox - Float64(py=c.pos[0])
                var ey = oy - Float64(py=c.pos[1])
                var ez = oz - Float64(py=c.pos[2])
                var rr = sqrt(ex * ex + ey * ey + ez * ez)
                if rr < best:
                    best = rr
                    bd = abs(od - Float64(py=c.dist))
            if best > 1e-5:
                # the un-perturbed INITIAL contact between two parallel faces
                # is EPA's witness coin (MuJoCo returns the face centre, ours
                # a point 2 cm along the face on mesh-over-box); it is not
                # what this gate is about, so it only has to lie in the
                # contact plane — same z as MuJoCo's points
                n_far += 1
                var mz = Float64(py=dat.contact[0].pos[2])
                for j in range(mjn):
                    var c = dat.contact[j]
                    var x1 = Int(py=m.geom_bodyid[Int(py=c.geom1)])
                    if x1 == ba or x1 == bb:
                        mz = Float64(py=c.pos[2])
                        break
                var dz = abs(oz - mz)
                if dz > far_dz:
                    far_dz = dz
            else:
                if best > worst:
                    worst = best
            if bd > worst_dist:
                worst_dist = bd
        print("  group", g, " MuJoCo", n_mj, " ours", n_ours,
              " matched worst |d pos|", worst, " |d dist|", worst_dist,
              " unmatched", n_far, " (plane |dz|", far_dz, ")")
        total_mj += n_mj
        assert_true(
            n_mj == n_ours,
            "group " + String(g) + ": " + String(n_ours) + " points vs MuJoCo's "
            + String(n_mj) + " — a mesh pair with margin must take the"
            " perturbation loop",
        )
        # the perturbation loop's witnesses converge to the ccd tolerance
        # (1e-6) on both sides; the polygon-clip gate is tighter because
        # its points are exact vertices
        assert_true(
            worst < 1e-5 and worst_dist < 1e-8 and n_far <= 1 and far_dz < 1e-6,
            "group " + String(g) + ": matched points off by " + String(worst)
            + " (dist " + String(worst_dist) + "), " + String(n_far)
            + " unmatched, plane offset " + String(far_dz) + " — the four"
            " PERTURBED points must coincide; only the initial witness may drift",
        )
    assert_true(
        total_mj > NGROUP,
        "the fixture gives MuJoCo one point per group; the multi-point path is"
        " not being exercised",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
