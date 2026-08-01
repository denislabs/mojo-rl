"""box/box narrow phase, swept over many poses (task #42, the remaining half).

WHAT THIS MEASURES, AND WHY IT COMES BEFORE THE FIX. `mjc_BoxBox` does two
things: it picks a separating axis (its `code`, from 6 face axes on each box
plus 9 edge-edge cross products), and it then builds a CONTACT MANIFOLD of up
to four points on that axis. Our `box_box` does the first half by SAT and emits
a single point.

So there are two independent questions, and the size of the remaining work
depends entirely on which of them is already right:

    normal   does our separating AXIS agree with MuJoCo's `code`?
    depth    does our penetration agree with MuJoCo's deepest point?
    count    how many points does MuJoCo put on that axis? (we emit 1)

If the normal already matches, the remaining work is manifold generation alone.
If it does not, the axis selection has to be ported too — and a manifold built
on the wrong axis would be worse than one point on the right one.

⚠ THIS FILE DELIBERATELY DOES NOT ASSERT THE COUNT. That is the open defect;
`test_stacker_box_box_is_one_point_per_pair` pins it. What this file gates is
that the single point we DO emit is on the right axis at the right depth, which
is the part a manifold port must not regress.

⚠ COMPARED AGAINST MuJoCo'S DEEPEST POINT, and that IS the right choice here,
unlike capsule/box: on a face manifold every point shares one normal, and the
deepest is the one a single-point reduction should represent. Stating it because
picking the wrong reference point cost a debugging cycle on the capsule/box
sweep.

Run: pixi run mojo run -I . tests/physics3d/test_box_box_sweep.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
)


comptime DTYPE = DType.float64

# Two DELIBERATELY UNEQUAL boxes with different shapes. Two cubes cannot
# distinguish an axis mix-up from a correct answer.
comptime BB_XML = """
<mujoco model="box box sweep">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0">
      <geom name="g1" type="box" size=".03 .02 .05"/>
    </body>
    <body name="b2" pos="0 0 0">
      <joint name="j2" type="free"/>
      <geom name="g2" type="box" size=".025 .04 .015"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime bb = parse_xml(BB_XML)
comptime BBM = ModelDefFromXML[
    xml=BB_XML,
    nbody=bb.NBODY, njoint=bb.NJOINT, nq=bb.NQ, nv=bb.NV,
    ngeom=bb.NGEOM, nact=bb.NACT, ntex=bb.NTEX, nmat=bb.NMAT,
    nlight=bb.NLIGHT, ncam=bb.NCAM, nsite=bb.NSITE,
    max_tendon=bb.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=bb.TIMESTEP,
]

comptime NQ: Int = BBM.NQ
comptime NV: Int = BBM.NV
comptime NBODY: Int = BBM.NBODY
comptime MC: Int = BBM.MAX_CONTACTS

comptime Dat = Data[DTYPE, NQ, NV, NBODY, MC, BBM.NSITE, 1]
comptime Mod = Model[
    DTYPE, NV, NBODY, BBM.NJOINT, BBM.NGEOM, BBM.MAX_EQUALITY, BBM.MAX_TENDON,
    BBM.NSITE, BBM.NEXCLUDE, 0,
]

comptime NPOSE: Int = 400
comptime SPAN: Float64 = 0.075

comptime TOL_DIST: Float64 = 1e-11
comptime TOL_DIR: Float64 = 1e-9


struct Lcg(Copyable, Movable):
    """A fixed 64-bit LCG, so a failing pose index is reproducible."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / 9007199254740992.0

    def sym(mut self, span: Float64) -> Float64:
        return (self.next() * 2.0 - 1.0) * span


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    BBM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_box_box_sweep_vs_mujoco() raises:
    """Separating axis and penetration depth over NPOSE box poses."""
    print("--- box/box sweep:", NPOSE, "poses")
    var mf = _build()
    var d = Dat()

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(BB_XML))
    var dat = mujoco.MjData(m)

    var rng = Lcg(0x9E3779B97F4A7C15)

    var n_touch = 0
    var n_ours = 0
    var n_both = 0
    var worst_dist = Float64(0)
    var worst_dir = Float64(0)
    var worst_dist_pose = -1
    var worst_dir_pose = -1
    var n_bad_dist = 0
    var n_bad_dir = 0
    # How many points MuJoCo puts on the manifold, so the size of the missing
    # half is reported rather than assumed.
    var mj_points = 0
    var mj_max_points = 0

    for p in range(NPOSE):
        var px = rng.sym(SPAN)
        var py = rng.sym(SPAN)
        var pz = rng.sym(SPAN)
        var qx = rng.sym(1.0)
        var qy = rng.sym(1.0)
        var qz = rng.sym(1.0)
        var qw = rng.sym(1.0)
        var qn = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if qn < 1e-6:
            qx = 0.0
            qy = 0.0
            qz = 0.0
            qw = 1.0
            qn = 1.0
        qx /= qn
        qy /= qn
        qz /= qn
        qw /= qn

        # ⚠ Free-joint qpos is (x, y, z, qw, qx, qy, qz) — MuJoCo's order, NOT
        # the (x, y, z, w) order the contact records use.
        BBM.reset_data(d)
        d.qpos.data[0] = Scalar[DTYPE](px)
        d.qpos.data[1] = Scalar[DTYPE](py)
        d.qpos.data[2] = Scalar[DTYPE](pz)
        d.qpos.data[3] = Scalar[DTYPE](qw)
        d.qpos.data[4] = Scalar[DTYPE](qx)
        d.qpos.data[5] = Scalar[DTYPE](qy)
        d.qpos.data[6] = Scalar[DTYPE](qz)
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])

        dat.qpos[0] = px
        dat.qpos[1] = py
        dat.qpos[2] = pz
        dat.qpos[3] = qw
        dat.qpos[4] = qx
        dat.qpos[5] = qy
        dat.qpos[6] = qz
        for i in range(NV):
            dat.qvel[i] = 0.0
        mujoco.mj_forward(m, dat)
        var mjn = Int(py=dat.ncon)

        if mjn > 0:
            n_touch += 1
            mj_points += mjn
            if mjn > mj_max_points:
                mj_max_points = mjn
        if nc > 0:
            n_ours += 1
        if mjn == 0 or nc == 0:
            continue
        n_both += 1

        # MuJoCo's DEEPEST point on the manifold.
        var best = 0
        var bestd = Float64(py=dat.contact[0].dist)
        for k in range(1, mjn):
            var dk = Float64(py=dat.contact[k].dist)
            if dk < bestd:
                bestd = dk
                best = k
        var con = dat.contact[best]

        var ba = Int(d.contacts.data[CONTACT_IDX_BODY_A])
        var bb2 = Int(d.contacts.data[CONTACT_IDX_BODY_B])
        var od = Float64(d.contacts.data[CONTACT_IDX_DIST])

        var dd = abs(od - bestd)
        if dd > worst_dist:
            worst_dist = dd
            worst_dist_pose = p
        if dd > TOL_DIST:
            n_bad_dist += 1

        var mb1 = Int(py=m.geom_bodyid[con.geom1])
        var mb2 = Int(py=m.geom_bodyid[con.geom2])
        var sgn = Float64(1.0) if (mb1 == bb2 and mb2 == ba) else Float64(-1.0)
        var e = max(
            abs(Float64(d.contacts.data[CONTACT_IDX_NX])
                - sgn * Float64(py=con.frame[0])),
            max(
                abs(Float64(d.contacts.data[CONTACT_IDX_NY])
                    - sgn * Float64(py=con.frame[1])),
                abs(Float64(d.contacts.data[CONTACT_IDX_NZ])
                    - sgn * Float64(py=con.frame[2])),
            ),
        )
        if e > worst_dir:
            worst_dir = e
            worst_dir_pose = p
        if e > TOL_DIR:
            n_bad_dir += 1

    print("  poses with a MuJoCo contact =", n_touch,
          " with ours =", n_ours, " with both =", n_both)
    print("  MuJoCo manifold points: total", mj_points, " max on one pose",
          mj_max_points, " (we emit 1 — task #42)")
    print("  worst |d dist| =", worst_dist, " at pose", worst_dist_pose,
          " (", n_bad_dist, "poses over tol )")
    print("  worst |d normal| =", worst_dir, " at pose", worst_dir_pose,
          " (", n_bad_dir, "poses over tol )")

    assert_true(
        n_both >= NPOSE / 8,
        String("only ") + String(n_both) + " of " + String(NPOSE)
        + " poses produced a contact in both engines — this sweep gates almost"
        " nothing",
    )
    assert_true(
        worst_dir <= TOL_DIR,
        String("box/box SEPARATING AXIS diverges from MuJoCo by ")
        + String(worst_dir) + " at pose " + String(worst_dir_pose)
        + " (" + String(n_bad_dir) + " poses over tolerance) — the manifold"
        " port needs MuJoCo's axis selection too, not just its clipping",
    )
    assert_true(
        worst_dist <= TOL_DIST,
        String("box/box PENETRATION diverges from MuJoCo by ")
        + String(worst_dist) + " at pose " + String(worst_dist_pose)
        + " (" + String(n_bad_dist) + " poses over tolerance)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
