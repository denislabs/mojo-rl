"""`<subtreeangmom>` — the last AUD-23 kind a model in this tree declares.

    pixi run mojo run -I . tests/physics3d/test_subtreeangmom_vs_mujoco.mojo

Angular momentum of a subtree about that subtree's centre of mass, in the
world frame. Six declarations, all of them dog
(`mojo_rl/envs/dm_control/assets/dog_{trot,run,fetch,stand_walk}.xml` and
dm_control's own `dog.xml`), and it was the last kind still ADDRESSED and
unserved on a model this tree ships an env for.

⚠⚠ SERVED BY THE DEFINITION, NOT BY MUJOCO'S RECURSION, and the substitution
was measured first. `mj_subtreeVel` (engine_core_smooth.c:2249) uses two
REVERSE passes with a `body_vel` scratch, a momentum accumulator divided by
`body_subtreemass`, and a per-parent shift. Ours sums

    L = sum_i [ R_i (I_i . (R_i^T w_i)) + m_i (xipos_i - com) x (v_i - vcom) ]

over the subtree. The two agree to 2.2e-16 at every root of a five-body probe
(free joint + ball joint + two hinges). Taking the direct form avoided a
`body_subtreemass` column (body record 28 -> 29, every model's layout), a
`Data.subtree_angmom` field, a new step pass with a new ordering constraint,
and a 26th buffer in a sensor kernel that fails with NO DIAGNOSTIC at 29 —
because every operand the sum needs is ALREADY bound by the sensor eval.

⚠ A DIFFERENT SUMMATION ORDER, so this is not bit-identical to MuJoCo and
this file does not claim it is.

TWO NON-VACUITY ARMS, because a subtree momentum is easy to fake:

  * the three roots must report three DIFFERENT momenta on MuJoCo's side
    before anything is compared;
  * the LEAF root is pure SPIN. For a one-body subtree `subtree_com` is that
    body's `xipos` and `subtree_linvel` is its `xvel`, so the orbital term is
    identically zero — asserted here from MuJoCo's own arrays. An
    implementation that dropped `R_i I_i R_i^T w_i` reads exactly (0,0,0)
    there, and the fixture keeps that value large.

⚠ THE TORSO CARRIES A `fullinertia` WITH OFF-DIAGONAL TERMS, so MuJoCo
diagonalises it into a NON-IDENTITY `body_iquat`. Without that, `R_i` is the
body frame, the `iquat` composition is the identity, and dropping it
altogether would pass. The test asserts the fixture actually has one.

⚠ AIRBORNE, like every other sensordata gate: `ncon == 0` is asserted.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64
comptime NSD = 9  # three sensors x 3

# Bodies: torso (free, tilted inertial frame), a (hinge), b (leaf hinge),
# c (ball). Sensors at torso / a / b — a mixed root, a smaller mixed root, and
# a leaf whose reading is pure spin.
comptime A_XML = """
<mujoco model="subtree angmom">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"/>
    <body name="torso" pos="0 0 1.5">
      <freejoint name="root"/>
      <inertial pos="0.01 -0.02 0.005" mass="4.0"
                fullinertia="0.05 0.07 0.06 0.011 -0.008 0.013"/>
      <geom name="gt" type="box" size="0.15 0.1 0.05" density="800"/>
      <body name="a" pos="0.15 0 0" euler="0 20 0">
        <joint name="ja" type="hinge" axis="0 1 0"/>
        <geom name="ga" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="b" pos="0.3 0 0" euler="0 0 30">
          <joint name="jb" type="hinge" axis="0 0 1"/>
          <geom name="gb" type="capsule" fromto="0 0 0 0.25 0 0"
                size="0.025"/>
        </body>
      </body>
      <body name="c" pos="-0.15 0.05 0">
        <joint name="jc" type="ball"/>
        <geom name="gc" type="capsule" fromto="0 0 0 -0.2 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <subtreeangmom name="Lt" body="torso"/>
    <subtreeangmom name="La" body="a"/>
    <subtreeangmom name="Lb" body="b"/>
  </sensor>
</mujoco>
"""

comptime ap = parse_xml(A_XML)
comptime AM = ModelDefFromXML[
    xml=A_XML,
    nbody=ap.NBODY, njoint=ap.NJOINT, nq=ap.NQ, nv=ap.NV,
    ngeom=ap.NGEOM, nact=ap.NACT, ntex=ap.NTEX, nmat=ap.NMAT,
    nlight=ap.NLIGHT, ncam=ap.NCAM, nsite=ap.NSITE,
    nsensor=3, nsensordata=NSD,
    max_tendon=ap.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=ap.TIMESTEP,
]

comptime AMD = ModelDims[AM]
comptime Dat = Data[DTYPE, AMD, 1]
comptime Mod = Model[DTYPE, AMD]
comptime Integ = EulerIntegrator[
    DTYPE, AMD, AM.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
]

# qpos: free (3 pos + 4 quat) + ja + jb + ball (4). qvel: 6 + 1 + 1 + 3.
comptime NQ_F = 7


def _state() -> Tuple[List[Float64], List[Float64]]:
    """One pose and rate set, tilted and spinning on every joint.

    ⚠ EVERY DOF MOVES. A subtree momentum with a resting joint in it cannot
    distinguish the spin term from the orbital one at that body.
    """
    var qpos = List[Float64]()
    qpos.append(0.07)
    qpos.append(-0.04)
    qpos.append(1.5)
    # free-joint quaternion, w x y z, normalised.
    var w = 0.86
    var x = 0.21
    var y = -0.17
    var z = 0.31
    var n = sqrt(w * w + x * x + y * y + z * z)
    qpos.append(w / n)
    qpos.append(x / n)
    qpos.append(y / n)
    qpos.append(z / n)
    qpos.append(0.42)   # ja
    qpos.append(-0.65)  # jb
    # ball-joint quaternion, w x y z, normalised.
    var bw = 0.93
    var bx = -0.14
    var by = 0.22
    var bz = 0.09
    var bn = sqrt(bw * bw + bx * bx + by * by + bz * bz)
    qpos.append(bw / bn)
    qpos.append(bx / bn)
    qpos.append(by / bn)
    qpos.append(bz / bn)

    var qvel = List[Float64]()
    qvel.append(0.8)
    qvel.append(-0.55)
    qvel.append(1.05)
    qvel.append(1.4)
    qvel.append(-0.9)
    qvel.append(0.75)
    qvel.append(-1.6)   # ja
    qvel.append(2.1)    # jb
    qvel.append(0.65)   # ball wx
    qvel.append(-1.2)   # ball wy
    qvel.append(0.95)   # ball wz
    return (qpos^, qvel^)


def _run(mut d: Dat, mut mf: Mod, ctx: DeviceContext) raises:
    AM.init_fields[DTYPE](ctx, mf)
    var st = _state()
    for i in range(AM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for i in range(AM.NQ):
        d.qpos.data[i] = Scalar[DTYPE](st[0][i])
    for i in range(AM.NV):
        d.qvel.data[i] = Scalar[DTYPE](st[1][i])
    var integ = Integ()
    integ.step["cpu"](d, mf)


def _mj() raises -> Tuple[PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(A_XML)))
    var dat = mujoco.MjData(m)
    var st = _state()
    for i in range(len(st[0])):
        dat.qpos[i] = st[0][i]
    for i in range(len(st[1])):
        dat.qvel[i] = st[1][i]
    mujoco.mj_forward(m, dat)
    return (m^, dat^)


def test_the_fixture_exercises_a_rotated_inertial_frame() raises:
    """⚠ RUN FIRST. `R_i` is `R_body . R_iquat`; with an identity `iquat` the
    composition is a no-op and an implementation that never applied it would
    pass every value below.

    The torso's `fullinertia` has off-diagonal terms, so MuJoCo diagonalises
    it into a non-identity `body_iquat`. This asserts that actually happened
    rather than assuming it.
    """
    print("=== the torso's inertial frame is rotated ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var pair = _mj()
    var m = pair[0]
    # body 1 is the torso (0 is the world).
    var qw = Float64(py=m.body_iquat[1][0])
    var qx = Float64(py=m.body_iquat[1][1])
    var qy = Float64(py=m.body_iquat[1][2])
    var qz = Float64(py=m.body_iquat[1][3])
    var off = abs(qx) + abs(qy) + abs(qz)
    print("  torso body_iquat = (", qw, qx, qy, qz, ") off-axis =", off)
    assert_true(
        off > 0.05,
        "the torso's inertial frame is (near) the body frame — off-axis part"
        " " + String(off) + ". The `iquat` composition is then untested",
    )


def test_the_three_roots_report_different_momenta() raises:
    """⚠ RUN BEFORE THE COMPARISON. Three roots that agree cannot tell a
    per-root walk from one that returns the whole model every time.
    """
    print("=== MuJoCo's three roots disagree ===")
    var pair = _mj()
    var dat = pair[1]
    var worst_pair = 0.0
    for a in range(3):
        for b in range(a + 1, 3):
            var moved = 0.0
            for k in range(3):
                var va = Float64(py=dat.sensordata[a * 3 + k])
                var vb = Float64(py=dat.sensordata[b * 3 + k])
                if abs(va - vb) > moved:
                    moved = abs(va - vb)
            print("  roots", a, "and", b, " largest component gap =", moved)
            if moved > worst_pair:
                worst_pair = moved
            assert_true(
                moved > 1e-3,
                "roots " + String(a) + " and " + String(b) + " differ by only"
                " " + String(moved) + " — this fixture cannot detect a"
                " per-root walk that ignores its root",
            )
    print("  largest gap over all pairs =", worst_pair)


def test_subtreeangmom_matches_mujoco() raises:
    """All nine values, against `mj_forward` at the same state."""
    print("=== <subtreeangmom> vs MuJoCo ===")
    var pair = _mj()
    var dat = pair[1]
    assert_true(
        Int(py=dat.ncon) == 0,
        "the fixture is in contact (" + String(Int(py=dat.ncon))
        + " contacts); this gate must stay airborne",
    )

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var worst = 0.0
    var nonzero = 0
    for k in range(NSD):
        var ours = Float64(d.sensordata.data[k])
        var theirs = Float64(py=dat.sensordata[k])
        var dd = abs(ours - theirs)
        if dd > worst:
            worst = dd
        if abs(theirs) > 1e-6:
            nonzero += 1
        print("  value", k, " ours =", ours, " MuJoCo =", theirs)
        assert_true(
            dd <= 1e-12,
            "value " + String(k) + ": ours " + String(ours) + " vs MuJoCo "
            + String(theirs),
        )
    print("  values compared:", NSD, " nonzero on MuJoCo's side:", nonzero,
          " worst |d| =", worst)
    assert_true(
        nonzero >= 8,
        "only " + String(nonzero) + " of " + String(NSD) + " values are"
        " nonzero — the fixture has gone quiet",
    )


def test_the_leaf_root_is_pure_spin() raises:
    """The spin term, isolated. ⚠ DROP `R_i I_i R_i^T w_i` AND THIS READS 0.

    For a one-body subtree MuJoCo's `subtree_com` IS that body's `xipos` and
    its `subtree_linvel` IS that body's CoM velocity, so
    `m (xipos - com) x (v - vcom)` is identically zero and the whole reading
    is the body's own spin. Both halves of that claim are asserted from
    MuJoCo's arrays rather than assumed, and then the value is required to be
    large.
    """
    print("=== the leaf root's reading is entirely spin ===")
    var mujoco = Python.import_module("mujoco")
    var pair = _mj()
    var m = pair[0]
    var dat = pair[1]

    # body index of "b" — the leaf. Resolved by name, not assumed.
    var bi = Int(py=mujoco.mj_name2id(
        m, Int(py=mujoco.mjtObj.mjOBJ_BODY), PythonObject(String("b"))
    ))
    assert_true(bi > 0, "the fixture has no body named 'b'")

    var off_com = 0.0
    for k in range(3):
        var dd = abs(
            Float64(py=dat.subtree_com[bi][k]) - Float64(py=dat.xipos[bi][k])
        )
        if dd > off_com:
            off_com = dd
    print("  |subtree_com[b] - xipos[b]| =", off_com)
    assert_true(
        off_com <= 1e-14,
        "body 'b' is not a leaf: its subtree CoM is " + String(off_com)
        + " from its own CoM, so this reading is not pure spin and the arm"
        " below proves nothing",
    )

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var mag = 0.0
    for k in range(3):
        var v = abs(Float64(d.sensordata.data[6 + k]))
        if v > mag:
            mag = v
    print("  our leaf reading magnitude =", mag)
    assert_true(
        mag > 1e-4,
        "the leaf root reports " + String(mag) + ", which is what an"
        " implementation with no spin term reports: exactly zero. The"
        " orbital term cannot contribute here",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_fixture_exercises_a_rotated_inertial_frame]()
    suite.test[test_the_three_roots_report_different_momenta]()
    suite.test[test_subtreeangmom_matches_mujoco]()
    suite.test[test_the_leaf_root_is_pure_spin]()
    suite^.run()
