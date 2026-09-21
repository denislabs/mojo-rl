"""`<tendonpos>` on the BATCHED CPU leg, vs MuJoCo per env.

`d.ten_length` is filled on demand — `compute_tendon_lengths` returns at its
first lines unless a SERVED `<tendonpos>` row exists, because a spatial
tendon's length is a polyline walk over its wrap geoms and materialising the
whole array every step would pay for 700 of them on `ms_human_700` to serve a
sensor almost nothing declares.

That pass ran under `comptime if target == "cpu" and BATCH == 1` while the
sensor table marked `<tendonpos>` **SERVED**, so a batched model got a slot
the contract says is computed and a NaN in it. This file is the gate on the
env loop that closes it.

⚠⚠ THE FAILURE THIS FILE EXISTS FOR HAS A NAMED VALUE: under the BATCH=1
pass, envs 1..3 read **NaN** for both tendon rows while env 0 reads the right
number, and `jointpos` — computed in the eval pass itself, never through
`ten_length` — reads correctly in all four. Both halves are asserted
separately, so a failure says which leg moved.

⚠ ONE FIXED AND ONE SPATIAL TENDON. They reach `d.ten_length` through
different helpers (`fixed_tendon_length_jac` over `qpos`,
`spatial_tendon_length_jac` over the world polyline), and only the second one
is the expensive walk the guard exists for. A fixture with one kind would
gate half the pass.

⚠ AIRBORNE. `ncon == 0` is asserted per env, as in every other sensordata
gate.

Run with:
    pixi run mojo run -I . tests/physics3d/test_tendon_length_batched_vs_mujoco.mojo
"""

from std.math import abs, isnan
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64
comptime NENV = 4
comptime NSD = 3

# The elbow drives BOTH tendons: the fixed one reads `0.5 * qpos[el]`
# directly, and the spatial one is the imu->wrist distance, which the elbow
# swings. So a per-env elbow angle moves every value this file compares.
comptime T_XML = """
<mujoco model="batched tendon length">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0.02 0.05" size="0.02"/>
      <body name="link" pos="0.2 0 0">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0.01 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="jp" joint="el"/>
    <tendonpos name="tpf" tendon="tf"/>
    <tendonpos name="tps" tendon="ts"/>
  </sensor>
  <tendon>
    <fixed name="tf">
      <joint joint="el" coef="0.5"/>
    </fixed>
    <spatial name="ts">
      <site site="imu"/>
      <site site="wrist"/>
    </spatial>
  </tendon>
</mujoco>
"""

comptime tp = parse_xml(T_XML)
comptime TM = ModelDefFromXML[
    xml=T_XML,
    nbody=tp.NBODY, njoint=tp.NJOINT, nq=tp.NQ, nv=tp.NV,
    ngeom=tp.NGEOM, nact=tp.NACT, ntex=tp.NTEX, nmat=tp.NMAT,
    nlight=tp.NLIGHT, ncam=tp.NCAM, nsite=tp.NSITE,
    # 3 sensors, 1 value each.
    nsensor=3, nsensordata=NSD,
    max_tendon=tp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=tp.TIMESTEP,
]

comptime TMD = ModelDims[TM]
comptime Dat = Data[DTYPE, TMD, NENV]
comptime Mod = Model[DTYPE, TMD]
comptime Integ = EulerIntegrator[
    DTYPE, TMD, TM.CONE_TYPE, NENV, SOLVER="newton", RNE_POST=True
]


def _state(e: Int) -> Tuple[List[Float64], List[Float64]]:
    """Env `e`'s pose and rates — DIFFERENT per env, by construction.

    ⚠ THE ELBOW SPREAD IS THE TEST. Four envs at the same elbow angle give
    four identical tendon lengths, and then "the pass indexed the env" and
    "the pass wrote env 0 four times" are the same output.
    """
    var f = Float64(e)
    var qpos = List[Float64]()
    qpos.append(0.05 * f)
    qpos.append(-0.03 * f)
    qpos.append(2.0 + 0.1 * f)
    var wx = 0.9 - 0.1 * f
    var xx = 0.2 + 0.05 * f
    var yy = 0.15 + 0.04 * f
    var zz = 0.1 + 0.06 * f
    var n = (wx * wx + xx * xx + yy * yy + zz * zz) ** 0.5
    qpos.append(wx / n)
    qpos.append(xx / n)
    qpos.append(yy / n)
    qpos.append(zz / n)
    qpos.append(0.2 + 0.35 * f)  # elbow

    var qvel = List[Float64]()
    qvel.append(0.7 - 0.2 * f)
    qvel.append(-0.4 + 0.3 * f)
    qvel.append(1.1 - 0.15 * f)
    qvel.append(0.9 + 0.2 * f)
    qvel.append(-1.3 + 0.4 * f)
    qvel.append(0.5 - 0.25 * f)
    qvel.append(2.0 - 0.5 * f)
    return (qpos^, qvel^)


def _run(mut d: Dat, mut mf: Mod, ctx: DeviceContext) raises:
    """⚠ NO `reset_data` — IT IS BATCH=1 ONLY, as the batched sensordata gate
    records. Every `qpos`/`qvel` is written here and `Data.__init__` zeroes
    the rest, so the pose is fully determined without it."""
    TM.init_fields[DTYPE](ctx, mf)
    for i in range(NENV * TM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for e in range(NENV):
        var st = _state(e)
        for i in range(TM.NQ):
            d.qpos.data[e * TM.NQ + i] = Scalar[DTYPE](st[0][i])
        for i in range(TM.NV):
            d.qvel.data[e * TM.NV + i] = Scalar[DTYPE](st[1][i])
    var integ = Integ()
    integ.step["cpu"](d, mf)


def _mj_at(mujoco: PythonObject, e: Int) raises -> PythonObject:
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(T_XML)))
    var dat = mujoco.MjData(m)
    var st = _state(e)
    for i in range(len(st[0])):
        dat.qpos[i] = st[0][i]
    for i in range(len(st[1])):
        dat.qvel[i] = st[1][i]
    mujoco.mj_forward(m, dat)
    return dat^


def test_the_four_envs_have_different_tendon_lengths() raises:
    """⚠ RUN FIRST. Without it the whole file passes on a pass that wrote env
    0's lengths into all four rows.

    MuJoCo's own `ten_length` must separate the envs before anything here can
    claim to be testing the env index.
    """
    print("=== the four envs are at four different tendon lengths ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var d0 = _mj_at(mujoco, 0)
    for e in range(1, NENV):
        var de = _mj_at(mujoco, e)
        for t in range(2):
            var a = Float64(py=d0.ten_length[t])
            var b = Float64(py=de.ten_length[t])
            print(
                "  tendon", t, " env 0 =", a, " env", e, "=", b,
                " |d| =", abs(a - b),
            )
            assert_true(
                abs(a - b) > 1e-4,
                "tendon " + String(t) + " has the same length in env 0 and"
                " env " + String(e) + " — this fixture cannot detect a"
                " dropped env index",
            )


def test_batched_tendonpos_matches_mujoco_per_env() raises:
    """Both tendon rows, all four envs, against MuJoCo forwarded at each state.

    ⚠ THE NaN ASSERT IS THE NEGATIVE CONTROL AND IT NAMES ITS VALUE. Under
    the BATCH=1 pass `d.ten_length` was written for env 0 only, so envs 1..3
    reported exactly `nan` here — not a wrong number, the `Data` fill.
    """
    print("=== batched <tendonpos> vs MuJoCo, per env ===")
    var mujoco = Python.import_module("mujoco")

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var worst = 0.0
    var compared = 0
    var nan_rows = 0
    for e in range(NENV):
        var dat = _mj_at(mujoco, e)
        assert_true(
            Int(py=dat.ncon) == 0,
            "env " + String(e) + " is in contact (" + String(Int(py=dat.ncon))
            + " contacts); this gate must stay airborne",
        )
        for k in range(NSD):
            var ours = Float64(d.sensordata.data[e * NSD + k])
            var theirs = Float64(py=dat.sensordata[k])
            if isnan(ours):
                nan_rows += 1
            assert_true(
                not isnan(ours),
                "env " + String(e) + " sensor value " + String(k) + " is NaN."
                " That is `Data`'s fill, i.e. the BATCH=1 pass: env 0 written,"
                " every other env left uncomputed while the table says SERVED",
            )
            var dd = abs(ours - theirs)
            if dd > worst:
                worst = dd
            assert_true(
                dd <= 1e-12,
                "env " + String(e) + " sensor value " + String(k) + ": ours "
                + String(ours) + " vs MuJoCo " + String(theirs),
            )
            compared += 1
        print(
            "  env", e,
            " jointpos =", Float64(d.sensordata.data[e * NSD + 0]),
            " fixed =", Float64(d.sensordata.data[e * NSD + 1]),
            " spatial =", Float64(d.sensordata.data[e * NSD + 2]),
        )
    print("  values compared:", compared, " values NaN:", nan_rows,
          " worst |d| =", worst)
    assert_true(compared == NENV * NSD,
                "expected " + String(NENV * NSD) + " comparisons")


def test_only_the_tendon_rows_were_ever_at_risk() raises:
    """The contrast arm: `jointpos` was batched-correct before this change.

    `jointpos` is computed inside the eval pass from `qpos[jnt_qposadr]` and
    never reaches `d.ten_length`, so it read correctly in all four envs even
    while the tendon rows were NaN. Asserting that here is what makes the
    NaN test above evidence about `compute_tendon_lengths` rather than about
    the sensor pass as a whole.
    """
    print("=== jointpos was correct in every env all along ===")
    var mujoco = Python.import_module("mujoco")

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    for e in range(NENV):
        var dat = _mj_at(mujoco, e)
        var ours = Float64(d.sensordata.data[e * NSD + 0])
        var theirs = Float64(py=dat.sensordata[0])
        assert_true(
            not isnan(ours) and abs(ours - theirs) <= 1e-12,
            "env " + String(e) + " jointpos: ours " + String(ours)
            + " vs MuJoCo " + String(theirs),
        )
    print("  4 envs, jointpos matches MuJoCo in each")


def test_no_env_shares_a_tendon_length_with_another() raises:
    """OUR rows must differ from each other too, not only MuJoCo's.

    The per-env comparison above already catches a dropped env index — env 1
    would be compared against MuJoCo's env 1 and fail. This states the
    property directly, so a failure reads as "the pass is not per-env" and
    survives someone loosening the tolerance.
    """
    print("=== our four tendon readings are four different readings ===")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    _run(d, mf, ctx)

    var pairs = 0
    for a in range(NENV):
        for b in range(a + 1, NENV):
            for k in range(1, NSD):
                var va = Float64(d.sensordata.data[a * NSD + k])
                var vb = Float64(d.sensordata.data[b * NSD + k])
                # ⚠ BOTH VALUES IN THE MESSAGE, because NaN fails this
                # comparison too and the two failures are different bugs.
                # Under the BATCH=1 pass `vb` reads `nan` — printing only
                # `va` made that failure read as "two envs agree on 0.1".
                assert_true(
                    abs(va - vb) > 1e-4,
                    "our env " + String(a) + " reads " + String(va)
                    + " and env " + String(b) + " reads " + String(vb)
                    + " for tendon row " + String(k) + " — either the pass is"
                    " writing one env's reading into several rows, or the"
                    " later envs were never computed at all",
                )
            pairs += 1
    print("  every one of the", pairs, "env pairs differs on both tendons")


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_four_envs_have_different_tendon_lengths]()
    suite.test[test_batched_tendonpos_matches_mujoco_per_env]()
    suite.test[test_only_the_tendon_rows_were_ever_at_risk]()
    suite.test[test_no_env_shares_a_tendon_length_with_another]()
    suite^.run()
