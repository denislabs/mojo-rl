"""`d.sensordata` on the DEVICE, against the CPU leg (AUD-53).

The sensor pass is one function — `_eval_sensor_env`, twenty-nine tensors and
an `env` — called from a CPU loop or from `_sensor_stage_kernel`, one thread
per env. This is the gate that says the kernel exists at all: a green BUILD
proves nothing about a generic GPU kernel, because precompilation stops at
elaboration and the kernel is only instantiated when something launches it.

⚠⚠ THE TWO LEGS ARE COMPARED AGAINST EACH OTHER, NOT AGAINST MuJoCo, AND THAT
IS DELIBERATE. `test_batched_sensordata_vs_mujoco` already pins the CPU leg to
the oracle at 1.7e-15; what is open here is whether the DEVICE computes the
same thing from the same state. Comparing the GPU against MuJoCo directly
would fold the two questions together, and a float32 device leg would then
fail on precision rather than on correctness.

⚠ FLOAT32. Metal has no `double`; every GPU env in this tree is float32 and
so is this one. The bound below is set from the measurement and is a float32
bound, not a claim about the algorithm.

⚠ THE STATE IS SET AND THEN STEPPED ONCE ON BOTH LEGS, from the same `Data`
contents. A rollout would accumulate solver divergence and the sensor
comparison would be measuring that instead.

Run with:
    pixi run mojo run -I . tests/physics3d/test_sensordata_gpu_vs_cpu.mojo
"""

from std.math import abs, isnan, sqrt
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float32
comptime NENV = 4

# Every served kind the device leg can reach. ⚠ A RANGEFINDER IS IN HERE ON
# PURPOSE: it is the one sensor that pulls the whole ray/mesh/hfield set into
# the kernel's argument table — four of the twenty-nine buffers exist only for
# it — so leaving it out would test the easy half.
comptime G_XML = """
<mujoco model="gpu sensordata">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0.02 0.05" size="0.02" euler="0 25 40"/>
      <site name="down" pos="0 0 -0.08" size="0.01"/>
      <site name="pad" pos="0 0 -0.09" type="box" size="0.1 0.1 0.02"/>
      <body name="link" pos="0.2 0 0" euler="0 0 15">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0.01 0" size="0.02" euler="10 0 0"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf" site="down"/>
    <jointpos name="jp" joint="el"/>
    <framepos name="fp" objtype="site" objname="wrist"/>
    <framequat name="fq" objtype="body" objname="link"/>
    <subtreecom name="scm" body="torso"/>
    <velocimeter name="vel" site="imu"/>
    <gyro name="gyr" site="imu"/>
    <jointvel name="jv" joint="el"/>
    <framelinvel name="flv" objtype="site" objname="wrist"/>
    <frameangvel name="fav" objtype="geom" objname="gl"/>
    <subtreelinvel name="slv" body="torso"/>
    <touch name="tch" site="pad"/>
  </sensor>
</mujoco>
"""

comptime gp = parse_xml(G_XML)
comptime GM = ModelDefFromXML[
    xml=G_XML,
    nbody=gp.NBODY, njoint=gp.NJOINT, nq=gp.NQ, nv=gp.NV,
    ngeom=gp.NGEOM, nact=gp.NACT, ntex=gp.NTEX, nmat=gp.NMAT,
    nlight=gp.NLIGHT, ncam=gp.NCAM, nsite=gp.NSITE,
    # 12 sensors: rf 1, jp 1, fp 3, fq 4, scm 3, vel 3, gyr 3, jv 1, flv 3,
    # fav 3, slv 3, tch 1 = 29. ⚠ EXACT, not padded — `init_fields`
    # cross-checks the sum against the parse and raises on a mismatch.
    nsensor=12, nsensordata=29,
    max_tendon=gp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=gp.TIMESTEP,
]

comptime GMD = ModelDims[GM]
comptime Integ = EulerIntegrator[
    DTYPE, GMD, GM.CONE_TYPE, NENV, SOLVER="newton", RNE_POST=False
]
comptime NSDATA = 29


def _set_state(mut d: Data[DTYPE, GMD, NENV]):
    """The same four distinct states both legs are evaluated at."""
    for i in range(NENV * GM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
        d.qacc.data[i] = Scalar[DTYPE](0)
        d.qacc_warmstart.data[i] = Scalar[DTYPE](0)
    for e in range(NENV):
        var f = Float32(e)
        var q = e * GM.NQ
        d.qpos.data[q + 0] = Scalar[DTYPE](0.05 * f)
        d.qpos.data[q + 1] = Scalar[DTYPE](-0.03 * f)
        d.qpos.data[q + 2] = Scalar[DTYPE](2.0 + 0.1 * f)
        var wx = 0.9 - 0.1 * f
        var xx = 0.2 + 0.05 * f
        var yy = 0.15 + 0.04 * f
        var zz = 0.1 + 0.06 * f
        var n = sqrt(wx * wx + xx * xx + yy * yy + zz * zz)
        d.qpos.data[q + 3] = Scalar[DTYPE](wx / n)
        d.qpos.data[q + 4] = Scalar[DTYPE](xx / n)
        d.qpos.data[q + 5] = Scalar[DTYPE](yy / n)
        d.qpos.data[q + 6] = Scalar[DTYPE](zz / n)
        d.qpos.data[q + 7] = Scalar[DTYPE](0.2 + 0.35 * f)
        var v = e * GM.NV
        d.qvel.data[v + 0] = Scalar[DTYPE](0.7 - 0.2 * f)
        d.qvel.data[v + 1] = Scalar[DTYPE](-0.4 + 0.3 * f)
        d.qvel.data[v + 2] = Scalar[DTYPE](1.1 - 0.15 * f)
        d.qvel.data[v + 3] = Scalar[DTYPE](0.9 + 0.2 * f)
        d.qvel.data[v + 4] = Scalar[DTYPE](-1.3 + 0.4 * f)
        d.qvel.data[v + 5] = Scalar[DTYPE](0.5 - 0.25 * f)
        d.qvel.data[v + 6] = Scalar[DTYPE](2.0 - 0.5 * f)


def test_the_device_fills_sensordata_at_all() raises:
    """⚠ RUN FIRST, AND IT IS NOT A FORMALITY. `Data` NaN-fills `sensordata`,
    so a kernel that never launched — or launched and wrote nothing — leaves
    every slot NaN. Without this arm a comparison loop that skipped NaNs, or a
    tolerance applied to NaN, could read as agreement.
    """
    print("=== the device leg writes sensordata ===")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, GMD]()
    GM.init_fields[DTYPE](ctx, mf)
    var dg = Data[DTYPE, GMD, NENV]()
    _set_state(dg)
    dg.upload_all(ctx)

    var ig = Integ()
    ig.prepare_gpu(ctx)
    ig.step["gpu"](dg, mf, ctx)
    dg.sensordata.download(ctx)

    var n_nan = 0
    for i in range(NENV * NSDATA):
        if isnan(Float64(dg.sensordata.data[i])):
            n_nan += 1
    print("  slots still NaN after one GPU step:", n_nan, "/",
          NENV * NSDATA)
    assert_true(
        n_nan == 0,
        String(n_nan) + " of " + String(NENV * NSDATA) + " slots are still"
        " NaN — the device pass did not run, or did not cover every served"
        " sensor",
    )


def test_device_sensordata_matches_the_cpu_leg() raises:
    """Same model, same state, one step each; compare all four envs."""
    print("=== GPU vs CPU sensordata, per env ===")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, GMD]()
    GM.init_fields[DTYPE](ctx, mf)

    var dg = Data[DTYPE, GMD, NENV]()
    var dc = Data[DTYPE, GMD, NENV]()
    _set_state(dg)
    _set_state(dc)
    dg.upload_all(ctx)

    var ig = Integ()
    ig.prepare_gpu(ctx)
    var ic = Integ()
    ig.step["gpu"](dg, mf, ctx)
    ic.step["cpu"](dc, mf)
    dg.sensordata.download(ctx)

    var worst = 0.0
    var nonzero = 0
    var compared = 0
    for e in range(NENV):
        var e_worst = 0.0
        for k in range(NSDATA):
            var a = Float64(dc.sensordata.data[e * NSDATA + k])
            var b = Float64(dg.sensordata.data[e * NSDATA + k])
            assert_true(
                not isnan(a) and not isnan(b),
                "env " + String(e) + " value " + String(k) + " is NaN (cpu "
                + String(a) + ", gpu " + String(b) + ")",
            )
            var dd = abs(a - b)
            if dd > e_worst:
                e_worst = dd
            if abs(a) > 1e-6:
                nonzero += 1
            compared += 1
        if e_worst > worst:
            worst = e_worst
        print("  env", e, " worst |cpu - gpu| =", e_worst)
    print("  values compared:", compared, " worst |d| =", worst)
    print("  values the CPU leg reports NONZERO:", nonzero, "/", compared)
    # ⚠ NON-VACUITY. Two all-zero buffers agree perfectly.
    assert_true(
        nonzero >= 90,
        "only " + String(nonzero) + " of " + String(compared) + " values are"
        " nonzero — the fixture has gone quiet and the agreement is empty",
    )
    # ⚠ BOUND FROM THE MEASUREMENT, NOT INHERITED. Measured 1.19e-07 on
    # Metal across all four envs — one float32 ulp at these magnitudes. It is
    # a FLOAT32 bound and says nothing about the algorithm; the CPU leg's
    # agreement with MuJoCo (1.7e-15, `test_batched_sensordata_vs_mujoco`) is
    # what says that.
    assert_true(
        worst <= 1e-6,
        "GPU and CPU sensordata differ by " + String(worst),
    )


def test_the_device_rows_are_four_different_readings() raises:
    """One thread per env — so a dropped `env` index writes one reading into
    every row, and every row would still look plausible. Stated directly so a
    failure reads as "the kernel is not per-env".
    """
    print("=== the device's four rows are four readings ===")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, GMD]()
    GM.init_fields[DTYPE](ctx, mf)
    var dg = Data[DTYPE, GMD, NENV]()
    _set_state(dg)
    dg.upload_all(ctx)
    var ig = Integ()
    ig.prepare_gpu(ctx)
    ig.step["gpu"](dg, mf, ctx)
    dg.sensordata.download(ctx)

    var pairs = 0
    for a in range(NENV):
        for b in range(a + 1, NENV):
            var moved = 0
            for k in range(NSDATA):
                if abs(
                    Float64(dg.sensordata.data[a * NSDATA + k])
                    - Float64(dg.sensordata.data[b * NSDATA + k])
                ) > 1e-5:
                    moved += 1
            assert_true(
                moved >= 20,
                "device env " + String(a) + " and env " + String(b)
                + " agree on all but " + String(NSDATA - moved) + " values —"
                " the kernel is writing one env's readings into several rows",
            )
            pairs += 1
    print("  every one of the", pairs, "env pairs differs in >= 20 of",
          NSDATA)


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_device_fills_sensordata_at_all]()
    suite.test[test_device_sensordata_matches_the_cpu_leg]()
    suite.test[test_the_device_rows_are_four_different_readings]()
    suite^.run()
