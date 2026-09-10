"""The live 463-D privileged observation against the reference's function — G3.0's gate.

    pixi run mojo run -I . tests/robots/test_unitree_g1_privileged_obs.mojo
    G1_LAFAN_STORE=/path/to/store.h5 pixi run mojo run -I . tests/robots/test_unitree_g1_privileged_obs.mojo

`unitree_g1_config.mojo`'s CPU observation hook appends BFM-Zero's
`max_local_self` (`unitree_g1_priv_obs.mojo`) to the 64-D proprio state.
This gate feeds OUR engine's raw per-body state — the 30 skeleton bodies'
`xpos`, `xquat`, `xipos`, `xvel` (COM point), `xangvel` — through
`tools/g1/privileged_obs_oracle.py::max_local_self_from_sim`, a numpy
transcription of the reference's `compute_humanoid_observations_max` plus
its head extension and the COM-to-origin velocity conversion, certified
against the reference's torch function by `--selfcheck` (9.7e-7, float32
torch). The two must agree to `TOL` (float64 both) on every sampled row.

Two situations per row, because they exercise different code paths:
  1. `set_state` from a store row (FK + body velocities refreshed by
     `set_state` itself) — the observation of an injected reference state,
     i.e. what a reset-from-store lane sees (G3.1);
  2. `N_STEPS` driven control steps after it — the observation the policy
     sees during a rollout, which is only right if `SYNC_FK_AFTER_STEP`
     refreshes the FK products and body velocities after the last substep.
     Without the flag this half fails by the size of one substep's motion.

The store's `qvel[3:6]` is the reference's WORLD-frame root angular
velocity; it is injected as-is here — any qvel is a valid state for a
formula gate — and the observation compares the engine's own body
velocities, not the reference's finite differences.

⚠ RUN FROM THE REPO ROOT, like every gate that loads `unitree_g1.xml`.
"""

from std.math import abs, sin, cos, atan2
from std.os import getenv
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_STATE_DIM, UNITREE_G1_OBS_DIM,
)
from mojo_rl.envs.robots.unitree_g1_priv_obs import (
    G1_N_SKELETON, G1_PRIV_DIM, g1_skeleton_body, g1_atan2f,
)


comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime ACT = UnitreeG1Model.ACTION_DIM
comptime TOL = 1e-10
comptime STRIDE = 4999      # rows between samples: ~88 rows over the 40 clips
comptime N_STEPS = 3        # driven steps after each injection


def _store_path() -> String:
    var p = getenv("G1_LAFAN_STORE")
    if p.byte_length() > 0:
        return p
    return String("lafan_g1_50hz.h5")


def _py_list(builtins: PythonObject, xs: List[Float64]) raises -> PythonObject:
    var out = builtins.list()
    for i in range(len(xs)):
        _ = out.append(xs[i])
    return out


def _raw_bodies(
    env: UnitreeG1[DType.float64], builtins: PythonObject
) raises -> Tuple[PythonObject, PythonObject, PythonObject, PythonObject, PythonObject]:
    """The 30 skeleton bodies' raw state from the engine, in the reference's order."""
    var xpos = List[Float64](capacity=G1_N_SKELETON * 3)
    var xquat = List[Float64](capacity=G1_N_SKELETON * 4)
    var xipos = List[Float64](capacity=G1_N_SKELETON * 3)
    var xvel = List[Float64](capacity=G1_N_SKELETON * 3)
    var xang = List[Float64](capacity=G1_N_SKELETON * 3)
    for s in range(G1_N_SKELETON):
        var b = g1_skeleton_body(s)
        for c in range(3):
            xpos.append(Float64(env.d.xpos.data[b * 3 + c]))
            xipos.append(Float64(env.d.xipos.data[b * 3 + c]))
            xvel.append(Float64(env.d.xvel.data[b * 3 + c]))
            xang.append(Float64(env.d.xangvel.data[b * 3 + c]))
        for c in range(4):
            xquat.append(Float64(env.d.xquat.data[b * 4 + c]))
    return (
        _py_list(builtins, xpos), _py_list(builtins, xquat), _py_list(builtins, xipos),
        _py_list(builtins, xvel), _py_list(builtins, xang),
    )


def _compare(
    obs: List[Scalar[DType.float64]], want: PythonObject, mut worst: Float64
) raises:
    for i in range(G1_PRIV_DIM):
        var e = abs(Float64(obs[UNITREE_G1_STATE_DIM + i]) - Float64(py=want[i]))
        if e > worst:
            worst = e


def test_privileged_obs_matches_reference_function() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var builtins = Python.import_module("builtins")
    var oracle = Python.import_module("privileged_obs_oracle")

    var store = TrajectoryStore(_store_path())
    var n = store.n_rows()
    var qpos = store.load_column[DType.float32](String("qpos"))
    var qvel = store.load_column[DType.float32](String("qvel"))

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    assert_true(len(env.get_obs_list()) == UNITREE_G1_OBS_DIM, "obs dim is not 64 + 463")

    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var worst_set = 0.0
    var worst_step = 0.0
    var n_rows = 0
    var n_steps = 0
    var r = 0
    while r < n:
        for i in range(NQ):
            qp[i] = Float64(qpos[r * NQ + i])
        for i in range(NV):
            qv[i] = Float64(qvel[r * NV + i])
        env.set_state(qp, qv)
        var obs0 = env.get_obs_list()
        var raw = _raw_bodies(env, builtins)
        var want0 = oracle.max_local_self_from_sim(raw[0], raw[1], raw[2], raw[3], raw[4])
        _compare(obs0, want0, worst_set)
        n_rows += 1
        for t in range(N_STEPS):
            var act = ContAction[ACT]()
            for j in range(ACT):
                act.data[j] = 0.3 * sin(Float64(t + r) * 0.23 + Float64(j) * 0.61)
            var res = env.step(act)
            var obs = List[Scalar[DType.float64]](capacity=UNITREE_G1_OBS_DIM)
            for i in range(UNITREE_G1_OBS_DIM):
                obs.append(res[0].data[i])
            var raw2 = _raw_bodies(env, builtins)
            var want2 = oracle.max_local_self_from_sim(raw2[0], raw2[1], raw2[2], raw2[3], raw2[4])
            _compare(obs, want2, worst_step)
            n_steps += 1
        r += STRIDE

    print("  injected rows:", n_rows, " worst |d| vs the reference's function:", worst_set)
    print("  post-step rows:", n_steps, " worst |d|:", worst_step, " (SYNC_FK_AFTER_STEP)")
    assert_true(n_rows >= 50, "too few rows to gate anything")
    assert_true(worst_set < TOL, "privileged obs after set_state differs by " + String(worst_set))
    assert_true(worst_step < TOL, "privileged obs after a step differs by " + String(worst_step))


def test_device_atan2_matches_libm() raises:
    """`g1_atan2f` (the device path's heading) against the stdlib's `atan2`
    around the whole circle at four radii, plus the axes and the origin:
    worst error below 5e-7 rad, float32's own resolution near π."""
    var worst = 0.0
    var n = 0
    for k in range(4):
        var r: Float32
        if k == 0:
            r = Float32(1e-3)
        elif k == 1:
            r = Float32(1.0)
        elif k == 2:
            r = Float32(37.5)
        else:
            r = Float32(1e4)
        for i in range(200001):
            var ang = Float64(i) / 200000.0 * 2.0 * 3.141592653589793 - 3.141592653589793
            var y = Float32(Float64(r) * sin(ang))
            var x = Float32(Float64(r) * cos(ang))
            var got = Float64(g1_atan2f(y, x))
            var want = atan2(Float64(y), Float64(x))
            var e = abs(got - want)
            if e > 6.0:
                e = abs(e - 2.0 * 3.141592653589793)   # ±π at the branch cut
            if e > worst:
                worst = e
            n += 1
    var axes = List[Float32]()
    axes.append(Float32(1)); axes.append(Float32(0))
    axes.append(Float32(0)); axes.append(Float32(1))
    axes.append(Float32(-1)); axes.append(Float32(0))
    axes.append(Float32(0)); axes.append(Float32(-1))
    for k in range(4):
        var e = abs(Float64(g1_atan2f(axes[2 * k + 1], axes[2 * k])) - atan2(Float64(axes[2 * k + 1]), Float64(axes[2 * k])))
        if e > worst:
            worst = e
    assert_true(g1_atan2f(Float32(0), Float32(0)) == Float32(0), "atan2(0, 0) is 0, as torch")
    print("  g1_atan2f vs libm: worst", worst, "rad over", n + 4, "points")
    assert_true(worst < 5e-7, "device atan2 differs from libm by " + String(worst))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
