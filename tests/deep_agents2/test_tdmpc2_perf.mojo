"""TD-MPC2 train_step throughput micro-benchmark (Apple).

Times N train_steps (after filling replay) for CPU B=256, GPU B=32, GPU
B=256 — to see whether the GPU path is fixed-overhead-bound (per-step
allocs / D2H syncs → bigger B amortizes) or compute-bound. Not a pass/fail
gate; prints ms/step + samples/sec.

Run: `pixi run -e apple mojo run -I . tests/deep_agents2/test_tdmpc2_perf.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent

comptime OBS = 17
comptime ENC = 256
comptime ACT = 6
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime H = 3
comptime CAP = 20000
comptime FILL = 1100
comptime N = 20


def main() raises:
    print("=" * 70)
    print("TD-MPC2 train_step throughput (Apple) — N =", N, "steps")
    print("=" * 70)
    var ctx = DeviceContext()
    seed(1)

    # CPU B=256
    comptime AgC = TDMPC2Agent[
        "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, 256, H, CAP
    ]
    var agc = AgC.make(lr=Scalar[DT](1e-3), learning_starts=0)
    var _ob = alloc[Scalar[DT]](OBS)
    var _ac = alloc[Scalar[DT]](ACT)
    for _ in range(FILL):
        for i in range(OBS):
            _ob[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        _ac[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        agc.record(_ob, _ac, Scalar[DT](random_float64() - 1.0), Scalar[DT](0.0))
    _ob.free(); _ac.free()
    _ = agc.train_step()  # warmup
    var t0 = perf_counter_ns()
    for _ in range(N):
        _ = agc.train_step()
    var t1 = perf_counter_ns()
    var ms_c = Float64(t1 - t0) / 1.0e6 / Float64(N)
    print("  CPU  B=256 :", ms_c, "ms/step  (", 256.0 / (ms_c / 1000.0), "samp/s )")

    # GPU B=32
    comptime AgG32 = TDMPC2Agent[
        "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, 32, H, CAP
    ]
    var agg32 = AgG32.make(lr=Scalar[DT](1e-3), learning_starts=0, ctx=ctx)
    var _ob2 = alloc[Scalar[DT]](OBS)
    var _ac2 = alloc[Scalar[DT]](ACT)
    for _ in range(FILL):
        for i in range(OBS):
            _ob2[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        _ac2[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        agg32.record(_ob2, _ac2, Scalar[DT](random_float64() - 1.0), Scalar[DT](0.0))
    _ob2.free(); _ac2.free()
    _ = agg32.train_step()
    var t2 = perf_counter_ns()
    for _ in range(N):
        _ = agg32.train_step()
    var t3 = perf_counter_ns()
    var ms_g32 = Float64(t3 - t2) / 1.0e6 / Float64(N)
    print("  GPU  B=32  :", ms_g32, "ms/step  (", 32.0 / (ms_g32 / 1000.0), "samp/s )")

    # GPU B=256
    comptime AgG256 = TDMPC2Agent[
        "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, 256, H, CAP
    ]
    var agg256 = AgG256.make(lr=Scalar[DT](1e-3), learning_starts=0, ctx=ctx)
    var _ob3 = alloc[Scalar[DT]](OBS)
    var _ac3 = alloc[Scalar[DT]](ACT)
    for _ in range(FILL):
        for i in range(OBS):
            _ob3[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        _ac3[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        agg256.record(_ob3, _ac3, Scalar[DT](random_float64() - 1.0), Scalar[DT](0.0))
    _ob3.free(); _ac3.free()
    _ = agg256.train_step()
    var t4 = perf_counter_ns()
    for _ in range(N):
        _ = agg256.train_step()
    var t5 = perf_counter_ns()
    var ms_g256 = Float64(t5 - t4) / 1.0e6 / Float64(N)
    print("  GPU  B=256 :", ms_g256, "ms/step  (", 256.0 / (ms_g256 / 1000.0), "samp/s )")
    print("=" * 70)
