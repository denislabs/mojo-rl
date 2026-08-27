"""TD-MPC2 MPC inference on Apple — the viewer's frame cost.

    pixi run -e apple mojo run -I . benchmarks/bench_tdmpc2_mpc_inference.mojo

Times `select_action_mpc` at the walker viewer's exact budget (H=3, S=256,
12 pi-trajs, 32 elites, 4 MPPI iterations, BATCH_TOTAL=268) plus the
non-planning `prior` path for reference.

## Measured, M1 Pro, QUIET machine, 2026-08-12

    MPC    70.94 ms (14.1 Hz)  ->  53.59 ms (18.7 Hz)   1.32x
    prior   1.81 ms (551 Hz)   ->   1.51 ms ( 664 Hz)   1.20x

("before" = commit bbe31819, i.e. before the K/N GEMM padding, the MPPI grid
fixes and the LayerNorm+activation fusion.)

## ⚠ HOW TO MEASURE THIS WITHOUT FOOLING YOURSELF

This benchmark produced three DIFFERENT answers on the same machine in one
afternoon — 7.8%, "no change", and 32% — and only the last is real.

  * A/B AGAINST A WORKTREE, INTERLEAVED. Never against a number recorded
    earlier in the session. Same bench file, two module trees:

        git worktree add /tmp/wt_before <baseline-commit>
        pixi run --manifest-path <repo>/pixi.toml -e apple \
            mojo run -I <repo>      benchmarks/bench_tdmpc2_mpc_inference.mojo
        pixi run --manifest-path <repo>/pixi.toml -e apple \
            mojo run -I /tmp/wt_before benchmarks/bench_tdmpc2_mpc_inference.mojo

  * REPORT MIN, not mean. Contention is one-sided — it only ever adds time.
  * CHECK THE SPREAD FIRST. Quiet: max/min ~1.04x, and a 32% effect is
    obvious. Under a concurrent build: 1.7x, and the same 32% is invisible.
    If the spread exceeds the effect you are looking for, the run is worthless
    — wait for an idle machine rather than reporting the number.
  * Warm both JIT caches before timing; `mojo run` compiles first.
"""


from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2

comptime OBS = 24
comptime ACT = 6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime CAP = 1024


def bench_mpc[
    H: Int, SAMPLES: Int, PI_TRAJS: Int, ELITES: Int, ITERS: Int
](ctx: DeviceContext, label: String, reps: Int) raises:
    var agent = TDMPC2[
        "gpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN,
        VMIN, VMAX, H, SAMPLES, PI_TRAJS, ELITES, ITERS,
    ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
    agent.mpc_start_episode()

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.1))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    # warm up (first call builds the callback + compiles every pipeline state)
    for _ in range(5):
        agent.select_action_mpc(obs, act, explore=False)

    var t0 = perf_counter_ns()
    for _ in range(reps):
        agent.select_action_mpc(obs, act, explore=False)
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6 / Float64(reps)
    print(
        "  ", label, ": ", ms, " ms/action  -> ", 1000.0 / ms, " Hz",
        sep="",
    )


def bench_prior(ctx: DeviceContext, reps: Int) raises:
    """The `prior` variant — one encoder + one policy forward, no planning.
    This is the floor a viewer frame could reach with the SAME nets."""
    var agent = TDMPC2[
        "gpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN,
        VMIN, VMAX, 3, 256, 12, 32, 4,
    ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.1))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    for _ in range(5):
        agent.select_action(obs, act, explore=False)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        agent.select_action(obs, act, explore=False)
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6 / Float64(reps)
    print(
        "   prior (encoder+policy, no plan): ", ms, " ms/action  -> ",
        1000.0 / ms, " Hz", sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print()
    print("== the viewer's config (was 76.8 ms / 13.0 Hz) ==")
    bench_mpc[3, 256, 12, 32, 4](ctx, "H=3 S=256 pi=12 el=32 iters=4", 40)
    bench_mpc[3, 256, 12, 32, 4](ctx, "H=3 S=256 pi=12 el=32 iters=4", 40)
    print()
    print("== reference: the non-planning path (was 1.98 ms / 504 Hz) ==")
    bench_prior(ctx, 200)
