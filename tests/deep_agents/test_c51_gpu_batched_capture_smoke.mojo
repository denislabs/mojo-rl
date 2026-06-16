"""Smoke: C51 + Rainbow GPU-batched training + CUDA-graph capture compile/run.

Runs a few iterations of `train_gpu_batched` with `USE_TRAIN_CUDA_GRAPH=True`
for both a plain (uniform-replay) C51 agent and a Rainbow agent (PER + N-step).
On Apple the capture harness no-ops, so this validates that the C51 capture
surface — CE `forward_accumulate`/`read_accum`, the device distributional diag
folded into the captured sequence, and the device-PER replay (Rainbow) — all
compile and run with finite losses. Real CUDA-graph capture is on NVIDIA.
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.c51.config import C51, Rainbow
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongEnv


comptime OBS_DIM = PongEnv[DT].OBS_DIM
comptime NUM_ACTIONS = PongEnv[DT].NUM_ACTIONS
comptime N_ENVS = 8
comptime BATCH = 32
comptime CAP = 10_000
comptime NA = 51
comptime HIDDEN = 64
comptime N_STEP = 3

comptime PongBatched = BatchedGpuDiscreteEnv[
    PongEnv[DT, Float64(0.0)], N_ENVS, OBS_DIM, 1
]


def main() raises:
    seed(42)
    print("C51 + Rainbow GPU-batched + CUDA-graph capture smoke")
    with DeviceContext() as ctx:
        # ── Plain C51 (uniform replay) ────────────────────────────────
        var c51 = C51["gpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, NA, HIDDEN](
            ctx=ctx, learning_starts=64, v_min=Scalar[DT](-2.0),
            v_max=Scalar[DT](2.0),
        )
        var env_c = PongBatched(ctx)
        _ = c51.train_gpu_batched[
            PongBatched, N_ENVS, 1, USE_TRAIN_CUDA_GRAPH=True
        ](
            env_c, 512, updates_per_step=2, print_every=0, verbose=False,
        )
        var mc = c51.flush_metrics()
        var lc = mc.loss.to_f64()
        if lc != lc:
            raise Error("C51 capture smoke: non-finite loss")
        print("C51 loss:", lc, " train_steps:", mc.train_steps.to_f64())

        # ── Rainbow (PER + N-step + device sum-tree) ──────────────────
        var rb = Rainbow[
            "gpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, NA, HIDDEN, N_STEP
        ](
            ctx=ctx, learning_starts=64, v_min=Scalar[DT](-2.0),
            v_max=Scalar[DT](2.0),
        )
        var env_r = PongBatched(ctx)
        _ = rb.train_gpu_batched[
            PongBatched, N_ENVS, N_STEP, USE_TRAIN_CUDA_GRAPH=True
        ](
            env_r, 512, updates_per_step=2, print_every=0, verbose=False,
            nstep_gamma=Scalar[DT](0.99),
        )
        var mr = rb.flush_metrics()
        var lr = mr.loss.to_f64()
        if lr != lr:
            raise Error("Rainbow capture smoke: non-finite loss")
        print("Rainbow loss:", lr, " train_steps:", mr.train_steps.to_f64())
        print("PASS")
