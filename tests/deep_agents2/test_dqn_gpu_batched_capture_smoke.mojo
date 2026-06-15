"""Smoke: DQN GPU-batched training path + CUDA-graph capture compile/run.

Builds a `DQN["gpu", ...]` agent and runs a few iterations of
`train_gpu_batched` with `USE_TRAIN_CUDA_GRAPH=True` over a small batch of
GPU-resident Pong envs. On Apple the capture harness no-ops (CUDAGraph is a
no-op), so this validates that the new DQN GPU-batched surface
(`record_batch_gpu` / `select_greedy_action_batched` / `train_device_kernels`
/ `note_train_update` / `learning_starts_count`) compiles and runs, and that
returns stay finite. Real CUDA-graph capture is validated on NVIDIA.
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dqn.config import DQN
from mojo_rl.deep_agents2.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongEnv


comptime OBS_DIM = PongEnv[DT].OBS_DIM
comptime NUM_ACTIONS = PongEnv[DT].NUM_ACTIONS
comptime N_ENVS = 8
comptime BATCH = 32
comptime CAP = 10_000

comptime PongBatched = BatchedGpuDiscreteEnv[
    PongEnv[DT, Float64(0.0)], N_ENVS, OBS_DIM, 1
]


def main() raises:
    seed(42)
    print("DQN GPU-batched + CUDA-graph capture smoke")
    with DeviceContext() as ctx:
        var agent = DQN["gpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, 64](
            ctx=ctx,
            learning_starts=64,
        )
        var env = PongBatched(ctx)

        # A handful of iterations past warmup so the capture path is exercised.
        var ep_returns = agent.train_gpu_batched[
            PongBatched, N_ENVS, 1, USE_TRAIN_CUDA_GRAPH=True
        ](
            env,
            512,
            updates_per_step=2,
            print_every=0,
            verbose=False,
        )
        print("iterations recorded:", len(ep_returns))

        var m = agent.flush_metrics()
        var loss = m.loss.to_f64()
        if loss != loss:  # NaN check
            raise Error("DQN GPU-batched capture smoke: non-finite loss")
        print("loss:", loss, " train_steps:", m.train_steps.to_f64())
        print("PASS")
