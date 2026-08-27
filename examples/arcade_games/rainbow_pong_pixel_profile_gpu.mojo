"""Rainbow DQN CNN GPU on Pong Pixel — short run for nsys profiling.

Profiling counterpart of `rainbow_pong_pixel_training_gpu.mojo`. Same env,
network (Nature CNN + noisy dueling distributional heads), N_ENVS, batch size,
replay capacity, and grad-steps/iter so an nsys trace is representative of the
real training loop — the only differences are a short step count, minimal
warmup, and no logger / checkpoint / eval machinery.

The point of this script: figure out why the im2col+GEMM Conv2D rewrite gave a
~7× speedup on AlphaZero but only ~5% here. nsys (or `--stats`) will show how
much of the iteration is conv forward/backward vs the rest (env pixel render,
GPU replay store/sample, C51 projection, PER sum-tree, NoisyLinear resample).

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_pixel_profile_gpu.mojo  # compile/smoke
    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        examples/arcade_games/rainbow_pong_pixel_profile_gpu.mojo                              # profile
"""

from std.random import seed
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn.primitives.dueling_head_c51 import DuelingHeadC51

from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.training.blocks import NStepSampleStep
from mojo_rl.deep_agents.data.any_per_replay import AnyPerReplay
from mojo_rl.deep_agents.training import (
    BatchedGpuDiscreteEnv,
    run_offpolicy_discrete_train_gpu_batched,
)
from mojo_rl.envs.arcade_games.pong import PongPixelEnv
from mojo_rl.core.fmt import fit


# =============================================================================
# Constants (mirror rainbow_pong_pixel_training_gpu.mojo, short run)
# =============================================================================

comptime OBS_DIM = PongPixelEnv[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = PongPixelEnv[DType.float64].NUM_ACTIONS  # 3
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

comptime BUFFER_CAPACITY = 12_000
comptime BATCH_SIZE = 32
comptime N_ENVS = 64

comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)
comptime HIT_REWARD = 0.0

comptime GRAD_STEPS = 16
# Short run: warmup just past the n-step fill so train_step actually fires, and
# enough total steps to amortize compile + capture a representative steady state.
comptime WARMUP = 2_000
comptime NUM_STEPS = 50_000
comptime LR = Scalar[DT](6.25e-5)


comptime RainbowCNNNet = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84],
    ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20],
    ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9],
    ReLU[64 * 7 * 7],
    Flatten[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, HIDDEN],
    NoisyLinear[HIDDEN, (1 + NUM_ACTIONS) * NUM_ATOMS],
    DuelingHeadC51[NUM_ACTIONS, NUM_ATOMS],
]

comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["gpu", OBS_DIM, 1, BUFFER_CAPACITY], BATCH_SIZE
]
comptime RainbowTrainer = C51Trainer[
    "gpu", SAMPLE, RainbowCNNNet, NUM_ATOMS, NUM_ACTIONS, True
]
comptime PongPixelBatched = BatchedGpuDiscreteEnv[
    PongPixelEnv[DT, HIT_REWARD], N_ENVS, OBS_DIM, 1
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=== Rainbow Pong Pixel nsys profile (deep_agents / nn) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP)
    print("  N_ENVS:", N_ENVS, "| BATCH:", BATCH_SIZE, "| Grad/iter:", GRAD_STEPS)
    print("  Obs: 4x84x84 =", OBS_DIM, "| N-step:", N_STEP)
    print()

    with DeviceContext() as ctx:
        var trainer = RainbowTrainer.make(
            ctx=ctx,
            lr=LR,
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](0.0),
            learning_starts=WARMUP,
            target_update_freq=500,
            max_grad_norm=Scalar[DT](10.0),
            per_alpha=Scalar[DT](0.5),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
            nstep=N_STEP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = PongPixelBatched(ctx)

        var start = perf_counter_ns()

        # No logger, no checkpoints, no eval — just the collect→train loop.
        _ = run_offpolicy_discrete_train_gpu_batched[
            RainbowTrainer, PongPixelBatched, N_ENVS, N_STEP
        ](
            ctx,
            trainer,
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=GRAD_STEPS,
            print_every=10_000,
            verbose=True,
            nstep_gamma=Scalar[DT](0.99),
            episode_sync_every=32,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Time:", fit(String(elapsed), 6), "s")
        print(
            "Transitions/second:",
            fit(String(Float64(NUM_STEPS) / elapsed), 9),
        )
        print("=== Done ===")
