"""Rainbow DQN CNN CarRacing — Live Render Eval, COLOR scene (pixel obs).

Like `rainbow_car_racing_pixel_eval_render.mojo`, but renders the REAL top-down
COLOR scene (track, car, wheels) in SDL3 — the nice view — while the agent acts
on its 4x84x84 grayscale pixel observation. Uses the CPU `CarRacingMB` env,
which has the same multi-body physics as the GPU training env and a CPU pixel
rasterizer matching `CarRacingPixel` exactly (same camera + colors), so the CNN
sees in-distribution input.

(The GPU eval `rainbow_car_racing_pixel_eval_render.mojo` instead shows the
agent's literal 84x84 grayscale view. This one shows the pretty color scene.)

The CNN q-net identity below MUST match the training script.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render_cpu.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render_cpu.mojo

Reads checkpoints/rainbow_car_racing_pixel.ckpt.
"""

from std.gpu.host import DeviceContext

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
from mojo_rl.envs.car_racing import CarRacingMB, CarRacingPixel


# =============================================================================
# Config — must match rainbow_car_racing_pixel_training_gpu.mojo's q-net identity
# =============================================================================

comptime OBS_DIM = CarRacingPixel[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = CarRacingPixel[DType.float64].NUM_ACTIONS  # 5
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3
comptime V_MIN = Scalar[DT](-30.0)
comptime V_MAX = Scalar[DT](30.0)

comptime CKPT_PATH = "checkpoints/rainbow_car_racing_pixel.ckpt"

comptime EVAL_EPISODES = 5
comptime FRAME_DELAY_MS = 20  # ~50 FPS
comptime MAX_STEPS = 2_000

comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

comptime RainbowCNNNet = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84], ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20], ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9], ReLU[64 * 7 * 7],
    Flatten[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, HIDDEN],
    NoisyLinear[HIDDEN, (1 + NUM_ACTIONS) * NUM_ATOMS],
    DuelingHeadC51[NUM_ACTIONS, NUM_ATOMS],
]
comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["gpu", OBS_DIM, 1, EVAL_CAP], BATCH_SIZE
]
comptime RainbowTrainer = C51Trainer[
    "gpu", SAMPLE, RainbowCNNNet, NUM_ATOMS, NUM_ACTIONS, True
]


def main() raises:
    print("=" * 70)
    print("Rainbow CarRacing — Live COLOR Render Eval (pixel-trained agent)")
    print("=" * 70)
    print("  Checkpoint:", CKPT_PATH, "  Episodes:", EVAL_EPISODES)
    print("  Window: real color scene; agent acts on its 84x84 pixel view.")
    print()

    with DeviceContext() as ctx:
        var trainer = RainbowTrainer.make(
            ctx=ctx,
            lr=Scalar[DT](6.25e-5),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](0.0),
            learning_starts=0,
            target_update_freq=500,
            max_grad_norm=Scalar[DT](10.0),
            per_alpha=Scalar[DT](0.5),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
            nstep=N_STEP,
            v_min=V_MIN,
            v_max=V_MAX,
        )
        trainer.load_state(String(CKPT_PATH))
        print("Checkpoint loaded. Starting live play...")
        trainer.set_noise_scale(Scalar[DT](0.0))  # deterministic greedy

        var env = CarRacingMB[DT](max_steps=MAX_STEPS)
        _ = env.init_renderer()

        var ep = 0
        var obs = env.reset_pixel()
        var ep_return = Scalar[DT](0.0)
        var ep_steps = 0

        while env.is_renderer_open() and ep < EVAL_EPISODES:
            env.render_frame()  # real top-down color scene

            var action = trainer.select_greedy_action(obs)
            var result = env.step_action_pixel(action)
            obs = result[0].copy()
            ep_return += Scalar[DT](result[1])
            ep_steps += 1

            if result[2] or ep_steps >= MAX_STEPS:
                ep += 1
                print(
                    "Race", ep, " return:", ep_return, " steps:", ep_steps,
                    " tiles:", env.tiles_visited, "/", env.track_length(),
                )
                obs = env.reset_pixel()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

            env.renderer_delay(FRAME_DELAY_MS)
            if env.check_renderer_quit():
                break

        env.close_renderer()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
