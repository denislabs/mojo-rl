"""Rainbow DQN CarRacing — Live Render Eval (clean obs, multi-body physics).

Loads a checkpoint saved by `rainbow_car_racing_training_gpu.mojo` and drives
CarRacing in an SDL3 window using deterministic greedy actions (NoisyLinear
noise off), so you can watch the trained agent race.

Eval runs on the CPU `CarRacingMB` env, which shares the EXACT multi-body
physics (`CarDynamicsMB`), the 13-D normalized observation, and the discrete
action decode of the GPU training env (`CarRacingDiscrete`), so the trained
q-net sees the same inputs it trained on. (The procedural track differs per
reset — the agent is trained on per-env random tracks, so it generalizes.)

The trainer config below MUST match the training script's q-net identity
(OBS_DIM, NUM_ACTIONS, NUM_ATOMS, HIDDEN_DIM, V_MIN, V_MAX, N_STEP) — the
checkpoint only stores the q-net + optimizer + epsilon.

Run with (the trainer is a GPU trainer; the env+render are CPU):
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_eval_render.mojo

Reads checkpoints/rainbow_car_racing.ckpt. Window closes on quit (ESC / window
close) or after EVAL_EPISODES races.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.c51.config import RainbowNet
from mojo_rl.deep_agents.training.blocks import NStepSampleStep
from mojo_rl.deep_agents.data.any_per_replay import AnyPerReplay
from mojo_rl.envs.car_racing import CarRacingMB, CarRacingDiscrete


# =============================================================================
# Config — must match rainbow_car_racing_training_gpu.mojo's q-net identity
# =============================================================================

comptime OBS_DIM = CarRacingDiscrete[DT].OBS_DIM  # 13
comptime NUM_ACTIONS = CarRacingDiscrete[DT].NUM_ACTIONS  # 5

comptime HIDDEN_DIM = 256
comptime NUM_ATOMS = 51
comptime N_STEP = 3
# Must match the training script (checkpoint stores only the q-net).
comptime V_MIN = Scalar[DT](-30.0)
comptime V_MAX = Scalar[DT](30.0)

comptime CKPT_PATH = "checkpoints/rainbow_car_racing.ckpt"

# Eval-only knobs.
comptime EVAL_EPISODES = 5
comptime FRAME_DELAY_MS = 20  # ~50 FPS (CarRacing runs at 50 FPS)
comptime MAX_STEPS = 2_000

# Replay is unused at eval time — keep it tiny (checkpoint excludes it).
comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["gpu", OBS_DIM, 1, EVAL_CAP], BATCH_SIZE
]
comptime QNET = RainbowNet[OBS_DIM, NUM_ACTIONS, NUM_ATOMS, HIDDEN_DIM]
comptime RainbowTrainer = C51Trainer[
    "gpu", SAMPLE, QNET, NUM_ATOMS, NUM_ACTIONS, True
]


def main() raises:
    print("=" * 70)
    print("Rainbow CarRacing — Live Render Eval (clean obs, multi-body)")
    print("=" * 70)
    print("  Checkpoint:", CKPT_PATH)
    print("  Episodes:", EVAL_EPISODES)
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

        # Deterministic greedy: zero out NoisyLinear exploration noise.
        trainer.set_noise_scale(Scalar[DT](0.0))

        var env = CarRacingMB[DT](max_steps=MAX_STEPS)
        _ = env.init_renderer()

        var ep: Int = 0
        var obs = env.reset()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        while env.is_renderer_open() and ep < EVAL_EPISODES:
            env.render_frame()

            var action = trainer.select_greedy_action(obs)
            var result = env.step_action(action)
            obs = result[0].copy()
            ep_return += Scalar[DT](result[1])
            ep_steps += 1

            var done = result[2]
            if done or ep_steps >= MAX_STEPS:
                ep += 1
                print(
                    "Race", ep, "return:", ep_return,
                    "steps:", ep_steps,
                    "tiles:", env.tiles_visited, "/", env.track_length(),
                )
                obs = env.reset()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

            env.renderer_delay(FRAME_DELAY_MS)
            if env.check_renderer_quit():
                break

        env.close_renderer()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
