"""Rainbow DQN CNN CarRacing — Live Render Eval (pixel obs, multi-body).

Loads a checkpoint saved by `rainbow_car_racing_pixel_training_gpu.mojo` and
drives CarRacing with deterministic greedy actions (NoisyLinear noise off), so
you can watch the trained CNN agent race. CarRacingPixel is a GPU-only env (no
CPU path), so this drives the GPU env directly with a batch of 1 and renders the
agent's ACTUAL 84x84 grayscale view (the newest frame of its 4-frame stack) —
i.e. exactly what the agent sees: a top-down, car-centered, car-aligned image
with the car at center, the road, and grass.

The CNN q-net identity below MUST match the training script (conv geometry,
HIDDEN, NUM_ATOMS, V_MIN, V_MAX, N_STEP). The checkpoint stores only the q-net +
optimizer + epsilon.

Run with (GPU):
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render.mojo

Reads checkpoints/rainbow_car_racing_pixel.ckpt. Window closes on quit
(ESC / window close) or after EVAL_EPISODES races.
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
from mojo_rl.physics2d import dtype
from mojo_rl.envs.car_racing import CarRacingPixel
from mojo_rl.render import Renderer2D, SDL_Color


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

# Eval-only knobs.
comptime EVAL_EPISODES = 5
comptime FRAME_DELAY_MS = 20  # ~50 FPS (CarRacing runs at 50 FPS)
comptime VIEW_SCALE = 6  # upscale 84x84 -> 504x504 window

comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

# Same Nature-CNN backbone + noisy dueling distributional heads as training.
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

comptime E = CarRacingPixel[DT]
comptime SSZ = E.STATE_SIZE
comptime OBS = E.OBS_DIM
comptime WS = E.STEP_WS_PER_ENV
comptime FS = E.FRAME_SIZE
comptime OW = E.OBS_W
comptime OH = E.OBS_H


def draw_agent_view(
    mut renderer: Renderer2D, host_obs: List[Scalar[dtype]]
):
    """Render the newest 84x84 grayscale frame, upscaled. Grass is the
    background; brighter pixels (road, car) are drawn on top."""
    var bg = SDL_Color(77, 77, 77, 255)  # grass gray (~0.30)
    if not renderer.begin_frame_with_color(bg):
        return
    var nb = 3 * FS  # newest frame in the chronological stack
    for dy in range(OH):
        for dx in range(OW):
            var v = Float64(host_obs[nb + dy * OW + dx])
            if v > 0.45:  # road (~0.70) or car (~1.0)
                var g = UInt8(min(Int(v * 255.0), 255))
                renderer.draw_rect(
                    dx * VIEW_SCALE, dy * VIEW_SCALE,
                    VIEW_SCALE, VIEW_SCALE, SDL_Color(g, g, g, 255), 0,
                )
    renderer.flip()


def main() raises:
    print("=" * 70)
    print("Rainbow CarRacing — Live Render Eval (pixel obs, multi-body)")
    print("=" * 70)
    print("  Checkpoint:", CKPT_PATH, "  Episodes:", EVAL_EPISODES)
    print("  Window shows the agent's 84x84 grayscale view (car-centered).")
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

        # GPU env, batch of 1.
        var states = ctx.enqueue_create_buffer[dtype](SSZ)
        var actions = ctx.enqueue_create_buffer[dtype](1)
        var rewards = ctx.enqueue_create_buffer[dtype](1)
        var dones = ctx.enqueue_create_buffer[dtype](1)
        var term = ctx.enqueue_create_buffer[dtype](1)
        var obs = ctx.enqueue_create_buffer[dtype](OBS)
        var ws = ctx.enqueue_create_buffer[dtype](WS)
        ctx.synchronize()
        var wsp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](ws.unsafe_ptr())

        var ahost = ctx.enqueue_create_host_buffer[dtype](1)
        var rhost = ctx.enqueue_create_host_buffer[dtype](1)
        var dhost = ctx.enqueue_create_host_buffer[dtype](1)
        var ohost = ctx.enqueue_create_host_buffer[dtype](OBS)
        ctx.synchronize()

        E.init_step_workspace_gpu[1](ctx, ws)
        E.reset_kernel_gpu[1, SSZ](ctx, states, rng_seed=7)
        ctx.synchronize()

        var renderer = Renderer2D(OW * VIEW_SCALE, OH * VIEW_SCALE, 50, "CarRacing Pixel Eval")

        var ep = 0
        var ep_return = Scalar[DT](0.0)
        var ep_steps = 0

        while not renderer.get_should_quit() and ep < EVAL_EPISODES:
            # Render current obs (agent's view).
            ctx.enqueue_copy(ohost, obs)
            ctx.synchronize()
            var obs_list = List[Scalar[dtype]](capacity=OBS)
            for i in range(OBS):
                obs_list.append(ohost[i])
            draw_agent_view(renderer, obs_list)

            # Greedy action from the pixel obs, then step.
            var action = trainer.select_greedy_action(obs_list)
            ahost[0] = Scalar[dtype](action)
            ctx.enqueue_copy(actions, ahost)
            E.step_kernel_gpu[1, SSZ, OBS](
                ctx, states, actions, rewards, dones, term, obs,
                rng_seed=0, workspace_ptr=wsp,
            )
            ctx.enqueue_copy(rhost, rewards)
            ctx.enqueue_copy(dhost, dones)
            ctx.synchronize()
            ep_return += Scalar[DT](Float64(rhost[0]))
            ep_steps += 1

            if Float64(dhost[0]) > 0.5:
                ep += 1
                print("Race", ep, " return:", ep_return, " steps:", ep_steps)
                E.reset_kernel_gpu[1, SSZ](ctx, states, rng_seed=UInt64(ep * 101 + 7))
                E.init_step_workspace_gpu[1](ctx, ws)
                ctx.synchronize()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

            renderer.renderer_delay(FRAME_DELAY_MS)

        renderer.close()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
