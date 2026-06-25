"""Rainbow DQN CNN CarRacing — Live COLOR Render Eval (pixel obs, multi-body).

Watch a pixel-trained CNN agent race in the real top-down SDL3 COLOR scene
(track, car, wheels) while it acts on its 4x84x84 grayscale view.

IMPORTANT — why this drives the GPU env (not the CPU CarRacingMB): the agent is
trained on the GPU `CarRacingPixel` env, whose embedded track generator differs
from `CarRacingMB`'s faithful one (different road width + shape). Rendering a
DIFFERENT env's track would give the agent out-of-distribution input and it
would freeze. So this drives the GPU env for BOTH the observation and the
physics, and color-renders that env's OWN embedded state (car pose + embedded
track) — guaranteeing the picture matches exactly what the agent perceives.

The CNN q-net identity below MUST match the training script.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render_color.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_eval_render_color.mojo

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

from mojo_rl.physics2d import dtype
from mojo_rl.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE, BODY_STATE_SIZE
from mojo_rl.physics2d.car.constants import TILE_DATA_SIZE
from mojo_rl.envs.car_racing import CarRacingPixel, CarRacingDiscrete
from mojo_rl.envs.car_racing.constants import CRConstants
from mojo_rl.render import (
    Renderer2D, RotatingCamera, Transform2D, SDL_Color,
    Vec2 as RenderVec2, car_red, black,
)


# =============================================================================
# Config — must match rainbow_car_racing_pixel_training_gpu.mojo
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
comptime FRAME_DELAY_MS = 20

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

comptime E = CarRacingPixel[DT]
comptime D = CarRacingDiscrete[DT]
comptime SSZ = E.STATE_SIZE
comptime OBS = E.OBS_DIM
comptime WS = E.STEP_WS_PER_ENV


def render_color(mut renderer: Renderer2D, st: List[Scalar[dtype]], num_tiles: Int):
    """Color top-down scene from the GPU env's embedded state row."""
    var ho = D.BODIES_OFFSET
    var hx = Float64(st[ho + IDX_X])
    var hy = Float64(st[ho + IDX_Y])
    var ha = Float64(st[ho + IDX_ANGLE])

    var grass = SDL_Color(102, 204, 102, 255)
    if not renderer.begin_frame_with_color(grass):
        return
    var zoom = CRConstants.ZOOM * CRConstants.SCALE
    var cam = renderer.make_rotating_camera_offset(
        hx, hy, -ha, zoom,
        Float64(CRConstants.WINDOW_W) / 2.0,
        Float64(CRConstants.WINDOW_H) * 3.0 / 4.0,
    )

    # Track tiles (embedded at TRACK_OFFSET).
    var road = SDL_Color(102, 102, 102, 255)
    for i in range(num_tiles):
        var to = D.TRACK_OFFSET + i * TILE_DATA_SIZE
        var verts = List[RenderVec2]()
        verts.append(RenderVec2(Float64(st[to + 0]), Float64(st[to + 1])))
        verts.append(RenderVec2(Float64(st[to + 2]), Float64(st[to + 3])))
        verts.append(RenderVec2(Float64(st[to + 4]), Float64(st[to + 5])))
        verts.append(RenderVec2(Float64(st[to + 6]), Float64(st[to + 7])))
        renderer.draw_polygon_rotating(verts, cam, road, filled=True)

    # Car hull (4 polys) at the hull pose.
    var sz = CRConstants.SIZE
    var red = car_red()
    var tf = Transform2D(hx, hy, ha)
    var p1 = List[RenderVec2]()
    p1.append(RenderVec2(-60.0 * sz, 130.0 * sz)); p1.append(RenderVec2(60.0 * sz, 130.0 * sz))
    p1.append(RenderVec2(60.0 * sz, 110.0 * sz)); p1.append(RenderVec2(-60.0 * sz, 110.0 * sz))
    renderer.draw_transformed_polygon_rotating(p1, tf, cam, red, filled=True)
    var p3 = List[RenderVec2]()
    p3.append(RenderVec2(25.0 * sz, 20.0 * sz)); p3.append(RenderVec2(50.0 * sz, -10.0 * sz))
    p3.append(RenderVec2(50.0 * sz, -40.0 * sz)); p3.append(RenderVec2(20.0 * sz, -90.0 * sz))
    p3.append(RenderVec2(-20.0 * sz, -90.0 * sz)); p3.append(RenderVec2(-50.0 * sz, -40.0 * sz))
    p3.append(RenderVec2(-50.0 * sz, -10.0 * sz)); p3.append(RenderVec2(-25.0 * sz, 20.0 * sz))
    renderer.draw_transformed_polygon_rotating(p3, tf, cam, red, filled=True)

    # Wheels at their rigid-body poses (bodies 1..4).
    var blk = black()
    var hw = 14.0 * sz
    var hr = 27.0 * sz
    for w in range(4):
        var wbo = D.BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
        var wx = Float64(st[wbo + IDX_X])
        var wy = Float64(st[wbo + IDX_Y])
        var wa = Float64(st[wbo + IDX_ANGLE])
        var wtf = Transform2D(wx, wy, wa)
        var wv = List[RenderVec2]()
        wv.append(RenderVec2(-hw, hr)); wv.append(RenderVec2(hw, hr))
        wv.append(RenderVec2(hw, -hr)); wv.append(RenderVec2(-hw, -hr))
        renderer.draw_transformed_polygon_rotating(wv, wtf, cam, blk, filled=True)

    renderer.flip()


def main() raises:
    print("=" * 70)
    print("Rainbow CarRacing — Live COLOR Render Eval (pixel agent, GPU env)")
    print("=" * 70)
    print("  Checkpoint:", CKPT_PATH, "  Episodes:", EVAL_EPISODES)
    print()

    with DeviceContext() as ctx:
        var trainer = RainbowTrainer.make(
            ctx=ctx, lr=Scalar[DT](6.25e-5), gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005), epsilon=Scalar[DT](0.0), learning_starts=0,
            target_update_freq=500, max_grad_norm=Scalar[DT](10.0),
            per_alpha=Scalar[DT](0.5), per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6), nstep=N_STEP, v_min=V_MIN, v_max=V_MAX,
        )
        trainer.load_state(String(CKPT_PATH))
        print("Checkpoint loaded. Starting live play...")
        trainer.set_noise_scale(Scalar[DT](0.0))

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
        var dhost = ctx.enqueue_create_host_buffer[dtype](1)
        var ohost = ctx.enqueue_create_host_buffer[dtype](OBS)
        var shost = ctx.enqueue_create_host_buffer[dtype](SSZ)
        ctx.synchronize()

        E.init_step_workspace_gpu[1](ctx, ws)
        E.reset_kernel_gpu[1, SSZ](ctx, states, rng_seed=7)
        ctx.synchronize()

        var renderer = Renderer2D(
            CRConstants.WINDOW_W, CRConstants.WINDOW_H, 50, "CarRacing Pixel Eval (color)"
        )

        var ep = 0
        var ep_return = Scalar[DT](0.0)
        var ep_steps = 0

        while not renderer.get_should_quit() and ep < EVAL_EPISODES:
            # Copy state (for color render) + obs (for the agent).
            ctx.enqueue_copy(shost, states)
            ctx.enqueue_copy(ohost, obs)
            ctx.synchronize()
            var st_list = List[Scalar[dtype]](capacity=SSZ)
            for i in range(SSZ):
                st_list.append(shost[i])
            var num_tiles = Int(
                Float64(shost[D.METADATA_OFFSET + D.META_NUM_TILES])
            )
            render_color(renderer, st_list, num_tiles)

            var obs_list = List[Scalar[dtype]](capacity=OBS)
            for i in range(OBS):
                obs_list.append(ohost[i])
            var action = trainer.select_greedy_action(obs_list)
            ahost[0] = Scalar[dtype](action)
            ctx.enqueue_copy(actions, ahost)
            E.step_kernel_gpu[1, SSZ, OBS](
                ctx, states, actions, rewards, dones, term, obs,
                rng_seed=0, workspace_ptr=wsp,
            )
            ctx.enqueue_copy(dhost, dones)
            ctx.synchronize()
            ep_steps += 1

            if Float64(dhost[0]) > 0.5:
                ep += 1
                print("Race", ep, " steps:", ep_steps)
                E.reset_kernel_gpu[1, SSZ](ctx, states, rng_seed=UInt64(ep * 101 + 7))
                E.init_step_workspace_gpu[1](ctx, ws)
                ctx.synchronize()
                ep_steps = 0

            renderer.renderer_delay(FRAME_DELAY_MS)

        renderer.close()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
