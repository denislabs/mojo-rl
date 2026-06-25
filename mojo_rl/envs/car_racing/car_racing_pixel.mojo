"""CarRacingPixel — GPU-batched, PIXEL-observation CarRacing (multi-body physics).

The faithful-to-Gymnasium observation: a top-down rendered image, NOT a state
vector. Gymnasium CarRacing is pixels-only (96x96x3); a state vector cannot
convey where the road is relative to the car, so a clean-obs agent can't learn
to follow the track. This env renders a car-centered, car-aligned top-down view
into a 4x84x84 grayscale frame stack (Atari-style), so the agent SEES the road.

It reuses `CarRacingDiscrete` wholesale for physics / embedded track / discrete
actions / reward / termination (identical state layout) and adds only:
  - a per-env GPU workspace holding the 4-frame stack + ring index, and
  - an inverse-camera rasterizer (1 GPU thread per output pixel): each pixel is
    inverse-transformed into world space and tested against the embedded track
    tiles (road vs grass); the car is a fixed sprite at the view center since
    the camera follows + rotates with it.

Observation: 4x84x84 grayscale in [0,1] (OBS_DIM = 28224), chronological stack.
Actions: 5 discrete (same as CarRacingDiscrete). Reward: Gymnasium-faithful.

The frame stack lives in STEP_WS_PER_ENV workspace (not state), exactly the
pong_pixel convention that BatchedGpuDiscreteEnv already supports (it skips the
state-prefix obs extraction when OBS_DIM > STATE_SIZE and renders on first step).
"""

from std.math import sin, cos
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core import GPUDiscreteEnv
from mojo_rl.physics2d import dtype
from mojo_rl.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE
from mojo_rl.physics2d.car import TileCollision
from mojo_rl.physics2d.car.constants import TILE_DATA_SIZE
from .car_racing_discrete import CarRacingDiscrete


struct CarRacingPixel[DTYPE: DType](GPUDiscreteEnv, Copyable, Movable):
    """GPU-batched discrete CarRacing with 4x84x84 pixel observations."""

    comptime dtype = Self.DTYPE
    comptime NAME: String = "CarRacingPixel"

    # Reuse the discrete env's physics / track / reward / layout verbatim.
    comptime D = CarRacingDiscrete[Self.DTYPE]
    comptime STATE_SIZE: Int = Self.D.STATE_SIZE
    comptime NUM_ACTIONS: Int = Self.D.NUM_ACTIONS  # 5

    # Pixel observation geometry (Atari-style).
    comptime OBS_W: Int = 84
    comptime OBS_H: Int = 84
    comptime FRAME_STACK: Int = 4
    comptime FRAME_SIZE: Int = Self.OBS_W * Self.OBS_H  # 7056
    comptime OBS_DIM: Int = Self.FRAME_STACK * Self.FRAME_SIZE  # 28224

    # Per-env workspace: [4 frames | frame_idx].
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = Self.FRAME_STACK * Self.FRAME_SIZE + 1
    comptime WS_IDX: Int = Self.FRAME_STACK * Self.FRAME_SIZE  # idx slot in env_ws

    # Camera: car drawn at (CX, CY), forward = screen up; ZOOM_PX px per world
    # unit. CY below center -> more road visible ahead. Tunable.
    comptime CX: Float64 = 42.0
    comptime CY: Float64 = 52.0
    comptime ZOOM_PX: Float64 = 1.3
    comptime CAR_HW: Float64 = 1.4  # car half-width (world units)
    comptime CAR_HL: Float64 = 2.8  # car half-length
    # Grayscale surface values.
    comptime C_GRASS: Float64 = 0.30
    comptime C_ROAD: Float64 = 0.70
    comptime C_CAR: Float64 = 1.0

    comptime TPB: Int = 64  # pixel rendering is heavy — fewer threads/block

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Per-pixel inverse-camera rasterizer
    # =========================================================================
    @always_inline
    @staticmethod
    def _render_pixel[
        BATCH: Int, STATE_SIZE: Int
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        dx: Int,
        dy: Int,
        num_tiles: Int,
    ) -> Scalar[dtype]:
        """Grayscale value [0,1] for output pixel (dx, dy).

        All math is float32 — Metal GPU kernels do not support float64."""
        var zero = Scalar[dtype](0.0)
        var cx = Scalar[dtype](Self.CX)
        var cy = Scalar[dtype](Self.CY)
        var zoom = Scalar[dtype](Self.ZOOM_PX)
        # Camera-frame coords (world units): +x right, +y forward (screen-up).
        var camx = (Scalar[dtype](dx) - cx) / zoom
        var camy = (cy - Scalar[dtype](dy)) / zoom

        # Car sprite: fixed at view center (camera follows + aligns with car).
        var acx = camx if camx >= zero else -camx
        var acy = camy if camy >= zero else -camy
        if acx < Scalar[dtype](Self.CAR_HW) and acy < Scalar[dtype](Self.CAR_HL):
            return Scalar[dtype](Self.C_CAR)

        # Inverse rotate+translate camera -> world.
        var ho = Self.D.BODIES_OFFSET  # hull = body 0
        var car_x = rebind[Scalar[dtype]](states[env, ho + IDX_X])
        var car_y = rebind[Scalar[dtype]](states[env, ho + IDX_Y])
        var a = rebind[Scalar[dtype]](states[env, ho + IDX_ANGLE])
        var ca = cos(a)
        var sa = sin(a)
        # right=(cos,sin), forward=(-sin,cos): world = car + camx*right + camy*fwd
        var wx = car_x + camx * ca - camy * sa
        var wy = car_y + camx * sa + camy * ca

        for i in range(num_tiles):
            var to = Self.D.TRACK_OFFSET + i * TILE_DATA_SIZE
            var v0x = rebind[Scalar[dtype]](states[env, to + 0])
            var v0y = rebind[Scalar[dtype]](states[env, to + 1])
            var v1x = rebind[Scalar[dtype]](states[env, to + 2])
            var v1y = rebind[Scalar[dtype]](states[env, to + 3])
            var v2x = rebind[Scalar[dtype]](states[env, to + 4])
            var v2y = rebind[Scalar[dtype]](states[env, to + 5])
            var v3x = rebind[Scalar[dtype]](states[env, to + 6])
            var v3y = rebind[Scalar[dtype]](states[env, to + 7])
            if TileCollision.point_in_quad(
                wx, wy, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
            ):
                return Scalar[dtype](Self.C_ROAD)
        return Scalar[dtype](Self.C_GRASS)

    # =========================================================================
    # GPUDiscreteEnv launchers
    # =========================================================================
    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int
    ](ctx: DeviceContext, mut states: DeviceBuffer[dtype], rng_seed: UInt64 = 0) raises:
        # Physics/track reset is identical to the discrete env; the frame stack
        # is zeroed by init_step_workspace_gpu and warms up over the first steps.
        Self.D.reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](ctx, states, rng_seed)

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[UnsafePointer[Scalar[dtype], MutAnyOrigin]] = None,
        rng_counter_ptr: Optional[UnsafePointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var st = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)](states)
        var dn = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE)](dones)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var ws = workspace_ptr.value()

        @parameter
        @always_inline
        def sel_wrap(
            st: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
            dn: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
            seed: Scalar[DType.uint64],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            if rebind[Scalar[dtype]](dn[env]) > Scalar[dtype](0.5):
                var s = Int(seed) * 2654435761 + env * 40503
                CarRacingDiscrete[Self.dtype]._reset_env[BATCH_SIZE, STATE_SIZE](st, env, s)
                dn[env] = Scalar[dtype](0.0)
                # Clear this env's frame stack + ring index.
                var env_ws = ws_ptr + env * Self.STEP_WS_PER_ENV
                for i in range(Self.STEP_WS_PER_ENV):
                    env_ws[i] = Scalar[dtype](0.0)

        ctx.enqueue_function[sel_wrap](
            st, dn, ws, Scalar[DType.uint64](rng_seed),
            grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[UnsafePointer[Scalar[dtype], MutAnyOrigin]] = None,
        rng_counter_ptr: Optional[UnsafePointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var st = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)](states)
        var ac = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1)](actions)
        var rw = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE)](rewards)
        var dn = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE)](dones)
        var tm = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE)](terminated)
        var ws = workspace_ptr.value()
        var obs_ptr = obs.unsafe_ptr()

        # ── Kernel A: physics + reward + termination (1 thread / env) ──
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @parameter
        @always_inline
        def phys_wrap(
            st: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
            ac: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), ImmutAnyOrigin],
            rw: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            dn: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            tm: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            CarRacingDiscrete[Self.dtype]._step_env[BATCH_SIZE, STATE_SIZE, 1](
                env, st, ac, rw, dn
            )
            var is_tr = rebind[Scalar[dtype]](
                st[env, Self.D.METADATA_OFFSET + Self.D.META_TRUNCATED]
            )
            tm[env] = dn[env] * (Scalar[dtype](1.0) - is_tr)

        ctx.enqueue_function[phys_wrap](
            st, ac, rw, dn, tm, grid_dim=(BLOCKS,), block_dim=(Self.TPB,)
        )

        # ── Kernel B: render + frame-stack output (1 thread / (env, pixel)) ──
        comptime RT = BATCH_SIZE * Self.FRAME_SIZE
        comptime RTPB = 256
        comptime RBLOCKS = (RT + RTPB - 1) // RTPB

        @parameter
        @always_inline
        def render_wrap(
            st: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
            ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
            o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= RT:
                return
            var env = tid // Self.FRAME_SIZE
            var pix = tid % Self.FRAME_SIZE
            var dy = pix // Self.OBS_W
            var dx = pix % Self.OBS_W

            var n = Int(
                rebind[Scalar[dtype]](
                    st[env, Self.D.METADATA_OFFSET + Self.D.META_NUM_TILES]
                )
            )
            if n <= 0:
                n = 1
            var gray = CarRacingPixel[Self.dtype]._render_pixel[BATCH_SIZE, STATE_SIZE](
                env, st, dx, dy, n
            )

            var env_ws = ws_ptr + env * Self.STEP_WS_PER_ENV
            var slot = Int(env_ws[Self.WS_IDX]) % Self.FRAME_STACK
            env_ws[slot * Self.FRAME_SIZE + pix] = gray

            var env_obs = o_ptr + env * Self.OBS_DIM
            for f in range(Self.FRAME_STACK):
                var rs = (slot + 1 + f) % Self.FRAME_STACK
                env_obs[f * Self.FRAME_SIZE + pix] = env_ws[rs * Self.FRAME_SIZE + pix]

        ctx.enqueue_function[render_wrap](
            st, ws, obs_ptr, grid_dim=(RBLOCKS,), block_dim=(RTPB,)
        )

        # ── Kernel C: advance ring index (1 thread / env) ──
        @parameter
        @always_inline
        def adv_wrap(ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            var env_ws = ws_ptr + env * Self.STEP_WS_PER_ENV
            var slot = Int(env_ws[Self.WS_IDX])
            env_ws[Self.WS_IDX] = Scalar[dtype]((slot + 1) % Self.FRAME_STACK)

        ctx.enqueue_function[adv_wrap](
            ws, grid_dim=(BLOCKS,), block_dim=(Self.TPB,)
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype]) raises:
        comptime WS_TOTAL = BATCH_SIZE * Self.STEP_WS_PER_ENV
        comptime BLK = (WS_TOTAL + 256 - 1) // 256
        var ws = workspace_buf.unsafe_ptr()

        @parameter
        @always_inline
        def zero_wrap(ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= WS_TOTAL:
                return
            ws_ptr[i] = Scalar[dtype](0.0)

        ctx.enqueue_function[zero_wrap](ws, grid_dim=(BLK,), block_dim=(256,))

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        pass

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Never used for pixel obs (OBS_DIM > STATE_SIZE → BatchedGpuDiscreteEnv
        skips it and renders on the first step). Provided for trait conformance;
        zero-fills so it is well-defined if ever called."""
        var ob = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OBS_DIM)](obs)
        comptime TOT = BATCH_SIZE * OBS_DIM
        comptime BLK = (TOT + 256 - 1) // 256

        @parameter
        @always_inline
        def zero_obs(o: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin]):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= TOT:
                return
            o[i // OBS_DIM, i % OBS_DIM] = Scalar[dtype](0.0)

        ctx.enqueue_function[zero_obs](ob, grid_dim=(BLK,), block_dim=(256,))
