"""Native Pong — Pixel observation variant for CNN-based DQN training.

Wraps PongEnv physics with GPU-rendered pixel observations.
After each physics step, renders the game to a 160×210 grayscale framebuffer,
resizes to 84×84, and maintains a 4-frame stack for temporal context.

State layout: same as PongEnv (STATE_SIZE = 12)
Pixel obs (OBS_DIM = 28224): 4 × 84 × 84 grayscale frames, normalized to [0, 1]
Actions: 0=NOOP, 1=UP, 2=DOWN (same as PongEnv)

GPU workspace per env (PIXEL_WS_PER_ENV = 36625 float32):
  [0..8399]          160×210 grayscale framebuffer (packed 4 UInt8 per float32)
  [8400..36623]      4 × 84×84 frame stack (float32)
  [36624]            frame_idx (ring buffer write position)

Usage:
    from mojo_rl.envs.arcade_games.pong import PongPixelEnv

    # GPU training with CNN DQN
    var metrics = agent.train_gpu[PongPixelEnv[DType.float32]](ctx, ...)
"""

from std.random import random_float64
from std.memory import alloc, memset
from mojo_rl.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.nn.gpu import random_range, xorshift32
from layout import LayoutTensor, Layout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ..core.gpu_env import ArcadeGameState, ArcadeGameAction, gpu_dtype
from ..core.gpu_env import (
    FRAME_BUF_F32_SIZE,
    FRAME_STACK_F32_SIZE,
    PIXEL_WS_PER_ENV,
)
from ..core.colors import (
    SCREEN_W,
    SCREEN_H,
    OBS_W,
    OBS_H,
    FRAME_STACK,
    PIXEL_OBS_DIM,
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_GRAY,
)
from ..core.gpu_renderer import (
    clear_frame,
    draw_filled_rect,
    draw_dashed_vline,
    draw_number,
)

# Import PongEnv for physics kernels
from .pong import (
    PongEnv,
    S_BALL_X,
    S_BALL_Y,
    S_BALL_VX,
    S_BALL_VY,
    S_PADDLE_Y,
    S_CPU_PADDLE_Y,
    S_PLAYER_SCORE,
    S_CPU_SCORE,
    S_STEP_COUNT,
    PADDLE_WIDTH,
    PADDLE_HEIGHT,
    BALL_SIZE,
    LEFT_PADDLE_X,
    RIGHT_PADDLE_X,
    MAX_BALL_VY,
)


# ============================================================================
# Inline helpers for GPU pixel rendering
# ============================================================================


@always_inline
fn _render_pong_frame(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    bx: Int,
    by: Int,
    pad_y: Int,
    cpu_y: Int,
    p_score: Int,
    c_score: Int,
):
    """Render Pong game state to 160×210 grayscale buffer."""
    clear_frame(buf)

    # Center dashed line
    draw_dashed_vline(buf, SCREEN_W // 2, 0, SCREEN_H, 4, 4, COLOR_GRAY)

    # CPU paddle (left)
    draw_filled_rect(
        buf,
        LEFT_PADDLE_X,
        cpu_y - PADDLE_HEIGHT // 2,
        PADDLE_WIDTH,
        PADDLE_HEIGHT,
        COLOR_WHITE,
    )

    # Agent paddle (right)
    draw_filled_rect(
        buf,
        RIGHT_PADDLE_X,
        pad_y - PADDLE_HEIGHT // 2,
        PADDLE_WIDTH,
        PADDLE_HEIGHT,
        COLOR_WHITE,
    )

    # Ball
    draw_filled_rect(
        buf,
        bx - BALL_SIZE,
        by - BALL_SIZE,
        BALL_SIZE * 2,
        BALL_SIZE * 2,
        COLOR_WHITE,
    )

    # Scores at top
    draw_number(buf, 30, 5, c_score, COLOR_WHITE, 2)
    draw_number(buf, 110, 5, p_score, COLOR_WHITE, 2)


@always_inline
fn _resize_and_push[
    WS_FRAME_STACK: Int,
    WS_FRAME_IDX: Int,
](
    frame_buf: UnsafePointer[UInt8, MutAnyOrigin],
    workspace: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
    obs_out: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
):
    """Resize 160×210 to 84×84, push to frame stack, output 4-frame obs.

    Combines resize + normalize + frame stack update in one pass.
    """
    comptime FRAME_SIZE = OBS_W * OBS_H  # 7056

    # Get current write slot
    var slot = Int(workspace[WS_FRAME_IDX]) % FRAME_STACK
    var slot_base = WS_FRAME_STACK + slot * FRAME_SIZE

    # Resize from 160×210 to 84×84, normalize to [0,1], write to frame stack
    for dy in range(OBS_H):
        var sy0 = dy * SCREEN_H // OBS_H
        var sy1 = (dy + 1) * SCREEN_H // OBS_H
        if sy1 == sy0:
            sy1 = sy0 + 1

        for dx in range(OBS_W):
            var sx0 = dx * SCREEN_W // OBS_W
            var sx1 = (dx + 1) * SCREEN_W // OBS_W
            if sx1 == sx0:
                sx1 = sx0 + 1

            # Box-filter average
            var total: Int = 0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Int(frame_buf[sy * SCREEN_W + sx])
                    count += 1

            workspace[slot_base + dy * OBS_W + dx] = (
                Scalar[gpu_dtype](total // count) / 255.0
            )

    # Advance ring buffer index
    workspace[WS_FRAME_IDX] = Scalar[gpu_dtype]((slot + 1) % FRAME_STACK)

    # Output chronological frame stack: oldest → newest
    for f in range(FRAME_STACK):
        var read_slot = (slot + 1 + f) % FRAME_STACK
        var read_base = WS_FRAME_STACK + read_slot * FRAME_SIZE
        var out_base = f * FRAME_SIZE
        for i in range(FRAME_SIZE):
            obs_out[out_base + i] = workspace[read_base + i]


# ============================================================================
# PongPixelEnv
# ============================================================================


struct PongPixelEnv[DTYPE: DType where DTYPE.is_floating_point()](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Native Pong with pixel observations for CNN-based training.

    Uses the same physics as PongEnv but produces 4×84×84 pixel observations
    instead of 6D clean observations.

    CPU: Renders to internal grayscale buffer for get_obs_list().
    GPU: Renders in-kernel, maintains per-env frame stacks in workspace.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = ArcadeGameState
    comptime ActionType = ArcadeGameAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = 12
    comptime OBS_DIM: Int = PIXEL_OBS_DIM  # 4 × 84 × 84 = 28224
    comptime NUM_ACTIONS: Int = 3  # NOOP, UP, DOWN
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = PIXEL_WS_PER_ENV  # 36625

    # Delegate to PongEnv for physics + rendering
    var inner: PongEnv[Self.DTYPE]

    # CPU pixel observation buffers
    var _frame_buf: UnsafePointer[UInt8, MutAnyOrigin]  # 160×210 grayscale
    var _frame_stack: UnsafePointer[Scalar[Self.DTYPE], MutAnyOrigin]  # 4×84×84
    var _frame_idx: Int

    fn __init__(out self):
        self.inner = PongEnv[Self.DTYPE]()
        self._frame_buf = alloc[UInt8](SCREEN_W * SCREEN_H)
        self._frame_stack = alloc[Scalar[Self.DTYPE]](PIXEL_OBS_DIM)
        self._frame_idx = 0
        # Zero frame stack
        for i in range(PIXEL_OBS_DIM):
            self._frame_stack[i] = 0

    fn __del__(deinit self):
        if self._frame_buf:
            self._frame_buf.free()
        if self._frame_stack:
            self._frame_stack.free()

    # ========================================================================
    # CPU: render game to pixel obs
    # ========================================================================

    fn _render_to_buf(self):
        """Render current Pong state to internal grayscale buffer."""
        _render_pong_frame(
            self._frame_buf,
            Int(self.inner.state[S_BALL_X]),
            Int(self.inner.state[S_BALL_Y]),
            Int(self.inner.state[S_PADDLE_Y]),
            Int(self.inner.state[S_CPU_PADDLE_Y]),
            Int(self.inner.state[S_PLAYER_SCORE]),
            Int(self.inner.state[S_CPU_SCORE]),
        )

    fn _push_frame(mut self):
        """Resize framebuffer to 84×84 and push to frame stack."""
        comptime FRAME_SIZE = OBS_W * OBS_H  # 7056
        var slot_offset = self._frame_idx * FRAME_SIZE

        for dy in range(OBS_H):
            var sy0 = dy * SCREEN_H // OBS_H
            var sy1 = (dy + 1) * SCREEN_H // OBS_H
            if sy1 == sy0:
                sy1 = sy0 + 1

            for dx in range(OBS_W):
                var sx0 = dx * SCREEN_W // OBS_W
                var sx1 = (dx + 1) * SCREEN_W // OBS_W
                if sx1 == sx0:
                    sx1 = sx0 + 1

                var total: Int = 0
                var count: Int = 0
                for sy in range(sy0, sy1):
                    for sx in range(sx0, sx1):
                        total += Int(self._frame_buf[sy * SCREEN_W + sx])
                        count += 1

                self._frame_stack[slot_offset + dy * OBS_W + dx] = (
                    Scalar[Self.DTYPE](total // count) / 255.0
                )

        self._frame_idx = (self._frame_idx + 1) % FRAME_STACK

    # ========================================================================
    # Env trait — delegate to inner PongEnv
    # ========================================================================

    fn reset(mut self) -> ArcadeGameState:
        var state = self.inner.reset()
        # Render initial frame and fill all 4 slots
        self._render_to_buf()
        self._frame_idx = 0
        for _ in range(FRAME_STACK):
            self._push_frame()
        return state

    fn step(
        mut self, action: ArcadeGameAction, verbose: Bool = False
    ) -> Tuple[ArcadeGameState, Scalar[Self.DTYPE], Bool]:
        var result = self.inner.step(action, verbose)
        self._render_to_buf()
        self._push_frame()
        return result

    fn get_state(self) -> ArcadeGameState:
        return self.inner.get_state()

    fn close(mut self):
        self.inner.close()

    fn action_from_index(self, action_idx: Int) -> ArcadeGameAction:
        return self.inner.action_from_index(action_idx)

    fn num_actions(self) -> Int:
        return 3

    fn obs_dim(self) -> Int:
        return PIXEL_OBS_DIM

    fn num_states(self) -> Int:
        return 1

    fn state_to_index(self, state: ArcadeGameState) -> Int:
        return state.index

    # ========================================================================
    # BoxDiscreteActionEnv — pixel observations
    # ========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        """Return 4×84×84 pixel observations as chronological frame stack."""
        comptime FRAME_SIZE = OBS_W * OBS_H
        var obs = List[Scalar[Self.DTYPE]](capacity=PIXEL_OBS_DIM)
        for f in range(FRAME_STACK):
            var read_slot = (self._frame_idx + f) % FRAME_STACK  # oldest first
            var read_base = read_slot * FRAME_SIZE
            for i in range(FRAME_SIZE):
                obs.append(self._frame_stack[read_base + i])
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        var result = self.inner._step_impl(action)
        self._render_to_buf()
        self._push_frame()
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # RenderableEnv — delegate to inner PongEnv
    # ========================================================================

    fn init_renderer(mut self) raises -> Bool:
        return self.inner.init_renderer()

    fn render_frame(mut self) raises -> None:
        self.inner.render_frame()

    fn close_renderer(mut self) raises -> None:
        self.inner.close_renderer()

    fn is_renderer_open(self) -> Bool:
        return self.inner.is_renderer_open()

    fn check_renderer_quit(mut self) -> Bool:
        return self.inner.check_renderer_quit()

    fn renderer_delay(self, ms: Int) -> None:
        self.inner.renderer_delay(ms)

    fn renderer_is_paused(self) -> Bool:
        return False

    fn renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU: Pixel observation step/reset kernels
    # ========================================================================

    comptime TPB = 64  # Fewer threads per block — pixel rendering is heavy

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
    ) raises:
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var terminated_out = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](terminated_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        fn pixel_step_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs_ptr: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
            ws_ptr: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
            rng_seed: Scalar[DType.uint64],
        ):
            # 1. Run Pong physics (shared with clean obs PongEnv)
            PongEnv[DType.float32].step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones, rng_seed
            )

            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH_SIZE:
                return

            terminated_out[idx] = dones[idx]

            # 2. Render game state to grayscale framebuffer in workspace
            var env_ws = ws_ptr + idx * PIXEL_WS_PER_ENV
            var frame_buf = env_ws.bitcast[UInt8]()

            _render_pong_frame(
                frame_buf,
                Int(states[idx, S_BALL_X]),
                Int(states[idx, S_BALL_Y]),
                Int(states[idx, S_PADDLE_Y]),
                Int(states[idx, S_CPU_PADDLE_Y]),
                Int(states[idx, S_PLAYER_SCORE]),
                Int(states[idx, S_CPU_SCORE]),
            )

            # 3. Resize to 84×84 and push to frame stack, output obs
            var env_obs = obs_ptr + idx * PIXEL_OBS_DIM
            _resize_and_push[
                FRAME_BUF_F32_SIZE, FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE
            ](frame_buf, env_ws, env_obs)

        ctx.enqueue_function[pixel_step_wrapper, pixel_step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs_buf.unsafe_ptr(),
            workspace_ptr,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments. Frame stack is initialized in init_step_workspace_gpu.
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            PongEnv[DType.float32].reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
    ) raises:
        """Reset done environments and clear their frame stacks."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            ws_ptr: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
            rng_seed: Scalar[DType.uint64],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH_SIZE:
                return
            if dones[idx] < Scalar[gpu_dtype](0.5):
                return

            # Reset physics
            PongEnv[DType.float32].selective_reset_kernel[
                BATCH_SIZE, STATE_SIZE
            ](states, dones, Scalar[DType.uint32](rng_seed))

            # Clear frame stack for this env
            var env_ws = ws_ptr + idx * PIXEL_WS_PER_ENV
            # Zero frame stack region
            for i in range(FRAME_STACK_F32_SIZE):
                env_ws[FRAME_BUF_F32_SIZE + i] = 0.0
            # Reset frame index
            env_ws[FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE] = 0.0

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            workspace_ptr,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        """Initialize workspace: zero frame stacks and frame indices."""
        comptime WS_TOTAL = BATCH_SIZE * PIXEL_WS_PER_ENV
        comptime BLOCKS = (WS_TOTAL + 256 - 1) // 256

        var ws_ptr = workspace_buf.unsafe_ptr()

        @always_inline
        fn init_ws_wrapper(
            ws: UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= WS_TOTAL:
                return
            ws[i] = 0.0

        ctx.enqueue_function[init_ws_wrapper, init_ws_wrapper](
            ws_ptr,
            grid_dim=(BLOCKS,),
            block_dim=(256,),
        )

    @staticmethod
    fn update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass
