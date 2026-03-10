"""Native Breakout — CPU+GPU environment for RL training.

Ball + paddle + brick grid. No Atari emulation.
Follows the CartPole/Pong pattern.

State layout (STATE_SIZE = 56):
  [0]  ball_x             \
  [1]  ball_y              |
  [2]  ball_vx             | CLEAN_OBS_DIM = 7
  [3]  ball_vy             |
  [4]  paddle_x            |
  [5]  bricks_remaining    |
  [6]  lives              /
  [7..48]  brick_alive (6 rows × 7 cols = 42 floats, 0.0 or 1.0)
  [49] score
  [50] ball_stuck  (1.0 = ball on paddle before serve)
  [51] step_count
  [52..55] reserved

Actions: 0=NOOP, 1=FIRE (release ball), 2=LEFT, 3=RIGHT
"""

from std.random import random_float64
from std.memory import alloc
from core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from render import Renderer2D, SDL_Color, Vec2, Camera, black, white
from nn.gpu import random_range, xorshift32
from layout import LayoutTensor, Layout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ..core.gpu_env import AtariGameState, AtariGameAction, gpu_dtype
from ..core.colors import SCREEN_W, SCREEN_H

# ============================================================================
# Physics Constants
# ============================================================================

comptime PADDLE_WIDTH: Int = 16
comptime PADDLE_HEIGHT: Int = 4
comptime PADDLE_Y: Int = 195  # Near bottom
comptime PADDLE_SPEED: Float64 = 4.0
comptime BALL_SIZE: Int = 2
comptime BALL_SPEED_X: Float64 = 1.5
comptime BALL_SPEED_Y: Float64 = -2.0  # Upward
comptime MAX_BALL_VX: Float64 = 3.0
comptime MAX_BALL_VY: Float64 = 3.0

# Brick grid
comptime BRICK_ROWS: Int = 6
comptime BRICK_COLS: Int = 7
comptime BRICK_WIDTH: Int = 20
comptime BRICK_HEIGHT: Int = 8
comptime BRICK_TOP: Int = 30  # Y start of brick area
comptime BRICK_LEFT: Int = 10  # X start
comptime BRICK_GAP: Int = 2

comptime TOTAL_BRICKS: Int = BRICK_ROWS * BRICK_COLS  # 42
comptime INITIAL_LIVES: Int = 5
comptime BREAKOUT_MAX_STEPS: Int = 10000

# Score per brick row (top rows = more points, like original)
comptime ROW_SCORES: InlineArray[Int, 6] = [7, 7, 4, 4, 1, 1]

# State slot indices
comptime S_BALL_X: Int = 0
comptime S_BALL_Y: Int = 1
comptime S_BALL_VX: Int = 2
comptime S_BALL_VY: Int = 3
comptime S_PADDLE_X: Int = 4
comptime S_BRICKS_REM: Int = 5
comptime S_LIVES: Int = 6
comptime S_BRICKS_START: Int = 7  # 42 brick slots
comptime S_SCORE: Int = 49
comptime S_BALL_STUCK: Int = 50
comptime S_STEP_COUNT: Int = 51

# ============================================================================
# BreakoutEnv
# ============================================================================


struct BreakoutEnv[DTYPE: DType where DTYPE.is_floating_point()](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Native Breakout environment — CPU+GPU dual path."""

    comptime dtype = Self.DTYPE
    comptime StateType = AtariGameState
    comptime ActionType = AtariGameAction

    comptime STATE_SIZE: Int = 56
    comptime OBS_DIM: Int = 7
    comptime NUM_ACTIONS: Int = 4  # NOOP, FIRE, LEFT, RIGHT
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    var state: InlineArray[Scalar[Self.dtype], 56]
    var done: Bool
    var _rng_counter: UInt32

    var _renderer: UnsafePointer[Renderer2D, MutAnyOrigin]
    var _renderer_initialized: Bool

    fn __init__(out self):
        self.state = InlineArray[Scalar[Self.dtype], 56](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._rng_counter = 42
        self._renderer = UnsafePointer[Renderer2D, MutAnyOrigin]()
        self._renderer_initialized = False

    # ========================================================================
    # CPU reset + step
    # ========================================================================

    fn reset(mut self) -> AtariGameState:
        self._rng_counter += 1
        # Ball stuck on paddle
        self.state[S_PADDLE_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BALL_Y] = Scalar[Self.dtype](PADDLE_Y - BALL_SIZE - 1)
        self.state[S_BALL_VX] = 0.0
        self.state[S_BALL_VY] = 0.0
        self.state[S_BALL_STUCK] = 1.0
        self.state[S_LIVES] = Scalar[Self.dtype](INITIAL_LIVES)
        self.state[S_SCORE] = 0.0
        self.state[S_STEP_COUNT] = 0.0
        self.state[S_BRICKS_REM] = Scalar[Self.dtype](TOTAL_BRICKS)
        # All bricks alive
        for b in range(TOTAL_BRICKS):
            self.state[S_BRICKS_START + b] = 1.0
        self.done = False
        return AtariGameState(index=0)

    fn step(
        mut self, action: AtariGameAction, verbose: Bool = False
    ) -> Tuple[AtariGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            AtariGameState(index=Int(self.state[S_STEP_COUNT])),
            result[0],
            result[1],
        )

    fn _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        var paddle_x = self.state[S_PADDLE_X]

        # Move paddle
        if action == 2:  # LEFT
            paddle_x -= Scalar[Self.dtype](PADDLE_SPEED)
            if paddle_x < Scalar[Self.dtype](PADDLE_WIDTH // 2):
                paddle_x = Scalar[Self.dtype](PADDLE_WIDTH // 2)
        elif action == 3:  # RIGHT
            paddle_x += Scalar[Self.dtype](PADDLE_SPEED)
            if paddle_x > Scalar[Self.dtype](SCREEN_W - PADDLE_WIDTH // 2):
                paddle_x = Scalar[Self.dtype](SCREEN_W - PADDLE_WIDTH // 2)
        self.state[S_PADDLE_X] = paddle_x

        # Fire (release ball)
        if action == 1 and self.state[S_BALL_STUCK] > 0.5:
            self.state[S_BALL_STUCK] = 0.0
            self.state[S_BALL_VX] = Scalar[Self.dtype](
                (random_float64() - 0.5) * 2.0 * BALL_SPEED_X
            )
            self.state[S_BALL_VY] = Scalar[Self.dtype](BALL_SPEED_Y)

        var reward = Scalar[Self.dtype](0.0)

        if self.state[S_BALL_STUCK] > 0.5:
            # Ball follows paddle
            self.state[S_BALL_X] = paddle_x
            self.state[S_BALL_Y] = Scalar[Self.dtype](
                PADDLE_Y - BALL_SIZE - 1
            )
            self.state[S_STEP_COUNT] += 1.0
            return (reward, self.done)

        # Move ball
        self.state[S_BALL_X] += self.state[S_BALL_VX]
        self.state[S_BALL_Y] += self.state[S_BALL_VY]

        var bx = self.state[S_BALL_X]
        var by = self.state[S_BALL_Y]

        # Wall bounce (left/right)
        if bx < 0:
            self.state[S_BALL_X] = 0.0
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]
        elif bx > Scalar[Self.dtype](SCREEN_W):
            self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W)
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]

        # Top bounce
        if by < 0:
            self.state[S_BALL_Y] = 0.0
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]

        # Bottom: lose life
        if by > Scalar[Self.dtype](SCREEN_H):
            self.state[S_LIVES] -= 1.0
            if self.state[S_LIVES] <= 0:
                self.done = True
                self.state[S_STEP_COUNT] += 1.0
                return (reward, self.done)
            # Reset ball on paddle
            self.state[S_BALL_STUCK] = 1.0
            self.state[S_BALL_X] = paddle_x
            self.state[S_BALL_Y] = Scalar[Self.dtype](
                PADDLE_Y - BALL_SIZE - 1
            )
            self.state[S_BALL_VX] = 0.0
            self.state[S_BALL_VY] = 0.0

        # Paddle collision
        bx = self.state[S_BALL_X]
        by = self.state[S_BALL_Y]
        if (
            by >= Scalar[Self.dtype](PADDLE_Y - PADDLE_HEIGHT)
            and by <= Scalar[Self.dtype](PADDLE_Y)
            and bx >= paddle_x - Scalar[Self.dtype](PADDLE_WIDTH // 2)
            and bx <= paddle_x + Scalar[Self.dtype](PADDLE_WIDTH // 2)
            and self.state[S_BALL_VY] > 0
        ):
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]
            var hit_pos = (bx - paddle_x) / Scalar[Self.dtype](
                PADDLE_WIDTH // 2
            )
            self.state[S_BALL_VX] = hit_pos * Scalar[Self.dtype](MAX_BALL_VX)
            self.state[S_BALL_Y] = Scalar[Self.dtype](
                PADDLE_Y - PADDLE_HEIGHT - 1
            )

        # Brick collision
        bx = self.state[S_BALL_X]
        by = self.state[S_BALL_Y]
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                var idx = row * BRICK_COLS + col
                if self.state[S_BRICKS_START + idx] < 0.5:
                    continue
                var brick_x = Scalar[Self.dtype](
                    BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                )
                var brick_y = Scalar[Self.dtype](
                    BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                )
                if (
                    bx >= brick_x
                    and bx <= brick_x + Scalar[Self.dtype](BRICK_WIDTH)
                    and by >= brick_y
                    and by <= brick_y + Scalar[Self.dtype](BRICK_HEIGHT)
                ):
                    self.state[S_BRICKS_START + idx] = 0.0
                    self.state[S_BALL_VY] = -self.state[S_BALL_VY]
                    self.state[S_BRICKS_REM] -= 1.0
                    var scores = materialize[ROW_SCORES]()
                    var points = scores[row]
                    self.state[S_SCORE] += Scalar[Self.dtype](points)
                    reward += Scalar[Self.dtype](points)
                    break  # Only one brick per step

        # Check win
        if self.state[S_BRICKS_REM] <= 0:
            self.done = True

        self.state[S_STEP_COUNT] += 1.0
        if Int(self.state[S_STEP_COUNT]) >= BREAKOUT_MAX_STEPS:
            self.done = True

        return (reward, self.done)

    # ========================================================================
    # Trait methods
    # ========================================================================

    fn get_state(self) -> AtariGameState:
        return AtariGameState(index=Int(self.state[S_STEP_COUNT]))

    fn close(mut self):
        if self._renderer_initialized:
            self._renderer[].close()
            self._renderer.free()
            self._renderer_initialized = False

    fn action_from_index(self, action_idx: Int) -> AtariGameAction:
        return AtariGameAction(value=action_idx)

    fn num_actions(self) -> Int:
        return 4

    fn obs_dim(self) -> Int:
        return 7

    fn num_states(self) -> Int:
        return 1

    fn state_to_index(self, state: AtariGameState) -> Int:
        return state.index

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=7)
        obs.append(self.state[S_BALL_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BALL_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_BALL_VX] / Scalar[Self.dtype](MAX_BALL_VX))
        obs.append(self.state[S_BALL_VY] / Scalar[Self.dtype](MAX_BALL_VY))
        obs.append(self.state[S_PADDLE_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(
            self.state[S_BRICKS_REM] / Scalar[Self.dtype](TOTAL_BRICKS)
        )
        obs.append(self.state[S_LIVES] / Scalar[Self.dtype](INITIAL_LIVES))
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action)
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # RenderableEnv
    # ========================================================================

    fn init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D](1)
        self._renderer.init_pointee_move(Renderer2D())
        self._renderer_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer[])

    fn _render(self, mut renderer: Renderer2D):
        if not renderer.begin_frame():
            return

        var sx = Float64(renderer.screen_width) / Float64(SCREEN_W)
        var sy = Float64(renderer.screen_height) / Float64(SCREEN_H)

        var white_c = SDL_Color(255, 255, 255, 255)
        # Brick row colors
        # Brick row colors
        var c_red = SDL_Color(255, 50, 50, 255)
        var c_orange = SDL_Color(255, 140, 50, 255)
        var c_yellow = SDL_Color(255, 255, 50, 255)
        var c_green = SDL_Color(50, 255, 50, 255)
        var c_aqua = SDL_Color(50, 255, 255, 255)
        var c_blue = SDL_Color(50, 100, 255, 255)

        # Draw bricks
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                var idx = row * BRICK_COLS + col
                if self.state[S_BRICKS_START + idx] < 0.5:
                    continue
                var bx = BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                var by = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                var color = c_red
                if row == 1:
                    color = c_orange
                elif row == 2:
                    color = c_yellow
                elif row == 3:
                    color = c_green
                elif row == 4:
                    color = c_aqua
                elif row == 5:
                    color = c_blue
                renderer.draw_rect(
                    Int(Float64(bx) * sx),
                    Int(Float64(by) * sy),
                    max(1, Int(Float64(BRICK_WIDTH) * sx)),
                    max(1, Int(Float64(BRICK_HEIGHT) * sy)),
                    color,
                )

        # Draw paddle
        var px = Int(self.state[S_PADDLE_X])
        renderer.draw_rect(
            Int(Float64(px - PADDLE_WIDTH // 2) * sx),
            Int(Float64(PADDLE_Y - PADDLE_HEIGHT) * sy),
            max(1, Int(Float64(PADDLE_WIDTH) * sx)),
            max(1, Int(Float64(PADDLE_HEIGHT) * sy)),
            white_c,
        )

        # Draw ball
        var ball_x = Int(self.state[S_BALL_X])
        var ball_y = Int(self.state[S_BALL_Y])
        renderer.draw_rect(
            Int(Float64(ball_x) * sx) - 1,
            Int(Float64(ball_y) * sy) - 1,
            max(2, Int(Float64(BALL_SIZE) * sx)),
            max(2, Int(Float64(BALL_SIZE) * sy)),
            white_c,
        )

        # Info
        var info = List[String]()
        info.append("Score: " + String(Int(self.state[S_SCORE])))
        info.append("Lives: " + String(Int(self.state[S_LIVES])))
        info.append("Bricks: " + String(Int(self.state[S_BRICKS_REM])))
        renderer.draw_info_box(info)

        renderer.flip()

    fn close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    fn is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return not self._renderer[].get_should_quit()

    fn check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].get_should_quit()

    fn renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer[].renderer_delay(ms)

    fn renderer_is_paused(self) -> Bool:
        return False

    fn renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU Inline Kernels
    # ========================================================================

    @staticmethod
    @always_inline
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
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
        rng_seed: Scalar[DType.uint64],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var action = Int(actions[i])

        # Load state
        var bx = states[i, S_BALL_X]
        var by = states[i, S_BALL_Y]
        var bvx = states[i, S_BALL_VX]
        var bvy = states[i, S_BALL_VY]
        var paddle_x = states[i, S_PADDLE_X]
        var bricks_rem = states[i, S_BRICKS_REM]
        var lives = states[i, S_LIVES]
        var score = states[i, S_SCORE]
        var stuck = states[i, S_BALL_STUCK]
        var steps = states[i, S_STEP_COUNT]
        var reward = Scalar[gpu_dtype](0.0)

        # Move paddle
        if action == 2:  # LEFT
            paddle_x = paddle_x - Scalar[gpu_dtype](PADDLE_SPEED)
            if paddle_x < Scalar[gpu_dtype](PADDLE_WIDTH // 2):
                paddle_x = Scalar[gpu_dtype](PADDLE_WIDTH // 2)
        elif action == 3:  # RIGHT
            paddle_x = paddle_x + Scalar[gpu_dtype](PADDLE_SPEED)
            if paddle_x > Scalar[gpu_dtype](SCREEN_W - PADDLE_WIDTH // 2):
                paddle_x = Scalar[gpu_dtype](SCREEN_W - PADDLE_WIDTH // 2)

        # Fire
        if action == 1 and stuck > Scalar[gpu_dtype](0.5):
            stuck = 0.0
            var rng = xorshift32(
                Scalar[DType.uint32](
                    UInt32(i) * 2654435761 + UInt32(rng_seed)
                )
            )
            var vx_result = random_range[gpu_dtype](
                rng, Scalar[gpu_dtype](-BALL_SPEED_X), Scalar[gpu_dtype](BALL_SPEED_X)
            )
            bvx = vx_result[0]
            bvy = Scalar[gpu_dtype](BALL_SPEED_Y)

        var is_done = False

        if stuck > Scalar[gpu_dtype](0.5):
            bx = paddle_x
            by = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        else:
            # Move ball
            bx = bx + bvx
            by = by + bvy

            # Wall bounce
            if bx < 0:
                bx = Scalar[gpu_dtype](0)
                bvx = -bvx
            elif bx > Scalar[gpu_dtype](SCREEN_W):
                bx = Scalar[gpu_dtype](SCREEN_W)
                bvx = -bvx

            # Top bounce
            if by < 0:
                by = Scalar[gpu_dtype](0)
                bvy = -bvy

            # Bottom: lose life
            if by > Scalar[gpu_dtype](SCREEN_H):
                lives = lives - 1.0
                if lives <= 0:
                    is_done = True
                else:
                    stuck = 1.0
                    bx = paddle_x
                    by = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
                    bvx = 0.0
                    bvy = 0.0

            # Paddle collision
            if (
                by >= Scalar[gpu_dtype](PADDLE_Y - PADDLE_HEIGHT)
                and by <= Scalar[gpu_dtype](PADDLE_Y)
                and bx >= paddle_x - Scalar[gpu_dtype](PADDLE_WIDTH // 2)
                and bx <= paddle_x + Scalar[gpu_dtype](PADDLE_WIDTH // 2)
                and bvy > 0
            ):
                bvy = -bvy
                var hit_pos = (bx - paddle_x) / Scalar[gpu_dtype](
                    PADDLE_WIDTH // 2
                )
                bvx = hit_pos * Scalar[gpu_dtype](MAX_BALL_VX)
                by = Scalar[gpu_dtype](PADDLE_Y - PADDLE_HEIGHT - 1)

            # Brick collision (check all bricks)
            for row in range(BRICK_ROWS):
                for col in range(BRICK_COLS):
                    var idx = row * BRICK_COLS + col
                    if states[i, S_BRICKS_START + idx] < Scalar[gpu_dtype](0.5):
                        continue
                    var brick_x = Scalar[gpu_dtype](
                        BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                    )
                    var brick_y = Scalar[gpu_dtype](
                        BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                    )
                    if (
                        bx >= brick_x
                        and bx <= brick_x + Scalar[gpu_dtype](BRICK_WIDTH)
                        and by >= brick_y
                        and by <= brick_y + Scalar[gpu_dtype](BRICK_HEIGHT)
                    ):
                        states[i, S_BRICKS_START + idx] = 0.0
                        bvy = -bvy
                        bricks_rem = bricks_rem - 1.0
                        # Row-based scoring (simplified: top=7, mid=4, bot=1)
                        var pts = Scalar[gpu_dtype](1.0)
                        if row < 2:
                            pts = 7.0
                        elif row < 4:
                            pts = 4.0
                        score = score + pts
                        reward = reward + pts
                        break  # One brick per step

        # Check win
        if bricks_rem <= 0:
            is_done = True

        steps = steps + 1.0
        if Int(steps) >= BREAKOUT_MAX_STEPS:
            is_done = True

        # Write state back
        states[i, S_BALL_X] = bx
        states[i, S_BALL_Y] = by
        states[i, S_BALL_VX] = bvx
        states[i, S_BALL_VY] = bvy
        states[i, S_PADDLE_X] = paddle_x
        states[i, S_BRICKS_REM] = bricks_rem
        states[i, S_LIVES] = lives
        states[i, S_SCORE] = score
        states[i, S_BALL_STUCK] = stuck
        states[i, S_STEP_COUNT] = steps

        rewards[i] = reward
        dones[i] = Scalar[gpu_dtype](is_done)

    @staticmethod
    @always_inline
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        state[i, S_BALL_VX] = 0.0
        state[i, S_BALL_VY] = 0.0
        state[i, S_PADDLE_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BRICKS_REM] = Scalar[gpu_dtype](TOTAL_BRICKS)
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_SCORE] = 0.0
        state[i, S_BALL_STUCK] = 1.0
        state[i, S_STEP_COUNT] = 0.0
        # All bricks alive
        for b in range(TOTAL_BRICKS):
            state[i, S_BRICKS_START + b] = 1.0
        # Zero reserved
        for r in range(52, 56):
            state[i, r] = 0.0

    @staticmethod
    @always_inline
    fn selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
        dones: LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE),
            MutAnyOrigin,
        ],
        rng_seed: Scalar[DType.uint32],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        state[i, S_BALL_VX] = 0.0
        state[i, S_BALL_VY] = 0.0
        state[i, S_PADDLE_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BRICKS_REM] = Scalar[gpu_dtype](TOTAL_BRICKS)
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_SCORE] = 0.0
        state[i, S_BALL_STUCK] = 1.0
        state[i, S_STEP_COUNT] = 0.0
        for b in range(TOTAL_BRICKS):
            state[i, S_BRICKS_START + b] = 1.0
        for r in range(52, 56):
            state[i, r] = 0.0
        dones[i] = 0.0

    # ========================================================================
    # GPU Launcher Methods
    # ========================================================================

    comptime TPB = 256

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
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        fn step_wrapper(
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
            obs: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
            rng_seed: Scalar[DType.uint64],
        ):
            Self.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones, rng_seed
            )
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]
                obs[idx, 0] = states[idx, S_BALL_X] / Scalar[gpu_dtype](SCREEN_W)
                obs[idx, 1] = states[idx, S_BALL_Y] / Scalar[gpu_dtype](SCREEN_H)
                obs[idx, 2] = states[idx, S_BALL_VX] / Scalar[gpu_dtype](MAX_BALL_VX)
                obs[idx, 3] = states[idx, S_BALL_VY] / Scalar[gpu_dtype](MAX_BALL_VY)
                obs[idx, 4] = states[idx, S_PADDLE_X] / Scalar[gpu_dtype](SCREEN_W)
                obs[idx, 5] = states[idx, S_BRICKS_REM] / Scalar[gpu_dtype](TOTAL_BRICKS)
                obs[idx, 6] = states[idx, S_LIVES] / Scalar[gpu_dtype](INITIAL_LIVES)

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states, actions, rewards, dones, terminated_out, obs, seed,
            grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
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
            Self.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
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
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        fn sel_reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            rng_seed: Scalar[DType.uint64],
        ):
            Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones, Scalar[DType.uint32](rng_seed)
            )

        ctx.enqueue_function[sel_reset_wrapper, sel_reset_wrapper](
            states, dones, seed, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    fn init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        pass

    @staticmethod
    fn update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass
