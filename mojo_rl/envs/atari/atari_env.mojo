"""AtariEnv — BoxDiscreteActionEnv wrapper for the Atari 2600 emulator.

Wraps AtariEnvironment with the standard RL trait interface, providing
RAM or pixel observations for use with DQN and other discrete-action agents.

Usage:
    from mojo_rl.envs.atari import AtariEnv, load_rom
    from mojo_rl.envs.atari.games import PongDef

    var rom = load_rom("pong.bin")
    var env = AtariEnv[PongDef](rom.data, rom.size)  # RAM mode (128D obs)
    var obs = env.reset_obs_list()
    var result = env.step_obs(0)

    # Pixel mode (4×84×84 = 28224D obs)
    var env_px = AtariEnv[PongDef, 1](rom.data, rom.size)
"""

from std.memory import alloc, memset
from mojo_rl.core import State, Action, BoxDiscreteActionEnv
from .environment import AtariEnvironment, GameDef
from .cpu6502 import run_frame, run_frame_with_video
from .riot import set_action
from .flags import (
    RAM_SIZE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    OBS_WIDTH,
    OBS_HEIGHT,
    ACTION_NOOP,
)


# Pixel buffer sizes
comptime FRAME_BGRA_SIZE: Int = FRAME_WIDTH * FRAME_HEIGHT * 4  # 160*210*4
comptime GRAY_FRAME_SIZE: Int = FRAME_WIDTH * FRAME_HEIGHT  # 160*210
comptime OBS_FRAME_SIZE: Int = OBS_WIDTH * OBS_HEIGHT  # 84*84
comptime FRAME_STACK_SIZE: Int = 4 * OBS_FRAME_SIZE  # 4*84*84 = 28224


# ============================================================================
# State and Action types
# ============================================================================


@fieldwise_init
struct AtariEnvState(Copyable, ImplicitlyCopyable, Movable, State):
    """Minimal state wrapper — frame number as index."""

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit take: Self):
        self.index = take.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct AtariAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Thin wrapper around action index."""

    var action_idx: Int

    def __init__(out self, *, copy: Self):
        self.action_idx = copy.action_idx

    def __init__(out self, *, deinit take: Self):
        self.action_idx = take.action_idx


# ============================================================================
# Resize helper
# ============================================================================


def _resize_160x210_to_84x84(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Resize 160×210 grayscale to 84×84 using area (box-filter) interpolation.

    Each output pixel averages the source pixels in its corresponding
    rectangle. Scale factors: x = 160/84 ≈ 1.905, y = 210/84 = 2.5.
    """
    for oy in range(OBS_HEIGHT):
        # Source y range for this output row
        var sy0 = (oy * FRAME_HEIGHT) // OBS_HEIGHT
        var sy1 = ((oy + 1) * FRAME_HEIGHT) // OBS_HEIGHT
        if sy1 <= sy0:
            sy1 = sy0 + 1

        for ox in range(OBS_WIDTH):
            # Source x range for this output column
            var sx0 = (ox * FRAME_WIDTH) // OBS_WIDTH
            var sx1 = ((ox + 1) * FRAME_WIDTH) // OBS_WIDTH
            if sx1 <= sx0:
                sx1 = sx0 + 1

            # Average source pixels in the rectangle
            var total: Int = 0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Int(src[sy * FRAME_WIDTH + sx])
                    count += 1

            dst[oy * OBS_WIDTH + ox] = UInt8(total // count)


# ============================================================================
# AtariEnv
# ============================================================================


struct AtariEnv[
    GAME: GameDef,
    OBS_MODE: Int = 0,
    DTYPE: DType = DType.float32,
](BoxDiscreteActionEnv):
    """Atari environment conforming to BoxDiscreteActionEnv.

    Parameters:
        GAME: Game-specific RAM extraction (PongDef, BreakoutDef, etc.).
        OBS_MODE: 0 = RAM (128 floats), 1 = pixels (4×84×84 = 28224 floats).
        DTYPE: Observation dtype (default float32).
    """

    comptime dtype = Self.DTYPE
    comptime StateType = AtariEnvState
    comptime ActionType = AtariAction

    var env: AtariEnvironment
    var episode_reward: Float64
    var done: Bool
    var _steps: Int

    # Pixel-mode buffers (allocated only when OBS_MODE==1)
    var frame_stack: UnsafePointer[UInt8, MutAnyOrigin]  # 4 * 84 * 84
    var frame_idx: Int  # ring buffer index
    var raw_frame_a: UnsafePointer[UInt8, MutAnyOrigin]  # 160*210*4 BGRA
    var raw_frame_b: UnsafePointer[UInt8, MutAnyOrigin]  # 160*210*4 BGRA
    var gray_buf: UnsafePointer[UInt8, MutAnyOrigin]  # 160*210 grayscale

    def __init__(
        out self,
        rom: UnsafePointer[UInt8, MutAnyOrigin],
        rom_size: Int,
        frame_skip: Int = 4,
        max_frames: Int = 108000,
    ):
        """Create an AtariEnv.

        For pixel mode (OBS_MODE=1), frame_skip is managed internally
        (env.frame_skip is set to 1, skip loop is in step_obs).

        Args:
            rom: ROM data pointer.
            rom_size: ROM size in bytes.
            frame_skip: Number of frames to repeat each action (default 4).
            max_frames: Max frames per episode (default 108000).
        """
        comptime if Self.OBS_MODE == 1:
            # Pixel mode: we drive frame skip manually
            self.env = AtariEnvironment(
                rom, rom_size, frame_skip=1, max_frames=max_frames
            )
        else:
            self.env = AtariEnvironment(
                rom, rom_size, frame_skip=frame_skip, max_frames=max_frames
            )

        self.episode_reward = 0.0
        self.done = False
        self._steps = 0

        # Pixel mode buffers
        comptime if Self.OBS_MODE == 1:
            self.frame_stack = alloc[UInt8](FRAME_STACK_SIZE)
            self.raw_frame_a = alloc[UInt8](FRAME_BGRA_SIZE)
            self.raw_frame_b = alloc[UInt8](FRAME_BGRA_SIZE)
            self.gray_buf = alloc[UInt8](GRAY_FRAME_SIZE)
            self.frame_idx = 0
            memset(self.frame_stack, 0, FRAME_STACK_SIZE)
        else:
            self.frame_stack = UnsafePointer[UInt8, MutAnyOrigin]()
            self.raw_frame_a = UnsafePointer[UInt8, MutAnyOrigin]()
            self.raw_frame_b = UnsafePointer[UInt8, MutAnyOrigin]()
            self.gray_buf = UnsafePointer[UInt8, MutAnyOrigin]()
            self.frame_idx = 0

    def __init__(out self, *, deinit take: Self):
        self.env = take.env^
        self.episode_reward = take.episode_reward
        self.done = take.done
        self._steps = take._steps
        self.frame_stack = take.frame_stack
        self.frame_idx = take.frame_idx
        self.raw_frame_a = take.raw_frame_a
        self.raw_frame_b = take.raw_frame_b
        self.gray_buf = take.gray_buf

    # ========================================================================
    # Pixel-mode helpers
    # ========================================================================

    def _bgra_to_gray_maxpool(self):
        """Max-pool raw_frame_a and raw_frame_b element-wise, convert to grayscale.

        Handles Atari sprite flickering by taking the max of 2 consecutive frames.
        Result stored in gray_buf (160×210 grayscale).
        """
        for i in range(GRAY_FRAME_SIZE):
            var offset = i * 4
            # Max of two frames per channel
            var b = max(
                Int(self.raw_frame_a[offset + 0]),
                Int(self.raw_frame_b[offset + 0]),
            )
            var g = max(
                Int(self.raw_frame_a[offset + 1]),
                Int(self.raw_frame_b[offset + 1]),
            )
            var r = max(
                Int(self.raw_frame_a[offset + 2]),
                Int(self.raw_frame_b[offset + 2]),
            )
            # Luminance: Y = (77*R + 150*G + 29*B) >> 8
            self.gray_buf[i] = UInt8((77 * r + 150 * g + 29 * b) >> 8)

    def _push_frame_to_stack(mut self):
        """Resize gray_buf (160×210) to 84×84 and push into frame_stack ring buffer.
        """
        var slot_offset = self.frame_idx * OBS_FRAME_SIZE
        _resize_160x210_to_84x84(
            self.gray_buf,
            self.frame_stack + slot_offset,
        )
        self.frame_idx = (self.frame_idx + 1) % 4

    def _render_current_frame(mut self):
        """Render the current emulator state into raw_frame_a using run_frame_with_video.

        Runs one NOOP frame with video output to capture the display.
        """
        run_frame_with_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_a
        )

    # ========================================================================
    # Env trait (base)
    # ========================================================================

    def reset(mut self) -> AtariEnvState:
        """Reset the environment."""
        self.env.reset()
        self.episode_reward = 0.0
        self.done = False
        self._steps = 0

        comptime if Self.OBS_MODE == 1:
            # Render initial frame into all 4 stack slots
            run_frame_with_video(
                self.env.state,
                self.env.rom,
                self.env.rom_size,
                self.raw_frame_a,
            )
            # Copy frame_a to frame_b for maxpool (both identical after reset)
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b[i] = self.raw_frame_a[i]
            self._bgra_to_gray_maxpool()
            # Fill all 4 slots with the same initial frame
            self.frame_idx = 0
            for _ in range(4):
                self._push_frame_to_stack()

        return AtariEnvState(index=0)

    def step(
        mut self, action: AtariAction, verbose: Bool = False
    ) -> Tuple[AtariEnvState, Scalar[Self.DTYPE], Bool]:
        """Take action and return (state, reward, done)."""
        var result = self.step_obs(action.action_idx)
        return (
            AtariEnvState(index=self._steps),
            result[1],
            result[2],
        )

    def get_state(self) -> AtariEnvState:
        return AtariEnvState(index=self._steps)

    def close(mut self):
        """Free pixel-mode buffers."""
        comptime if Self.OBS_MODE == 1:
            if self.frame_stack:
                self.frame_stack.free()
                self.frame_stack = UnsafePointer[UInt8, MutAnyOrigin]()
            if self.raw_frame_a:
                self.raw_frame_a.free()
                self.raw_frame_a = UnsafePointer[UInt8, MutAnyOrigin]()
            if self.raw_frame_b:
                self.raw_frame_b.free()
                self.raw_frame_b = UnsafePointer[UInt8, MutAnyOrigin]()
            if self.gray_buf:
                self.gray_buf.free()
                self.gray_buf = UnsafePointer[UInt8, MutAnyOrigin]()

    # ========================================================================
    # ContinuousStateEnv trait
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        """Return current observation as a list of floats.

        RAM mode: 128 floats in [0, 1].
        Pixel mode: 28224 floats in [0, 1] (4 stacked 84×84 grayscale frames).
        """
        comptime if Self.OBS_MODE == 1:
            # Read 4 stacked frames in chronological order from ring buffer
            var obs = List[Scalar[Self.DTYPE]](capacity=FRAME_STACK_SIZE)
            for i in range(4):
                var slot = (self.frame_idx + i) % 4  # oldest first
                var offset = slot * OBS_FRAME_SIZE
                for j in range(OBS_FRAME_SIZE):
                    obs.append(
                        Scalar[Self.DTYPE](self.frame_stack[offset + j]) / 255.0
                    )
            return obs^
        else:
            # RAM mode: 128 bytes normalized to [0, 1]
            var ram = self.env.get_ram()
            var obs = List[Scalar[Self.DTYPE]](capacity=RAM_SIZE)
            for i in range(RAM_SIZE):
                obs.append(Scalar[Self.DTYPE](ram[i]) / 255.0)
            return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        """Reset environment and return initial observation."""
        _ = self.reset()
        return self.get_obs_list()

    def obs_dim(self) -> Int:
        """Return observation dimension."""
        comptime if Self.OBS_MODE == 1:
            return FRAME_STACK_SIZE  # 4 * 84 * 84 = 28224
        else:
            return RAM_SIZE  # 128

    # ========================================================================
    # DiscreteActionEnv trait
    # ========================================================================

    def action_from_index(self, action_idx: Int) -> AtariAction:
        return AtariAction(action_idx=action_idx)

    def num_actions(self) -> Int:
        return Self.GAME.NUM_ACTIONS

    # ========================================================================
    # BoxDiscreteActionEnv trait — step_obs
    # ========================================================================

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """Take action and return (obs, reward, done).

        RAM mode: delegates to env.step_with_game, reads RAM.
        Pixel mode: drives frame_skip sub-frames manually, rendering
        the last 2 for max-pooling (handles sprite flickering).
        """
        self._steps += 1

        comptime if Self.OBS_MODE == 1:
            return self._step_obs_pixel(action)
        else:
            return self._step_obs_ram(action)

    def _step_obs_ram(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """RAM mode step: use step_with_game, read 128 RAM bytes."""
        var reward = self.env.step_with_game[Self.GAME](action)
        self.done = self.env.is_terminal()
        self.episode_reward += Float64(reward)
        var obs = self.get_obs_list()
        return (obs^, Scalar[Self.DTYPE](reward), self.done)

    def _step_obs_pixel(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """Pixel mode step: manual frame-skip with per-scanline rendering.

        Frame skip = 4 (default):
          - Frames 0,1: set_action + run_frame (no render, fast)
          - Frame 2: set_action + run_frame_with_video → raw_frame_a
          - Frame 3: set_action + run_frame_with_video → raw_frame_b
        Then max-pool a/b → grayscale → resize → push to stack.
        """
        var ale_action = Self.GAME.map_action(action)
        var prev_score = Int(self.env.state.score)

        # We use a fixed frame_skip of 4 for pixel mode
        comptime PIXEL_FRAME_SKIP: Int = 4

        # Frames 0 .. skip-3: run without rendering
        for _ in range(PIXEL_FRAME_SKIP - 2):
            set_action(self.env.state, ale_action)
            run_frame(self.env.state, self.env.rom, self.env.rom_size)

        # Frame skip-2: render into raw_frame_a
        set_action(self.env.state, ale_action)
        run_frame_with_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_a
        )

        # Frame skip-1: render into raw_frame_b
        set_action(self.env.state, ale_action)
        run_frame_with_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_b
        )

        # Extract RL signals from RAM (same as step_with_game)
        var new_score = Self.GAME.get_score(self.env.state.ram)
        var reward = new_score - prev_score
        self.env.state.score = Int32(new_score)
        self.env.state.reward = Int32(reward)
        self.env.state.lives = UInt8(Self.GAME.get_lives(self.env.state.ram))
        self.env.state.terminal = Self.GAME.is_terminal(self.env.state.ram)

        # Check max frames truncation
        if (
            self.env.max_frames > 0
            and Int(self.env.state.frame_number) >= self.env.max_frames
        ):
            self.env.state.terminal = True

        self.done = self.env.is_terminal()
        self.episode_reward += Float64(reward)

        # Max-pool → grayscale → resize → push to frame stack
        self._bgra_to_gray_maxpool()
        self._push_frame_to_stack()

        var obs = self.get_obs_list()
        return (obs^, Scalar[Self.DTYPE](reward), self.done)
