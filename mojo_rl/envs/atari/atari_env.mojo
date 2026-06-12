"""AtariEnv — BoxDiscreteActionEnv wrapper for the Atari 2600 emulator.

Wraps AtariEnvironment with the standard RL trait interface, providing
RAM or pixel observations for use with DQN and other discrete-action agents.

The game is a RUNTIME parameter (`AtariGame` registry value): one compiled
binary plays every registered game. Observations are game-agnostic — RAM
(128 floats) or pixels (4×84×84 frame stack) — so the same env type works
for all of them.

Usage:
    from mojo_rl.envs.atari import AtariEnv, AtariGame

    # RAM mode (128D obs); loads roms/pong.bin
    var env = AtariEnv(AtariGame.PONG)
    var obs = env.reset_obs_list()
    var result = env.step_obs(0)

    # Pixel mode (4×84×84 = 28224D obs)
    var env_px = AtariEnv[1](AtariGame.MS_PACMAN)

    # Or with an explicitly loaded ROM:
    var rom = load_rom("roms/pong.bin")
    var env2 = AtariEnv(AtariGame.PONG, rom.data.value(), rom.size)
"""

from std.memory import alloc, memset
from mojo_rl.core import State, Action, BoxDiscreteActionEnv
from .environment import AtariEnvironment, load_rom
from .games.registry import AtariGame, game_signals
from .cpu6502 import run_frame, run_frame_video
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

    SIMD-restructured but BIT-EXACT vs the original scalar double loop:
    per output row, the 2-3 source rows are first summed column-wise into
    a u16 row buffer (vectorized — this is where the 33,600 reads live),
    then the cheap horizontal pass sums 1-2 columns per output pixel and
    performs the SAME single `total // count` integer division (max total
    3·2·255 = 1530 fits u16). NOT a separable two-pass resize — that would
    divide twice and round differently."""
    comptime W = 16
    comptime assert FRAME_WIDTH % W == 0, "row width must be SIMD-divisible"

    var vsum = InlineArray[UInt16, FRAME_WIDTH](fill=0)
    for oy in range(OBS_HEIGHT):
        # Source y range for this output row
        var sy0 = (oy * FRAME_HEIGHT) // OBS_HEIGHT
        var sy1 = ((oy + 1) * FRAME_HEIGHT) // OBS_HEIGHT
        if sy1 <= sy0:
            sy1 = sy0 + 1

        # Column-wise sums of rows [sy0, sy1) — vectorized.
        for x in range(0, FRAME_WIDTH, W):
            var acc = SIMD[DType.uint16, W](0)
            for sy in range(sy0, sy1):
                acc += src.load[width=W](sy * FRAME_WIDTH + x).cast[
                    DType.uint16
                ]()
            vsum.unsafe_ptr().store(x, acc)

        var cy = sy1 - sy0
        for ox in range(OBS_WIDTH):
            # Source x range for this output column
            var sx0 = (ox * FRAME_WIDTH) // OBS_WIDTH
            var sx1 = ((ox + 1) * FRAME_WIDTH) // OBS_WIDTH
            if sx1 <= sx0:
                sx1 = sx0 + 1

            var total: Int = 0
            for sx in range(sx0, sx1):
                total += Int(vsum[sx])

            dst[oy * OBS_WIDTH + ox] = UInt8(total // ((sx1 - sx0) * cy))


# ============================================================================
# AtariEnv
# ============================================================================


struct AtariEnv[
    OBS_MODE: Int = 0,
    DTYPE: DType = DType.float32,
](BoxDiscreteActionEnv, Movable):
    """Atari environment conforming to BoxDiscreteActionEnv.

    The game is a runtime field (AtariGame registry) — score/lives/terminal
    extraction and the minimal action set dispatch on it per step, which is
    negligible next to the per-frame emulation cost.

    Parameters:
        OBS_MODE: 0 = RAM (128 floats), 1 = pixels (4×84×84 = 28224 floats).
        DTYPE: Observation dtype (default float32).
    """

    comptime dtype = Self.DTYPE
    comptime StateType = AtariEnvState
    comptime ActionType = AtariAction

    var game: AtariGame
    var env: AtariEnvironment
    var episode_reward: Float64
    var done: Bool
    var _steps: Int

    # Pixel-mode buffers (allocated only when OBS_MODE==1)
    var frame_stack: Optional[UnsafePointer[UInt8, MutAnyOrigin]]  # 4 * 84 * 84
    var frame_idx: Int  # ring buffer index
    var raw_frame_a: Optional[UnsafePointer[UInt8, MutAnyOrigin]]  # 160*210*4 BGRA
    var raw_frame_b: Optional[UnsafePointer[UInt8, MutAnyOrigin]]  # 160*210*4 BGRA
    var gray_buf: Optional[UnsafePointer[UInt8, MutAnyOrigin]]  # 160*210 grayscale

    def __init__(
        out self,
        game: AtariGame,
        frame_skip: Int = 4,
        max_frames: Int = 108000,
    ) raises:
        """Create an AtariEnv for a registry game, loading its ROM from
        `roms/<name>.bin` (relative to the working directory).

        The loaded ROM buffer lives as long as the process (not freed on
        close), matching the explicit-ROM constructor's ownership model.
        """
        var rom_data = load_rom(game.rom_file())
        self = Self(
            game,
            rom_data.data.value(),
            rom_data.size,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )

    def __init__(
        out self,
        game: AtariGame,
        rom: UnsafePointer[UInt8, MutAnyOrigin],
        rom_size: Int,
        frame_skip: Int = 4,
        max_frames: Int = 108000,
    ):
        """Create an AtariEnv from an explicitly loaded ROM.

        For pixel mode (OBS_MODE=1), frame_skip is managed internally
        (env.frame_skip is set to 1, skip loop is in step_obs).

        Args:
            game: Registry game (drives action set + RAM signal extraction).
            rom: ROM data pointer.
            rom_size: ROM size in bytes.
            frame_skip: Number of frames to repeat each action (default 4).
            max_frames: Max frames per episode (default 108000).
        """
        self.game = game
        comptime if Self.OBS_MODE == 1:
            # Pixel mode: we drive frame skip manually
            self.env = AtariEnvironment(
                rom,
                rom_size,
                frame_skip=1,
                max_frames=max_frames,
                mapper=game.mapper(),
                swap_ports=game.swap_ports(),
                paddles=game.uses_paddles(),
            )
        else:
            self.env = AtariEnvironment(
                rom,
                rom_size,
                frame_skip=frame_skip,
                max_frames=max_frames,
                mapper=game.mapper(),
                swap_ports=game.swap_ports(),
                paddles=game.uses_paddles(),
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
            memset(self.frame_stack.value(), 0, FRAME_STACK_SIZE)
        else:
            self.frame_stack = None
            self.raw_frame_a = None
            self.raw_frame_b = None
            self.gray_buf = None
            self.frame_idx = 0

    def __init__(out self, *, deinit take: Self):
        self.game = take.game
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

        SIMD, bit-exact vs the scalar reference: each BGRA pixel is loaded
        as one little-endian u32 lane (B | G<<8 | R<<16 | A<<24), channels
        extracted by shift+mask, per-channel max of the two frames, then
        the integer luma `(77R + 150G + 29B) >> 8` (max 65,280 — fits u32)
        in 16 lanes at a time. 33,600 px % 16 == 0 → no scalar tail.
        """
        comptime W = 16
        comptime assert GRAY_FRAME_SIZE % W == 0, "frame must be SIMD-divisible"
        var a32 = self.raw_frame_a.value().bitcast[UInt32]()
        var b32 = self.raw_frame_b.value().bitcast[UInt32]()
        var gray = self.gray_buf.value()
        for i in range(0, GRAY_FRAME_SIZE, W):
            var va = a32.load[width=W](i)
            var vb = b32.load[width=W](i)
            var b = max(va & 0xFF, vb & 0xFF)
            var g = max((va >> 8) & 0xFF, (vb >> 8) & 0xFF)
            var r = max((va >> 16) & 0xFF, (vb >> 16) & 0xFF)
            # Luminance: Y = (77*R + 150*G + 29*B) >> 8
            gray.store(i, ((77 * r + 150 * g + 29 * b) >> 8).cast[DType.uint8]())

    def _push_frame_to_stack(mut self):
        """Resize gray_buf (160×210) to 84×84 and push into frame_stack ring buffer.
        """
        var slot_offset = self.frame_idx * OBS_FRAME_SIZE
        _resize_160x210_to_84x84(self.gray_buf.value(),
            self.frame_stack.value() + slot_offset,
        )
        self.frame_idx = (self.frame_idx + 1) % 4

    def _render_current_frame(mut self):
        """Render the current emulator state into raw_frame_a using run_frame_video.

        Runs one NOOP frame with video output to capture the display.
        """
        run_frame_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_a.value()
        )

    # ========================================================================
    # Env trait (base)
    # ========================================================================

    def reset(mut self) -> AtariEnvState:
        """Reset the environment."""
        self.env.reset_game(self.game)
        self.episode_reward = 0.0
        self.done = False
        self._steps = 0

        comptime if Self.OBS_MODE == 1:
            # Render initial frame into all 4 stack slots
            run_frame_video(
                self.env.state,
                self.env.rom,
                self.env.rom_size,
                self.raw_frame_a.value(),
            )
            # Copy frame_a to frame_b for maxpool (both identical after reset)
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[i] = self.raw_frame_a.value()[i]
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

    def was_terminated(self) -> Bool:
        """True iff the last step ended via GAME termination (game over),
        not max_frames truncation — the TD bootstrap is dropped only on
        the former. Overrides the base-Env `False` default, which silently
        classified every Atari game-over as a truncation."""
        return self.env.natural_terminal

    def close(mut self):
        """Free pixel-mode buffers."""
        comptime if Self.OBS_MODE == 1:
            if Bool(self.frame_stack):
                self.frame_stack.value().free()
                self.frame_stack = None
            if Bool(self.raw_frame_a):
                self.raw_frame_a.value().free()
                self.raw_frame_a = None
            if Bool(self.raw_frame_b):
                self.raw_frame_b.value().free()
                self.raw_frame_b = None
            if Bool(self.gray_buf):
                self.gray_buf.value().free()
                self.gray_buf = None

    # ========================================================================
    # ContinuousStateEnv trait
    # ========================================================================

    def _write_stack_obs_into(
        self, obs_out: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    ):
        """Write the 4-frame stack (chronological order, oldest first) as
        normalized floats into `obs_out` (FRAME_STACK_SIZE scalars).
        SIMD uint8→float `/255` — bit-exact vs the per-element scalar
        conversion (each uint8 value maps to the identical float)."""
        comptime W = 16
        comptime assert OBS_FRAME_SIZE % W == 0, "obs frame must be SIMD-divisible"
        var fs = self.frame_stack.value()
        var out_off = 0
        for i in range(4):
            var slot = (self.frame_idx + i) % 4  # oldest first
            var src = fs + slot * OBS_FRAME_SIZE
            for j in range(0, OBS_FRAME_SIZE, W):
                obs_out.store(
                    out_off + j,
                    src.load[width=W](j).cast[Self.dtype]() / 255.0,
                )
            out_off += OBS_FRAME_SIZE

    def _write_ram_obs_into(
        self, obs_out: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    ):
        """Write the 128 RAM bytes as normalized floats into `obs_out`."""
        comptime W = 16
        comptime assert RAM_SIZE % W == 0, "RAM size must be SIMD-divisible"
        var ram = self.env.get_ram()
        for i in range(0, RAM_SIZE, W):
            obs_out.store(
                i, ram.unsafe_ptr().load[width=W](i).cast[Self.dtype]() / 255.0
            )

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        """Return current observation as a list of floats.

        RAM mode: 128 floats in [0, 1].
        Pixel mode: 28224 floats in [0, 1] (4 stacked 84×84 grayscale frames).
        """
        comptime if Self.OBS_MODE == 1:
            var obs = List[Scalar[Self.DTYPE]](
                length=FRAME_STACK_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_stack_obs_into(obs.unsafe_ptr())
            return obs^
        else:
            var obs = List[Scalar[Self.DTYPE]](
                length=RAM_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_ram_obs_into(obs.unsafe_ptr())
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
        return self.game.num_actions()

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
        """RAM mode step: use step_game, read 128 RAM bytes."""
        var reward = self.env.step_game(self.game, action)
        self.done = self.env.is_terminal()
        self.episode_reward += Float64(reward)
        var obs = self.get_obs_list()
        return (obs^, Scalar[Self.DTYPE](reward), self.done)

    def step_obs_into(
        mut self,
        action: Int,
        obs_out: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ) -> Tuple[Scalar[Self.DTYPE], Bool]:
        """Allocation-free step (trait override): advance the emulator and
        write the observation directly into `obs_out`, skipping the
        per-step List the `step_obs` path materializes (28,224 floats in
        pixel mode). Hot path for `BatchedCpuDiscreteEnv`."""
        self._steps += 1
        comptime if Self.OBS_MODE == 1:
            var reward = self._advance_pixel(action)
            self._write_stack_obs_into(obs_out)
            return (reward, self.done)
        else:
            var reward = self.env.step_game(self.game, action)
            self.done = self.env.is_terminal()
            self.episode_reward += Float64(reward)
            self._write_ram_obs_into(obs_out)
            return (Scalar[Self.DTYPE](reward), self.done)

    def _step_obs_pixel(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """Pixel mode step: list-returning wrapper over `_advance_pixel`."""
        var reward = self._advance_pixel(action)
        var obs = self.get_obs_list()
        return (obs^, reward, self.done)

    def _advance_pixel(mut self, action: Int) -> Scalar[Self.DTYPE]:
        """Pixel mode step: manual frame-skip with per-scanline rendering.

        Frame skip = 4 (default):
          - Frames 0,1: set_action + run_frame (no render, fast)
          - Frame 2: set_action + run_frame_video → raw_frame_a
          - Frame 3: set_action + run_frame_video → raw_frame_b
        Then max-pool a/b → grayscale → resize → push to stack. Returns
        the step reward; `self.done` carries the terminal flag.
        """
        var ale_action = self.game.action(action)
        var prev_score = Int(self.env.state.score)

        # We use a fixed frame_skip of 4 for pixel mode
        comptime PIXEL_FRAME_SKIP: Int = 4

        # Frames 0 .. skip-3: run without rendering
        for _ in range(PIXEL_FRAME_SKIP - 2):
            set_action(self.env.state, ale_action)
            run_frame(self.env.state, self.env.rom, self.env.rom_size)

        # Frame skip-2: render into raw_frame_a
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_a.value()
        )

        # Frame skip-1: render into raw_frame_b
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state, self.env.rom, self.env.rom_size, self.raw_frame_b.value()
        )

        # Extract RL signals from RAM (registry; includes per-game reward
        # quirks like Space Invaders' score-wrap correction, which the old
        # plain score-delta here missed)
        var sig = game_signals(self.game, self.env.state, prev_score)
        var reward = sig.reward
        self.env.state.score = Int32(sig.score)
        self.env.state.reward = Int32(reward)
        self.env.state.lives = UInt8(sig.lives)
        self.env.state.terminal = sig.terminal
        self.env.natural_terminal = sig.terminal

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

        return Scalar[Self.DTYPE](reward)
