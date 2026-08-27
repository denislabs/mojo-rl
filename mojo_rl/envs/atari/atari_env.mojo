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

from std.memory import alloc, unsafe_memset
from std.random import random_float64
from mojo_rl.core import State, Action, BoxDiscreteActionEnv
from mojo_rl.nn.constants import LAYOUT_NCHW, LAYOUT_NHWC
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

# RGB-96 pixel buffer sizes (OBS_MODE==2 — the EfficientZero-V2 Atari preprocessing:
# RGB, area-resized to 96×96, 4 frames stacked → [12,96,96]). Additive; the
# grayscale-84 path (OBS_MODE==1) is untouched.
comptime RGB_OBS_W: Int = 96
comptime RGB_OBS_H: Int = 96
comptime RGB_OBS_PLANE: Int = RGB_OBS_W * RGB_OBS_H  # 9216 (one 96×96 channel)
comptime RGB_FRAME_SIZE: Int = 3 * RGB_OBS_PLANE  # 27648 (one [3,96,96] frame)
comptime RGB_STACK_SIZE: Int = 4 * RGB_FRAME_SIZE  # 110592 ([12,96,96] stack)
comptime RGB_SRC_PLANE: Int = FRAME_WIDTH * FRAME_HEIGHT  # 33600 (one src channel)

# Grayscale-96 SINGLE-FRAME buffer size (OBS_MODE==3 — the DreamerV3 Atari
# preprocessing: one grayscale frame area-resized to 96×96, NO frame stacking
# (the RSSM carries temporal state). 96 (not 84) so the CNN's H%16==0 holds.
# Reuses `_resize_plane_160x210_to_96x96` on the single gray plane.
comptime GRAY96_OBS_W: Int = 96
comptime GRAY96_OBS_H: Int = 96
comptime GRAY96_SIZE: Int = GRAY96_OBS_W * GRAY96_OBS_H  # 9216 (one 96×96 frame)
# Grayscale-96 4-frame STACK (OBS_MODE==4) — same gray-96 preprocessing as mode 3
# but a 4-frame ring so the observation carries MOTION directly (velocity), which
# the single-frame mode 3 lacks. Empirically the RSSM prior can't generalize fast
# small-object dynamics (Pong ball) from single frames → imagination collapses;
# stacking hands it velocity so the prior's job is a one-step propagation.
comptime GRAY96_STACK_SIZE: Int = 4 * GRAY96_SIZE  # 36864 ([4,96,96] stack)
# Grayscale-64 SINGLE-FRAME buffer size (OBS_MODE==5) — the reference DreamerV3
# atari100k RESOLUTION (configs.yaml: 64×64; the published curves run at 64², so
# our 96² modes do ~2.25× the reference's conv FLOPs + im2col traffic). Same
# preprocessing as mode 3 (maxpool → gray → area box-filter resize), just 64².
# 64 % 16 == 0 → CNN minres 4 (reference-exact conv geometry end-to-end).
comptime GRAY64_OBS_W: Int = 64
comptime GRAY64_OBS_H: Int = 64
comptime GRAY64_SIZE: Int = GRAY64_OBS_W * GRAY64_OBS_H  # 4096 (one 64×64 frame)


# ============================================================================
# State and Action types
# ============================================================================


@fieldwise_init
struct AtariEnvState(Copyable, ImplicitlyCopyable, Movable, State):
    """Minimal state wrapper — frame number as index."""

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct AtariAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Thin wrapper around action index."""

    var action_idx: Int

    def __init__(out self, *, copy: Self):
        self.action_idx = copy.action_idx

    def __init__(out self, *, deinit move: Self):
        self.action_idx = move.action_idx


# ============================================================================
# Resize helper
# ============================================================================


def _resize_160x210_to_84x84[
    so: MutOrigin, do: MutOrigin, //
](
    src: Pointer[UInt8, so],
    dst: Pointer[UInt8, do],
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
                acc += src.unsafe_load[width=W](sy * FRAME_WIDTH + x).cast[
                    DType.uint16
                ]()
            vsum.unsafe_ptr().unsafe_store(x, acc)

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

            dst[unsafe_offset=oy * OBS_WIDTH + ox] = UInt8(total // ((sx1 - sx0) * cy))


def _bgra_maxpool_to_rgb_planar[
    ao: MutOrigin, bo: MutOrigin, do: MutOrigin, //
](
    a: Pointer[UInt8, ao],
    b: Pointer[UInt8, bo],
    dst: Pointer[UInt8, do],
):
    """Max-pool two BGRA frames (flicker handling, like the gray path) and
    write the result as three PLANAR channels R,G,B into `dst`
    (3 × 160×210). Channel order R,G,B matches gym/cv2's RGB observation
    that EfficientZero-V2's `WarpFrame` consumes.

    SIMD over little-endian u32 BGRA lanes (B|G<<8|R<<16|A<<24), identical
    channel extraction to `_bgra_to_gray_maxpool` but kept per-channel."""
    comptime W = 16
    comptime assert RGB_SRC_PLANE % W == 0, "src plane must be SIMD-divisible"
    var a32 = a.unsafe_bitcast[UInt32]()
    var b32 = b.unsafe_bitcast[UInt32]()
    var rp = dst
    var gp = dst.unsafe_offset(RGB_SRC_PLANE)
    var bp = dst.unsafe_offset(2 * RGB_SRC_PLANE)
    for i in range(0, RGB_SRC_PLANE, W):
        var va = a32.unsafe_load[width=W](i)
        var vb = b32.unsafe_load[width=W](i)
        bp.unsafe_store(i, max(va & 0xFF, vb & 0xFF).cast[DType.uint8]())
        gp.unsafe_store(i, max((va >> 8) & 0xFF, (vb >> 8) & 0xFF).cast[DType.uint8]())
        rp.unsafe_store(
            i, max((va >> 16) & 0xFF, (vb >> 16) & 0xFF).cast[DType.uint8]()
        )


def _resize_plane_160x210[
    OW: Int, OH: Int
](
    src: Pointer[UInt8, MutAnyOrigin],
    dst: Pointer[UInt8, MutAnyOrigin],
):
    """Area (box-filter) resize one 160×210 plane to OW×OH, the SAME
    integer-boundary box filter the 84×84 gray path uses (each output pixel
    averages its source rectangle). (Documented deviation: cv2's INTER_AREA
    uses fractional-coverage area weighting; this integer-boundary
    approximation is the same one already blessed for the gray-84 path — see
    docs/EZV2_ATARI_PARITY.md §A.) Comptime-generic over the output size —
    instantiated at 96 (modes 2/3/4, arithmetic identical to the previous
    fixed-96 function) and 64 (mode 5, the reference DreamerV3 resolution).
    """
    for oy in range(OH):
        var sy0 = (oy * FRAME_HEIGHT) // OH
        var sy1 = ((oy + 1) * FRAME_HEIGHT) // OH
        if sy1 <= sy0:
            sy1 = sy0 + 1
        for ox in range(OW):
            var sx0 = (ox * FRAME_WIDTH) // OW
            var sx1 = ((ox + 1) * FRAME_WIDTH) // OW
            if sx1 <= sx0:
                sx1 = sx0 + 1
            var total: Int = 0
            for sy in range(sy0, sy1):
                var row = sy * FRAME_WIDTH
                for sx in range(sx0, sx1):
                    total += Int(src[unsafe_offset=row + sx])
            dst[unsafe_offset=oy * OW + ox] = UInt8(
                total // ((sx1 - sx0) * (sy1 - sy0))
            )


def _resize_plane_160x210_to_96x96[
    so: MutOrigin, do: MutOrigin, //
](
    src: Pointer[UInt8, so],
    dst: Pointer[UInt8, do],
):
    """96×96 alias of `_resize_plane_160x210` (kept so the mode-2/3/4 call
    sites read unchanged). Scale: x = 160/96 ≈ 1.667, y = 210/96 ≈ 2.1875."""
    _resize_plane_160x210[RGB_OBS_W, RGB_OBS_H](
        src.as_unsafe_any_origin(), dst.as_unsafe_any_origin()
    )


# ============================================================================
# AtariEnv
# ============================================================================


struct AtariEnv[
    OBS_MODE: Int = 0,
    DTYPE: DType = DType.float32,
    LAYOUT: Int = LAYOUT_NCHW,
](BoxDiscreteActionEnv, Movable):
    """Atari environment conforming to BoxDiscreteActionEnv.

    The game is a runtime field (AtariGame registry) — score/lives/terminal
    extraction and the minimal action set dispatch on it per step, which is
    negligible next to the per-frame emulation cost.

    Parameters:
        OBS_MODE: 0 = RAM (128 floats), 1 = pixels grayscale (4×84×84 = 28224
            floats), 2 = pixels RGB-96 (4×[3,96,96] = 110592 floats — the
            EfficientZero-V2 Atari preprocessing), 3 = pixels grayscale-96
            SINGLE frame (96×96 = 9216 floats — the DreamerV3 Atari
            preprocessing: no frame stacking, the RSSM carries temporal state),
            4 = pixels grayscale-96 4-frame STACK (4×96×96 = 36864 floats — mode
            3 plus a 4-frame ring so the obs carries motion/velocity directly),
            5 = pixels grayscale-64 SINGLE frame (64×64 = 4096 floats — mode 3
            at the reference DreamerV3 atari100k RESOLUTION; ~2.25× cheaper
            conv stack than 96²).
        DTYPE: Observation dtype (default float32).
        LAYOUT: Observation layout (default NCHW).

    EfficientZero-V2 parity flags (all default off → existing behavior):
        clip_reward: emit sign(reward) ∈ {−1,0,1} (episode_reward stays RAW
            for correct score logging; training reward is clipped).
        episodic_life: treat loss of a life as a (non-bootstrapping) terminal
            without a true game reset (DeepMind EpisodicLifeEnv); a real reset
            happens only on game over. Inert for games with no lives (Pong).
        full_action_set: expose the full 18-action ALE set (policy head width
            18) instead of the game's minimal set, as EZv2 uses.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = AtariEnvState
    comptime ActionType = AtariAction

    var game: AtariGame
    var env: AtariEnvironment
    var episode_reward: Float64
    var done: Bool
    var _steps: Int

    # EfficientZero-V2 parity flags (default off; see struct doc).
    var clip_reward: Bool
    var episodic_life: Bool
    var full_action_set: Bool
    # DreamerV3 / Machado Atari protocol (default off → existing modes unchanged).
    var sticky_prob: Float64  # prob of repeating the previous action (0.25 = ALE sticky)
    var noop_max: Int  # random no-op starts: up to this many NOOPs after reset (0 = off)
    var _last_ale_action: UInt8  # last executed ALE action (for sticky repeat)
    # Episodic-life bookkeeping (only used when episodic_life=True).
    var _prev_lives: Int
    var _was_real_done: Bool  # true game-over on the last step (drives reset)
    var _life_lost: Bool  # last step lost a life (bootstrap-terminal, no reset)

    # Pixel-mode buffers (allocated only when OBS_MODE>=1)
    var frame_stack: Optional[Pointer[UInt8, MutUntrackedOrigin]]  # stack ring
    var frame_idx: Int  # ring buffer index
    var raw_frame_a: Optional[
        Pointer[UInt8, MutUntrackedOrigin]
    ]  # 160*210*4 BGRA
    var raw_frame_b: Optional[
        Pointer[UInt8, MutUntrackedOrigin]
    ]  # 160*210*4 BGRA
    var gray_buf: Optional[
        Pointer[UInt8, MutUntrackedOrigin]
    ]  # 160*210 grayscale
    var rgb_buf: Optional[
        Pointer[UInt8, MutUntrackedOrigin]
    ]  # 3*160*210 planar RGB

    def __init__(
        out self,
        game: AtariGame,
        frame_skip: Int = 4,
        max_frames: Int = 108000,
        clip_reward: Bool = False,
        episodic_life: Bool = False,
        full_action_set: Bool = False,
        sticky_prob: Float64 = 0.0,
        noop_max: Int = 0,
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
            clip_reward=clip_reward,
            episodic_life=episodic_life,
            full_action_set=full_action_set,
            sticky_prob=sticky_prob,
            noop_max=noop_max,
        )

    def __init__(
        out self,
        game: AtariGame,
        rom: Pointer[UInt8, MutUntrackedOrigin],
        rom_size: Int,
        frame_skip: Int = 4,
        max_frames: Int = 108000,
        clip_reward: Bool = False,
        episodic_life: Bool = False,
        full_action_set: Bool = False,
        sticky_prob: Float64 = 0.0,
        noop_max: Int = 0,
    ):
        """Create an AtariEnv from an explicitly loaded ROM.

        For pixel modes (OBS_MODE>=1), frame_skip is managed internally
        (env.frame_skip is set to 1, skip loop is in step_obs).

        Args:
            game: Registry game (drives action set + RAM signal extraction).
            rom: ROM data pointer.
            rom_size: ROM size in bytes.
            frame_skip: Number of frames to repeat each action (default 4).
            max_frames: Max frames per episode (default 108000).
            clip_reward: Emit sign(reward); see struct doc.
            episodic_life: Life loss = terminal; see struct doc.
            full_action_set: Expose full 18-action ALE set; see struct doc.
            sticky_prob: Probability of repeating the previous action
                (ALE sticky actions); 0.0 disables.
            noop_max: Max random no-op actions at reset; 0 disables.
        """
        self.game = game
        comptime if Self.OBS_MODE >= 1:
            # Pixel modes: we drive frame skip manually
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
        self.clip_reward = clip_reward
        self.episodic_life = episodic_life
        self.full_action_set = full_action_set
        self.sticky_prob = sticky_prob
        self.noop_max = noop_max
        self._last_ale_action = 0  # ALE NOOP
        self._prev_lives = 0
        self._was_real_done = True
        self._life_lost = False

        # Pixel mode buffers
        comptime if Self.OBS_MODE == 1:
            self.frame_stack = alloc[UInt8](
                {count = FRAME_STACK_SIZE}
            ).unsafe_leak()
            self.raw_frame_a = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.raw_frame_b = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.gray_buf = alloc[UInt8](
                {count = GRAY_FRAME_SIZE}
            ).unsafe_leak()
            self.rgb_buf = None
            self.frame_idx = 0
            unsafe_memset(self.frame_stack.value(), 0, FRAME_STACK_SIZE)
        elif Self.OBS_MODE == 2:
            self.frame_stack = alloc[UInt8](
                {count = RGB_STACK_SIZE}
            ).unsafe_leak()
            self.raw_frame_a = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.raw_frame_b = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.gray_buf = None
            self.rgb_buf = alloc[UInt8](
                {count = 3 * RGB_SRC_PLANE}
            ).unsafe_leak()
            self.frame_idx = 0
            unsafe_memset(self.frame_stack.value(), 0, RGB_STACK_SIZE)
        elif Self.OBS_MODE == 3:
            # Grayscale-96 single frame: `frame_stack` holds ONE 96×96 frame
            # (no ring). Reuse gray_buf (160×210) as the maxpool scratch.
            self.frame_stack = alloc[UInt8]({count = GRAY96_SIZE}).unsafe_leak()
            self.raw_frame_a = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.raw_frame_b = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.gray_buf = alloc[UInt8](
                {count = GRAY_FRAME_SIZE}
            ).unsafe_leak()
            self.rgb_buf = None
            self.frame_idx = 0
            unsafe_memset(self.frame_stack.value(), 0, GRAY96_SIZE)
        elif Self.OBS_MODE == 4:
            # Grayscale-96 4-frame stack: `frame_stack` is a 4-slot ring.
            self.frame_stack = alloc[UInt8](
                {count = GRAY96_STACK_SIZE}
            ).unsafe_leak()
            self.raw_frame_a = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.raw_frame_b = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.gray_buf = alloc[UInt8](
                {count = GRAY_FRAME_SIZE}
            ).unsafe_leak()
            self.rgb_buf = None
            self.frame_idx = 0
            unsafe_memset(self.frame_stack.value(), 0, GRAY96_STACK_SIZE)
        elif Self.OBS_MODE == 5:
            # Grayscale-64 single frame (reference DreamerV3 resolution):
            # one 64×64 slot, no ring; gray_buf as the maxpool scratch.
            self.frame_stack = alloc[UInt8]({count = GRAY64_SIZE}).unsafe_leak()
            self.raw_frame_a = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.raw_frame_b = alloc[UInt8](
                {count = FRAME_BGRA_SIZE}
            ).unsafe_leak()
            self.gray_buf = alloc[UInt8](
                {count = GRAY_FRAME_SIZE}
            ).unsafe_leak()
            self.rgb_buf = None
            self.frame_idx = 0
            unsafe_memset(self.frame_stack.value(), 0, GRAY64_SIZE)
        else:
            self.frame_stack = None
            self.raw_frame_a = None
            self.raw_frame_b = None
            self.gray_buf = None
            self.rgb_buf = None
            self.frame_idx = 0

    def __init__(out self, *, deinit move: Self):
        self.game = move.game
        self.env = move.env^
        self.episode_reward = move.episode_reward
        self.done = move.done
        self._steps = move._steps
        self.clip_reward = move.clip_reward
        self.episodic_life = move.episodic_life
        self.full_action_set = move.full_action_set
        self.sticky_prob = move.sticky_prob
        self.noop_max = move.noop_max
        self._last_ale_action = move._last_ale_action
        self._prev_lives = move._prev_lives
        self._was_real_done = move._was_real_done
        self._life_lost = move._life_lost
        self.frame_stack = move.frame_stack
        self.frame_idx = move.frame_idx
        self.raw_frame_a = move.raw_frame_a
        self.raw_frame_b = move.raw_frame_b
        self.gray_buf = move.gray_buf
        self.rgb_buf = move.rgb_buf

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
        var a32 = self.raw_frame_a.value().unsafe_bitcast[UInt32]()
        var b32 = self.raw_frame_b.value().unsafe_bitcast[UInt32]()
        var gray = self.gray_buf.value()
        for i in range(0, GRAY_FRAME_SIZE, W):
            var va = a32.unsafe_load[width=W](i)
            var vb = b32.unsafe_load[width=W](i)
            var b = max(va & 0xFF, vb & 0xFF)
            var g = max((va >> 8) & 0xFF, (vb >> 8) & 0xFF)
            var r = max((va >> 16) & 0xFF, (vb >> 16) & 0xFF)
            # Luminance: Y = (77*R + 150*G + 29*B) >> 8
            gray.unsafe_store(
                i, ((77 * r + 150 * g + 29 * b) >> 8).cast[DType.uint8]()
            )

    def _push_frame_to_stack(mut self):
        """Resize gray_buf (160×210) to 84×84 and push into frame_stack ring buffer.
        """
        var slot_offset = self.frame_idx * OBS_FRAME_SIZE
        _resize_160x210_to_84x84(
            self.gray_buf.value(),
            self.frame_stack.value().unsafe_offset(slot_offset),
        )
        self.frame_idx = (self.frame_idx + 1) % 4

    def _render_current_frame(mut self):
        """Render the current emulator state into raw_frame_a using run_frame_video.

        Runs one NOOP frame with video output to capture the display.
        """
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_a.value().as_unsafe_any_origin(),
        )

    # ── RGB-96 pixel helpers (OBS_MODE==2) ──────────────────────────────

    def _push_rgb_frame_to_stack(mut self):
        """Max-pool a/b → planar RGB (rgb_buf) → area-resize each of the 3
        channels into the frame_stack ring slot [3,96,96], advance the ring.
        """
        _bgra_maxpool_to_rgb_planar(
            self.raw_frame_a.value().as_unsafe_any_origin(),
            self.raw_frame_b.value().as_unsafe_any_origin(),
            self.rgb_buf.value(),
        )
        var slot = self.frame_stack.value().unsafe_offset(self.frame_idx * RGB_FRAME_SIZE)
        var src = self.rgb_buf.value()
        for c in range(3):
            _resize_plane_160x210_to_96x96(
                src.unsafe_offset(c * RGB_SRC_PLANE),
                slot.unsafe_offset(c * RGB_OBS_PLANE),
            )
        self.frame_idx = (self.frame_idx + 1) % 4

    def _write_rgb_stack_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the 4-frame RGB stack (chronological, oldest first) as
        normalized floats into `obs_out`. The 12 logical channels are frame-major
        (f0R,f0G,f0B, f1R,…, f3B), each from the source ring slot's [3,96,96]
        planar frame. Channel placement follows `Self.LAYOUT` (channels_last
        migration — see CHANNELS_LAST_NHWC_MIGRATION_PLAN.md), default NCHW so
        every existing consumer (muzero/DQN/…) is byte-identical:
          NCHW → [12,96,96] (channel-outer, contiguous per channel)
          NHWC → [96,96,12] (channel-inner, interleaved per pixel)
        Only OBS_MODE==2 (RGB-96) honors NHWC; the gray-84 / RAM writers are
        NCHW-only (no NHWC consumer)."""
        var fs = self.frame_stack.value()
        comptime if Self.LAYOUT == LAYOUT_NHWC:
            # Channel-inner scatter: obs[p*12 + ch]. The source frame is planar
            # [3,96,96], so the contiguous-per-channel SIMD copy can't apply;
            # gather each channel for pixel p. (CPU obs gen — emulation dominates.)
            comptime CH = RGB_STACK_SIZE // RGB_OBS_PLANE  # 12
            for i in range(4):
                var slot = (self.frame_idx + i) % 4  # oldest first
                var src = fs.unsafe_offset(slot * RGB_FRAME_SIZE)
                for c in range(3):
                    var ch = i * 3 + c
                    var src_c = src.unsafe_offset(c * RGB_OBS_PLANE)
                    for p in range(RGB_OBS_PLANE):
                        obs_out[unsafe_offset=p * CH + ch] = (
                            src_c[unsafe_offset=p].cast[Self.dtype]() / 255.0
                        )
        else:
            # NCHW: contiguous per-channel → SIMD frame-block copy (bit-identical
            # to the pre-LAYOUT path).
            comptime W = 16
            comptime assert (
                RGB_OBS_PLANE % W == 0
            ), "RGB plane must be SIMD-divisible"
            var out_off = 0
            for i in range(4):
                var slot = (self.frame_idx + i) % 4  # oldest first
                var src = fs.unsafe_offset(slot * RGB_FRAME_SIZE)
                for j in range(0, RGB_FRAME_SIZE, W):
                    obs_out.unsafe_store(
                        out_off + j,
                        src.unsafe_load[width=W](j).cast[Self.dtype]() / 255.0,
                    )
                out_off += RGB_FRAME_SIZE

    # ── grayscale-96 single-frame helpers (OBS_MODE==3 — DreamerV3 Atari) ─

    def _push_gray96_frame(mut self):
        """Max-pool a/b → grayscale (gray_buf 160×210) → area-resize to the
        single 96×96 `frame_stack` slot (no ring — DreamerV3 doesn't stack)."""
        self._bgra_to_gray_maxpool()
        _resize_plane_160x210_to_96x96(
            self.gray_buf.value(), self.frame_stack.value()
        )

    def _write_gray96_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the single 96×96 grayscale frame as normalized floats into
        `obs_out` (GRAY96_SIZE scalars). SIMD uint8→float `/255`."""
        comptime W = 16
        comptime assert GRAY96_SIZE % W == 0, "gray96 must be SIMD-divisible"
        var fs = self.frame_stack.value()
        for j in range(0, GRAY96_SIZE, W):
            obs_out.unsafe_store(
                j, fs.unsafe_load[width=W](j).cast[Self.dtype]() / 255.0
            )

    # ── grayscale-64 single-frame helpers (OBS_MODE==5 — DreamerV3 ref res) ─

    def _push_gray64_frame(mut self):
        """Max-pool a/b → grayscale (gray_buf 160×210) → area-resize to the
        single 64×64 `frame_stack` slot (no ring — mode 3 at the reference
        DreamerV3 atari100k resolution)."""
        self._bgra_to_gray_maxpool()
        _resize_plane_160x210[GRAY64_OBS_W, GRAY64_OBS_H](
            self.gray_buf.value().as_unsafe_any_origin(),
            self.frame_stack.value().as_unsafe_any_origin(),
        )

    def _write_gray64_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the single 64×64 grayscale frame as normalized floats into
        `obs_out` (GRAY64_SIZE scalars). SIMD uint8→float `/255`."""
        comptime W = 16
        comptime assert GRAY64_SIZE % W == 0, "gray64 must be SIMD-divisible"
        var fs = self.frame_stack.value()
        for j in range(0, GRAY64_SIZE, W):
            obs_out.unsafe_store(
                j, fs.unsafe_load[width=W](j).cast[Self.dtype]() / 255.0
            )

    def _push_gray96_stack(mut self):
        """Max-pool a/b → grayscale → area-resize to 96×96 into the current ring
        slot, then advance the ring (OBS_MODE==4)."""
        self._bgra_to_gray_maxpool()
        var slot = self.frame_stack.value().unsafe_offset(self.frame_idx * GRAY96_SIZE)
        _resize_plane_160x210_to_96x96(self.gray_buf.value(), slot)
        self.frame_idx = (self.frame_idx + 1) % 4

    def _write_gray96_stack_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the 4-frame gray-96 stack (chronological, oldest first) as
        normalized floats into `obs_out` (GRAY96_STACK_SIZE scalars)."""
        comptime W = 16
        comptime assert GRAY96_SIZE % W == 0, "gray96 must be SIMD-divisible"
        var fs = self.frame_stack.value()
        var out_off = 0
        for i in range(4):
            var slot = (self.frame_idx + i) % 4  # oldest first
            var src = fs.unsafe_offset(slot * GRAY96_SIZE)
            for j in range(0, GRAY96_SIZE, W):
                obs_out.unsafe_store(
                    out_off + j, src.unsafe_load[width=W](j).cast[Self.dtype]() / 255.0
                )
            out_off += GRAY96_SIZE

    def _apply_sticky(mut self, ale_action: UInt8) -> UInt8:
        """Machado sticky actions: with prob `sticky_prob`, repeat the previous
        ALE action instead of the requested one; record the executed action.
        `sticky_prob=0.0` (default) → always the requested action."""
        var a = ale_action
        if self.sticky_prob > 0.0 and random_float64() < self.sticky_prob:
            a = self._last_ale_action
        self._last_ale_action = a
        return a

    # ========================================================================
    # Env trait (base)
    # ========================================================================

    def reset(mut self) -> AtariEnvState:
        """Reset the environment.

        With `episodic_life` and no true game-over on the previous step
        (DeepMind EpisodicLifeEnv semantics), this does NOT reset the game:
        it advances one no-op action-step past the lost-life state so all
        states stay reachable, keeping episode bookkeeping intact. A true
        game reset happens only on real game over (or when episodic_life is
        off — the default)."""
        if self.episodic_life and not self._was_real_done:
            # Advance past the lost-life state; the real episode continues.
            _ = self.step_obs(0)  # action 0 == ALE NOOP (minimal + full sets)
            self._prev_lives = Int(self.env.state.lives)
            self.done = False
            self._life_lost = False
            return AtariEnvState(index=self._steps)

        self.env.reset_game(self.game)
        self.episode_reward = 0.0
        self.done = False
        self._steps = 0
        self._was_real_done = True
        self._life_lost = False
        self._prev_lives = 0
        self._last_ale_action = 0  # ALE NOOP (sticky-action history)

        # Random no-op starts (Machado): advance a random number of NOOP frames
        # for initial-state diversity before capturing the first obs (noop_max=0
        # → off, existing behavior).
        comptime if Self.OBS_MODE >= 1:
            if self.noop_max > 0:
                var nk = Int(random_float64() * Float64(self.noop_max + 1))
                for _ in range(nk):
                    set_action(self.env.state, ACTION_NOOP)
                    run_frame(
                        self.env.state,
                        self.env.rom.as_unsafe_any_origin(),
                        self.env.rom_size,
                    )

        comptime if Self.OBS_MODE == 1:
            # Render initial frame into all 4 stack slots
            run_frame_video(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
                self.raw_frame_a.value().as_unsafe_any_origin(),
            )
            # Copy frame_a to frame_b for maxpool (both identical after reset)
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[unsafe_offset=i] = self.raw_frame_a.value()[unsafe_offset=i]
            self._bgra_to_gray_maxpool()
            # Fill all 4 slots with the same initial frame
            self.frame_idx = 0
            for _ in range(4):
                self._push_frame_to_stack()
        elif Self.OBS_MODE == 2:
            # Render initial frame into all 4 RGB stack slots
            run_frame_video(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
                self.raw_frame_a.value().as_unsafe_any_origin(),
            )
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[unsafe_offset=i] = self.raw_frame_a.value()[unsafe_offset=i]
            self.frame_idx = 0
            for _ in range(4):
                self._push_rgb_frame_to_stack()
        elif Self.OBS_MODE == 3:
            # Grayscale-96 single frame: render the initial frame (a==b after
            # reset) → maxpool → resize into the single 96×96 slot.
            run_frame_video(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
                self.raw_frame_a.value().as_unsafe_any_origin(),
            )
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[unsafe_offset=i] = self.raw_frame_a.value()[unsafe_offset=i]
            self._push_gray96_frame()
        elif Self.OBS_MODE == 4:
            # Grayscale-96 4-stack: render the initial frame (a==b) and fill all
            # 4 ring slots with it.
            run_frame_video(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
                self.raw_frame_a.value().as_unsafe_any_origin(),
            )
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[unsafe_offset=i] = self.raw_frame_a.value()[unsafe_offset=i]
            self.frame_idx = 0
            for _ in range(4):
                self._push_gray96_stack()
        elif Self.OBS_MODE == 5:
            # Grayscale-64 single frame: render (a==b) → maxpool → resize into
            # the single 64×64 slot.
            run_frame_video(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
                self.raw_frame_a.value().as_unsafe_any_origin(),
            )
            for i in range(FRAME_BGRA_SIZE):
                self.raw_frame_b.value()[unsafe_offset=i] = self.raw_frame_a.value()[unsafe_offset=i]
            self._push_gray64_frame()

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

    def get_state(mut self) -> AtariEnvState:
        return AtariEnvState(index=self._steps)

    def was_terminated(self) -> Bool:
        """True iff the last step ended in a (bootstrap-)terminal — game over,
        or (with episodic_life) loss of a life — but NOT max_frames
        truncation, on which the TD bootstrap is kept. Overrides the base-Env
        `False` default, which silently classified every Atari game-over as a
        truncation."""
        if self.episodic_life:
            return self.env.natural_terminal or self._life_lost
        return self.env.natural_terminal

    def close(mut self):
        """Free pixel-mode buffers."""
        comptime if Self.OBS_MODE >= 1:
            if Bool(self.frame_stack):
                self.frame_stack.value().unsafe_free()
                self.frame_stack = None
            if Bool(self.raw_frame_a):
                self.raw_frame_a.value().unsafe_free()
                self.raw_frame_a = None
            if Bool(self.raw_frame_b):
                self.raw_frame_b.value().unsafe_free()
                self.raw_frame_b = None
            if Bool(self.gray_buf):
                self.gray_buf.value().unsafe_free()
                self.gray_buf = None
            if Bool(self.rgb_buf):
                self.rgb_buf.value().unsafe_free()
                self.rgb_buf = None

    # ========================================================================
    # ContinuousStateEnv trait
    # ========================================================================

    def _write_stack_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the 4-frame stack (chronological order, oldest first) as
        normalized floats into `obs_out` (FRAME_STACK_SIZE scalars).
        SIMD uint8→float `/255` — bit-exact vs the per-element scalar
        conversion (each uint8 value maps to the identical float)."""
        comptime W = 16
        comptime assert (
            OBS_FRAME_SIZE % W == 0
        ), "obs frame must be SIMD-divisible"
        var fs = self.frame_stack.value()
        var out_off = 0
        for i in range(4):
            var slot = (self.frame_idx + i) % 4  # oldest first
            var src = fs.unsafe_offset(slot * OBS_FRAME_SIZE)
            for j in range(0, OBS_FRAME_SIZE, W):
                obs_out.unsafe_store(
                    out_off + j,
                    src.unsafe_load[width=W](j).cast[Self.dtype]() / 255.0,
                )
            out_off += OBS_FRAME_SIZE

    def _write_ram_obs_into[
        o: MutOrigin
    ](self, obs_out: Pointer[Scalar[Self.dtype], o]):
        """Write the 128 RAM bytes as normalized floats into `obs_out`."""
        comptime W = 16
        comptime assert RAM_SIZE % W == 0, "RAM size must be SIMD-divisible"
        var ram = self.env.get_ram()
        for i in range(0, RAM_SIZE, W):
            obs_out.unsafe_store(
                i, ram.unsafe_ptr().unsafe_load[width=W](i).cast[Self.dtype]() / 255.0
            )

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        """Return current observation as a list of floats.

        RAM mode: 128 floats in [0, 1].
        Pixel mode 1: 28224 floats (4 stacked 84×84 grayscale frames).
        Pixel mode 2: 110592 floats ([12,96,96] RGB stack — EZv2 Atari).
        """
        comptime if Self.OBS_MODE == 1:
            var obs = List[Scalar[Self.DTYPE]](
                length=FRAME_STACK_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_stack_obs_into(obs.unsafe_ptr())
            return obs^
        elif Self.OBS_MODE == 2:
            var obs = List[Scalar[Self.DTYPE]](
                length=RGB_STACK_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_rgb_stack_obs_into(obs.unsafe_ptr())
            return obs^
        elif Self.OBS_MODE == 3:
            var obs = List[Scalar[Self.DTYPE]](
                length=GRAY96_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_gray96_obs_into(obs.unsafe_ptr())
            return obs^
        elif Self.OBS_MODE == 4:
            var obs = List[Scalar[Self.DTYPE]](
                length=GRAY96_STACK_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_gray96_stack_obs_into(obs.unsafe_ptr())
            return obs^
        elif Self.OBS_MODE == 5:
            var obs = List[Scalar[Self.DTYPE]](
                length=GRAY64_SIZE, fill=Scalar[Self.DTYPE](0.0)
            )
            self._write_gray64_obs_into(obs.unsafe_ptr())
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
        elif Self.OBS_MODE == 2:
            return RGB_STACK_SIZE  # 4 * 3 * 96 * 96 = 110592
        elif Self.OBS_MODE == 3:
            return GRAY96_SIZE  # 96 * 96 = 9216 (single grayscale frame)
        elif Self.OBS_MODE == 4:
            return GRAY96_STACK_SIZE  # 4 * 96 * 96 = 36864 (gray-96 4-stack)
        elif Self.OBS_MODE == 5:
            return GRAY64_SIZE  # 64 * 64 = 4096 (single grayscale frame)
        else:
            return RAM_SIZE  # 128

    # ========================================================================
    # DiscreteActionEnv trait
    # ========================================================================

    def action_from_index(self, action_idx: Int) -> AtariAction:
        return AtariAction(action_idx=action_idx)

    def num_actions(self) -> Int:
        # full_action_set exposes the full 18-action ALE set (EZv2); the
        # minimal set is the default. (full_action_set is honored by the
        # pixel paths; the RAM path keeps its minimal-set mapping.)
        if self.full_action_set:
            return 18
        return self.game.num_actions()

    def _ale_action(self, action_idx: Int) -> UInt8:
        """Map an agent action index to an ALE action id. With
        full_action_set the index IS the ALE id (0..17); otherwise it
        routes through the game's minimal action set (ascending id order)."""
        if self.full_action_set:
            return UInt8(action_idx)
        return self.game.action(action_idx)

    def _clip(self, raw_reward: Int) -> Scalar[Self.DTYPE]:
        """sign(reward) ∈ {−1,0,1} when clip_reward, else the raw reward."""
        if self.clip_reward:
            if raw_reward > 0:
                return Scalar[Self.DTYPE](1.0)
            elif raw_reward < 0:
                return Scalar[Self.DTYPE](-1.0)
            return Scalar[Self.DTYPE](0.0)
        return Scalar[Self.DTYPE](raw_reward)

    def _apply_episodic(mut self):
        """After self.done is set from is_terminal(): record the true
        game-over (drives reset) and, with episodic_life, upgrade loss of a
        life to a (bootstrap-)terminal without a real reset."""
        self._was_real_done = self.env.natural_terminal
        self._life_lost = False
        if self.episodic_life:
            var lives = Int(self.env.state.lives)
            if lives < self._prev_lives and lives > 0:
                self.done = True
                self._life_lost = True
            self._prev_lives = lives

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
        elif Self.OBS_MODE == 2:
            return self._step_obs_rgb(action)
        elif Self.OBS_MODE == 3:
            return self._step_obs_gray96(action)
        elif Self.OBS_MODE == 4:
            return self._step_obs_gray96(action)
        elif Self.OBS_MODE == 5:
            return self._step_obs_gray96(action)  # size-dispatch in the push/obs
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
        obs_out: Pointer[Scalar[Self.dtype], MutAnyOrigin],
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
        elif Self.OBS_MODE == 2:
            var reward = self._advance_pixel_rgb(action)
            self._write_rgb_stack_obs_into(obs_out)
            return (reward, self.done)
        elif Self.OBS_MODE == 3:
            var reward = self._advance_pixel_gray96(action)
            self._write_gray96_obs_into(obs_out)
            return (reward, self.done)
        elif Self.OBS_MODE == 4:
            var reward = self._advance_pixel_gray96(action)
            self._write_gray96_stack_obs_into(obs_out)
            return (reward, self.done)
        elif Self.OBS_MODE == 5:
            var reward = self._advance_pixel_gray96(action)
            self._write_gray64_obs_into(obs_out)
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

        Honors `sticky_prob`, `full_action_set`, `clip_reward` and
        `episodic_life` exactly like the RGB-96/gray-96 paths (this mode
        used to silently ignore all four — non-protocol training with no
        warning). All are identity at the default flag values.
        """
        var ale_action = self._apply_sticky(self._ale_action(action))
        var prev_score = Int(self.env.state.score)

        # We use a fixed frame_skip of 4 for pixel mode
        comptime PIXEL_FRAME_SKIP: Int = 4

        # Frames 0 .. skip-3: run without rendering
        for _ in range(PIXEL_FRAME_SKIP - 2):
            set_action(self.env.state, ale_action)
            run_frame(self.env.state, self.env.rom.as_unsafe_any_origin(), self.env.rom_size)

        # Frame skip-2: render into raw_frame_a
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_a.value().as_unsafe_any_origin(),
        )

        # Frame skip-1: render into raw_frame_b
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_b.value().as_unsafe_any_origin(),
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
        self.episode_reward += Float64(reward)  # raw reward for score logging
        self._apply_episodic()  # may upgrade done on life loss

        # Max-pool → grayscale → resize → push to frame stack
        self._bgra_to_gray_maxpool()
        self._push_frame_to_stack()

        return self._clip(reward)

    # ── RGB-96 step (OBS_MODE==2 — EfficientZero-V2 Atari) ──────────────

    def _step_obs_rgb(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """RGB-96 pixel step: list-returning wrapper over `_advance_pixel_rgb`.
        """
        var reward = self._advance_pixel_rgb(action)
        var obs = self.get_obs_list()
        return (obs^, reward, self.done)

    def _advance_pixel_rgb(mut self, action: Int) -> Scalar[Self.DTYPE]:
        """RGB-96 pixel step (EfficientZero-V2 preprocessing): manual
        frame-skip with per-scanline rendering of the last 2 frames, then
        max-pool a/b → planar RGB → area-resize each channel to 96×96 →
        push to the [12,96,96] stack. Honors `full_action_set`, `clip_reward`
        and `episodic_life`. Mirrors `_advance_pixel` (the gray-84 path) but
        keeps RGB; episode_reward stays RAW for score logging."""
        var ale_action = self._ale_action(action)
        var prev_score = Int(self.env.state.score)

        comptime PIXEL_FRAME_SKIP: Int = 4

        for _ in range(PIXEL_FRAME_SKIP - 2):
            set_action(self.env.state, ale_action)
            run_frame(self.env.state, self.env.rom.as_unsafe_any_origin(), self.env.rom_size)

        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_a.value().as_unsafe_any_origin(),
        )
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_b.value().as_unsafe_any_origin(),
        )

        var sig = game_signals(self.game, self.env.state, prev_score)
        var reward = sig.reward
        self.env.state.score = Int32(sig.score)
        self.env.state.reward = Int32(reward)
        self.env.state.lives = UInt8(sig.lives)
        self.env.state.terminal = sig.terminal
        self.env.natural_terminal = sig.terminal

        if (
            self.env.max_frames > 0
            and Int(self.env.state.frame_number) >= self.env.max_frames
        ):
            self.env.state.terminal = True

        self.done = self.env.is_terminal()
        self.episode_reward += Float64(reward)  # raw reward for score logging
        self._apply_episodic()  # may upgrade done on life loss

        # Max-pool → planar RGB → resize → push to [12,96,96] stack
        self._push_rgb_frame_to_stack()

        return self._clip(reward)

    # ── grayscale-96 single-frame step (OBS_MODE==3 — DreamerV3 Atari) ────

    def _step_obs_gray96(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        """Gray-96 single-frame step: list-returning wrapper over
        `_advance_pixel_gray96`."""
        var reward = self._advance_pixel_gray96(action)
        var obs = self.get_obs_list()
        return (obs^, reward, self.done)

    def _advance_pixel_gray96(mut self, action: Int) -> Scalar[Self.DTYPE]:
        """Gray-96 single-frame step (DreamerV3 preprocessing): sticky-action
        gating, manual frame-skip=4 rendering the last 2 sub-frames, max-pool
        a/b → grayscale → area-resize to a single 96×96 obs (NO stacking).
        Honors `sticky_prob`, `full_action_set`, `clip_reward`, `episodic_life`.
        Mirrors `_advance_pixel_rgb` but keeps a single gray frame."""
        var ale_action = self._apply_sticky(self._ale_action(action))
        var prev_score = Int(self.env.state.score)

        comptime PIXEL_FRAME_SKIP: Int = 4

        for _ in range(PIXEL_FRAME_SKIP - 2):
            set_action(self.env.state, ale_action)
            run_frame(
                self.env.state,
                self.env.rom.as_unsafe_any_origin(),
                self.env.rom_size,
            )

        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_a.value().as_unsafe_any_origin(),
        )
        set_action(self.env.state, ale_action)
        run_frame_video(
            self.env.state,
            self.env.rom.as_unsafe_any_origin(),
            self.env.rom_size,
            self.raw_frame_b.value().as_unsafe_any_origin(),
        )

        var sig = game_signals(self.game, self.env.state, prev_score)
        var reward = sig.reward
        self.env.state.score = Int32(sig.score)
        self.env.state.reward = Int32(reward)
        self.env.state.lives = UInt8(sig.lives)
        self.env.state.terminal = sig.terminal
        self.env.natural_terminal = sig.terminal

        if (
            self.env.max_frames > 0
            and Int(self.env.state.frame_number) >= self.env.max_frames
        ):
            self.env.state.terminal = True

        self.done = self.env.is_terminal()
        self.episode_reward += Float64(reward)  # raw reward for score logging
        self._apply_episodic()  # may upgrade done on life loss

        # Max-pool → grayscale → resize → gray obs (single 96² slot for mode 3,
        # ring slot for the 4-stack mode 4, single 64² slot for mode 5).
        comptime if Self.OBS_MODE == 4:
            self._push_gray96_stack()
        elif Self.OBS_MODE == 5:
            self._push_gray64_frame()
        else:
            self._push_gray96_frame()

        return self._clip(reward)
