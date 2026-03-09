"""Atari 2600 environment — ties together CPU, TIA, RIOT, and game logic.

This module provides the high-level step/reset interface for running
Atari games. It manages ROM loading, emulation, and RL signal extraction.

Usage:
    var rom_data = load_rom("/path/to/pong.bin")
    var env = AtariEnvironment(rom_data.ptr, rom_data.size)
    env.reset()
    for step in range(1000):
        var reward = env.step(ACTION_FIRE)
"""

from .atari_state import AtariState
from .cpu6502 import cpu_reset, run_frame, mem_read
from .cartridge import init_bank
from .riot import set_action
from .tia import tia_update_frame_scanline
from .flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_RESET,
    TOTAL_SCANLINES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)


struct AtariEnvironment(Movable):
    """High-level Atari 2600 environment for RL.

    Manages the emulator state and provides step/reset interface.
    Frame skip and action repeat are handled at this level.
    """

    var state: AtariState
    var rom: UnsafePointer[UInt8, MutAnyOrigin]
    var rom_size: Int
    var frame_skip: Int
    var max_frames: Int  # Max frames per episode (0 = unlimited)

    fn __init__(
        out self,
        rom: UnsafePointer[UInt8, MutAnyOrigin],
        rom_size: Int,
        frame_skip: Int = 4,
        max_frames: Int = 108000,  # Standard ALE default (~30 min at 60fps)
    ):
        self.state = AtariState()
        self.rom = rom
        self.rom_size = rom_size
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    fn __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.rom = take.rom
        self.rom_size = take.rom_size
        self.frame_skip = take.frame_skip
        self.max_frames = take.max_frames

    fn reset(mut self):
        """Reset the environment to initial state."""
        self.state = AtariState()
        init_bank(self.state, self.rom_size)
        cpu_reset(self.state, self.rom, self.rom_size)

        # Run a few frames to get past the title screen / initialization
        # (Most games need ~60 frames of NOOP to start)
        for _ in range(60):
            set_action(self.state, ACTION_NOOP)
            run_frame(self.state, self.rom, self.rom_size)

        # Press RESET to start the game
        set_action(self.state, ACTION_RESET)
        run_frame(self.state, self.rom, self.rom_size)

        # A few more NOOP frames
        for _ in range(10):
            set_action(self.state, ACTION_NOOP)
            run_frame(self.state, self.rom, self.rom_size)

        # Initialize RL state
        self.state.reward = 0
        self.state.score = 0
        self.state.terminal = False
        self.state.started = True

    fn step(mut self, action: UInt8) -> Int:
        """Execute one step (frame_skip frames with the same action).

        Returns the cumulative reward over the skipped frames.
        """
        var total_reward: Int = 0
        var prev_score = Int(self.state.score)

        for frame in range(self.frame_skip):
            set_action(self.state, action)
            run_frame(self.state, self.rom, self.rom_size)

        # Return score delta as reward (game-specific extraction happens externally)
        return total_reward

    fn step_with_game[GameDef: _GameDefTrait](mut self, action_idx: Int) -> Int:
        """Execute one step using a game definition for reward/terminal extraction.

        This is the preferred interface — it handles:
        - Action mapping (agent index → ALE action)
        - Frame skipping
        - Score/reward/lives/terminal extraction from RAM
        """
        var ale_action = GameDef.map_action(action_idx)
        var prev_score = Int(self.state.score)

        for frame in range(self.frame_skip):
            set_action(self.state, ale_action)
            run_frame(self.state, self.rom, self.rom_size)

        # Extract RL signals from RAM
        var new_score = GameDef.get_score(self.state.ram)
        var reward = new_score - prev_score
        self.state.score = Int32(new_score)
        self.state.reward = Int32(reward)
        self.state.lives = UInt8(GameDef.get_lives(self.state.ram))
        self.state.terminal = GameDef.is_terminal(self.state.ram)

        # Check max frames truncation
        if (
            self.max_frames > 0
            and Int(self.state.frame_number) >= self.max_frames
        ):
            self.state.terminal = True

        return reward

    fn get_ram(self) -> InlineArray[UInt8, RAM_SIZE]:
        """Get a copy of the 128-byte RAM (for RAM observations)."""
        return self.state.ram.copy()

    fn is_terminal(self) -> Bool:
        return self.state.terminal

    fn get_score(self) -> Int:
        return Int(self.state.score)

    fn get_lives(self) -> Int:
        return Int(self.state.lives)

    fn get_frame_number(self) -> Int:
        return Int(self.state.frame_number)


# ============================================================================
# Game Definition Trait (duck-typed for now)
# ============================================================================


trait _GameDefTrait:
    @staticmethod
    fn get_score(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        ...

    @staticmethod
    fn get_reward(ram: InlineArray[UInt8, RAM_SIZE], prev_score: Int) -> Int:
        ...

    @staticmethod
    fn get_lives(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        ...

    @staticmethod
    fn is_terminal(ram: InlineArray[UInt8, RAM_SIZE]) -> Bool:
        ...

    @staticmethod
    fn map_action(action_idx: Int) -> UInt8:
        ...


# ============================================================================
# ROM Loading Utility
# ============================================================================


struct RomData(Movable):
    """Holds ROM data loaded from a file."""

    var data: UnsafePointer[UInt8, MutAnyOrigin]
    var size: Int

    fn __init__(out self):
        self.data = UnsafePointer[UInt8, MutAnyOrigin]()
        self.size = 0

    fn __init__(out self, *, deinit take: Self):
        self.data = take.data
        self.size = take.size


fn load_rom(path: String) raises -> RomData:
    """Load a ROM file from disk.

    Atari 2600 ROMs are typically 2K, 4K, 8K, or 16K binary files.
    """
    var result = RomData()

    # Read file using Mojo's file I/O
    with open(path, "r") as f:
        var content = f.read_bytes()
        result.size = len(content)
        result.data = alloc[UInt8](result.size)
        for i in range(result.size):
            result.data[i] = content[i]

    return result^
