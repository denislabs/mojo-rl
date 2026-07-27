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
from mojo_rl.nn.core.ptr import untracked
from .cpu6502 import cpu_reset, run_frame, mem_read
from .cartridge import init_bank
from .riot import set_action
from .flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_RESET,
    FLAG_CON_SELECT,
    FLAG_SWAP_PORTS,
    FLAG_PADDLES,
    ROM_AUTO,
    TOTAL_SCANLINES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from .games.registry import AtariGame, game_signals


struct AtariEnvironment(Movable):
    """High-level Atari 2600 environment for RL.

    Manages the emulator state and provides step/reset interface.
    Frame skip and action repeat are handled at this level.
    """

    var state: AtariState
    var rom: UnsafePointer[UInt8, MutUntrackedOrigin]
    var rom_size: Int
    var frame_skip: Int
    var max_frames: Int  # Max frames per episode (0 = unlimited)
    var mapper: UInt8  # ROM_* mapper (ROM_AUTO = resolve from size)
    var swap_ports: Bool  # Player 1 on the RIGHT joystick port (WoW)
    var paddles: Bool  # Paddle cart (ALE PADDLES controller mapping)
    # True iff the last step's terminal came from the GAME (game over /
    # lives exhausted), BEFORE the max_frames overlay. Distinguishes natural
    # termination (drop the TD bootstrap) from time-limit truncation (keep
    # it); surfaced through `AtariEnv.was_terminated`.
    var natural_terminal: Bool

    def __init__(
        out self,
        rom: UnsafePointer[UInt8, MutAnyOrigin],
        rom_size: Int,
        frame_skip: Int = 4,
        max_frames: Int = 108000,  # Standard ALE default (~30 min at 60fps)
        mapper: UInt8 = ROM_AUTO,
        swap_ports: Bool = False,
        paddles: Bool = False,
    ):
        self.state = AtariState()
        self.rom = untracked(rom)
        self.rom_size = rom_size
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.mapper = mapper
        self.swap_ports = swap_ports
        self.paddles = paddles
        self.natural_terminal = False

    def __init__(out self, *, deinit move: Self):
        self.state = move.state^
        self.rom = move.rom
        self.rom_size = move.rom_size
        self.frame_skip = move.frame_skip
        self.max_frames = move.max_frames
        self.mapper = move.mapper
        self.swap_ports = move.swap_ports
        self.paddles = move.paddles
        self.natural_terminal = move.natural_terminal

    def reset(mut self):
        """Reset the environment to initial state."""
        self.state = AtariState()
        self.natural_terminal = False
        if self.swap_ports:
            self.state.sys_flags = self.state.sys_flags | FLAG_SWAP_PORTS
        if self.paddles:
            self.state.sys_flags = self.state.sys_flags | FLAG_PADDLES
        init_bank(self.state, self.rom_size, self.mapper)
        cpu_reset(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # Run a few frames to get past the title screen / initialization
        # (Most games need ~60 frames of NOOP to start)
        for _ in range(60):
            set_action(self.state, ACTION_NOOP)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # Hold the console RESET switch to actually start the game. A single
        # frame is too short for some games (e.g. Space Invaders), which then
        # stay in attract/demo mode — showing a colored background and a
        # garbage demo score. Holding RESET for several frames matches a real
        # button press and reliably starts gameplay.
        for _ in range(10):
            set_action(self.state, ACTION_RESET)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # A few more NOOP frames to let the game settle into play.
        for _ in range(10):
            set_action(self.state, ACTION_NOOP)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # Initialize RL state
        self.state.reward = 0
        self.state.score = 0
        self.state.terminal = False
        self.state.started = True

    def reset_game(mut self, game: AtariGame):
        """Reset() plus the game's starting actions (ALE getStartingActions):
        leaves the title screen for games where console RESET alone is not
        enough (Asterix/Enduro FIRE, BeamRider RIGHT, DarkChambers' 486-frame
        boot animation, ElevatorAction 16×FIRE)."""
        self.reset()
        # Console-SELECT game-mode selection (ALE setMode default path):
        # press SELECT 2 frames on / 2 off until the mode byte matches,
        # then soft-reset (hold RESET) to apply.
        var su = game.select_until()
        if su[0] >= 0:
            var guard = 0
            while Int(self.state.ram[su[0] & 0x7F]) != su[1] and guard < 100:
                self.state.sys_flags = self.state.sys_flags | FLAG_CON_SELECT
                for _ in range(su[2]):
                    set_action(self.state, ACTION_NOOP)
                    run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
                self.state.sys_flags = self.state.sys_flags & ~FLAG_CON_SELECT
                for _ in range(2):
                    set_action(self.state, ACTION_NOOP)
                    run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
                guard += 1
            for _ in range(4):
                set_action(self.state, ACTION_RESET)
                run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
            for _ in range(4):
                set_action(self.state, ACTION_NOOP)
                run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
        var sa = game.starting_actions()
        for _ in range(sa[1]):
            set_action(self.state, sa[0])
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
        for _ in range(sa[3]):
            set_action(self.state, sa[2])
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
        for _ in range(sa[5]):
            set_action(self.state, sa[4])
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
        # FIRE-mash start (Mario Bros): press FIRE until the byte latches.
        var fu = game.fire_until()
        if fu >= 0:
            var tries = 0
            while Int(self.state.ram[fu & 0x7F]) == 0 and tries < 30:
                for _ in range(2):
                    set_action(self.state, ACTION_FIRE)
                    run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
                for _ in range(28):
                    set_action(self.state, ACTION_NOOP)
                    run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)
                tries += 1
        # Sync the RL signals to the post-reset RAM, like ALE (settings step
        # during reset, rewards not exposed). Without this, games whose score
        # doesn't start at 0 leak a bogus first-step reward (Pitfall starts
        # at 2000; Skiing's timer baseline).
        var sig = game_signals(game, self.state, 0)
        self.state.score = Int32(sig.score)
        self.state.lives = UInt8(sig.lives)
        self.state.reward = 0
        self.state.terminal = False
        self.natural_terminal = False

    def step(mut self, action: UInt8) -> Int:
        """Execute one step (frame_skip frames with the same action).

        Returns the cumulative reward over the skipped frames.
        """
        var total_reward: Int = 0
        var prev_score = Int(self.state.score)

        for frame in range(self.frame_skip):
            set_action(self.state, action)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # Return score delta as reward (game-specific extraction happens externally)
        return total_reward

    def step_with_game[GAME: GameDef](mut self, action_idx: Int) -> Int:
        """Execute one step using a game definition for reward/terminal extraction.

        This is the preferred interface — it handles:
        - Action mapping (agent index → ALE action)
        - Frame skipping
        - Score/reward/lives/terminal extraction from RAM
        """
        var ale_action = GAME.map_action(action_idx)
        var prev_score = Int(self.state.score)

        for frame in range(self.frame_skip):
            set_action(self.state, ale_action)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        # Extract RL signals from RAM
        var new_score = GAME.get_score(self.state.ram)
        var reward = new_score - prev_score
        self.state.score = Int32(new_score)
        self.state.reward = Int32(reward)
        self.state.lives = UInt8(GAME.get_lives(self.state.ram))
        self.state.terminal = GAME.is_terminal(self.state.ram)
        self.natural_terminal = self.state.terminal

        # Check max frames truncation
        if (
            self.max_frames > 0
            and Int(self.state.frame_number) >= self.max_frames
        ):
            self.state.terminal = True

        return reward

    def step_game(mut self, game: AtariGame, action_idx: Int) -> Int:
        """Runtime-game variant of step_with_game: one env binary, any game.

        Maps the agent's action index through the game's minimal action set
        (registry, ALE ordering), runs frame_skip frames, then extracts
        score/reward/lives/terminal from RAM via game_signals.
        """
        var ale_action = game.action(action_idx)
        var prev_score = Int(self.state.score)

        for _ in range(self.frame_skip):
            set_action(self.state, ale_action)
            run_frame(self.state, self.rom.as_unsafe_any_origin(), self.rom_size)

        var sig = game_signals(game, self.state, prev_score)
        self.state.score = Int32(sig.score)
        self.state.reward = Int32(sig.reward)
        self.state.lives = UInt8(sig.lives)
        self.state.terminal = sig.terminal
        self.natural_terminal = sig.terminal

        # Check max frames truncation
        if (
            self.max_frames > 0
            and Int(self.state.frame_number) >= self.max_frames
        ):
            self.state.terminal = True

        return sig.reward

    def get_ram(self) -> InlineArray[UInt8, RAM_SIZE]:
        """Get a copy of the 128-byte RAM (for RAM observations)."""
        return self.state.ram.copy()

    def is_terminal(self) -> Bool:
        return self.state.terminal

    def get_score(self) -> Int:
        return Int(self.state.score)

    def get_lives(self) -> Int:
        return Int(self.state.lives)

    def get_frame_number(self) -> Int:
        return Int(self.state.frame_number)


# ============================================================================
# Game Definition Trait (duck-typed for now)
# ============================================================================


trait GameDef:
    comptime NUM_ACTIONS: Int

    @staticmethod
    def get_score(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        ...

    @staticmethod
    def get_reward(ram: InlineArray[UInt8, RAM_SIZE], prev_score: Int) -> Int:
        ...

    @staticmethod
    def get_lives(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        ...

    @staticmethod
    def is_terminal(ram: InlineArray[UInt8, RAM_SIZE]) -> Bool:
        ...

    @staticmethod
    def map_action(action_idx: Int) -> UInt8:
        ...


# ============================================================================
# ROM Loading Utility
# ============================================================================


struct RomData(Movable):
    """Holds ROM data loaded from a file."""

    var data: Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]
    var size: Int

    def __init__(out self):
        self.data = None
        self.size = 0

    def __init__(out self, *, deinit move: Self):
        self.data = move.data
        self.size = move.size


def load_rom(path: String) raises -> RomData:
    """Load a ROM file from disk.

    Atari 2600 ROMs are typically 2K, 4K, 8K, or 16K binary files.
    """
    var result = RomData()

    # Read file using Mojo's file I/O
    with open(path, "r") as f:
        var content = f.read_bytes()
        result.size = len(content)
        result.data = alloc[UInt8](result.size)
        var raw = result.data.value()
        for i in range(result.size):
            raw[i] = content[i]

    return result^
