"""Breakout game definition.

RAM mapping (from CuLE):
  RAM[77]: score ones/tens (BCD)
  RAM[76]: score hundreds (BCD)
  RAM[57]: lives remaining
  Terminal: lives == 0 after game has started (starts with 5 lives)

Minimal actions: NOOP, FIRE, RIGHT, LEFT (4 actions)

Ported from CuLE (BSD-3): cule/atari/games/breakout.hpp
"""

from ..flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_RIGHT,
    ACTION_LEFT,
)


struct BreakoutDef:
    """Breakout game definition."""

    comptime GAME_NAME: String = "Breakout"
    comptime NUM_ACTIONS: Int = 4
    comptime INITIAL_LIVES: Int = 5

    @staticmethod
    @always_inline
    fn get_score(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        """Decode BCD score from RAM."""
        var x = Int(ram[77])
        var y = Int(ram[76])
        return 1 * (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F)

    @staticmethod
    @always_inline
    fn get_reward(ram: InlineArray[UInt8, RAM_SIZE], prev_score: Int) -> Int:
        return BreakoutDef.get_score(ram) - prev_score

    @staticmethod
    @always_inline
    fn get_lives(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        return Int(ram[57])

    @staticmethod
    @always_inline
    fn is_terminal(ram: InlineArray[UInt8, RAM_SIZE]) -> Bool:
        """Terminal when lives reach 0 after game started."""
        # Game starts with 5 lives; terminal once lives drop to 0
        return Int(ram[57]) == 0

    @staticmethod
    @always_inline
    fn map_action(action_idx: Int) -> UInt8:
        if action_idx == 0:
            return ACTION_NOOP
        elif action_idx == 1:
            return ACTION_FIRE
        elif action_idx == 2:
            return ACTION_RIGHT
        else:
            return ACTION_LEFT
