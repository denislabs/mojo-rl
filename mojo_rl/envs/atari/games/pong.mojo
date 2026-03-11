"""Pong game definition.

RAM mapping (from CuLE):
  RAM[13] = CPU score
  RAM[14] = Player score
  Terminal: when either player reaches 21

Minimal actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE (6 actions)

Ported from CuLE (BSD-3): cule/atari/games/pong.hpp
"""

from ..flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_RIGHTFIRE,
    ACTION_LEFTFIRE,
)
from ..environment import GameDef


struct PongDef(GameDef):
    """Pong game definition."""

    comptime GAME_NAME: String = "Pong"
    comptime NUM_ACTIONS: Int = 6
    comptime INITIAL_LIVES: Int = 0  # Pong doesn't use lives

    @staticmethod
    @always_inline
    fn get_score(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        """Get current score (player score - CPU score)."""
        return Int(ram[14]) - Int(ram[13])

    @staticmethod
    @always_inline
    fn get_reward(ram: InlineArray[UInt8, RAM_SIZE], prev_score: Int) -> Int:
        """Get reward as score delta."""
        return PongDef.get_score(ram) - prev_score

    @staticmethod
    @always_inline
    fn get_lives(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        return 0  # Pong doesn't have lives

    @staticmethod
    @always_inline
    fn is_terminal(ram: InlineArray[UInt8, RAM_SIZE]) -> Bool:
        """Game over when either player reaches 21."""
        return Int(ram[13]) == 21 or Int(ram[14]) == 21

    @staticmethod
    @always_inline
    fn map_action(action_idx: Int) -> UInt8:
        """Map [0, NUM_ACTIONS) to ALE action constant."""
        if action_idx == 0:
            return ACTION_NOOP
        elif action_idx == 1:
            return ACTION_FIRE
        elif action_idx == 2:
            return ACTION_RIGHT
        elif action_idx == 3:
            return ACTION_LEFT
        elif action_idx == 4:
            return ACTION_RIGHTFIRE
        else:
            return ACTION_LEFTFIRE
