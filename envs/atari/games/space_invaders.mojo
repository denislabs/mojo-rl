"""Space Invaders game definition.

RAM mapping (from CuLE):
  Score: BCD decoded from RAM[0xE8] (lower) and RAM[0xE6] (upper)
  Lives: RAM[0xC9]
  Terminal: RAM[0x98] & 0x80 is set, or lives == 0

Minimal actions: NOOP, LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE (6 actions)

Ported from CuLE (BSD-3): cule/atari/games/spaceinvaders.hpp
"""

from ..flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_LEFTFIRE,
    ACTION_RIGHTFIRE,
)
from ..environment import GameDef
from .helpers import get_decimal_score_2


struct SpaceInvadersDef(GameDef):
    """Space Invaders game definition."""

    comptime GAME_NAME: String = "SpaceInvaders"
    comptime NUM_ACTIONS: Int = 6
    comptime INITIAL_LIVES: Int = 3

    @staticmethod
    @always_inline
    fn get_score(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        """Decode BCD score."""
        return get_decimal_score_2(ram, 0xE8, 0xE6)

    @staticmethod
    @always_inline
    fn get_reward(ram: InlineArray[UInt8, RAM_SIZE], prev_score: Int) -> Int:
        var score = SpaceInvadersDef.get_score(ram)
        var reward = score - prev_score
        if reward < 0:
            # Score overflow (max 10000)
            reward = (10000 - prev_score) + score
        return reward

    @staticmethod
    @always_inline
    fn get_lives(ram: InlineArray[UInt8, RAM_SIZE]) -> Int:
        return Int(ram[0xC9])

    @staticmethod
    @always_inline
    fn is_terminal(ram: InlineArray[UInt8, RAM_SIZE]) -> Bool:
        var some_byte = Int(ram[0x98])
        return (some_byte & 0x80) != 0 or Int(ram[0xC9]) == 0

    @staticmethod
    @always_inline
    fn map_action(action_idx: Int) -> UInt8:
        if action_idx == 0:
            return ACTION_NOOP
        elif action_idx == 1:
            return ACTION_LEFT
        elif action_idx == 2:
            return ACTION_RIGHT
        elif action_idx == 3:
            return ACTION_FIRE
        elif action_idx == 4:
            return ACTION_LEFTFIRE
        else:
            return ACTION_RIGHTFIRE
