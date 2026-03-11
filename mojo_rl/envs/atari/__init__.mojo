"""Atari 2600 emulator — pure Mojo port of CuLE.

GPU-capable Atari environment for reinforcement learning.
Emulates the MOS 6502 CPU, TIA graphics chip, and RIOT timer/IO.

Usage:
    from envs.atari import AtariEnvironment, load_rom
    from envs.atari.games import PongDef

    var rom = load_rom("pong.bin")
    var env = AtariEnvironment(rom.data, rom.size)
    env.reset()
    var reward = env.step_with_game[PongDef](action_idx=0)
"""

from .atari_state import AtariState
from .environment import AtariEnvironment, RomData, load_rom
from .frame_render import render_frame_bgra, render_frame_rgb, render_frame_grayscale
from .atari_env import AtariEnv, AtariEnvState, AtariAction
from .flags import (
    ACTION_NOOP, ACTION_FIRE, ACTION_UP, ACTION_RIGHT,
    ACTION_LEFT, ACTION_DOWN,
    FRAME_WIDTH, FRAME_HEIGHT,
    OBS_WIDTH, OBS_HEIGHT,
    NUM_TOTAL_ACTIONS,
)
