"""Headless Space Invaders laser-accuracy probe.

The user reported the laser "goes through invaders" and is hard to land. This
drives SI without a window: it fires on a cooldown and sweeps the cannon left/
right, counting score increases (kills) over many frames. With Stella-style
per-color-clock collision the laser should latch missile-vs-player at the exact
beam position it occupied, so kills-per-shot should be healthy rather than the
laser tunnelling past invaders.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_si_laser.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import run_frame_with_video
from mojo_rl.envs.atari.games.space_invaders import SpaceInvadersDef
from mojo_rl.envs.atari.flags import (
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_LEFTFIRE,
    ACTION_RIGHTFIRE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)
from mojo_rl.envs.atari.riot import set_action
from std.memory import alloc


def main() raises:
    var rom_path = "roms/space_invaders.bin"
    print("Loading ROM: " + rom_path)
    var rom_data = load_rom(rom_path)
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()

    var buf = alloc[UInt8](FRAME_WIDTH * FRAME_HEIGHT * 4)

    var step = 0
    comptime MAX_STEPS = 40000
    var prev_score = 0
    var kills = 0
    var shots = 0
    var total_points = 0  # sum of positive score deltas across all games
    var prev_lives = SpaceInvadersDef.get_lives(env.state.ram)

    while step < MAX_STEPS:
        # Sweep the cannon back and forth and fire continuously. A continuous
        # fire + sweep covers every column, so a correct laser should rack up
        # kills; a tunnelling laser would leave the score nearly flat.
        var phase = (step // 60) % 4
        var act = ACTION_FIRE
        if phase == 0:
            act = ACTION_RIGHTFIRE
        elif phase == 1:
            act = ACTION_LEFTFIRE
        elif phase == 2:
            act = ACTION_RIGHTFIRE
        else:
            act = ACTION_LEFTFIRE

        set_action(env.state, act)
        run_frame_with_video(env.state, env.rom, env.rom_size, buf)
        step += 1
        shots += 1

        var cur_score = SpaceInvadersDef.get_score(env.state.ram)
        if cur_score > prev_score:
            kills += 1
            total_points += cur_score - prev_score
        prev_score = cur_score

        var lives = SpaceInvadersDef.get_lives(env.state.ram)
        prev_lives = lives

        if SpaceInvadersDef.is_terminal(env.state.ram):
            env.reset()
            prev_score = 0
            prev_lives = SpaceInvadersDef.get_lives(env.state.ram)

        if step % 5000 == 0:
            print(
                "step="
                + String(step)
                + " score="
                + String(cur_score)
                + " lives="
                + String(lives)
                + " score_events="
                + String(kills)
            )

    print(
        "DONE. steps="
        + String(step)
        + " final_score="
        + String(SpaceInvadersDef.get_score(env.state.ram))
        + " score_events="
        + String(kills)
        + " total_points="
        + String(total_points)
    )
    buf.free()
