"""Diag: does a game actually enter gameplay after our reset sequence?

Boots a game, mirrors AtariEnvironment.reset()'s start sequence (60 NOOP,
10 RESET, 10 NOOP), then plays random actions, dumping the game's key RAM
bytes every 60 frames. Used to root-cause games that sit in attract/demo
mode (score frozen at 0, or terminal loops) under the smoke harness.

Usage: diag_atari_game_start <game> [hold_fire]
"""

from std.sys import argv

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame, game_signals
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import run_frame
from mojo_rl.envs.atari.flags import ACTION_NOOP, ACTION_FIRE, ACTION_RESET


def dump(mut env: AtariEnvironment, game: AtariGame, label: String):
    var s = label + "  ram["
    # Bytes of interest for the currently suspect games (harmless extras
    # for others): berzerk lives 0xDA score 93-95; asterix lives 0xD3
    # death 0xC7 score 0xDE-0xE0.
    for addr in [0xDA, 93, 94, 95, 0xD3, 0xC7, 0xDE, 0xDF, 0xE0]:
        s += hex(addr) + "=" + hex(Int(env.state.ram[addr & 0x7F])) + " "
    var sig = game_signals(game, env.state, Int(env.state.score))
    s += (
        "]  score="
        + String(sig.score)
        + " lives="
        + String(sig.lives)
        + " term="
        + String(sig.terminal)
    )
    print(s)


def main() raises:
    if len(argv()) < 2:
        print("usage: diag_atari_game_start <game> [hold_fire]")
        return
    var game = AtariGame.from_name(String(argv()[1]))
    var hold_fire = len(argv()) > 2

    var rom_data = load_rom(game.rom_file())
    var env = AtariEnvironment(
        rom_data.data.value(),
        rom_data.size,
        frame_skip=1,
        max_frames=0,
        mapper=game.mapper(),
    )
    # Manual boot (NOT env.reset()) so we can observe each phase.
    from mojo_rl.envs.atari.atari_state import AtariState
    from mojo_rl.envs.atari.cartridge import init_bank
    from mojo_rl.envs.atari.cpu6502 import cpu_reset

    env.state = AtariState()
    init_bank(env.state, env.rom_size, game.mapper())
    cpu_reset(env.state, env.rom, env.rom_size)

    print("== phase A: 60 NOOP frames after power-on")
    for i in range(60):
        set_action(env.state, ACTION_NOOP)
        run_frame(env.state, env.rom, env.rom_size)
        if i % 20 == 19:
            dump(env, game, "A+" + String(i + 1))

    print("== phase B: hold console RESET 10 frames")
    for _ in range(10):
        set_action(env.state, ACTION_RESET)
        run_frame(env.state, env.rom, env.rom_size)
    dump(env, game, "B")

    print("== phase C: 1800 frames of play (FIRE-heavy random)")
    var rng: UInt64 = 12345
    for i in range(1800):
        rng = rng * 6364136223846793005 + 1442695040888963407
        var act = ACTION_NOOP
        if hold_fire or (rng >> 33) % 3 == 0:
            act = ACTION_FIRE
        set_action(env.state, act)
        run_frame(env.state, env.rom, env.rom_size)
        if i % 180 == 179:
            dump(env, game, "C+" + String(i + 1))
