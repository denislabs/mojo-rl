"""Diag: berzerk scores 0 under random play — headless vs video divergence?

Plays the same deterministic random action stream through (a) the headless
run_frame path and (b) the run_frame_video path, comparing score
trajectories. If only (b) scores, the headless collision path is at fault
(Phase-1 run_frame probe pattern); if both stay 0, it's a game-start or
action-application issue.
"""

from std.memory import alloc

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame, game_signals
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import run_frame, run_frame_video


def run_path(video: Bool) raises -> Int:
    var game = AtariGame.BERZERK
    var rom_data = load_rom(game.rom_file())
    var env = AtariEnvironment(
        rom_data.data.value(),
        rom_data.size,
        frame_skip=4,
        max_frames=0,
        mapper=game.mapper(),
    )
    var buf = alloc[UInt8](160 * 210 * 4)
    env.reset_game(game)

    var rng: UInt64 = 99991
    var total = 0
    var terminals = 0
    var n = game.num_actions()
    for step in range(2500):
        rng = rng * 6364136223846793005 + 1442695040888963407
        var idx = Int((rng >> 33) % UInt64(n))
        var ale_action = game.action(idx)
        var prev_score = Int(env.state.score)
        for _ in range(4):
            set_action(env.state, ale_action)
            if video:
                run_frame_video(env.state, env.rom, env.rom_size, buf)
            else:
                run_frame(env.state, env.rom, env.rom_size)
        var sig = game_signals(game, env.state, prev_score)
        env.state.score = Int32(sig.score)
        env.state.lives = UInt8(sig.lives)
        if sig.reward > 0:
            total += sig.reward
            print(
                ("video" if video else "headless")
                + " step "
                + String(step)
                + ": +"
                + String(sig.reward)
                + " (score "
                + String(sig.score)
                + ")"
            )
        if sig.terminal:
            terminals += 1
            env.reset_game(game)
    print(
        ("video" if video else "headless")
        + " TOTAL="
        + String(total)
        + " terminals="
        + String(terminals)
    )
    return total


def main() raises:
    _ = run_path(False)
    _ = run_path(True)
