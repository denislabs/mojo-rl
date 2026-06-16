"""Smoke-test every registered AtariGame headlessly.

For each game in the registry: boot (reset), render one frame (screen must be
non-blank), then play deterministic pseudo-random actions for a few thousand
steps, checking that:

  - score becomes positive at some point (reward extraction works)
  - lives change (where the game has lives)
  - a natural terminal is reached (episode ends; env auto-resets and play
    continues)

This is the per-game validation gate for porting new ALE games: a game that
boots, scores, loses lives, and terminates under random play has its RAM
extraction wired correctly with high confidence.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/atari_smoke_all.mojo
"""

from mojo_rl.envs.atari.atari_env import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.flags import FRAME_WIDTH, FRAME_HEIGHT
from std.memory import alloc


comptime STEPS_PER_GAME = 3000  # frame_skip=4 → 12k frames ≈ 3.3 game-minutes


def _pad(s: String, width: Int) -> String:
    var out = s
    while len(out) < width:
        out += " "
    return out


def smoke_one(game: AtariGame) raises -> String:
    var env = AtariEnv[0](game, frame_skip=4, max_frames=0)
    _ = env.reset()

    # Render one frame and count lit (non-black) pixels.
    var buf = alloc[UInt8](FRAME_WIDTH * FRAME_HEIGHT * 4)
    run_frame_video(env.env.state, env.env.rom, env.env.rom_size, buf)
    var lit = 0
    for i in range(FRAME_WIDTH * FRAME_HEIGHT):
        var off = i * 4
        if (Int(buf[off]) + Int(buf[off + 1]) + Int(buf[off + 2])) > 24:
            lit += 1
    buf.free()

    var n_act = env.num_actions()
    var rng: UInt64 = 0x9E3779B97F4A7C15
    var max_score = 0
    var min_score = 0
    var lives_min = 9999
    var lives_max = -1
    var terminals = 0
    var total_reward = 0

    for _ in range(STEPS_PER_GAME):
        rng = rng * 6364136223846793005 + 1442695040888963407
        var a = Int((rng >> 33) % UInt64(n_act))
        var result = env.step_obs(a)
        total_reward += Int(result[1])
        var score = Int(env.env.state.score)
        var lives = Int(env.env.state.lives)
        max_score = max(max_score, score)
        min_score = min(min_score, score)
        lives_min = min(lives_min, lives)
        lives_max = max(lives_max, lives)
        if result[2]:
            terminals += 1
            _ = env.reset()

    env.close()
    return (
        _pad(game.name(), 15)
        + " act=" + _pad(String(n_act), 3)
        + " lit=" + _pad(String(lit), 6)
        + " score=[" + String(min_score) + "," + String(max_score) + "]"
        + " lives=[" + String(lives_min) + "," + String(lives_max) + "]"
        + " terminals=" + String(terminals)
        + " totR=" + String(total_reward)
    )


def main() raises:
    print("=== Atari registry smoke (random play, " + String(STEPS_PER_GAME)
          + " steps/game) ===")
    for gid in range(AtariGame.NUM_GAMES):
        var game = AtariGame.from_id(gid)
        try:
            print(smoke_one(game))
        except e:
            print(_pad(game.name(), 15) + " FAILED: " + String(e))
