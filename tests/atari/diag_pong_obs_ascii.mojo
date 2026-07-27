"""Diagnostic: does the DreamerV3 Pong obs (OBS_MODE=3, gray-96 single frame)
actually contain the ball + paddles, or does the 160x210->96x96 downscale
destroy them? Resets Pong, steps with paddle-moving actions, then ASCII-renders
the 96x96 obs and reports bright-pixel stats by region.

Run: pixi run -e apple mojo run -I . tests/atari/diag_pong_obs_ascii.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.envs.atari import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame

comptime IMG = 96
comptime OBS = IMG * IMG


def _render(obs: List[Scalar[DT]], thr: Scalar[DT]) raises:
    # 96x96 -> 32 rows x 48 cols ascii (row stride 3, col stride 2).
    for y in range(0, IMG, 3):
        var line = String("")
        for x in range(0, IMG, 2):
            var v = obs[y * IMG + x]
            if v > thr + Scalar[DT](0.25):
                line += "#"
            elif v > thr:
                line += "+"
            elif v > Scalar[DT](0.15):
                line += "."
            else:
                line += " "
        print(line)


def main() raises:
    print("=" * 60)
    print("Pong OBS_MODE=3 (gray-96) — what the agent sees")
    print("=" * 60)
    var env = AtariEnv[3, DT](AtariGame.PONG)
    var obs = env.reset_obs_list()

    # Step ~40 times, alternating paddle-move actions (2=RIGHT/up, 3=LEFT/down)
    # so the paddle moves and the ball is in play.
    for t in range(40):
        var a = 2 if (t // 5) % 2 == 0 else 3
        var r = env.step_obs(a)
        obs = r[0].copy()

    # Stats
    var mn = obs[0]
    var mx = obs[0]
    var sm = Scalar[DT](0.0)
    for i in range(OBS):
        var v = obs[i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        sm += v
    print("obs min/mean/max =", mn, "/", sm / Scalar[DT](OBS), "/", mx)

    # Distinct intensity levels (rounded) — Pong has few (bg / border / ball / paddle).
    var lv = Scalar[DT](0.5)  # bright threshold
    var bright_total = 0
    var bright_top = 0   # rows 0..17 (score area)
    var bright_play = 0  # rows 18..90 (play area: ball + paddles)
    for y in range(IMG):
        for x in range(IMG):
            if obs[y * IMG + x] > lv:
                bright_total += 1
                if y < 18:
                    bright_top += 1
                elif y < 91:
                    bright_play += 1
    print("bright(>0.5) px: total", bright_total, " score-area", bright_top,
          " play-area", bright_play)
    print("(if play-area bright px ~= 0, the ball/paddles are NOT in the obs)")
    print("-" * 60)
    print("ASCII (row/3 x col/2;  '#'=bright '+'=mid '.'=dim ' '=dark):")
    print("-" * 60)
    _render(obs, lv)
    env.close()
    _ = env^
    print("=" * 60)
