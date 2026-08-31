"""Procgen fruitbot — render a level to PNG for visual inspection.

Resets fruitbot (Hard) and steps a while so enemy waves fly in + the ship fires,
then writes a high-resolution frame: space background, the player ship, enemy
flyers/meteors/turrets, bullets, and explosions. See `docs/PROCGEN_STARPILOT_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/fruitbot_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import FruitbotGame, FruitbotAssets
from mojo_rl.envs.procgen.games.fruitbot import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime WARM_STEPS = 70
comptime OUT = String("procgen_fruitbot_seed7.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [4, 1, 4, 7, 9, 4, 1, 7, 4]
    return t[step % len(t)]


def main() raises:
    var assets = FruitbotAssets(ASSET_ROOT)
    var game = FruitbotGame(DIST_HARD)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
    var frame = game.render(assets, OUT_RES)

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for fruitbot seed", SEED)
