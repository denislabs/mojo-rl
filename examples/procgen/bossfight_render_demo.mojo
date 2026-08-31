"""Procgen bossfight — render a level to PNG for visual inspection.

Resets bossfight (Hard) and steps a while so the boss fires + the ship shoots,
then writes a high-resolution frame: space background, the boss + shields, the
player ship, laser bullets/trails, meteor barriers, and explosions.
See `docs/PROCGEN_BOSSFIGHT_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/bossfight_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import BossfightGame, BossfightAssets
from mojo_rl.envs.procgen.games.bossfight import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime WARM_STEPS = 40
comptime OUT = String("procgen_bossfight_seed7.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [9, 4, 7, 9, 1, 9, 5, 9, 3]
    return t[step % len(t)]


def main() raises:
    var assets = BossfightAssets(ASSET_ROOT)
    var game = BossfightGame(DIST_HARD)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
    var frame = game.render(assets, OUT_RES)

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for bossfight seed", SEED)
