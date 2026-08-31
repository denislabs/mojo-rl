"""Procgen caveflyer — render a level to PNG for visual inspection.

Resets caveflyer (Hard) and thrusts around the cave a while, then writes a
high-resolution frame: space background, carved cave walls, the green UFO goal,
meteors/target ships/enemies, bullets, exhaust, and the player ship.
See `docs/PROCGEN_CAVEFLYER_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/caveflyer_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import CaveflyerGame, CaveflyerAssets
from mojo_rl.envs.procgen.games.caveflyer import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime WARM_STEPS = 16
comptime OUT = String("procgen_caveflyer_seed0.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [5, 5, 8, 5, 2, 5, 7]
    return t[step % len(t)]


def main() raises:
    var assets = CaveflyerAssets(ASSET_ROOT)
    var game = CaveflyerGame(DIST_HARD)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
    var frame = game.render(assets, OUT_RES)

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for caveflyer seed", SEED)
