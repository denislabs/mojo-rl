"""Procgen leaper — render a level to PNG for visual inspection.

Resets leaper (Hard, more lanes) and writes a high-resolution frame. Confirms the
visual-approx renderer: grass start row, road lanes with cars, water lanes with
logs, the finish line at the top, and the frog at the bottom. See
`docs/PROCGEN_LEAPER_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/leaper_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import LeaperGame, LeaperAssets
from mojo_rl.envs.procgen.games.leaper import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime OUT = String("procgen_leaper_seed0.png")


def main() raises:
    var assets = LeaperAssets(ASSET_ROOT)
    var game = LeaperGame(DIST_HARD)
    game.reset(SEED)
    var frame = game.render(assets, OUT_RES)

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for leaper seed", SEED)
