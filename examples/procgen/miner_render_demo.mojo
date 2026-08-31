"""Procgen miner — render a level to PNG for visual inspection.

Resets miner (Hard) and writes a high-resolution frame: dirt-filled grid with
scattered blue diamonds and stone boulders, the dug-out area around the robot, and
the exit window. See `docs/PROCGEN_MINER_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/miner_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import MinerGame, MinerAssets
from mojo_rl.envs.procgen.games.miner import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime OUT = String("procgen_miner_seed0.png")


def main() raises:
    var assets = MinerAssets(ASSET_ROOT)
    var game = MinerGame(DIST_HARD)
    game.reset(SEED)
    var frame = game.render(assets, OUT_RES)

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for miner seed", SEED)
