"""Procgen miner — render a level to PNG for visual inspection.

Resets miner (Hard) and writes a high-resolution frame: dirt-filled grid with
scattered blue diamonds and stone boulders, the dug-out area around the robot, and
the exit window. See `docs/PROCGEN_MINER_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/miner_render_demo.mojo
"""

from std.python import Python
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

    var pil = Python.import_module("PIL.Image")
    var img = pil.new("RGB", Python.tuple(OUT_RES, OUT_RES))
    var px = img.load()
    for oy in range(OUT_RES):
        for ox in range(OUT_RES):
            var off = (oy * OUT_RES + ox) * 3
            px[Python.tuple(ox, oy)] = Python.tuple(
                Int(frame[off + 0]), Int(frame[off + 1]), Int(frame[off + 2])
            )
    img.save(OUT)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for miner seed", SEED)
