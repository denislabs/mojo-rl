"""Procgen heist — render a level to PNG for visual inspection.

Resets heist at a fixed seed and writes a high-resolution human-play frame.
Confirms the visual-approx renderer produces a recognisable level: dirt-block
maze walls, colored locked doors, colored keys, the gem exit, and the astronaut
player. See `docs/PROCGEN_HEIST_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/heist_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import HeistGame, HeistAssets
from mojo_rl.envs.procgen.games.heist import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime OUT = String("procgen_heist_seed0.png")


def main() raises:
    var assets = HeistAssets(ASSET_ROOT)
    var game = HeistGame(DIST_HARD)  # 13×13 world, more doors/keys to see
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
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for heist seed", SEED)
