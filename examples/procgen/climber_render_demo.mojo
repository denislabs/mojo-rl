"""Procgen climber — render a level to PNG for visual inspection.

Run: pixi run mojo run -I . examples/procgen/climber_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import ClimberGame, ClimberAssets
from mojo_rl.envs.procgen.games.climber import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime WARM_STEPS = 10
comptime OUT = String("procgen_climber_seed0.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [8, 8, 7, 5, 8]
    return t[step % len(t)]


def main() raises:
    var assets = ClimberAssets(ASSET_ROOT)
    var game = ClimberGame(DIST_HARD)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
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
    print("wrote", OUT, "for climber seed", SEED)
