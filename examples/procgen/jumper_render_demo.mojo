"""Procgen jumper — render a level to PNG for visual inspection.

Run: pixi run mojo run -I . examples/procgen/jumper_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import JumperGame, JumperAssets
from mojo_rl.envs.procgen.games.jumper import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime WARM_STEPS = 10
comptime OUT = String("procgen_jumper_seed0.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [8, 8, 7, 5, 8]
    return t[step % len(t)]


def main() raises:
    var assets = JumperAssets(ASSET_ROOT)
    var game = JumperGame(DIST_HARD)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
    var frame = game.render(assets, OUT_RES)
    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "for jumper seed", SEED)
