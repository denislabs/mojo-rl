"""Procgen bigfish — render a level to PNG for visual inspection.

Resets bigfish, steps a while so fish populate the water, and writes a
high-resolution frame. Confirms the visual-approx renderer: water background, the
player fish, and themed enemy fish of varying sizes swimming across. See
`docs/PROCGEN_BIGFISH_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/bigfish_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import BigfishGame, BigfishAssets
from mojo_rl.envs.procgen.games.bigfish import DIST_EASY

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime WARM_STEPS = 80  # let fish spawn and drift in
comptime OUT = String("procgen_bigfish_seed7.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 5, 5, 8, 6, 2, 4, 1, 3, 5]
    return t[step % len(t)]


def main() raises:
    var assets = BigfishAssets(ASSET_ROOT)
    var game = BigfishGame(DIST_EASY)
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
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for bigfish seed", SEED)
