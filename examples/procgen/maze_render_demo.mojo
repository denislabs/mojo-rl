"""Procgen maze — render a level to PNG for visual inspection.

Resets the maze at a fixed seed and writes a high-resolution human-play frame
(`render(OUT_RES)`, not the tiny 64×64 training obs) to a PNG. Confirms the
visual-approx rasterizer produces a recognisable maze with crisp sprites. See
`docs/PROCGEN_PORT.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/maze_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import MazeGame

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime OUT = String("procgen_maze_seed7.png")


def main() raises:
    var game = MazeGame(ASSET_ROOT)
    game.reset(SEED)
    var frame = game.render(OUT_RES)  # OUT_RES*OUT_RES*3 RGB, row-major

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
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for maze seed", SEED)
