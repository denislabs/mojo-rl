"""Procgen maze — render a level to PNG for visual inspection.

Resets the maze at a fixed seed and writes a high-resolution human-play frame
(`render(OUT_RES)`, not the tiny 64×64 training obs) to a PNG. Confirms the
visual-approx rasterizer produces a recognisable maze with crisp sprites. See
`docs/PROCGEN_PORT.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/maze_render_demo.mojo
"""

from mojo_rl.io.png import save_png
from mojo_rl.envs.procgen.games import MazeGame

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime OUT = String("procgen_maze_seed7.png")


def main() raises:
    var game = MazeGame(ASSET_ROOT)
    game.reset(SEED)
    var frame = game.render(OUT_RES)  # OUT_RES*OUT_RES*3 RGB, row-major

    # ⚠ The old path built the image PIXEL BY PIXEL through the Python
    # interpreter — OUT_RES x OUT_RES round trips per frame — because that was the
    # only writer available. `io/png.save_png` takes the buffer whole.
    save_png(String(OUT), frame, OUT_RES, OUT_RES, 3)
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for maze seed", SEED)
