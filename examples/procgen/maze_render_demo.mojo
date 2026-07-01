"""Procgen maze spike — render a level to PNG for visual inspection.

Resets the maze at a fixed seed and writes the 64×64 observation (nearest-
upscaled ×8 to 512 for visibility) to a PNG via PIL. Confirms the visual-approx
rasterizer produces a recognisable maze. See `docs/PROCGEN_PORT.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/maze_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import MazeSpikeGame

comptime ASSET_ROOT = String("references/procgen-master/procgen/data/assets/")
comptime SEED = 7
comptime UP = 8  # nearest upscale for the saved PNG
comptime OUT = String("procgen_maze_seed7.png")


def main() raises:
    var game = MazeSpikeGame(ASSET_ROOT)
    game.reset(SEED)
    var obs = game.render()  # 64*64*3 RGB, row-major

    var pil = Python.import_module("PIL.Image")
    var big = 64 * UP
    var img = pil.new("RGB", Python.tuple(big, big))
    var px = img.load()
    for oy in range(64):
        for ox in range(64):
            var off = (oy * 64 + ox) * 3
            var color = Python.tuple(
                Int(obs[off + 0]), Int(obs[off + 1]), Int(obs[off + 2])
            )
            for dy in range(UP):
                for dx in range(UP):
                    px[Python.tuple(ox * UP + dx, oy * UP + dy)] = color
    img.save(OUT)
    print("wrote", OUT, "for maze seed", SEED)
