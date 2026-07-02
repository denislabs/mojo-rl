"""Procgen chaser — render a level to PNG for visual inspection.

Resets chaser at a fixed seed, steps a few frames so eggs/enemies appear, and
writes a high-resolution human-play frame (`render(OUT_RES)`, not the 64×64
training obs) to a PNG. Confirms the visual-approx rasterizer produces a
recognisable level: stone-block maze walls, green orbs filling the corridors, the
floating player, and enemy eggs/enemies. See `docs/PROCGEN_CHASER_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/chaser_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import ChaserGame, ChaserAssets, DIST_EASY

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 7
comptime OUT_RES = 512
comptime WARM_STEPS = 60  # let eggs hatch into enemies so they show up
comptime OUT = String("procgen_chaser_seed7.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 5, 5, 1, 1, 3, 3, 8, 6, 2, 0, 4]
    return t[step % len(t)]


def main() raises:
    var assets = ChaserAssets(ASSET_ROOT)
    var game = ChaserGame(DIST_EASY)
    game.reset(SEED)
    for s in range(WARM_STEPS):
        _ = game.step(_tape(s))
    var frame = game.render(assets, OUT_RES)  # OUT_RES² RGB, row-major

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
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for chaser seed", SEED)
