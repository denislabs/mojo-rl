"""Procgen coinrun — render a level to PNG for visual inspection.

Resets coinrun (Hard) and runs right + jumps a while, then writes a high-resolution
frame: platform background, themed ground blocks, the coin, saws/enemies/crates,
lava, and the alien player. See `docs/PROCGEN_COINRUN_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . examples/procgen/coinrun_render_demo.mojo
"""

from std.python import Python
from mojo_rl.envs.procgen.games import CoinrunGame, CoinrunAssets
from mojo_rl.envs.procgen.games.coinrun import DIST_HARD

comptime ASSET_ROOT = String("assets/procgen/")
comptime SEED = 0
comptime OUT_RES = 512
comptime WARM_STEPS = 12
comptime OUT = String("procgen_coinrun_seed0.png")


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 8, 7, 7, 5, 7]
    return t[step % len(t)]


def main() raises:
    var assets = CoinrunAssets(ASSET_ROOT)
    var game = CoinrunGame(DIST_HARD)
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
    print("wrote", OUT, "(", OUT_RES, "x", OUT_RES, ") for coinrun seed", SEED)
