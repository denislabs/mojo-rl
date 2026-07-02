"""Playable Procgen Coinrun — run right + jump to the coin.

Run:
  pixi run mojo run -I . examples/procgen/coinrun_playable.mojo

Controls:
  Left/Right   Run
  Space / Up   Jump (only when standing on ground/crate)
  Close window / Esc  Quit

Run right along the ground, jump gaps + over saws/enemies/lava, and grab the gold
coin at the far right (+10). One touch of an enemy or saw, falling in lava, or
going off-screen ends the run. See `docs/PROCGEN_COINRUN_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import CoinrunEnv
from mojo_rl.envs.procgen.games.coinrun import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Coinrun ===")
    print("Left/Right = run    Space/Up = jump    Close = quit")
    print("Reach the gold coin on the right; avoid saws, enemies, lava + gaps.")

    var env = CoinrunEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Coinrun"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var jump = (
            keys[Int(Scancode.SCANCODE_SPACE)] or keys[Int(Scancode.SCANCODE_UP)]
        )
        var right = keys[Int(Scancode.SCANCODE_RIGHT)]
        var left = keys[Int(Scancode.SCANCODE_LEFT)]
        # move id: vx = move//3-1, vy = move%3-1.
        var vx = 1 if right else (-1 if left else 0)  # -1 / 0 / +1
        var vy = 1 if jump else 0
        var action = (vx + 1) * 3 + (vy + 1)

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Got the coin! (seed", env.current_level_seed, ") — next level…")
            else:
                print("You died! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)

    numkeys.free()
    window.close()
    print("=== Done ===")
