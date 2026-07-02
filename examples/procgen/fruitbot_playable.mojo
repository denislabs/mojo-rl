"""Playable Procgen Fruitbot — steer the auto-scrolling robot; collect fruit.

Run:
  pixi run mojo run -I . examples/procgen/fruitbot_playable.mojo

Controls:
  Left/Right   Steer (the robot always drifts upward)
  Space        Fire (shoot a lock to open its door)
  Close window / Esc  Quit

Fly up through the wall gaps, collect fruit (+1), avoid food (-4), and reach the
presents at the top (+10). Crashing into a wall or closed door ends the run.
See `docs/PROCGEN_FRUITBOT_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import FruitbotEnv
from mojo_rl.envs.procgen.games.fruitbot import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Fruitbot ===")
    print("Left/Right = steer    Space = fire    Close = quit")
    print("Fly up through gaps, collect fruit, avoid food, reach the presents.")

    var env = FruitbotEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Fruitbot"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var action = 4
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 9  # fire
        elif keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 7

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Level", episode, "complete! reached the presents (seed", env.current_level_seed, ")")
            else:
                print("Crashed! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)

    numkeys.free()
    window.close()
    print("=== Done ===")
