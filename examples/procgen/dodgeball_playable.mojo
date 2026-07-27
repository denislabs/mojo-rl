"""Playable Procgen Dodgeball — dodge, throw, clear the arena, reach the door.

Run: pixi run mojo run -I . examples/procgen/dodgeball_playable.mojo

Controls:
  Arrows      Move        Space  Throw a ball (in your last-moved direction)
  Close/Esc   Quit

Throw balls to kill every enemy (+2 each), then reach the door (+10, turns green
when the arena is clear). Death on touching an enemy, an enemy ball, or a lava wall.
See `docs/PROCGEN_DODGEBALL_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import DodgeballEnv
from mojo_rl.envs.procgen.games.dodgeball import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Dodgeball ===")
    print("Arrows = move    Space = throw ball    Close = quit")
    print("Kill every enemy with balls, then reach the (green) door; avoid balls + lava.")
    var env = DodgeballEnv(ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD)
    var window = PixelWindow(WIN, WIN, String("Procgen Dodgeball"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)
    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1
    while window.is_open():
        window.show(frame, WIN, WIN)
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var action: Int
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 9  # throw
        else:
            var vx = 1 if keys[Int(Scancode.SCANCODE_RIGHT)] else (
                -1 if keys[Int(Scancode.SCANCODE_LEFT)] else 0
            )
            var vy = 1 if keys[Int(Scancode.SCANCODE_UP)] else (
                -1 if keys[Int(Scancode.SCANCODE_DOWN)] else 0
            )
            action = (vx + 1) * 3 + (vy + 1)
        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Arena cleared + door reached! (seed", env.current_level_seed, ")")
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
