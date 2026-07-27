"""Playable Procgen Starpilot — fly and shoot down the enemy waves.

Run:
  pixi run mojo run -I . examples/procgen/starpilot_playable.mojo

Controls:
  Arrow keys   Move the ship
  Space        Fire right      Z  Fire left
  Close window / Esc  Quit

Shoot down enemies (+1 each) and survive; a finish line appears late in the level —
touch it to win. You can move OR fire on a given frame (firing holds position).
`num_levels=0` draws from the full level set. See `docs/PROCGEN_STARPILOT_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import StarpilotEnv
from mojo_rl.envs.procgen.games.starpilot import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Starpilot ===")
    print("Arrows = move    Space = fire right    Z = fire left    Close = quit")
    print("Shoot down enemies; reach the finish to win.")

    var env = StarpilotEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Starpilot"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # Fire takes priority (action >= 9 forces stand); else move.
        var action = 4
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 9  # fire right (special_action 1)
        elif keys[Int(Scancode.SCANCODE_Z)]:
            action = 10  # fire left (special_action 2)
        else:
            var vx = 0
            var vy = 0
            if keys[Int(Scancode.SCANCODE_UP)]:
                vy = 1
            elif keys[Int(Scancode.SCANCODE_DOWN)]:
                vy = -1
            if keys[Int(Scancode.SCANCODE_LEFT)]:
                vx = -1
            elif keys[Int(Scancode.SCANCODE_RIGHT)]:
                vx = 1
            action = (vx + 1) * 3 + (vy + 1)

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Level", episode, "complete! reached the finish (seed", env.current_level_seed, ")")
            else:
                print("Destroyed! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)

        window.delay(66)  # ~15 fps

    numkeys.free()
    window.close()
    print("=== Done ===")
