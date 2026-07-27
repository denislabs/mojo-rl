"""Playable Procgen Plunder — sail the cannon ship, sink the target ships.

Run:
  pixi run mojo run -I . examples/procgen/plunder_playable.mojo

Controls:
  Left/Right   Move the cannon ship
  Space        Fire a cannonball (upward)
  Close window / Esc  Quit

Shoot the ships matching the target shown in the corner legend (+1 each, refills
juice); shooting a decoy costs juice. Juice drains over time — if it empties you
lose. Sink 20 targets to win. See `docs/PROCGEN_PLUNDER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import PlunderEnv
from mojo_rl.envs.procgen.games.plunder import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Plunder ===")
    print("Left/Right = move    Space = fire    Close = quit")
    print("Sink the ships matching the corner legend; avoid decoys; watch your juice.")

    var env = PlunderEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Plunder"), fps=15)
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
                print("Level", episode, "complete! sank 20 targets (seed", env.current_level_seed, ")")
            else:
                print("Out of juice! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)

    numkeys.free()
    window.close()
    print("=== Done ===")
