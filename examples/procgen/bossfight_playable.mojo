"""Playable Procgen Bossfight — dodge the boss's bullets and shoot it down.

Run:
  pixi run mojo run -I . examples/procgen/bossfight_playable.mojo

Controls:
  Arrow keys   Move the ship (all directions)
  Space        Fire (upward)
  Close window / Esc  Quit

Shoot the boss only when its shields are DOWN (+1 per health chunk cleared, +10
for the kill). While the shields are up your shots reflect. Dodge the boss's
bullet patterns and the drifting meteor barriers — one touch of the boss, a
meteor, or an enemy bullet ends the run. See `docs/PROCGEN_BOSSFIGHT_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import BossfightEnv
from mojo_rl.envs.procgen.games.bossfight import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Bossfight ===")
    print("Arrows = move    Space = fire    Close = quit")
    print("Shoot the boss only when its shields are down; dodge bullets + meteors.")

    var env = BossfightEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Bossfight"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # move ids: vx=move//3-1, vy=move%3-1 → 1=left, 7=right, 5=up, 3=down.
        var action = 4  # no move
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 9  # fire (no move)
        elif keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 7
        elif keys[Int(Scancode.SCANCODE_UP)]:
            action = 5
        elif keys[Int(Scancode.SCANCODE_DOWN)]:
            action = 3

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Boss down! (seed", env.current_level_seed, ") — next level…")
            else:
                print("You were hit! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)

    numkeys.free()
    window.close()
    print("=== Done ===")
