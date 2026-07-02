"""Playable Procgen Heist — arrow keys navigate; grab keys, open doors, reach the gem.

Run:
  pixi run mojo run -I . examples/procgen/heist_playable.mojo

Controls:
  Arrow keys  Move (momentum-based; diagonals supported)
  Close window / Esc  Quit

Collect a colored key to open its matching locked door, then reach the yellow gem
to complete the level; a new procedurally-generated level then loads.
`num_levels=0` draws from the full level set. See `docs/PROCGEN_HEIST_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import HeistEnv
from mojo_rl.envs.procgen.games.heist import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Heist ===")
    print("Arrow keys = move    Close window = quit")
    print("Grab keys to open matching doors; reach the gem to win.")

    var env = HeistEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Heist"), fps=30)
    _ = env.reset()
    print("Level seed:", env.current_level_seed, " keys:", env.game.num_keys)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # move = (vx+1)*3 + (vy+1) (inverse of set_action_xy); no key → 4 (stand).
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
        var action = (vx + 1) * 3 + (vy + 1)

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print(
                    "Level",
                    episode,
                    "complete! reached the gem (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
            else:
                print("Reset (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed, " keys:", env.game.num_keys)
        frame = env.render(WIN)

        window.delay(33)  # ~30 fps

    numkeys.free()
    window.close()
    print("=== Done ===")
