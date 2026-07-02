"""Playable Procgen Leaper — arrow keys hop; cross the road and river to the finish.

Run:
  pixi run mojo run -I . examples/procgen/leaper_playable.mojo

Controls:
  Arrow keys  Hop (one cell per press; can't change direction mid-hop)
  Close window / Esc  Quit

Dodge the cars on the road and ride the logs across the water (stepping into open
water drowns you). Reach the finish line at the top to win; a new level then loads.
`num_levels=0` draws from the full level set. See `docs/PROCGEN_LEAPER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import LeaperEnv
from mojo_rl.envs.procgen.games.leaper import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Leaper ===")
    print("Arrow keys = hop    Close window = quit")
    print("Cross the road and river to the finish; dodge cars, ride logs.")

    var env = LeaperEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Leaper"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # move = (vx+1)*3 + (vy+1); no key → 4 (a hop auto-completes over 5 steps).
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
                    "complete! reached the finish (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
            else:
                print("Splat/splash! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)

        window.delay(66)  # ~15 fps

    numkeys.free()
    window.close()
    print("=== Done ===")
