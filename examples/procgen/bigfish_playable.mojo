"""Playable Procgen Bigfish — arrow keys swim; eat smaller fish, avoid bigger ones.

Run:
  pixi run mojo run -I . examples/procgen/bigfish_playable.mojo

Controls:
  Arrow keys  Swim (momentum-based; diagonals supported)
  Close window / Esc  Quit

Eat fish smaller than you to grow; touching a bigger fish ends the run. Eat 30 to
complete the level; a new level then loads. See `docs/PROCGEN_BIGFISH_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import BigfishEnv
from mojo_rl.envs.procgen.games.bigfish import DIST_EASY
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Bigfish ===")
    print("Arrow keys = swim    Close window = quit")
    print("Eat smaller fish to grow; avoid bigger ones; eat 30 to win.")

    var env = BigfishEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_EASY
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Bigfish"), fps=30)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # move = (vx+1)*3 + (vy+1); no key → 4 (agent coasts to a stop via decay).
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
                    "complete! ate 30 fish (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
            else:
                print(
                    "Eaten! (seed",
                    env.current_level_seed,
                    ",",
                    env.game.fish_eaten,
                    "fish) — loading next…",
                )
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)

        window.delay(33)  # ~30 fps

    numkeys.free()
    window.close()
    print("=== Done ===")
