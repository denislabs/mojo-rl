"""Playable Procgen Climber — climb the platforms, grab every crystal.

Run: pixi run mojo run -I . examples/procgen/climber_playable.mojo

Controls:
  Left/Right  Run        Up / Space  Jump        Close / Esc  Quit

Climb up the platforms and collect every yellow crystal (+1 each; all → +10 and
done). One touch of a patrolling enemy ends the run. See `docs/PROCGEN_CLIMBER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import ClimberEnv
from mojo_rl.envs.procgen.games.climber import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Climber ===")
    print("Left/Right = run    Up/Space = jump    Close = quit")
    print("Collect every yellow crystal; avoid the enemies.")

    var env = ClimberEnv(ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD)
    var window = PixelWindow(WIN, WIN, String("Procgen Climber"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)
    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1
    while window.is_open():
        window.show(frame, WIN, WIN)
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var jump = keys[Int(Scancode.SCANCODE_UP)] or keys[Int(Scancode.SCANCODE_SPACE)]
        var vx = 1 if keys[Int(Scancode.SCANCODE_RIGHT)] else (
            -1 if keys[Int(Scancode.SCANCODE_LEFT)] else 0
        )
        var vy = 1 if jump else 0
        var action = (vx + 1) * 3 + (vy + 1)
        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("All crystals! (seed", env.current_level_seed, ") — next level…")
            else:
                print("Caught! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)
    numkeys.free()
    window.close()
    print("=== Done ===")
