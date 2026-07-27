"""Playable Procgen Ninja — charge jumps + throw stars to the mushroom.

Run: pixi run mojo run -I . examples/procgen/ninja_playable.mojo

Controls:
  Left/Right  Run
  Up          Hold to CHARGE a jump, release to leap (higher = longer hold)
  Space       Throw a star (forward, in your facing direction)
  Close/Esc   Quit

Charge-jump across the platforms to the mushroom goal (+10); throw stars to
detonate bombs. Death on touching fire, bombs, or explosions.
See `docs/PROCGEN_NINJA_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import NinjaEnv
from mojo_rl.envs.procgen.games.ninja import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Ninja ===")
    print("Left/Right = run    Up = charge+release jump    Space = throw star")
    print("Reach the mushroom; detonate bombs with stars; avoid fire/bombs/blasts.")

    var env = NinjaEnv(ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD)
    var window = PixelWindow(WIN, WIN, String("Procgen Ninja"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)
    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1
    while window.is_open():
        window.show(frame, WIN, WIN)
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var vx = 1 if keys[Int(Scancode.SCANCODE_RIGHT)] else (
            -1 if keys[Int(Scancode.SCANCODE_LEFT)] else 0
        )
        var vy = 1 if keys[Int(Scancode.SCANCODE_UP)] else 0
        # Space throws a star (special action 1 = straight, facing-directed).
        var action: Int
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 9
        else:
            action = (vx + 1) * 3 + (vy + 1)
        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Reached the mushroom! (seed", env.current_level_seed, ") — next level…")
            else:
                print("Killed! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print("Level seed:", env.current_level_seed)
        frame = env.render(WIN)
        window.delay(66)
    numkeys.free()
    window.close()
    print("=== Done ===")
