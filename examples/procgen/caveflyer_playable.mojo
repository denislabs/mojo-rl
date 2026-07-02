"""Playable Procgen Caveflyer — fly the cave to the green UFO.

Run:
  pixi run mojo run -I . examples/procgen/caveflyer_playable.mojo

Controls:
  Up / Down    Thrust forward / reverse
  Left / Right Rotate the ship
  Space        Fire (coasts while firing)
  Close window / Esc  Quit

Rotate-and-thrust (Asteroids-style) through the carved cave to the green UFO goal
(+10); shoot the red target ships (+3 each). One touch of a meteor, an enemy ship,
a target, or a cave wall ends the run. See `docs/PROCGEN_CAVEFLYER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import CaveflyerEnv
from mojo_rl.envs.procgen.games.caveflyer import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Caveflyer ===")
    print("Up/Down = thrust    Left/Right = rotate    Space = fire    Close = quit")
    print("Reach the green UFO; shoot red targets; avoid meteors, enemies + walls.")

    var env = CaveflyerEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Caveflyer"), fps=15)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events
        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        var fire = keys[Int(Scancode.SCANCODE_SPACE)]
        # move = (vrot+1)*3 + (accel+1); accel = fwd/rev, vrot = rotate.
        var accel = 1 if keys[Int(Scancode.SCANCODE_UP)] else (
            -1 if keys[Int(Scancode.SCANCODE_DOWN)] else 0
        )
        var vrot = 1 if keys[Int(Scancode.SCANCODE_RIGHT)] else (
            -1 if keys[Int(Scancode.SCANCODE_LEFT)] else 0
        )
        var move = (vrot + 1) * 3 + (accel + 1)
        var action = 9 if fire else move

        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print("Reached the UFO! (seed", env.current_level_seed, ") — next level…")
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
