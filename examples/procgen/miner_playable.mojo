"""Playable Procgen Miner — arrow keys dig; collect diamonds, avoid falling boulders.

Run:
  pixi run mojo run -I . examples/procgen/miner_playable.mojo

Controls:
  Arrow keys  Move / dig (one cell per step; push boulders by walking into them)
  Close window / Esc  Quit

Dig through the dirt to collect every blue diamond, then reach the exit window —
but don't let a falling boulder land on you. `num_levels=0` draws from the full
level set. The world steps continuously (boulders fall), so hold a direction to
keep moving. See `docs/PROCGEN_MINER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import MinerEnv
from mojo_rl.envs.procgen.games.miner import DIST_HARD
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512


def main() raises:
    print("=== Playable Procgen Miner ===")
    print("Arrow keys = dig/move    Close window = quit")
    print("Collect all diamonds, reach the exit, dodge falling boulders.")

    var env = MinerEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_HARD
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Miner"), fps=8)
    _ = env.reset()
    print("Level seed:", env.current_level_seed, " diamonds:", env.game.diamonds_remaining)
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # Grid-step, no diagonal (set_action_xy zeroes vy when vx set). x priority.
        var action = 4  # stand
        if keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 7
        elif keys[Int(Scancode.SCANCODE_UP)]:
            action = 5
        elif keys[Int(Scancode.SCANCODE_DOWN)]:
            action = 3

        # Step every frame (boulders fall continuously) at ~8 fps.
        var res = env.step(action)
        if res.done:
            if res.level_complete:
                print(
                    "Level",
                    episode,
                    "complete! all diamonds + exit (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
            else:
                print("Crushed! (seed", env.current_level_seed, ") — loading next…")
            episode += 1
            _ = env.reset()
            print(
                "Level seed:",
                env.current_level_seed,
                " diamonds:",
                env.game.diamonds_remaining,
            )
        frame = env.render(WIN)

        window.delay(125)  # ~8 fps

    numkeys.free()
    window.close()
    print("=== Done ===")
