"""Playable Procgen Chaser — arrow keys steer the player; eat all the orbs.

Run:
  pixi run mojo run -I . examples/procgen/chaser_playable.mojo

Controls:
  Arrow keys  Steer (hold to change direction; diagonals supported)
  Close window / Esc  Quit

Chaser is Pac-Man-like: the player glides continuously at max speed in the last
direction you pressed until it hits a wall — so the game steps every frame (no
grid-step throttle like maze). Eat every green orb to complete the level (the big
yellow crystals are power pellets that briefly let you eat the enemies); touching
an enemy while not powered ends the run. Either way a new procedurally-generated
level then loads. `num_levels=0` draws from the full level set. See
`docs/PROCGEN_CHASER_SCOPE.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import ChaserEnv, DIST_EASY
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512  # window size (world upscaled from maze_dim cells)


def main() raises:
    print("=== Playable Procgen Chaser ===")
    print("Arrow keys = steer    Close window = quit")
    print("Eat all the orbs; crystals let you eat enemies; don't get caught.")

    var env = ChaserEnv(
        ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0, dist_mode=DIST_EASY
    )
    var window = PixelWindow(WIN, WIN, String("Procgen Chaser"), fps=15)
    _ = env.reset()
    print(
        "Level seed:",
        env.current_level_seed,
        " orbs:",
        env.game.total_orbs,
    )
    # Render the human view at full window resolution (not the tiny 64×64 obs).
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # Continuous move: build (action_vx, action_vy) ∈ {-1,0,1}² from held
        # arrows, then move = (vx+1)*3 + (vy+1) (inverse of set_action_xy). No key
        # → move 4 (the player keeps its current velocity, Pac-Man style).
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
                    "complete! ate all",
                    env.game.total_orbs,
                    "orbs (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
            else:
                print(
                    "Caught! (seed",
                    env.current_level_seed,
                    ",",
                    env.game.orbs_collected,
                    "orbs) — loading next…",
                )
            episode += 1
            _ = env.reset()
            print(
                "Level seed:",
                env.current_level_seed,
                " orbs:",
                env.game.total_orbs,
            )
        frame = env.render(WIN)

        window.delay(66)  # ~15 fps; the player glides ~7 cells/sec

    numkeys.free()
    window.close()
    print("=== Done ===")
