"""Playable Procgen Maze — arrow keys move the mouse to the cheese.

Run:
  pixi run mojo run -I . examples/procgen/maze_playable.mojo

Controls:
  Arrow keys  Move (up/down/left/right)
  Close window / Esc  Quit

Reach the cheese (goal) to complete the level; a new procedurally-generated
level then loads. `num_levels=0` draws from the full level set. See
`docs/PROCGEN_PORT.md`.
"""

from std.memory import alloc
from mojo_rl.envs.procgen.games import MazeEnv
from mojo_rl.envs.procgen.core.pixel_window import PixelWindow
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode

comptime ASSET_ROOT = String("assets/procgen/")
comptime WIN = 512  # window size (64×64 obs upscaled ×8)


def main() raises:
    print("=== Playable Procgen Maze ===")
    print("Arrow keys = move    Close window = quit")
    print("Reach the cheese to complete the level.")

    var env = MazeEnv(ASSET_ROOT, rand_seed=0, num_levels=0, start_level=0)
    var window = PixelWindow(WIN, WIN, String("Procgen Maze"), fps=30)
    _ = env.reset()
    print("Level seed:", env.current_level_seed)
    # Render the human view at full window resolution (not the tiny 64×64 obs)
    # so the mouse/cheese stay crisp instead of flickering at ~2 px.
    var frame = env.render(WIN)

    var numkeys = alloc[Int32](1)
    numkeys[] = 0
    var episode = 1
    # Grid-step throttle: take one move every STEP_EVERY frames while a key is
    # held, so movement is a comfortable pace independent of the frame rate.
    comptime STEP_EVERY = 5  # ~12 grid-steps/sec while a key is held (60fps × 5)
    var since_step = STEP_EVERY

    while window.is_open():
        window.show(frame, WIN, WIN)  # pumps SDL events

        var keys = get_keyboard_state(numkeys.as_unsafe_any_origin())
        # Maze move codes: up(+y)=5, down(-y)=3, left=1, right=7, stand=4.
        var action = 4
        if keys[Int(Scancode.SCANCODE_UP)]:
            action = 5
        elif keys[Int(Scancode.SCANCODE_DOWN)]:
            action = 3
        elif keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 7

        since_step += 1
        if action != 4 and since_step >= STEP_EVERY:
            since_step = 0
            var res = env.step(action)
            if res.level_complete:
                print(
                    "Level",
                    episode,
                    "complete! (seed",
                    env.current_level_seed,
                    ") — loading next…",
                )
                episode += 1
                _ = env.reset()
                print("Level seed:", env.current_level_seed)
            frame = env.render(WIN)

        window.delay(16)  # ~60 fps display; movement paced by STEP_EVERY

    numkeys.free()
    window.close()
    print("=== Done ===")
