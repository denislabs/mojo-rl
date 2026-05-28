"""Playable Breakout with GIF recording — LEFT/RIGHT + SPACE to fire.

Close the window to stop recording and save the GIF.

Run with:
    pixi run mojo run -I . examples/arcade_games/breakout_playable_gif.mojo
"""

from std.memory import alloc
from mojo_rl.envs.arcade_games.breakout import BreakoutEnv
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode


def main() raises:
    print("=== Playable Breakout — Recording to GIF ===")
    print("Controls: LEFT/RIGHT arrows, SPACE to fire, close window to stop & save")

    var env = BreakoutEnv[DType.float64]()
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    env.start_recording("gifs/breakout_playable.gif", fps=30, skip=2)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0
    var frames = 0

    while env.is_renderer_open():
        env.render_frame()

        var keys = get_keyboard_state(numkeys_ptr)

        var action = 0
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 2
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 3

        var result = env.step_obs(action)

        if result[2]:
            print(
                "Game over! Score:",
                Int(env.state[49]),
                " Lives left:",
                Int(env.state[6]),
            )
            _ = env.reset_obs_list()

        env.renderer_delay(16)
        frames += 1

    env.stop_recording()
    numkeys_ptr.free()
    env.close_renderer()

    print("Recorded", frames, "frames (skip=2)")
    print("Saved: gifs/breakout_playable.gif")
    print("=== Done ===")
