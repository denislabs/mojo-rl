"""Playable Space Invaders — LEFT/RIGHT arrows to move, SPACE to fire."""

from std.memory import alloc
from envs.atari_games.space_invaders import SpaceInvadersEnv
from render.sdl.sdl_keyboard import get_keyboard_state
from render.sdl.sdl_scancode import Scancode


fn main() raises:
    print("=== Playable Space Invaders ===")
    print("Controls: LEFT/RIGHT arrows to move, SPACE to fire, close window to quit")

    var env = SpaceInvadersEnv[DType.float64]()
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while env.is_renderer_open():
        # Render first — pumps SDL events
        env.render_frame()

        # Read keyboard
        var keys = get_keyboard_state(numkeys_ptr)

        # Actions: 0=NOOP, 1=LEFT, 2=RIGHT, 3=FIRE
        var action = 0
        if keys[Int(Scancode.SCANCODE_SPACE)]:
            action = 3
        elif keys[Int(Scancode.SCANCODE_LEFT)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            action = 2

        var result = env.step_obs(action)
        var done = result[2]

        if done:
            print(
                "Game over! Score:",
                Int(env.state[67]),
                " Lives left:",
                Int(env.state[68]),
            )
            _ = env.reset_obs_list()

        env.renderer_delay(16)  # ~60 fps

    numkeys_ptr.free()
    env.close_renderer()
    print("=== Done ===")
