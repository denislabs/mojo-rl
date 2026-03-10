"""Playable Pong — use UP/DOWN arrow keys to control the right paddle."""

from std.memory import alloc
from envs.atari_games.pong import PongEnv
from render.sdl.sdl_keyboard import get_keyboard_state
from render.sdl.sdl_scancode import Scancode


fn main() raises:
    print("=== Playable Pong ===")
    print("Controls: UP/DOWN arrows to move paddle, close window to quit")

    var env = PongEnv[DType.float64]()
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while env.is_renderer_open():
        # Render first — begin_frame() pumps SDL events, updating keyboard state
        env.render_frame()

        # Read keyboard state (valid after events pumped)
        var keys = get_keyboard_state(numkeys_ptr)

        # Map arrow keys to actions: 0=NOOP, 1=UP, 2=DOWN
        var action = 0
        if keys[Int(Scancode.SCANCODE_UP)]:
            action = 1
        elif keys[Int(Scancode.SCANCODE_DOWN)]:
            action = 2

        var result = env.step_obs(action)
        var done = result[2]

        if done:
            print(
                "Game over! Score:",
                Int(env.state[6]),
                "-",
                Int(env.state[7]),
                "(you - cpu)",
            )
            _ = env.reset_obs_list()

        env.renderer_delay(16)  # ~60 fps

    numkeys_ptr.free()
    env.close_renderer()
    print("=== Done ===")
