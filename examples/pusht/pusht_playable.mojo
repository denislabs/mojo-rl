"""Playable PushT — move the agent toward the mouse cursor.

Mirrors the pymunk `teleop_agent` from gym_pusht: each frame, we feed the
current mouse position as the continuous-action target. The PD controller
inside PushTEnv drives the agent there over the next step's substeps.

Controls:
    Mouse        : move the agent toward the cursor (target position)
    Space        : pause / resume (target frozen while paused)
    R            : reset the episode
    Close window : quit
"""

from std.memory import alloc
from mojo_rl.envs.pusht import PushTEnv, PConstants, PushTAction
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state


def main() raises:
    print("=== Playable PushT ===")
    print("Controls:")
    print("  Mouse: move the blue agent toward the cursor")
    print("  Space: pause/resume   R: reset   Close window: quit")

    var env = PushTEnv[DType.float32](seed=UInt64(42))
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(256.0)
    mouse_y_ptr[] = Float32(256.0)

    var paused = False
    var prev_space = False
    var prev_r = False
    var total_reward = Float32(0.0)
    var episode_count = 1

    while env.is_renderer_open():
        # Drawing the frame also pumps SDL events (begin_frame_with_color does
        # that for us inside env.render_frame). We must do this before reading
        # mouse / keyboard state.
        env.render_frame()

        # Keyboard
        var keys = get_keyboard_state(numkeys_ptr)
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])
        if cur_space and not prev_space:
            paused = not paused
            print(("[paused]" if paused else "[resumed]"))
        if cur_r and not prev_r:
            _ = env.reset_obs_list()
            total_reward = Float32(0.0)
            episode_count += 1
            print("[reset] episode", episode_count)
        prev_space = cur_space
        prev_r = cur_r

        # Mouse → world target
        _ = get_mouse_state(
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var wxy = env.screen_to_world(
            Int(mouse_x_ptr[]), Int(mouse_y_ptr[])
        )
        var target_x = wxy[0]
        var target_y = wxy[1]

        if not paused:
            var action = PushTAction[DType.float32](
                target_x=target_x, target_y=target_y
            )
            var result = env.step(action)
            total_reward = total_reward + result[1]
            if result[2]:
                var cov = env.coverage()
                var threshold = Scalar[DType.float32](
                    PConstants.SUCCESS_THRESHOLD
                )
                var outcome = (
                    "WIN (cov > "
                    + String(Float64(threshold))
                    + ")" if cov > threshold else "timeout after "
                    + String(PConstants.MAX_STEPS)
                    + " steps"
                )
                print(
                    "[done]",
                    outcome,
                    " total_reward=",
                    total_reward,
                    " coverage=",
                    cov,
                )
                _ = env.reset_obs_list()
                total_reward = Float32(0.0)
                episode_count += 1

        env.renderer_delay(16)  # ~60 fps

    env.close_renderer()
    mouse_x_ptr.free()
    mouse_y_ptr.free()
    numkeys_ptr.free()
    print("=== Done ===")
