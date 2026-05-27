"""Render any Physics3D environment with random actions.

Provides a reusable function to visually test any Phyics3dEnv (or any
BoxContinuousActionEnv & RenderableEnv) by running random actions with
3D rendering.

Usage:
    from mojo_rl.envs.render_random import render_random
    from mojo_rl.envs.half_cheetah import HalfCheetah

    def main() raises:
        var env = HalfCheetah()
        render_random(env, num_steps=1000)
"""

from std.random import random_float64
from std.time import perf_counter_ns

from mojo_rl.core import BoxContinuousActionEnv, RenderableEnv


def render_random[
    E: BoxContinuousActionEnv & RenderableEnv
](
    mut env: E,
    num_steps: Int = 2000,
    frame_delay_ms: Int = 16,
    verbose: Bool = True,
    print_every: Int = 200,
    show_velocity: Bool = True,
    record_path: String = "",
    record_fps: Int = 30,
    record_skip: Int = 1,
) raises:
    """Render an environment with uniform random actions.

    Resets the environment, then runs `num_steps` steps with random actions
    sampled uniformly from [action_low, action_high]. Renders each frame
    and prints periodic telemetry.

    Close the window to stop early.

    Args:
        env: Environment to render (must support continuous actions + rendering).
        num_steps: Maximum steps to run.
        frame_delay_ms: Delay between frames in ms (~16 for 60 FPS).
        verbose: Print step telemetry.
        print_every: Steps between telemetry prints.
        show_velocity: Show velocity vectors in the renderer.
        record_path: If non-empty, record to this file (.gif or .mp4).
        record_fps: FPS for the recording (default 30).
        record_skip: Only record every Nth frame (reduces file size).
    """
    var action_dim = env.action_dim()
    var lo = Float64(env.action_low())
    var hi = Float64(env.action_high())

    if verbose:
        print("=" * 60)
        print("Random Action Rendering")
        print("=" * 60)
        print("  OBS_DIM:", env.obs_dim())
        print("  ACTION_DIM:", action_dim)
        print("  Action range: [", lo, ",", hi, "]")
        print("  Steps:", num_steps, " Frame delay:", frame_delay_ms, "ms")
        print()

    # Init renderer
    _ = env.init_renderer(show_velocity=show_velocity)

    # Start recording if requested
    var recording = record_path.byte_length() > 0
    if recording:
        env.start_recording(record_path, record_fps, record_skip)
        if verbose:
            print("Recording to:", record_path)

    # Reset environment
    _ = env.reset_obs_list()

    if verbose:
        print("Renderer open. Close window to exit.")
        print("-" * 60)

    var start_ns = perf_counter_ns()
    var step = 0
    var total_reward: Float64 = 0.0

    while step < num_steps:
        # Check quit
        if env.check_renderer_quit():
            break
        if not env.is_renderer_open():
            break

        # Sample random action
        var action = List[Float64](capacity=action_dim)
        for _ in range(action_dim):
            action.append(random_float64(lo, hi))

        # Step
        var result = env.step_continuous_vec(action)
        var reward = Float64(result[1])
        var done = result[2]
        total_reward += reward

        # Render
        env.render_frame()
        env.renderer_delay(frame_delay_ms)

        # Telemetry
        if verbose and step % print_every == 0:
            var elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000
            var fps = Float64(0)
            if elapsed_ms > 0:
                fps = Float64(step + 1) / (Float64(elapsed_ms) / 1000.0)
            print(
                "Step",
                step,
                " | reward:",
                String(reward)[byte=:8],
                " | total:",
                String(total_reward)[byte=:10],
                " | FPS:",
                Int(fps),
            )

        step += 1

        # Reset on done
        if done:
            _ = env.reset_obs_list()
            if verbose:
                print("  [Episode done at step", step, "- resetting]")

    # Stop recording before closing
    if recording:
        env.stop_recording()
        if verbose:
            print("Saved recording:", record_path)

    # Cleanup
    env.close_renderer()

    var total_ms = (perf_counter_ns() - start_ns) / 1_000_000
    if verbose:
        print("-" * 60)
        print("Done!", step, "steps in", Int(total_ms), "ms")
        if total_ms > 0:
            print("FPS:", Int(Float64(step) / (Float64(total_ms) / 1000.0)))
        print("Total reward:", total_reward)
