"""Demo of BipedalWalker v2 GPU batch mode.

Run with:
  # Apple Silicon (Metal)
  pixi run -e apple mojo run examples/bipedal_walker_v2_gpu_demo.mojo

  # NVIDIA GPUs (CUDA)
  pixi run -e nvidia mojo run examples/bipedal_walker_v2_gpu_demo.mojo

Features:
- Batched physics simulation on GPU
- Motor-enabled revolute joints
- Parallel environment execution
- State/observation extraction

This demo runs multiple environments in parallel on the GPU.
"""

from gpu.host import DeviceContext, DeviceBuffer
from random import random_float64, seed
from time import perf_counter

from envs.bipedal_walker import BipedalWalkerV2, BWConstants
from physics_gpu import dtype


fn main() raises:
    print("=== BipedalWalker v2 GPU Batch Demo ===")
    print()

    # Configuration
    comptime BATCH_SIZE: Int = 64
    comptime STATE_SIZE: Int = BWConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = BWConstants.OBS_DIM_VAL
    comptime ACTION_DIM: Int = BWConstants.ACTION_DIM_VAL

    print("Batch size:", BATCH_SIZE)
    print("State size:", STATE_SIZE)
    print("Observation dim:", OBS_DIM)
    print("Action dim:", ACTION_DIM)
    print()

    # Create GPU context
    var ctx = DeviceContext()
    print("GPU initialized")

    # Allocate buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)

    # Initialize dones buffer to zeros (prevent garbage value issues)
    var host_dones_init = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_dones_init.append(Float32(0))
    ctx.enqueue_copy(dones_buf, host_dones_init.unsafe_ptr())
    ctx.synchronize()

    # Reset all environments
    print("Resetting", BATCH_SIZE, "environments...")
    var start = perf_counter()

    BipedalWalkerV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, rng_seed=42
    )
    ctx.synchronize()

    var reset_time = perf_counter() - start
    print("Reset time:", reset_time * 1000, "ms")
    print()

    # Initialize actions with random values on host, then copy to device
    var host_actions = List[Float32](capacity=BATCH_SIZE * ACTION_DIM)
    seed(42)
    for _ in range(BATCH_SIZE * ACTION_DIM):
        host_actions.append(Float32(random_float64() * 2.0 - 1.0) * 0.5)

    # Copy actions to device
    ctx.enqueue_copy(actions_buf, host_actions.unsafe_ptr())
    ctx.synchronize()

    # Run steps
    var num_steps = 100
    print("Running", num_steps, "steps on", BATCH_SIZE, "environments...")
    start = perf_counter()

    for step in range(num_steps):
        BipedalWalkerV2[DType.float32].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
            rng_seed=UInt64(step + 1),
        )

        # Selective reset done environments every 10 steps
        if step % 10 == 9:
            BipedalWalkerV2[DType.float32].selective_reset_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states_buf, dones_buf, rng_seed=UInt64(step + 100))

    ctx.synchronize()
    var step_time = perf_counter() - start
    var total_steps = num_steps * BATCH_SIZE
    var steps_per_sec = total_steps / step_time

    print("Step time:", step_time * 1000, "ms")
    print("Total env steps:", total_steps)
    print("Steps per second:", Int(steps_per_sec))
    print()

    # Copy results back to host
    var host_obs = List[Float32](capacity=BATCH_SIZE * OBS_DIM)
    for _ in range(BATCH_SIZE * OBS_DIM):
        host_obs.append(Float32(0))
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)

    var host_rewards = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_rewards.append(Float32(0))
    ctx.enqueue_copy(host_rewards.unsafe_ptr(), rewards_buf)

    var host_dones = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_dones.append(Float32(0))
    ctx.enqueue_copy(host_dones.unsafe_ptr(), dones_buf)
    ctx.synchronize()

    # Print sample results
    print("=== Sample Results ===")
    print()
    print("Environment 0:")
    print("  Hull angle:", Float64(host_obs[0]))
    print("  Hull angular velocity:", Float64(host_obs[1]))
    print("  Vel x:", Float64(host_obs[2]))
    print("  Vel y:", Float64(host_obs[3]))
    print("  Reward:", Float64(host_rewards[0]))
    print("  Done:", Float64(host_dones[0]))

    print()
    print("Environment 1:")
    print("  Hull angle:", Float64(host_obs[OBS_DIM]))
    print("  Hull angular velocity:", Float64(host_obs[OBS_DIM + 1]))
    print("  Vel x:", Float64(host_obs[OBS_DIM + 2]))
    print("  Vel y:", Float64(host_obs[OBS_DIM + 3]))
    print("  Reward:", Float64(host_rewards[1]))
    print("  Done:", Float64(host_dones[1]))

    # Count done environments
    var num_done = 0
    for i in range(BATCH_SIZE):
        if host_dones[i] > 0.5:
            num_done += 1
    print()
    print("Done environments:", num_done, "/", BATCH_SIZE)

    print()
    print("=== GPU Demo Complete ===")
