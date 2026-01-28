"""Demo of the HalfCheetah3D environment with 3D visualization and GPU support."""

from envs import HalfCheetah3D, HalfCheetah3DState, HalfCheetah3DAction
from envs.half_cheetah_3d import HalfCheetah3DRenderer
from envs.half_cheetah_3d.constants3d import HC3DConstantsCPU
from physics3d import dtype
from random import random_float64
from gpu.host import DeviceContext, DeviceBuffer


fn run_headless_demo() raises:
    """Run demo without visualization (for testing)."""
    print("=== HalfCheetah3D Headless Demo ===")

    # Create environment
    var env = HalfCheetah3D(seed=42)
    print("\nEnvironment created:")
    print("  State size:", env.STATE_SIZE)
    print("  Obs dim:", env.obs_dim())
    print("  Action dim:", env.action_dim())
    print("  Action bounds: [", env.action_low(), ",", env.action_high(), "]")

    # Reset environment
    var state = env.reset()
    print("\nInitial state (first 5 obs):")
    var obs = state.to_list()
    print("  torso_z:", obs[0])
    print("  torso_pitch:", obs[1])
    print("  bthigh_angle:", obs[2])
    print("  vel_x:", obs[8])
    print("  vel_z:", obs[9])

    # Run a few steps with random actions
    print("\nRunning 100 steps with random actions...")
    var total_reward = Float64(0.0)

    for step in range(100):
        # Create random action
        var action = HalfCheetah3DAction()
        action.bthigh = Scalar[dtype](random_float64(-1.0, 1.0))
        action.bshin = Scalar[dtype](random_float64(-1.0, 1.0))
        action.bfoot = Scalar[dtype](random_float64(-1.0, 1.0))
        action.fthigh = Scalar[dtype](random_float64(-1.0, 1.0))
        action.fshin = Scalar[dtype](random_float64(-1.0, 1.0))
        action.ffoot = Scalar[dtype](random_float64(-1.0, 1.0))

        # Step environment
        var result = env.step(action)
        var next_state = result[0]
        var reward = result[1]
        var done = result[2]
        total_reward += Float64(reward)

        if step % 25 == 0:
            var next_obs = next_state.to_list()
            print("  Step", step, ": reward =", Float64(reward), ", torso_z =", Float64(next_obs[0]))

        if done:
            print("  Episode done at step", step)
            break

    print("\nTotal reward:", total_reward)
    print("\n=== Headless Demo Complete ===")


fn run_gpu_demo() raises:
    """Run GPU demo to test batched simulation."""
    print("=== HalfCheetah3D GPU Demo ===")

    # Constants
    comptime BATCH_SIZE = 4  # Small batch for testing
    comptime STATE_SIZE = HC3DConstantsCPU.STATE_SIZE
    comptime OBS_DIM = HC3DConstantsCPU.OBS_DIM
    comptime ACTION_DIM = HC3DConstantsCPU.ACTION_DIM

    print("\nGPU Configuration:")
    print("  Batch size:", BATCH_SIZE)
    print("  State size:", STATE_SIZE)
    print("  Obs dim:", OBS_DIM)
    print("  Action dim:", ACTION_DIM)

    # Create GPU context
    var ctx = DeviceContext()
    print("\nGPU device acquired")

    # Allocate buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)
    print("  Buffers allocated")

    # Reset all environments
    print("\nResetting environments on GPU...")
    HalfCheetah3D.reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, rng_seed=42
    )
    ctx.synchronize()
    print("  Reset complete")

    # Copy states to host to verify reset
    var states_host = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    ctx.enqueue_copy(states_host.unsafe_ptr(), states_buf.unsafe_ptr(), BATCH_SIZE * STATE_SIZE)
    ctx.synchronize()

    # Check torso Z position for each environment
    print("\nInitial torso positions (z):")
    from physics3d import BODY_STATE_SIZE_3D, IDX_PZ
    for i in range(BATCH_SIZE):
        var torso_z_off = i * STATE_SIZE + HC3DConstantsCPU.BODIES_OFFSET + IDX_PZ
        print("  Env", i, ": z =", Float64(states_host.unsafe_ptr()[torso_z_off]))

    # Create random actions on host
    print("\nStepping environments on GPU...")
    var actions_host = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE * ACTION_DIM)
    for i in range(BATCH_SIZE):
        for j in range(ACTION_DIM):
            actions_host.unsafe_ptr()[i * ACTION_DIM + j] = Scalar[dtype](random_float64(-1.0, 1.0))
    ctx.enqueue_copy(actions_buf.unsafe_ptr(), actions_host.unsafe_ptr(), BATCH_SIZE * ACTION_DIM)

    # Step all environments
    HalfCheetah3D.step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf, rng_seed=42
    )
    ctx.synchronize()
    print("  Step complete")

    # Copy results to host
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE * OBS_DIM)
    ctx.enqueue_copy(rewards_host.unsafe_ptr(), rewards_buf.unsafe_ptr(), BATCH_SIZE)
    ctx.enqueue_copy(obs_host.unsafe_ptr(), obs_buf.unsafe_ptr(), BATCH_SIZE * OBS_DIM)
    ctx.synchronize()

    # Print results
    print("\nResults after one step:")
    for i in range(BATCH_SIZE):
        print("  Env", i, ": reward =", Float64(rewards_host.unsafe_ptr()[i]),
              ", torso_z =", Float64(obs_host.unsafe_ptr()[i * OBS_DIM + 0]))

    # Run multiple steps
    print("\nRunning 10 more steps...")
    var total_rewards = InlineArray[Float64, BATCH_SIZE](fill=Float64(0))
    for step in range(10):
        # Generate random actions
        for i in range(BATCH_SIZE):
            for j in range(ACTION_DIM):
                actions_host.unsafe_ptr()[i * ACTION_DIM + j] = Scalar[dtype](random_float64(-1.0, 1.0))
        ctx.enqueue_copy(actions_buf.unsafe_ptr(), actions_host.unsafe_ptr(), BATCH_SIZE * ACTION_DIM)

        # Step
        HalfCheetah3D.step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf, rng_seed=UInt64(42 + step)
        )
        ctx.synchronize()

        # Accumulate rewards
        ctx.enqueue_copy(rewards_host.unsafe_ptr(), rewards_buf.unsafe_ptr(), BATCH_SIZE)
        ctx.synchronize()
        for i in range(BATCH_SIZE):
            total_rewards[i] += Float64(rewards_host.unsafe_ptr()[i])

    print("\nTotal rewards after 11 steps:")
    for i in range(BATCH_SIZE):
        print("  Env", i, ":", total_rewards[i])

    print("\n=== GPU Demo Complete ===")


fn run_visual_demo() raises:
    """Run demo with 3D visualization."""
    print("=== HalfCheetah3D Visual Demo ===")
    print("Controls: Close window to exit")

    # Create environment and renderer
    var env = HalfCheetah3D(seed=42)
    var renderer = HalfCheetah3DRenderer(
        width=1024,
        height=576,
        follow_cheetah=True,
        show_velocity=True,
    )

    # Initialize renderer
    renderer.init()
    print("Renderer initialized. Running simulation...")

    # Reset environment
    _ = env.reset()

    # Debug: print body positions
    print("\nBody positions after reset:")
    from physics3d import BODY_STATE_SIZE_3D, IDX_PX, IDX_PY, IDX_PZ
    from physics3d import JOINT_DATA_SIZE_3D, JOINT3D_TYPE, JOINT3D_BODY_A, JOINT3D_BODY_B
    from physics3d import JOINT3D_ANCHOR_AX, JOINT3D_ANCHOR_AZ, JOINT3D_ANCHOR_BZ
    var names = List[String]()
    names.append("Torso")
    names.append("BThigh")
    names.append("BShin")
    names.append("BFoot")
    names.append("FThigh")
    names.append("FShin")
    names.append("FFoot")
    for i in range(7):
        var off = env.BODIES_OFFSET + i * BODY_STATE_SIZE_3D
        print("  ", names[i], ": x=", Float64(env.state[off + IDX_PX]),
              " y=", Float64(env.state[off + IDX_PY]),
              " z=", Float64(env.state[off + IDX_PZ]))

    # Debug: print joint data
    print("\nJoint data after reset:")
    var joint_names = List[String]()
    joint_names.append("Back hip")
    joint_names.append("Back knee")
    joint_names.append("Back ankle")
    joint_names.append("Front hip")
    joint_names.append("Front knee")
    joint_names.append("Front ankle")
    for j in range(6):
        var joff = env.JOINTS_OFFSET + j * JOINT_DATA_SIZE_3D
        print("  ", joint_names[j], ": type=", Int(env.state[joff + JOINT3D_TYPE]),
              " body_a=", Int(env.state[joff + JOINT3D_BODY_A]),
              " body_b=", Int(env.state[joff + JOINT3D_BODY_B]),
              " anchor_ax=", Float64(env.state[joff + JOINT3D_ANCHOR_AX]),
              " anchor_az=", Float64(env.state[joff + JOINT3D_ANCHOR_AZ]),
              " anchor_bz=", Float64(env.state[joff + JOINT3D_ANCHOR_BZ]))

    var step = 0
    var total_reward = Float64(0.0)

    # Main loop
    while not renderer.check_quit():
        # Create random action (replace with policy for actual training)
        var action = HalfCheetah3DAction()
        action.bthigh = Scalar[dtype](random_float64(-1.0, 1.0))
        action.bshin = Scalar[dtype](random_float64(-1.0, 1.0))
        action.bfoot = Scalar[dtype](random_float64(-1.0, 1.0))
        action.fthigh = Scalar[dtype](random_float64(-1.0, 1.0))
        action.fshin = Scalar[dtype](random_float64(-1.0, 1.0))
        action.ffoot = Scalar[dtype](random_float64(-1.0, 1.0))

        # Step environment
        var result = env.step(action)
        var reward = result[1]
        var done = result[2]
        total_reward += Float64(reward)

        # Get torso position and velocity for rendering
        var torso_x = Float64(env.state[env.BODIES_OFFSET + 0])  # IDX_PX = 0
        var vel_x = Float64(env.cached_state.vel_x)

        # Render
        renderer.render(env.state, torso_x, vel_x)

        # Delay for ~20 FPS (slower for easier observation)
        renderer.delay(50)

        step += 1

        # Reset if episode done
        if done:
            print("Episode completed. Steps:", step, "Total reward:", total_reward)
            _ = env.reset()
            step = 0
            total_reward = 0.0

    # Cleanup
    renderer.close()
    print("\n=== Visual Demo Complete ===")


fn main() raises:
    # Run headless demo first (quick test)
    run_headless_demo()

    # GPU demo - uses scalar-only functions in physics3d for Metal compatibility
    print("\n" + "=" * 50)
    print("Starting GPU demo...")
    print("=" * 50 + "\n")
    run_gpu_demo()

    print("\n" + "=" * 50)
    print("Starting visual demo...")
    print("=" * 50 + "\n")

    # Run visual demo
    run_visual_demo()
