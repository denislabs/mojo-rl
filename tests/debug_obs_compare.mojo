"""Debug: Compare CPU and GPU observations."""

from random import seed
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from envs.hopper_3d import Hopper3D
from deep_rl import dtype as gpu_dtype

comptime HopperEnv = Hopper3D[DType.float64]


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Comparing CPU vs GPU Observations")
    print("=" * 70)
    print()

    # Create CPU environment and get initial observations
    var cpu_env = HopperEnv()
    var cpu_obs = cpu_env.reset_obs_list()

    print("CPU Initial Observations:")
    print("  [0] torso_z:     ", cpu_obs[0])
    print("  [1] torso_pitch: ", cpu_obs[1])
    print("  [2] hip_angle:   ", cpu_obs[2])
    print("  [3] knee_angle:  ", cpu_obs[3])
    print("  [4] ankle_angle: ", cpu_obs[4])
    print("  [5] vel_x:       ", cpu_obs[5])
    print("  [6] vel_z:       ", cpu_obs[6])
    print("  [7] omega_y:     ", cpu_obs[7])
    print("  [8] hip_omega:   ", cpu_obs[8])
    print("  [9] knee_omega:  ", cpu_obs[9])
    print("  [10] ankle_omega:", cpu_obs[10])
    print()

    # Create GPU environment and get initial observations
    with DeviceContext() as ctx:
        comptime STATE_SIZE = HopperEnv.STATE_SIZE
        comptime OBS_DIM = HopperEnv.OBS_DIM
        comptime BATCH = 1

        # Allocate GPU buffers
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH * STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH * OBS_DIM)

        # Reset GPU environment
        Hopper3D.reset_kernel_gpu[BATCH, STATE_SIZE](ctx, states_buf, rng_seed=42)
        ctx.synchronize()

        # Extract observations (using a zero action)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH * 3)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH)

        # Initialize actions to zero
        var actions_host = List[Scalar[gpu_dtype]](capacity=3)
        for _ in range(3):
            actions_host.append(Scalar[gpu_dtype](0.0))
        ctx.enqueue_copy(actions_buf, actions_host.unsafe_ptr())
        ctx.synchronize()

        # Step to extract observations (this calls _extract_obs_rewards_dones_gpu)
        Hopper3D.step_kernel_gpu[BATCH, STATE_SIZE, OBS_DIM, 3](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf
        )
        ctx.synchronize()

        # Copy observations back to host
        var gpu_obs = List[Scalar[gpu_dtype]](capacity=OBS_DIM)
        for _ in range(OBS_DIM):
            gpu_obs.append(Scalar[gpu_dtype](0.0))
        ctx.enqueue_copy(gpu_obs.unsafe_ptr(), obs_buf)
        ctx.synchronize()

        print("GPU Initial Observations (after 1 step with zero action):")
        print("  [0] torso_z:     ", gpu_obs[0])
        print("  [1] torso_pitch: ", gpu_obs[1])
        print("  [2] hip_angle:   ", gpu_obs[2])
        print("  [3] knee_angle:  ", gpu_obs[3])
        print("  [4] ankle_angle: ", gpu_obs[4])
        print("  [5] vel_x:       ", gpu_obs[5])
        print("  [6] vel_z:       ", gpu_obs[6])
        print("  [7] omega_y:     ", gpu_obs[7])
        print("  [8] hip_omega:   ", gpu_obs[8])
        print("  [9] knee_omega:  ", gpu_obs[9])
        print("  [10] ankle_omega:", gpu_obs[10])
        print()

        # Now step CPU once for fair comparison
        var zero_action = List[Scalar[DType.float64]]()
        for _ in range(3):
            zero_action.append(Scalar[DType.float64](0.0))
        var cpu_result = cpu_env.step_continuous_vec(zero_action)
        var cpu_obs_step1 = cpu_result[0].copy()

        print("CPU Observations (after 1 step with zero action):")
        print("  [0] torso_z:     ", cpu_obs_step1[0])
        print("  [1] torso_pitch: ", cpu_obs_step1[1])
        print("  [2] hip_angle:   ", cpu_obs_step1[2])
        print("  [3] knee_angle:  ", cpu_obs_step1[3])
        print("  [4] ankle_angle: ", cpu_obs_step1[4])
        print("  [5] vel_x:       ", cpu_obs_step1[5])
        print("  [6] vel_z:       ", cpu_obs_step1[6])
        print("  [7] omega_y:     ", cpu_obs_step1[7])
        print("  [8] hip_omega:   ", cpu_obs_step1[8])
        print("  [9] knee_omega:  ", cpu_obs_step1[9])
        print("  [10] ankle_omega:", cpu_obs_step1[10])
        print()

        # Print differences
        print("Differences (GPU - CPU):")
        var names = List[String]()
        names.append("torso_z")
        names.append("torso_pitch")
        names.append("hip_angle")
        names.append("knee_angle")
        names.append("ankle_angle")
        names.append("vel_x")
        names.append("vel_z")
        names.append("omega_y")
        names.append("hip_omega")
        names.append("knee_omega")
        names.append("ankle_omega")

        for i in range(OBS_DIM):
            var diff = Float64(gpu_obs[i]) - Float64(cpu_obs_step1[i])
            print("  [", i, "]", names[i], ":", diff)
