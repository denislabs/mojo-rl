"""Test Ant GPU step kernel isolation."""

from gpu.host import DeviceContext
from memory import UnsafePointer
from envs.ant import Ant
from deep_rl import dtype as gpu_dtype

comptime AntEnv = Ant[gpu_dtype, True]
comptime N_ENVS = 4


fn main() raises:
    with DeviceContext() as ctx:
        # Step 1: Init workspace FIRST (before reset to avoid stack overflow)
        print("Step 1: Init workspace...")
        print("  STEP_WS_SHARED:", AntEnv.STEP_WS_SHARED)
        print("  STEP_WS_PER_ENV:", AntEnv.STEP_WS_PER_ENV)
        comptime TOTAL_WS = AntEnv.STEP_WS_SHARED + N_ENVS * AntEnv.STEP_WS_PER_ENV
        print("  TOTAL_WS:", TOTAL_WS)
        var workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](TOTAL_WS)
        AntEnv.init_step_workspace_gpu[N_ENVS](ctx, workspace_buf)
        ctx.synchronize()
        print("  Workspace init PASSED!")

        # Step 2: Reset
        print("Step 2: Reset...")
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * AntEnv.STATE_SIZE
        )
        AntEnv.reset_kernel_gpu[N_ENVS, AntEnv.STATE_SIZE](ctx, states_buf)
        ctx.synchronize()
        print("  Reset PASSED!")

        # Step 3: Extract obs
        print("Step 3: Extract obs...")
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * AntEnv.OBS_DIM
        )
        AntEnv.extract_obs_kernel_gpu[N_ENVS, AntEnv.STATE_SIZE, AntEnv.OBS_DIM](
            ctx, states_buf, obs_buf
        )
        ctx.synchronize()
        print("  Extract obs PASSED!")

        # Step 4: Step with zero actions
        print("Step 4: Step with zero actions...")
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * AntEnv.ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        AntEnv.step_kernel_gpu[
            N_ENVS,
            AntEnv.STATE_SIZE,
            AntEnv.OBS_DIM,
            AntEnv.ACTION_DIM,
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        ctx.synchronize()
        print("  Step PASSED!")

        print("ALL TESTS PASSED!")
