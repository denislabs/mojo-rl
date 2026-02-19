"""Test Ant: reversed order - init_step_workspace first, then reset."""

from gpu.host import DeviceContext
from envs.ant import Ant
from deep_rl import dtype as gpu_dtype

comptime AntEnv = Ant[gpu_dtype, True]
comptime N_ENVS = 4

fn main() raises:
    with DeviceContext() as ctx:
        # Step 1: Init workspace FIRST
        print("Step 1: Init workspace...")
        comptime TOTAL_WS = AntEnv.STEP_WS_SHARED + N_ENVS * AntEnv.STEP_WS_PER_ENV
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

        print("ALL PASSED!")
