"""Test Ant init_step_workspace_gpu isolation."""

from gpu.host import DeviceContext
from envs.ant import Ant
from deep_rl import dtype as gpu_dtype

comptime AntEnv = Ant[gpu_dtype, True]
comptime N_ENVS = 4

fn main() raises:
    with DeviceContext() as ctx:
        print("STEP_WS_SHARED:", AntEnv.STEP_WS_SHARED)
        print("STEP_WS_PER_ENV:", AntEnv.STEP_WS_PER_ENV)
        comptime TOTAL_WS = AntEnv.STEP_WS_SHARED + N_ENVS * AntEnv.STEP_WS_PER_ENV
        print("TOTAL_WS:", TOTAL_WS)

        print("Creating workspace buffer...")
        var workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](TOTAL_WS)
        print("Buffer created.")

        print("Calling init_step_workspace_gpu...")
        AntEnv.init_step_workspace_gpu[N_ENVS](ctx, workspace_buf)
        print("init called, synchronizing...")
        ctx.synchronize()
        print("PASSED!")
