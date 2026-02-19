"""Test Ant reset_kernel_gpu to isolate the crash."""

from gpu.host import DeviceContext
from envs.ant import Ant
from deep_rl import dtype as gpu_dtype

comptime AntEnv = Ant[gpu_dtype, True]
comptime N_ENVS = 4


fn main() raises:
    print("=== Test Ant GPU Reset ===")
    print("STATE_SIZE:", AntEnv.STATE_SIZE)
    print("STEP_WS_SHARED:", AntEnv.STEP_WS_SHARED)

    with DeviceContext() as ctx:
        print("Creating states buffer...")
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * AntEnv.STATE_SIZE
        )
        print("Calling reset_kernel_gpu...")
        AntEnv.reset_kernel_gpu[N_ENVS, AntEnv.STATE_SIZE](ctx, states_buf)
        ctx.synchronize()
        print("reset_kernel_gpu completed!")
        print("=== PASSED ===")
