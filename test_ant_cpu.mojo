"""Test HalfCheetah and Ant GPU reset+FK."""

from gpu.host import DeviceContext
from envs.half_cheetah import HalfCheetah
from envs.ant import Ant
from deep_rl import dtype as gpu_dtype

comptime HCEnv = HalfCheetah[gpu_dtype, True]
comptime AntEnv = Ant[gpu_dtype, True]
comptime N_ENVS = 4


fn main() raises:
    with DeviceContext() as ctx:
        # Test HalfCheetah first
        print("=== HalfCheetah reset+FK ===")
        var hc_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * HCEnv.STATE_SIZE
        )
        HCEnv.reset_kernel_gpu[N_ENVS, HCEnv.STATE_SIZE](ctx, hc_buf)
        ctx.synchronize()
        print("HalfCheetah PASSED!")

        # Test Ant
        print("=== Ant reset+FK ===")
        var ant_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * AntEnv.STATE_SIZE
        )
        AntEnv.reset_kernel_gpu[N_ENVS, AntEnv.STATE_SIZE](ctx, ant_buf)
        ctx.synchronize()
        print("Ant PASSED!")
