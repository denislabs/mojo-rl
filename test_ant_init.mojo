"""Test Ant init_model_gpu isolation."""

from gpu.host import DeviceContext
from envs.ant.ant_def import AntModel
from deep_rl import dtype as gpu_dtype
from physics3d.gpu.constants import model_size_with_invweight

fn main() raises:
    with DeviceContext() as ctx:
        comptime MODEL_SIZE = model_size_with_invweight[
            AntModel.NBODY,
            AntModel.NJOINT,
            AntModel.NV,
            AntModel.NGEOM,
        ]()
        print("MODEL_SIZE:", MODEL_SIZE)

        print("Creating device buffer...")
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        print("Buffer created.")

        print("Calling init_model_gpu...")
        AntModel.init_model_gpu[gpu_dtype](ctx, model_buf)
        print("init_model_gpu returned.")

        ctx.synchronize()
        print("PASSED!")
