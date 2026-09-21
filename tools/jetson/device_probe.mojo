# What the MAX runtime believes the GPU is — name, compute capability, arch.
#
# Run on the board through the jetson env (the interceptor's NVML shims must
# be preloaded, and the compile target set):
#
#     pixi run -e jetson run-jetson tools/jetson/device_probe.mojo
#
# The arch printed here is what the run-time `ptxas` step targets. If it is
# not `sm_87` on an Orin, every kernel load fails with
# CUDA_ERROR_INVALID_SOURCE ("device kernel image is invalid"), whatever
# `--target-accelerator` said at build time: the PTX MAX embeds for sm_87 is
# `.target sm_80`, generic within the major, and the real arch is picked here.
from max.gpu.host import DeviceContext


def main() raises:
    var ctx = DeviceContext()
    print("api                :", ctx.api())
    print("name               :", ctx.name())
    print("compute_capability :", ctx.compute_capability())
    print("arch_name          :", ctx.arch_name())
    print("compile-time info  :", ctx.default_device_info.name, "compute", ctx.default_device_info.compute)
    var buf = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.synchronize()
    print("buffer of 4 floats : ok")
