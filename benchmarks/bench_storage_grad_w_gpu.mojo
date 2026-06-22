"""grad_w backward microbench: is the missing `transpose_a` worth chasing?

Thin launcher — all kernels, timing, and the (heavy) `max_matmul` shape
instantiations live in the cached package module
`mojo_rl/nn/storage/primitives/linear_transpose_a.mojo`, so this file stays
tiny and the package compile is cached in `.mojoc` across runs. See that
module's docstring for the variant definitions and read-out.

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_grad_w_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_grad_w_gpu.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.storage.primitives.linear_transpose_a import run_grad_w_bench


def main() raises:
    var ctx = DeviceContext()
    run_grad_w_bench(ctx)
