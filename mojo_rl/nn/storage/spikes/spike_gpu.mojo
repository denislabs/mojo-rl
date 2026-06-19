"""GPU proof for the storage-passing design — binary add, CPU vs GPU parity.

Same `ref Tensor` / `mut Tensor` surface as the CPU slice. The GPU path builds
a device `LayoutTensor` INTERNALLY via `Tensor.lt["gpu", layout]()` and launches a
kernel; the kernel args are `MutAnyOrigin` (the GPU ABI boundary — the one,
expected erasure). No origin params on the storage surface either way.

Run (Apple Metal): pixi run -e apple mojo run -I . \
    mojo_rl/nn/storage/spike_gpu.mojo
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor


def _add_kernel[
    B: Int, DIM: Int
](
    a: LayoutTensor[DT, Layout.row_major(B, DIM), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(B, DIM), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < B * DIM:
        var i = idx // DIM
        var d = idx % DIM
        o[i, d] = rebind[Scalar[DT]](a[i, d]) + rebind[Scalar[DT]](b[i, d])


# Storage-passing surface — identical shape to the CPU leaves.
def add_cpu[B: Int, DIM: Int](ref a: Tensor, ref b: Tensor, mut out: Tensor):
    out.ensure(B * DIM)
    for i in range(B * DIM):
        out.data[i] = a.data[i] + b.data[i]


def add_gpu[
    B: Int, DIM: Int
](ctx: DeviceContext, mut a: Tensor, mut b: Tensor, mut out: Tensor) raises:
    # inputs are `mut` so the origin-linking `lt["gpu", …](mut self)` ctor can
    # supply the MutAnyOrigin device view (the buffer is read, not written).
    # In the trait path, `inputs[k]` is already a mutable MutAnyOrigin ref, so
    # it composes without this `mut`.
    out.ensure_gpu(ctx, B * DIM)
    comptime layout = Layout.row_major(B, DIM)
    var al = a.lt["gpu", layout]()
    var bl = b.lt["gpu", layout]()
    var ol = out.lt["gpu", layout]()
    comptime kernel = _add_kernel[B, DIM]
    comptime nblk = (B * DIM + 255) // 256
    ctx.enqueue_function[kernel](al, bl, ol, grid_dim=nblk, block_dim=256)
    ctx.synchronize()


def main() raises:
    comptime B = 4
    comptime DIM = 8
    var ctx = DeviceContext()

    var a = Tensor.alloc(B * DIM)
    var b = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        a.data[i] = Scalar[DT](i + 1)
        b.data[i] = Scalar[DT](10 + i)

    # CPU reference.
    var out_cpu = Tensor.alloc(B * DIM)
    add_cpu[B, DIM](a, b, out_cpu)

    # GPU: upload inputs, run through the same ref-Tensor surface, download.
    a.upload(ctx)
    b.upload(ctx)
    var out_gpu = Tensor.alloc_gpu(ctx, B * DIM)
    add_gpu[B, DIM](ctx, a, b, out_gpu)
    out_gpu.download(ctx)

    var maxdiff: Scalar[DT] = 0
    for i in range(B * DIM):
        var d = out_cpu.data[i] - out_gpu.data[i]
        if d < 0:
            d = -d
        if d > maxdiff:
            maxdiff = d
    print("cpu[0],cpu[31]:", out_cpu.data[0], out_cpu.data[B * DIM - 1])
    print("gpu[0],gpu[31]:", out_gpu.data[0], out_gpu.data[B * DIM - 1])
    print("max|cpu-gpu|:", maxdiff)
    if maxdiff == Scalar[DT](0):
        print("GPU PARITY OK — storage surface, kernel-arg MutAnyOrigin only")
    else:
        print("GPU PARITY FAIL")
