"""Gate for the zero-series adapter boundary (gap #1): bridge an MCTS planner's
EXTERNALLY-owned device buffers into the storage `Module` surface.

Two device-buffer wrap strategies are probed here:

  (1) `Tensor.view_gpu` — a NON-OWNING view (zero-copy). PROVEN to work as a
      single kernel operand, but TWO simultaneous non-owning views as operands
      of one kernel miscompile on Metal (deterministic prefix-drop — same
      exclusivity/wildcard-origin class as the prior ExternalRef GPU bug).

  (2) `Tensor.copy_from_device` / `copy_to_device` — OWNED scratch + a small
      device→device `enqueue_copy` at the boundary (LeWM's proven approach).
      Robust for the input+output case the adapters actually need.

This gate asserts: (1) single non-owning operand is correct, and (2) the
copy-at-boundary path matches an owned reference.

Run (Apple Metal): pixi run -e apple mojo run -I . \
    mojo_rl/nn/storage/spikes/spike_view_gpu.mojo
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


def _scale_kernel[
    N: Int
](
    inp: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    outp: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        outp[idx] = rebind[Scalar[DT]](inp[idx]) * Scalar[DT](2)


def scale[
    N: Int
](ctx: DeviceContext, mut t_in: Tensor, mut t_out: Tensor) raises:
    comptime layout = Layout.row_major(N)
    var il = t_in.lt["gpu", layout]()
    var ol = t_out.lt["gpu", layout]()
    comptime nblk = (N + 255) // 256
    ctx.enqueue_function[_scale_kernel[N]](il, ol, grid_dim=nblk, block_dim=256)
    ctx.synchronize()


def _ptr_of(mut buf: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())


def main() raises:
    comptime N = 32
    comptime layout = Layout.row_major(N)
    var ctx = DeviceContext()

    # ----- owned reference -----
    var ref_in = Tensor.alloc(N)
    for i in range(N):
        ref_in.data[i] = Scalar[DT](i + 1)
    ref_in.upload(ctx)
    var ref_out = Tensor.alloc_gpu(ctx, N)
    scale[N](ctx, ref_in, ref_out)
    ref_out.download(ctx)

    # planner-owned buffers
    var ext_in = ctx.enqueue_create_buffer[DT](N)
    var ext_out = ctx.enqueue_create_buffer[DT](N)
    ext_out.enqueue_fill(Scalar[DT](0))
    var h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        h[i] = Scalar[DT](i + 1)
    ctx.enqueue_copy(ext_in, h)
    ctx.synchronize()

    # ----- strategy (1): single non-owning operand (output), owned input -----
    var s1_in = Tensor.alloc(N)
    for i in range(N):
        s1_in.data[i] = Scalar[DT](i + 1)
    s1_in.upload(ctx)
    var s1_out = Tensor.view_gpu(ctx, _ptr_of(ext_out), N)
    scale[N](ctx, s1_in, s1_out)
    var hb1 = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(hb1, ext_out)
    ctx.synchronize()
    var d1: Scalar[DT] = 0
    for i in range(N):
        var d = ref_out.data[i] - hb1[i]
        d1 = max(d1, d if d >= 0 else -d)

    # ----- strategy (2): owned scratch + D2D copy at the boundary -----
    ext_out.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    var sc_in = Tensor.alloc_gpu(ctx, N)
    var sc_out = Tensor.alloc_gpu(ctx, N)
    sc_in.copy_from_device(ctx, _ptr_of(ext_in), N)   # planner in  -> scratch
    scale[N](ctx, sc_in, sc_out)
    sc_out.copy_to_device(ctx, _ptr_of(ext_out), N)   # scratch out -> planner
    var hb2 = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(hb2, ext_out)
    ctx.synchronize()
    var d2: Scalar[DT] = 0
    for i in range(N):
        var d = ref_out.data[i] - hb2[i]
        d2 = max(d2, d if d >= 0 else -d)

    print("strategy(1) single non-owning operand  max|ref-got|:", d1)
    print("strategy(2) copy-at-boundary           max|ref-got|:", d2)
    if d1 == Scalar[DT](0) and d2 == Scalar[DT](0):
        print("ADAPTER-BOUNDARY OK (view_gpu single-operand + copy-at-boundary)")
    else:
        print("ADAPTER-BOUNDARY FAIL")
