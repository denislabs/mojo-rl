"""GRUCell CPU↔GPU parity (audit L1 — GPU path).

Builds a CPU and a GPU GRUCell, copies the CPU weights into the GPU
device buffers (so params are identical by construction — no init-RNG
coupling), then runs the SAME (x, h) through both forward + vjp and
asserts:

  * forward output matches,
  * grad_x / grad_h match,
  * param grads (dW_ih / dW_hh / db_ih / db_hh) match.

GPU uses block-reduction sums (different summation order than the CPU
sequential loops), so the tolerance is 1e-4, not bit-exact.

Run:  pixi run -e apple mojo run -I . tests/nn/test_gru_cell_gpu_parity.mojo
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.gru_cell import GRUCell
from mojo_rl.nn.initializer import Kaiming


# IN_DIM == H: GRUCell.forward / vjp take a homogeneous TileTensor
# variadic, so the (x, h) inputs must share shape — the same constraint
# the existing CPU GRU tests run under. The kernels still treat IN_/H as
# independent comptime dims, so this exercises the full GPU path.
comptime IN_DIM = 4
comptime H = 4
comptime BATCH = 3
comptime TOL: Scalar[DT] = 1e-4


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _upload(
    ctx: DeviceContext,
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: DeviceBuffer[DT],
    n: Int,
) raises:
    var hb = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for k in range(n):
        hb.unsafe_ptr()[k] = src[k]
    ctx.enqueue_copy(dst, hb)
    ctx.synchronize()


def _download(
    ctx: DeviceContext, src: DeviceBuffer[DT], n: Int
) raises -> List[Scalar[DT]]:
    var hb = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    ctx.enqueue_copy(hb, src)
    ctx.synchronize()
    var out = List[Scalar[DT]](length=n, fill=0.0)
    for k in range(n):
        out[k] = hb.unsafe_ptr()[k]
    return out^


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin], b: List[Scalar[DT]], n: Int
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for k in range(n):
        var d = fabs(a[k] - b[k])
        if d > m:
            m = d
    return m


def main() raises:
    print("=" * 70)
    print("GRUCell CPU↔GPU parity (L1)")
    print("=" * 70)
    seed(3)
    var ctx = DeviceContext()

    var cpu = GRUCell[IN_DIM, H].make[target="cpu", INIT=Kaiming]()
    var gpu = GRUCell[IN_DIM, H].make[target="gpu", INIT=Kaiming](ctx)

    # Make params identical: copy CPU → GPU device buffers.
    _upload(ctx, cpu.W_ih.value_unsafe_ptr_cpu(), gpu.W_ih.val.dev.value(), cpu.W_IH_SIZE)
    _upload(ctx, cpu.W_hh.value_unsafe_ptr_cpu(), gpu.W_hh.val.dev.value(), cpu.W_HH_SIZE)
    _upload(ctx, cpu.b_ih.value_unsafe_ptr_cpu(), gpu.b_ih.val.dev.value(), cpu.B_IH_SIZE)
    _upload(ctx, cpu.b_hh.value_unsafe_ptr_cpu(), gpu.b_hh.val.dev.value(), cpu.B_IH_SIZE)

    # ---- Host inputs (shared) ----
    var x_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var h_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var go_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    for k in range(BATCH * IN_DIM):
        x_h[k] = Scalar[DT](-0.3 + 0.17 * Float64(k))
    for k in range(BATCH * H):
        h_h[k] = Scalar[DT](0.2 - 0.09 * Float64(k))
        go_h[k] = Scalar[DT](0.5 + 0.11 * Float64(k))  # non-uniform upstream grad

    # ---- CPU forward + vjp ----
    var cpu_out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cpu_dx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var cpu_dh: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    cpu.zero_grad["cpu"]()
    var cx = TileTensor(x_h, row_major[BATCH, IN_DIM]())
    var ch = TileTensor(h_h, row_major[BATCH, H]())
    var co = TileTensor(cpu_out, row_major[BATCH, H]())
    cpu.forward["cpu", BATCH](
            TensorPack[2].of(cx, ch), output=co,
        )
    var cgo = TileTensor(go_h, row_major[BATCH, H]())
    var cdx = TileTensor(cpu_dx, row_major[BATCH, IN_DIM]())
    var cdh = TileTensor(cpu_dh, row_major[BATCH, H]())
    cpu.vjp["cpu", BATCH](cgo, TensorPack[2].of(cdx, cdh))

    # ---- GPU forward + vjp ----
    var x_d = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    var h_d = ctx.enqueue_create_buffer[DT](BATCH * H)
    var o_d = ctx.enqueue_create_buffer[DT](BATCH * H)
    var go_d = ctx.enqueue_create_buffer[DT](BATCH * H)
    var dx_d = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    var dh_d = ctx.enqueue_create_buffer[DT](BATCH * H)
    _upload(ctx, x_h, x_d, BATCH * IN_DIM)
    _upload(ctx, h_h, h_d, BATCH * H)
    _upload(ctx, go_h, go_d, BATCH * H)

    gpu.zero_grad["gpu"]()
    var gx = TileTensor(_p(x_d), row_major[BATCH, IN_DIM]())
    var gh = TileTensor(_p(h_d), row_major[BATCH, H]())
    var go = TileTensor(_p(o_d), row_major[BATCH, H]())
    gpu.forward["gpu", BATCH](
            TensorPack[2].of(gx, gh), output=go,
        )
    var ggo = TileTensor(_p(go_d), row_major[BATCH, H]())
    var gdx = TileTensor(_p(dx_d), row_major[BATCH, IN_DIM]())
    var gdh = TileTensor(_p(dh_d), row_major[BATCH, H]())
    gpu.vjp["gpu", BATCH](ggo, TensorPack[2].of(gdx, gdh))
    ctx.synchronize()

    # ---- Compare ----
    var out_diff = _maxdiff(cpu_out, _download(ctx, o_d, BATCH * H), BATCH * H)
    var dx_diff = _maxdiff(cpu_dx, _download(ctx, dx_d, BATCH * IN_DIM), BATCH * IN_DIM)
    var dh_diff = _maxdiff(cpu_dh, _download(ctx, dh_d, BATCH * H), BATCH * H)
    print("  max|forward|  =", out_diff)
    print("  max|grad_x|   =", dx_diff)
    print("  max|grad_h|   =", dh_diff)
    assert_true(out_diff < TOL, "forward CPU/GPU mismatch")
    assert_true(dx_diff < TOL, "grad_x CPU/GPU mismatch")
    assert_true(dh_diff < TOL, "grad_h CPU/GPU mismatch")

    var dwih_diff = _maxdiff(
        cpu.W_ih.grad_unsafe_ptr_cpu(),
        _download(ctx, gpu.W_ih.grd.dev.value(), cpu.W_IH_SIZE),
        cpu.W_IH_SIZE,
    )
    var dwhh_diff = _maxdiff(
        cpu.W_hh.grad_unsafe_ptr_cpu(),
        _download(ctx, gpu.W_hh.grd.dev.value(), cpu.W_HH_SIZE),
        cpu.W_HH_SIZE,
    )
    var dbih_diff = _maxdiff(
        cpu.b_ih.grad_unsafe_ptr_cpu(),
        _download(ctx, gpu.b_ih.grd.dev.value(), cpu.B_IH_SIZE),
        cpu.B_IH_SIZE,
    )
    var dbhh_diff = _maxdiff(
        cpu.b_hh.grad_unsafe_ptr_cpu(),
        _download(ctx, gpu.b_hh.grd.dev.value(), cpu.B_IH_SIZE),
        cpu.B_IH_SIZE,
    )
    print("  max|dW_ih|    =", dwih_diff)
    print("  max|dW_hh|    =", dwhh_diff)
    print("  max|db_ih|    =", dbih_diff)
    print("  max|db_hh|    =", dbhh_diff)
    assert_true(dwih_diff < TOL, "dW_ih CPU/GPU mismatch")
    assert_true(dwhh_diff < TOL, "dW_hh CPU/GPU mismatch")
    assert_true(dbih_diff < TOL, "db_ih CPU/GPU mismatch")
    assert_true(dbhh_diff < TOL, "db_hh CPU/GPU mismatch")

    x_h.free(); h_h.free(); go_h.free()
    cpu_out.free(); cpu_dx.free(); cpu_dh.free()
    _ = cpu^
    _ = gpu^
    print("=" * 70)
    print("PASS — GRUCell CPU/GPU parity (forward + grads)")
    print("=" * 70)
