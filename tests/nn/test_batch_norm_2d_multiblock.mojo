"""BatchNorm2D multi-block GPU kernel — CPU↔GPU parity at the large-G path.

The `test_nn_gpu_parity` BN2D case uses BATCH=4 (→ G=4, 1 row/shard), which
checks the multi-block split but not the realistic regime. This drives the
rewrite at BATCH=128 → G=BN2D_RBLOCKS=64 with **bpb=2** (two batch rows per
reduction shard) and bigger C/SPATIAL, so the partial→finalize→normalize (fwd)
and partial→finalize→scatter (bwd) reductions are exercised the way the EZv2
rep ResNet uses them. Compares forward output, running stats, grad_input, and
grad_gamma/beta against the CPU two-pass reference within ~1e-3 (the new GPU
path uses the Σx/Σx² one-pass var, so it is close-but-not-bit-identical).

Run:
    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_multiblock.mojo
"""

from std.memory import alloc
from std.math import abs as _abs
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D


comptime BATCH = 128          # → G = min(128, 64) = 64, bpb = 2
comptime C = 32
comptime HH = 12
comptime WW = 12
comptime FLAT = C * HH * WW   # 4608
comptime N = BATCH * FLAT     # 589824
comptime ATOL = Scalar[DT](1e-3)


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Scalar[DT]:
    var m = Scalar[DT](0.0)
    for i in range(n):
        var d = _abs(a[i] - b[i])
        if d > m:
            m = d
    return m


def main() raises:
    print("=" * 70)
    print("BatchNorm2D multi-block CPU↔GPU parity (BATCH=128, C=32, 12×12)")
    print("=" * 70)
    var ctx = DeviceContext()

    var bn_cpu = BatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    var bn_gpu = BatchNorm2D[C, HH, WW].make[target="gpu", INIT=Zero](ctx)
    bn_cpu.training = True
    bn_gpu.training = True

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    # bounded, varied input (keeps batch stats moderate so the one-pass var is
    # well-conditioned — the realistic post-conv regime).
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](Float64((i % 97) - 48) * 0.05)
        go_h.unsafe_ptr()[i] = Scalar[DT](Float64((i % 31) - 15) * 0.04)

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x_cpu[i] = x_h.unsafe_ptr()[i]
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, FLAT]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, FLAT]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, FLAT]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, FLAT]())
    bn_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    bn_cpu.vjp["cpu", BATCH](goc_t, gic_t)

    var x_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var go_d = ctx.enqueue_create_buffer[DT](N)
    var gi_d = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(go_d, go_h)
    var xg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_d.unsafe_ptr())
    var yg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_d.unsafe_ptr())
    var gog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_d.unsafe_ptr())
    var gig_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_d.unsafe_ptr())
    var xg_t = TileTensor(xg_p, row_major[BATCH, FLAT]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, FLAT]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, FLAT]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, FLAT]())
    bn_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    bn_gpu.vjp["gpu", BATCH](gog_t, gig_t)

    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    var dg_h = ctx.enqueue_create_host_buffer[DT](C)
    var db_h = ctx.enqueue_create_host_buffer[DT](C)
    var rm_h = ctx.enqueue_create_host_buffer[DT](C)
    var rv_h = ctx.enqueue_create_host_buffer[DT](C)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.enqueue_copy(dg_h, bn_gpu.gamma.grd.dev.value())
    ctx.enqueue_copy(db_h, bn_gpu.beta.grd.dev.value())
    ctx.enqueue_copy(rm_h, bn_gpu.running_mean.t.dev.value())
    ctx.enqueue_copy(rv_h, bn_gpu.running_var.t.dev.value())
    ctx.synchronize()

    var dy = _maxdiff(y_cpu, y_h.unsafe_ptr(), N)
    var dgi = _maxdiff(gi_cpu, gi_h.unsafe_ptr(), N)
    var max_dg = Scalar[DT](0.0)
    var max_db = Scalar[DT](0.0)
    var max_rm = Scalar[DT](0.0)
    var max_rv = Scalar[DT](0.0)
    for f in range(C):
        var ed = _abs(bn_cpu.gamma.grd.cpu[f] - dg_h.unsafe_ptr()[f])
        var eb = _abs(bn_cpu.beta.grd.cpu[f] - db_h.unsafe_ptr()[f])
        var er = _abs(bn_cpu.running_mean.t.cpu[f] - rm_h.unsafe_ptr()[f])
        var ev = _abs(bn_cpu.running_var.t.cpu[f] - rv_h.unsafe_ptr()[f])
        if ed > max_dg: max_dg = ed
        if eb > max_db: max_db = eb
        if er > max_rm: max_rm = er
        if ev > max_rv: max_rv = ev
    print("  max|y|=", dy, " max|gi|=", dgi, " max|dγ|=", max_dg,
          " max|dβ|=", max_db, " max|run_mean|=", max_rm,
          " max|run_var|=", max_rv)
    assert_true(dy < ATOL, "fwd output parity")
    assert_true(dgi < ATOL, "grad_input parity")
    assert_true(max_dg < ATOL, "grad_gamma parity")
    assert_true(max_db < ATOL, "grad_beta parity")
    assert_true(max_rm < ATOL, "running_mean parity")
    assert_true(max_rv < ATOL, "running_var parity")

    x_cpu.free(); y_cpu.free(); go_cpu.free(); gi_cpu.free()
    _ = bn_cpu^
    _ = bn_gpu^
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
