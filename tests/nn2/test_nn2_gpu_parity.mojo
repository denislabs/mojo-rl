"""GPU parity smoke tests for the Phase 2 / 4 / 5 nn2 primitives.

For each layer that grew a GPU path in `PORTING_PLAN.md`'s consumer-gated
follow-ups, build a CPU instance and a GPU instance with identical
parameters, run the same forward + backward, and check that the GPU
output / grad_input / param grads match the CPU within fp32 noise.

This is a parity test — not a separate correctness test. The CPU paths
already have their own FD gradcheck / analytic-reference tests; the
GPU paths inherit correctness via parity.

Covered:
    - SimNorm[DIM, GROUPS]
    - MinMaxNorm[DIM]
    - Dropout[DIM, p, SEED] (train + eval)
    - BatchNorm1D[DIM]
    - BatchNorm2D[C, H, W]
    - MaxPool2D[C, K, S, P, H, W]
    - AvgPool2D[C, K, S, P, H, W]
    - Conv2D[IC, OC, K, S, P, H, W]
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming
from mojo_rl.nn2.primitives.sim_norm import SimNorm
from mojo_rl.nn2.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn2.primitives.dropout import Dropout
from mojo_rl.nn2.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn2.primitives.batch_norm_2d import BatchNorm2D
from mojo_rl.nn2.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn2.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn2.primitives.conv2d import Conv2D


comptime PARITY_ATOL: Scalar[DT] = 5e-5


def _abs(v: Scalar[DT]) -> Scalar[DT]:
    return v if v >= Scalar[DT](0) else -v


def _max_diff(
    a_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = _abs(a_p[i] - b_p[i])
        if d > m:
            m = d
    return m


def test_sim_norm_gpu_parity() raises:
    print("test_sim_norm_gpu_parity ...")
    comptime BATCH = 4
    comptime DIM = 16
    comptime GROUPS = 4
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var s_cpu = SimNorm[DIM, GROUPS].make[target="cpu", INIT=Zero]()
    var s_gpu = SimNorm[DIM, GROUPS].make[target="gpu", INIT=Zero](ctx)

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](-1.0 + 0.13 * Float64(i))
        go_h.unsafe_ptr()[i] = Scalar[DT](0.5 + 0.07 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x_cpu[i] = x_h.unsafe_ptr()[i]
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, DIM]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, DIM]())
    s_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    s_cpu.vjp["cpu", BATCH](goc_t, gic_t)

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
    var xg_t = TileTensor(xg_p, row_major[BATCH, DIM]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, DIM]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, DIM]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, DIM]())
    s_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    s_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N)
    print("  max|y_cpu-y_gpu| =", dy, "  max|gi_cpu-gi_gpu| =", dgi)
    assert_true(dy < PARITY_ATOL, "SimNorm GPU forward parity failed")
    assert_true(dgi < PARITY_ATOL, "SimNorm GPU backward parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_min_max_norm_gpu_parity() raises:
    print("test_min_max_norm_gpu_parity ...")
    comptime BATCH = 3
    comptime DIM = 32
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var m_cpu = MinMaxNorm[DIM].make[target="cpu", INIT=Zero]()
    var m_gpu = MinMaxNorm[DIM].make[target="gpu", INIT=Zero](ctx)

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](-1.5 + 0.21 * Float64(i % 17))
        go_h.unsafe_ptr()[i] = Scalar[DT](0.3 + 0.05 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x_cpu[i] = x_h.unsafe_ptr()[i]
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, DIM]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, DIM]())
    m_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    m_cpu.vjp["cpu", BATCH](goc_t, gic_t)

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
    var xg_t = TileTensor(xg_p, row_major[BATCH, DIM]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, DIM]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, DIM]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, DIM]())
    m_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    m_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N)
    print("  max|y_cpu-y_gpu| =", dy, "  max|gi_cpu-gi_gpu| =", dgi)
    assert_true(dy < PARITY_ATOL, "MinMaxNorm GPU forward parity failed")
    assert_true(dgi < PARITY_ATOL, "MinMaxNorm GPU backward parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_dropout_gpu_eval_identity() raises:
    """In eval mode Dropout is the identity — GPU should match input."""
    print("test_dropout_gpu_eval_identity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var d_gpu = Dropout[DIM, 0.3, 42].make[
        target="gpu", INIT=Zero,
    ](ctx)
    d_gpu.training = False

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](0.5 + 0.1 * Float64(i))
        go_h.unsafe_ptr()[i] = Scalar[DT](2.0 - 0.05 * Float64(i))

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
    var x_t = TileTensor(xg_p, row_major[BATCH, DIM]())
    var y_t = TileTensor(yg_p, row_major[BATCH, DIM]())
    var go_t = TileTensor(gog_p, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gig_p, row_major[BATCH, DIM]())
    d_gpu.forward["gpu", BATCH](x_t, output=y_t)
    d_gpu.vjp["gpu", BATCH](go_t, gi_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()
    var dy = _max_diff(x_h.unsafe_ptr(), y_h.unsafe_ptr(), N)
    var dgi = _max_diff(go_h.unsafe_ptr(), gi_h.unsafe_ptr(), N)
    print("  eval: max|y-x| =", dy, "  max|gi-go| =", dgi)
    assert_true(dy == Scalar[DT](0.0), "Dropout eval should be identity")
    assert_true(dgi == Scalar[DT](0.0), "Dropout eval grad should be identity")
    print("  ok")


def test_dropout_gpu_train_mask_stats() raises:
    """In train mode the mask should be Bernoulli(1-p); check:
    (1) backward equals grad_y * mask (with x ≡ 1, mask = y).
    (2) drop fraction is approximately p over many lanes."""
    print("test_dropout_gpu_train_mask_stats ...")
    comptime BATCH = 64
    comptime DIM = 256
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var d_gpu = Dropout[DIM, 0.3, 123].make[
        target="gpu", INIT=Zero,
    ](ctx)
    d_gpu.training = True

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](1.0)
        go_h.unsafe_ptr()[i] = Scalar[DT](2.0)

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
    var x_t = TileTensor(xg_p, row_major[BATCH, DIM]())
    var y_t = TileTensor(yg_p, row_major[BATCH, DIM]())
    var go_t = TileTensor(gog_p, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gig_p, row_major[BATCH, DIM]())
    d_gpu.forward["gpu", BATCH](x_t, output=y_t)
    d_gpu.vjp["gpu", BATCH](go_t, gi_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()

    var scale = Scalar[DT](1.0 / (1.0 - 0.3))
    var n_zero: Int = 0
    var n_scale: Int = 0
    var max_grad_err: Scalar[DT] = 0.0
    for i in range(N):
        var v = y_h.unsafe_ptr()[i]
        if v == Scalar[DT](0.0):
            n_zero += 1
        elif _abs(v - scale) < Scalar[DT](1e-5):
            n_scale += 1
        # x ≡ 1, so mask = y. gi = go * mask = 2.0 * y.
        var exp_gi = Scalar[DT](2.0) * v
        var err = _abs(gi_h.unsafe_ptr()[i] - exp_gi)
        if err > max_grad_err:
            max_grad_err = err
    var frac_zero = Float64(n_zero) / Float64(N)
    print(
        "  frac_zero =", frac_zero,
        " (target p=0.3)  max|gi - go·mask| =", max_grad_err,
    )
    assert_true(
        n_zero + n_scale == N,
        "Dropout mask values should be only 0 or 1/(1-p)",
    )
    assert_true(
        max_grad_err < Scalar[DT](1e-5),
        "Dropout gradient must equal go · mask",
    )
    assert_true(
        Scalar[DT](frac_zero) > Scalar[DT](0.27)
        and Scalar[DT](frac_zero) < Scalar[DT](0.33),
        "Dropout frac_zero out of 3σ band",
    )
    print("  ok")


def test_batch_norm_1d_gpu_parity() raises:
    print("test_batch_norm_1d_gpu_parity ...")
    comptime BATCH = 8
    comptime DIM = 6
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var bn_cpu = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    var bn_gpu = BatchNorm1D[DIM].make[target="gpu", INIT=Zero](ctx)
    bn_cpu.training = True
    bn_gpu.training = True

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](-0.7 + 0.11 * Float64(i))
        go_h.unsafe_ptr()[i] = Scalar[DT](0.5 + 0.07 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x_cpu[i] = x_h.unsafe_ptr()[i]
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, DIM]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, DIM]())
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
    var xg_t = TileTensor(xg_p, row_major[BATCH, DIM]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, DIM]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, DIM]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, DIM]())
    bn_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    bn_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N)
    var dg_h = ctx.enqueue_create_host_buffer[DT](DIM)
    var db_h = ctx.enqueue_create_host_buffer[DT](DIM)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.enqueue_copy(dg_h, bn_gpu.gamma.grad_dev.value())
    ctx.enqueue_copy(db_h, bn_gpu.beta.grad_dev.value())
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N)
    var max_dg: Scalar[DT] = 0.0
    var max_db: Scalar[DT] = 0.0
    for f in range(DIM):
        var ed = _abs(bn_cpu.gamma.grad[f] - dg_h.unsafe_ptr()[f])
        var eb = _abs(bn_cpu.beta.grad[f]  - db_h.unsafe_ptr()[f])
        if ed > max_dg: max_dg = ed
        if eb > max_db: max_db = eb
    print(
        "  max|y| =", dy, " max|gi| =", dgi,
        " max|dγ| =", max_dg, " max|dβ| =", max_db,
    )
    assert_true(dy < PARITY_ATOL, "BN1D fwd parity failed")
    assert_true(dgi < PARITY_ATOL, "BN1D dx parity failed")
    assert_true(max_dg < PARITY_ATOL, "BN1D dγ parity failed")
    assert_true(max_db < PARITY_ATOL, "BN1D dβ parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_batch_norm_2d_gpu_parity() raises:
    print("test_batch_norm_2d_gpu_parity ...")
    comptime BATCH = 4
    comptime C = 3
    comptime HH = 2
    comptime WW = 2
    comptime FLAT = C * HH * WW
    comptime N = BATCH * FLAT
    var ctx = DeviceContext()

    var bn_cpu = BatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    var bn_gpu = BatchNorm2D[C, HH, WW].make[target="gpu", INIT=Zero](ctx)
    bn_cpu.training = True
    bn_gpu.training = True

    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var go_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        x_h.unsafe_ptr()[i] = Scalar[DT](-0.5 + 0.09 * Float64(i))
        go_h.unsafe_ptr()[i] = Scalar[DT](0.4 + 0.06 * Float64(i))

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
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.enqueue_copy(dg_h, bn_gpu.gamma.grad_dev.value())
    ctx.enqueue_copy(db_h, bn_gpu.beta.grad_dev.value())
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N)
    var max_dg: Scalar[DT] = 0.0
    var max_db: Scalar[DT] = 0.0
    for f in range(C):
        var ed = _abs(bn_cpu.gamma.grad[f] - dg_h.unsafe_ptr()[f])
        var eb = _abs(bn_cpu.beta.grad[f]  - db_h.unsafe_ptr()[f])
        if ed > max_dg: max_dg = ed
        if eb > max_db: max_db = eb
    print(
        "  max|y| =", dy, " max|gi| =", dgi,
        " max|dγ| =", max_dg, " max|dβ| =", max_db,
    )
    assert_true(dy < PARITY_ATOL, "BN2D fwd parity failed")
    assert_true(dgi < PARITY_ATOL, "BN2D dx parity failed")
    assert_true(max_dg < PARITY_ATOL, "BN2D dγ parity failed")
    assert_true(max_db < PARITY_ATOL, "BN2D dβ parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_max_pool_2d_gpu_parity() raises:
    print("test_max_pool_2d_gpu_parity ...")
    comptime BATCH = 2
    comptime C = 2
    comptime HH = 4
    comptime WW = 4
    comptime K_ = 2
    comptime S_ = 2
    comptime P_ = 0
    comptime OH = (HH + 2 * P_ - K_) // S_ + 1
    comptime OW = (WW + 2 * P_ - K_) // S_ + 1
    comptime IN_FLAT = C * HH * WW
    comptime OUT_FLAT = C * OH * OW
    comptime N_IN = BATCH * IN_FLAT
    comptime N_OUT = BATCH * OUT_FLAT
    var ctx = DeviceContext()

    var mp_cpu = MaxPool2D[C, K_, S_, P_, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var mp_gpu = MaxPool2D[C, K_, S_, P_, HH, WW].make[
        target="gpu", INIT=Zero,
    ](ctx)

    var x_h = ctx.enqueue_create_host_buffer[DT](N_IN)
    var go_h = ctx.enqueue_create_host_buffer[DT](N_OUT)
    ctx.synchronize()
    for i in range(N_IN):
        x_h.unsafe_ptr()[i] = Scalar[DT](Float64((i * 31) % 47) * 0.1)
    for i in range(N_OUT):
        go_h.unsafe_ptr()[i] = Scalar[DT](0.5 + 0.13 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    for i in range(N_IN):
        x_cpu[i] = x_h.unsafe_ptr()[i]
    for i in range(N_OUT):
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, IN_FLAT]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, OUT_FLAT]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, OUT_FLAT]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, IN_FLAT]())
    mp_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    mp_cpu.vjp["cpu", BATCH](goc_t, gic_t)

    var x_d = ctx.enqueue_create_buffer[DT](N_IN)
    var y_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var go_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var gi_d = ctx.enqueue_create_buffer[DT](N_IN)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(go_d, go_h)
    var xg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_d.unsafe_ptr())
    var yg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_d.unsafe_ptr())
    var gog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_d.unsafe_ptr())
    var gig_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_d.unsafe_ptr())
    var xg_t = TileTensor(xg_p, row_major[BATCH, IN_FLAT]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, OUT_FLAT]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, OUT_FLAT]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, IN_FLAT]())
    mp_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    mp_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N_OUT)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N_IN)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N_OUT)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N_IN)
    print("  max|y| =", dy, " max|gi| =", dgi)
    assert_true(dy < PARITY_ATOL, "MaxPool fwd parity failed")
    assert_true(dgi < PARITY_ATOL, "MaxPool bwd parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_avg_pool_2d_gpu_parity() raises:
    print("test_avg_pool_2d_gpu_parity ...")
    comptime BATCH = 2
    comptime C = 2
    comptime HH = 4
    comptime WW = 4
    comptime K_ = 2
    comptime S_ = 2
    comptime P_ = 0
    comptime OH = (HH + 2 * P_ - K_) // S_ + 1
    comptime OW = (WW + 2 * P_ - K_) // S_ + 1
    comptime IN_FLAT = C * HH * WW
    comptime OUT_FLAT = C * OH * OW
    comptime N_IN = BATCH * IN_FLAT
    comptime N_OUT = BATCH * OUT_FLAT
    var ctx = DeviceContext()

    var ap_cpu = AvgPool2D[C, K_, S_, P_, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var ap_gpu = AvgPool2D[C, K_, S_, P_, HH, WW].make[
        target="gpu", INIT=Zero,
    ](ctx)

    var x_h = ctx.enqueue_create_host_buffer[DT](N_IN)
    var go_h = ctx.enqueue_create_host_buffer[DT](N_OUT)
    ctx.synchronize()
    for i in range(N_IN):
        x_h.unsafe_ptr()[i] = Scalar[DT](-0.5 + 0.05 * Float64(i))
    for i in range(N_OUT):
        go_h.unsafe_ptr()[i] = Scalar[DT](0.3 + 0.11 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    for i in range(N_IN):
        x_cpu[i] = x_h.unsafe_ptr()[i]
    for i in range(N_OUT):
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, IN_FLAT]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, OUT_FLAT]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, OUT_FLAT]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, IN_FLAT]())
    ap_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    ap_cpu.vjp["cpu", BATCH](goc_t, gic_t)

    var x_d = ctx.enqueue_create_buffer[DT](N_IN)
    var y_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var go_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var gi_d = ctx.enqueue_create_buffer[DT](N_IN)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(go_d, go_h)
    var xg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_d.unsafe_ptr())
    var yg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_d.unsafe_ptr())
    var gog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_d.unsafe_ptr())
    var gig_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_d.unsafe_ptr())
    var xg_t = TileTensor(xg_p, row_major[BATCH, IN_FLAT]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, OUT_FLAT]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, OUT_FLAT]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, IN_FLAT]())
    ap_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    ap_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_h = ctx.enqueue_create_host_buffer[DT](N_OUT)
    var gi_h = ctx.enqueue_create_host_buffer[DT](N_IN)
    ctx.enqueue_copy(y_h, y_d)
    ctx.enqueue_copy(gi_h, gi_d)
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_h.unsafe_ptr(), N_OUT)
    var dgi = _max_diff(gi_cpu, gi_h.unsafe_ptr(), N_IN)
    print("  max|y| =", dy, " max|gi| =", dgi)
    assert_true(dy < PARITY_ATOL, "AvgPool fwd parity failed")
    assert_true(dgi < PARITY_ATOL, "AvgPool bwd parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def test_conv2d_gpu_parity() raises:
    print("test_conv2d_gpu_parity ...")
    comptime BATCH = 2
    comptime IC = 3
    comptime OC = 4
    comptime K_ = 3
    comptime S_ = 1
    comptime P_ = 1
    comptime HH = 5
    comptime WW = 5
    comptime OH = (HH + 2 * P_ - K_) // S_ + 1
    comptime OW = (WW + 2 * P_ - K_) // S_ + 1
    comptime IN_FLAT = IC * HH * WW
    comptime OUT_FLAT = OC * OH * OW
    comptime W_SIZE = OC * IC * K_ * K_
    comptime B_SIZE = OC
    comptime N_IN = BATCH * IN_FLAT
    comptime N_OUT = BATCH * OUT_FLAT
    var ctx = DeviceContext()

    var cv_cpu = Conv2D[IC, OC, K_, S_, P_, HH, WW].make[
        target="cpu", INIT=Kaiming,
    ]()
    var cv_gpu = Conv2D[IC, OC, K_, S_, P_, HH, WW].make[
        target="gpu", INIT=Kaiming,
    ](ctx)

    # Align weights & bias: copy CPU init → GPU buffer.
    var w_h = ctx.enqueue_create_host_buffer[DT](W_SIZE)
    var b_h = ctx.enqueue_create_host_buffer[DT](B_SIZE)
    ctx.synchronize()
    for k in range(W_SIZE):
        w_h.unsafe_ptr()[k] = cv_cpu.weight.value[k]
    for k in range(B_SIZE):
        b_h.unsafe_ptr()[k] = cv_cpu.bias.value[k]
    ctx.enqueue_copy(cv_gpu.weight.value_dev.value(), w_h)
    ctx.enqueue_copy(cv_gpu.bias.value_dev.value(),   b_h)
    ctx.synchronize()

    var x_h = ctx.enqueue_create_host_buffer[DT](N_IN)
    var go_h = ctx.enqueue_create_host_buffer[DT](N_OUT)
    ctx.synchronize()
    for i in range(N_IN):
        x_h.unsafe_ptr()[i] = Scalar[DT](-0.3 + 0.07 * Float64(i))
    for i in range(N_OUT):
        go_h.unsafe_ptr()[i] = Scalar[DT](0.4 + 0.03 * Float64(i))

    var x_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    var y_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_OUT)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_IN)
    for i in range(N_IN):
        x_cpu[i] = x_h.unsafe_ptr()[i]
    for i in range(N_OUT):
        go_cpu[i] = go_h.unsafe_ptr()[i]
    var xc_t = TileTensor(x_cpu, row_major[BATCH, IN_FLAT]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, OUT_FLAT]())
    var goc_t = TileTensor(go_cpu, row_major[BATCH, OUT_FLAT]())
    var gic_t = TileTensor(gi_cpu, row_major[BATCH, IN_FLAT]())
    cv_cpu.forward["cpu", BATCH](xc_t, output=yc_t)
    cv_cpu.zero_grad["cpu"]()
    cv_cpu.vjp["cpu", BATCH](goc_t, gic_t)

    var x_d = ctx.enqueue_create_buffer[DT](N_IN)
    var y_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var go_d = ctx.enqueue_create_buffer[DT](N_OUT)
    var gi_d = ctx.enqueue_create_buffer[DT](N_IN)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(go_d, go_h)
    var xg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_d.unsafe_ptr())
    var yg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_d.unsafe_ptr())
    var gog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_d.unsafe_ptr())
    var gig_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi_d.unsafe_ptr())
    var xg_t = TileTensor(xg_p, row_major[BATCH, IN_FLAT]())
    var yg_t = TileTensor(yg_p, row_major[BATCH, OUT_FLAT]())
    var gog_t = TileTensor(gog_p, row_major[BATCH, OUT_FLAT]())
    var gig_t = TileTensor(gig_p, row_major[BATCH, IN_FLAT]())
    cv_gpu.forward["gpu", BATCH](xg_t, output=yg_t)
    cv_gpu.zero_grad["gpu"]()
    cv_gpu.vjp["gpu", BATCH](gog_t, gig_t)
    var y_host = ctx.enqueue_create_host_buffer[DT](N_OUT)
    var gi_host = ctx.enqueue_create_host_buffer[DT](N_IN)
    var dw_host = ctx.enqueue_create_host_buffer[DT](W_SIZE)
    var db_host = ctx.enqueue_create_host_buffer[DT](B_SIZE)
    ctx.enqueue_copy(y_host,  y_d)
    ctx.enqueue_copy(gi_host, gi_d)
    ctx.enqueue_copy(dw_host, cv_gpu.weight.grad_dev.value())
    ctx.enqueue_copy(db_host, cv_gpu.bias.grad_dev.value())
    ctx.synchronize()

    var dy = _max_diff(y_cpu, y_host.unsafe_ptr(), N_OUT)
    var dgi = _max_diff(gi_cpu, gi_host.unsafe_ptr(), N_IN)
    var max_dw: Scalar[DT] = 0.0
    var max_db: Scalar[DT] = 0.0
    for k in range(W_SIZE):
        var e = _abs(cv_cpu.weight.grad[k] - dw_host.unsafe_ptr()[k])
        if e > max_dw: max_dw = e
    for k in range(B_SIZE):
        var e = _abs(cv_cpu.bias.grad[k] - db_host.unsafe_ptr()[k])
        if e > max_db: max_db = e
    print(
        "  max|y| =", dy, " max|gi| =", dgi,
        " max|dW| =", max_dw, " max|db| =", max_db,
    )
    # Convolution accumulates O(IC·K·K) terms per output, so use a
    # slightly looser tolerance than the elementwise layers.
    var conv_tol = Scalar[DT](2e-4)
    assert_true(dy < conv_tol, "Conv2D fwd parity failed")
    assert_true(dgi < conv_tol, "Conv2D dx parity failed")
    assert_true(max_dw < conv_tol, "Conv2D dW parity failed")
    assert_true(max_db < conv_tol, "Conv2D db parity failed")
    x_cpu.free()
    y_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("nn2 GPU parity smoke (PORTING_PLAN.md consumer-gated follow-ups)")
    print("=" * 70)
    test_sim_norm_gpu_parity()
    test_min_max_norm_gpu_parity()
    test_dropout_gpu_eval_identity()
    test_dropout_gpu_train_mask_stats()
    test_batch_norm_1d_gpu_parity()
    test_batch_norm_2d_gpu_parity()
    test_max_pool_2d_gpu_parity()
    test_avg_pool_2d_gpu_parity()
    test_conv2d_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
