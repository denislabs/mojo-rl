"""Parity test: LinearAct[IN, OUT, ReLUOp] vs Linear[IN, OUT] + ReLU[OUT].

Validates that the fused LinearAct primitive is numerically equivalent to
the unfused Sequential[Linear, ReLU] chain. With weights matched via
`seed(42)` before each `.make[..., Kaiming]()` call, the two paths execute
the same arithmetic and should produce bit-identical results on CPU.

GPU path is checked with a small absolute-diff tolerance (1e-5) because
the fused epilogue executes the bias-add + activation in a different lane
order than the unfused separate kernels — same fp32 operations, but
potential FMA-flush differences across the parallel reductions in
`grad_w`. CPU path is fully deterministic.

Coverage:
  1. CPU forward: y_linact == y_unfused
  2. CPU backward: grad_in / grad_w / grad_b all match
  3. GPU forward: y_linact ≈ y_unfused (< 1e-5)
  4. GPU backward: grad_in / grad_w / grad_b all ≈ within 1e-5
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.linear_act import LinearAct
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.initializer import Kaiming


comptime BATCH = 4
comptime IN = 6
comptime OUT = 5
comptime N_X = BATCH * IN
comptime N_Y = BATCH * OUT


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _maxdiff(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
             b: UnsafePointer[Scalar[DT], MutAnyOrigin],
             n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = _abs(a[i] - b[i])
        if d > m:
            m = d
    return m


def test_cpu_parity() raises:
    print("test_cpu_parity ...")

    seed(42)
    var linact = LinearAct[IN, OUT, ReLUOp].make[target="cpu", INIT=Kaiming]()
    seed(42)
    var lin = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    var relu = ReLU[OUT].make[target="cpu", INIT=Kaiming]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var y_act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var y_lin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var y_relu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi_act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var gi_lin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var mid_grad: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    # Mixed-sign inputs so ReLU's branch fires for both x>0 and x<0 lanes.
    for i in range(N_X):
        x[i] = Scalar[DT](-1.0 + 0.17 * Float64(i))
    for i in range(N_Y):
        go[i] = Scalar[DT](0.3 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var y_act_t = TileTensor(y_act, row_major[BATCH, OUT]())
    var y_lin_t = TileTensor(y_lin, row_major[BATCH, OUT]())
    var y_relu_t = TileTensor(y_relu, row_major[BATCH, OUT]())
    var gi_act_t = TileTensor(gi_act, row_major[BATCH, IN]())
    var gi_lin_t = TileTensor(gi_lin, row_major[BATCH, IN]())
    var mid_grad_t = TileTensor(mid_grad, row_major[BATCH, OUT]())

    # ── Forward ────────────────────────────────────────────────────────
    linact.forward["cpu", BATCH](x_t, output=y_act_t)
    lin.forward["cpu", BATCH](x_t, output=y_lin_t)
    relu.forward["cpu", BATCH](y_lin_t, output=y_relu_t)

    var fwd_diff = _maxdiff(y_act, y_relu, N_Y)
    print("  forward max-diff:", fwd_diff)
    assert_true(fwd_diff == Scalar[DT](0), "CPU forward mismatch")

    # ── Backward ───────────────────────────────────────────────────────
    linact.zero_grad["cpu"]()
    lin.zero_grad["cpu"]()
    linact.vjp["cpu", BATCH](go_t, gi_act_t)
    relu.vjp["cpu", BATCH](go_t, mid_grad_t)
    lin.vjp["cpu", BATCH](mid_grad_t, gi_lin_t)

    var gi_diff = _maxdiff(gi_act, gi_lin, N_X)
    print("  grad_in max-diff:", gi_diff)
    assert_true(gi_diff == Scalar[DT](0), "CPU grad_in mismatch")

    var gw_act = linact.weight.grad_unsafe_ptr_cpu()
    var gw_lin = lin.weight.grad_unsafe_ptr_cpu()
    var gw_diff = _maxdiff(gw_act, gw_lin, IN * OUT)
    print("  grad_w  max-diff:", gw_diff)
    assert_true(gw_diff == Scalar[DT](0), "CPU grad_w mismatch")

    var gb_act = linact.bias.grad_unsafe_ptr_cpu()
    var gb_lin = lin.bias.grad_unsafe_ptr_cpu()
    var gb_diff = _maxdiff(gb_act, gb_lin, OUT)
    print("  grad_b  max-diff:", gb_diff)
    assert_true(gb_diff == Scalar[DT](0), "CPU grad_b mismatch")

    x.free(); go.free()
    y_act.free(); y_lin.free(); y_relu.free()
    gi_act.free(); gi_lin.free(); mid_grad.free()
    print("  ok")


def test_gpu_parity() raises:
    print("test_gpu_parity ...")
    var ctx = DeviceContext()

    seed(42)
    var linact = LinearAct[IN, OUT, ReLUOp].make[target="gpu", INIT=Kaiming](ctx)
    seed(42)
    var lin = Linear[IN, OUT].make[target="gpu", INIT=Kaiming](ctx)
    var relu = ReLU[OUT].make[target="gpu", INIT=Kaiming](ctx)

    # Host stage buffers.
    var x_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var go_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var y_act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var y_relu_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi_act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var gi_lin_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var gw_act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN * OUT)
    var gw_lin_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN * OUT)
    var gb_act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT)
    var gb_lin_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT)
    for i in range(N_X):
        x_h[i] = Scalar[DT](-1.0 + 0.17 * Float64(i))
    for i in range(N_Y):
        go_h[i] = Scalar[DT](0.3 + 0.05 * Float64(i))

    # Device buffers.
    var x_dev = ctx.enqueue_create_buffer[DT](N_X)
    var go_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var y_act_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var y_lin_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var y_relu_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var gi_act_dev = ctx.enqueue_create_buffer[DT](N_X)
    var gi_lin_dev = ctx.enqueue_create_buffer[DT](N_X)
    var mid_grad_dev = ctx.enqueue_create_buffer[DT](N_Y)

    var x_host = ctx.enqueue_create_host_buffer[DT](N_X)
    var go_host = ctx.enqueue_create_host_buffer[DT](N_Y)
    ctx.synchronize()
    for i in range(N_X):
        x_host.unsafe_ptr()[i] = x_h[i]
    for i in range(N_Y):
        go_host.unsafe_ptr()[i] = go_h[i]
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(go_dev, go_host)

    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = x_dev.unsafe_ptr()
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = go_dev.unsafe_ptr()
    var y_act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = y_act_dev.unsafe_ptr()
    var y_lin_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = y_lin_dev.unsafe_ptr()
    var y_relu_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = y_relu_dev.unsafe_ptr()
    var gi_act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = gi_act_dev.unsafe_ptr()
    var gi_lin_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = gi_lin_dev.unsafe_ptr()
    var mid_grad_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = mid_grad_dev.unsafe_ptr()

    var x_t = TileTensor(x_p, row_major[BATCH, IN]())
    var go_t = TileTensor(go_p, row_major[BATCH, OUT]())
    var y_act_t = TileTensor(y_act_p, row_major[BATCH, OUT]())
    var y_lin_t = TileTensor(y_lin_p, row_major[BATCH, OUT]())
    var y_relu_t = TileTensor(y_relu_p, row_major[BATCH, OUT]())
    var gi_act_t = TileTensor(gi_act_p, row_major[BATCH, IN]())
    var gi_lin_t = TileTensor(gi_lin_p, row_major[BATCH, IN]())
    var mid_grad_t = TileTensor(mid_grad_p, row_major[BATCH, OUT]())

    # ── Forward ────────────────────────────────────────────────────────
    linact.forward["gpu", BATCH](x_t, output=y_act_t)
    lin.forward["gpu", BATCH](x_t, output=y_lin_t)
    relu.forward["gpu", BATCH](y_lin_t, output=y_relu_t)

    # ── Backward ───────────────────────────────────────────────────────
    linact.zero_grad["gpu"]()
    lin.zero_grad["gpu"]()
    linact.vjp["gpu", BATCH](go_t, gi_act_t)
    relu.vjp["gpu", BATCH](go_t, mid_grad_t)
    lin.vjp["gpu", BATCH](mid_grad_t, gi_lin_t)

    # Download results.
    var y_act_host = ctx.enqueue_create_host_buffer[DT](N_Y)
    var y_relu_host = ctx.enqueue_create_host_buffer[DT](N_Y)
    var gi_act_host = ctx.enqueue_create_host_buffer[DT](N_X)
    var gi_lin_host = ctx.enqueue_create_host_buffer[DT](N_X)
    var gw_act_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var gw_lin_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var gb_act_host = ctx.enqueue_create_host_buffer[DT](OUT)
    var gb_lin_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.enqueue_copy(y_act_host, y_act_dev)
    ctx.enqueue_copy(y_relu_host, y_relu_dev)
    ctx.enqueue_copy(gi_act_host, gi_act_dev)
    ctx.enqueue_copy(gi_lin_host, gi_lin_dev)
    ctx.enqueue_copy(gw_act_host, linact.weight.grd.dev.value())
    ctx.enqueue_copy(gw_lin_host, lin.weight.grd.dev.value())
    ctx.enqueue_copy(gb_act_host, linact.bias.grd.dev.value())
    ctx.enqueue_copy(gb_lin_host, lin.bias.grd.dev.value())
    ctx.synchronize()
    for i in range(N_Y):
        y_act_h[i] = y_act_host.unsafe_ptr()[i]
        y_relu_h[i] = y_relu_host.unsafe_ptr()[i]
    for i in range(N_X):
        gi_act_h[i] = gi_act_host.unsafe_ptr()[i]
        gi_lin_h[i] = gi_lin_host.unsafe_ptr()[i]
    for i in range(IN * OUT):
        gw_act_h[i] = gw_act_host.unsafe_ptr()[i]
        gw_lin_h[i] = gw_lin_host.unsafe_ptr()[i]
    for i in range(OUT):
        gb_act_h[i] = gb_act_host.unsafe_ptr()[i]
        gb_lin_h[i] = gb_lin_host.unsafe_ptr()[i]

    var fwd_diff = _maxdiff(y_act_h, y_relu_h, N_Y)
    var gi_diff  = _maxdiff(gi_act_h, gi_lin_h, N_X)
    var gw_diff  = _maxdiff(gw_act_h, gw_lin_h, IN * OUT)
    var gb_diff  = _maxdiff(gb_act_h, gb_lin_h, OUT)
    print("  forward  max-diff:", fwd_diff)
    print("  grad_in  max-diff:", gi_diff)
    print("  grad_w   max-diff:", gw_diff)
    print("  grad_b   max-diff:", gb_diff)

    var tol: Scalar[DT] = 1e-5
    assert_true(fwd_diff < tol, "GPU forward outside tolerance")
    assert_true(gi_diff  < tol, "GPU grad_in outside tolerance")
    assert_true(gw_diff  < tol, "GPU grad_w outside tolerance")
    assert_true(gb_diff  < tol, "GPU grad_b outside tolerance")

    x_h.free(); go_h.free()
    y_act_h.free(); y_relu_h.free()
    gi_act_h.free(); gi_lin_h.free()
    gw_act_h.free(); gw_lin_h.free()
    gb_act_h.free(); gb_lin_h.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LinearAct[ReLUOp] vs Linear+ReLU parity")
    print("=" * 70)
    test_cpu_parity()
    test_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
