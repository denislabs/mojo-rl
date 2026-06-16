"""NoisyLinear[IN, OUT] CPU + GPU smoke test.

Validates:
  - forward produces finite output of shape [B, OUT]
  - backward produces finite grads on all 4 param groups + grad_x
  - algebraic relationship: grad_σ_W[i, j] = grad_μ_W[i, j] · f(ε_in[i]) · f(ε_out[j])
    (and similarly for the bias pair) — exercises the noise-cache path
  - on GPU, same invariant holds (noise sampled via Philox/Box-Muller)
"""

from std.math import isnan, isinf
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn.initializer import Zero


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_forward_backward_cpu() raises:
    print("test_forward_backward_cpu ...")
    comptime BATCH = 4
    comptime IN = 3
    comptime OUT = 5

    seed(42)
    var nl = NoisyLinear[IN, OUT].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * IN):
        x[k] = Scalar[DT](0.1 * Float64(k) - 0.5)
    for k in range(BATCH * OUT):
        go[k] = Scalar[DT](0.05 * Float64(k) + 0.2)

    var x_t = TileTensor(rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x), row_major[BATCH, IN]())
    var y_t = TileTensor(rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y), row_major[BATCH, OUT]())
    var go_t = TileTensor(rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go), row_major[BATCH, OUT]())
    var gi_t = TileTensor(rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi), row_major[BATCH, IN]())
    nl.forward["cpu", BATCH](x_t, output=y_t)
    nl.vjp["cpu", BATCH](go_t, gi_t)

    for k in range(BATCH * OUT):
        assert_true(not isnan(y[k]), "y NaN")
        assert_true(not isinf(y[k]), "y Inf")
    for k in range(BATCH * IN):
        assert_true(not isnan(gi[k]), "gi NaN")
        assert_true(not isinf(gi[k]), "gi Inf")

    # Algebraic invariant: grad_σ_W[i, j] = grad_μ_W[i, j] · noise_in[i] · noise_out[j]
    # (and grad_σ_b[j] = grad_μ_b[j] · noise_out[j]) — direct from the noisy
    # forward derivative. Holds to FP32 ULP since both grads accumulated from
    # the same Σ_b x[b,i]·grad_out[b,j].
    var ni_p = nl._noise_in.cpu_ptr()
    var no_p = nl._noise_out.cpu_ptr()
    var g_mu_w = nl.mu_w.grad_unsafe_ptr_cpu()
    var g_sg_w = nl.sigma_w.grad_unsafe_ptr_cpu()
    var g_mu_b = nl.mu_b.grad_unsafe_ptr_cpu()
    var g_sg_b = nl.sigma_b.grad_unsafe_ptr_cpu()

    var max_w_diff: Scalar[DT] = 0.0
    for i in range(IN):
        for j in range(OUT):
            var idx = i * OUT + j
            var expected = g_mu_w[idx] * ni_p[i] * no_p[j]
            var d = _abs(g_sg_w[idx] - expected)
            if d > max_w_diff:
                max_w_diff = d
    var max_b_diff: Scalar[DT] = 0.0
    for j in range(OUT):
        var expected = g_mu_b[j] * no_p[j]
        var d = _abs(g_sg_b[j] - expected)
        if d > max_b_diff:
            max_b_diff = d

    print("  max |g_σ_w - g_μ_w·n_w| =", max_w_diff)
    print("  max |g_σ_b - g_μ_b·n_b| =", max_b_diff)
    assert_true(
        max_w_diff < Scalar[DT](1e-5), "σ_w grad invariant broken"
    )
    assert_true(
        max_b_diff < Scalar[DT](1e-6), "σ_b grad invariant broken"
    )
    print("  ok")


def test_forward_backward_gpu() raises:
    print("test_forward_backward_gpu ...")
    comptime BATCH = 4
    comptime IN = 3
    comptime OUT = 5

    var ctx = DeviceContext()
    seed(42)
    var nl = NoisyLinear[IN, OUT].make[target="gpu", INIT=Zero](ctx=ctx)
    nl.set_noise_seed(UInt64(123))

    var x_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var y_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var x_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.synchronize()
    for k in range(BATCH * IN):
        x_host.unsafe_ptr()[k] = Scalar[DT](0.1 * Float64(k) - 0.5)
    for k in range(BATCH * OUT):
        go_host.unsafe_ptr()[k] = Scalar[DT](0.05 * Float64(k) + 0.2)
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(go_dev, go_host)
    ctx.synchronize()

    var x_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        x_dev.unsafe_ptr()
    )
    var y_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        y_dev.unsafe_ptr()
    )
    var go_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        go_dev.unsafe_ptr()
    )
    var gi_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        gi_dev.unsafe_ptr()
    )
    var x_t = TileTensor(x_dev_p, row_major[BATCH, IN]())
    var y_t = TileTensor(y_dev_p, row_major[BATCH, OUT]())
    var go_t = TileTensor(go_dev_p, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi_dev_p, row_major[BATCH, IN]())
    nl.forward["gpu", BATCH](x_t, output=y_t)
    nl.vjp["gpu", BATCH](go_t, gi_t)

    # Download y, gi, noise, grads for inspection.
    var y_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    var ni_host = ctx.enqueue_create_host_buffer[DT](IN)
    var no_host = ctx.enqueue_create_host_buffer[DT](OUT)
    var g_mu_w_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var g_sg_w_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var g_mu_b_host = ctx.enqueue_create_host_buffer[DT](OUT)
    var g_sg_b_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.enqueue_copy(y_host, y_dev)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.enqueue_copy(ni_host, nl._noise_in.dev.value())
    ctx.enqueue_copy(no_host, nl._noise_out.dev.value())
    ctx.enqueue_copy(g_mu_w_host, nl.mu_w.grd.dev.value())
    ctx.enqueue_copy(g_sg_w_host, nl.sigma_w.grd.dev.value())
    ctx.enqueue_copy(g_mu_b_host, nl.mu_b.grd.dev.value())
    ctx.enqueue_copy(g_sg_b_host, nl.sigma_b.grd.dev.value())
    ctx.synchronize()

    for k in range(BATCH * OUT):
        assert_true(not isnan(y_host.unsafe_ptr()[k]), "y NaN")
        assert_true(not isinf(y_host.unsafe_ptr()[k]), "y Inf")
    for k in range(BATCH * IN):
        assert_true(not isnan(gi_host.unsafe_ptr()[k]), "gi NaN")
        assert_true(not isinf(gi_host.unsafe_ptr()[k]), "gi Inf")

    var max_w_diff: Scalar[DT] = 0.0
    for i in range(IN):
        for j in range(OUT):
            var idx = i * OUT + j
            var expected = (
                g_mu_w_host.unsafe_ptr()[idx]
                * ni_host.unsafe_ptr()[i]
                * no_host.unsafe_ptr()[j]
            )
            var d = _abs(g_sg_w_host.unsafe_ptr()[idx] - expected)
            if d > max_w_diff:
                max_w_diff = d
    var max_b_diff: Scalar[DT] = 0.0
    for j in range(OUT):
        var expected = (
            g_mu_b_host.unsafe_ptr()[j] * no_host.unsafe_ptr()[j]
        )
        var d = _abs(g_sg_b_host.unsafe_ptr()[j] - expected)
        if d > max_b_diff:
            max_b_diff = d
    print("  max |g_σ_w - g_μ_w·n_w| =", max_w_diff)
    print("  max |g_σ_b - g_μ_b·n_b| =", max_b_diff)
    assert_true(
        max_w_diff < Scalar[DT](1e-5), "σ_w grad invariant broken (GPU)"
    )
    assert_true(
        max_b_diff < Scalar[DT](1e-6), "σ_b grad invariant broken (GPU)"
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("NoisyLinear[IN, OUT] CPU + GPU smoke")
    print("=" * 70)
    test_forward_backward_cpu()
    test_forward_backward_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
