"""LayerNorm[DIM] CPU + GPU tests — Phase 5.4.

Covers:
  - forward: hand-computed normalize on a known [1,2,3,4,5] input
  - backward: finite-difference gradcheck on grad_input
  - for_each_param: γ + β visits, both apply_decay=False
  - GPU parity vs CPU on forward, dx, dgamma, dbeta
"""

from std.math import abs as fabs, sqrt
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.layer_norm import LayerNorm


struct ParamRecord(Movable & ImplicitlyDestructible):
    var name: String
    var n_elems: Int
    var apply_decay: Bool

    def __init__(out self, name: String, n_elems: Int, apply_decay: Bool):
        self.name = name
        self.n_elems = n_elems
        self.apply_decay = apply_decay


struct RecordVisitor(ParamVisitor):
    var records: List[ParamRecord]

    def __init__(out self):
        self.records = List[ParamRecord]()

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
        ) raises:
        self.records.append(ParamRecord(name, n_elems, apply_decay))


def test_forward_cpu() raises:
    """Input [1,2,3,4,5] → x_hat ≈ [-√2, -√2/2, 0, √2/2, √2]
    (γ=1, β=0, eps=1e-5 — negligible)."""
    comptime DIM = 5
    comptime BATCH = 1
    comptime TOL: Scalar[DT] = 1e-3   # eps=1e-5 perturbs ~3e-6

    var ln = LayerNorm[DIM].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] = 1.0
    in_buf[1] = 2.0
    in_buf[2] = 3.0
    in_buf[3] = 4.0
    in_buf[4] = 5.0
    for k in range(BATCH * DIM):
        out_buf[k] = -999.0

    var input  = TileTensor(in_buf,  row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())
    ln.forward["cpu", BATCH](input, output)

    # Expected: (x - mean=3) * inv_std where inv_std = 1/sqrt(2+1e-5).
    var inv_std = Scalar[DT](1.0) / sqrt(Scalar[DT](2.0 + 1e-5))
    for d in range(DIM):
        var x = Scalar[DT](Float32(d + 1))    # 1, 2, 3, 4, 5
        var expected = (x - Scalar[DT](3.0)) * inv_std
        var diff = fabs(output[0, d] - expected)
        assert_true(diff < TOL, "d=" + String(d) + " expected " + String(expected)
            + " got " + String(output[0, d]) + " diff " + String(diff))

    # Mean of output must be ~0, var ~1 (sanity check of LN invariant).
    var s: Scalar[DT] = 0.0
    for d in range(DIM):
        s += output[0, d]
    assert_true(fabs(s) < TOL, "output mean not ~0: " + String(s))
    var s2v: Scalar[DT] = 0.0
    for d in range(DIM):
        s2v += output[0, d] * output[0, d]
    var var_actual = s2v / Scalar[DT](DIM)
    assert_true(fabs(var_actual - 1.0) < TOL, "output var not ~1: " + String(var_actual))

    in_buf.free()
    out_buf.free()
    print("  test_forward_cpu PASSED (output zero-mean unit-var)")


def test_backward_cpu_gradcheck() raises:
    """Finite-difference gradcheck on grad_input. Uses a fixed
    grad_output and a non-trivial input."""
    comptime DIM = 4
    comptime BATCH = 2
    comptime EPS_FD: Scalar[DT] = 1e-2
    comptime TOL_REL: Scalar[DT] = 1e-2     # FD vs analytical rel-err

    var ln = LayerNorm[DIM].make["cpu", INIT=Zero]()

    # Non-trivial input + grad_output.
    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] =  1.5; in_buf[1] = -0.5; in_buf[2] = 2.0; in_buf[3] = -1.0
    in_buf[4] = -2.0; in_buf[5] =  3.0; in_buf[6] = 0.5; in_buf[7] =  1.0
    for k in range(BATCH * DIM):
        go_buf[k] = Scalar[DT](0.1 + Float32(k) * 0.05)
        gi_buf[k] = 0.0

    var input  = TileTensor(in_buf,  row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())
    var grad_out = TileTensor(go_buf, row_major[BATCH, DIM]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, DIM]())

    # Forward, then analytical backward.
    ln.forward["cpu", BATCH](input, output)
    ln.backward["cpu", BATCH](grad_out, grad_in)

    # FD gradcheck: ∂L/∂x[b,d] = lim (L(x+ε) - L(x-ε)) / 2ε
    # where L = Σ_{b',d'} grad_output[b',d'] * y[b',d'] (linearizes y).
    var max_rel: Scalar[DT] = 0.0
    var max_abs_analytical: Scalar[DT] = 0.0
    for bi in range(BATCH):
        for di in range(DIM):
            # Loss at x + eps
            in_buf[bi * DIM + di] += EPS_FD
            ln.forward["cpu", BATCH](input, output)
            var L_plus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for d2 in range(DIM):
                    L_plus += go_buf[b2 * DIM + d2] * output[b2, d2]
            # Loss at x - eps
            in_buf[bi * DIM + di] -= Scalar[DT](2.0) * EPS_FD
            ln.forward["cpu", BATCH](input, output)
            var L_minus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for d2 in range(DIM):
                    L_minus += go_buf[b2 * DIM + d2] * output[b2, d2]
            # Restore
            in_buf[bi * DIM + di] += EPS_FD
            var fd = (L_plus - L_minus) / (Scalar[DT](2.0) * EPS_FD)
            var an = grad_in[bi, di]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel: max_rel = rel
            if fabs(an) > max_abs_analytical: max_abs_analytical = fabs(an)

    print("LayerNorm backward gradcheck: max-rel-err = " + String(max_rel)
          + ", max|analytical| = " + String(max_abs_analytical))
    assert_true(max_rel < TOL_REL,
        "gradcheck failed: max-rel-err " + String(max_rel) + " > " + String(TOL_REL))

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_backward_cpu_gradcheck PASSED")


def test_for_each_param() raises:
    """Two visits: 'gamma' and 'beta', both with apply_decay=False."""
    var ln = LayerNorm[8].make["cpu", INIT=Zero]()
    var v = RecordVisitor()
    ln.for_each_param["cpu"](String("ln0"), v)
    assert_equal(len(v.records), 2)
    assert_equal(v.records[0].name, String("ln0.gamma"))
    assert_equal(v.records[0].n_elems, 8)
    assert_true(not v.records[0].apply_decay, "gamma must have apply_decay=False")
    assert_equal(v.records[1].name, String("ln0.beta"))
    assert_equal(v.records[1].n_elems, 8)
    assert_true(not v.records[1].apply_decay, "beta must have apply_decay=False")
    print("  test_for_each_param PASSED (gamma+beta, both apply_decay=False)")


def test_gpu_parity() raises:
    """GPU forward + backward parity vs CPU. Includes dgamma/dbeta
    accumulation."""
    comptime DIM = 6
    comptime BATCH = 4
    comptime TOL: Scalar[DT] = 1e-4

    var ctx = DeviceContext()
    var ln_cpu = LayerNorm[DIM].make["cpu", INIT=Zero]()
    var ln_gpu = LayerNorm[DIM].make["gpu", INIT=Zero](ctx)

    # Non-default γ + β to make grad_input depend on them.
    for d in range(DIM):
        ln_cpu.gamma[d] = Scalar[DT](0.5 + Float32(d) * 0.1)
        ln_cpu.beta[d]  = Scalar[DT](-0.2 + Float32(d) * 0.05)

    var gamma_host = ctx.enqueue_create_host_buffer[DT](DIM)
    var beta_host  = ctx.enqueue_create_host_buffer[DT](DIM)
    ctx.synchronize()
    for d in range(DIM):
        gamma_host.unsafe_ptr()[d] = ln_cpu.gamma[d]
        beta_host.unsafe_ptr()[d]  = ln_cpu.beta[d]
    ctx.enqueue_copy(ln_gpu.gamma_dev.value(), gamma_host)
    ctx.enqueue_copy(ln_gpu.beta_dev.value(),  beta_host)

    # Input.
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    var ip = in_host.unsafe_ptr()
    for k in range(BATCH * DIM):
        ip[k] = Scalar[DT](Float32(k) * 0.3 - 2.0)

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_buf_cpu[k] = ip[k]
    var input_cpu  = TileTensor(in_buf_cpu, row_major[BATCH, DIM]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, DIM]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, DIM]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, DIM]())

    ln_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    ln_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(output_cpu[b, d] - out_host.unsafe_ptr()[b * DIM + d])
            if diff > max_fwd: max_fwd = diff
    print("forward max-diff = " + String(max_fwd))
    assert_true(max_fwd < TOL, "forward parity: " + String(max_fwd))

    # ── Backward ────────────────────────────────────────────────────────
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        go_host.unsafe_ptr()[k] = Scalar[DT](0.1 + Float32(k) * 0.05)

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
        gi_buf_cpu[k] = 0.0
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, DIM]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, DIM]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, DIM]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, DIM]())

    ln_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    ln_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    var gg_host = ctx.enqueue_create_host_buffer[DT](DIM)
    var gb_host = ctx.enqueue_create_host_buffer[DT](DIM)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.enqueue_copy(gg_host, ln_gpu.grad_gamma_dev.value())
    ctx.enqueue_copy(gb_host, ln_gpu.grad_beta_dev.value())
    ctx.synchronize()

    var max_gi: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(grad_in_cpu[b, d] - gi_host.unsafe_ptr()[b * DIM + d])
            if diff > max_gi: max_gi = diff
    var max_gg: Scalar[DT] = 0.0
    var max_gb: Scalar[DT] = 0.0
    for d in range(DIM):
        var dg = fabs(ln_cpu.grad_gamma[d] - gg_host.unsafe_ptr()[d])
        var db = fabs(ln_cpu.grad_beta[d]  - gb_host.unsafe_ptr()[d])
        if dg > max_gg: max_gg = dg
        if db > max_gb: max_gb = db
    print("grad_input max-diff = " + String(max_gi))
    print("grad_gamma max-diff = " + String(max_gg))
    print("grad_beta  max-diff = " + String(max_gb))
    assert_true(max_gi < TOL, "grad_input parity: " + String(max_gi))
    assert_true(max_gg < TOL, "grad_gamma parity: " + String(max_gg))
    assert_true(max_gb < TOL, "grad_beta parity: "  + String(max_gb))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 LayerNorm tests (CPU + GPU, Phase 5.4)")
    print("=" * 60)
    test_forward_cpu()
    test_backward_cpu_gradcheck()
    test_for_each_param()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
