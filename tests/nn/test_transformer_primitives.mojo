"""Wave A transformer primitives — CPU forward + finite-diff gradcheck.

Covers Transpose2D, TokenMean, BiasAdd, Embedding (docs/NN_TRANSFORMER_PORT.md
Phase 1 Wave A). All four are linear ops, so finite differences match the
analytic vjp to rounding. We check:
  * forward against a hand-computed reference, and
  * grad_input (all) + grad_param (BiasAdd.bias, Embedding.weight) vs FD.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.transpose_2d import Transpose2D
from mojo_rl.nn.primitives.token_mean import TokenMean
from mojo_rl.nn.primitives.bias_add import BiasAdd
from mojo_rl.nn.primitives.embedding import Embedding


comptime EPS: Float64 = 1e-2
comptime TOL: Float64 = 1e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _fill(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, seed: Float64):
    for i in range(n):
        p[i] = Scalar[DT](0.37 * sin_approx(seed + 0.7 * Float64(i)))


def sin_approx(x: Float64) -> Float64:
    # Cheap deterministic spread in [-1,1]; exact value irrelevant for a test.
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return t - (t * t * t) / 6.0 + (t * t * t * t * t) / 120.0


def _loss(
    vals: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(vals[i]) * Float64(go[i])
    return s


def _maxabs_diff_vs_fd_input[
    op_out_dim: Int
](
    name: String,
    # closure-free FD: caller supplies forward via a re-run lambda is awkward
    # in Mojo, so each test inlines its own FD. This helper just reports.
    analytic: UnsafePointer[Scalar[DT], MutAnyOrigin],
    fd: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(analytic[i]) - Float64(fd[i]))
        if d > m:
            m = d
    print("  ", name, " max|analytic - FD| =", m)
    return m


# ──────────────────────────────────────────────────────────────────────
# Transpose2D
# ──────────────────────────────────────────────────────────────────────


def test_transpose2d() raises:
    print("test_transpose2d ...")
    comptime BATCH = 2
    comptime A = 3
    comptime B = 4
    comptime N = BATCH * A * B
    var op = Transpose2D[A, B].make[target="cpu", INIT=Zero]()

    var x = _alloc(N)
    var y = _alloc(N)
    var go = _alloc(N)
    var gi = _alloc(N)
    _fill(x, N, 1.0)
    _fill(go, N, 5.0)
    var x_t = TileTensor(x, row_major[BATCH, A * B]())
    var y_t = TileTensor(y, row_major[BATCH, A * B]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    # Forward reference: y[b, j*A+i] == x[b, i*B+j].
    var fwd_err: Float64 = 0.0
    for b in range(BATCH):
        for i in range(A):
            for j in range(B):
                var d = abs(
                    Float64(y[b * A * B + j * A + i])
                    - Float64(x[b * A * B + i * B + j])
                )
                if d > fwd_err:
                    fwd_err = d
    print("   forward max err =", fwd_err)
    assert_true(fwd_err == 0.0, "Transpose2D forward mismatch")

    # Backward: analytic grad_input vs FD.
    var go_t = TileTensor(go, row_major[BATCH, A * B]())
    var gi_t = TileTensor(gi, row_major[BATCH, A * B]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    var fd = _alloc(N)
    for k in range(N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, N)
        x[k] = orig
        fd[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m = _maxabs_diff_vs_fd_input[A * B]("grad_input", gi, fd, N)
    assert_true(m < TOL, "Transpose2D grad_input vs FD")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# TokenMean
# ──────────────────────────────────────────────────────────────────────


def test_token_mean() raises:
    print("test_token_mean ...")
    comptime BATCH = 2
    comptime SEQ = 5
    comptime DIM = 3
    comptime IN_N = BATCH * SEQ * DIM
    comptime OUT_N = BATCH * DIM
    var op = TokenMean[SEQ, DIM].make[target="cpu", INIT=Zero]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    _fill(x, IN_N, 2.0)
    _fill(go, OUT_N, 9.0)
    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    var fwd_err: Float64 = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var s: Float64 = 0.0
            for t in range(SEQ):
                s += Float64(x[b * SEQ * DIM + t * DIM + d])
            var d2 = abs(Float64(y[b * DIM + d]) - s / Float64(SEQ))
            if d2 > fwd_err:
                fwd_err = d2
    print("   forward max err =", fwd_err)
    assert_true(fwd_err < 1e-5, "TokenMean forward mismatch")

    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * DIM]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    var fd = _alloc(IN_N)
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, OUT_N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, OUT_N)
        x[k] = orig
        fd[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m = _maxabs_diff_vs_fd_input[DIM]("grad_input", gi, fd, IN_N)
    assert_true(m < TOL, "TokenMean grad_input vs FD")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# BiasAdd
# ──────────────────────────────────────────────────────────────────────


def test_bias_add() raises:
    print("test_bias_add ...")
    comptime BATCH = 4
    comptime DIM = 6
    comptime N = BATCH * DIM
    var op = BiasAdd[DIM].make[target="cpu", INIT=Zero]()
    # Set a non-zero bias.
    var b_ptr = op.bias.value_unsafe_ptr_cpu()
    for i in range(DIM):
        b_ptr[i] = Scalar[DT](0.1 * Float64(i) - 0.25)

    var x = _alloc(N)
    var y = _alloc(N)
    var go = _alloc(N)
    var gi = _alloc(N)
    _fill(x, N, 3.0)
    _fill(go, N, 7.0)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    var fwd_err: Float64 = 0.0
    for b in range(BATCH):
        for i in range(DIM):
            var d = abs(
                Float64(y[b * DIM + i])
                - (Float64(x[b * DIM + i]) + Float64(b_ptr[i]))
            )
            if d > fwd_err:
                fwd_err = d
    print("   forward max err =", fwd_err)
    assert_true(fwd_err < 1e-6, "BiasAdd forward mismatch")

    # Backward (analytic). grad accumulates, so zero it first.
    op.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    # grad_input FD.
    var fd_in = _alloc(N)
    for k in range(N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, N)
        x[k] = orig
        fd_in[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m_in = _maxabs_diff_vs_fd_input[DIM]("grad_input", gi, fd_in, N)
    assert_true(m_in < TOL, "BiasAdd grad_input vs FD")

    # grad_bias FD.
    var ga = op.bias.grad_unsafe_ptr_cpu()
    var fd_b = _alloc(DIM)
    for k in range(DIM):
        var orig = b_ptr[k]
        b_ptr[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, N)
        b_ptr[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, N)
        b_ptr[k] = orig
        fd_b[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m_b = _maxabs_diff_vs_fd_input[DIM]("grad_bias", ga, fd_b, DIM)
    assert_true(m_b < TOL, "BiasAdd grad_bias vs FD")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────


def test_embedding() raises:
    print("test_embedding ...")
    comptime BATCH = 3
    comptime VOCAB = 5
    comptime EMBED = 4
    comptime IN_N = BATCH * VOCAB
    comptime OUT_N = BATCH * EMBED
    comptime W_N = VOCAB * EMBED
    var op = Embedding[VOCAB, EMBED].make[target="cpu", INIT=Zero]()
    # Set a non-trivial table.
    var w_ptr = op.weight.value_unsafe_ptr_cpu()
    for i in range(W_N):
        w_ptr[i] = Scalar[DT](0.2 * Float64(i) - 0.5)

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    # One-hot rows: token b -> index (b % VOCAB).
    for i in range(IN_N):
        x[i] = Scalar[DT](0.0)
    for b in range(BATCH):
        x[b * VOCAB + (b % VOCAB)] = Scalar[DT](1.0)
    _fill(go, OUT_N, 4.0)
    var x_t = TileTensor(x, row_major[BATCH, VOCAB]())
    var y_t = TileTensor(y, row_major[BATCH, EMBED]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    # Forward reference: y[b,:] == W[token_b, :].
    var fwd_err: Float64 = 0.0
    for b in range(BATCH):
        var tok = b % VOCAB
        for j in range(EMBED):
            var d = abs(
                Float64(y[b * EMBED + j]) - Float64(w_ptr[tok * EMBED + j])
            )
            if d > fwd_err:
                fwd_err = d
    print("   forward max err =", fwd_err)
    assert_true(fwd_err < 1e-6, "Embedding forward mismatch")

    op.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, EMBED]())
    var gi_t = TileTensor(gi, row_major[BATCH, VOCAB]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    # grad_input FD (treat one-hot input as continuous — op is linear in it).
    var fd_in = _alloc(IN_N)
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, OUT_N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, OUT_N)
        x[k] = orig
        fd_in[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m_in = _maxabs_diff_vs_fd_input[VOCAB]("grad_input", gi, fd_in, IN_N)
    assert_true(m_in < TOL, "Embedding grad_input vs FD")

    # grad_weight FD. Re-run forward to refresh the cache after perturbations.
    op.forward["cpu", BATCH](x_t, output=y_t)
    var gw = op.weight.grad_unsafe_ptr_cpu()
    var fd_w = _alloc(W_N)
    for k in range(W_N):
        var orig = w_ptr[k]
        w_ptr[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, OUT_N)
        w_ptr[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, OUT_N)
        w_ptr[k] = orig
        fd_w[k] = Scalar[DT]((lp - lm) / (2.0 * EPS))
    var m_w = _maxabs_diff_vs_fd_input[EMBED]("grad_weight", gw, fd_w, W_N)
    assert_true(m_w < TOL, "Embedding grad_weight vs FD")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Wave A transformer primitives (CPU) — docs/NN_TRANSFORMER_PORT.md")
    print("=" * 70)
    test_transpose2d()
    test_token_mean()
    test_bias_add()
    test_embedding()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
