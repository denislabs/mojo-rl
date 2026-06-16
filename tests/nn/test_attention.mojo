"""ScaledDotProductAttention — CPU forward + finite-diff gradcheck (Wave C 6a/6b).

Attention is nonlinear (softmax), so we finite-difference the input grads
against the analytic vjp for both non-causal and causal modes, multi-head.
Also a direct single-head softmax forward reference for one position.

Docs: docs/NN_TRANSFORMER_PORT.md Phase 1 Wave C.
"""

from std.memory import alloc
from std.math import abs, exp, sqrt
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.attention import ScaledDotProductAttention


comptime EPS: Float64 = 2e-3
comptime TOL: Float64 = 1.5e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _loss(
    y: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(y[i]) * Float64(go[i])
    return s


def _run_gradcheck[
    DIM: Int, N_HEADS: Int, SEQ: Int, CAUSAL: Bool
](name: String) raises:
    print(name, "...")
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM

    var op = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make[target="cpu", INIT=Zero]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        go[i] = _spread(i, 4.1)

    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM * 3]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * DIM]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    var go_t = TileTensor(go, row_major[BATCH, SEQ * DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * DIM * 3]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    # FD over every input element.
    var max_err: Float64 = 0.0
    var max_rel: Float64 = 0.0
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, OUT_N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, OUT_N)
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
        var denom = abs(fd) + 1.0
        if d / denom > max_rel:
            max_rel = d / denom
    print("   max|analytic - FD| =", max_err, "  max rel =", max_rel)
    assert_true(max_err < TOL, name + ": grad_input vs FD")
    print("  ok")


def test_softmax_forward_reference() raises:
    """Single (b,h)=(0,0), query i=0 forward against a direct softmax."""
    print("test_softmax_forward_reference ...")
    comptime DIM = 2
    comptime N_HEADS = 1
    comptime SEQ = 3
    comptime BATCH = 1
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM
    var op = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, False
    ].make[target="cpu", INIT=Zero]()
    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = _spread(i, 0.9)
    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM * 3]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * DIM]())
    op.forward["cpu", BATCH](x_t, output=y_t)

    # Reference for query i=0 over all keys j.
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var scale = 1.0 / sqrt(Float64(DIM))  # head_dim == DIM here
    var scores = List[Float64]()
    var mx = -1e30
    for j in range(SEQ):
        var s: Float64 = 0.0
        for d in range(DIM):
            s += Float64(x[0 * DIM + d]) * Float64(x[KOFF + j * DIM + d])
        s *= scale
        scores.append(s)
        if s > mx:
            mx = s
    var se: Float64 = 0.0
    for j in range(SEQ):
        se += exp(scores[j] - mx)
    var ref0: Float64 = 0.0
    var ref1: Float64 = 0.0
    for j in range(SEQ):
        var w = exp(scores[j] - mx) / se
        ref0 += w * Float64(x[VOFF + j * DIM + 0])
        ref1 += w * Float64(x[VOFF + j * DIM + 1])
    var e0 = abs(Float64(y[0]) - ref0)
    var e1 = abs(Float64(y[1]) - ref1)
    print("   out[0] err =", e0, "  out[1] err =", e1)
    assert_true(e0 < 1e-5 and e1 < 1e-5, "softmax forward reference")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ScaledDotProductAttention CPU (Wave C 6a/6b)")
    print("=" * 70)
    test_softmax_forward_reference()
    _run_gradcheck[4, 2, 3, False]("test_gradcheck_noncausal_mh")
    _run_gradcheck[4, 2, 3, True]("test_gradcheck_causal_mh")
    _run_gradcheck[6, 1, 4, False]("test_gradcheck_singlehead")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
