"""SequenceCrossEntropyLoss — CPU forward ref_loss + FD gradcheck.

Treats (BATCH, SEQ*VOCAB) as (BATCH*SEQ, VOCAB) per-token softmax-CE,
averaged over all positions. Docs: NN_TRANSFORMER_PORT.md Wave D.
"""

from std.memory import alloc
from std.math import abs, exp, log
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss import SequenceCrossEntropyLoss


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 5e-3


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _spread(i: Int, s: Float64) -> Scalar[DT]:
    var x = s + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.6 * (t - (t * t * t) / 6.0))


def main() raises:
    print("=" * 70)
    print("SequenceCrossEntropyLoss CPU (forward ref_loss + FD gradcheck)")
    print("=" * 70)
    comptime BATCH = 2
    comptime SEQ = 3
    comptime VOCAB = 4
    comptime N = BATCH * SEQ * VOCAB
    comptime BT = BATCH * SEQ

    var loss = SequenceCrossEntropyLoss[SEQ, VOCAB].make["cpu"]()

    var logits = _alloc(N)
    var tgt = _alloc(N)
    var grad = _alloc(N)
    for i in range(N):
        logits[i] = _spread(i, 1.0)
    # One-hot target per token row r → class r % VOCAB.
    for i in range(N):
        tgt[i] = 0.0
    for r in range(BT):
        tgt[r * VOCAB + (r % VOCAB)] = 1.0

    var lg_t = TileTensor(logits, row_major[BATCH, SEQ * VOCAB]())
    var tg_t = TileTensor(tgt, row_major[BATCH, SEQ * VOCAB]())
    var L = Float64(loss.forward["cpu", BATCH](lg_t, tg_t))

    # Reference: mean per-token CE.
    var ref_loss: Float64 = 0.0
    for r in range(BT):
        var base = r * VOCAB
        var m = Float64(logits[base])
        for c in range(1, VOCAB):
            if Float64(logits[base + c]) > m:
                m = Float64(logits[base + c])
        var se: Float64 = 0.0
        for c in range(VOCAB):
            se += exp(Float64(logits[base + c]) - m)
        var lse = m + log(se)
        var tgt_c = r % VOCAB
        ref_loss += -(Float64(logits[base + tgt_c]) - lse)
    ref_loss /= Float64(BT)
    print("   forward L =", L, "  ref_loss =", ref_loss, "  err =", abs(L - ref_loss))
    assert_true(abs(L - ref_loss) < 1e-5, "forward vs manual per-token CE")

    # FD gradcheck on logits.
    var grad_t = TileTensor(grad, row_major[BATCH, SEQ * VOCAB]())
    loss.vjp["cpu", BATCH](tg_t, grad_t)
    var max_err: Float64 = 0.0
    for k in range(N):
        var orig = logits[k]
        logits[k] = orig + Scalar[DT](EPS)
        var lp = Float64(loss.forward["cpu", BATCH](lg_t, tg_t))
        logits[k] = orig - Scalar[DT](EPS)
        var lm = Float64(loss.forward["cpu", BATCH](lg_t, tg_t))
        logits[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(grad[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| grad =", max_err)
    assert_true(max_err < TOL, "grad_logits vs FD")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
