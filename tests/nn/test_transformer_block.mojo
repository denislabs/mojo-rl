"""TransformerBlock — end-to-end CPU gradcheck (Wave D checkpoint 1).

Finite-differences grad_input through the full composed graph
(Tokenwise+LayerNorm → MHA(QKV proj → attention → out proj) → Residual,
then Tokenwise+LayerNorm → FFN(Linear→GELU→Linear) → Residual), for both
non-causal and causal. If FD matches the analytic vjp, the whole
composition's backward is wired correctly. Docs: NN2_TRANSFORMER_PORT.md.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.models.transformer import TransformerBlock


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 3e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.4 * (t - (t * t * t) / 6.0))


def _loss(
    y: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(y[i]) * Float64(go[i])
    return s


def _run[
    DIM: Int, N_HEADS: Int, SEQ: Int, FF: Int, CAUSAL: Bool
](name: String) raises:
    print(name, "...")
    comptime BATCH = 2
    comptime N = BATCH * SEQ * DIM

    var blk = TransformerBlock[
        DIM, N_HEADS, SEQ, FF, CAUSAL
    ].make[target="cpu", INIT=Kaiming]()

    var x = _alloc(N)
    var y = _alloc(N)
    var go = _alloc(N)
    var gi = _alloc(N)
    for i in range(N):
        x[i] = _spread(i, 1.1)
        go[i] = _spread(i, 3.7)

    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * DIM]())
    blk.forward["cpu", BATCH](x_t, output=y_t)

    var fin: Float64 = 0.0
    for i in range(N):
        fin += Float64(y[i])
    print("   forward sum (finite check) =", fin)

    blk.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, SEQ * DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * DIM]())
    blk.vjp["cpu", BATCH](go_t, gi_t)

    var max_err: Float64 = 0.0
    for k in range(N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        blk.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, N)
        x[k] = orig - Scalar[DT](EPS)
        blk.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, N)
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| grad_input =", max_err)
    assert_true(max_err < TOL, name + ": grad_input vs FD")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("TransformerBlock end-to-end CPU gradcheck (Wave D)")
    print("=" * 70)
    _run[8, 2, 4, 16, False]("transformer_block_noncausal")
    _run[8, 2, 4, 16, True]("transformer_block_causal")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
