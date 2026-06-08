"""LSTMSeq (recurrent encoder Module) — CPU finite-difference gradcheck.

Validates the unroll + BPTT wrapped behind `Module.forward`/`vjp`: the
analytic `grad_input` and the cell's `W_ih` gradient match central finite
differences of the scalar loss `<output, grad_output>`. (The CPU↔GPU
parity of the same primitive is exercised separately on GPU hardware.)

Run:
    pixi run mojo run -I . tests/nn2/test_lstm_seq.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.lstm_seq import LSTMSeq
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.core.module import mptr

comptime V = 4
comptime H = 3
comptime S = 5
comptime B = 2
comptime IN = S * V
comptime OUT = S * H


def _fwd_loss(
    mut net: LSTMSeq[V, H, S],
    inp: List[Scalar[DT]],
    go: List[Scalar[DT]],
) raises -> Float64:
    var out = List[Scalar[DT]](length=B * OUT, fill=0.0)
    var in_tt = TileTensor(mptr(inp.unsafe_ptr()), row_major[B, IN]())
    var out_tt = TileTensor(mptr(out.unsafe_ptr()), row_major[B, OUT]())
    net.forward["cpu", B](in_tt, output=out_tt)
    var l: Float64 = 0.0
    for i in range(B * OUT):
        l += Float64(out[i]) * Float64(go[i])
    return l


def test_lstm_seq_gradcheck() raises:
    print("test_lstm_seq_gradcheck ...", end=" ")
    seed(3)
    var net = LSTMSeq[V, H, S].make[target="cpu", INIT=Xavier]()
    var inp = List[Scalar[DT]](length=B * IN, fill=0.0)
    var go = List[Scalar[DT]](length=B * OUT, fill=0.0)
    for i in range(B * IN):
        inp[i] = Scalar[DT](random_float64(-1.0, 1.0))
    for i in range(B * OUT):
        go[i] = Scalar[DT](random_float64(-1.0, 1.0))

    # Analytic grads: grad_input + cell param grads from one forward+vjp.
    var gin = List[Scalar[DT]](length=B * IN, fill=0.0)
    var out = List[Scalar[DT]](length=B * OUT, fill=0.0)
    var in_tt = TileTensor(mptr(inp.unsafe_ptr()), row_major[B, IN]())
    var out_tt = TileTensor(mptr(out.unsafe_ptr()), row_major[B, OUT]())
    var gin_tt = TileTensor(mptr(gin.unsafe_ptr()), row_major[B, IN]())
    var go_tt = TileTensor(mptr(go.unsafe_ptr()), row_major[B, OUT]())
    net.forward["cpu", B](in_tt, output=out_tt)
    net.zero_grad["cpu"]()
    net.vjp["cpu", B](go_tt, gin_tt)

    comptime eps = 1e-3
    var max_gin_err: Float64 = 0.0
    for k in range(B * IN):
        var save = inp[k]
        inp[k] = save + Scalar[DT](eps)
        var lp = _fwd_loss(net, inp, go)
        inp[k] = save - Scalar[DT](eps)
        var lm = _fwd_loss(net, inp, go)
        inp[k] = save
        var fd = (lp - lm) / (2.0 * eps)
        max_gin_err = max(max_gin_err, abs(fd - Float64(gin[k])))

    var max_w_err: Float64 = 0.0
    for j in range(LSTMSeq[V, H, S].Cell.W_IH_SIZE):
        var save = net.cell.W_ih.val.cpu[j]
        net.cell.W_ih.val.cpu[j] = save + Scalar[DT](eps)
        var lp = _fwd_loss(net, inp, go)
        net.cell.W_ih.val.cpu[j] = save - Scalar[DT](eps)
        var lm = _fwd_loss(net, inp, go)
        net.cell.W_ih.val.cpu[j] = save
        var fd = (lp - lm) / (2.0 * eps)
        max_w_err = max(max_w_err, abs(fd - Float64(net.cell.W_ih.grd.cpu[j])))

    assert_true(
        max_gin_err < 1e-2,
        "grad_input FD mismatch: " + String(max_gin_err),
    )
    assert_true(
        max_w_err < 1e-2,
        "W_ih grad FD mismatch: " + String(max_w_err),
    )
    print(
        "PASS (grad_input err=" + String(max_gin_err)[byte=:9]
        + ", W_ih err=" + String(max_w_err)[byte=:9] + ")"
    )


def main() raises:
    print("=" * 60)
    print("LSTMSeq tests")
    print("=" * 60)
    test_lstm_seq_gradcheck()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
