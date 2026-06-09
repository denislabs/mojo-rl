"""Dreamer4Tokenizer — CPU overfit training step (Phase 1).

Wires encoder → decoder → masked-MSE loss → Adam over one fixed batch with a
fixed MAE mask (advance_rng NOT called). The masked-reconstruction loss must
drop substantially — the end-to-end forward/backward/optimizer loop learns.
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents2.dreamer4.recon_loss import masked_recon_loss


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    # smooth, bounded target in [0.1, 0.9] (sigmoid-reachable patches)
    return Scalar[DT](0.5 + 0.4 * sin(seed + 0.7 * Float64(i)))


def main() raises:
    print("=" * 70)
    print("Dreamer4Tokenizer — CPU overfit (Phase 1)")
    print("=" * 70)
    comptime DP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime L = 2
    comptime NP = 4
    comptime D_BOT = 4
    comptime HID = 16
    comptime DEPTH = 2
    comptime B = 1
    comptime BATCH = B * T
    comptime N = BATCH * NP * DP
    comptime STEPS = 300
    comptime LR = Scalar[DT](3e-3)

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, 0.4, 0.4, 7
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(tok)](tok)
    optim.lr = LR

    var x = _alloc(N)         # fixed batch (targets)
    var pred = _alloc(N)
    var gpred = _alloc(N)
    var gin = _alloc(N)       # throwaway grad_input
    for i in range(N):
        x[i] = _spread(i, 1.1)
    var xt = TileTensor(x, row_major[BATCH, NP * DP]())
    var pt = TileTensor(pred, row_major[BATCH, NP * DP]())
    var git = TileTensor(gin, row_major[BATCH, NP * DP]())

    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(STEPS):
        optim.zero_grad["cpu"](tok)
        tok.forward["cpu", BATCH](xt, output=pt)
        var mask = tok.mae_mask_ptr()  # fixed (advance_rng not called)
        # loss compares the reconstruction `pred` to the original patches `x`
        # on the dropped positions; fills `gpred` = dL/dpred.
        var loss = masked_recon_loss[NP, DP, BATCH](pred, x, mask, gpred)
        var got = TileTensor(gpred, row_major[BATCH, NP * DP]())
        tok.vjp["cpu", BATCH](got, git)
        optim.step["cpu"](tok)
        if step == 0:
            first = loss
        last = loss
        if step % 60 == 0:
            print("   step", step, " loss =", loss)

    print("   first =", first, "  last =", last)
    assert_true(last < first, "loss must decrease")
    assert_true(last < 0.4 * first, "loss must drop substantially (overfit)")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
