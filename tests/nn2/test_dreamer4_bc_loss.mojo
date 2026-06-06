"""Behavior-cloning MTP loss (eq. 9) — overfit + grad_h flow (Phase 3.6).

    pixi run mojo run -I . tests/nn2/test_dreamer4_bc_loss.mojo

Drives `bc_mtp_loss` on a synthetic dataset: per (sequence b, window-position
j) a target action and reward. The agent embeddings h_t are treated as a
trainable input (SGD via the returned grad_h) alongside the heads (Adam). If
the loss machinery — MTP alignment + end-of-window masking, both head losses,
and the grad-wrt-h_t reduction — is correct, the loss collapses and the policy
recovers the dataset actions (BC), proving grad_h carries real signal back to
the dynamics. Also checks the masked valid-prediction count.
"""

from std.memory import alloc
from std.math import sin, abs
from std.testing import assert_true, assert_equal

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.heads import (
    Dreamer4PolicyHead, Dreamer4RewardHead,
)
from mojo_rl.deep_agents2.dreamer4.bc_loss import bc_mtp_loss, bc_n_valid
from mojo_rl.deep_agents2.dreamerv3.dists_discrete import cat_argmax
from mojo_rl.deep_agents2.dreamerv3.twohot import symexp_twohot_bins


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime D_IN = 8
comptime HID = 32
comptime NACT = 4
comptime NBINS = 41
comptime NMTP = 3
comptime B = 3
comptime T = 4
comptime BT = B * T
comptime PLOG = NMTP * NACT
comptime RLOG = NMTP * NBINS


def main() raises:
    print("=" * 70)
    print("Dreamer 4 BC MTP loss — overfit + grad_h (Phase 3.6)")
    print("=" * 70)

    var nv = bc_n_valid(B, T, NMTP)
    print("   n_valid =", nv, " (expected", B * (3 + 3 + 2 + 1), ")")
    assert_equal(nv, B * (3 + 3 + 2 + 1), "valid-prediction count (masked)")

    comptime PH = Dreamer4PolicyHead[D_IN, HID, NACT, NMTP]
    comptime RH = Dreamer4RewardHead[D_IN, HID, NBINS, NMTP]
    var ph = PH.make[target="cpu", INIT=Xavier]()
    var rh = RH.make[target="cpu", INIT=Xavier]()
    var ap = Adam.make["cpu", M=PH](ph)
    var ar = Adam.make["cpu", M=RH](rh)
    ap.lr = Scalar[DT](3e-3)
    ar.lr = Scalar[DT](3e-3)

    # trainable agent embeddings (stand in for the dynamics h_t)
    var h = _alloc(BT * D_IN)
    for i in range(BT * D_IN):
        h[i] = Scalar[DT](0.3 * sin(0.4 + 0.6 * Float64(i)))

    # dataset targets per (b, window-position)
    var actions = _alloc(BT)
    var rewards = _alloc(BT)
    for b in range(B):
        for j in range(T):
            var p = b * T + j
            actions[p] = Scalar[DT](Float64((b + j) % NACT))
            rewards[p] = Scalar[DT](0.5 * sin(0.2 + 0.9 * Float64(p)))

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    # scratch
    var plog = _alloc(BT * PLOG)
    var rlog = _alloc(BT * RLOG)
    var gpl = _alloc(BT * PLOG)
    var grl = _alloc(BT * RLOG)
    var grad_h = _alloc(BT * D_IN)
    var grad_h_tmp = _alloc(BT * D_IN)

    comptime LR_H = Scalar[DT](0.1)
    var first: Float64 = 0.0
    var last: Float64 = 0.0
    var max_gh: Float64 = 0.0
    for step in range(400):
        ap.zero_grad["cpu"](ph)
        ar.zero_grad["cpu"](rh)
        var loss = bc_mtp_loss[PH, RH, B, T, NMTP, NACT, NBINS, D_IN](
            ph, rh, h, actions, rewards, bins,
            plog, rlog, gpl, grl, grad_h, grad_h_tmp,
        )
        ap.step["cpu"](ph)
        ar.step["cpu"](rh)
        # SGD on the embeddings via grad_h (validates grad_h flow)
        for i in range(BT * D_IN):
            h[i] = h[i] - LR_H * grad_h[i]
        if step == 0:
            first = loss
            for i in range(BT * D_IN):
                var g = abs(Float64(grad_h[i]))
                if g > max_gh:
                    max_gh = g
        last = loss
        if step % 80 == 0:
            print("   step", step, " BC loss =", loss)
    print("   BC loss", first, "->", last)
    print("   grad_h max|·| (step 0) =", max_gh)

    # refresh logits with the final params (the last loop iteration stepped
    # h + heads AFTER its forward, so `plog` would otherwise be one step stale)
    var _refresh = bc_mtp_loss[PH, RH, B, T, NMTP, NACT, NBINS, D_IN](
        ph, rh, h, actions, rewards, bins,
        plog, rlog, gpl, grl, grad_h, grad_h_tmp,
    )

    # policy accuracy over the valid (masked) MTP predictions
    var n_correct = 0
    for b in range(B):
        for j in range(T):
            var bt = b * T + j
            for n in range(NMTP):
                if j + n >= T:
                    break
                var pbase = bt * PLOG + n * NACT
                var pos = b * T + (j + n)
                var k = Int(Float64(actions[pos]) + 0.5)
                if cat_argmax[NACT](plog, pbase) == k:
                    n_correct += 1
    print("   action accuracy =", n_correct, "/", nv)

    assert_true(last < 0.3 * first, "BC loss must collapse")
    assert_true(max_gh > 1e-6, "grad_h must carry signal to the dynamics")
    assert_true(n_correct == nv, "BC recovers every (masked) dataset action")

    print("=" * 70)
    print("ALL PASSED — Dreamer 4 BC MTP loss (Phase 3.6)")
    print("=" * 70)
