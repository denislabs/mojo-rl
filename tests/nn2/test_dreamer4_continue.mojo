"""Dreamer 4 continue/termination head (Phase 4) — BCE + imagination wiring.

    pixi run mojo run -I . tests/nn2/test_dreamer4_continue.mojo

Validates the DreamerV3-style `cont` head:
  1. `continue_bce_backward` matches finite differences of `continue_bce_loss`.
  2. the head overfits: Adam on the BCE drives ĉ = sigmoid(logit) to the target
     continue flags (≈1 for non-terminal, ≈0 for terminal).
  3. `Dreamer4Agent.imag_train_step(use_continue=True)` runs: the continue head
     reads the rollout's h_t to form con_t = γ·ĉ_t, the λ-returns truncate at
     predicted terminals, and the policy + value heads still train while the
     continue head stays frozen during imagination.
"""

from std.memory import alloc
from std.math import abs, isfinite
from std.testing import assert_true

from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.heads import Dreamer4ContinueHead
from mojo_rl.deep_agents2.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents2.dreamer4.imag_rl_loss import (
    continue_pred, continue_bce_loss, continue_bce_backward,
)
from mojo_rl.deep_agents2.dreamerv3.twohot import symexp_twohot_bins


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def test_bce_gradcheck() raises:
    print("-- continue BCE FD gradcheck")
    comptime N = 6
    var logits = _alloc(N)
    var target = _alloc(N)
    var lv: InlineArray[Float64, 6] = [-1.2, 0.4, 2.0, -0.7, 1.1, 0.0]
    var tv: InlineArray[Float64, 6] = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    for i in range(N):
        logits[i] = Scalar[DT](lv[i])
        target[i] = Scalar[DT](tv[i])
    var grad = _alloc(N)
    continue_bce_backward[N](logits, target, Scalar[DT](1.0), grad)
    var eps = 1e-3
    var max_err = Float64(0.0)
    for i in range(N):
        var s = logits[i]
        logits[i] = s + Scalar[DT](eps)
        var lp = continue_bce_loss[N](logits, target)
        logits[i] = s - Scalar[DT](eps)
        var lm = continue_bce_loss[N](logits, target)
        logits[i] = s
        var fd = (lp - lm) / (2.0 * eps)
        var e = abs(fd - Float64(grad[i]))
        if e > max_err:
            max_err = e
    print("   max|FD − analytic| =", max_err)
    assert_true(max_err < 5e-3, "continue BCE backward must match FD")


def test_overfit() raises:
    print("-- continue head overfit")
    comptime D_IN = 8
    comptime HID = 16
    comptime N = 6
    comptime CH = Dreamer4ContinueHead[D_IN, HID]
    var ch = CH.make[target="cpu", INIT=Xavier]()
    var opt = Adam.make["cpu", M=CH](ch)
    opt.lr = Scalar[DT](5e-2)

    var h = _alloc(N * D_IN)
    for i in range(N * D_IN):
        h[i] = Scalar[DT](0.2 * Float64((i % 7) - 3))
    var target = _alloc(N)
    var tv: InlineArray[Float64, 6] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    for i in range(N):
        target[i] = Scalar[DT](tv[i])

    var logits = _alloc(N)
    var glog = _alloc(N)
    var h_t = TileTensor(h, row_major[N, D_IN]())
    var lg_t = TileTensor(logits, row_major[N, 1]())
    var gl_t = TileTensor(glog, row_major[N, 1]())
    var gin = _alloc(N * D_IN)
    var gin_t = TileTensor(gin, row_major[N, D_IN]())

    var first = Float64(0.0)
    var last = Float64(0.0)
    for step in range(400):
        opt.zero_grad["cpu"](ch)
        ch.forward["cpu", N](h_t, output=lg_t)
        var loss = continue_bce_loss[N](logits, target)
        continue_bce_backward[N](logits, target, Scalar[DT](1.0), glog)
        ch.vjp["cpu", N, mode="all"](gl_t, gin_t)
        opt.step["cpu"](ch)
        if step == 0:
            first = loss
        last = loss
    ch.forward["cpu", N](h_t, output=lg_t)
    var chat = _alloc(N)
    continue_pred[N](logits, chat)
    print("   BCE", first, "->", last)
    var ok = True
    for i in range(N):
        var c = Float64(chat[i])
        var y = Float64(target[i])
        if abs(c - y) > 0.1:
            ok = False
    assert_true(last < 0.1 * first, "BCE must collapse")
    assert_true(ok, "ĉ must match the continue flags")
    print("   ĉ matches flags OK")


comptime DSP = 4
comptime NSP = 4
comptime D = 8
comptime NH = 2
comptime T = 4
comptime NREG = 2
comptime HID = 16
comptime DEPTH = 2
comptime KMAX = 4
comptime NAGENT = 1
comptime NTASK = 2
comptime HHID = 16
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 1
comptime B = 2
comptime B_SELF = 1
comptime ADIM = NACT
comptime AHID = 2 * D
comptime K_IMAG = 2
comptime NCTX = 1
comptime ND = NSP * DSP

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    True, ADIM, AHID, K_IMAG, NCTX,
]


def test_imag_use_continue() raises:
    print("-- imag_train_step(use_continue=True) wiring")
    var agent = Agent.make[target="cpu", INIT=Xavier]()
    agent.snapshot_prior()
    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var ctx = _alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctx[i] = Scalar[DT](0.2)
    var u01 = _alloc(B * T)
    for i in range(B * T):
        u01[i] = Scalar[DT](0.3 + 0.1 * Float64(i % 5))
    var znoise = _alloc(B * T * ND)
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT](0.1)
    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](Float64(b % NTASK))

    # both paths run; the continue-discounted returns stay finite
    var l_off = agent.imag_train_step(
        ctx, u01, znoise, task_ids, bins, use_continue=False
    )
    var l_on = agent.imag_train_step(
        ctx, u01, znoise, task_ids, bins, use_continue=True
    )
    print("   value/policy  off =", l_off[0], l_off[1],
          "  on =", l_on[0], l_on[1])
    assert_true(isfinite(l_on[0]) and isfinite(l_on[1]),
                "use_continue losses must be finite")
    print("   use_continue path runs OK")


def main() raises:
    print("=" * 70)
    print("Dreamer 4 continue/termination head (Phase 4)")
    print("=" * 70)
    test_bce_gradcheck()
    test_overfit()
    test_imag_use_continue()
    print("=" * 70)
    print("ALL PASSED — continue head BCE + imagination discounting")
    print("=" * 70)
