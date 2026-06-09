"""Dreamer4Dynamics — action conditioning (ActionEncoder, Phase 2 follow-up).

    pixi run mojo run -I . tests/nn2/test_dreamer4_dynamics_action.mojo

With ADIM>0 the action token = action_base + act_mlp(clamp(mask⊙a)), where
act_mlp = Linear→SiLU→ZeroLinear (model.py:ActionEncoder; ZeroLinear fc2 ⇒
the action contribution starts EXACTLY 0). Two checks, both on a batch whose
samples share identical packed input + signal/step indices so the ONLY thing
that can distinguish their outputs is the action:

  1. INVARIANT (any training stage): same action ⇒ identical output. Holds
     because identical input+indices+action ⇒ identical assembled grid.
  2. CONDITIONING: overfit two samples to DISTINCT targets using DISTINCT
     actions. Loss must drop hard (grads flow through the act-MLP), and the
     trained outputs for the distinct actions must clearly differ — the model
     can only fit both by routing the action token. At init both outputs are 0
     (ZeroLinear flow head), so the differentiation is learned end-to-end.
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    print("=" * 70)
    print("Dreamer4Dynamics — action conditioning (ActionEncoder)")
    print("=" * 70)

    comptime DSP = 4
    comptime NSP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime NREG = 2
    comptime HID = 16
    comptime DEPTH = 2
    comptime KMAX = 4
    comptime ADIM = 3
    comptime BATCH = 2        # two samples sharing input+indices
    comptime IO = NSP * DSP
    comptime N = BATCH * IO
    comptime STEPS = 400
    comptime LR = Scalar[DT](3e-3)

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, ADIM
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(dyn)](dyn)
    optim.lr = LR

    var z = _alloc(N)
    var sig = _alloc(BATCH)
    var stp = _alloc(BATCH)
    var tgt = _alloc(N)
    var pred = _alloc(N)
    var gpred = _alloc(N)
    var gin = _alloc(N)
    var actions = _alloc(BATCH * ADIM)
    var act_mask = _alloc(ADIM)

    # identical packed input + indices across BOTH samples
    for j in range(IO):
        var v = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(j)))
        z[0 * IO + j] = v
        z[1 * IO + j] = v
    sig[0] = 2.0; sig[1] = 2.0
    stp[0] = 1.0; stp[1] = 1.0
    for a in range(ADIM):
        act_mask[a] = 1.0

    # distinct targets per sample (only the action can distinguish them)
    for j in range(IO):
        tgt[0 * IO + j] = Scalar[DT](0.4 * sin(1.1 + 0.5 * Float64(j)))
        tgt[1 * IO + j] = Scalar[DT](-0.4 * sin(0.7 + 0.6 * Float64(j)))
    # distinct actions per sample
    var a0 = [Scalar[DT](0.8), Scalar[DT](-0.5), Scalar[DT](0.3)]
    var a1 = [Scalar[DT](-0.6), Scalar[DT](0.4), Scalar[DT](-0.9)]
    for a in range(ADIM):
        actions[0 * ADIM + a] = a0[a]
        actions[1 * ADIM + a] = a1[a]

    var zt = TileTensor(z, row_major[BATCH, IO]())
    var pt = TileTensor(pred, row_major[BATCH, IO]())
    var git = TileTensor(gin, row_major[BATCH, IO]())
    dyn.set_indices(sig, stp, BATCH)

    # ── 1. invariant: same action ⇒ identical output (pre-training) ──────
    var same = _alloc(BATCH * ADIM)
    for a in range(ADIM):
        same[0 * ADIM + a] = a0[a]
        same[1 * ADIM + a] = a0[a]      # both samples get action a0
    dyn.set_actions(same, act_mask, BATCH)
    dyn.forward["cpu", BATCH](zt, output=pt)
    var d_same: Float64 = 0.0
    for j in range(IO):
        d_same = max(d_same, abs(Float64(pred[0 * IO + j]) - Float64(pred[1 * IO + j])))
    print("  same-action  max|out0-out1| =", d_same)
    assert_true(d_same < 1e-6, "same action ⇒ identical output")

    # ── 2. overfit distinct targets via distinct actions ────────────────
    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(STEPS):
        optim.zero_grad["cpu"](dyn)
        dyn.set_actions(actions, act_mask, BATCH)
        dyn.forward["cpu", BATCH](zt, output=pt)
        var loss: Float64 = 0.0
        for i in range(N):
            var diff = Float64(pred[i]) - Float64(tgt[i])
            loss += diff * diff
            gpred[i] = Scalar[DT](2.0 * diff / Float64(N))
        loss /= Float64(N)
        var got = TileTensor(gpred, row_major[BATCH, IO]())
        dyn.vjp["cpu", BATCH](got, git)
        optim.step["cpu"](dyn)
        if step == 0:
            first = loss
        last = loss
        if step % 80 == 0:
            print("   step", step, " loss =", loss)
    print("   first =", first, "  last =", last)

    # trained outputs for the distinct actions must clearly differ
    dyn.set_actions(actions, act_mask, BATCH)
    dyn.forward["cpu", BATCH](zt, output=pt)
    var d_diff: Float64 = 0.0
    for j in range(IO):
        d_diff = max(d_diff, abs(Float64(pred[0 * IO + j]) - Float64(pred[1 * IO + j])))
    print("   distinct-action max|out0-out1| =", d_diff)

    assert_true(last < 0.2 * first, "loss must drop hard (act-MLP learns)")
    assert_true(d_diff > 0.1, "distinct actions ⇒ clearly distinct outputs")
    print("=" * 70)
    print("ALL PASSED — action conditioning (CPU)")
    print("=" * 70)
