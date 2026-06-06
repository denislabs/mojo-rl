"""Dreamer4Dynamics — CPU forward sanity + overfit (Phase 2.2).

Two checks on the bespoke dynamics module (no shortcut loss yet — plain MSE):
  1. Zero-init: the flow_x_head is ZeroLinear, so the very first forward is
     all zeros regardless of input (x-prediction starts at the shortcut fixed
     point).
  2. Overfit: train output → fixed bounded target with Adam; the loss must
     drop substantially. Exercises tail.vjp + proj.vjp + the conditioning
     param grads (action_base / signal_table / step_table / register).
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
    print("Dreamer4Dynamics — CPU forward sanity + overfit (Phase 2.2)")
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
    comptime B = 1
    comptime BATCH = B * T
    comptime IO = NSP * DSP
    comptime N = BATCH * IO
    comptime STEPS = 300
    comptime LR = Scalar[DT](3e-3)

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(dyn)](dyn)
    optim.lr = LR

    var z = _alloc(N)          # packed latents (input)
    var sig = _alloc(BATCH)    # signal idx per sample
    var stp = _alloc(BATCH)    # step idx per sample
    var tgt = _alloc(N)        # fixed target
    var pred = _alloc(N)
    var gpred = _alloc(N)
    var gin = _alloc(N)        # throwaway grad_input (packed)

    for i in range(N):
        z[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
        tgt[i] = Scalar[DT](0.5 + 0.4 * sin(1.3 + 0.6 * Float64(i)))
    for bt in range(BATCH):
        sig[bt] = Scalar[DT](Float64((bt + 1) % (KMAX + 1)))    # in [0,KMAX]
        stp[bt] = Scalar[DT](Float64(bt % 2))                   # in [0,NSTEP)

    var zt = TileTensor(z, row_major[BATCH, IO]())
    var pt = TileTensor(pred, row_major[BATCH, IO]())
    var git = TileTensor(gin, row_major[BATCH, IO]())
    dyn.set_indices(sig, stp, BATCH)

    # ── 1. zero-init forward ──────────────────────────────────────────
    dyn.forward["cpu", BATCH](zt, output=pt)
    var maxabs: Float64 = 0.0
    for i in range(N):
        var a = abs(Float64(pred[i]))
        if a > maxabs:
            maxabs = a
    print("  zero-init max|pred| =", maxabs)
    assert_true(maxabs < 1e-6, "zero-init flow head ⇒ first forward all zeros")

    # ── 2. overfit ────────────────────────────────────────────────────
    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(STEPS):
        optim.zero_grad["cpu"](dyn)
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
        if step % 60 == 0:
            print("   step", step, " loss =", loss)

    print("   first =", first, "  last =", last)
    assert_true(last < first, "loss must decrease")
    assert_true(last < 0.3 * first, "loss must drop substantially (overfit)")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
