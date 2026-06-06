"""Shortcut-forcing flow-matching loss — CPU convergence (Phase 2.3).

Trains Dreamer4Dynamics with `dynamics_pretrain_loss` on a fixed batch of
clean latents (fixed σ / step / z0 noise). Checks:
  1. step-0 loss is finite & positive (ẑ=0 at the zero-init flow head);
  2. the loss drops substantially over training — the empirical flow term +
     the bootstrap consistency term produce correct-sign gradients that the
     vjp + Adam loop turns into learning.

Bootstrap is enabled after `BOOT_START` steps (matches the reference's
ramp-in). Numeric parity vs a PyTorch fixture (≈1e-5) is the §6 gate and is
deferred to fixture work; this test validates the end-to-end optimisation.
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics
from mojo_rl.deep_agents2.dreamer4.shortcut_loss import dynamics_pretrain_loss


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    print("=" * 70)
    print("Dreamer4 shortcut-forcing loss — CPU convergence (Phase 2.3)")
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
    comptime B = 2
    comptime B_SELF = 1
    comptime BF = B * T
    comptime ND = NSP * DSP
    comptime N = BF * ND
    comptime STEPS = 250
    comptime BOOT_START = 20
    comptime LR = Scalar[DT](2e-3)

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(dyn)](dyn)
    optim.lr = LR

    var z1 = _alloc(N)
    var z0 = _alloc(N)
    var sigma = _alloc(BF)
    var sig_idx = _alloc(BF)
    var step_idx = _alloc(BF)
    var grad_zhat = _alloc(N)
    var zhat = _alloc(N)
    var gin = _alloc(N)

    # fixed targets + noise
    for i in range(N):
        z1[i] = Scalar[DT](0.5 + 0.4 * sin(0.3 + 0.5 * Float64(i)))
        z0[i] = Scalar[DT](0.2 * sin(2.1 + 0.9 * Float64(i)))
    # row 0 = empirical (finest step e_max=2); row 1 = self (coarser step 1)
    for t in range(T):
        sigma[0 * T + t] = 0.5
        sig_idx[0 * T + t] = 2.0
        step_idx[0 * T + t] = 2.0          # e_max = log2(KMAX)
        sigma[1 * T + t] = 0.3
        sig_idx[1 * T + t] = 1.0
        step_idx[1 * T + t] = 1.0          # coarser → d = 1/2

    var git = TileTensor(gin, row_major[BF, ND]())

    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(STEPS):
        var do_boot = step >= BOOT_START
        optim.zero_grad["cpu"](dyn)
        var loss = dynamics_pretrain_loss[
            type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX
        ](dyn, z1, z0, sigma, sig_idx, step_idx, do_boot, grad_zhat, zhat)
        var gz = TileTensor(grad_zhat, row_major[BF, ND]())
        dyn.vjp["cpu", BF](gz, git)
        optim.step["cpu"](dyn)
        if step == 0:
            first = loss
            assert_true(loss > 0.0, "step-0 loss positive")
            assert_true(loss == loss, "step-0 loss finite (not NaN)")
        last = loss
        if step % 50 == 0:
            print("   step", step, " do_boot=", do_boot, " loss =", loss)

    print("   first =", first, "  last =", last)
    assert_true(last < first, "loss must decrease")
    assert_true(last < 0.5 * first, "loss must drop substantially")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
