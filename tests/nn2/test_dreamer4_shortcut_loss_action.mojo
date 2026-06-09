"""Shortcut-forcing loss with action conditioning (Phase 2 follow-up).

    pixi run mojo run -I . tests/nn2/test_dreamer4_shortcut_loss_action.mojo

End-to-end check that actions flow through ALL passes of the shortcut-forcing
loss (`dynamics_pretrain_loss[ADIM>0]` sets actions before the main forward
and the self-row subset before the two bootstrap halves). Trains a conditioned
Dreamer4Dynamics through the loss to fit fixed targets with a fixed action set,
then verifies the loss is ACTION-SENSITIVE: re-evaluating with a different
action (same z0/σ) yields a clearly higher loss. At init the action head is
ZeroLinear (contribution 0), so the sensitivity is entirely learned — proof
that the act-MLP gradients propagate through the multi-pass loss.
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
    print("Shortcut-forcing loss — action conditioning (CPU)")
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
    comptime B = 2
    comptime B_SELF = 1
    comptime BF = B * T
    comptime ND = NSP * DSP
    comptime N = BF * ND
    comptime STEPS = 200
    comptime LR = Scalar[DT](3e-3)

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, ADIM
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(dyn)](dyn)
    optim.lr = LR

    var z1 = _alloc(N)
    var z0 = _alloc(N)
    var sigma = _alloc(BF)
    var sidx = _alloc(BF)
    var pidx = _alloc(BF)
    var actA = _alloc(BF * ADIM)
    var actB = _alloc(BF * ADIM)
    var amask = _alloc(ADIM)
    var gz = _alloc(N)
    var zhat = _alloc(N)
    var gin = _alloc(N)

    for i in range(N):
        z1[i] = Scalar[DT](0.5 + 0.4 * sin(0.3 + 0.5 * Float64(i)))
        z0[i] = Scalar[DT](0.2 * sin(2.1 + 0.9 * Float64(i)))
    # σ must lie on the step grid (σ = j/2^step); use step=1, σ=0.5 (j=1) so
    # self-row σ_plus = σ + d/2 = 0.75 < 1 (avoids the 1/(1−σ) singularity).
    for bt in range(BF):
        sigma[bt] = 0.5
        sidx[bt] = 2.0          # j·(KMAX/K) = 1·(4/2)
        pidx[bt] = 1.0          # step=1 ⇒ d=0.5, d_half=0.25
    for a in range(ADIM):
        amask[a] = 1.0
    for i in range(BF * ADIM):
        actA[i] = Scalar[DT](0.7 * sin(0.2 + 0.8 * Float64(i)))
        actB[i] = Scalar[DT](-0.7 * sin(1.1 + 0.5 * Float64(i)))

    var gzt = TileTensor(gz, row_major[BF, ND]())
    var git = TileTensor(gin, row_major[BF, ND]())

    # ── train through the loss with action A ────────────────────────────
    var first: Float64 = 0.0
    var last: Float64 = 0.0
    for step in range(STEPS):
        optim.zero_grad["cpu"](dyn)
        var loss = dynamics_pretrain_loss[
            type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX, "cpu", ADIM
        ](
            dyn, z1, z0, sigma, sidx, pidx, step >= 30, gz, zhat,
            actions=actA, act_mask=amask,
        )
        dyn.vjp["cpu", BF](gzt, git)
        optim.step["cpu"](dyn)
        if step == 0:
            first = loss
        last = loss
        if step % 40 == 0:
            print("   step", step, " loss =", loss)
    print("   first =", first, "  last =", last)

    # ── action sensitivity: same z0/σ, action A vs B ────────────────────
    var loss_A = dynamics_pretrain_loss[
        type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX, "cpu", ADIM
    ](dyn, z1, z0, sigma, sidx, pidx, False, gz, zhat, actions=actA, act_mask=amask)
    var loss_B = dynamics_pretrain_loss[
        type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX, "cpu", ADIM
    ](dyn, z1, z0, sigma, sidx, pidx, False, gz, zhat, actions=actB, act_mask=amask)
    print("   loss(action A) =", loss_A, "  loss(action B) =", loss_B)

    assert_true(last < 0.5 * first, "conditioned loss must decrease (learns)")
    assert_true(loss_B > loss_A * 1.2, "loss must be action-sensitive (A≪B)")
    print("=" * 70)
    print("ALL PASSED — shortcut-forcing loss action conditioning (CPU)")
    print("=" * 70)
