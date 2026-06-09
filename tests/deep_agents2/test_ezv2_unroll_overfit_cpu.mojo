"""EZv2 K-step unroll BPTT + consistency — overfit a fixed batch on CPU.

The core correctness check for `efficient_zero_v2/blocks.mojo::
ezv2_unroll_train_step_cpu`: repeatedly training on ONE fixed batch must drive
the total loss far down. This exercises the MuZero BPTT carry + ½ dynamics scale
(shared with the validated MuZero unroll) AND the new SimSiam consistency wiring
— the online ``h_pred(g_proj(z_k))`` branch, the stop-grad ``g_proj(h(obs_k))``
target pre-pass, the consistency gradient folded into the z_k carry, and the
re-forward-rep-before-vjp cache discipline. If any of those were wrong the loss
would stall, diverge, or NaN.

The reported total folds the MuZero CE terms (policy one-hot → ~0; value/reward
two-hot floor at bin entropy) plus the consistency term ``Σ_k −cos`` (bounded in
``[−K, K]`` — pushes the total *down* as alignment improves), so the strong
signal is a large finite reduction.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_ezv2_unroll_overfit_cpu.mojo
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.blocks import (
    ezv2_unroll_train_step_cpu,
)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    comptime B = 4
    comptime K = 3
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 8
    comptime BINS = 21
    comptime H = 16
    comptime PROJ = 16
    comptime PROJ_HID = 16
    comptime BOTTLENECK = 8
    var v_min = Scalar[DT](-10.0)
    var v_max = Scalar[DT](10.0)

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var proj = Proj.make["cpu", INIT=Kaiming]()
    var predh = Predh.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    var oproj = Adam.make["cpu", M=Proj](proj)
    var opredh = Adam.make["cpu", M=Predh](predh)
    orep.lr = Scalar[DT](0.01)
    odyn.lr = Scalar[DT](0.01)
    opred.lr = Scalar[DT](0.01)
    oproj.lr = Scalar[DT](0.01)
    opredh.lr = Scalar[DT](0.01)

    var xs = UInt64(0x9E3779B97F4A7C15)

    # ── one fixed batch (time-major) — full obs sequence [K+1, B, OBS] ──
    var obs_seq = _a((K + 1) * B * OBS)
    for i in range((K + 1) * B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs_seq[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](
            1.0
        )

    var actions = _a(K * B)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        actions[i] = Scalar[DT](Int(xs % ACT))

    var policy_tgt = _a((K + 1) * B * ACT)
    for i in range((K + 1) * B * ACT):
        policy_tgt[i] = Scalar[DT](0.0)
    for k in range(K + 1):
        for b in range(B):
            xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
            var a = Int(xs % ACT)
            policy_tgt[k * B * ACT + b * ACT + a] = Scalar[DT](1.0)

    var value_tgt = _a((K + 1) * B)
    for i in range((K + 1) * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        value_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[
            DT
        ](1.0)

    var reward_tgt = _a(K * B)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        reward_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[
            DT
        ](1.0)

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(400):
        var l = ezv2_unroll_train_step_cpu[
            Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT, LATENT, BINS
        ](
            rep, dyn, pred, proj, predh,
            orep, odyn, opred, oproj, opredh,
            obs_seq, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max,
        )
        if it == 0:
            first = l
        last = l
        if it % 80 == 0:
            print("it", it, "loss", l)

    print("first", first, "last", last)
    assert_true(first == first and last == last, "loss became NaN")
    # CE terms drop toward their bin-entropy floor; consistency −cos pushes lower.
    # A large finite reduction confirms the BPTT carry + consistency wiring.
    assert_true(last < first * Scalar[DT](0.2), "unroll failed to overfit (≥5×)")

    obs_seq.free(); actions.free(); policy_tgt.free()
    value_tgt.free(); reward_tgt.free()
    print("EZv2 unroll BPTT + consistency overfit (CPU): OK")
