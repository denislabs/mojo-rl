"""MuZero K-step unroll BPTT — overfit a fixed batch on CPU (no GPU).

The core correctness check for `muzero/blocks.mojo::mz_unroll_train_step_cpu`:
repeatedly training on ONE fixed batch must drive the total loss (policy + value
+ reward, all categorical soft-CE through the learned h/g/f unroll) far down. If
the forward scan, the reverse-scan carry gradient, the ½ dynamics-input scale,
or any vjp wiring were wrong, the loss would stall or diverge. Policy targets are
one-hot (policy CE floors at 0); value/reward two-hot targets floor at their
(small) bin entropy.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mz_unroll_overfit_cpu.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.blocks import mz_unroll_train_step_cpu


def main() raises:
    comptime B = 4
    comptime K = 3
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 8
    comptime BINS = 21
    comptime H = 16
    var v_min = Scalar[DT](-10.0)
    var v_max = Scalar[DT](10.0)

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](0.01))
    var odyn = Adam(lr=Scalar[DT](0.01))
    var opred = Adam(lr=Scalar[DT](0.01))

    # ── one fixed batch (time-major) — owned Lists (List-input unroll) ──
    var obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    var xs = UInt64(0x9E3779B97F4A7C15)
    for i in range(B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs0[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)

    var actions = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        actions[i] = Scalar[DT](Int(xs % ACT))

    # one-hot policy targets per (k,b) → policy CE floors at 0
    var policy_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    for i in range((K + 1) * B * ACT):
        policy_tgt[i] = Scalar[DT](0.0)
    for k in range(K + 1):
        for b in range(B):
            xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
            var a = Int(xs % ACT)
            policy_tgt[k * B * ACT + b * ACT + a] = Scalar[DT](1.0)

    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for i in range((K + 1) * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        value_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)

    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        reward_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(400):
        var l = mz_unroll_train_step_cpu[
            Rep, Dyn, Pred, B, K, OBS, ACT, LATENT, BINS
        ](
            rep, dyn, pred, orep, odyn, opred,
            obs0, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max,
        )
        if it == 0:
            first = l
        last = l
        if it % 80 == 0:
            print("it", it, "loss", l)

    print("first", first, "last", last)
    assert_true(first == first and last == last, "loss became NaN")
    # The total is summed over K+1 value + K+1 policy + K reward soft-CE terms;
    # one-hot policy CE → ~0, but the two-hot value/reward targets floor at their
    # (nonzero) bin entropy, so the irreducible floor is a few nats. The strong
    # signal that the BPTT carry + ½ dyn-scale + vjp wiring are correct is the
    # large reduction toward that floor (≥5×).
    assert_true(last < first * Scalar[DT](0.2), "unroll failed to overfit (≥5×)")

    print("MuZero unroll BPTT overfit (CPU): OK")
