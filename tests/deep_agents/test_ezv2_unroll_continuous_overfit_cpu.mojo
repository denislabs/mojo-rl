"""EZv2 continuous K-step unroll — overfit a fixed batch on CPU (no GPU).

Correctness check for `efficient_zero_v2/blocks_continuous.mojo::
ezv2_unroll_train_step_continuous_cpu`: training on ONE fixed batch must drive
the total loss far down. Exercises the MuZero BPTT carry + ½ dynamics scale, the
SimSiam consistency wiring, AND the new squashed-Gaussian policy head (NLL of the
target action over the [μ_raw|σ_raw] slice + continuous-action dynamics input).

The reported total folds the policy NLL (floors low / can go negative as μ→target,
σ→min_std), the value/reward soft-CE (floor at bin entropy), and consistency
(−cos). The strong signal is a large finite reduction.

Run:
    pixi run mojo run -I . tests/deep_agents/test_ezv2_unroll_continuous_overfit_cpu.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents.efficient_zero_v2.blocks_continuous import (
    ezv2_unroll_train_step_continuous_cpu,
)



def main() raises:
    comptime B = 4
    comptime K = 3
    comptime OBS = 3       # Pendulum
    comptime ACT_DIM = 1
    comptime LATENT = 8
    comptime BINS = 21
    comptime H = 16
    comptime PROJ = 16
    comptime PROJ_HID = 16
    comptime BOTTLENECK = 8
    var v_min = Scalar[DT](-10.0)
    var v_max = Scalar[DT](10.0)

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT_DIM, BINS, H]
    comptime Pred = EZContPredNet[LATENT, ACT_DIM, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var proj = Proj.make["cpu", Kaiming]()
    var predh = Predh.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](1e-3))
    var odyn = Adam(lr=Scalar[DT](1e-3))
    var opred = Adam(lr=Scalar[DT](1e-3))
    var oproj = Adam(lr=Scalar[DT](1e-3))
    var opredh = Adam(lr=Scalar[DT](1e-3))
    orep.lr = Scalar[DT](0.01)
    odyn.lr = Scalar[DT](0.01)
    opred.lr = Scalar[DT](0.01)
    oproj.lr = Scalar[DT](0.01)
    opredh.lr = Scalar[DT](0.01)

    var xs = UInt64(0x9E3779B97F4A7C15)

    var obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    for i in range((K + 1) * B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs_seq[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](
            1.0
        )

    # continuous actions in [-0.8, 0.8].
    var actions = List[Scalar[DT]](length=K * B * ACT_DIM, fill=0)
    for i in range(K * B * ACT_DIM):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        actions[i] = Scalar[DT](Int(xs % 160)) / Scalar[DT](100.0) - Scalar[DT](
            0.8
        )
    # policy targets at K+1 positions (independent draw).
    var policy_act_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT_DIM, fill=0)
    for i in range((K + 1) * B * ACT_DIM):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        policy_act_tgt[i] = Scalar[DT](Int(xs % 160)) / Scalar[DT](
            100.0
        ) - Scalar[DT](0.8)

    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for i in range((K + 1) * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        value_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[
            DT
        ](1.0)

    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        reward_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[
            DT
        ](1.0)

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(400):
        var l = ezv2_unroll_train_step_continuous_cpu[
            Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT_DIM, LATENT, BINS
        ](
            rep, dyn, pred, proj, predh,
            orep, odyn, opred, oproj, opredh,
            obs_seq, actions, policy_act_tgt, value_tgt, reward_tgt,
            v_min, v_max,
        )
        if it == 0:
            first = l
        last = l
        if it % 80 == 0:
            print("it", it, "loss", l)

    print("first", first, "last", last)
    assert_true(first == first and last == last, "loss became NaN")
    assert_true(last < first * Scalar[DT](0.3), "continuous unroll failed to overfit")

    print("EZv2 continuous unroll overfit (CPU): OK")
