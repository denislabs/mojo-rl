"""EZv2-Atari value-prefix unroll BPTT — CPU end-to-end + overfit trend (Stage 3).

Exercises `ezv2_unroll_train_step_cpu_vp` on the real Atari nets (z'-only dynamics
+ stateful LSTM value-prefix reward head, 6 nets / 6 optimizers). The Atari nets
are fixed at full dims (LATENT=2304, OBS=110592), so this is a *trend* check on a
small fixed batch (few iters) rather than a 400-step overfit: the loss must be
finite throughout and decrease (BPTT carry + (h,c) recurrence + ½ dyn-scale all
wired). Per-component correctness is pinned by the net + target unit smokes.

Run:
    pixi run mojo run -I . tests/deep_agents/test_ezv2_atari_value_prefix_unroll_cpu.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZRepNetResNetAtari, EZDynZNetAtari, EZRewardLSTMAtari, EZPredNetAtari,
    EZ_C, EZ_LATENT, EZ_LSTM_HIDDEN, EZ_LSTM_HORIZON,
    ez_atari_init_zero_pred, ez_atari_init_zero_reward,
)
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.blocks import (
    ezv2_unroll_train_step_cpu_vp,
)
from mojo_rl.deep_agents.zero import value_prefix_from_rewards


def main() raises:
    comptime FRAMES = 4
    comptime IN_CH = FRAMES * 3        # 12
    comptime OBS = IN_CH * 96 * 96     # 110592
    comptime ACT = 18
    comptime BINS = 601
    comptime LATENT = EZ_LATENT        # 2304
    comptime PROJ = 1024
    comptime HID = EZ_LSTM_HIDDEN      # 512
    comptime HORIZON = EZ_LSTM_HORIZON # 5
    comptime B = 2
    comptime K = 2
    comptime ITERS = 6
    var v_min = Scalar[DT](-300.0)
    var v_max = Scalar[DT](300.0)

    comptime Rep = EZRepNetResNetAtari[IN_CH, EZ_C]
    comptime DynZ = EZDynZNetAtari[ACT]
    comptime Rew = EZRewardLSTMAtari[BINS]
    comptime Pred = EZPredNetAtari[ACT, BINS]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ]
    comptime Predh = EZPredictorNet[PROJ, 256]

    var rep = Rep.make["cpu", Kaiming]()
    var dynz = DynZ.make["cpu", Kaiming]()
    var rew = Rew.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var proj = Proj.make["cpu", Kaiming]()
    var predh = Predh.make["cpu", Kaiming]()
    # init_zero heads (neutral value/reward + uniform policy at init), train mode.
    ez_atari_init_zero_pred["cpu", ACT, BINS](pred)
    ez_atari_init_zero_reward["cpu", BINS](rew)
    rep.set_attr["training"](Scalar[DT](1.0))
    dynz.set_attr["training"](Scalar[DT](1.0))
    rew.set_attr["training"](Scalar[DT](1.0))
    pred.set_attr["training"](Scalar[DT](1.0))
    proj.set_attr["training"](Scalar[DT](1.0))

    var orep = Adam(lr=Scalar[DT](0.02))
    var odynz = Adam(lr=Scalar[DT](0.02))
    var orew = Adam(lr=Scalar[DT](0.02))
    var opred = Adam(lr=Scalar[DT](0.02))
    var oproj = Adam(lr=Scalar[DT](0.02))
    var opredh = Adam(lr=Scalar[DT](0.02))

    # ── one fixed batch (time-major) ──
    var xs = UInt64(0x9E3779B97F4A7C15)
    var obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    for i in range((K + 1) * B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs_seq[i] = Scalar[DT](Int(xs % 256)) / Scalar[DT](255.0)  # pixel in [0,1]

    var actions = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        actions[i] = Scalar[DT](Int(xs % ACT))

    var policy_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    for k in range(K + 1):
        for b in range(B):
            xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
            policy_tgt[k * B * ACT + b * ACT + Int(xs % ACT)] = Scalar[DT](1.0)

    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for i in range((K + 1) * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        value_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)

    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        reward_tgt[i] = Scalar[DT](Int(xs % 100)) / Scalar[DT](100.0)
    # cumulative value-prefix targets (reset every HORIZON)
    value_prefix_from_rewards[K, HORIZON](reward_tgt, B)

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(ITERS):
        var l = ezv2_unroll_train_step_cpu_vp[
            Rep, DynZ, Pred, Proj, Predh,
            B, K, OBS, ACT, LATENT, BINS, HID, HORIZON,
        ](
            rep, dynz, rew, pred, proj, predh,
            orep, odynz, orew, opred, oproj, opredh,
            obs_seq, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max,
            consistency_coef=Scalar[DT](2.0),
        )
        if it == 0:
            first = l
        last = l
        print("it", it, "loss", l)
        assert_true(l == l, "loss became NaN")

    print("first", first, "last", last)
    assert_true(last < first, "VP unroll failed to reduce loss on a fixed batch")

    _ = rep^; _ = dynz^; _ = rew^; _ = pred^; _ = proj^; _ = predh^
    print("EZv2-Atari value-prefix unroll BPTT (CPU): OK")
