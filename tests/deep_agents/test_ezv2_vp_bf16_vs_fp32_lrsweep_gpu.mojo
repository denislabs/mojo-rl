"""EZv2 VP unroll — fp32 vs bf16 convergence at the REALISTIC lr (diagnostic).

The 6-iter overfit (`test_ezv2_atari_value_prefix_unroll_gpu`) uses an aggressive
lr 0.02 to converge fast; at that step size even fp32 bounces, and bf16's coarser
gradients stall (~40→28, unstable). Real EZv2 training uses lr ~1e-3 (ResNet-20
bf16 matched fp32 at 1e-3). This diagnostic trains BOTH fp32 and bf16 nets on the
SAME fixed batch at lr 1e-3 for 30 iters and prints both curves, so we can tell
whether the bf16 gap is lr/step-size (then bf16 tracks fp32 here) or a real bf16
precision floor (bf16 plateaus above fp32).

⚠️ NVIDIA only (Apple Metal bf16 GEMM is broken). Compare the two final losses.

Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/deep_agents/test_ezv2_vp_bf16_vs_fp32_lrsweep_gpu.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.module import Module
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
    ezv2_unroll_train_step_gpu_vp,
)
from mojo_rl.deep_agents.zero import value_prefix_from_rewards

comptime BF16 = DType.bfloat16
comptime FRAMES = 4
comptime IN_CH = FRAMES * 3
comptime OBS = IN_CH * 96 * 96
comptime ACT = 18
comptime BINS = 601
comptime LATENT = EZ_LATENT
comptime PROJ = 1024
comptime HID = EZ_LSTM_HIDDEN
comptime HORIZON = EZ_LSTM_HORIZON
comptime B = 4
comptime K = 2
comptime ITERS = 30


def _run[
    Rep: Module, DynZ: Module, Pred: Module, Proj: Module, Predh: Module
](
    ctx: DeviceContext,
    mut rep: Rep, mut dynz: DynZ, mut rew: EZRewardLSTMAtari[BINS],
    mut pred: Pred, mut proj: Proj, mut predh: Predh,
    obs_seq: List[Scalar[DT]], actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]], value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]], v_min: Scalar[DT], v_max: Scalar[DT],
    label: StaticString,
) raises:
    var orep = Adam(lr=Scalar[DT](0.001))
    var odynz = Adam(lr=Scalar[DT](0.001))
    var orew = Adam(lr=Scalar[DT](0.001))
    var opred = Adam(lr=Scalar[DT](0.001))
    var oproj = Adam(lr=Scalar[DT](0.001))
    var opredh = Adam(lr=Scalar[DT](0.001))
    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(ITERS):
        var l = ezv2_unroll_train_step_gpu_vp[
            Rep, DynZ, Pred, Proj, Predh,
            B, K, OBS, ACT, LATENT, BINS, HID, HORIZON,
        ](
            ctx, rep, dynz, rew, pred, proj, predh,
            orep, odynz, orew, opred, oproj, opredh,
            obs_seq, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max,
            consistency_coef=Scalar[DT](2.0),
        )
        if it == 0:
            first = l
        last = l
        if it % 5 == 0 or it == ITERS - 1:
            print(label, "it", it, "loss", l)
    print(label, "FINAL  first", first, "last", last)


def main() raises:
    var v_min = Scalar[DT](-300.0)
    var v_max = Scalar[DT](300.0)
    var ctx = DeviceContext()

    # SAME fixed-seed batch for both runs.
    var xs = UInt64(0x9E3779B97F4A7C15)
    var obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    for i in range((K + 1) * B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs_seq[i] = Scalar[DT](Int(xs % 256)) / Scalar[DT](255.0)
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
    value_prefix_from_rewards[K, HORIZON](reward_tgt, B)

    print("=== fp32 (lr 1e-3, ", ITERS, " iters) ===")
    comptime RepF = EZRepNetResNetAtari[IN_CH, EZ_C]
    comptime DynZF = EZDynZNetAtari[ACT]
    comptime PredF = EZPredNetAtari[ACT, BINS]
    comptime ProjF = EZProjectorNet[LATENT, PROJ, PROJ]
    comptime PredhF = EZPredictorNet[PROJ, 256]
    var repf = RepF.make["gpu", Kaiming](ctx)
    var dynzf = DynZF.make["gpu", Kaiming](ctx)
    var rewf = EZRewardLSTMAtari[BINS].make["gpu", Kaiming](ctx)
    var predf = PredF.make["gpu", Kaiming](ctx)
    var projf = ProjF.make["gpu", Kaiming](ctx)
    var predhf = PredhF.make["gpu", Kaiming](ctx)
    ez_atari_init_zero_pred["gpu", ACT, BINS](predf, ctx)
    ez_atari_init_zero_reward["gpu", BINS](rewf, ctx)
    repf.set_attr["training"](Scalar[DT](1.0)); dynzf.set_attr["training"](Scalar[DT](1.0))
    rewf.set_attr["training"](Scalar[DT](1.0)); predf.set_attr["training"](Scalar[DT](1.0))
    projf.set_attr["training"](Scalar[DT](1.0))
    _run[RepF, DynZF, PredF, ProjF, PredhF](
        ctx, repf, dynzf, rewf, predf, projf, predhf,
        obs_seq, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max, "fp32")
    _ = repf^; _ = dynzf^; _ = rewf^; _ = predf^; _ = projf^; _ = predhf^

    print("=== bf16 (lr 1e-3, ", ITERS, " iters) ===")
    comptime RepB = EZRepNetResNetAtari[IN_CH, EZ_C, ADT=BF16]
    comptime DynZB = EZDynZNetAtari[ACT, ADT=BF16]
    comptime PredB = EZPredNetAtari[ACT, BINS, ADT=BF16]
    comptime ProjB = EZProjectorNet[LATENT, PROJ, PROJ, ADT=BF16]
    comptime PredhB = EZPredictorNet[PROJ, 256, ADT=BF16]
    var repb = RepB.make["gpu", Kaiming](ctx)
    var dynzb = DynZB.make["gpu", Kaiming](ctx)
    var rewb = EZRewardLSTMAtari[BINS].make["gpu", Kaiming](ctx)
    var predb = PredB.make["gpu", Kaiming](ctx)
    var projb = ProjB.make["gpu", Kaiming](ctx)
    var predhb = PredhB.make["gpu", Kaiming](ctx)
    ez_atari_init_zero_pred["gpu", ACT, BINS, BF16](predb, ctx)
    ez_atari_init_zero_reward["gpu", BINS](rewb, ctx)
    repb.set_attr["training"](Scalar[DT](1.0)); dynzb.set_attr["training"](Scalar[DT](1.0))
    rewb.set_attr["training"](Scalar[DT](1.0)); predb.set_attr["training"](Scalar[DT](1.0))
    projb.set_attr["training"](Scalar[DT](1.0))
    _run[RepB, DynZB, PredB, ProjB, PredhB](
        ctx, repb, dynzb, rewb, predb, projb, predhb,
        obs_seq, actions, policy_tgt, value_tgt, reward_tgt, v_min, v_max, "bf16")
    _ = repb^; _ = dynzb^; _ = rewb^; _ = predb^; _ = projb^; _ = predhb^

    print("DONE — compare fp32 vs bf16 FINAL last loss at lr 1e-3")
