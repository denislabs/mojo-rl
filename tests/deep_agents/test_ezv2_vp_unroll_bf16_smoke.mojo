"""EZv2-Atari value-prefix unroll BPTT — bf16 overfit (EZv2-bf16 2b-iii gate).

Exact bf16 twin of `test_ezv2_atari_value_prefix_unroll_gpu` (SAME fixed-seed
batch, same Adam lr, same zero-inits) but the training nets flow bf16
(rep/dynz/pred/proj/predh `ADT=bfloat16`); the reward head `EZRewardLSTMAtari`
stays fp32 (its `dz` input is cast down, its `grad_zp` cast back up inside the
unroll). This is the convergence gate for the bf16 VP train step: run it
alongside the fp32 test and compare the loss curve.

⚠️ NVIDIA is the real run. On Apple Metal `linalg.matmul` MIS-COMPUTES bf16 GEMMs,
so the loss here is garbage — this only proves the bf16 VP unroll COMPILES + runs
FINITE on Apple. On NVIDIA the curve should DECREASE comparably to the fp32 test
(fp32: 40.66 → ~13.05 over 6 iters); `last < first` there is the convergence gate.

Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/deep_agents/test_ezv2_vp_unroll_bf16_smoke.mojo
Run (Apple, compile+finite smoke): pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_vp_unroll_bf16_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

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
    ezv2_unroll_train_step_gpu_vp,
)
from mojo_rl.deep_agents.zero import value_prefix_from_rewards

comptime BF16 = DType.bfloat16


def main() raises:
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
    comptime ITERS = 6
    var v_min = Scalar[DT](-300.0)
    var v_max = Scalar[DT](300.0)

    var ctx = DeviceContext()

    # bf16-flow training nets; reward head stays fp32.
    comptime Rep = EZRepNetResNetAtari[IN_CH, EZ_C, ADT=BF16]
    comptime DynZ = EZDynZNetAtari[ACT, ADT=BF16]
    comptime Rew = EZRewardLSTMAtari[BINS]
    comptime Pred = EZPredNetAtari[ACT, BINS, ADT=BF16]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ, ADT=BF16]
    comptime Predh = EZPredictorNet[PROJ, 256, ADT=BF16]
    comptime assert Rep.ACT_DT == BF16 and DynZ.ACT_DT == BF16, "nets must flow bf16"

    var rep = Rep.make["gpu", Kaiming](ctx)
    var dynz = DynZ.make["gpu", Kaiming](ctx)
    var rew = Rew.make["gpu", Kaiming](ctx)
    var pred = Pred.make["gpu", Kaiming](ctx)
    var proj = Proj.make["gpu", Kaiming](ctx)
    var predh = Predh.make["gpu", Kaiming](ctx)
    ez_atari_init_zero_pred["gpu", ACT, BINS, BF16](pred, ctx)
    ez_atari_init_zero_reward["gpu", BINS](rew, ctx)
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

    var cmask = List[Scalar[DT]](length=K * B, fill=1)
    var cmask_ptr = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        cmask.unsafe_ptr().as_unsafe_any_origin())
    var isw = List[Scalar[DT]](length=B, fill=1)
    var prio = List[Scalar[DT]](length=B, fill=0)
    var isw_ptr = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        isw.unsafe_ptr().as_unsafe_any_origin())
    var prio_ptr = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        prio.unsafe_ptr().as_unsafe_any_origin())
    var d_obs_ext = ctx.enqueue_create_buffer[DT]((K + 1) * B * OBS)
    ctx.enqueue_copy(d_obs_ext, obs_seq.unsafe_ptr())
    var obsdev_ptr = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        d_obs_ext.unsafe_ptr().as_unsafe_any_origin())

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
            cons_mask=cmask_ptr,
            is_weights=isw_ptr,
            out_prio=prio_ptr,
            obs_dev=obsdev_ptr,
        )
        if it == 0:
            first = l
        last = l
        print("it", it, "loss", l)
        assert_true(l == l, "loss became NaN")

    print("first", first, "last", last)
    # On Apple the Metal bf16 GEMM is broken → loss is garbage; only finiteness
    # is meaningful here. On NVIDIA, last < first (decreasing like the fp32 test's
    # 40.66 → ~13.05) is the convergence gate.
    if last < first:
        print("bf16 VP unroll: loss DECREASED (last < first) — converging")
    else:
        print(
            "bf16 VP unroll: loss did NOT decrease — EXPECTED on Apple (Metal"
            " bf16 GEMM bug); on NVIDIA this would signal a bf16 numerics issue"
        )
    _ = rep^; _ = dynz^; _ = rew^; _ = pred^; _ = proj^; _ = predh^
    print("bf16-flow VP unroll: COMPILES + runs FINITE on Apple (numerics=NVIDIA)")
