"""EZv2 continuous K-step unroll GPU↔CPU parity (Apple/CUDA) — Phase-C2 GPU gate.

The continuous analogue of `test_ezv2_unroll_gpu_parity.mojo`: build a CPU and a
GPU continuous-EZv2 model (rep/dyn + the squashed-Gaussian prediction head
`EZContPredNet` + SimSiam projector/predictor) with **identical** initial params
(CPU init → serialized → uploaded to the GPU buffers), run one
`ezv2_unroll_train_step_continuous_*` on each over the SAME host batch (full obs
sequence `[K+1, B, OBS]`, action **vectors** `[K, B, ACT_DIM]`, per-position
target actions `[K+1, B, ACT_DIM]`), and check the returned loss + every
post-step Param match within fp32 noise. Validates the device mirror of the two
continuous-specific pieces — the action-vector dynamics input
(`_ez_build_dyn_in_cont_k`) and the squashed-Gaussian policy NLL
(`continuous_policy_loss_grad_k`) — on top of the parity-checked consistency
branch + MuZero BPTT.

Run (GPU env required):
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_unroll_continuous_gpu_parity.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.core.hard_copy import _CollectVisitor, _InjectVisitor
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents.efficient_zero_v2.blocks_continuous import (
    ezv2_unroll_train_step_continuous_cpu,
    ezv2_unroll_train_step_continuous_gpu,
)
from mojo_rl.deep_agents.efficient_zero_v2.unroll_scratch import (
    EZV2UnrollContScratch,
)


comptime OBS = 3       # Pendulum
comptime ACT_DIM = 1
comptime LATENT = 8
comptime BINS = 11
comptime H = 16
comptime PROJ = 16
comptime PROJ_HID = 16
comptime BOTTLENECK = 8
comptime B = 5
comptime K = 4
comptime ATOL: Scalar[DT] = 2e-3

comptime Rep = MZRepNet[OBS, LATENT, H]
comptime Dyn = MZDynNet[LATENT, ACT_DIM, BINS, H]
comptime Pred = EZContPredNet[LATENT, ACT_DIM, BINS, H]
comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]



def _abs(v: Scalar[DT]) -> Scalar[DT]:
    return v if v >= 0 else -v


def _sync_cpu_to_gpu[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises:
    var c = _CollectVisitor()
    cpu.for_each_param["cpu"](c, None)
    var inj = _InjectVisitor(c.names.copy(), c.vals.copy())
    gpu.for_each_param["gpu"](inj, Optional(ctx))


def _param_maxdiff[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises -> Scalar[DT]:
    var cc = _CollectVisitor()
    cpu.for_each_param["cpu"](cc, None)
    var gc = _CollectVisitor()
    gpu.for_each_param["gpu"](gc, Optional(ctx))
    assert_true(len(cc.vals) == len(gc.vals), "param section count mismatch")
    var md = Scalar[DT](0.0)
    for s in range(len(cc.vals)):
        for i in range(len(cc.vals[s])):
            var d = _abs(cc.vals[s][i] - gc.vals[s][i])
            if d > md:
                md = d
    return md


def main() raises:
    print("test_ezv2_unroll_continuous_gpu_parity ...")
    seed(11)
    var ctx = DeviceContext()

    var crep = Rep.make["cpu", Kaiming]()
    var cdyn = Dyn.make["cpu", Kaiming]()
    var cpred = Pred.make["cpu", Kaiming]()
    var cproj = Proj.make["cpu", Kaiming]()
    var cpredh = Predh.make["cpu", Kaiming]()
    var grep = Rep.make["gpu", Kaiming](Optional(ctx))
    var gdyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var gpred = Pred.make["gpu", Kaiming](Optional(ctx))
    var gproj = Proj.make["gpu", Kaiming](Optional(ctx))
    var gpredh = Predh.make["gpu", Kaiming](Optional(ctx))
    _sync_cpu_to_gpu(crep, grep, ctx)
    _sync_cpu_to_gpu(cdyn, gdyn, ctx)
    _sync_cpu_to_gpu(cpred, gpred, ctx)
    _sync_cpu_to_gpu(cproj, gproj, ctx)
    _sync_cpu_to_gpu(cpredh, gpredh, ctx)

    var pre = _param_maxdiff(crep, grep, ctx)
    pre = max(pre, _param_maxdiff(cdyn, gdyn, ctx))
    pre = max(pre, _param_maxdiff(cpred, gpred, ctx))
    pre = max(pre, _param_maxdiff(cproj, gproj, ctx))
    pre = max(pre, _param_maxdiff(cpredh, gpredh, ctx))
    print("  pre-step max|param diff| =", pre)
    assert_true(pre < Scalar[DT](1e-6), "CPU→GPU param sync failed")

    var corep = Adam(lr=Scalar[DT](1e-3))
    var codyn = Adam(lr=Scalar[DT](1e-3))
    var copred = Adam(lr=Scalar[DT](1e-3))
    var coproj = Adam(lr=Scalar[DT](1e-3))
    var copredh = Adam(lr=Scalar[DT](1e-3))
    var gorep = Adam(lr=Scalar[DT](1e-3))
    var godyn = Adam(lr=Scalar[DT](1e-3))
    var gopred = Adam(lr=Scalar[DT](1e-3))
    var goproj = Adam(lr=Scalar[DT](1e-3))
    var gopredh = Adam(lr=Scalar[DT](1e-3))
    corep.lr = Scalar[DT](3e-4); codyn.lr = Scalar[DT](3e-4)
    copred.lr = Scalar[DT](3e-4); coproj.lr = Scalar[DT](3e-4)
    copredh.lr = Scalar[DT](3e-4)
    gorep.lr = Scalar[DT](3e-4); godyn.lr = Scalar[DT](3e-4)
    gopred.lr = Scalar[DT](3e-4); goproj.lr = Scalar[DT](3e-4)
    gopredh.lr = Scalar[DT](3e-4)

    # ── deterministic host batch (time-major) ──
    var obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    var actions = List[Scalar[DT]](length=K * B * ACT_DIM, fill=0)
    var policy_act_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT_DIM, fill=0)
    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for i in range((K + 1) * B * OBS):
        obs_seq[i] = Scalar[DT](-0.4 + 0.13 * Float64(i % 7))
    # continuous actions in [-0.8, 0.8].
    for i in range(K * B * ACT_DIM):
        actions[i] = Scalar[DT](-0.8 + 0.21 * Float64(i % 8))
    for i in range((K + 1) * B * ACT_DIM):
        policy_act_tgt[i] = Scalar[DT](-0.7 + 0.19 * Float64(i % 9))
    for i in range(K * B):
        reward_tgt[i] = Scalar[DT](-0.5 + 0.2 * Float64(i % 5))
    for i in range((K + 1) * B):
        value_tgt[i] = Scalar[DT](-0.3 + 0.17 * Float64(i % 6))

    comptime VMIN = Scalar[DT](-1.0)
    comptime VMAX = Scalar[DT](1.0)
    comptime VCOEF = Scalar[DT](0.25)
    comptime CCOEF = Scalar[DT](2.0)

    var lc = ezv2_unroll_train_step_continuous_cpu[
        Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT_DIM, LATENT, BINS
    ](
        crep, cdyn, cpred, cproj, cpredh,
        corep, codyn, copred, coproj, copredh,
        obs_seq, actions, policy_act_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF, CCOEF,
    )
    var gscratch = EZV2UnrollContScratch[
        B, K, OBS, ACT_DIM, LATENT, BINS, PROJ
    ].make(ctx)
    var lg = ezv2_unroll_train_step_continuous_gpu[
        Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT_DIM, LATENT, BINS
    ](
        ctx, gscratch, grep, gdyn, gpred, gproj, gpredh,
        gorep, godyn, gopred, goproj, gopredh,
        obs_seq, actions, policy_act_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF, CCOEF,
    )

    print("  loss  cpu =", lc, "  gpu =", lg, "  |diff| =", _abs(lc - lg))
    assert_true(_abs(lc - lg) < ATOL, "continuous unroll loss parity failed")

    var dr = _param_maxdiff(crep, grep, ctx)
    var dd = _param_maxdiff(cdyn, gdyn, ctx)
    var dp = _param_maxdiff(cpred, gpred, ctx)
    var dj = _param_maxdiff(cproj, gproj, ctx)
    var dh = _param_maxdiff(cpredh, gpredh, ctx)
    print("  post-step max|param diff|  rep =", dr, " dyn =", dd,
          " pred =", dp, " proj =", dj, " predh =", dh)
    assert_true(dr < ATOL, "rep param parity failed")
    assert_true(dd < ATOL, "dyn param parity failed")
    assert_true(dp < ATOL, "pred param parity failed")
    assert_true(dj < ATOL, "proj param parity failed")
    assert_true(dh < ATOL, "predh param parity failed")

    print("  ok — EZv2 continuous GPU unroll matches CPU within", ATOL)
