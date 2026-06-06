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
        tests/deep_agents2/test_ezv2_unroll_continuous_gpu_parity.mojo
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body,
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
)
from mojo_rl.deep_agents2.core.checkpoint_helpers import split_lines_v2
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents2.efficient_zero_v2.blocks_continuous import (
    ezv2_unroll_train_step_continuous_cpu,
    ezv2_unroll_train_step_continuous_gpu,
)
from mojo_rl.deep_agents2.efficient_zero_v2.unroll_scratch import (
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


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _abs(v: Scalar[DT]) -> Scalar[DT]:
    return v if v >= 0 else -v


def _sync_cpu_to_gpu[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises:
    var body = String("")
    save_state_v2_body(cpu, body, String(""))
    var lines = split_lines_v2(body)
    var idx = 0
    load_state_v2_body_gpu(gpu, lines, idx, String(""), ctx)


def _param_maxdiff[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises -> Scalar[DT]:
    var cbody = String("")
    save_state_v2_body(cpu, cbody, String(""))
    var gbody = String("")
    save_state_v2_body_gpu(gpu, gbody, String(""), ctx)
    var lc = split_lines_v2(cbody)
    var lg = split_lines_v2(gbody)
    assert_true(len(lc) == len(lg), "serialized param line count mismatch")
    var md = Scalar[DT](0.0)
    for i in range(len(lc)):
        if lc[i].find(String("#size=")) >= 0:
            assert_true(lc[i] == lg[i], "param section header mismatch")
            continue
        var d = _abs(Scalar[DT](atof(lc[i])) - Scalar[DT](atof(lg[i])))
        if d > md:
            md = d
    return md


def main() raises:
    print("test_ezv2_unroll_continuous_gpu_parity ...")
    seed(11)
    var ctx = DeviceContext()

    var crep = Rep.make["cpu", INIT=Kaiming]()
    var cdyn = Dyn.make["cpu", INIT=Kaiming]()
    var cpred = Pred.make["cpu", INIT=Kaiming]()
    var cproj = Proj.make["cpu", INIT=Kaiming]()
    var cpredh = Predh.make["cpu", INIT=Kaiming]()
    var grep = Rep.make["gpu", INIT=Kaiming](ctx)
    var gdyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var gpred = Pred.make["gpu", INIT=Kaiming](ctx)
    var gproj = Proj.make["gpu", INIT=Kaiming](ctx)
    var gpredh = Predh.make["gpu", INIT=Kaiming](ctx)
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

    var corep = Adam.make["cpu", M=Rep](crep)
    var codyn = Adam.make["cpu", M=Dyn](cdyn)
    var copred = Adam.make["cpu", M=Pred](cpred)
    var coproj = Adam.make["cpu", M=Proj](cproj)
    var copredh = Adam.make["cpu", M=Predh](cpredh)
    var gorep = Adam.make["gpu", M=Rep](grep, ctx)
    var godyn = Adam.make["gpu", M=Dyn](gdyn, ctx)
    var gopred = Adam.make["gpu", M=Pred](gpred, ctx)
    var goproj = Adam.make["gpu", M=Proj](gproj, ctx)
    var gopredh = Adam.make["gpu", M=Predh](gpredh, ctx)
    corep.lr = Scalar[DT](3e-4); codyn.lr = Scalar[DT](3e-4)
    copred.lr = Scalar[DT](3e-4); coproj.lr = Scalar[DT](3e-4)
    copredh.lr = Scalar[DT](3e-4)
    gorep.lr = Scalar[DT](3e-4); godyn.lr = Scalar[DT](3e-4)
    gopred.lr = Scalar[DT](3e-4); goproj.lr = Scalar[DT](3e-4)
    gopredh.lr = Scalar[DT](3e-4)

    # ── deterministic host batch (time-major) ──
    var obs_seq = _a((K + 1) * B * OBS)
    var actions = _a(K * B * ACT_DIM)
    var policy_act_tgt = _a((K + 1) * B * ACT_DIM)
    var value_tgt = _a((K + 1) * B)
    var reward_tgt = _a(K * B)
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

    obs_seq.free(); actions.free(); policy_act_tgt.free()
    value_tgt.free(); reward_tgt.free()
    print("  ok — EZv2 continuous GPU unroll matches CPU within", ATOL)
