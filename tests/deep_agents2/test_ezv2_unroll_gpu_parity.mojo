"""EZv2 K-step unroll GPU↔CPU parity (Apple/CUDA) — the Phase-C GPU gate.

The EZv2 analogue of `test_mz_unroll_gpu_parity.mojo`: build a CPU and a GPU EZv2
model (rep/dyn/pred + the SimSiam projector/predictor) with **identical** initial
params (CPU init → serialized → uploaded to the GPU buffers), run one
`ezv2_unroll_train_step_*` on each over the SAME host batch (full obs sequence
`[K+1, B, OBS]`), and check that the returned loss and every post-step Param match
within fp32 noise. This validates the device mirror of the consistency branch —
the target pre-pass `g_proj(h(obs_k))`, the online `h_pred(g_proj(z_k))` vjp, the
`consistency_loss_grad_k` kernel, and the gradient fold into the z_k carry — on
top of the already-parity-checked MuZero BPTT.

The projector/predictor carry BatchNorm1D; its GPU forward computes batch stats on
device, so a slightly looser tolerance than the BN-free MuZero unroll is used.

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_ezv2_unroll_gpu_parity.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
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
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.blocks import (
    ezv2_unroll_train_step_cpu,
    ezv2_unroll_train_step_gpu,
)
from mojo_rl.deep_agents2.efficient_zero_v2.unroll_scratch import (
    EZV2UnrollScratch,
)


comptime OBS = 4
comptime ACT = 3
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
comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
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
    print("test_ezv2_unroll_gpu_parity ...")
    seed(7)
    var ctx = DeviceContext()

    # ── identical-init CPU + GPU models (5 nets) ──
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

    # ── optimizers (same lr both devices) ──
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

    # ── deterministic host batch (time-major) — full obs sequence ──
    var obs_seq = _a((K + 1) * B * OBS)
    var actions = _a(K * B)
    var policy_tgt = _a((K + 1) * B * ACT)
    var value_tgt = _a((K + 1) * B)
    var reward_tgt = _a(K * B)
    for i in range((K + 1) * B * OBS):
        obs_seq[i] = Scalar[DT](-0.4 + 0.13 * Float64(i % 7))
    for i in range(K * B):
        actions[i] = Scalar[DT](Float64(i % ACT))
        reward_tgt[i] = Scalar[DT](-0.5 + 0.2 * Float64(i % 5))
    for p in range((K + 1) * B):
        var s = Scalar[DT](0.0)
        for a in range(ACT):
            var v = Scalar[DT](0.1 + random_float64())
            policy_tgt[p * ACT + a] = v
            s += v
        for a in range(ACT):
            policy_tgt[p * ACT + a] = policy_tgt[p * ACT + a] / s
    for i in range((K + 1) * B):
        value_tgt[i] = Scalar[DT](-0.3 + 0.17 * Float64(i % 6))

    comptime VMIN = Scalar[DT](-1.0)
    comptime VMAX = Scalar[DT](1.0)
    comptime VCOEF = Scalar[DT](0.25)
    comptime CCOEF = Scalar[DT](2.0)

    # ── one step on each device, same batch ──
    var lc = ezv2_unroll_train_step_cpu[
        Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT, LATENT, BINS
    ](
        crep, cdyn, cpred, cproj, cpredh,
        corep, codyn, copred, coproj, copredh,
        obs_seq, actions, policy_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF, CCOEF,
    )
    var gscratch = EZV2UnrollScratch[
        B, K, OBS, ACT, LATENT, BINS, PROJ
    ].make(ctx)
    var lg = ezv2_unroll_train_step_gpu[
        Rep, Dyn, Pred, Proj, Predh, B, K, OBS, ACT, LATENT, BINS
    ](
        ctx, gscratch, grep, gdyn, gpred, gproj, gpredh,
        gorep, godyn, gopred, goproj, gopredh,
        obs_seq, actions, policy_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF, CCOEF,
    )

    print("  loss  cpu =", lc, "  gpu =", lg, "  |diff| =", _abs(lc - lg))
    assert_true(_abs(lc - lg) < ATOL, "unroll loss parity failed")

    # ── post-step param parity ──
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

    obs_seq.free(); actions.free(); policy_tgt.free()
    value_tgt.free(); reward_tgt.free()
    print("  ok — EZv2 GPU unroll matches CPU within", ATOL)
