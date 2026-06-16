"""MuZero K-step unroll GPU↔CPU parity (Apple/CUDA) — the #24 GPU branch gate.

Builds a CPU and a GPU MuZero model (rep/dyn/pred) with **identical** initial
params (CPU init → serialized → uploaded to the GPU buffers), runs one
`mz_unroll_train_step_*` on each over the SAME host batch, and checks that:

  * the returned mean loss matches within fp32 noise, and
  * every post-step Param matches within fp32 noise (the device forward-scan /
    reverse-scan + the ½ dynamics-gradient + the two-hot/soft-CE kernels all
    reproduce the validated CPU unroll).

This is the lighthouse for Phase-B #24: the CPU unroll is the oracle, the GPU
unroll mirrors it. Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_unroll_gpu_parity.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.checkpoint import (
    save_state_v2_body,
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
)
from mojo_rl.deep_agents.core.checkpoint_helpers import split_lines_v2
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.blocks import (
    mz_unroll_train_step_cpu,
    mz_unroll_train_step_gpu,
    MZScratch,
)


comptime OBS = 4
comptime ACT = 3
comptime LATENT = 8
comptime BINS = 11
comptime H = 16
comptime B = 5
comptime K = 4
comptime ATOL: Scalar[DT] = 2e-4

comptime Rep = MZRepNet[OBS, LATENT, H]
comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
comptime Pred = MZPredNet[LATENT, ACT, BINS, H]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _abs(v: Scalar[DT]) -> Scalar[DT]:
    return v if v >= 0 else -v


def _sync_cpu_to_gpu[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises:
    """Serialize CPU params and upload into the GPU model's device buffers."""
    var body = String("")
    save_state_v2_body(cpu, body, String(""))
    var lines = split_lines_v2(body)
    var idx = 0
    load_state_v2_body_gpu(gpu, lines, idx, String(""), ctx)


def _param_maxdiff[M: Module](
    mut cpu: M, mut gpu: M, ctx: DeviceContext
) raises -> Scalar[DT]:
    """Max |CPU − GPU| over every serialized Param value line."""
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
    print("test_mz_unroll_gpu_parity ...")
    seed(7)
    var ctx = DeviceContext()

    # ── identical-init CPU + GPU models ──
    var crep = Rep.make["cpu", INIT=Kaiming]()
    var cdyn = Dyn.make["cpu", INIT=Kaiming]()
    var cpred = Pred.make["cpu", INIT=Kaiming]()
    var grep = Rep.make["gpu", INIT=Kaiming](ctx)
    var gdyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var gpred = Pred.make["gpu", INIT=Kaiming](ctx)
    _sync_cpu_to_gpu(crep, grep, ctx)
    _sync_cpu_to_gpu(cdyn, gdyn, ctx)
    _sync_cpu_to_gpu(cpred, gpred, ctx)

    # confirm starting params are bit-identical (sanity: sync worked)
    var pre = _param_maxdiff(crep, grep, ctx)
    pre = max(pre, _param_maxdiff(cdyn, gdyn, ctx))
    pre = max(pre, _param_maxdiff(cpred, gpred, ctx))
    print("  pre-step max|param diff| =", pre)
    assert_true(pre < Scalar[DT](1e-6), "CPU→GPU param sync failed")

    # ── optimizers (same lr both devices) ──
    var corep = Adam.make["cpu", M=Rep](crep)
    var codyn = Adam.make["cpu", M=Dyn](cdyn)
    var copred = Adam.make["cpu", M=Pred](cpred)
    var gorep = Adam.make["gpu", M=Rep](grep, ctx)
    var godyn = Adam.make["gpu", M=Dyn](gdyn, ctx)
    var gopred = Adam.make["gpu", M=Pred](gpred, ctx)
    corep.lr = Scalar[DT](3e-4); codyn.lr = Scalar[DT](3e-4)
    copred.lr = Scalar[DT](3e-4)
    gorep.lr = Scalar[DT](3e-4); godyn.lr = Scalar[DT](3e-4)
    gopred.lr = Scalar[DT](3e-4)

    # ── deterministic host batch (time-major, same as replay produces) ──
    var obs0 = _a(B * OBS)
    var actions = _a(K * B)
    var policy_tgt = _a((K + 1) * B * ACT)
    var value_tgt = _a((K + 1) * B)
    var reward_tgt = _a(K * B)
    for i in range(B * OBS):
        obs0[i] = Scalar[DT](-0.4 + 0.13 * Float64(i % 7))
    for i in range(K * B):
        actions[i] = Scalar[DT](Float64(i % ACT))
        reward_tgt[i] = Scalar[DT](-0.5 + 0.2 * Float64(i % 5))
    # policy targets: random positive then row-normalized over ACT
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

    # ── one step on each device, same batch ──
    var lc = mz_unroll_train_step_cpu[
        Rep, Dyn, Pred, B, K, OBS, ACT, LATENT, BINS
    ](
        crep, cdyn, cpred, corep, codyn, copred,
        obs0, actions, policy_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF,
    )
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)
    var lg = mz_unroll_train_step_gpu[
        Rep, Dyn, Pred, B, K, OBS, ACT, LATENT, BINS
    ](
        ctx, grep, gdyn, gpred, gorep, godyn, gopred,
        scratch,
        obs0, actions, policy_tgt, value_tgt, reward_tgt,
        VMIN, VMAX, VCOEF,
    )

    print("  loss  cpu =", lc, "  gpu =", lg, "  |diff| =", _abs(lc - lg))
    assert_true(_abs(lc - lg) < ATOL, "unroll loss parity failed")

    # ── post-step param parity ──
    var dr = _param_maxdiff(crep, grep, ctx)
    var dd = _param_maxdiff(cdyn, gdyn, ctx)
    var dp = _param_maxdiff(cpred, gpred, ctx)
    print("  post-step max|param diff|  rep =", dr, " dyn =", dd, " pred =", dp)
    assert_true(dr < ATOL, "rep param parity failed")
    assert_true(dd < ATOL, "dyn param parity failed")
    assert_true(dp < ATOL, "pred param parity failed")

    obs0.free(); actions.free(); policy_tgt.free()
    value_tgt.free(); reward_tgt.free()
    print("  ok — MuZero GPU unroll matches CPU within", ATOL)
