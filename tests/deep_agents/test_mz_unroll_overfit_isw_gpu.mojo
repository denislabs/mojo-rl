"""Does activating the PER GPU path (kScalePred/kScaleRew) CORRUPT the gradient?

PER is the confirmed MuZero regression, but its IS-weights/priorities are healthy
and its code is byte-identical to the last working commit. Hypothesis (user): PER
logic is fine, but turning it on injects the kScalePred/kScaleRew kernels (+ the
d_isw buffer) into the reverse-scan, BETWEEN pred.forward and pred.vjp — and that
GPU path corrupts the following vjp matmuls (the ExternalRef-class exclusivity
miscompile), poisoning the GRADIENT (not the IS-weights).

Decisive A/B on the same machine, same fixed overfit batch:
  Phase A: is_weights=None              (PER path OFF) — known to overfit on NVIDIA
  Phase B: is_weights = ALL 1.0         (PER path ON, but x1.0 is a NO-OP in math)
           + out_prio active (kPrioCE)

If the code is sound, B must equal A (multiplying grads by 1.0 changes nothing).
  B overfits like A (policy->0)  -> PER GPU path is CLEAN; harm is logic/tuning
  B breaks (policy stuck)        -> the PER kernels CORRUPT the gradient with no-op
                                    weights => data-poisoning / exclusivity bug CONFIRMED
apple should pass both. Run on NVIDIA to compare.
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
)
from mojo_rl.deep_agents.muzero.blocks import (
    mz_unroll_train_step_gpu, MZScratch,
)


def main() raises:
    comptime CH = 8
    comptime ACT = 7
    comptime BINS = 51
    comptime HH = 6
    comptime WW = 7
    comptime NB = 2
    comptime OBS = 3 * HH * WW       # 126
    comptime LATENT = CH * HH * WW   # 336
    comptime K = 5
    comptime B = 16
    comptime Rep = MZRepNetC4Spatial[CH, HH, WW, NB]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW, NB]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW, NB]

    var ctx = DeviceContext()

    # ── fixed, consistent synthetic batch (overfit target), shared by A and B ──
    var obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    for b in range(B):
        for j in range(OBS):
            obs0[b * OBS + j] = Scalar[DT](0.1) * Scalar[DT](((b * 13 + j) % 7) - 3)
    var actions = List[Scalar[DT]](length=K * B, fill=0)
    for k in range(K):
        for b in range(B):
            actions[k * B + b] = Scalar[DT]((b + k) % ACT)
    var policy_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    for k in range(K + 1):
        for b in range(B):
            policy_tgt[(k * B + b) * ACT + (b % ACT)] = Scalar[DT](1.0)
    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for k in range(K + 1):
        for b in range(B):
            value_tgt[k * B + b] = Scalar[DT](0.5) if (b % 2 == 0) else Scalar[DT](-0.5)
    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for k in range(K):
        for b in range(B):
            reward_tgt[k * B + b] = Scalar[DT](0.5) if (b % 3 == 0) else Scalar[DT](0.0)

    # PER buffers for phase B: uniform IS-weights = 1.0 (math no-op) + priority sink
    var isw = List[Scalar[DT]](length=B, fill=Scalar[DT](1.0))
    var prio = List[Scalar[DT]](length=B, fill=0)

    for phase in range(2):
        var per_on = phase == 1
        if per_on:
            print("\n=== Phase B: PER path ON (is_weights = ALL 1.0, NO-OP math) ===")
        else:
            print("=== Phase A: PER path OFF (is_weights=None) — baseline ===")

        var rep = Rep.make["gpu", INIT=Kaiming](Optional(ctx))
        var dyn = Dyn.make["gpu", INIT=Kaiming](Optional(ctx))
        var pred = Pred.make["gpu", INIT=Kaiming](Optional(ctx))
        var orep = Adam(lr=Scalar[DT](2e-3))
        var odyn = Adam(lr=Scalar[DT](2e-3))
        var opred = Adam(lr=Scalar[DT](2e-3))
        var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

        var lp = List[Scalar[DT]](length=3, fill=0)
        var lp_opt = Optional[Pointer[Scalar[DT], MutAnyOrigin]](
            lp.unsafe_ptr().as_unsafe_any_origin()
        )
        var isw_opt = Optional[Pointer[Scalar[DT], MutAnyOrigin]](None)
        var prio_opt = Optional[Pointer[Scalar[DT], MutAnyOrigin]](None)
        if per_on:
            isw_opt = Optional(isw.unsafe_ptr().as_unsafe_any_origin())
            prio_opt = Optional(prio.unsafe_ptr().as_unsafe_any_origin())

        print("step | loss_policy | loss_value | loss_reward")
        for step in range(401):
            _ = mz_unroll_train_step_gpu[
                Rep, Dyn, Pred, B, K, OBS, ACT, LATENT, BINS, obs_on_device=False,
            ](
                ctx, rep, dyn, pred, orep, odyn, opred, scratch,
                obs0, actions, policy_tgt, value_tgt, reward_tgt,
                Scalar[DT](-1.0), Scalar[DT](1.0),
                value_coef=Scalar[DT](0.25), max_grad_norm=1.0,
                loss_parts=lp_opt,
                is_weights=isw_opt,
                out_prio=prio_opt,
            )
            ctx.synchronize()
            if step % 100 == 0:
                print(step, "|", lp[0], "|", lp[1], "|", lp[2])
        print("FINAL loss_policy =", lp[0], " loss_value =", lp[1],
              " loss_reward =", lp[2])

    print(
        "\n>>> If Phase B's losses DON'T match Phase A (policy stuck high),"
        " the PER GPU path corrupts the gradient with no-op weights = the bug. <<<"
    )
