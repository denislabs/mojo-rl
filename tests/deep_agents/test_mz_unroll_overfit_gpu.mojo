"""Does the FULL K-step unroll (rep -> K*dyn -> pred, BPTT) learn on GPU?

Every isolated piece is correct on NVIDIA and the live reward data is good, yet
loss_reward is stuck. The only untested-as-an-assembly path is the multi-step
BPTT unroll. This drives the REAL mz_unroll_train_step_gpu on B consistent
synthetic examples (fixed obs -> fixed policy/value/reward targets) and overfits.

  loss_reward (and policy/value) -> ~0  → the unroll learns; regression is in the
                                          live training dynamics (sims/bootstrap)
  loss_reward stuck high                 → the GPU BPTT unroll is broken = the bug

apple GPU should overfit. Run on NVIDIA to compare.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
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
    var rep = Rep.make["gpu", INIT=Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", INIT=Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", INIT=Kaiming](Optional(ctx))
    var orep = Adam(lr=Scalar[DT](2e-3))
    var odyn = Adam(lr=Scalar[DT](2e-3))
    var opred = Adam(lr=Scalar[DT](2e-3))
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # ── fixed, consistent synthetic batch (overfit target) ──
    var obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    for b in range(B):
        for j in range(OBS):
            obs0[b * OBS + j] = Scalar[DT](0.1) * Scalar[DT](((b * 13 + j) % 7) - 3)
    var actions = List[Scalar[DT]](length=K * B, fill=0)
    for k in range(K):
        for b in range(B):
            actions[k * B + b] = Scalar[DT]((b + k) % ACT)
    # peaked policy targets (one-hot on action b%ACT), per (k,b)
    var policy_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    for k in range(K + 1):
        for b in range(B):
            policy_tgt[(k * B + b) * ACT + (b % ACT)] = Scalar[DT](1.0)
    # consistent value + reward targets per sample
    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for k in range(K + 1):
        for b in range(B):
            value_tgt[k * B + b] = Scalar[DT](0.5) if (b % 2 == 0) else Scalar[DT](-0.5)
    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for k in range(K):
        for b in range(B):
            reward_tgt[k * B + b] = Scalar[DT](0.5) if (b % 3 == 0) else Scalar[DT](0.0)

    var lp = List[Scalar[DT]](length=3, fill=0)
    var lp_opt = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        lp.unsafe_ptr().as_unsafe_any_origin()
    )

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
        )
        ctx.synchronize()
        if step % 50 == 0:
            print(step, "|", lp[0], "|", lp[1], "|", lp[2])

    print("FINAL loss_reward =", lp[2], "(want -> ~0 if the unroll learns)")
