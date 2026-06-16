"""R.5 GPU smoke for `EnsembleActorLoss`.

Same invariants as the R.2 CPU smoke (loss finite, actor moves,
critic params unchanged via input_only, target nets unchanged, loss
decreases over K steps) but on GPU. Apple Metal — gates the kernel
+ glue wiring; numeric convergence on real hardware is a separate
NVIDIA-gated step.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.named_params import named_params
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq import (
    CriticEnsemble,
    EnsembleActorLoss,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 8
comptime N = 4

comptime ActorNet = Sequential[
    Linear[OBS, 16], ReLU[16], Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16], ReLU[16], Linear[16, 1],
]


def _fill_mb_s_gpu(
    ctx: DeviceContext,
    mut state: TrainerState[OBS, ACT, BATCH],
) raises:
    var host_buf = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    ctx.synchronize()
    var hp = host_buf.unsafe_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            hp[b * OBS + d] = Scalar[DT](
                0.05 * Float64(b) + 0.07 * Float64(d) - 0.2
            )
    ctx.enqueue_copy(state.mb_s.dev.value(), hp)


def _gpu_param_sum[M: Module](
    ctx: DeviceContext, mut model: M,
) raises -> Float64:
    """D2H every leaf's param buffer and sum |value| on host. Coarse
    "did this leaf change" signature, robust to FP noise. Wraps each
    `param_ptr` in a non-owning DeviceBuffer view (named_params returns
    raw ptrs, but `enqueue_copy` needs a buffer)."""
    var ps = named_params["gpu", M](model)
    var acc: Float64 = 0.0
    for i in range(len(ps)):
        ref p = ps[i]
        var host = ctx.enqueue_create_host_buffer[DT](p.n_elems)
        ctx.synchronize()
        var src_view = DeviceBuffer[DT](
            ctx, p.param_ptr, p.n_elems, owning=False,
        )
        ctx.enqueue_copy(host, src_view)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        for k in range(p.n_elems):
            var v = Float64(hp[k])
            if v < 0.0:
                v = -v
            acc += v
    return acc


def test_ensemble_actor_loss_gpu() raises:
    print("--- EnsembleActorLoss GPU N=4 smoke ---")
    var ctx = DeviceContext()

    var actor = ActorNet.make["gpu", Xavier](ctx=ctx)
    var actor_opt = Adam.make["gpu", M=ActorNet](actor, ctx=ctx)
    actor_opt.lr = Scalar[DT](1e-3)
    var ensemble = CriticEnsemble[CriticNet, N].make["gpu", Xavier](
        ctx=ctx,
    )
    var block = EnsembleActorLoss[
        ActorNet, CriticNet, N, BATCH, OBS, ACT,
    ].make["gpu"](action_scale=Scalar[DT](1.0), ctx=ctx)
    var state = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx=ctx)

    _fill_mb_s_gpu(ctx, state)
    var alpha = Scalar[DT](0.2)

    var actor_before = _gpu_param_sum[ActorNet](ctx, actor)
    var online_before = List[Float64](length=N, fill=0.0)
    var target_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _gpu_param_sum[CriticNet](
            ctx, ensemble.pairs[i].online
        )
        target_before[i] = _gpu_param_sum[CriticNet](
            ctx, ensemble.pairs[i].target_net
        )

    var res0 = block.forward_backward["gpu"](
        actor, actor_opt, ensemble, state.mb_s.dev_ptr(), alpha,
    )
    var loss_first = Float64(res0.loss)
    var lp_first = Float64(res0.log_prob_mean)
    print("  step 0 loss =", loss_first, " log_prob_mean =", lp_first)
    assert_true(loss_first == loss_first, "step 0 loss finite")
    assert_true(lp_first == lp_first, "step 0 log_prob_mean finite")

    comptime K = 10
    var loss_last: Float64 = loss_first
    for _ in range(K - 1):
        var res = block.forward_backward["gpu"](
            actor, actor_opt, ensemble, state.mb_s.dev_ptr(), alpha,
        )
        loss_last = Float64(res.loss)
    print("  step", K - 1, "loss =", loss_last)
    assert_true(loss_last == loss_last, "final loss finite")

    var actor_after = _gpu_param_sum[ActorNet](ctx, actor)
    var actor_d = actor_after - actor_before
    if actor_d < 0.0:
        actor_d = -actor_d
    print("  actor |Δ|sum =", actor_d)
    assert_true(actor_d > 0.0, "actor params must change")

    for i in range(N):
        var on_after = _gpu_param_sum[CriticNet](
            ctx, ensemble.pairs[i].online
        )
        var tg_after = _gpu_param_sum[CriticNet](
            ctx, ensemble.pairs[i].target_net
        )
        var on_d = on_after - online_before[i]
        if on_d < 0.0:
            on_d = -on_d
        var tg_d = tg_after - target_before[i]
        if tg_d < 0.0:
            tg_d = -tg_d
        print(
            "  member", i,
            " online |Δ|sum =", on_d,
            " target |Δ|sum =", tg_d,
        )
        assert_true(
            on_d < 1e-5,
            "online critic byte-identical on GPU (input_only)",
        )
        assert_true(
            tg_d < 1e-5,
            "target net byte-identical on GPU",
        )

    assert_true(
        loss_last < loss_first,
        "actor loss must decrease (soft-V ascent)",
    )

    print("PASS — EnsembleActorLoss N=4 GPU smoke green.")


def main() raises:
    test_ensemble_actor_loss_gpu()
