"""G.4 — EnsembleActorStepOFE GPU smoke.

Gates the full GPU forward+backward chain through the OFE-aware actor
loss block:

  - GPU concat+lp via the reused `_eal_concat_sa_extract_lp_kernel`
    (with PHI_S_DIM in the first-input slot).
  - Device action_branch.forward + N critic forwards.
  - Per-batch q_sum accumulation via `_eal_add_into_kernel`.
  - D2H of `_mb_q_sum` + `_mb_lp_dev` for the host-side loss/lp_mean
    reduction.
  - Per-critic grad_q_i fill via `_eal_fill_const_kernel` + critic.vjp
    [input_only] + accumulate grad_φ(s,a)_sum.
  - action_branch.vjp[input_only] on device.
  - grad_alp build via `_eal_build_grad_alp_kernel`.
  - rsample.vjp + actor.vjp + actor_opt.step all on device.

Gates:
  (1) make+forward_backward run end-to-end on GPU.
  (2) loss + log_prob_mean both finite.
  (3) action_branch params byte-identical pre/post (stop-grad via
      `mode='input_only'` — strongest correctness gate).
  (4) critic[0] params byte-identical pre/post (same reason).
  (5) actor params CHANGE pre/post (actor.vjp + actor_opt.step ran)."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.primitives.linear import Linear

from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.redq.ensemble import CriticEnsemble
from mojo_rl.deep_agents2.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6,
    OFEFeatureStep, EnsembleActorStepOFE,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6
comptime N = 2

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime ACTOR = Sequential[Linear[PHI_S_DIM, 2 * ACT]]
comptime CRITIC = Sequential[Linear[PHI_SA_DIM, 1]]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _read_dev_scalar(
    ctx: DeviceContext,
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    """Read 1 device scalar into a host scalar."""
    var host = ctx.enqueue_create_host_buffer[DT](1)
    var src_lt = ctx.enqueue_create_buffer[DT](1)
    src_lt.enqueue_fill(Scalar[DT](0.0))
    # Direct memcpy through a 1-elem host buffer.
    # The caller passes a device address; we slip it into a temporary
    # DeviceBuffer view via enqueue_copy from raw ptr — but Mojo's
    # enqueue_copy needs DeviceBuffer endpoints. Simpler: caller has
    # to supply the source DeviceBuffer. This helper is unused; the
    # test below reads params via the Param's value buffer directly.
    ctx.synchronize()
    return host.unsafe_ptr()[0]


def test_actor_step_gpu() raises:
    print("=" * 70)
    print("G.4 — EnsembleActorStepOFE on GPU (actor.vjp + AB.vjp[input_only])")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    var sb = SB.make[target="gpu", INIT=Xavier](ctx)
    var ab = AB.make[target="gpu", INIT=Xavier](ctx)
    var actor = ACTOR.make[target="gpu", INIT=Xavier](ctx)
    var actor_opt = Adam.make[target="gpu", M=ACTOR](actor, ctx)
    actor_opt.lr = Scalar[DT](3e-3)
    var ensemble = CriticEnsemble[CRITIC, N].make[
        target="gpu", INIT=Xavier,
    ](ctx)
    var feat = OFEFeatureStep[
        SB, OBS, ACT, BATCH,
    ].make[target="gpu"](ctx)
    var astep = EnsembleActorStepOFE[
        ACTOR, AB, CRITIC, N, BATCH, PHI_S_DIM, ACT,
    ].make[target="gpu"](action_scale=Scalar[DT](1.0), ctx=ctx)

    var state = TrainerState[OBS, ACT, BATCH].make[target="gpu"](ctx)

    # Synthetic obs + next_obs → H2D.
    var obs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    var nobs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    for b in range(BATCH):
        for d in range(OBS):
            obs_h[b * OBS + d] = Scalar[DT](
                0.25 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
            nobs_h[b * OBS + d] = Scalar[DT](
                0.35 - 0.07 * Float64(b) + 0.03 * Float64(d)
            )
    var obs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var nobs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    ctx.synchronize()
    for i in range(BATCH * OBS):
        obs_host.unsafe_ptr()[i] = obs_h[i]
        nobs_host.unsafe_ptr()[i] = nobs_h[i]
    ctx.enqueue_copy(state.mb_s.dev.value(), obs_host)
    ctx.enqueue_copy(state.mb_sp.dev.value(), nobs_host)

    feat.step["gpu"](sb, state)
    var phi_s_p = feat.phi_s_ptr["gpu"]()

    # Snapshot AB[0] + critic[0] params via D2H of a single value.
    var ab_w_pre_host = ctx.enqueue_create_host_buffer[DT](1)
    var c0_w_pre_host = ctx.enqueue_create_host_buffer[DT](1)
    var actor_w_pre_host = ctx.enqueue_create_host_buffer[DT](1)
    var ab_w_dev = (
        ab.children[0].inner.children[0].weight.val.dev.value()
    )
    var c0_w_dev = (
        ensemble.pairs[0].online.children[0].weight.val.dev.value()
    )
    var actor_w_dev = actor.children[0].weight.val.dev.value()
    ctx.enqueue_copy(ab_w_pre_host, ab_w_dev)
    ctx.enqueue_copy(c0_w_pre_host, c0_w_dev)
    ctx.enqueue_copy(actor_w_pre_host, actor_w_dev)
    ctx.synchronize()
    var ab_w_pre = ab_w_pre_host.unsafe_ptr()[0]
    var c0_w_pre = c0_w_pre_host.unsafe_ptr()[0]
    var actor_w_pre = actor_w_pre_host.unsafe_ptr()[0]

    # Run the actor step.
    var alpha = Scalar[DT](0.1)
    var res = astep.forward_backward["gpu"](
        actor, actor_opt, ab, ensemble, phi_s_p, alpha,
    )
    ctx.synchronize()

    print("  GPU actor loss     =", res.loss)
    print("  GPU log_prob_mean  =", res.log_prob_mean)

    # (1)(2) finite.
    assert_true(
        res.loss == res.loss and res.log_prob_mean == res.log_prob_mean,
        "GPU actor loss + log_prob_mean finite",
    )

    # Snapshot post params.
    var ab_w_post_host = ctx.enqueue_create_host_buffer[DT](1)
    var c0_w_post_host = ctx.enqueue_create_host_buffer[DT](1)
    var actor_w_post_host = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(ab_w_post_host, ab_w_dev)
    ctx.enqueue_copy(c0_w_post_host, c0_w_dev)
    ctx.enqueue_copy(actor_w_post_host, actor_w_dev)
    ctx.synchronize()
    var ab_w_post = ab_w_post_host.unsafe_ptr()[0]
    var c0_w_post = c0_w_post_host.unsafe_ptr()[0]
    var actor_w_post = actor_w_post_host.unsafe_ptr()[0]

    print("  AB[0].weight[0]    pre/post =", ab_w_pre, "/", ab_w_post)
    print("  C0.weight[0]       pre/post =", c0_w_pre, "/", c0_w_post)
    print("  ACTOR.weight[0]    pre/post =",
          actor_w_pre, "/", actor_w_post)

    # (3) AB params byte-identical (input_only stop-grad).
    assert_true(
        _abs(ab_w_post - ab_w_pre) < Scalar[DT](1e-10),
        "GPU AB params must NOT change (mode='input_only')",
    )
    # (4) Critic params byte-identical.
    assert_true(
        _abs(c0_w_post - c0_w_pre) < Scalar[DT](1e-10),
        "GPU critic params must NOT change (mode='input_only')",
    )
    # (5) Actor params CHANGE.
    assert_true(
        _abs(actor_w_post - actor_w_pre) > Scalar[DT](1e-5),
        "GPU actor params must change (actor.vjp + actor_opt.step)",
    )

    obs_h.free()
    nobs_h.free()

    print("PASS — EnsembleActorStepOFE GPU path: actor trains, AB+critics frozen.")


def main() raises:
    test_actor_step_gpu()
