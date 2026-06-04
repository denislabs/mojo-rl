"""Arena-gated AlphaZero self-play with symmetry augmentation — full AlphaZero.

Extends `run_alphazero_selfplay` with the two production pieces:

  * **Best/learner split + Arena gating.** A frozen *best* net generates all
    self-play (MCTS expansion + the policy/value priors); a *learner* net trains
    on that data. Every `arena_every` iterations the learner plays the best in
    the Arena (`candidate_winrate`, both colors, random openings); if it wins a
    clear majority of decisive games (`should_promote`) it is copied over the
    best (`hard_copy_params`) and becomes the new generator. This is the
    accept/reject loop that stops a transiently-worse learner from poisoning the
    self-play distribution.

  * **Symmetry augmentation.** Each recorded `(obs, π)` sample is replicated
    under the board's symmetry group (`AUG`, e.g. TicTacToe's 8 D4 symmetries);
    the scalar value target `z` is symmetry-invariant and copied. Multiplies
    effective data and bakes in the game's invariances.

The passed-in `net` is the *best* and holds the final (best) weights on return.
`AUG = IdentityAugmenter` recovers the un-augmented behaviour.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.initializer import Zero, Kaiming
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.planners.tree_search import (
    GenericGPUMCTS, AlphaGoPUCT, DirichletNoise, SelfPlay,
)

from .loss_ops import AZLossOp
from .arena import candidate_winrate, should_promote
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.example_replay import MCTSExampleReplay
from ..zero.symmetries import BoardAugmenter


@fieldwise_init
struct ArenaRunResult(Copyable, Movable):
    """Summary of an arena-gated run: last mean train loss + how many times the
    learner was accepted over the best."""
    var last_loss: Float64
    var promotions: Int


def run_alphazero_selfplay_arena[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    AUG: BoardAugmenter,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
    ARENA_GAMES: Int = 32,        # arena games per color (comptime: sizes buffers)
    RESULT_IDX: Int = 10,
    MAX_PLIES: Int = 9,
](
    ctx: DeviceContext,
    mut net: NET,                 # the BEST net — holds final weights on return
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
    arena_every: Int = 100,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
) raises -> ArenaRunResult:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM          # ACT + 1
    comptime STATE = ENV.STATE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime MCTS = GenericGPUMCTS[
        N_ENVS, ACT, OBS, 1, MAX_NODES, NUM_SIMS, 1,
        AlphaGoPUCT[2.5], DirichletNoise[0.25, 0.25], SelfPlay,
        STATE_SIZE=STATE,
    ]
    comptime Graph = ComputeGraph[
        1,
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    # Learner net (trains) initialised to the best net's weights.
    var learner = NET.make["gpu", INIT=Kaiming](ctx=ctx)
    hard_copy_params["gpu", M=NET](net, learner, ctx)

    var opt = Adam.make["gpu", M=NET](learner, ctx)
    opt.lr = lr
    var graph = Graph.make["gpu", INIT=Zero](ctx=ctx)
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Device buffers ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var tb_obs = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var tb_tgt = ctx.enqueue_create_buffer[DT](BATCH * W)
    var tb_loss = ctx.enqueue_create_buffer[DT](BATCH)
    var tb_grad = ctx.enqueue_create_buffer[DT](BATCH)

    # ── Host buffers ──
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var tb_obs_h = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var tb_tgt_h = ctx.enqueue_create_host_buffer[DT](BATCH * W)
    var loss_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    var grad_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    # ── Host trajectory storage (per-env in-progress game) ──
    # Augmenter signatures pin MutAnyOrigin — rebind the owned slabs to match.
    var traj_obs = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * OBS)
    )
    var traj_pol = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * ACT)
    )
    var traj_len = alloc[Int](N_ENVS)
    var tmp_tgt = alloc[Scalar[DT]](W)
    var aug_obs = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](OBS)
    )
    var aug_pol = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](ACT)
    )
    for e in range(N_ENVS):
        traj_len[e] = 0

    # Constant 1/BATCH grad seed (uploaded once).
    for i in range(BATCH):
        grad_h.unsafe_ptr()[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    ctx.enqueue_copy(tb_grad, grad_h)
    ctx.synchronize()

    var tbo_t = TileTensor(tb_obs, row_major[BATCH, OBS]())
    var tbt_t = TileTensor(tb_tgt, row_major[BATCH, W]())
    var loss_t = TileTensor(tb_loss, row_major[BATCH, 1]())
    var grad_t = TileTensor(tb_grad, row_major[BATCH, 1]())

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs_dev, legal_dev)
    ctx.synchronize()

    var last_loss: Float64 = 0.0
    var promotions = 0

    for it in range(iterations):
        # 1. MCTS search with the BEST net (eval mode for any BatchNorm).
        net.set_attr["training"](Scalar[DT](0.0))
        var pred = AZPredGPU[OBS, ACT, NET].make(net)
        var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
        var root_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
        ](obs_dev.unsafe_ptr())
        var root_legal = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
        ](legal_dev.unsafe_ptr())
        mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
            ctx, pred, env_ad, root_obs, states, root_legal,
            rng_seed=seed + UInt64(it),
        )

        # 2. Pull root obs + visit-count policy, record into trajectory.
        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, mcts.policies_out)
        ctx.synchronize()
        for e in range(N_ENVS):
            var k = traj_len[e]
            if k < MAX_TRAJ:
                var ob = (e * MAX_TRAJ + k) * OBS
                for j in range(OBS):
                    traj_obs[ob + j] = obs_h.unsafe_ptr()[e * OBS + j]
                var pb = (e * MAX_TRAJ + k) * ACT
                for a in range(ACT):
                    traj_pol[pb + a] = pol_h.unsafe_ptr()[e * ACT + a]
                traj_len[e] = k + 1

        # 3. Step every game by its chosen action.
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, mcts.actions_out, rew, done, term, obs_next, legal_next,
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # 4. Flush finished games with value target z, augmented by symmetry.
        for e in range(N_ENVS):
            if done_h.unsafe_ptr()[e] > 0.5:
                var L = traj_len[e]
                var win = Float64(rew_h.unsafe_ptr()[e]) > 0.5
                for k in range(L):
                    var z: Float64 = 0.0
                    if win:
                        z = 1.0 if ((L - 1 - k) % 2 == 0) else -1.0
                    var ob = (e * MAX_TRAJ + k) * OBS
                    var pb = (e * MAX_TRAJ + k) * ACT
                    for s in range(NSYM):
                        AUG.augment_obs[OBS](traj_obs + ob, s, aug_obs)
                        AUG.augment_policy[ACT](traj_pol + pb, s, aug_pol)
                        for a in range(ACT):
                            tmp_tgt[a] = aug_pol[a]
                        tmp_tgt[ACT] = Scalar[DT](z)
                        replay.record(aug_obs, tmp_tgt)
                traj_len[e] = 0

        # 5. Reset finished games, refresh obs/legal.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs_dev, legal_dev)
        ctx.synchronize()

        # 6. Train the LEARNER (train mode for BatchNorm).
        if len(replay) >= BATCH and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch[BATCH](
                    tb_obs_h.unsafe_ptr(), tb_tgt_h.unsafe_ptr()
                )
                ctx.enqueue_copy(tb_obs, tb_obs_h)
                ctx.enqueue_copy(tb_tgt, tb_tgt_h)
                ctx.synchronize()
                opt.zero_grad["gpu", M=NET](learner)
                graph.set_external["pred", NET](learner)
                graph.set_input["obs", BATCH](tbo_t)
                graph.set_input["tgt", BATCH](tbt_t)
                graph.forward["gpu", BATCH](loss_t)
                graph.vjp["gpu", BATCH](grad_t)
                opt.step["gpu", M=NET](learner)
            ctx.enqueue_copy(loss_h, tb_loss)
            ctx.synchronize()
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_h.unsafe_ptr()[b])
            last_loss = ml / Float64(BATCH)

        # 7. Arena gating: periodically challenge the best with the learner.
        if (
            it >= learning_starts
            and arena_every > 0
            and (it + 1) % arena_every == 0
            and len(replay) >= BATCH
        ):
            var rec = candidate_winrate[
                ENV, NET, NET, ARENA_GAMES, RESULT_IDX, MAX_PLIES,
            ](ctx, learner, net, seed=seed + UInt64(it) * 7 + 1,
              open_plies=arena_open_plies)
            if should_promote(rec, promote_threshold, min_decisive=ARENA_GAMES // 2):
                hard_copy_params["gpu", M=NET](learner, net, ctx)
                promotions += 1

    # Final flush: ensure the returned best is at least as new as the learner if
    # the learner ended clearly ahead (covers runs shorter than arena_every).
    var final_rec = candidate_winrate[
        ENV, NET, NET, ARENA_GAMES, RESULT_IDX, MAX_PLIES,
    ](ctx, learner, net, seed=seed + 9991, open_plies=arena_open_plies)
    if should_promote(final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2):
        hard_copy_params["gpu", M=NET](learner, net, ctx)
        promotions += 1

    traj_obs.free()
    traj_pol.free()
    traj_len.free()
    tmp_tgt.free()
    aug_obs.free()
    aug_pol.free()
    return ArenaRunResult(last_loss=last_loss, promotions=promotions)
