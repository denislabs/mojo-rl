"""AlphaZero self-play training driver — ties search → record → outcome-z →
replay → train into one host-orchestrated loop (GPU MCTS + GPU net + GPU env).

One `run_alphazero_selfplay` call drives `iterations` self-play *moves* across
`N_ENVS` parallel games:

  1. `search_gpu_alphazero` (true game rules) → action + visit-count policy + root value.
  2. record (canonical_obs, policy) into the per-env in-progress trajectory.
  3. step every game by its chosen action (`env.step_kernel_gpu`).
  4. on a finished game, assign value targets z and flush (obs, [π | z]) to replay:
       * strict-alternation zero-sum ⇒ no absolute player index needed — the last
         mover gets z=+1 on a win, signs alternate backward by step parity
         (`z_k = +1 if (L-1-k) even else -1`); a draw gives z=0 for every step.
         (canonical obs + canonical z = the AlphaZero training convention.)
  5. selective-reset finished games, re-extract obs/legal.
  6. once replay ≥ BATCH, run `train_per_iter` graph train steps (the bit-identical
     CPU/GPU AZ loss graph) and update the net.

Returns the last observed mean train loss (0.0 if training never triggered).
This is the driver; the agent/checkpoint facade (a thin struct over it) lands next.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.planners.tree_search import (
    GenericGPUMCTS, AlphaGoPUCT, DirichletNoise, SelfPlay,
)

from .loss_ops import AZLossOp
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.example_replay import MCTSExampleReplay


def run_alphazero_selfplay[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
) raises -> Float64:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM          # ACT + 1
    comptime STATE = ENV.STATE_SIZE
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

    var opt = Adam.make["gpu", M=NET](net, ctx)
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
    var traj_obs = alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * OBS)
    var traj_pol = alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * ACT)
    var traj_len = alloc[Int](N_ENVS)
    var tmp_tgt = alloc[Scalar[DT]](W)
    for e in range(N_ENVS):
        traj_len[e] = 0

    # Constant 1/BATCH grad seed (uploaded once).
    for i in range(BATCH):
        grad_h.unsafe_ptr()[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    ctx.enqueue_copy(tb_grad, grad_h)
    ctx.synchronize()

    # Train-graph IO tiles (stable buffers; created once, reused each step).
    var tbo_t = TileTensor(tb_obs, row_major[BATCH, OBS]())
    var tbt_t = TileTensor(tb_tgt, row_major[BATCH, W]())
    var loss_t = TileTensor(tb_loss, row_major[BATCH, 1]())
    var grad_t = TileTensor(tb_grad, row_major[BATCH, 1]())

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var last_loss: Float64 = 0.0

    for it in range(iterations):
        # 1. MCTS search across all envs. Net runs in eval mode so any
        #    BatchNorm (CNN / ResNet torsos) uses running stats during the
        #    single-position inference of MCTS expansion (no-op for the MLP).
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

        # 2. Pull root obs + visit-count policy to host, record into trajectory.
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
            ctx, states, mcts.actions_out, rew, done, term,
            obs_next, legal_next,
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # 4. Flush finished games into replay with value targets z.
        for e in range(N_ENVS):
            if done_h.unsafe_ptr()[e] > 0.5:
                var L = traj_len[e]
                var win = Float64(rew_h.unsafe_ptr()[e]) > 0.5
                for k in range(L):
                    var z: Float64 = 0.0
                    if win:
                        z = 1.0 if ((L - 1 - k) % 2 == 0) else -1.0
                    var pb = (e * MAX_TRAJ + k) * ACT
                    for a in range(ACT):
                        tmp_tgt[a] = traj_pol[pb + a]
                    tmp_tgt[ACT] = Scalar[DT](z)
                    replay.record(traj_obs + (e * MAX_TRAJ + k) * OBS, tmp_tgt)
                traj_len[e] = 0

        # 5. Reset finished games, refresh obs/legal for the next move.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # 6. Train. Net back in train mode so BatchNorm uses batch stats and
        #    updates its running averages (no-op for the MLP).
        if len(replay) >= BATCH and it >= learning_starts:
            net.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch[BATCH](
                    tb_obs_h.unsafe_ptr(), tb_tgt_h.unsafe_ptr()
                )
                ctx.enqueue_copy(tb_obs, tb_obs_h)
                ctx.enqueue_copy(tb_tgt, tb_tgt_h)
                ctx.synchronize()
                opt.zero_grad["gpu", M=NET](net)
                graph.set_external["pred", NET](net)
                graph.set_input["obs", BATCH](tbo_t)
                graph.set_input["tgt", BATCH](tbt_t)
                graph.forward["gpu", BATCH](loss_t)
                graph.vjp["gpu", BATCH](grad_t)
                opt.step["gpu", M=NET](net)
            ctx.enqueue_copy(loss_h, tb_loss)
            ctx.synchronize()
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_h.unsafe_ptr()[b])
            last_loss = ml / Float64(BATCH)

    traj_obs.free()
    traj_pol.free()
    traj_len.free()
    tmp_tgt.free()
    return last_loss
