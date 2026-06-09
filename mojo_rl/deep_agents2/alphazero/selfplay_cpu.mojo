"""AlphaZero self-play training driver — CPU path (single-env, GenericCPUMCTS).

The CPU counterpart to `selfplay.mojo`'s GPU driver. Where the GPU driver runs
`N_ENVS` games in parallel through one batched `GenericGPUMCTS` search, the CPU
driver plays a single game at a time: each move runs a `GenericCPUMCTS` search
(true game rules via the `AZ*CPU` adapters) from the live env state, records
`(canonical_obs, visit_policy)`, samples a move from the visit policy, and steps
the env. On a finished game the strict-alternation value target `z` is assigned
(last mover z=+1 on a win, signs alternate back; draw → 0) and the trajectory is
flushed to the (host-resident) `MCTSExampleReplay`; once it holds ≥ BATCH samples
the same nn2 AZ loss graph is trained on the CPU (`forward/vjp["cpu"]`).

This is the reference / debuggable path; it shares the loss graph, replay, and
value-target convention with the GPU driver, so a CPU-trained net is a faithful
(if slower, lower-throughput) AlphaZero. Returns the last mean train loss.
"""

from std.memory import alloc, UnsafePointer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS, AlphaGoPUCT, DirichletNoise, SelfPlay,
)

from .loss_ops import AZLossOp
from ..zero.mcts_adapters_cpu import AZRepCPU, AZDynCPU, AZPredCPU
from ..zero.example_replay import MCTSExampleReplay


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


def run_alphazero_selfplay_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & ImplicitlyDestructible,
    NET: Module,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
](
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
    comptime LATENT = ENV.SAVE_SIZE
    comptime MCTS = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        AlphaGoPUCT[1.0], DirichletNoise[0.25, 0.25], SelfPlay,
        NORMALIZE_Q=False,  # raw Q∈[-1,1] like legacy (MinMax over-explores)
    ]
    comptime Graph = ComputeGraph[
        1,
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var env = ENV()
    var opt = Adam.make["cpu", M=NET](net)
    opt.lr = lr
    var graph = Graph.make["cpu", INIT=Zero]()
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Host trajectory storage (single in-progress game) ──
    # Slabs rebound to MutAnyOrigin so the replay/tile signatures accept them.
    var traj_obs = mptr(alloc[Scalar[DT]](MAX_TRAJ * OBS))
    var traj_pol = mptr(alloc[Scalar[DT]](MAX_TRAJ * ACT))
    var tmp_tgt = alloc[Scalar[DT]](W)
    var root_save = alloc[Scalar[DT]](LATENT)
    var traj_len = 0

    # ── Train-batch host buffers + graph IO tiles ──
    var tb_obs = mptr(alloc[Scalar[DT]](BATCH * OBS))
    var tb_tgt = mptr(alloc[Scalar[DT]](BATCH * W))
    var tb_loss = mptr(alloc[Scalar[DT]](BATCH))
    var tb_grad = mptr(alloc[Scalar[DT]](BATCH))
    for i in range(BATCH):
        tb_grad[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    var tbo_t = TileTensor(tb_obs, row_major[BATCH, OBS]())
    var tbt_t = TileTensor(tb_tgt, row_major[BATCH, W]())
    var loss_t = TileTensor(tb_loss, row_major[BATCH, 1]())
    var grad_t = TileTensor(tb_grad, row_major[BATCH, 1]())

    _ = env.reset()
    var last_loss: Float64 = 0.0
    var rng = seed | 1

    for it in range(iterations):
        # 1. MCTS search from the live env state (eval mode for any BatchNorm).
        net.set_attr["training"](Scalar[DT](0.0))
        env.save_env_state(root_save)            # snapshot root (search trashes it)
        var env_ptr = UnsafePointer(to=env)
        var rep = AZRepCPU[ENV, OBS](env=env_ptr)
        var dyn = AZDynCPU[ENV, ACT](env=env_ptr)
        var pred = AZPredCPU[ENV, OBS, ACT, NET](
            env=env_ptr, net=UnsafePointer(to=net)
        )
        var mcts = MCTS(gamma=1.0)
        var legal = env.legal_action_mask()
        var root_obs = List[Float64](length=OBS, fill=Float64(0.0))  # ignored
        var policy = mcts.search[
            AZRepCPU[ENV, OBS], AZDynCPU[ENV, ACT], AZPredCPU[ENV, OBS, ACT, NET]
        ](rep, dyn, pred, root_obs, add_noise=True, legal_mask=legal)
        env.load_env_state(root_save)            # restore env to root

        # 2. Record (canonical obs, visit policy) into the trajectory.
        if traj_len < MAX_TRAJ:
            var obs_raw = env.get_obs_list()
            var ob = traj_len * OBS
            for j in range(OBS):
                traj_obs[ob + j] = Scalar[DT](obs_raw[j])
            var pb = traj_len * ACT
            for a in range(ACT):
                traj_pol[pb + a] = Scalar[DT](policy[a])
            traj_len += 1

        # 3. Sample a move from the visit policy (exploration), step the env.
        rng = _xs(rng)
        var u = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
        var cum: Float64 = 0.0
        var chosen = -1
        for a in range(ACT):
            cum += policy[a]
            if u <= cum and policy[a] > 0.0:
                chosen = a
                break
        if chosen < 0:                            # numeric fallback: argmax legal
            var bestv = Float64(-1.0)
            for a in range(ACT):
                if policy[a] > bestv:
                    bestv = policy[a]
                    chosen = a
            if chosen < 0:
                chosen = 0
        var step_res = env.step(env.action_from_index(chosen))
        var done = step_res[2]

        # 4. On a finished game: assign z, flush to replay, reset.
        if done:
            var gr = env.game_result()            # 1=P0 win, 2=P1 win, 3=draw
            for k in range(traj_len):
                var z: Float64 = 0.0
                if gr == 1:
                    z = 1.0 if (k % 2 == 0) else -1.0
                elif gr == 2:
                    z = 1.0 if (k % 2 == 1) else -1.0
                var pb = k * ACT
                for a in range(ACT):
                    tmp_tgt[a] = traj_pol[pb + a]
                tmp_tgt[ACT] = Scalar[DT](z)
                replay.record(traj_obs + k * OBS, tmp_tgt)
            traj_len = 0
            _ = env.reset()

        # 5. Train (train mode so any BatchNorm uses batch stats).
        if len(replay) >= BATCH and it >= learning_starts:
            net.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch[BATCH](tb_obs, tb_tgt)
                opt.zero_grad["cpu", M=NET](net)
                graph.set_external["pred", NET](net)
                graph.set_input["obs", BATCH](tbo_t)
                graph.set_input["tgt", BATCH](tbt_t)
                graph.forward["cpu", BATCH](loss_t)
                graph.vjp["cpu", BATCH](grad_t)
                opt.step["cpu", M=NET](net)
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(tb_loss[b])
            last_loss = ml / Float64(BATCH)

    traj_obs.free()
    traj_pol.free()
    tmp_tgt.free()
    root_save.free()
    tb_obs.free()
    tb_tgt.free()
    tb_loss.free()
    tb_grad.free()
    return last_loss
