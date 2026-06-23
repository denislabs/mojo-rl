"""AlphaZero self-play training driver — CPU path (single-env, GenericCPUMCTS).

The CPU counterpart to `selfplay.mojo`'s GPU driver. Where the GPU driver runs
`N_ENVS` games in parallel through one batched `GenericGPUMCTS` search, the CPU
driver plays a single game at a time: each move runs a `GenericCPUMCTS` search
(true game rules via the `AZ*CPU` adapters) from the live env state, records
`(canonical_obs, visit_policy)`, samples a move from the visit policy, and steps
the env. On a finished game the strict-alternation value target `z` is assigned
(last mover z=+1 on a win, signs alternate back; draw → 0) and the trajectory is
flushed to the (host-resident) `MCTSExampleReplay`; once it holds ≥ BATCH samples
the same nn AZ loss graph is trained on the CPU (`forward/vjp["cpu"]`).

Storage surface: the loss graph runs on the storage ComputeGraph (net threaded
as a forward/vjp external arg; `set_input` takes a `Tensor`; Adam via
`begin_step` + `for_each_param`). The single-game trajectory + the example
replay stay category-B raw host buffers (the env/replay interop boundary); the
batch is bridged into storage Tensors by `sample_batch_tensors`.
"""

from std.memory import UnsafePointer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.initializer import Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import (
    InputSlot,
    Node,
    ExternalNode,
)
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    AlphaGoPUCT,
    DirichletNoise,
    SelfPlay,
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
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & ImplicitlyDeletable,
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
    comptime W = NET.OUT_DIM  # ACT + 1
    comptime LATENT = ENV.SAVE_SIZE
    comptime MCTS = GenericCPUMCTS[
        ACT,
        LATENT,
        NUM_SIMS,
        MAX_NODES,
        AlphaGoPUCT[1.0],
        DirichletNoise[0.25, 0.25],
        SelfPlay,
        NORMALIZE_Q=False,  # raw Q∈[-1,1] like legacy (MinMax over-explores)
    ]
    comptime Graph = ComputeGraph[
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var env = ENV()
    var opt = Adam(lr=lr)
    var graph = Graph.make["cpu", Zero]()
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Host trajectory storage (single in-progress game) — owned Lists. ──
    var traj_obs = List[Scalar[DT]](length=MAX_TRAJ * OBS, fill=0)
    var traj_pol = List[Scalar[DT]](length=MAX_TRAJ * ACT, fill=0)
    var tmp_tgt = List[Scalar[DT]](length=W, fill=0)
    var root_save = List[Scalar[DT]](length=LATENT, fill=0)
    var traj_len = 0

    # ── Train-batch storage Tensors (the nn surface) ──
    var obs_t = Tensor.alloc(BATCH * OBS)
    var tgt_t = Tensor.alloc(BATCH * W)
    var loss_t = Tensor.alloc(BATCH)
    var grad_t = Tensor.alloc(BATCH)
    for i in range(BATCH):
        grad_t.data[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)

    _ = env.reset()
    var last_loss: Float64 = 0.0
    var rng = seed | 1

    for it in range(iterations):
        # 1. MCTS search from the live env state (eval mode for any BatchNorm).
        net.set_attr["training"](Scalar[DT](0.0))
        env.save_env_state(root_save)  # snapshot root (search trashes it)
        # TODO(unsafe-origin): rep/dyn/pred each hold a NON-OWNING mutable handle
        # to the SAME `env` (the planner mutates it through them during search) —
        # a deliberate 3-way mutable alias that `GenericCPUMCTS.search` takes all
        # at once. A concrete `Pointer[E, o]` (TensorRefs-style, as `AZPredGPU`'s
        # single-ptr `o: Origin` does) would trip exclusivity here, so the handles
        # stay `MutAnyOrigin` and we discard the origin explicitly via
        # `as_unsafe_any_origin()`. A clean fix needs the planner to take `env`
        # ONCE (e.g. thread it through `search`) instead of embedding 3 aliases.
        var env_ptr = UnsafePointer(to=env)
        var net_ptr = UnsafePointer(to=net)
        var rep = AZRepCPU[ENV, OBS](env=env_ptr.as_unsafe_any_origin())
        var dyn = AZDynCPU[ENV, ACT](env=env_ptr.as_unsafe_any_origin())
        var pred = AZPredCPU[ENV, OBS, ACT, NET](
            env=env_ptr.as_unsafe_any_origin(),
            net=net_ptr.as_unsafe_any_origin(),
        )
        var mcts = MCTS(gamma=1.0)
        var legal = env.legal_action_mask()
        var root_obs = List[Float64](length=OBS, fill=Float64(0.0))  # ignored
        var policy = mcts.search[
            AZRepCPU[ENV, OBS],
            AZDynCPU[ENV, ACT],
            AZPredCPU[ENV, OBS, ACT, NET],
        ](rep, dyn, pred, root_obs, add_noise=True, legal_mask=legal)
        env.load_env_state(root_save)  # restore env to root

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
        if chosen < 0:  # numeric fallback: argmax legal
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
            var gr = env.game_result()  # 1=P0 win, 2=P1 win, 3=draw
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
                replay.record(traj_obs, k * OBS, tmp_tgt, 0)
            traj_len = 0
            _ = env.reset()

        # 5. Train (train mode so any BatchNorm uses batch stats).
        if len(replay) >= BATCH and it >= learning_starts:
            net.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch_tensors[BATCH](obs_t, tgt_t)
                net.zero_grad["cpu"](None)
                graph.set_input["obs", BATCH](obs_t, None)
                graph.set_input["tgt", BATCH](tgt_t, None)
                graph.forward[BATCH, "cpu"](loss_t, None, net)
                graph.vjp[BATCH, "cpu"](grad_t, None, net)
                opt.begin_step()
                net.for_each_param["cpu"](opt, None)
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_t.data[b])
            last_loss = ml / Float64(BATCH)

    return last_loss
