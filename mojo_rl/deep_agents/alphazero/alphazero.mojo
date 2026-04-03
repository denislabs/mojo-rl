"""AlphaZero Agent — Self-play RL with true game rules and MCTS.

Training is simple supervised learning:
  loss = CE(policy_pred, mcts_policy) + MSE(value_pred, game_outcome)

No representation network, no dynamics network, no K-step unroll.
Just one prediction network trained on self-play data.

Reference: Silver et al., 2017 — Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning Algorithm
"""

from std.math import exp, log, sqrt
from std.random import random_float64
from std.memory import alloc, memset
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    save_checkpoint_file,
    read_checkpoint_file,
    parse_checkpoint_header,
    read_metadata_section,
    get_metadata_value,
)
from mojo_rl.core import (
    TrainingMetrics,
    TwoPlayerDiscreteEnv,
    GPUTwoPlayerDiscreteEnv,
    DataAugmentable,
    Saveable,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    increment_rng_counter_kernel,
    sample_indices_kernel,
    log_and_reset_completed_kernel,
    uniform_random_discrete_actions_kernel,
    uniform_random_legal_actions_kernel,
)
from mojo_rl.deep_agents.muzero.gpu_mcts import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_extract_actions_temp_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_copy_root_state_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_kernel,
    TPB,
    MAX_DEPTH,
)

# extract_hidden_kernel not needed for AlphaZero (no dynamics)
from mojo_rl.deep_agents.muzero.evaluators import (
    Evaluator,
    GPUEvaluator,
    RandomOpponent,
)
from mojo_rl.cuda.graph import CUDAGraph
from .configs import AlphaZeroConfig
from .state import AlphaZeroCPUState, AlphaZeroGPUState


# ═══════════════════════════════════════════════════════════════════════════
# GPU Kernels for AlphaZero Training
# ═══════════════════════════════════════════════════════════════════════════


def az_policy_value_grad_kernel[
    BATCH: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    pred_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    target_policy: LayoutTensor[
        dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
    ],
    target_value: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Compute combined policy CE + value MSE gradient.

    grad_policy = (softmax(logits) - target_policy) / BATCH
    grad_value = 2 * (pred_value - target_value) / BATCH

    One thread per batch sample.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var pred_off = b * PRED_OUT
    var pol_off = b * ACT
    var inv_batch = Scalar[dtype](1.0) / Scalar[dtype](BATCH)

    # Policy gradient: softmax(logits) - target
    var max_logit = rebind[Scalar[dtype]](pred_out[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_out[pred_off + a])
        if v > max_logit:
            max_logit = v
    var sum_exp = Scalar[dtype](0.0)
    for a in range(ACT):
        sum_exp += exp(
            rebind[Scalar[dtype]](pred_out[pred_off + a]) - max_logit
        )
    # Pure CE gradient: softmax(logits) - target
    for a in range(ACT):
        var prob = (
            exp(rebind[Scalar[dtype]](pred_out[pred_off + a]) - max_logit)
            / sum_exp
        )
        var target = rebind[Scalar[dtype]](target_policy[pol_off + a])
        grad_out[pred_off + a] = (prob - target) * inv_batch

    # Value gradient: MSE through tanh activation
    # loss = (tanh(raw) - target)^2
    # d/draw = 2 * (tanh(raw) - target) * (1 - tanh(raw)^2)
    var raw_v = rebind[Scalar[dtype]](pred_out[pred_off + ACT])
    var target_v = rebind[Scalar[dtype]](target_value[b])
    var ev_p = exp(raw_v)
    var ev_n = exp(-raw_v)
    var tanh_v = (ev_p - ev_n) / (ev_p + ev_n)
    var dtanh = Scalar[dtype](1.0) - tanh_v * tanh_v  # tanh derivative
    grad_out[pred_off + ACT] = (
        Scalar[dtype](2.0) * (tanh_v - target_v) * dtanh * inv_batch
    )


@always_inline
def az_gather_batch_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    OBS: Int,
    ACT: Int,
    CAPACITY: Int,
](
    # Output batch
    batch_obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    batch_policy: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ],
    batch_value: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    # GPU replay storage
    replay_obs: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS), MutAnyOrigin
    ],
    replay_policy: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, ACT), MutAnyOrigin
    ],
    replay_value: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    # Sampled indices
    indices: LayoutTensor[DType.int32, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Gather AlphaZero training batch from GPU replay buffer by indices.

    Each thread gathers one sample (obs, policy, value).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return

    var buf_idx = Int(indices[i])

    for d in range(OBS):
        batch_obs[i, d] = replay_obs[buf_idx, d]
    for d in range(ACT):
        batch_policy[i, d] = replay_policy[buf_idx, d]
    batch_value[i] = replay_value[buf_idx]


# ═══════════════════════════════════════════════════════════════════════════
# AlphaZero Agent
# ═══════════════════════════════════════════════════════════════════════════


struct GenericAlphaZeroAgent[
    Config: AlphaZeroConfig, n_envs: Int = 64, L: Logger = NoOpLogger
](Movable):
    """AlphaZero agent for two-player board games.

    Uses GPU MCTS with true game rules for self-play, trains a single
    prediction network with supervised policy + value loss.

    Parameters:
        Config: Compile-time config (AlphaZeroTicTacToeConfig, etc.).
        n_envs: Number of parallel environments for GPU training.
        L: Logger type for diagnostic logging (default: NoOpLogger).
    """

    comptime StateType = AlphaZeroCPUState[Self.Config]
    comptime GPUStateType = AlphaZeroGPUState[Self.Config]
    comptime PredNet = Network[Self.Config.PredModel, Self.Config.OptType]
    comptime PRED_OUT: Int = Self.Config.action_dim + 1

    var state: Self.StateType
    var train_step_count: Int
    var total_steps: Int

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(out self):
        self.state = Self.StateType()
        self.train_step_count = 0
        self.total_steps = 0
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.train_step_count = take.train_step_count
        self.total_steps = take.total_steps
        self.logger = take.logger
        self.diag_every = take.diag_every

    # ══════════════════════════════════════════════════════════════
    # Policy-only action selection (for evaluation)
    # ══════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        legal_mask: List[Bool],
    ) -> Int:
        """Select action using raw policy network (argmax over legal)."""
        comptime B: Int = 1
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM

        var obs_ptr = alloc[Scalar[dtype]](OBS)
        for i in range(OBS):
            if i < len(obs):
                obs_ptr[i] = obs[i]
            else:
                obs_ptr[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](obs_ptr)

        var pred_ptr = alloc[Scalar[dtype]](PRED_OUT_DIM)
        memset(pred_ptr, 0, PRED_OUT_DIM)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT_DIM), MutAnyOrigin
        ](pred_ptr)

        Self.PredNet.forward[B](
            obs_t, pred_t, self.state.prediction.params_view()
        )

        var best_action = -1
        var best_logit = Float64(-1e18)
        for a in range(ACT):
            if a < len(legal_mask) and legal_mask[a]:
                var logit = Float64(rebind[Scalar[dtype]](pred_t[0, a]))
                if logit > best_logit:
                    best_logit = logit
                    best_action = a

        obs_ptr.free()
        pred_ptr.free()
        if best_action < 0:
            for a in range(ACT):
                if a < len(legal_mask) and legal_mask[a]:
                    return a
        return best_action

    def select_action_mcts[
        E: TwoPlayerDiscreteEnv & Saveable,
    ](
        mut self,
        obs: List[Scalar[dtype]],
        legal_mask: List[Bool],
        mut env: E,
        num_sims: Int = 0,
    ) -> Int:
        """MCTS with save/restore. Returns action with highest visit count."""
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime B: Int = 1
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime MAX_N = Self.Config.max_nodes
        comptime C_PUCT = Self.Config.PUCT.C_INIT

        var sims = num_sims if num_sims > 0 else Self.Config.num_simulations

        # Root network forward pass
        var obs_ptr = alloc[Scalar[dtype]](OBS)
        for i in range(OBS):
            obs_ptr[i] = obs[i] if i < len(obs) else Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](obs_ptr)
        var pred_ptr = alloc[Scalar[dtype]](PRED_OUT_DIM)
        memset(pred_ptr, 0, PRED_OUT_DIM)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT_DIM), MutAnyOrigin
        ](pred_ptr)
        Self.PredNet.forward[B](
            obs_t, pred_t, self.state.prediction.params_view()
        )

        # Softmax over legal actions for root prior
        var prior = alloc[Float64](ACT)
        var max_logit: Float64 = -1e18
        for a in range(ACT):
            var l = Float64(rebind[Scalar[dtype]](pred_t[0, a]))
            if a < len(legal_mask) and legal_mask[a] and l > max_logit:
                max_logit = l
        var sum_exp: Float64 = 0.0
        for a in range(ACT):
            if a < len(legal_mask) and legal_mask[a]:
                prior[a] = exp(
                    Float64(rebind[Scalar[dtype]](pred_t[0, a])) - max_logit
                )
                sum_exp += prior[a]
            else:
                prior[a] = 0.0
        if sum_exp > 0:
            for a in range(ACT):
                prior[a] /= sum_exp
        obs_ptr.free()
        pred_ptr.free()

        # MCTS tree (flat arrays)
        var visit_count = alloc[Int](MAX_N * ACT)
        var total_value = alloc[Float64](MAX_N * ACT)
        var child_idx = alloc[Int](MAX_N * ACT)
        var node_prior = alloc[Float64](MAX_N * ACT)
        var node_visits = alloc[Int](MAX_N)
        var node_count = 1

        memset(visit_count, 0, MAX_N * ACT)
        memset(total_value, 0, MAX_N * ACT)
        memset(node_visits, 0, MAX_N)
        for i in range(MAX_N * ACT):
            child_idx[i] = -1
        node_visits[0] = 1
        for a in range(ACT):
            node_prior[a] = prior[a]

        # Per-node saved env states
        var node_states = alloc[Scalar[dtype]](MAX_N * E.SAVE_SIZE)

        # Save root state
        var root_save = alloc[Scalar[dtype]](E.SAVE_SIZE)
        env.save_env_state(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](root_save)
        )
        for i in range(E.SAVE_SIZE):
            node_states[0 * E.SAVE_SIZE + i] = root_save[i]

        for sim in range(sims):
            if node_count >= MAX_N:
                break

            # Restore env to root
            env.load_env_state(
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](root_save)
            )

            var node = 0
            var path = List[Int]()
            var path_actions = List[Int]()

            # Selection: descend tree using PUCT
            var found_leaf = False
            while not found_leaf:
                var total_n = Float64(node_visits[node])
                var sqrt_n = sqrt(total_n)
                var best_a = -1
                var best_puct: Float64 = -1e18

                for a in range(ACT):
                    var p = node_prior[node * ACT + a]
                    if p <= 0:
                        continue
                    var n_a = Float64(visit_count[node * ACT + a])
                    var q: Float64 = 0.0
                    if n_a > 0:
                        q = total_value[node * ACT + a] / n_a
                    var puct = q + C_PUCT * p * sqrt_n / (1.0 + n_a)
                    if puct > best_puct:
                        best_puct = puct
                        best_a = a

                if best_a < 0:
                    break

                path.append(node)
                path_actions.append(best_a)

                var ci = child_idx[node * ACT + best_a]
                if ci < 0:
                    # Restore parent state and step
                    var parent_state = node_states + node * E.SAVE_SIZE
                    env.load_env_state(
                        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                            parent_state
                        )
                    )
                    _ = env.step(env.action_from_index(best_a))
                    var game_result = env.game_result()

                    var leaf_value: Float64
                    if game_result != 0:
                        # Terminal node
                        if game_result == 3:
                            leaf_value = 0.0
                        else:
                            var moved_player = 1 - env.current_player()
                            leaf_value = (
                                1.0 if game_result == moved_player + 1 else -1.0
                            )
                    else:
                        # Expand: create child node
                        var ci_new = node_count
                        node_count += 1
                        child_idx[node * ACT + best_a] = ci_new
                        node_visits[ci_new] = 1

                        # Save child env state
                        var child_state = node_states + ci_new * E.SAVE_SIZE
                        env.save_env_state(
                            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                                child_state
                            )
                        )

                        # Network forward on child observation
                        var child_obs_raw = env.get_obs_list()
                        var child_obs = alloc[Scalar[dtype]](OBS)
                        for i in range(OBS):
                            child_obs[i] = Scalar[dtype](
                                child_obs_raw[i]
                            ) if i < len(child_obs_raw) else Scalar[dtype](0.0)
                        var c_obs_t = LayoutTensor[
                            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
                        ](child_obs)
                        var c_pred_ptr = alloc[Scalar[dtype]](PRED_OUT_DIM)
                        memset(c_pred_ptr, 0, PRED_OUT_DIM)
                        var c_pred_t = LayoutTensor[
                            dtype,
                            Layout.row_major(B, PRED_OUT_DIM),
                            MutAnyOrigin,
                        ](c_pred_ptr)
                        Self.PredNet.forward[B](
                            c_obs_t,
                            c_pred_t,
                            self.state.prediction.params_view(),
                        )

                        # Child prior (softmax over legal)
                        var child_legal = env.legal_action_mask()
                        var c_max: Float64 = -1e18
                        for a2 in range(ACT):
                            var l2 = Float64(
                                rebind[Scalar[dtype]](c_pred_t[0, a2])
                            )
                            if (
                                a2 < len(child_legal)
                                and child_legal[a2]
                                and l2 > c_max
                            ):
                                c_max = l2
                        var c_sum: Float64 = 0.0
                        for a2 in range(ACT):
                            if a2 < len(child_legal) and child_legal[a2]:
                                node_prior[ci_new * ACT + a2] = exp(
                                    Float64(
                                        rebind[Scalar[dtype]](c_pred_t[0, a2])
                                    )
                                    - c_max
                                )
                                c_sum += node_prior[ci_new * ACT + a2]
                            else:
                                node_prior[ci_new * ACT + a2] = 0.0
                        if c_sum > 0:
                            for a2 in range(ACT):
                                node_prior[ci_new * ACT + a2] /= c_sum

                        # Value from tanh output
                        var raw_v = Float64(
                            rebind[Scalar[dtype]](c_pred_t[0, ACT])
                        )
                        if raw_v > 15.0:
                            leaf_value = 1.0
                        elif raw_v < -15.0:
                            leaf_value = -1.0
                        else:
                            var ev = exp(2.0 * raw_v)
                            leaf_value = (ev - 1.0) / (ev + 1.0)

                        child_obs.free()
                        c_pred_ptr.free()

                    # Backup with negation (two-player)
                    var v = leaf_value
                    for p_idx in range(len(path) - 1, -1, -1):
                        v = -v
                        visit_count[
                            path[p_idx] * ACT + path_actions[p_idx]
                        ] += 1
                        total_value[
                            path[p_idx] * ACT + path_actions[p_idx]
                        ] += v
                        node_visits[path[p_idx]] += 1
                    found_leaf = True
                else:
                    # Restore child state and continue descent
                    var child_state = node_states + ci * E.SAVE_SIZE
                    env.load_env_state(
                        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                            child_state
                        )
                    )
                    node = ci

        # Restore env to root state
        env.load_env_state(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](root_save)
        )

        # Pick action with highest visit count
        var best_action = -1
        var best_visits = -1
        for a in range(ACT):
            if visit_count[a] > best_visits:
                best_visits = visit_count[a]
                best_action = a

        prior.free()
        visit_count.free()
        total_value.free()
        child_idx.free()
        node_prior.free()
        node_visits.free()
        node_states.free()
        root_save.free()

        if best_action < 0:
            for a in range(ACT):
                if a < len(legal_mask) and legal_mask[a]:
                    return a
        return best_action

    # ══════════════════════════════════════════════════════════════
    # Evaluation
    # ══════════════════════════════════════════════════════════════

    def evaluate_against[
        E: TwoPlayerDiscreteEnv,
        EvalType: Evaluator,
    ](
        mut self,
        mut env: E,
        mut evaluator: EvalType,
        num_games: Int = 50,
    ) -> Tuple[Int, Int, Int]:
        """Play agent vs evaluator. Returns (wins, draws, losses)."""
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        var wins = 0
        var draws = 0
        var losses = 0

        for game_idx in range(num_games):
            var agent_is_p0 = game_idx < num_games // 2
            _ = env.reset()
            evaluator.reset()

            while env.game_result() == 0:
                var player = env.current_player()
                var is_agent = (player == 0 and agent_is_p0) or (
                    player == 1 and not agent_is_p0
                )
                var legal = env.legal_action_mask()
                var action: Int

                if is_agent:
                    var obs = List[Scalar[dtype]](capacity=OBS)
                    var obs_raw = env.get_obs_list()
                    for i in range(OBS):
                        if i < len(obs_raw):
                            obs.append(Scalar[dtype](obs_raw[i]))
                        else:
                            obs.append(Scalar[dtype](0.0))
                    action = self.select_action(obs, legal)
                else:
                    action = evaluator.select_action(legal, ACT)

                if action < 0 or action >= ACT or not legal[action]:
                    for a in range(ACT):
                        if legal[a]:
                            action = a
                            break
                evaluator.observe_action(action, player)
                _ = env.step(env.action_from_index(action))

            var result = env.game_result()
            if result == 1:
                if agent_is_p0:
                    wins += 1
                else:
                    losses += 1
            elif result == 2:
                if agent_is_p0:
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1

        return (wins, draws, losses)

    def print_eval[
        E: TwoPlayerDiscreteEnv, EvalType: Evaluator
    ](mut self, mut env: E, mut evaluator: EvalType, num_games: Int = 50,):
        var r = self.evaluate_against[E, EvalType](env, evaluator, num_games)
        print(
            "  vs",
            evaluator.name(),
            "| W:",
            r[0],
            "D:",
            r[1],
            "L:",
            r[2],
            "| Win%:",
            r[0] * 100 // num_games,
            "Draw%:",
            r[1] * 100 // num_games,
        )

    # ══════════════════════════════════════════════════════════════
    # Checkpointing
    # ══════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: String) raises:
        comptime PS = Self.Config.PredModel.PARAM_SIZE
        comptime SS = PS * Self.Config.OptType.STATE_PER_PARAM
        var content = write_checkpoint_header("alphazero_agent", PS, SS)
        content += self.state.prediction.write_sections("pred_")
        var metadata = List[String]()
        metadata.append("config_name=" + Self.Config.NAME)
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("total_steps=" + String(self.total_steps))
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises:
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)
        self.state.prediction.read_sections(content, "pred_")
        var metadata = read_metadata_section(content)
        var s1 = get_metadata_value(metadata, "train_step_count")
        if len(s1) > 0:
            self.train_step_count = Int(atol(s1))
        var s2 = get_metadata_value(metadata, "total_steps")
        if len(s2) > 0:
            self.total_steps = Int(atol(s2))

    def start_new_iteration(mut self):
        """Mark the start of a new self-play iteration.

        Evicts oldest iteration data when history_window is exceeded.
        Call this before each self-play collection phase.
        """
        self.state.start_new_iteration()

    # ══════════════════════════════════════════════════════════════
    # GPU Training Step (simple supervised learning)
    # ══════════════════════════════════════════════════════════════

    def train_step_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
        diag_pred_host: HostBuffer[dtype],
        diag_go_host: HostBuffer[dtype],
        diag_params_host: HostBuffer[dtype],
        diag_grads_host: HostBuffer[dtype],
    ) raises:
        """One training step: sample batch from CPU replay, train on GPU.

        loss = CE(policy, mcts_π) + MSE(value, outcome)

        Diagnostic host buffers are pre-allocated by the caller
        and reused across steps to avoid memory bloat.
        """
        comptime BATCH = Self.Config.batch_size
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ── Sample random batch from CPU replay ──────────────────
        for b in range(BATCH):
            var idx = Int(random_float64() * Float64(self.state.buf_size))
            if idx >= self.state.buf_size:
                idx = self.state.buf_size - 1
            for i in range(OBS):
                gpu.obs_host[b * OBS + i] = self.state.buf_obs[idx * OBS + i]
            for i in range(ACT):
                gpu.policy_host[b * ACT + i] = self.state.buf_policy[
                    idx * ACT + i
                ]
            gpu.value_host[b] = self.state.buf_value[idx]

        # ── DEBUG: First training step — sample batch inspection ──
        if self.train_step_count == 0:
            print("╔══════════════════════════════════════════╗")
            print("║  DEBUG: Training batch (first step)      ║")
            print("╚══════════════════════════════════════════╝")
            # Print 3 samples from batch
            for _db in range(min(3, BATCH)):
                var _db_idx = _db
                print("Sample", _db_idx, ":")
                # Value target
                print(
                    "  value_target:",
                    Float64(gpu.value_host[_db_idx]),
                )
                # Policy target
                print("  policy_target:", end="")
                for a in range(ACT):
                    print(
                        "",
                        Int(
                            Float64(
                                gpu.policy_host[_db_idx * ACT + a]
                            )
                            * 1000.0
                        )
                        / 1000.0,
                        end="",
                    )
                print()
                # Obs summary (plane sums)
                var _db_p0: Float64 = 0.0
                var _db_p1: Float64 = 0.0
                var _db_p2: Float64 = 0.0
                for i in range(42):
                    _db_p0 += Float64(
                        gpu.obs_host[_db_idx * OBS + i]
                    )
                    _db_p1 += Float64(
                        gpu.obs_host[_db_idx * OBS + 42 + i]
                    )
                    _db_p2 += Float64(
                        gpu.obs_host[_db_idx * OBS + 84 + i]
                    )
                print(
                    "  obs planes: my=",
                    _db_p0,
                    "opp=",
                    _db_p1,
                    "legal=",
                    _db_p2,
                )
            print("─" * 44)

        # ── Upload to GPU ────────────────────────────────────────
        ctx.enqueue_copy(gpu.batch_obs, gpu.obs_host)
        ctx.enqueue_copy(gpu.batch_policy, gpu.policy_host)
        ctx.enqueue_copy(gpu.batch_value, gpu.value_host)

        # ── Forward with cache ───────────────────────────────────
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin
        ](gpu.batch_obs.unsafe_ptr())
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
        ](gpu.pred_out.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
        ](gpu.pred_cache.unsafe_ptr())

        Self.PredNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            pred_t,
            gpu.prediction.params_view(),
            cache_t,
            gpu.workspace,
        )

        # ── Compute gradient on GPU ──────────────────────────────
        var grad_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin
        ](gpu.grad_out.unsafe_ptr())
        var pred_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin
        ](gpu.pred_out.unsafe_ptr())
        var pol_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
        ](gpu.batch_policy.unsafe_ptr())
        var val_1d = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu.batch_value.unsafe_ptr()
        )

        comptime run_grad = az_policy_value_grad_kernel[
            BATCH, ACT, Self.PRED_OUT, dtype
        ]
        ctx.enqueue_function[run_grad, run_grad](
            grad_1d,
            pred_1d,
            pol_1d,
            val_1d,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── Backward + optimizer ─────────────────────────────────
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
        ](gpu.grad_out.unsafe_ptr())
        ctx.enqueue_memset(gpu.grad_in, 0)
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin
        ](gpu.grad_in.unsafe_ptr())

        gpu.prediction.zero_grads(ctx)
        var grads = gpu.prediction.grads_view()
        Self.PredNet.backward_gpu[BATCH](
            ctx,
            grad_out_t,
            grad_in_t,
            gpu.prediction.params_view(),
            cache_t,
            grads,
            gpu.workspace,
        )
        # Log param delta across optimizer step (every diag_every steps)
        if (
            self.logger
            and self.train_step_count > 0
            and (
                self.diag_every <= 0
                or (self.train_step_count + 1) % self.diag_every == 0
            )
        ):
            try:
                comptime _PS = Self.Config.PredModel.PARAM_SIZE
                # Reuse pre-allocated buffers (diag_params_host for pre, diag_grads_host for post)
                ctx.enqueue_copy(diag_params_host, gpu.prediction.params_buf)
                ctx.synchronize()

                gpu.prediction.optimizer_step(ctx)

                ctx.enqueue_copy(diag_grads_host, gpu.prediction.params_buf)
                ctx.synchronize()

                var delta_norm: Float64 = 0.0
                var delta_max: Float64 = 0.0
                for i in range(_PS):
                    var d = Float64(diag_grads_host[i]) - Float64(
                        diag_params_host[i]
                    )
                    delta_norm += d * d
                    var ad = d if d > 0 else -d
                    if ad > delta_max:
                        delta_max = ad
                self.logger[].log_scalar(
                    "param_delta_norm",
                    sqrt(delta_norm) if delta_norm == delta_norm else 0.0,
                    self.train_step_count + 1,
                )
                self.logger[].log_scalar(
                    "param_delta_max", delta_max, self.train_step_count + 1
                )
            except:
                gpu.prediction.optimizer_step(ctx)
        else:
            gpu.prediction.optimizer_step(ctx)
        self.train_step_count += 1

        # ── Log training diagnostics ────────────────────────────────
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                # Reuse pre-allocated diagnostic buffers
                ctx.enqueue_copy(diag_pred_host, gpu.pred_out)
                ctx.synchronize()

                var step = self.train_step_count
                var batch_ploss: Float64 = 0.0
                var batch_vloss: Float64 = 0.0
                var batch_entropy: Float64 = 0.0
                var batch_vmean: Float64 = 0.0

                for b in range(BATCH):
                    var pred_off = b * Self.PRED_OUT

                    # Policy CE
                    var max_l: Float64 = -1e18
                    for a in range(ACT):
                        var l = Float64(diag_pred_host[pred_off + a])
                        if l > max_l:
                            max_l = l
                    var sum_e: Float64 = 0.0
                    for a in range(ACT):
                        sum_e += exp(
                            Float64(diag_pred_host[pred_off + a]) - max_l
                        )

                    var ce: Float64 = 0.0
                    var ent: Float64 = 0.0
                    for a in range(ACT):
                        var prob = (
                            exp(Float64(diag_pred_host[pred_off + a]) - max_l)
                            / sum_e
                        )
                        var target = Float64(gpu.policy_host[b * ACT + a])
                        if target > 1e-8 and prob > 1e-8:
                            ce -= target * log(prob)
                        if prob > 1e-8:
                            ent -= prob * log(prob)
                    batch_ploss += ce
                    batch_entropy += ent

                    # Value MSE
                    var raw_v = Float64(diag_pred_host[pred_off + ACT])
                    var tanh_v: Float64 = 0.0
                    if raw_v != raw_v:
                        pass  # NaN — leave tanh_v = 0
                    elif raw_v > 15.0:
                        tanh_v = 1.0
                    elif raw_v < -15.0:
                        tanh_v = -1.0
                    else:
                        var e2 = exp(2.0 * raw_v)
                        tanh_v = (e2 - 1.0) / (e2 + 1.0)
                    var target_v = Float64(gpu.value_host[b])
                    batch_vloss += (tanh_v - target_v) * (tanh_v - target_v)
                    batch_vmean += tanh_v

                var n = Float64(BATCH)
                var pl = batch_ploss / n
                var vl = batch_vloss / n
                var pe = batch_entropy / n
                var vm = batch_vmean / n

                # Clamp to avoid NaN/inf being silently dropped by logger
                if vl != vl or vl > 1e10:
                    vl = 0.0
                if vm != vm or vm > 1e10 or vm < -1e10:
                    vm = 0.0

                self.logger[].log_scalar("policy_ce", pl, step)
                self.logger[].log_scalar("policy_entropy", pe, step)
                self.logger[].log_scalar("value_mse", vl, step)
                self.logger[].log_scalar("value_mean", vm, step)

                # MCTS target entropy — are targets sharp or uniform?
                var target_entropy_sum: Float64 = 0.0
                var target_max_sum: Float64 = 0.0
                for b2 in range(BATCH):
                    var tent: Float64 = 0.0
                    var tmax: Float64 = 0.0
                    for a2 in range(ACT):
                        var tp = Float64(gpu.policy_host[b2 * ACT + a2])
                        if tp > 1e-8:
                            tent -= tp * log(tp)
                        if tp > tmax:
                            tmax = tp
                    target_entropy_sum += tent
                    target_max_sum += tmax
                self.logger[].log_scalar(
                    "target_entropy", target_entropy_sum / n, step
                )
                self.logger[].log_scalar(
                    "target_max_prob", target_max_sum / n, step
                )

                # Gradient diagnostics: reuse pre-allocated buffers
                comptime PS = Self.Config.PredModel.PARAM_SIZE
                ctx.enqueue_copy(diag_go_host, gpu.grad_out)
                ctx.enqueue_copy(diag_grads_host, gpu.prediction.grads_buf)
                ctx.enqueue_copy(diag_params_host, gpu.prediction.params_buf)
                ctx.synchronize()

                # Grad output norm (from az_policy_value_grad_kernel)
                var go_norm: Float64 = 0.0
                var go_max: Float64 = 0.0
                for i in range(BATCH * Self.PRED_OUT):
                    var g = Float64(diag_go_host[i])
                    if g == g:  # skip NaN
                        go_norm += g * g
                        var ag = g if g > 0 else -g
                        if ag > go_max:
                            go_max = ag
                var go_n = sqrt(go_norm) if go_norm == go_norm else 0.0
                self.logger[].log_scalar("grad_output_norm", go_n, step)
                self.logger[].log_scalar("grad_output_max", go_max, step)

                # Param gradient norm (after backward)
                var grad_norm: Float64 = 0.0
                var grad_max: Float64 = 0.0
                for i in range(PS):
                    var g = Float64(diag_grads_host[i])
                    if g == g:
                        grad_norm += g * g
                        var ag = g if g > 0 else -g
                        if ag > grad_max:
                            grad_max = ag
                var gn = sqrt(grad_norm) if grad_norm == grad_norm else 0.0
                self.logger[].log_scalar("grad_param_norm", gn, step)
                self.logger[].log_scalar("grad_param_max", grad_max, step)

                # Param norm (are params changing?)
                var param_norm: Float64 = 0.0
                for i in range(PS):
                    var p = Float64(diag_params_host[i])
                    if p == p:
                        param_norm += p * p
                var pn = sqrt(param_norm) if param_norm == param_norm else 0.0
                self.logger[].log_scalar("param_norm", pn, step)

                # Value target stats (what is the network trying to learn?)
                var vt_sum: Float64 = 0.0
                var vt_pos = 0
                var vt_neg = 0
                for b2 in range(BATCH):
                    var t = Float64(gpu.value_host[b2])
                    vt_sum += t
                    if t > 0.5:
                        vt_pos += 1
                    elif t < -0.5:
                        vt_neg += 1
                self.logger[].log_scalar("value_target_mean", vt_sum / n, step)
                self.logger[].log_scalar(
                    "value_target_pos_frac",
                    Float64(vt_pos) / n,
                    step,
                )
            except:
                pass

    # ══════════════════════════════════════════════════════════════
    # CUDA Graph Helpers (split train_step_gpu for capturable kernels)
    # ══════════════════════════════════════════════════════════════

    def _sample_and_upload_batch(
        mut self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
    ) raises:
        """Sample random batch from CPU replay buffer and upload to GPU.

        This is NOT CUDA-graph-capturable (CPU work + H2D copies).
        Must be called before each graph replay so GPU buffers have fresh data.
        """
        comptime BATCH = Self.Config.batch_size
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim

        for b in range(BATCH):
            var idx = Int(random_float64() * Float64(self.state.buf_size))
            if idx >= self.state.buf_size:
                idx = self.state.buf_size - 1
            for i in range(OBS):
                gpu.obs_host[b * OBS + i] = self.state.buf_obs[idx * OBS + i]
            for i in range(ACT):
                gpu.policy_host[b * ACT + i] = self.state.buf_policy[
                    idx * ACT + i
                ]
            gpu.value_host[b] = self.state.buf_value[idx]

        ctx.enqueue_copy(gpu.batch_obs, gpu.obs_host)
        ctx.enqueue_copy(gpu.batch_policy, gpu.policy_host)
        ctx.enqueue_copy(gpu.batch_value, gpu.value_host)

    def _gpu_train_kernels(
        self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
    ) raises:
        """Pure GPU kernel sequence for one training step.

        Contains ONLY GPU kernel enqueues — no CPU counters, no diagnostics,
        no ctx.synchronize(). Fully CUDA graph capturable.

        Includes GPU-side sampling: increment RNG → sample indices → gather batch.
        """
        comptime BATCH = Self.Config.batch_size
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime CAP = Self.Config.buffer_capacity

        # ── GPU-side sampling ────────────────────────────────────
        # 1. Increment RNG counter (fresh seed each replay)
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu.rng_counter.unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        # 2. Sample random indices
        var idx_t = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.sample_indices.unsafe_ptr())
        var buf_sz_t = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](gpu.replay_size.unsafe_ptr())
        comptime sample_k = sample_indices_kernel[dtype, BATCH]
        ctx.enqueue_function[sample_k, sample_k](
            idx_t,
            buf_sz_t,
            rng_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 3. Gather batch from GPU replay buffer
        var b_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu.batch_obs.unsafe_ptr())
        var b_pol_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
        ](gpu.batch_policy.unsafe_ptr())
        var b_val_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.batch_value.unsafe_ptr())
        var r_obs_t = LayoutTensor[
            dtype, Layout.row_major(CAP, OBS), MutAnyOrigin
        ](gpu.replay_obs.unsafe_ptr())
        var r_pol_t = LayoutTensor[
            dtype, Layout.row_major(CAP, ACT), MutAnyOrigin
        ](gpu.replay_policy.unsafe_ptr())
        var r_val_t = LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin](
            gpu.replay_value.unsafe_ptr()
        )
        comptime gather_k = az_gather_batch_kernel[dtype, BATCH, OBS, ACT, CAP]
        ctx.enqueue_function[gather_k, gather_k](
            b_obs_t,
            b_pol_t,
            b_val_t,
            r_obs_t,
            r_pol_t,
            r_val_t,
            idx_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── Forward with cache ───────────────────────────────────
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin
        ](gpu.batch_obs.unsafe_ptr())
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
        ](gpu.pred_out.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
        ](gpu.pred_cache.unsafe_ptr())

        Self.PredNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            pred_t,
            gpu.prediction.params_view(),
            cache_t,
            gpu.workspace,
        )

        # Compute gradient on GPU
        var grad_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin
        ](gpu.grad_out.unsafe_ptr())
        var pred_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin
        ](gpu.pred_out.unsafe_ptr())
        var pol_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
        ](gpu.batch_policy.unsafe_ptr())
        var val_1d = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu.batch_value.unsafe_ptr()
        )

        comptime run_grad = az_policy_value_grad_kernel[
            BATCH, ACT, Self.PRED_OUT, dtype
        ]
        ctx.enqueue_function[run_grad, run_grad](
            grad_1d,
            pred_1d,
            pol_1d,
            val_1d,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backward + optimizer
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
        ](gpu.grad_out.unsafe_ptr())
        ctx.enqueue_memset(gpu.grad_in, 0)
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin
        ](gpu.grad_in.unsafe_ptr())

        gpu.prediction.zero_grads(ctx)
        var grads = gpu.prediction.grads_view()
        Self.PredNet.backward_gpu[BATCH](
            ctx,
            grad_out_t,
            grad_in_t,
            gpu.prediction.params_view(),
            cache_t,
            grads,
            gpu.workspace,
        )
        gpu.prediction.optimizer_step(ctx)

    def _gpu_train_diagnostics(
        mut self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
        num_steps: Int,
        diag_pred_host: HostBuffer[dtype],
        diag_go_host: HostBuffer[dtype],
        diag_params_host: HostBuffer[dtype],
        diag_grads_host: HostBuffer[dtype],
    ) raises:
        """CPU-side diagnostics after CUDA graph training replays.

        Increments counters and logs diagnostics for `num_steps` train steps.
        Loops per step to not miss diag_every boundaries.
        """
        var should_diag = False
        for _ in range(num_steps):
            self.train_step_count += 1
            if (
                self.logger
                and self.diag_every > 0
                and self.train_step_count % self.diag_every == 0
            ):
                should_diag = True

        if not should_diag:
            return

        comptime BATCH = Self.Config.batch_size
        comptime ACT = Self.Config.action_dim
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM

        try:
            # Download last batch targets (GPU sampling doesn't fill host buffers)
            ctx.enqueue_copy(gpu.policy_host, gpu.batch_policy)
            ctx.enqueue_copy(gpu.value_host, gpu.batch_value)
            # Reuse pre-allocated diagnostic buffers
            ctx.enqueue_copy(diag_pred_host, gpu.pred_out)
            ctx.synchronize()

            var step = self.train_step_count
            var batch_ploss: Float64 = 0.0
            var batch_vloss: Float64 = 0.0
            var batch_entropy: Float64 = 0.0
            var batch_vmean: Float64 = 0.0

            for b in range(BATCH):
                var pred_off = b * Self.PRED_OUT

                # Policy CE
                var max_l: Float64 = -1e18
                for a in range(ACT):
                    var l = Float64(diag_pred_host[pred_off + a])
                    if l > max_l:
                        max_l = l
                var sum_e: Float64 = 0.0
                for a in range(ACT):
                    sum_e += exp(Float64(diag_pred_host[pred_off + a]) - max_l)

                var ce: Float64 = 0.0
                var ent: Float64 = 0.0
                for a in range(ACT):
                    var prob = (
                        exp(Float64(diag_pred_host[pred_off + a]) - max_l)
                        / sum_e
                    )
                    var target = Float64(gpu.policy_host[b * ACT + a])
                    if target > 1e-8 and prob > 1e-8:
                        ce -= target * log(prob)
                    if prob > 1e-8:
                        ent -= prob * log(prob)
                batch_ploss += ce
                batch_entropy += ent

                # Value MSE
                var raw_v = Float64(diag_pred_host[pred_off + ACT])
                var tanh_v: Float64 = 0.0
                if raw_v != raw_v:
                    pass  # NaN — leave tanh_v = 0
                elif raw_v > 15.0:
                    tanh_v = 1.0
                elif raw_v < -15.0:
                    tanh_v = -1.0
                else:
                    var e2 = exp(2.0 * raw_v)
                    tanh_v = (e2 - 1.0) / (e2 + 1.0)
                var target_v = Float64(gpu.value_host[b])
                batch_vloss += (tanh_v - target_v) * (tanh_v - target_v)
                batch_vmean += tanh_v

            var n = Float64(BATCH)
            var pl = batch_ploss / n
            var vl = batch_vloss / n
            var pe = batch_entropy / n
            var vm = batch_vmean / n

            # Clamp to avoid NaN/inf being silently dropped by logger
            if vl != vl or vl > 1e10:
                vl = 0.0
            if vm != vm or vm > 1e10 or vm < -1e10:
                vm = 0.0

            self.logger[].log_scalar("policy_ce", pl, step)
            self.logger[].log_scalar("policy_entropy", pe, step)
            self.logger[].log_scalar("value_mse", vl, step)
            self.logger[].log_scalar("value_mean", vm, step)

            # MCTS target entropy
            var target_entropy_sum: Float64 = 0.0
            var target_max_sum: Float64 = 0.0
            for b2 in range(BATCH):
                var tent: Float64 = 0.0
                var tmax: Float64 = 0.0
                for a2 in range(ACT):
                    var tp = Float64(gpu.policy_host[b2 * ACT + a2])
                    if tp > 1e-8:
                        tent -= tp * log(tp)
                    if tp > tmax:
                        tmax = tp
                target_entropy_sum += tent
                target_max_sum += tmax
            self.logger[].log_scalar(
                "target_entropy", target_entropy_sum / n, step
            )
            self.logger[].log_scalar(
                "target_max_prob", target_max_sum / n, step
            )

            # Gradient diagnostics
            comptime PS = Self.Config.PredModel.PARAM_SIZE
            ctx.enqueue_copy(diag_go_host, gpu.grad_out)
            ctx.enqueue_copy(diag_grads_host, gpu.prediction.grads_buf)
            ctx.enqueue_copy(diag_params_host, gpu.prediction.params_buf)
            ctx.synchronize()

            var go_norm: Float64 = 0.0
            var go_max: Float64 = 0.0
            for i in range(BATCH * Self.PRED_OUT):
                var g = Float64(diag_go_host[i])
                if g == g:
                    go_norm += g * g
                    var ag = g if g > 0 else -g
                    if ag > go_max:
                        go_max = ag
            var go_n = sqrt(go_norm) if go_norm == go_norm else 0.0
            self.logger[].log_scalar("grad_output_norm", go_n, step)
            self.logger[].log_scalar("grad_output_max", go_max, step)

            var grad_norm: Float64 = 0.0
            var grad_max: Float64 = 0.0
            for i in range(PS):
                var g = Float64(diag_grads_host[i])
                if g == g:
                    grad_norm += g * g
                    var ag = g if g > 0 else -g
                    if ag > grad_max:
                        grad_max = ag
            var gn = sqrt(grad_norm) if grad_norm == grad_norm else 0.0
            self.logger[].log_scalar("grad_param_norm", gn, step)
            self.logger[].log_scalar("grad_param_max", grad_max, step)

            var param_norm: Float64 = 0.0
            for i in range(PS):
                var p = Float64(diag_params_host[i])
                if p == p:
                    param_norm += p * p
            var pn = sqrt(param_norm) if param_norm == param_norm else 0.0
            self.logger[].log_scalar("param_norm", pn, step)

            var vt_sum: Float64 = 0.0
            var vt_pos = 0
            var vt_neg = 0
            for b2 in range(BATCH):
                var t = Float64(gpu.value_host[b2])
                vt_sum += t
                if t > 0.5:
                    vt_pos += 1
                elif t < -0.5:
                    vt_neg += 1
            self.logger[].log_scalar("value_target_mean", vt_sum / n, step)
            self.logger[].log_scalar(
                "value_target_pos_frac",
                Float64(vt_pos) / n,
                step,
            )
        except:
            pass

    # ══════════════════════════════════════════════════════════════
    # MCTS Round Kernels (CUDA graph capturable)
    # ══════════════════════════════════════════════════════════════

    def _mcts_round_kernels[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
    ](
        self,
        ctx: DeviceContext,
        gpu: Self.GPUStateType,
        mut gpu_mcts: GPUMCTSState[
            Self.n_envs,
            Self.Config.max_nodes,
            Self.Config.action_dim,
            Self.Config.obs_dim,
            1,
            E.STATE_SIZE,
            Self.Config.batch_sims,
        ],
        mut exp_rewards: DeviceBuffer[dtype],
        mut exp_dones: DeviceBuffer[dtype],
        mut exp_terminated: DeviceBuffer[dtype],
        mut exp_obs: DeviceBuffer[dtype],
        mut mcts_ws: DeviceBuffer[dtype],
        # MCTS tree LayoutTensor views (pre-created by caller)
        vc: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * Self.Config.action_dim
            ),
            MutAnyOrigin,
        ],
        tv: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * Self.Config.action_dim
            ),
            MutAnyOrigin,
        ],
        pr: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * Self.Config.action_dim
            ),
            MutAnyOrigin,
        ],
        rw: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * Self.Config.action_dim
            ),
            MutAnyOrigin,
        ],
        ci: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * Self.Config.action_dim
            ),
            MutAnyOrigin,
        ],
        tvis: LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * Self.Config.max_nodes),
            MutAnyOrigin,
        ],
        nc: LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin],
        miq: LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin],
        mxq: LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin],
        gs: LayoutTensor[
            dtype,
            Layout.row_major(
                Self.n_envs * Self.Config.max_nodes * E.STATE_SIZE
            ),
            MutAnyOrigin,
        ],
        # Batched simulation buffers
        b_pp: LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * Self.Config.batch_sims), MutAnyOrigin
        ],
        b_pa: LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * Self.Config.batch_sims), MutAnyOrigin
        ],
        b_exp_st: LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * Self.Config.batch_sims * E.STATE_SIZE),
            MutAnyOrigin,
        ],
        b_sp: LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * Self.Config.batch_sims * MAX_DEPTH),
            MutAnyOrigin,
        ],
        b_ap: LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * Self.Config.batch_sims * MAX_DEPTH),
            MutAnyOrigin,
        ],
        b_pl: LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * Self.Config.batch_sims), MutAnyOrigin
        ],
    ) raises:
        """One MCTS simulation round: select → env.step → predict → backup.

        Contains ONLY GPU kernel enqueues — fully CUDA graph capturable.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime MAX_NODES = Self.Config.max_nodes
        comptime GS = E.STATE_SIZE
        comptime BATCH_SIMS = Self.Config.batch_sims
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime MCTS_PRED_OUT = ACT + 1
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB

        # 1. Fused select + copy
        comptime run_sel_cp = gpu_mcts_batched_select_and_copy_kernel[
            Self.n_envs, MAX_NODES, ACT, BATCH_SIMS, GS, dtype
        ]
        ctx.enqueue_function[run_sel_cp, run_sel_cp](
            vc,
            tv,
            pr,
            ci,
            tvis,
            nc,
            miq,
            mxq,
            gs,
            b_pp,
            b_pa,
            b_exp_st,
            b_sp,
            b_ap,
            b_pl,
            Scalar[dtype](Self.Config.PUCT.C_BASE),
            Scalar[dtype](Self.Config.PUCT.C_INIT),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # 2. Batched env.step
        E.step_kernel_gpu[TOTAL_EXPAND, GS, OBS](
            ctx,
            gpu_mcts.expansion_states,
            gpu_mcts.pending_action,
            exp_rewards,
            exp_dones,
            exp_terminated,
            exp_obs,
            gpu_mcts.expansion_legal_masks,
            rng_seed=UInt64(0),  # Fixed seed (graph-safe)
        )

        # 3. Batched prediction
        var p_in = LayoutTensor[
            dtype,
            Layout.row_major(TOTAL_EXPAND, PRED_IN),
            MutAnyOrigin,
        ](exp_obs.unsafe_ptr())
        var p_out = LayoutTensor[
            dtype,
            Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM),
            MutAnyOrigin,
        ](gpu_mcts.pred_output.unsafe_ptr())
        Self.PredNet.forward_gpu[TOTAL_EXPAND](
            ctx, p_in, p_out, gpu.prediction.params_view(), mcts_ws
        )

        # 4. Fused expand + backup + remove virtual losses
        var b_po = LayoutTensor[
            dtype,
            Layout.row_major(TOTAL_EXPAND * MCTS_PRED_OUT),
            MutAnyOrigin,
        ](gpu_mcts.pred_output.unsafe_ptr())
        var b_rew = LayoutTensor[
            dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
        ](exp_rewards.unsafe_ptr())
        comptime run_exp_bk = gpu_mcts_batched_expand_backup_kernel[
            Self.n_envs,
            MAX_NODES,
            ACT,
            BATCH_SIMS,
            MCTS_PRED_OUT,
            GS,
            dtype,
        ]
        ctx.enqueue_function[run_exp_bk, run_exp_bk](
            vc,
            tv,
            pr,
            rw,
            ci,
            tvis,
            nc,
            miq,
            mxq,
            gs,
            b_exp_st,
            b_pp,
            b_pa,
            b_po,
            b_rew,
            b_sp,
            b_ap,
            b_pl,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════
    # Data Augmentation (via DataAugmentable trait on environment)
    # ══════════════════════════════════════════════════════════════

    def _add_with_augmentation[
        AugEnv: DataAugmentable,
    ](
        mut self,
        obs: List[Scalar[dtype]],
        policy: List[Scalar[dtype]],
        value: Scalar[dtype],
    ):
        """Add training sample + all symmetries from the environment.

        Uses the DataAugmentable trait to generate augmented samples.
        The environment knows its own symmetries (rotations, reflections).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim

        for s in range(AugEnv.NUM_SYMMETRIES):
            var sym_obs = alloc[Scalar[dtype]](OBS)
            var sym_pol = alloc[Scalar[dtype]](ACT)

            var obs_in = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                obs.unsafe_ptr()
            )
            var pol_in = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                policy.unsafe_ptr()
            )
            var obs_out = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                sym_obs
            )
            var pol_out = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                sym_pol
            )

            AugEnv.augment_obs[OBS](obs_in, s, obs_out)
            AugEnv.augment_policy[ACT](pol_in, s, pol_out)

            self.state.add(
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](sym_obs),
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](sym_pol),
                value,
            )

            sym_obs.free()
            sym_pol.free()

    # ══════════════════════════════════════════════════════════════
    # Arena Comparison (accept/reject new model)
    # ══════════════════════════════════════════════════════════════

    def arena_compare[
        E: TwoPlayerDiscreteEnv,
        origin: MutOrigin,
    ](
        mut self,
        mut env: E,
        mut prev_params: UnsafePointer[Scalar[dtype], origin],
        num_games: Int = 40,
        threshold: Float64 = 0.55,
    ) -> Bool:
        """Play current model vs previous model. Accept if score >= threshold.

        Uses policy-only action selection for speed.
        Draws count as 0.5 for each player (Elo-style scoring):
          score = (wins + 0.5 * draws) / total_games

        This prevents the trap where two equally strong models that
        draw most games always reject the new model.

        Args:
            env: Environment for playing games.
            prev_params: Saved parameters of the previous best model.
            num_games: Games to play (half as each side).
            threshold: Score fraction needed to accept new model.

        Returns:
            True if new model is accepted.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime PS = Self.Config.PredModel.PARAM_SIZE

        # Save current (new) params
        var new_params = alloc[Scalar[dtype]](PS)
        for i in range(PS):
            new_params[i] = self.state.prediction.params[i]

        var new_wins = 0
        var draws = 0

        for game_idx in range(num_games):
            var new_is_p0 = game_idx < num_games // 2
            _ = env.reset()

            while env.game_result() == 0:
                var player = env.current_player()
                var is_new = (player == 0 and new_is_p0) or (
                    player == 1 and not new_is_p0
                )

                # Load appropriate params
                if is_new:
                    for i in range(PS):
                        self.state.prediction.params[i] = new_params[i]
                else:
                    for i in range(PS):
                        self.state.prediction.params[i] = prev_params[i]

                var legal = env.legal_action_mask()
                var obs_raw = env.get_obs_list()
                var obs = List[Scalar[dtype]](capacity=OBS)
                for i in range(OBS):
                    if i < len(obs_raw):
                        obs.append(Scalar[dtype](obs_raw[i]))
                    else:
                        obs.append(Scalar[dtype](0.0))

                var action = self.select_action(obs, legal)
                if action < 0 or action >= ACT or not legal[action]:
                    for a in range(ACT):
                        if legal[a]:
                            action = a
                            break
                _ = env.step(env.action_from_index(action))

            var result = env.game_result()
            if result == 3:
                draws += 1
            elif result == 1:
                if new_is_p0:
                    new_wins += 1
            elif result == 2:
                if not new_is_p0:
                    new_wins += 1

        # Restore new params (we'll decide whether to keep or revert)
        for i in range(PS):
            self.state.prediction.params[i] = new_params[i]
        new_params.free()

        # Like alpha-zero-general: draws excluded from denominator
        # win_rate = new_wins / (new_wins + old_wins), ignoring draws
        var old_wins = num_games - new_wins - draws
        var decisive = new_wins + old_wins
        var accepted: Bool
        if decisive == 0:
            accepted = False  # All draws — no evidence new is better
        else:
            var win_rate = Float64(new_wins) / Float64(decisive)
            accepted = win_rate >= threshold

        if not accepted:
            # Revert to previous params
            for i in range(PS):
                self.state.prediction.params[i] = prev_params[i]

        return accepted

    def _gpu_play_games[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
    ](
        mut self,
        ctx: DeviceContext,
        p0_params: DeviceBuffer[dtype],
        p1_params: DeviceBuffer[dtype],
        mut gpu_mcts: GPUMCTSState[
            Self.n_envs,
            Self.Config.max_nodes,
            Self.Config.action_dim,
            Self.Config.obs_dim,
            1,
            E.STATE_SIZE,
            Self.Config.batch_sims,
        ],
        mut mcts_ws: DeviceBuffer[dtype],
        mut states_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut legal_masks_buf: DeviceBuffer[dtype],
        mut exp_rewards: DeviceBuffer[dtype],
        mut exp_dones: DeviceBuffer[dtype],
        mut exp_terminated: DeviceBuffer[dtype],
        mut exp_obs: DeviceBuffer[dtype],
        mut rewards_host: HostBuffer[dtype],
        mut dones_host: HostBuffer[dtype],
        rng_offset: Int,
    ) raises -> Tuple[Int, Int, Int]:
        """Play n_envs games on GPU. P0 uses p0_params, P1 uses p1_params.

        Uses pre-allocated buffers (no per-call GPU allocation).
        Returns (p0_wins, draws, p1_wins).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime MAX_NODES = Self.Config.max_nodes
        comptime GS = E.STATE_SIZE
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime MCTS_PRED_OUT = ACT + 1
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime SIMS = Self.Config.num_simulations
        comptime BATCH_SIMS = Self.Config.batch_sims
        comptime NUM_ROUNDS = SIMS // BATCH_SIMS
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

        # Reset envs
        E.reset_kernel_gpu[Self.n_envs, GS](
            ctx, states_buf, rng_seed=UInt64(rng_offset)
        )
        E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
            ctx, states_buf, obs_buf, legal_masks_buf
        )
        ctx.synchronize()

        var game_done = InlineArray[Bool, Self.n_envs](fill=False)
        var game_result = InlineArray[Int, Self.n_envs](fill=0)
        var move_num = 0
        var all_done = False

        comptime MAX_ARENA_MOVES = ACT * ACT
        while not all_done and move_num < MAX_ARENA_MOVES:
            var p0_turn = move_num % 2 == 0

            # Pick params: even moves → P0, odd moves → P1
            comptime _PS = Self.Config.PredModel.PARAM_SIZE
            var active_params = LayoutTensor[
                dtype, Layout.row_major(_PS), MutAnyOrigin
            ](
                p0_params.unsafe_ptr()
                if p0_turn
                else p1_params.unsafe_ptr()
            )

            # LayoutTensor views over MCTS buffers
            var pred_obs = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs, PRED_IN), MutAnyOrigin
            ](obs_buf.unsafe_ptr())
            var pred_out = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs, PRED_OUT_DIM), MutAnyOrigin
            ](gpu_mcts.pred_output.unsafe_ptr())
            var vc = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                MutAnyOrigin,
            ](gpu_mcts.visit_count.unsafe_ptr())
            var tv = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                MutAnyOrigin,
            ](gpu_mcts.total_value.unsafe_ptr())
            var pr = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                MutAnyOrigin,
            ](gpu_mcts.prior.unsafe_ptr())
            var rw = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                MutAnyOrigin,
            ](gpu_mcts.reward.unsafe_ptr())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                MutAnyOrigin,
            ](gpu_mcts.child_idx.unsafe_ptr())
            var tvis = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * MAX_NODES), MutAnyOrigin
            ](gpu_mcts.total_visits.unsafe_ptr())
            var nc = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](gpu_mcts.node_count.unsafe_ptr())
            var po = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MCTS_PRED_OUT),
                MutAnyOrigin,
            ](gpu_mcts.pred_output.unsafe_ptr())
            var miq = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](gpu_mcts.min_q.unsafe_ptr())
            var mxq = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](gpu_mcts.max_q.unsafe_ptr())
            var lm = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
            ](legal_masks_buf.unsafe_ptr())
            var gs = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * MAX_NODES * GS),
                MutAnyOrigin,
            ](gpu_mcts.game_states.unsafe_ptr())
            var es = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * GS), MutAnyOrigin
            ](states_buf.unsafe_ptr())

            # Forward pass → prediction output
            Self.PredNet.forward_gpu[Self.n_envs](
                ctx, pred_obs, pred_out, active_params, mcts_ws
            )

            # Zero tree data via bulk memset, then init root
            gpu_mcts.zero_tree(ctx)
            comptime run_init = gpu_mcts_init_root_kernel[
                Self.n_envs, MAX_NODES, ACT, OBS, MCTS_PRED_OUT, dtype
            ]
            ctx.enqueue_function[run_init, run_init](
                vc,
                tv,
                pr,
                rw,
                ci,
                tvis,
                nc,
                po,
                miq,
                mxq,
                Scalar[dtype](0.0),  # No noise in arena
                Scalar[DType.uint32](UInt32(move_num)),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # Apply legal mask
            comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                Self.n_envs, MAX_NODES, ACT, dtype
            ]
            ctx.enqueue_function[run_mask, run_mask](
                pr, lm, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )

            # Copy root game states
            comptime run_rs = gpu_mcts_copy_root_state_kernel[
                Self.n_envs, MAX_NODES, GS, dtype
            ]
            ctx.enqueue_function[run_rs, run_rs](
                gs, es, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )

            # Batched MCTS simulations
            var b_pp = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
            ](gpu_mcts.pending_parent.unsafe_ptr())
            var b_pa = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
            ](gpu_mcts.pending_action.unsafe_ptr())
            var b_sp = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin
            ](gpu_mcts.search_paths.unsafe_ptr())
            var b_ap = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin
            ](gpu_mcts.action_paths.unsafe_ptr())
            var b_pl = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
            ](gpu_mcts.path_lengths.unsafe_ptr())
            var b_exp_st = LayoutTensor[
                dtype, Layout.row_major(TOTAL_EXPAND * GS), MutAnyOrigin
            ](gpu_mcts.expansion_states.unsafe_ptr())

            for _round in range(NUM_ROUNDS):
                comptime run_sel = gpu_mcts_batched_select_and_copy_kernel[
                    Self.n_envs, MAX_NODES, ACT, BATCH_SIMS, GS, dtype
                ]
                ctx.enqueue_function[run_sel, run_sel](
                    vc,
                    tv,
                    pr,
                    ci,
                    tvis,
                    nc,
                    miq,
                    mxq,
                    gs,
                    b_pp,
                    b_pa,
                    b_exp_st,
                    b_sp,
                    b_ap,
                    b_pl,
                    Scalar[dtype](Self.Config.PUCT.C_BASE),
                    Scalar[dtype](Self.Config.PUCT.C_INIT),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                E.step_kernel_gpu[TOTAL_EXPAND, GS, OBS](
                    ctx,
                    gpu_mcts.expansion_states,
                    gpu_mcts.pending_action,
                    exp_rewards,
                    exp_dones,
                    exp_terminated,
                    exp_obs,
                    gpu_mcts.expansion_legal_masks,
                    rng_seed=UInt64(
                        rng_offset + move_num * NUM_ROUNDS + _round
                    ),
                )

                var p_in = LayoutTensor[
                    dtype, Layout.row_major(TOTAL_EXPAND, PRED_IN), MutAnyOrigin
                ](exp_obs.unsafe_ptr())
                var p_out = LayoutTensor[
                    dtype,
                    Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM),
                    MutAnyOrigin,
                ](gpu_mcts.pred_output.unsafe_ptr())
                Self.PredNet.forward_gpu[TOTAL_EXPAND](
                    ctx, p_in, p_out, active_params, mcts_ws
                )

                var b_po = LayoutTensor[
                    dtype,
                    Layout.row_major(TOTAL_EXPAND * MCTS_PRED_OUT),
                    MutAnyOrigin,
                ](gpu_mcts.pred_output.unsafe_ptr())
                var b_rew = LayoutTensor[
                    dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                ](exp_rewards.unsafe_ptr())
                comptime run_exp = gpu_mcts_batched_expand_backup_kernel[
                    Self.n_envs,
                    MAX_NODES,
                    ACT,
                    BATCH_SIMS,
                    MCTS_PRED_OUT,
                    GS,
                    dtype,
                ]
                ctx.enqueue_function[run_exp, run_exp](
                    vc,
                    tv,
                    pr,
                    rw,
                    ci,
                    tvis,
                    nc,
                    miq,
                    mxq,
                    gs,
                    b_exp_st,
                    b_pp,
                    b_pa,
                    b_po,
                    b_rew,
                    b_sp,
                    b_ap,
                    b_pl,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Extract greedy actions (temp=0, matching alpha-zero-general arena)
            var act_out = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var pol_out = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
            ](gpu_mcts.policies_out.unsafe_ptr())
            comptime run_act = gpu_mcts_extract_actions_masked_kernel[
                Self.n_envs, MAX_NODES, ACT, dtype
            ]
            ctx.enqueue_function[run_act, run_act](
                vc,
                lm,
                act_out,
                pol_out,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # Env step
            E.step_kernel_gpu[Self.n_envs, GS, OBS](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                legal_masks_buf,
                rng_seed=UInt64(rng_offset + move_num + 1000),
            )

            # Check for completed games
            ctx.enqueue_copy(dones_host, dones_buf)
            ctx.enqueue_copy(rewards_host, rewards_buf)
            ctx.synchronize()

            for e in range(Self.n_envs):
                if not game_done[e] and Float64(dones_host[e]) > 0.5:
                    game_done[e] = True
                    var rew = Float64(rewards_host[e])
                    if rew > 0.5:
                        # Current player won
                        if p0_turn:
                            game_result[e] = 1  # P0 wins
                        else:
                            game_result[e] = 2  # P1 wins
                    elif rew < -0.5:
                        # Current player lost (illegal move penalty)
                        if p0_turn:
                            game_result[e] = 2  # P1 wins
                        else:
                            game_result[e] = 1  # P0 wins
                    else:
                        game_result[e] = 3  # Draw

            # Selective reset for completed games
            E.selective_reset_kernel_gpu[Self.n_envs, GS](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(rng_offset + move_num + 2000),
            )
            E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
                ctx, states_buf, obs_buf, legal_masks_buf
            )

            # Check if all done
            all_done = True
            for e in range(Self.n_envs):
                if not game_done[e]:
                    all_done = False
                    break

            move_num += 1

        # Tally results
        var p0_wins = 0
        var draws = 0
        var p1_wins = 0
        for e in range(Self.n_envs):
            var result = game_result[e]
            if result == 1:
                p0_wins += 1
            elif result == 2:
                p1_wins += 1
            elif result == 3:
                draws += 1
            # result == 0 means game didn't finish → count as draw
            else:
                draws += 1

        return (p0_wins, draws, p1_wins)

    def arena_compare_gpu[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
        origin: MutOrigin,
    ](
        mut self,
        ctx: DeviceContext,
        prev_params: UnsafePointer[Scalar[dtype], origin],
        prev_opt_state: UnsafePointer[Scalar[dtype], origin],
        # Pre-allocated arena param buffers
        mut arena_new_params: DeviceBuffer[dtype],
        mut arena_old_params: DeviceBuffer[dtype],
        mut arena_params_host: HostBuffer[dtype],
        # Pre-allocated shared buffers (reused from self-play)
        mut gpu_mcts: GPUMCTSState[
            Self.n_envs,
            Self.Config.max_nodes,
            Self.Config.action_dim,
            Self.Config.obs_dim,
            1,
            E.STATE_SIZE,
            Self.Config.batch_sims,
        ],
        mut mcts_ws: DeviceBuffer[dtype],
        mut states_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut legal_masks_buf: DeviceBuffer[dtype],
        mut exp_rewards: DeviceBuffer[dtype],
        mut exp_dones: DeviceBuffer[dtype],
        mut exp_terminated: DeviceBuffer[dtype],
        mut exp_obs: DeviceBuffer[dtype],
        mut rewards_host: HostBuffer[dtype],
        mut dones_host: HostBuffer[dtype],
        num_games: Int = 40,
        threshold: Float64 = 0.55,
    ) raises -> Tuple[Bool, Int, Int, Int]:
        """Compare current (new) model vs previous (old) model on GPU.

        Uses pre-allocated buffers (no per-call GPU allocation).
        Returns (accepted, new_wins, draws, old_wins).
        """
        comptime PS = Self.Config.PredModel.PARAM_SIZE

        # Upload new (current) params
        for i in range(PS):
            arena_params_host[i] = self.state.prediction.params[i]
        ctx.enqueue_copy(arena_new_params, arena_params_host)

        # Upload old (previous best) params
        for i in range(PS):
            arena_params_host[i] = prev_params[i]
        ctx.enqueue_copy(arena_old_params, arena_params_host)
        ctx.synchronize()

        # Phase 1: new=P0, old=P1
        var r1 = self._gpu_play_games[E](
            ctx,
            arena_new_params,
            arena_old_params,
            gpu_mcts,
            mcts_ws,
            states_buf,
            obs_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            legal_masks_buf,
            exp_rewards,
            exp_dones,
            exp_terminated,
            exp_obs,
            rewards_host,
            dones_host,
            rng_offset=42,
        )
        var new_wins = r1[0]
        var draws = r1[1]
        var old_wins = r1[2]

        # Phase 2: old=P0, new=P1
        var r2 = self._gpu_play_games[E](
            ctx,
            arena_old_params,
            arena_new_params,
            gpu_mcts,
            mcts_ws,
            states_buf,
            obs_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            legal_masks_buf,
            exp_rewards,
            exp_dones,
            exp_terminated,
            exp_obs,
            rewards_host,
            dones_host,
            rng_offset=9999,
        )
        new_wins += r2[2]
        draws += r2[1]
        old_wins += r2[0]

        # Acceptance decision (draws excluded from denominator)
        var decisive = new_wins + old_wins
        var accepted: Bool
        if decisive == 0:
            accepted = False  # All draws — no evidence new is better
        else:
            var win_rate = Float64(new_wins) / Float64(decisive)
            accepted = win_rate >= threshold

        if not accepted:
            # Revert params and optimizer state to previous best
            for i in range(PS):
                self.state.prediction.params[i] = prev_params[i]
            comptime OPT_STATE_SIZE = Self.Config.PredModel.PARAM_SIZE * Self.Config.OptType.STATE_PER_PARAM
            for i in range(OPT_STATE_SIZE):
                self.state.prediction.optimizer_state[i] = prev_opt_state[i]

        return (accepted, new_wins, draws, old_wins)

    # ══════════════════════════════════════════════════════════════
    # GPU Evaluation vs Random
    # ══════════════════════════════════════════════════════════════

    def gpu_eval[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
        Eval: GPUEvaluator,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
        mut eval_mcts: GPUMCTSState[
            Self.n_envs,
            Self.Config.max_nodes,
            Self.Config.action_dim,
            Self.Config.obs_dim,
            1,
            E.STATE_SIZE,
            Self.Config.batch_sims,
        ],
        mut mcts_ws: DeviceBuffer[dtype],
        mut eval_states: DeviceBuffer[dtype],
        mut eval_obs: DeviceBuffer[dtype],
        mut eval_acts: DeviceBuffer[dtype],
        mut eval_rews: DeviceBuffer[dtype],
        mut eval_dones: DeviceBuffer[dtype],
        mut eval_term: DeviceBuffer[dtype],
        mut eval_legal: DeviceBuffer[dtype],
        mut exp_rewards: DeviceBuffer[dtype],
        mut exp_dones: DeviceBuffer[dtype],
        mut exp_terminated: DeviceBuffer[dtype],
        mut exp_obs: DeviceBuffer[dtype],
        mut eval_dones_host: HostBuffer[dtype],
        mut eval_rews_host: HostBuffer[dtype],
        rng_offset: Int = 0,
    ) raises -> Tuple[Int, Int, Int]:
        """GPU evaluation: agent (MCTS temp=0) vs GPU evaluator.

        Uses pre-allocated buffers (no per-call GPU allocation).
        Returns (wins, draws, losses).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime MAX_NODES = Self.Config.max_nodes
        comptime GS = E.STATE_SIZE
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime MCTS_PRED_OUT = ACT + 1
        comptime BATCH_SIMS = Self.Config.batch_sims
        comptime NUM_ROUNDS = Self.Config.num_simulations // BATCH_SIMS
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

        E.reset_kernel_gpu[Self.n_envs, GS](
            ctx, eval_states, rng_seed=UInt64(rng_offset + 9999)
        )
        E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
            ctx, eval_states, eval_obs, eval_legal
        )
        ctx.synchronize()

        var eval_done = InlineArray[Bool, Self.n_envs](fill=False)
        var eval_result = InlineArray[Int, Self.n_envs](fill=0)
        var eval_move = 0
        var eval_all_done = False

        # Max moves: action_dim² is a safe upper bound for board games
        # (TTT=81, C4=49, Chess=~200). Must exceed max game length.
        comptime MAX_EVAL_MOVES = ACT * ACT
        while not eval_all_done and eval_move < MAX_EVAL_MOVES:
            var agent_turn = eval_move % 2 == 0

            if agent_turn:
                # Agent's turn: GPU MCTS (temp=0, no noise)
                var e_pred_obs = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_IN),
                    MutAnyOrigin,
                ](eval_obs.unsafe_ptr())
                var e_pred_out = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_OUT_DIM),
                    MutAnyOrigin,
                ](eval_mcts.pred_output.unsafe_ptr())
                Self.PredNet.forward_gpu[Self.n_envs](
                    ctx,
                    e_pred_obs,
                    e_pred_out,
                    gpu.prediction.params_view(),
                    mcts_ws,
                )

                var e_vc = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.visit_count.unsafe_ptr())
                var e_tv = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.total_value.unsafe_ptr())
                var e_pr = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.prior.unsafe_ptr())
                var e_rw = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.reward.unsafe_ptr())
                var e_ci = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.child_idx.unsafe_ptr())
                var e_tvis = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES),
                    MutAnyOrigin,
                ](eval_mcts.total_visits.unsafe_ptr())
                var e_nc = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ](eval_mcts.node_count.unsafe_ptr())
                var e_po = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MCTS_PRED_OUT),
                    MutAnyOrigin,
                ](eval_mcts.pred_output.unsafe_ptr())
                var e_miq = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ](eval_mcts.min_q.unsafe_ptr())
                var e_mxq = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ](eval_mcts.max_q.unsafe_ptr())
                var e_lm = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * ACT),
                    MutAnyOrigin,
                ](eval_legal.unsafe_ptr())
                var e_gs = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * GS),
                    MutAnyOrigin,
                ](eval_mcts.game_states.unsafe_ptr())
                var e_es = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * GS),
                    MutAnyOrigin,
                ](eval_states.unsafe_ptr())

                # Zero tree data via bulk memset, then init root
                eval_mcts.zero_tree(ctx)
                comptime e_run_init = gpu_mcts_init_root_kernel[
                    Self.n_envs,
                    MAX_NODES,
                    ACT,
                    OBS,
                    MCTS_PRED_OUT,
                    dtype,
                ]
                ctx.enqueue_function[e_run_init, e_run_init](
                    e_vc,
                    e_tv,
                    e_pr,
                    e_rw,
                    e_ci,
                    e_tvis,
                    e_nc,
                    e_po,
                    e_miq,
                    e_mxq,
                    Scalar[dtype](0.0),  # No noise
                    Scalar[DType.uint32](UInt32(rng_offset + eval_move)),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                comptime e_run_mask = gpu_mcts_apply_legal_mask_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                ctx.enqueue_function[e_run_mask, e_run_mask](
                    e_pr,
                    e_lm,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                comptime e_run_rs = gpu_mcts_copy_root_state_kernel[
                    Self.n_envs, MAX_NODES, GS, dtype
                ]
                ctx.enqueue_function[e_run_rs, e_run_rs](
                    e_gs,
                    e_es,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # MCTS simulations (reuse pre-allocated expansion buffers)
                comptime EVAL_TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

                var e_b_pp = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND),
                    MutAnyOrigin,
                ](eval_mcts.pending_parent.unsafe_ptr())
                var e_b_pa = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND),
                    MutAnyOrigin,
                ](eval_mcts.pending_action.unsafe_ptr())
                var e_b_sp = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND * MAX_DEPTH),
                    MutAnyOrigin,
                ](eval_mcts.search_paths.unsafe_ptr())
                var e_b_ap = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND * MAX_DEPTH),
                    MutAnyOrigin,
                ](eval_mcts.action_paths.unsafe_ptr())
                var e_b_pl = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND),
                    MutAnyOrigin,
                ](eval_mcts.path_lengths.unsafe_ptr())
                var e_b_exp = LayoutTensor[
                    dtype,
                    Layout.row_major(EVAL_TOTAL_EXPAND * GS),
                    MutAnyOrigin,
                ](eval_mcts.expansion_states.unsafe_ptr())

                for _r in range(NUM_ROUNDS):
                    comptime e_run_sel = gpu_mcts_batched_select_and_copy_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        BATCH_SIMS,
                        GS,
                        dtype,
                    ]
                    ctx.enqueue_function[e_run_sel, e_run_sel](
                        e_vc,
                        e_tv,
                        e_pr,
                        e_ci,
                        e_tvis,
                        e_nc,
                        e_miq,
                        e_mxq,
                        e_gs,
                        e_b_pp,
                        e_b_pa,
                        e_b_exp,
                        e_b_sp,
                        e_b_ap,
                        e_b_pl,
                        Scalar[dtype](Self.Config.PUCT.C_BASE),
                        Scalar[dtype](Self.Config.PUCT.C_INIT),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    E.step_kernel_gpu[EVAL_TOTAL_EXPAND, GS, OBS](
                        ctx,
                        eval_mcts.expansion_states,
                        eval_mcts.pending_action,
                        exp_rewards,
                        exp_dones,
                        exp_terminated,
                        exp_obs,
                        eval_mcts.expansion_legal_masks,
                        rng_seed=UInt64(rng_offset * 100 + eval_move * 10 + _r),
                    )
                    var ep_in = LayoutTensor[
                        dtype,
                        Layout.row_major(EVAL_TOTAL_EXPAND, PRED_IN),
                        MutAnyOrigin,
                    ](exp_obs.unsafe_ptr())
                    var ep_out = LayoutTensor[
                        dtype,
                        Layout.row_major(EVAL_TOTAL_EXPAND, PRED_OUT_DIM),
                        MutAnyOrigin,
                    ](eval_mcts.pred_output.unsafe_ptr())
                    Self.PredNet.forward_gpu[EVAL_TOTAL_EXPAND](
                        ctx,
                        ep_in,
                        ep_out,
                        gpu.prediction.params_view(),
                        mcts_ws,
                    )
                    var e_b_po = LayoutTensor[
                        dtype,
                        Layout.row_major(EVAL_TOTAL_EXPAND * MCTS_PRED_OUT),
                        MutAnyOrigin,
                    ](eval_mcts.pred_output.unsafe_ptr())
                    var e_b_rew = LayoutTensor[
                        dtype,
                        Layout.row_major(EVAL_TOTAL_EXPAND),
                        MutAnyOrigin,
                    ](exp_rewards.unsafe_ptr())
                    comptime e_run_exp = gpu_mcts_batched_expand_backup_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        BATCH_SIMS,
                        MCTS_PRED_OUT,
                        GS,
                        dtype,
                    ]
                    ctx.enqueue_function[e_run_exp, e_run_exp](
                        e_vc,
                        e_tv,
                        e_pr,
                        e_rw,
                        e_ci,
                        e_tvis,
                        e_nc,
                        e_miq,
                        e_mxq,
                        e_gs,
                        e_b_exp,
                        e_b_pp,
                        e_b_pa,
                        e_b_po,
                        e_b_rew,
                        e_b_sp,
                        e_b_ap,
                        e_b_pl,
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # Greedy action (temp=0)
                var e_act = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ](eval_acts.unsafe_ptr())
                var e_pol = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * ACT),
                    MutAnyOrigin,
                ](eval_mcts.policies_out.unsafe_ptr())
                comptime e_run_act = gpu_mcts_extract_actions_masked_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                ctx.enqueue_function[e_run_act, e_run_act](
                    e_vc,
                    e_lm,
                    e_act,
                    e_pol,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
            else:
                # Opponent's turn — use GPUEvaluator
                Eval.select_action_gpu[Self.n_envs, ACT, GS](
                    ctx,
                    eval_acts,
                    eval_legal,
                    eval_states,
                    rng_seed=UInt64(rng_offset + eval_move),
                )

            # Env step
            E.step_kernel_gpu[Self.n_envs, GS, OBS](
                ctx,
                eval_states,
                eval_acts,
                eval_rews,
                eval_dones,
                eval_term,
                eval_obs,
                eval_legal,
                rng_seed=UInt64(rng_offset + eval_move + 5000),
            )

            # Check completions
            ctx.enqueue_copy(eval_dones_host, eval_dones)
            ctx.enqueue_copy(eval_rews_host, eval_rews)
            ctx.synchronize()

            eval_all_done = True
            for e in range(Self.n_envs):
                if not eval_done[e] and Float64(eval_dones_host[e]) > 0.5:
                    eval_done[e] = True
                    var rew = Float64(eval_rews_host[e])
                    if rew > 0.5:
                        eval_result[e] = 1 if agent_turn else 2
                    elif rew < -0.5:
                        eval_result[e] = 2 if agent_turn else 1
                    else:
                        eval_result[e] = 3
                if not eval_done[e]:
                    eval_all_done = False

            # Selective reset
            E.selective_reset_kernel_gpu[Self.n_envs, GS](
                ctx,
                eval_states,
                eval_dones,
                rng_seed=UInt64(rng_offset + eval_move + 7000),
            )
            E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
                ctx, eval_states, eval_obs, eval_legal
            )
            eval_move += 1

        # Count results (agent = P0)
        var eval_wins = 0
        var eval_draws = 0
        var eval_losses = 0
        for e in range(Self.n_envs):
            if eval_result[e] == 1:
                eval_wins += 1
            elif eval_result[e] == 3:
                eval_draws += 1
            else:
                eval_losses += 1

        return (eval_wins, eval_draws, eval_losses)

    # ══════════════════════════════════════════════════════════════
    # Self-Play GPU Training
    # ══════════════════════════════════════════════════════════════

    def train_selfplay_gpu[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
        GPUEval: GPUEvaluator = RandomOpponent,
        GPUEval2: GPUEvaluator = RandomOpponent,
        USE_CUDA_GRAPH: Bool = True,
    ](
        mut self,
        ctx: DeviceContext,
        num_iters: Int = 25,
        steps_per_iter: Int = 1000,
        train_epochs: Int = 10,
        warmup_iters: Int = 1,
        arena_threshold: Float64 = 0.6,
        do_eval: Bool = True,
        do_eval2: Bool = False,
        do_arena: Bool = True,
        slow_window_start: Int = 0,  # 0=disabled. >0: start with this window, grow to full
        slow_window_growth: Int = 2,  # Grow window by 1 every N iterations
        checkpoint_every: Int = 0,
        checkpoint_path: String = "alphazero.ckpt",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 50,
        verbose: Bool = True,
    ) raises -> TrainingMetrics:
        """Train via batch-then-train (like alpha-zero-general).

        Each iteration:
          1. Collect self-play games (frozen network, GPU MCTS)
          2. Train for train_epochs on ALL collected data
          3. GPU evaluation vs GPUEval opponent
          4. GPU arena: new model vs best model (MCTS temp=0)
          5. Accept/reject based on arena threshold
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime SIMS = Self.Config.num_simulations
        comptime MAX_NODES = Self.Config.max_nodes
        comptime GS = E.STATE_SIZE
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM
        comptime MCTS_PRED_OUT = ACT + 1
        comptime BATCH_SIMS = Self.Config.batch_sims
        comptime NUM_ROUNDS = SIMS // BATCH_SIMS
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

        # GPU state (created once, re-uploaded each iteration)
        var gpu = Self.GPUStateType(ctx)
        gpu.upload_from(self.state, ctx)

        # GPU MCTS state (with game states for true-rules expansion)
        # Use a minimal LATENT=OBS since we run PredNet on obs directly
        var gpu_mcts = GPUMCTSState[
            Self.n_envs, MAX_NODES, ACT, OBS, 1, E.STATE_SIZE,
            Self.Config.batch_sims,
        ](ctx)

        # Network workspace (sized for batched prediction: n_envs * BATCH_SIMS)
        comptime BATCH_SIMS_C = Self.Config.batch_sims
        comptime WS = Self.Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_SIZE = Self.n_envs * BATCH_SIMS_C * WS if WS > 0 else 1
        var mcts_ws = ctx.enqueue_create_buffer[dtype](WS_SIZE)

        # Env buffers
        var states_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * E.STATE_SIZE
        )
        var obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var legal_masks_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * ACT
        )

        # Episode tracking
        var ep_rew_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var ep_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rew_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var ep_count_buf = ctx.enqueue_create_buffer[dtype](1)
        var rew_sum_host = ctx.enqueue_create_host_buffer[dtype](1)
        var ep_count_host = ctx.enqueue_create_host_buffer[dtype](1)

        # Expansion scratch for batched MCTS (BATCH_SIMS per env)
        comptime BATCH_SIMS_ALLOC = Self.Config.batch_sims
        comptime TOTAL_EXPAND_ALLOC = Self.n_envs * BATCH_SIMS_ALLOC
        var exp_rewards = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC)
        var exp_dones = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC)
        var exp_terminated = ctx.enqueue_create_buffer[dtype](
            TOTAL_EXPAND_ALLOC
        )
        var exp_obs = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC * OBS)

        # Host buffers for collecting self-play data
        var obs_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs * OBS)
        var policy_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * ACT
        )
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)

        # Pre-allocated diagnostic host buffers (reused every train step)
        comptime _DIAG_BATCH = Self.Config.batch_size
        comptime _DIAG_POUT = Self.Config.PredModel.OUT_DIM
        comptime _DIAG_PS = Self.Config.PredModel.PARAM_SIZE
        var diag_pred_host = ctx.enqueue_create_host_buffer[dtype](
            _DIAG_BATCH * _DIAG_POUT
        )
        var diag_go_host = ctx.enqueue_create_host_buffer[dtype](
            _DIAG_BATCH * _DIAG_POUT
        )
        var diag_params_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)
        var diag_grads_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)

        # Pre-allocated arena param buffers (reused every iteration)
        comptime _ARENA_PS = Self.Config.PredModel.PARAM_SIZE
        var arena_new_params = ctx.enqueue_create_buffer[dtype](_ARENA_PS)
        var arena_old_params = ctx.enqueue_create_buffer[dtype](_ARENA_PS)
        var arena_params_host = ctx.enqueue_create_host_buffer[dtype](_ARENA_PS)

        # Pre-allocated Q-value host buffers (for q-value training targets)
        comptime _Q_WEIGHT = Self.Config.value_target_q_weight
        comptime _Q_BUF_SIZE = Self.n_envs * MAX_NODES * ACT if _Q_WEIGHT > 0.0 else 1
        var q_vc_host = ctx.enqueue_create_host_buffer[dtype](_Q_BUF_SIZE)
        var q_tv_host = ctx.enqueue_create_host_buffer[dtype](_Q_BUF_SIZE)

        # Per-env episode data for computing game outcomes
        var env_rewards = List[List[Float64]](capacity=Self.n_envs)
        var env_obs_history = List[List[List[Scalar[dtype]]]](
            capacity=Self.n_envs
        )
        var env_policy_history = List[List[List[Scalar[dtype]]]](
            capacity=Self.n_envs
        )
        # Per-env root Q-value history (for q-value training targets)
        var env_q_history = List[List[Float64]](capacity=Self.n_envs)
        var root_q_host = List[Float64](capacity=Self.n_envs)
        for _ in range(Self.n_envs):
            env_rewards.append(List[Float64]())
            env_obs_history.append(List[List[Scalar[dtype]]]())
            env_policy_history.append(List[List[Scalar[dtype]]]())
            env_q_history.append(List[Float64]())
            root_q_host.append(0.0)

        # Initialize envs
        E.reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
            ctx, states_buf, obs_buf, legal_masks_buf
        )
        ep_rew_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_steps_buf.enqueue_fill(Scalar[dtype](0.0))
        rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        # Set logger for diagnostics
        self.logger = logger
        self.diag_every = diag_every

        var metrics = TrainingMetrics(algorithm_name="AlphaZero")
        var total_steps = 0

        # Save initial params + optimizer state for arena comparison
        comptime PS = Self.Config.PredModel.PARAM_SIZE
        comptime OPT_SS = PS * Self.Config.OptType.STATE_PER_PARAM
        var best_params = alloc[Scalar[dtype]](PS)
        var best_opt_state = alloc[Scalar[dtype]](OPT_SS)
        for i in range(PS):
            best_params[i] = self.state.prediction.params[i]
        for i in range(OPT_SS):
            best_opt_state[i] = self.state.prediction.optimizer_state[i]
        var arena_accepts = 0
        var arena_rejects = 0

        # CUDA Graph state (lazy-captured on first use)
        var _train_graph: Optional[CUDAGraph] = None
        var _mcts_round_graph: Optional[CUDAGraph] = None

        # Debug: fires once on first MCTS step and first completed game
        var _debug_mcts_done = False
        var _debug_game_done = False
        # Host buffers for debug (visit counts + states)
        var _dbg_vc_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * MAX_NODES * ACT
        )
        var _dbg_states_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * E.STATE_SIZE
        )

        # ══════════════════════════════════════════════════════════
        # Iteration loop (batch-then-train)
        # ══════════════════════════════════════════════════════════
        for iter in range(num_iters):
            # ── 1. Mark new iteration (sliding window eviction) ──
            self.state.start_new_iteration()

            # Slow window: limit effective history to a growing window
            # that starts small and reaches full size over time.
            if slow_window_start > 0:
                var effective_window = (
                    slow_window_start + iter // slow_window_growth
                )
                comptime FULL_WINDOW = Self.Config.history_window
                if effective_window > FULL_WINDOW:
                    effective_window = FULL_WINDOW
                while self.state.num_iters > effective_window + 1:
                    self.state._evict_oldest()

            # Determine if this is a warmup iteration (random actions)
            var use_mcts = iter >= warmup_iters

            # Re-upload params to GPU (training may have modified them)
            gpu.upload_from(self.state, ctx)

            # Reset episode counters for this iteration
            rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
            ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
            ctx.synchronize()

            # ── 2. Collect self-play data (frozen network) ───────
            var iter_steps = 0
            while iter_steps < steps_per_iter:
                # Download obs for episode tracking
                ctx.enqueue_copy(obs_host, obs_buf)
                ctx.synchronize()

                # GPU MCTS with true game rules
                if use_mcts:
                    # Prediction on obs for root
                    var pred_obs = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, PRED_IN),
                        MutAnyOrigin,
                    ](obs_buf.unsafe_ptr())
                    var pred_out = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, PRED_OUT_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    Self.PredNet.forward_gpu[Self.n_envs](
                        ctx,
                        pred_obs,
                        pred_out,
                        gpu.prediction.params_view(),
                        mcts_ws,
                    )

                    # Init root
                    comptime MCTS_PRED_OUT = ACT + 1
                    var vc = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.visit_count.unsafe_ptr())
                    var tv = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.total_value.unsafe_ptr())
                    var pr = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.prior.unsafe_ptr())
                    var rw = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.reward.unsafe_ptr())
                    var ci = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.child_idx.unsafe_ptr())
                    var tvis = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES),
                        MutAnyOrigin,
                    ](gpu_mcts.total_visits.unsafe_ptr())
                    var nc = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.node_count.unsafe_ptr())
                    var po = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MCTS_PRED_OUT),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    var miq = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.min_q.unsafe_ptr())
                    var mxq = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.max_q.unsafe_ptr())

                    gpu_mcts.zero_tree(ctx)
                    comptime run_init = gpu_mcts_init_root_kernel[
                        Self.n_envs, MAX_NODES, ACT, OBS, MCTS_PRED_OUT, dtype
                    ]
                    ctx.enqueue_function[run_init, run_init](
                        vc,
                        tv,
                        pr,
                        rw,
                        ci,
                        tvis,
                        nc,
                        po,
                        miq,
                        mxq,
                        Scalar[dtype](Self.Config.Noise.NOISE_FRACTION),
                        Scalar[DType.uint32](UInt32(total_steps)),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Legal mask on root
                    var lm = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](legal_masks_buf.unsafe_ptr())
                    comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                        Self.n_envs, MAX_NODES, ACT, dtype
                    ]
                    ctx.enqueue_function[run_mask, run_mask](
                        pr, lm, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
                    )

                    # Copy root game states
                    var gs = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * GS),
                        MutAnyOrigin,
                    ](gpu_mcts.game_states.unsafe_ptr())
                    var es = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * GS), MutAnyOrigin
                    ](states_buf.unsafe_ptr())
                    comptime run_rs = gpu_mcts_copy_root_state_kernel[
                        Self.n_envs, MAX_NODES, GS, dtype
                    ]
                    ctx.enqueue_function[run_rs, run_rs](
                        gs, es, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
                    )

                    # Batched simulations (BATCH_SIMS leaves per round)
                    comptime BATCH_SIMS = Self.Config.batch_sims
                    comptime NUM_ROUNDS = SIMS // BATCH_SIMS
                    comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

                    # Batched buffers [N_ENVS * BATCH_SIMS * ...]
                    var b_pp = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.pending_parent.unsafe_ptr())
                    var b_pa = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.pending_action.unsafe_ptr())
                    var b_sp = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                        MutAnyOrigin,
                    ](gpu_mcts.search_paths.unsafe_ptr())
                    var b_ap = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                        MutAnyOrigin,
                    ](gpu_mcts.action_paths.unsafe_ptr())
                    var b_pl = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.path_lengths.unsafe_ptr())
                    var b_exp_st = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND * GS), MutAnyOrigin
                    ](gpu_mcts.expansion_states.unsafe_ptr())

                    comptime if USE_CUDA_GRAPH:
                        if not _mcts_round_graph:
                            # Warmup: run one round without capture
                            self._mcts_round_kernels[E](
                                ctx,
                                gpu,
                                gpu_mcts,
                                exp_rewards,
                                exp_dones,
                                exp_terminated,
                                exp_obs,
                                mcts_ws,
                                vc,
                                tv,
                                pr,
                                rw,
                                ci,
                                tvis,
                                nc,
                                miq,
                                mxq,
                                gs,
                                b_pp,
                                b_pa,
                                b_exp_st,
                                b_sp,
                                b_ap,
                                b_pl,
                            )
                            ctx.synchronize()
                            # Capture
                            var graph = CUDAGraph(ctx)
                            graph.begin_capture()
                            self._mcts_round_kernels[E](
                                ctx,
                                gpu,
                                gpu_mcts,
                                exp_rewards,
                                exp_dones,
                                exp_terminated,
                                exp_obs,
                                mcts_ws,
                                vc,
                                tv,
                                pr,
                                rw,
                                ci,
                                tvis,
                                nc,
                                miq,
                                mxq,
                                gs,
                                b_pp,
                                b_pa,
                                b_exp_st,
                                b_sp,
                                b_ap,
                                b_pl,
                            )
                            graph.end_capture()
                            if verbose:
                                print(
                                    "[CUDA Graph] Captured MCTS round with "
                                    + String(graph.num_nodes())
                                    + " nodes"
                                )
                            _mcts_round_graph = graph^
                            # Replay remaining rounds on Mojo stream
                            # (preserves ordering with init_root/extract_actions)
                            for _ in range(NUM_ROUNDS - 1):
                                _mcts_round_graph.value().replay_on_mojo_stream()
                        else:
                            for _ in range(NUM_ROUNDS):
                                _mcts_round_graph.value().replay_on_mojo_stream()
                    else:
                        for _round in range(NUM_ROUNDS):
                            self._mcts_round_kernels[E](
                                ctx,
                                gpu,
                                gpu_mcts,
                                exp_rewards,
                                exp_dones,
                                exp_terminated,
                                exp_obs,
                                mcts_ws,
                                vc,
                                tv,
                                pr,
                                rw,
                                ci,
                                tvis,
                                nc,
                                miq,
                                mxq,
                                gs,
                                b_pp,
                                b_pa,
                                b_exp_st,
                                b_sp,
                                b_ap,
                                b_pl,
                            )

                    # Extract actions with temperature annealing
                    var act_out = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](actions_buf.unsafe_ptr())
                    var pol_out = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](gpu_mcts.policies_out.unsafe_ptr())
                    var ep_steps_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](ep_steps_buf.unsafe_ptr())
                    comptime run_act = gpu_mcts_extract_actions_temp_kernel[
                        Self.n_envs, MAX_NODES, ACT, dtype
                    ]
                    ctx.enqueue_function[run_act, run_act](
                        vc,
                        lm,
                        ep_steps_t,
                        act_out,
                        pol_out,
                        Self.Config.temp_threshold,
                        Scalar[DType.uint32](UInt32(total_steps)),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Download policies for training data
                    ctx.enqueue_copy(policy_host, gpu_mcts.policies_out)

                    # ── DEBUG: MCTS visit counts + obs + network output (env 0, first step) ──
                    if not _debug_mcts_done and use_mcts:
                        _debug_mcts_done = True
                        ctx.enqueue_copy(_dbg_vc_host, gpu_mcts.visit_count)
                        ctx.enqueue_copy(_dbg_states_host, states_buf)
                        ctx.synchronize()
                        print("╔══════════════════════════════════════════╗")
                        print("║  DEBUG: MCTS + Obs (env 0, first step)  ║")
                        print("╚══════════════════════════════════════════╝")
                        # Board state (raw bytes from ConnectFour state)
                        print("Board state (col-major, 0=empty 1=P0 2=P1):")
                        for row in range(5, -1, -1):
                            var row_str = String("  ")
                            for col in range(7):
                                var cell = Int(
                                    Float64(_dbg_states_host[col * 6 + row])
                                )
                                if cell == 0:
                                    row_str += ". "
                                elif cell == 1:
                                    row_str += "X "
                                else:
                                    row_str += "O "
                            print(row_str)
                        var cur_player = Int(
                            Float64(_dbg_states_host[42])
                        )
                        print("Current player:", cur_player)
                        # Observation planes
                        print("Obs plane 0 (my pieces) sum:", end="")
                        var p0_sum: Float64 = 0.0
                        for i in range(42):
                            p0_sum += Float64(obs_host[i])
                        print("", p0_sum)
                        print("Obs plane 1 (opp pieces) sum:", end="")
                        var p1_sum: Float64 = 0.0
                        for i in range(42, 84):
                            p1_sum += Float64(obs_host[i])
                        print("", p1_sum)
                        print("Obs plane 2 (legal) sum:", end="")
                        var p2_sum: Float64 = 0.0
                        for i in range(84, 126):
                            p2_sum += Float64(obs_host[i])
                        print("", p2_sum)
                        # MCTS visit counts for root (env 0)
                        print("MCTS root visit counts:", end="")
                        var total_vc: Float64 = 0.0
                        for a in range(ACT):
                            var vc_val = Float64(_dbg_vc_host[a])
                            print("", Int(vc_val), end="")
                            total_vc += vc_val
                        print("  (total:", Int(total_vc), ")")
                        # Policy target (from visit counts)
                        print("Policy target:", end="")
                        for a in range(ACT):
                            var p = Float64(policy_host[a])
                            print(
                                "",
                                Int(p * 1000.0) / 1000.0,
                                end="",
                            )
                        print()
                        # Network raw output (logits + value)
                        # pred_output was written during root forward pass
                        var _dbg_pred_host = ctx.enqueue_create_host_buffer[
                            dtype
                        ](Self.n_envs * MCTS_PRED_OUT)
                        ctx.enqueue_copy(
                            _dbg_pred_host, gpu_mcts.pred_output
                        )
                        ctx.synchronize()
                        print("Network policy logits:", end="")
                        for a in range(ACT):
                            var l = Float64(_dbg_pred_host[a])
                            print(
                                "",
                                Int(l * 1000.0) / 1000.0,
                                end="",
                            )
                        print()
                        var raw_v = Float64(_dbg_pred_host[ACT])
                        var tanh_v = (exp(raw_v) - exp(-raw_v)) / (
                            exp(raw_v) + exp(-raw_v)
                        )
                        print(
                            "Network value: raw=",
                            Int(raw_v * 1000.0) / 1000.0,
                            "tanh=",
                            Int(tanh_v * 1000.0) / 1000.0,
                        )
                        print("─" * 44)

                    # Compute root Q-value per env for q-value targets
                    # Q = sum(total_value[root,a]) / sum(visit_count[root,a])
                    comptime Q_WEIGHT = Self.Config.value_target_q_weight
                    comptime if Q_WEIGHT > 0.0:
                        ctx.enqueue_copy(q_vc_host, gpu_mcts.visit_count)
                        ctx.enqueue_copy(q_tv_host, gpu_mcts.total_value)
                        ctx.synchronize()
                        for e_q in range(Self.n_envs):
                            var root_off = (
                                e_q * MAX_NODES * ACT
                            )  # root = node 0
                            var total_v: Float64 = 0.0
                            var total_n: Float64 = 0.0
                            for a_q in range(ACT):
                                total_v += Float64(q_tv_host[root_off + a_q])
                                total_n += Float64(q_vc_host[root_off + a_q])
                            if total_n > 0.5:
                                root_q_host[e_q] = total_v / total_n
                            else:
                                root_q_host[e_q] = 0.0
                else:
                    # Warmup: random legal actions
                    comptime run_warmup = uniform_random_discrete_actions_kernel[
                        dtype, Self.n_envs, ACT
                    ]
                    var wa = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](actions_buf.unsafe_ptr())
                    ctx.enqueue_function[run_warmup, run_warmup](
                        wa,
                        Scalar[DType.uint32](UInt32(total_steps)),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # Uniform policies for warmup
                    for i in range(Self.n_envs * ACT):
                        policy_host[i] = Scalar[dtype](1.0 / Float64(ACT))

                # Env step
                E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                    ctx,
                    states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    legal_masks_buf,
                    rng_seed=UInt64(total_steps),
                )

                # Episode tracking
                var rew_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rewards_buf.unsafe_ptr())
                var don_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](dones_buf.unsafe_ptr())
                var epr_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](ep_rew_buf.unsafe_ptr())
                var eps_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](ep_steps_buf.unsafe_ptr())
                comptime run_accum = accumulate_rewards_kernel[
                    dtype, Self.n_envs
                ]
                ctx.enqueue_function[run_accum, run_accum](
                    epr_t, rew_t, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
                )
                comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
                ctx.enqueue_function[run_incr, run_incr](
                    eps_t, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
                )
                var rs_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](rew_sum_buf.unsafe_ptr())
                var ec_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](ep_count_buf.unsafe_ptr())
                comptime run_log = log_and_reset_completed_kernel[
                    dtype, Self.n_envs
                ]
                ctx.enqueue_function[run_log, run_log](
                    don_t,
                    epr_t,
                    eps_t,
                    rs_t,
                    ec_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Collect training data
                ctx.enqueue_copy(rewards_host, rewards_buf)
                ctx.enqueue_copy(dones_host, dones_buf)
                ctx.synchronize()

                if use_mcts:
                    for e in range(Self.n_envs):
                        # Store this step's obs + policy
                        var step_obs = List[Scalar[dtype]](capacity=OBS)
                        for i in range(OBS):
                            step_obs.append(obs_host[e * OBS + i])
                        var step_pol = List[Scalar[dtype]](capacity=ACT)
                        for i in range(ACT):
                            step_pol.append(policy_host[e * ACT + i])

                        env_obs_history[e].append(step_obs^)
                        env_policy_history[e].append(step_pol^)
                        env_rewards[e].append(Float64(rewards_host[e]))
                        env_q_history[e].append(root_q_host[e])

                        if Float64(dones_host[e]) > 0.5:
                            # Game ended — compute outcome and add all moves
                            var last_reward = Float64(rewards_host[e])
                            var ep_len = len(env_obs_history[e])

                            var is_draw = (
                                last_reward > -0.01 and last_reward < 0.01
                            )

                            # ── DEBUG: First completed game (env 0) ──
                            var _dbg_this_game = (
                                not _debug_game_done
                                and e == 0
                                and use_mcts
                            )
                            if _dbg_this_game:
                                _debug_game_done = True
                                print(
                                    "╔═══════════════════════════════"
                                    "═══════════════╗"
                                )
                                print(
                                    "║  DEBUG: First completed game"
                                    " (env 0)        ║"
                                )
                                print(
                                    "╚═══════════════════════════════"
                                    "═══════════════╝"
                                )
                                print(
                                    "last_reward:",
                                    last_reward,
                                    "is_draw:",
                                    is_draw,
                                    "ep_len:",
                                    ep_len,
                                )
                                # Print board from final state
                                ctx.enqueue_copy(
                                    _dbg_states_host, states_buf
                                )
                                ctx.synchronize()
                                print("Final board:")
                                for row in range(5, -1, -1):
                                    var row_str = String("  ")
                                    for col in range(7):
                                        var cell = Int(
                                            Float64(
                                                _dbg_states_host[
                                                    col * 6 + row
                                                ]
                                            )
                                        )
                                        if cell == 0:
                                            row_str += ". "
                                        elif cell == 1:
                                            row_str += "X "
                                        else:
                                            row_str += "O "
                                    print(row_str)

                            comptime QW = Self.Config.value_target_q_weight

                            for t in range(ep_len):
                                var steps_from_end = ep_len - 1 - t
                                # z = game outcome from this player's perspective
                                var z: Float64
                                if is_draw:
                                    z = 1e-4
                                elif steps_from_end % 2 == 0:
                                    z = last_reward
                                else:
                                    z = -last_reward

                                # Blend z with MCTS root Q-value
                                var value_target = z
                                comptime if QW > 0.0:
                                    var q = env_q_history[e][t]
                                    value_target = (1.0 - QW) * z + QW * q

                                if _dbg_this_game and t < 6:
                                    # Print first 6 moves' targets
                                    print(
                                        "  step",
                                        t,
                                        "steps_from_end:",
                                        steps_from_end,
                                        "z:",
                                        Int(z * 100.0) / 100.0,
                                        "policy:",
                                        end="",
                                    )
                                    for a in range(ACT):
                                        print(
                                            "",
                                            Int(
                                                Float64(
                                                    env_policy_history[e][
                                                        t
                                                    ][a]
                                                )
                                                * 100.0
                                            )
                                            / 100.0,
                                            end="",
                                        )
                                    print()

                                self._add_with_augmentation[E](
                                    env_obs_history[e][t],
                                    env_policy_history[e][t],
                                    Scalar[dtype](value_target),
                                )

                            if _dbg_this_game:
                                print("─" * 46)

                            env_obs_history[e].clear()
                            env_policy_history[e].clear()
                            env_rewards[e].clear()
                            env_q_history[e].clear()
                else:
                    # During warmup, just clear episode data on done
                    for e in range(Self.n_envs):
                        if Float64(dones_host[e]) > 0.5:
                            env_obs_history[e].clear()
                            env_policy_history[e].clear()
                            env_rewards[e].clear()
                            env_q_history[e].clear()

                # Selective reset
                E.selective_reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
                    ctx,
                    states_buf,
                    dones_buf,
                    rng_seed=UInt64(total_steps),
                )
                E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                    ctx,
                    states_buf,
                    obs_buf,
                    legal_masks_buf,
                )

                iter_steps += Self.n_envs
                total_steps += Self.n_envs
                self.total_steps += Self.n_envs

            # end collection loop

            # Read episode stats for this iteration
            ctx.enqueue_copy(rew_sum_host, rew_sum_buf)
            ctx.enqueue_copy(ep_count_host, ep_count_buf)
            ctx.synchronize()
            var iter_games = Int(Float64(ep_count_host[0]))

            # ── 3. Train for train_epochs on collected data ──────
            if use_mcts and self.state.is_ready(Self.Config.batch_size):
                var buf_sz = self.state.buf_size
                comptime BATCH = Self.Config.batch_size
                var num_train_steps = train_epochs * buf_sz // BATCH
                # Upload latest params to GPU for training
                gpu.upload_from(self.state, ctx)

                comptime if USE_CUDA_GRAPH:
                    # Upload replay buffer to GPU (one bulk copy per iteration)
                    gpu.upload_replay(self.state, ctx)

                    # Lazy capture: first time, warmup + capture
                    # (sampling is now GPU-side inside _gpu_train_kernels)
                    if not _train_graph:
                        self._gpu_train_kernels(ctx, gpu)
                        ctx.synchronize()
                        var graph = CUDAGraph(ctx)
                        graph.begin_capture()
                        self._gpu_train_kernels(ctx, gpu)
                        graph.end_capture()
                        if verbose:
                            print(
                                "[CUDA Graph] Captured train step with "
                                + String(graph.num_nodes())
                                + " nodes"
                            )
                        _train_graph = graph^
                    # All steps via async graph replay (no per-step CPU work)
                    for _ in range(num_train_steps):
                        _train_graph.value().replay_on_mojo_stream()

                    # Diagnostics outside graph (once after all replays)
                    self._gpu_train_diagnostics(
                        ctx,
                        gpu,
                        num_train_steps,
                        diag_pred_host,
                        diag_go_host,
                        diag_params_host,
                        diag_grads_host,
                    )
                else:
                    for _ in range(num_train_steps):
                        self.train_step_gpu(
                            ctx,
                            gpu,
                            diag_pred_host,
                            diag_go_host,
                            diag_params_host,
                            diag_grads_host,
                        )

                # Download trained params back to CPU
                gpu.download_to(self.state, ctx)

            # ── 4. GPU Evaluation ────────────────────────────────
            if do_eval and use_mcts:
                # Re-upload after training for eval
                gpu.upload_from(self.state, ctx)
                var eval_r = self.gpu_eval[E, GPUEval](
                    ctx,
                    gpu,
                    gpu_mcts,
                    mcts_ws,
                    states_buf,
                    obs_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    legal_masks_buf,
                    exp_rewards,
                    exp_dones,
                    exp_terminated,
                    exp_obs,
                    rewards_host,
                    dones_host,
                    rng_offset=total_steps,
                )
                print(
                    "  Iter",
                    iter,
                    "| Steps:",
                    total_steps,
                    "| Games:",
                    iter_games,
                    "| Replay:",
                    self.state.buf_size,
                    "| Train:",
                    self.train_step_count,
                    "| Eval vs",
                    GPUEval.NAME,
                    ": W",
                    eval_r[0],
                    "D",
                    eval_r[1],
                    "L",
                    eval_r[2],
                )
            else:
                print(
                    "  Iter",
                    iter,
                    "| Steps:",
                    total_steps,
                    "| Games:",
                    iter_games,
                    "| Replay:",
                    self.state.buf_size,
                    "| Train:",
                    self.train_step_count,
                )

            # ── 4b. Second GPU Evaluation ────────────────────────
            if do_eval2 and use_mcts:
                var eval_r2 = self.gpu_eval[E, GPUEval2](
                    ctx,
                    gpu,
                    gpu_mcts,
                    mcts_ws,
                    states_buf,
                    obs_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    legal_masks_buf,
                    exp_rewards,
                    exp_dones,
                    exp_terminated,
                    exp_obs,
                    rewards_host,
                    dones_host,
                    rng_offset=total_steps + 77777,
                )
                print(
                    "    vs",
                    GPUEval2.NAME,
                    ": W",
                    eval_r2[0],
                    "D",
                    eval_r2[1],
                    "L",
                    eval_r2[2],
                )

            # ── 5. GPU Arena ─────────────────────────────────────
            if do_arena and use_mcts and iter >= warmup_iters:
                var arena_r = self.arena_compare_gpu[E](
                    ctx,
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        best_params
                    ),
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        best_opt_state
                    ),
                    arena_new_params,
                    arena_old_params,
                    arena_params_host,
                    gpu_mcts,
                    mcts_ws,
                    states_buf,
                    obs_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    legal_masks_buf,
                    exp_rewards,
                    exp_dones,
                    exp_terminated,
                    exp_obs,
                    rewards_host,
                    dones_host,
                    num_games=40,
                    threshold=arena_threshold,
                )
                var accepted = arena_r[0]
                var a_new = arena_r[1]
                var a_draw = arena_r[2]
                var a_old = arena_r[3]
                if accepted:
                    for i in range(PS):
                        best_params[i] = self.state.prediction.params[i]
                    for i in range(OPT_SS):
                        best_opt_state[i] = self.state.prediction.optimizer_state[i]
                    arena_accepts += 1
                    print(
                        "  Arena: ACCEPTED",
                        "W",
                        a_new,
                        "D",
                        a_draw,
                        "L",
                        a_old,
                        "(",
                        arena_accepts,
                        "/",
                        arena_accepts + arena_rejects,
                        ")",
                    )
                else:
                    arena_rejects += 1
                    print(
                        "  Arena: rejected",
                        "W",
                        a_new,
                        "D",
                        a_draw,
                        "L",
                        a_old,
                        "(",
                        arena_accepts,
                        "/",
                        arena_accepts + arena_rejects,
                        ")",
                    )

            # ── 6. Checkpoint ───────────────────────────────────
            if checkpoint_every > 0 and (iter + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)

        # end iteration loop
        best_params.free()
        best_opt_state.free()
        if do_arena:
            print("Arena: accepted", arena_accepts, "/ rejected", arena_rejects)
        if checkpoint_every > 0:
            self.save_checkpoint(checkpoint_path)
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics
