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
    write_checkpoint_header, write_metadata_section,
    save_checkpoint_file, read_checkpoint_file,
    parse_checkpoint_header, read_metadata_section, get_metadata_value,
)
from mojo_rl.core import (
    TrainingMetrics, TwoPlayerDiscreteEnv, GPUTwoPlayerDiscreteEnv,
    DataAugmentable,
)
from mojo_rl.deep_agents.core.utils import print_progress_bar, clear_progress_bar
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel, increment_steps_kernel,
    log_and_reset_completed_kernel, uniform_random_discrete_actions_kernel,
)
from mojo_rl.deep_agents.muzero.gpu_mcts import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_extract_actions_masked_kernel, gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_copy_root_state_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_kernel,
    TPB, MAX_DEPTH,
)
# extract_hidden_kernel not needed for AlphaZero (no dynamics)
from mojo_rl.deep_agents.muzero.evaluators import Evaluator
from .configs import AlphaZeroConfig
from .state import AlphaZeroCPUState, AlphaZeroGPUState


# ═══════════════════════════════════════════════════════════════════════════
# GPU Kernels for AlphaZero Training
# ═══════════════════════════════════════════════════════════════════════════


fn az_policy_value_grad_kernel[
    BATCH: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin],
    pred_out: LayoutTensor[dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin],
    target_policy: LayoutTensor[dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin],
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
        sum_exp += exp(rebind[Scalar[dtype]](pred_out[pred_off + a]) - max_logit)
    # Entropy regularization coefficient (prevents policy collapse)
    var entropy_coef = Scalar[dtype](0.01)
    for a in range(ACT):
        var prob = exp(rebind[Scalar[dtype]](pred_out[pred_off + a]) - max_logit) / sum_exp
        var target = rebind[Scalar[dtype]](target_policy[pol_off + a])
        # CE gradient + entropy bonus: d/d_logit = (prob - target) + entropy_coef * (1 + log(prob))
        # Simplified: entropy gradient = entropy_coef * (log(prob) + 1) * prob * (1 - prob)
        # But simpler: just add entropy_coef to push prob toward uniform
        var entropy_grad = entropy_coef * (prob - Scalar[dtype](1.0) / Scalar[dtype](ACT))
        grad_out[pred_off + a] = ((prob - target) + entropy_grad) * inv_batch

    # Value gradient: MSE through tanh activation
    # loss = (tanh(raw) - target)^2
    # d/draw = 2 * (tanh(raw) - target) * (1 - tanh(raw)^2)
    var raw_v = rebind[Scalar[dtype]](pred_out[pred_off + ACT])
    var target_v = rebind[Scalar[dtype]](target_value[b])
    var ev_p = exp(raw_v)
    var ev_n = exp(-raw_v)
    var tanh_v = (ev_p - ev_n) / (ev_p + ev_n)
    var dtanh = Scalar[dtype](1.0) - tanh_v * tanh_v  # tanh derivative
    grad_out[pred_off + ACT] = Scalar[dtype](2.0) * (tanh_v - target_v) * dtanh * inv_batch


# ═══════════════════════════════════════════════════════════════════════════
# AlphaZero Agent
# ═══════════════════════════════════════════════════════════════════════════


struct GenericAlphaZeroAgent[Config: AlphaZeroConfig, n_envs: Int = 64](Movable):
    """AlphaZero agent for two-player board games.

    Uses GPU MCTS with true game rules for self-play, trains a single
    prediction network with supervised policy + value loss.
    """

    comptime StateType = AlphaZeroCPUState[Self.Config]
    comptime GPUStateType = AlphaZeroGPUState[Self.Config]
    comptime PredNet = Network[Self.Config.PredModel, Self.Config.OptType]
    comptime PRED_OUT: Int = Self.Config.action_dim + 1

    var state: Self.StateType
    var train_step_count: Int
    var total_steps: Int

    fn __init__(out self):
        self.state = Self.StateType()
        self.train_step_count = 0
        self.total_steps = 0

    fn __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.train_step_count = take.train_step_count
        self.total_steps = take.total_steps

    # ══════════════════════════════════════════════════════════════
    # Policy-only action selection (for evaluation)
    # ══════════════════════════════════════════════════════════════

    fn select_action(
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
        var obs_t = LayoutTensor[dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin](obs_ptr)

        var pred_ptr = alloc[Scalar[dtype]](PRED_OUT_DIM)
        memset(pred_ptr, 0, PRED_OUT_DIM)
        var pred_t = LayoutTensor[dtype, Layout.row_major(B, PRED_OUT_DIM), MutAnyOrigin](pred_ptr)

        Self.PredNet.forward[B](obs_t, pred_t, self.state.prediction.params_view())

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

    # ══════════════════════════════════════════════════════════════
    # Evaluation
    # ══════════════════════════════════════════════════════════════

    fn evaluate_against[
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

    fn print_eval[E: TwoPlayerDiscreteEnv, EvalType: Evaluator](
        mut self, mut env: E, mut evaluator: EvalType, num_games: Int = 50,
    ):
        var r = self.evaluate_against[E, EvalType](env, evaluator, num_games)
        print(
            "  vs", evaluator.name(),
            "| W:", r[0], "D:", r[1], "L:", r[2],
            "| Win%:", r[0] * 100 // num_games,
            "Draw%:", r[1] * 100 // num_games,
        )

    # ══════════════════════════════════════════════════════════════
    # Checkpointing
    # ══════════════════════════════════════════════════════════════

    fn save_checkpoint(self, path: String) raises:
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

    fn load_checkpoint(mut self, path: String) raises:
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

    fn start_new_iteration(mut self):
        """Mark the start of a new self-play iteration.

        Evicts oldest iteration data when history_window is exceeded.
        Call this before each self-play collection phase.
        """
        self.state.start_new_iteration()

    # ══════════════════════════════════════════════════════════════
    # GPU Training Step (simple supervised learning)
    # ══════════════════════════════════════════════════════════════

    fn train_step_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu: Self.GPUStateType,
    ) raises:
        """One training step: sample batch from CPU replay, train on GPU.

        loss = CE(policy, mcts_π) + MSE(value, outcome)
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
                gpu.policy_host[b * ACT + i] = self.state.buf_policy[idx * ACT + i]
            gpu.value_host[b] = self.state.buf_value[idx]

        # ── Upload to GPU ────────────────────────────────────────
        ctx.enqueue_copy(gpu.batch_obs, gpu.obs_host)
        ctx.enqueue_copy(gpu.batch_policy, gpu.policy_host)
        ctx.enqueue_copy(gpu.batch_value, gpu.value_host)

        # ── Forward with cache ───────────────────────────────────
        var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin](gpu.batch_obs.unsafe_ptr())
        var pred_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin](gpu.pred_out.unsafe_ptr())
        var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin](gpu.pred_cache.unsafe_ptr())

        Self.PredNet.forward_gpu_with_cache[BATCH](
            ctx, obs_t, pred_t, gpu.prediction.params_view(), cache_t, gpu.workspace
        )

        # ── Compute gradient on GPU ──────────────────────────────
        var grad_1d = LayoutTensor[dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin](gpu.grad_out.unsafe_ptr())
        var pred_1d = LayoutTensor[dtype, Layout.row_major(BATCH * Self.PRED_OUT), MutAnyOrigin](gpu.pred_out.unsafe_ptr())
        var pol_1d = LayoutTensor[dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin](gpu.batch_policy.unsafe_ptr())
        var val_1d = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](gpu.batch_value.unsafe_ptr())

        comptime run_grad = az_policy_value_grad_kernel[BATCH, ACT, Self.PRED_OUT, dtype]
        ctx.enqueue_function[run_grad, run_grad](
            grad_1d, pred_1d, pol_1d, val_1d,
            grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
        )

        # ── Backward + optimizer ─────────────────────────────────
        var grad_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin](gpu.grad_out.unsafe_ptr())
        ctx.enqueue_memset(gpu.grad_in, 0)
        var grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin](gpu.grad_in.unsafe_ptr())

        gpu.prediction.zero_grads(ctx)
        var grads = gpu.prediction.grads_view()
        Self.PredNet.backward_gpu[BATCH](
            ctx, grad_out_t, grad_in_t,
            gpu.prediction.params_view(), cache_t, grads, gpu.workspace
        )
        gpu.prediction.optimizer_step(ctx)
        self.train_step_count += 1

    # ══════════════════════════════════════════════════════════════
    # Data Augmentation (via DataAugmentable trait on environment)
    # ══════════════════════════════════════════════════════════════

    fn _add_with_augmentation[
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

    fn arena_compare[
        E: TwoPlayerDiscreteEnv,
        origin: MutOrigin,
    ](
        mut self,
        mut env: E,
        mut prev_params: UnsafePointer[Scalar[dtype], origin],
        num_games: Int = 40,
        threshold: Float64 = 0.55,
    ) -> Bool:
        """Play current model vs previous model. Accept if win rate >= threshold.

        Saves/restores network params to compare. Uses policy-only action
        selection (no MCTS) for speed during comparison.

        Args:
            env: Environment for playing games.
            prev_params: Saved parameters of the previous best model.
            num_games: Games to play (half as each side).
            threshold: Win fraction needed to accept new model.

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
        var prev_wins = 0

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
            if result == 1:
                if new_is_p0:
                    new_wins += 1
                else:
                    prev_wins += 1
            elif result == 2:
                if new_is_p0:
                    prev_wins += 1
                else:
                    new_wins += 1

        # Restore new params (we'll decide whether to keep or revert)
        for i in range(PS):
            self.state.prediction.params[i] = new_params[i]
        new_params.free()

        var win_rate = Float64(new_wins) / Float64(num_games)
        var accepted = win_rate >= threshold

        if not accepted:
            # Revert to previous params
            for i in range(PS):
                self.state.prediction.params[i] = prev_params[i]

        return accepted

    # ══════════════════════════════════════════════════════════════
    # Self-Play GPU Training
    # ══════════════════════════════════════════════════════════════

    fn train_selfplay_gpu[
        E: GPUTwoPlayerDiscreteEnv & DataAugmentable,
        ArenaEnv: TwoPlayerDiscreteEnv,
    ](
        mut self,
        ctx: DeviceContext,
        mut arena_env: ArenaEnv,
        num_steps: Int = 500000,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 1,
        print_every: Int = 10000,
        lr_decay_every: Int = 0,
        lr_decay_factor: Float64 = 0.5,
        arena_every: Int = 0,
        arena_games: Int = 40,
        arena_threshold: Float64 = 0.55,
    ) raises -> TrainingMetrics:
        """Train via GPU self-play with true game rules MCTS.

        Each iteration:
          1. GPU MCTS with env.step_kernel_gpu for expansion
          2. GPU env step
          3. Collect (obs, mcts_policy, game_outcome) on CPU
          4. GPU supervised training on collected data
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime SIMS = Self.Config.num_simulations
        comptime MAX_NODES = Self.Config.max_nodes
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime PRED_IN = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM

        # GPU state
        var gpu = Self.GPUStateType(ctx)
        gpu.upload_from(self.state, ctx)

        # GPU MCTS state (with game states for true-rules expansion)
        # Use a minimal LATENT=OBS since we run PredNet on obs directly
        var gpu_mcts = GPUMCTSState[Self.n_envs, MAX_NODES, ACT, OBS, 1, E.STATE_SIZE](ctx)

        # Network workspace (sized for batched prediction: n_envs * BATCH_SIMS)
        comptime BATCH_SIMS_C = 8  # Must match BATCH_SIMS in simulation loop
        comptime WS = Self.Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_SIZE = Self.n_envs * BATCH_SIMS_C * WS if WS > 0 else 1
        var mcts_ws = ctx.enqueue_create_buffer[dtype](WS_SIZE)

        # Env buffers
        var states_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * E.STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var legal_masks_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * ACT)

        # Episode tracking
        var ep_rew_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var ep_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rew_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var ep_count_buf = ctx.enqueue_create_buffer[dtype](1)
        var rew_sum_host = ctx.enqueue_create_host_buffer[dtype](1)
        var ep_count_host = ctx.enqueue_create_host_buffer[dtype](1)

        # Expansion scratch for batched MCTS (BATCH_SIMS per env)
        comptime BATCH_SIMS_ALLOC = 8  # Must match BATCH_SIMS in simulation loop
        comptime TOTAL_EXPAND_ALLOC = Self.n_envs * BATCH_SIMS_ALLOC
        var exp_rewards = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC)
        var exp_dones = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC)
        var exp_terminated = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC)
        var exp_obs = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND_ALLOC * OBS)

        # Host buffers for collecting self-play data
        var obs_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs * OBS)
        var policy_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs * ACT)
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)

        # Per-env episode data for computing game outcomes
        var env_rewards = List[List[Float64]](capacity=Self.n_envs)
        var env_obs_history = List[List[List[Scalar[dtype]]]](capacity=Self.n_envs)
        var env_policy_history = List[List[List[Scalar[dtype]]]](capacity=Self.n_envs)
        for _ in range(Self.n_envs):
            env_rewards.append(List[Float64]())
            env_obs_history.append(List[List[Scalar[dtype]]]())
            env_policy_history.append(List[List[Scalar[dtype]]]())

        # Initialize
        E.reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=42)
        E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
            ctx, states_buf, obs_buf, legal_masks_buf
        )
        ep_rew_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_steps_buf.enqueue_fill(Scalar[dtype](0.0))
        rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var metrics = TrainingMetrics(algorithm_name="AlphaZero")
        var total_steps = 0
        var next_print = print_every
        var next_arena = arena_every if arena_every > 0 else num_steps + 1

        # Save initial params for arena comparison
        comptime PS = Self.Config.PredModel.PARAM_SIZE
        var best_params = alloc[Scalar[dtype]](PS)
        for i in range(PS):
            best_params[i] = self.state.prediction.params[i]
        var arena_accepts = 0
        var arena_rejects = 0

        while total_steps < num_steps:
            # ── 1. Download obs for episode tracking ─────────────
            ctx.enqueue_copy(obs_host, obs_buf)
            ctx.synchronize()

            # ── 2. GPU MCTS with true game rules ────────────────
            if total_steps >= warmup_steps:
                # Prediction on obs for root
                var pred_obs = LayoutTensor[dtype, Layout.row_major(Self.n_envs, PRED_IN), MutAnyOrigin](obs_buf.unsafe_ptr())
                var pred_out = LayoutTensor[dtype, Layout.row_major(Self.n_envs, PRED_OUT_DIM), MutAnyOrigin](gpu_mcts.pred_output.unsafe_ptr())
                Self.PredNet.forward_gpu[Self.n_envs](
                    ctx, pred_obs, pred_out, gpu.prediction.params_view(), mcts_ws
                )

                # Init root
                comptime MCTS_PRED_OUT = ACT + 1
                var vc = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * ACT), MutAnyOrigin](gpu_mcts.visit_count.unsafe_ptr())
                var tv = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * ACT), MutAnyOrigin](gpu_mcts.total_value.unsafe_ptr())
                var pr = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * ACT), MutAnyOrigin](gpu_mcts.prior.unsafe_ptr())
                var rw = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * ACT), MutAnyOrigin](gpu_mcts.reward.unsafe_ptr())
                var ci = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * ACT), MutAnyOrigin](gpu_mcts.child_idx.unsafe_ptr())
                var tvis = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES), MutAnyOrigin](gpu_mcts.total_visits.unsafe_ptr())
                var nc = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](gpu_mcts.node_count.unsafe_ptr())
                var po = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MCTS_PRED_OUT), MutAnyOrigin](gpu_mcts.pred_output.unsafe_ptr())
                var miq = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](gpu_mcts.min_q.unsafe_ptr())
                var mxq = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](gpu_mcts.max_q.unsafe_ptr())

                comptime run_init = gpu_mcts_init_root_kernel[Self.n_envs, MAX_NODES, ACT, OBS, MCTS_PRED_OUT, dtype]
                ctx.enqueue_function[run_init, run_init](
                    vc, tv, pr, rw, ci, tvis, nc, po, miq, mxq,
                    Scalar[dtype](Self.Config.Noise.NOISE_FRACTION),
                    Scalar[DType.uint32](UInt32(total_steps)),
                    grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
                )

                # Legal mask on root
                var lm = LayoutTensor[dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin](legal_masks_buf.unsafe_ptr())
                comptime run_mask = gpu_mcts_apply_legal_mask_kernel[Self.n_envs, MAX_NODES, ACT, dtype]
                ctx.enqueue_function[run_mask, run_mask](pr, lm, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))

                # Copy root game states
                comptime GS = E.STATE_SIZE
                var gs = LayoutTensor[dtype, Layout.row_major(Self.n_envs * MAX_NODES * GS), MutAnyOrigin](gpu_mcts.game_states.unsafe_ptr())
                var es = LayoutTensor[dtype, Layout.row_major(Self.n_envs * GS), MutAnyOrigin](states_buf.unsafe_ptr())
                comptime run_rs = gpu_mcts_copy_root_state_kernel[Self.n_envs, MAX_NODES, GS, dtype]
                ctx.enqueue_function[run_rs, run_rs](gs, es, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))

                # ── Batched simulations (BATCH_SIMS leaves per round) ──
                comptime BATCH_SIMS = 8
                comptime NUM_ROUNDS = SIMS // BATCH_SIMS
                comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

                # Batched buffers [N_ENVS * BATCH_SIMS * ...]
                var b_pp = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin](gpu_mcts.pending_parent.unsafe_ptr())
                var b_pa = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin](gpu_mcts.pending_action.unsafe_ptr())
                var b_sp = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin](gpu_mcts.search_paths.unsafe_ptr())
                var b_ap = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin](gpu_mcts.action_paths.unsafe_ptr())
                var b_pl = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin](gpu_mcts.path_lengths.unsafe_ptr())
                var b_exp_st = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND * GS), MutAnyOrigin](gpu_mcts.expansion_states.unsafe_ptr())

                for _round in range(NUM_ROUNDS):
                    # 1. Fused select + copy (1 kernel, selects BATCH_SIMS leaves per env)
                    comptime run_sel_cp = gpu_mcts_batched_select_and_copy_kernel[
                        Self.n_envs, MAX_NODES, ACT, BATCH_SIMS, GS, dtype
                    ]
                    ctx.enqueue_function[run_sel_cp, run_sel_cp](
                        vc, tv, pr, ci, tvis, nc, miq, mxq, gs,
                        b_pp, b_pa, b_exp_st, b_sp, b_ap, b_pl,
                        Scalar[dtype](Self.Config.PUCT.C_BASE),
                        Scalar[dtype](Self.Config.PUCT.C_INIT),
                        grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
                    )

                    # 2. Batched env.step [TOTAL_EXPAND = n_envs * BATCH_SIMS]
                    E.step_kernel_gpu[TOTAL_EXPAND, GS, OBS](
                        ctx, gpu_mcts.expansion_states, gpu_mcts.pending_action,
                        exp_rewards, exp_dones, exp_terminated, exp_obs,
                        gpu_mcts.expansion_legal_masks,
                        rng_seed=UInt64(total_steps * NUM_ROUNDS + _round),
                    )

                    # 3. Batched prediction [BATCH = n_envs * BATCH_SIMS]
                    var p_in = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND, PRED_IN), MutAnyOrigin](exp_obs.unsafe_ptr())
                    var p_out = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM), MutAnyOrigin](gpu_mcts.pred_output.unsafe_ptr())
                    Self.PredNet.forward_gpu[TOTAL_EXPAND](
                        ctx, p_in, p_out, gpu.prediction.params_view(), mcts_ws
                    )

                    # 4. Fused expand + backup + remove virtual losses (1 kernel)
                    var b_po = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND * MCTS_PRED_OUT), MutAnyOrigin](gpu_mcts.pred_output.unsafe_ptr())
                    var b_rew = LayoutTensor[dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin](exp_rewards.unsafe_ptr())
                    comptime run_exp_bk = gpu_mcts_batched_expand_backup_kernel[
                        Self.n_envs, MAX_NODES, ACT, BATCH_SIMS, MCTS_PRED_OUT, GS, dtype
                    ]
                    ctx.enqueue_function[run_exp_bk, run_exp_bk](
                        vc, tv, pr, rw, ci, tvis, nc, miq, mxq,
                        gs, b_exp_st, b_pp, b_pa, b_po, b_rew,
                        b_sp, b_ap, b_pl,
                        grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
                    )

                # Extract actions (legal only)
                var act_out = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](actions_buf.unsafe_ptr())
                var pol_out = LayoutTensor[dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin](gpu_mcts.policies_out.unsafe_ptr())
                comptime run_act = gpu_mcts_extract_actions_masked_kernel[Self.n_envs, MAX_NODES, ACT, dtype]
                ctx.enqueue_function[run_act, run_act](vc, lm, act_out, pol_out, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))

                # Download policies for training data
                ctx.enqueue_copy(policy_host, gpu_mcts.policies_out)
            else:
                # Warmup: random legal actions
                comptime run_warmup = uniform_random_discrete_actions_kernel[dtype, Self.n_envs, ACT]
                var wa = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](actions_buf.unsafe_ptr())
                ctx.enqueue_function[run_warmup, run_warmup](
                    wa, Scalar[DType.uint32](UInt32(total_steps)),
                    grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
                )
                # Uniform policies for warmup
                for i in range(Self.n_envs * ACT):
                    policy_host[i] = Scalar[dtype](1.0 / Float64(ACT))

            # ── 3. Env step ──────────────────────────────────────
            E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf,
                terminated_buf, obs_buf, legal_masks_buf,
                rng_seed=UInt64(total_steps),
            )

            # ── 4. Episode tracking ──────────────────────────────
            var rew_t = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](rewards_buf.unsafe_ptr())
            var don_t = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](dones_buf.unsafe_ptr())
            var epr_t = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](ep_rew_buf.unsafe_ptr())
            var eps_t = LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin](ep_steps_buf.unsafe_ptr())
            comptime run_accum = accumulate_rewards_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_accum, run_accum](epr_t, rew_t, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))
            comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_incr, run_incr](eps_t, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))
            var rs_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](rew_sum_buf.unsafe_ptr())
            var ec_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](ep_count_buf.unsafe_ptr())
            comptime run_log = log_and_reset_completed_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_log, run_log](don_t, epr_t, eps_t, rs_t, ec_t, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,))

            # ── 5. Collect training data ─────────────────────────
            ctx.enqueue_copy(rewards_host, rewards_buf)
            ctx.enqueue_copy(dones_host, dones_buf)
            ctx.synchronize()

            # Only store data after warmup (warmup produces uniform policies = noise)
            if total_steps >= warmup_steps:
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

                    if Float64(dones_host[e]) > 0.5:
                        # Game ended — compute outcome and add all moves + symmetries
                        var last_reward = Float64(rewards_host[e])
                        var ep_len = len(env_obs_history[e])

                        for t in range(ep_len):
                            var steps_from_end = ep_len - 1 - t
                            var outcome: Float64
                            if steps_from_end % 2 == 0:
                                outcome = last_reward
                            else:
                                outcome = -last_reward

                            # Add original + all symmetries (up to 8x for 3x3 boards)
                            self._add_with_augmentation[E](
                                env_obs_history[e][t],
                                env_policy_history[e][t],
                                Scalar[dtype](outcome),
                            )

                        env_obs_history[e].clear()
                        env_policy_history[e].clear()
                        env_rewards[e].clear()
            else:
                # During warmup, just clear episode data on done (don't store)
                for e in range(Self.n_envs):
                    if Float64(dones_host[e]) > 0.5:
                        env_obs_history[e].clear()
                        env_policy_history[e].clear()
                        env_rewards[e].clear()

            # ── 6. Selective reset ───────────────────────────────
            E.selective_reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
                ctx, states_buf, dones_buf, rng_seed=UInt64(total_steps),
            )
            E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                ctx, states_buf, obs_buf, legal_masks_buf,
            )

            total_steps += Self.n_envs
            self.total_steps += Self.n_envs

            # ── 7. Training ──────────────────────────────────────
            if total_steps >= warmup_steps and self.state.is_ready(Self.Config.batch_size):
                for _ in range(gradient_steps):
                    self.train_step_gpu(ctx, gpu)

                # LR step-decay
                if lr_decay_every > 0 and self.train_step_count > 0:
                    if self.train_step_count % lr_decay_every == 0:
                        gpu.prediction.lr_scale *= lr_decay_factor

            # ── 7b. Arena comparison + new iteration ──────────────
            if arena_every > 0 and total_steps >= next_arena:
                gpu.download_to(self.state, ctx)
                self.state.start_new_iteration()
                var accepted = self.arena_compare[ArenaEnv](
                    arena_env, best_params,
                    num_games=arena_games, threshold=arena_threshold,
                )
                if accepted:
                    # Save new best params
                    for i in range(PS):
                        best_params[i] = self.state.prediction.params[i]
                    gpu.upload_from(self.state, ctx)
                    arena_accepts += 1
                else:
                    # Reverted to best params — re-upload
                    gpu.upload_from(self.state, ctx)
                    arena_rejects += 1
                next_arena += arena_every

            # ── 8. Progress ──────────────────────────────────────
            if total_steps >= next_print:
                ctx.enqueue_copy(rew_sum_host, rew_sum_buf)
                ctx.enqueue_copy(ep_count_host, ep_count_buf)
                ctx.synchronize()
                var total_eps = Int(Float64(ep_count_host[0]))
                gpu.download_to(self.state, ctx)  # Sync for eval
                clear_progress_bar()
                print(
                    "Steps:", total_steps,
                    "| Games:", total_eps,
                    "| Train:", self.train_step_count,
                    "| Replay:", self.state.buf_size,
                )
                rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
                ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
                next_print += print_every

        gpu.download_to(self.state, ctx)
        best_params.free()
        if arena_every > 0:
            print("Arena: accepted", arena_accepts, "/ rejected", arena_rejects)
        return metrics
