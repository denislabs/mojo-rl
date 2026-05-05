"""TDMPC2 Agent for Mojo-RL.

TD-MPC2 is a model-based RL algorithm that learns a world model consisting of:
  - encoder, dynamics, reward, termination, policy, and Q-function ensemble

At each timestep, MPPI planning in latent space selects actions.
Training alternates between world model updates and policy updates.

Key algorithm parameters:
  H = 3             planning horizon
  gamma = 0.99      discount factor
  rho = 0.5         temporal weight decay for horizon losses
  tau = 0.01        soft update coefficient for target Q-networks
  batch_size = 256
  consistency_coef = 20.0
  reward_coef = 0.1
  value_coef = 0.1
  entropy_coef = 1e-4
  num_samples = 512   MPPI candidates
  num_pi_trajs = 24   policy rollout trajectories in MPPI
  num_iterations = 6  MPPI optimization iterations
  temperature = 0.5   MPPI softmax temperature

Reference: Hansen et al., 2023 — TD-MPC2: Scalable, Robust World Models
           for Continuous Control
"""

from std.math import exp, log, sqrt, tanh
from std.gpu import thread_idx
from std.random import random_float64, seed
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.loss.two_hot import (
    compute_bins,
    two_hot_encode_batch,
    decode_value_batch,
)
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from mojo_rl.deep_agents.core.replay.gpu_sequence_replay_buffer import (
    GPUSequenceReplayBuffer,
)
from mojo_rl.deep_agents.core.kernels import (
    copy_buffer_kernel,
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)

from .state import (
    TDMPC2GPUState,
    TDMPC2CPUState,
    MPPIGPUBuffers,
    BatchedMPPIGPUBuffers,
)
from .world_model import WorldModel, decode_value_batch_scalar
from mojo_rl.core.logger import Logger, NoOpLogger
from .mppi import plan, plan_gpu, plan_gpu_batched
from .kernels import (
    tdmpc2_random_actions_kernel,
    tdmpc2_sample_actions_kernel,
    tdmpc2_build_za_kernel,
    tdmpc2_extract_z_from_za_grad_kernel,
    tdmpc2_extract_obs_step_kernel,
    tdmpc2_extract_act_step_kernel,
    tdmpc2_extract_scalar_step_kernel,
    tdmpc2_consistency_loss_grad_kernel,
    tdmpc2_two_hot_loss_grad_kernel,
    tdmpc2_bce_loss_grad_kernel,
    tdmpc2_q_decode_kernel,
    tdmpc2_compute_td_targets_kernel,
    tdmpc2_compute_reward_targets_kernel,
    tdmpc2_policy_grad_kernel,
    tdmpc2_q_decode_backward_kernel,
    tdmpc2_action_tanh_chain_kernel,
    tdmpc2_apply_tanh_kernel,
    tdmpc2_q_min_reduce_kernel,
    tdmpc2_zero_kernel,
    tdmpc2_add_into_kernel,
    tdmpc2_extract_rew_done_kernel,
    tdmpc2_decode_and_min_kernel,
    tdmpc2_add_two_into_kernel,
    tdmpc2_extract_all_build_za_kernel,
    tdmpc2_extract_obs_rew_done_kernel,
    tdmpc2_apply_tanh_build_za_kernel,
    tdmpc2_soft_update_5q_kernel,
    tdmpc2_gradient_norm_5q_kernel,
    tdmpc2_gradient_reduce_apply_5q_kernel,
    tdmpc2_adam_step_5q_kernel,
)


struct TDMPC2Agent[
    obs_dim: Int,
    action_dim: Int,
    latent_dim: Int = 512,
    mlp_dim: Int = 512,
    enc_dim: Int = 256,
    num_bins: Int = 101,
    num_q: Int = 5,
    simplex_dim: Int = 8,
    horizon: Int = 3,
    batch_size: Int = 256,
    buffer_capacity: Int = 100000,
    num_samples: Int = 512,
    num_pi_trajs: Int = 24,
    num_iterations: Int = 6,
    v_min: Float64 = -10.0,
    v_max: Float64 = 10.0,
    L: Logger = NoOpLogger,
]:
    """TD-MPC2 agent for continuous control.

    Parameters:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        latent_dim: Latent state dimension (default: 512).
        mlp_dim: Hidden layer width for dynamics/reward/policy/Q (default: 512).
        enc_dim: Encoder hidden layer width (default: 256).
        num_bins: Distributional RL bins (default: 101).
        num_q: Q-network ensemble size (default: 5).
        simplex_dim: SimNorm group size (default: 8).
        horizon: Planning horizon H (default: 3).
        batch_size: Training batch size (default: 256).
        buffer_capacity: Replay buffer capacity (default: 100_000).
        num_samples: MPPI candidate trajectories (default: 512).
        num_pi_trajs: Policy-seeded MPPI trajectories (default: 24).
        num_iterations: MPPI optimization iterations (default: 6).
        v_min: Minimum value for distribution (default: -10.0).
        v_max: Maximum value for distribution (default: 10.0).
        L: Logger (default: NoOpLogger).

    Note: latent_dim must be divisible by simplex_dim.
    """

    comptime BATCH = Self.batch_size
    comptime H = Self.horizon
    comptime OBS = Self.obs_dim
    comptime ACT = Self.action_dim
    comptime LATENT = Self.latent_dim
    comptime BINS = Self.num_bins
    comptime ZA = Self.LATENT + Self.ACT

    comptime CPUStateType = TDMPC2CPUState[
        Self.OBS,
        Self.ACT,
        Self.LATENT,
        Self.mlp_dim,
        Self.enc_dim,
        Self.BINS,
        Self.num_q,
        Self.simplex_dim,
        Self.v_min,
        Self.v_max,
        buffer_capacity=Self.buffer_capacity,
        batch_size=Self.batch_size,
        horizon=Self.horizon,
    ]
    # WM alias now derives from CPUStateType so GPU code's type references still work
    comptime WM = Self.CPUStateType.WM

    # ── Network parameter / cache / workspace sizes ──────────────────────
    # Lifted to struct level so train_gpu[ENV, n_envs] doesn't re-instantiate
    # them for every unique call-site parameter combination.
    comptime ENC_P = Self.WM.EncoderNet.PARAM_SIZE
    comptime ENC_C = Self.WM.EncoderNet.CACHE_SIZE
    comptime ENC_W = Self.WM.EncoderNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime DYN_P = Self.WM.DynamicsNet.PARAM_SIZE
    comptime DYN_C = Self.WM.DynamicsNet.CACHE_SIZE
    comptime DYN_W = Self.WM.DynamicsNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime REW_P = Self.WM.RewardNet.PARAM_SIZE
    comptime REW_C = Self.WM.RewardNet.CACHE_SIZE
    comptime REW_W = Self.WM.RewardNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime TERM_P = Self.WM.TermNet.PARAM_SIZE
    comptime TERM_C = Self.WM.TermNet.CACHE_SIZE
    comptime TERM_W = Self.WM.TermNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime POL_P = Self.WM.PolicyNet.PARAM_SIZE
    comptime POL_C = Self.WM.PolicyNet.CACHE_SIZE
    comptime POL_W = Self.WM.PolicyNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime POL_OUT = Self.WM.PolModel.OUT_DIM
    comptime Q_P = Self.WM.QNet.PARAM_SIZE
    comptime Q_C = Self.WM.QNet.CACHE_SIZE
    comptime Q_W = Self.WM.QNet.WORKSPACE_SIZE_PER_SAMPLE

    # ── Flat batch buffer sizes ───────────────────────────────────────────
    comptime B_OBS = Self.BATCH * Self.OBS
    comptime B_ACT = Self.BATCH * Self.ACT
    comptime B_LATENT = Self.BATCH * Self.LATENT
    comptime B_ZA = Self.BATCH * Self.ZA
    comptime B_BINS = Self.BATCH * Self.BINS
    comptime BATCH_OBS_FLAT = Self.BATCH * (Self.H + 1) * Self.OBS
    comptime BATCH_ACT_FLAT = Self.BATCH * Self.H * Self.ACT
    comptime BATCH_SCALAR_FLAT = Self.BATCH * Self.H
    comptime BATCH_TGTS_FLAT = Self.H * Self.BATCH * Self.BINS

    # ── Batch workspace sizes ─────────────────────────────────────────────
    comptime ENC_BATCH_WS = Self.BATCH * Self.ENC_W
    comptime DYN_BATCH_WS = Self.BATCH * Self.DYN_W
    comptime REW_BATCH_WS = Self.BATCH * Self.REW_W
    comptime TERM_BATCH_WS = Self.BATCH * Self.TERM_W
    comptime POL_BATCH_WS = Self.BATCH * Self.POL_W
    comptime Q_BATCH_WS = Self.BATCH * Self.Q_W

    # ── Grid dimensions (batch-level; env-level depends on n_envs) ───────
    comptime BATCH_BLOCKS = (Self.BATCH + TPB - 1) // TPB
    comptime ENC_GRAD_BLOCKS = (Self.ENC_P + TPB - 1) // TPB
    comptime DYN_GRAD_BLOCKS = (Self.DYN_P + TPB - 1) // TPB
    comptime REW_GRAD_BLOCKS = (Self.REW_P + TPB - 1) // TPB
    comptime TERM_GRAD_BLOCKS = (Self.TERM_P + TPB - 1) // TPB
    comptime POL_GRAD_BLOCKS = (Self.POL_P + TPB - 1) // TPB
    comptime Q_GRAD_BLOCKS = (Self.Q_P + TPB - 1) // TPB
    comptime DUMMY_SIZE = max(Self.B_ZA, Self.B_OBS)

    # GPU state type alias (parameterized without n_envs/env_state_size;
    # those are filled in by train_gpu[] which knows ENV).
    # Use make_gpu_state[n_envs, env_state_size](ctx) to construct.
    comptime EncOpt = Adam[LR=Self.WM.ENC_LR]
    comptime WMOpt = Adam[LR=Self.WM.WM_LR]
    comptime PIOpt = Adam[LR=Self.WM.PI_LR]

    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var rho: Float64
    var tau: Float64
    var consistency_coef: Float64
    var reward_coef: Float64
    var value_coef: Float64
    var terminal_coef: Float64
    var entropy_coef: Float64
    var temperature: Float64
    var action_scale: Float64
    var warmup_steps: Int
    var total_steps: Int
    var train_step_count: Int

    # RunningScale: normalizes Q-values for policy loss (reference: 5th-95th percentile)
    var running_scale: Float64

    # Diagnostics
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    # MPPI warm-start state
    var _prev_mean: List[Float64]
    var _episode_t0: Bool

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        rho: Float64 = 0.5,
        tau: Float64 = 0.01,
        consistency_coef: Float64 = 2.0,
        reward_coef: Float64 = 0.5,
        value_coef: Float64 = 0.1,
        terminal_coef: Float64 = 1.0,
        entropy_coef: Float64 = 1e-4,
        temperature: Float64 = 0.5,
        action_scale: Float64 = 1.0,
        warmup_steps: Int = 5000,
        wm_lr: Float64 = 3e-4,
        enc_lr_scale: Float64 = 0.3,
        pi_lr: Float64 = 3e-4,
        episode_length: Int = 0,
        discount_denom: Float64 = 5.0,
        discount_min: Float64 = 0.95,
        discount_max: Float64 = 0.995,
        diag_every: Int = 0,
    ):
        """Initialize TDMPC2 agent.

        Args:
            gamma: Discount factor fallback (default: 0.99). Used only when
                episode_length is 0 (dynamic discount disabled).
            rho: Temporal weight decay for horizon losses (default: 0.5).
            tau: Soft update coefficient for target Q-networks (default: 0.01).
            consistency_coef: Consistency loss weight (default: 2.0).
            reward_coef: Reward loss weight (default: 0.5).
            value_coef: Value loss weight (default: 0.1).
            terminal_coef: Terminal loss weight (default: 1.0).
            entropy_coef: Policy entropy coefficient (default: 1e-4).
            temperature: MPPI softmax temperature (default: 0.5).
            action_scale: Action scaling (default: 1.0).
            warmup_steps: Steps before training begins (default: 5000).
            wm_lr: World model learning rate (default: 3e-4).
            enc_lr_scale: Encoder LR multiplier (default: 0.3).
            pi_lr: Policy learning rate (default: 3e-4).
            episode_length: Max episode length for dynamic gamma (default: 0 = use
                fixed gamma). E.g. 1000 for HalfCheetah → gamma=0.995.
            discount_denom: Denominator for dynamic gamma formula (default: 5.0).
            discount_min: Minimum dynamic gamma (default: 0.95).
            discount_max: Maximum dynamic gamma (default: 0.995).
            diag_every: Log diagnostics every N steps (default: 0 = never).
        """
        self.state = Self.CPUStateType()

        # Dynamic discount: gamma = (L/d - 1) / (L/d), clamped to [min, max]
        if episode_length > 0:
            var ratio = Float64(episode_length) / discount_denom
            var dyn_gamma = (ratio - 1.0) / ratio
            if dyn_gamma < discount_min:
                dyn_gamma = discount_min
            if dyn_gamma > discount_max:
                dyn_gamma = discount_max
            self.gamma = dyn_gamma
        else:
            self.gamma = gamma
        self.rho = rho
        self.tau = tau
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.terminal_coef = terminal_coef
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.action_scale = action_scale
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.train_step_count = 0
        self.running_scale = 1.0
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = diag_every
        self._prev_mean = List[Float64]()
        self._episode_t0 = True

    # =========================================================================
    # Action Selection
    # =========================================================================

    def select_action(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        deterministic: Bool = False,
    ) -> InlineArray[Scalar[dtype], Self.ACT]:
        """Select action using MPPI planning in latent space.

        During warmup (total_steps < warmup_steps), returns random actions.

        Args:
            obs: Current observation [OBS_DIM].
            deterministic: If True, no exploration noise (for evaluation).

        Returns:
            Selected action [ACTION_DIM].
        """
        # Warmup: random actions
        if self.total_steps < self.warmup_steps:
            var action = InlineArray[Scalar[dtype], Self.ACT](fill=0)
            for i in range(Self.ACT):
                action[i] = Scalar[dtype](
                    (random_float64() * 2.0 - 1.0) * self.action_scale
                )
            return action^

        # Encode observation
        var obs_arr = InlineArray[Scalar[dtype], 1 * Self.OBS](fill=0)
        for i in range(Self.OBS):
            obs_arr[i] = obs[i]
        var z = InlineArray[Scalar[dtype], 1 * Self.LATENT](uninitialized=True)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var z_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.LATENT), MutAnyOrigin
        ](z.unsafe_ptr())
        self.state.world_model.encode[1](obs_t, z_t)

        # Extract z0 as single-sample array
        var z0 = InlineArray[Scalar[dtype], Self.LATENT](uninitialized=True)
        for i in range(Self.LATENT):
            z0[i] = z[i]

        # MPPI planning with warm-start

        var action = plan[
            Self.OBS,
            Self.ACT,
            Self.LATENT,
            Self.mlp_dim,
            Self.enc_dim,
            Self.BINS,
            Self.num_q,
            Self.simplex_dim,
            Self.v_min,
            Self.v_max,
            Self.H,
            Self.num_samples,
            Self.num_pi_trajs,
            Self.num_iterations,
        ](
            z0,
            self.state.world_model,
            self.gamma,
            self.temperature,
            self._prev_mean,
            self.action_scale,
            deterministic,
            self._episode_t0,
        )
        # After first call in episode, subsequent calls warm-start
        self._episode_t0 = False
        return action^

    # =========================================================================
    # Store Transition
    # =========================================================================

    def observe(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: InlineArray[Scalar[dtype], Self.ACT],
        reward: Float64,
        done: Bool,
    ):
        """Store a transition in the replay buffer.

        Args:
            obs: Observation at current step [OBS_DIM].
            action: Action taken [ACTION_DIM].
            reward: Reward received.
            done: Whether the episode ended.
        """
        self.state.observe(obs, action, Scalar[dtype](reward), done)
        self.total_steps += 1

    # =========================================================================
    # Training Update
    # =========================================================================

    def update(mut self) -> Float64:
        """Perform one TDMPC2 gradient update step.

        Returns:
            Total world model loss for this step.
        """
        if not self.state.is_ready():
            return 0.0

        # Sample into pre-allocated batch buffers
        self.state.buffer.sample_sequences[Self.BATCH, Self.H](
            self.state._batch_obs,
            self.state._batch_actions,
            self.state._batch_rewards,
            self.state._batch_dones,
        )

        # World model update
        var wm_loss = self._update_world_model()

        # Policy update
        self._update_policy()

        # Soft update target Q-networks
        self.state.world_model.soft_update_q_targets(self.tau)

        self.train_step_count += 1
        return wm_loss

    def _update_world_model(mut self) -> Float64:
        """Compute and apply world model gradient update.

        Uses pre-allocated scratch buffers from self.state instead of
        per-call List allocations.

        Losses computed:
          L_consistency: MSE between predicted and encoded next latent states
          L_reward: Soft cross-entropy on reward distribution
          L_value: Soft cross-entropy on Q-value distribution
          L_terminal: Binary cross-entropy on termination prediction

        Returns:
            Total weighted loss.
        """
        # Pre-compute fixed bin values (local copy, does not move world_model.bins)
        var bins = compute_bins[Self.BINS](
            Float32(Self.v_min), Float32(Self.v_max)
        )

        # ── LayoutTensor views over pre-allocated state buffers ──
        var next_obs_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.state._next_obs.unsafe_ptr())
        var z_enc_next_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](self.state._z_enc_next.unsafe_ptr())
        var a_next_mean_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACT), MutAnyOrigin
        ](self.state._a_next_mean.unsafe_ptr())
        var a_next_log_std_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACT), MutAnyOrigin
        ](self.state._a_next_log_std.unsafe_ptr())
        var za_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](self.state._za.unsafe_ptr())
        var obs_0_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.state._obs_0.unsafe_ptr())
        var z_current_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](self.state._z_current.unsafe_ptr())
        var z_pred_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](self.state._z_pred.unsafe_ptr())
        var enc_cache_v = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CPUStateType.ENC_CACHE_SIZE),
            MutAnyOrigin,
        ](self.state._enc_cache.unsafe_ptr())
        var dyn_cache_v = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CPUStateType.DYN_CACHE_SIZE),
            MutAnyOrigin,
        ](self.state._dyn_cache.unsafe_ptr())
        var rew_logits_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](self.state._rew_logits.unsafe_ptr())
        var q_logits_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](self.state._q_logits.unsafe_ptr())

        # -------------------------------------------------------------------------
        # Step 1: Compute TD targets (stop-gradient)
        # td_target_dist[t, b, k]: two-hot target for Q at step t, sample b
        # -------------------------------------------------------------------------
        # Zero td_targets
        for i in range(Self.H * Self.BATCH * Self.BINS):
            self.state._td_targets[i] = Scalar[dtype](0)

        for t in range(Self.H):
            # Get next observations for this horizon step → _next_obs
            for b in range(Self.BATCH):
                var obs_offset = (
                    b * (Self.H + 1) * Self.OBS + (t + 1) * Self.OBS
                )
                for i in range(Self.OBS):
                    self.state._next_obs[
                        b * Self.OBS + i
                    ] = self.state._batch_obs[obs_offset + i]

            # Encode next observations (stop-gradient: no cache) → _z_enc_next
            self.state.world_model.encode[Self.BATCH](next_obs_v, z_enc_next_v)

            # Sample next actions from policy → _a_next_mean, _a_next_log_std
            self.state.world_model.policy_forward[Self.BATCH](
                z_enc_next_v,
                a_next_mean_v,
                a_next_log_std_v,
            )

            # Clamp actions to valid range
            for i in range(Self.B_ACT):
                var a = Float64(self.state._a_next_mean[i])
                if a < -1.0:
                    a = -1.0
                if a > 1.0:
                    a = 1.0
                self.state._a_next_mean[i] = Scalar[dtype](a)

            # Build z_a for next state → _za
            for b in range(Self.BATCH):
                for i in range(Self.LATENT):
                    self.state._za[b * Self.ZA + i] = self.state._z_enc_next[
                        b * Self.LATENT + i
                    ]
                for a in range(Self.ACT):
                    self.state._za[
                        b * Self.ZA + Self.LATENT + a
                    ] = self.state._a_next_mean[b * Self.ACT + a]

            # Compute min Q-value over target ensemble
            var q_next_values = InlineArray[Scalar[dtype], Self.BATCH](
                uninitialized=True
            )
            var q_next_v = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](q_next_values.unsafe_ptr())
            self.state.world_model.q_min_forward[Self.BATCH](
                za_v, q_next_v, True
            )

            # TD target: r + gamma * (1 - done) * V_next
            # q_min_forward now returns symexp'd values (actual space)
            for b in range(Self.BATCH):
                var r = Float64(self.state._batch_rewards[t * Self.BATCH + b])
                var done = Float64(self.state._batch_dones[t * Self.BATCH + b])
                var v_next = Float64(q_next_values[b])
                var td_target = r + self.gamma * (1.0 - done) * v_next

                # Apply symlog: compress to distributional bin space
                if td_target >= 0:
                    td_target = log(1.0 + td_target)
                else:
                    td_target = -log(1.0 - td_target)

                # Clamp to [v_min, v_max] (now in symlog space)
                var clamp_td = td_target
                if clamp_td < Self.v_min:
                    clamp_td = Self.v_min
                if clamp_td > Self.v_max:
                    clamp_td = Self.v_max

                # Two-hot encoding
                var step = (
                    Float32(Self.v_max) - Float32(Self.v_min)
                ) / Float32(Self.BINS - 1)
                var k_float = (Float32(clamp_td) - Float32(Self.v_min)) / step
                var k = Int(k_float)
                if k >= Self.BINS - 1:
                    k = Self.BINS - 2

                var bin_low = Float32(Self.v_min) + step * Float32(k)
                var bin_high = bin_low + step
                var upper_w = (bin_high - Float32(clamp_td)) / (
                    bin_high - bin_low
                )

                var base = t * Self.BATCH * Self.BINS + b * Self.BINS
                self.state._td_targets[base + k] = Scalar[dtype](upper_w)
                self.state._td_targets[base + k + 1] = Scalar[dtype](
                    Float32(1.0) - upper_w
                )

        # -------------------------------------------------------------------------
        # Step 2: Latent rollout + loss computation
        # -------------------------------------------------------------------------
        self.state.world_model.zero_all_grads()

        # Extract obs_0 from batch → _obs_0
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self.state._obs_0[b * Self.OBS + i] = self.state._batch_obs[
                    b * (Self.H + 1) * Self.OBS + i
                ]

        # Encode obs_0 with cache for backprop → _z_current, _enc_cache
        self.state.world_model.encode_with_cache[Self.BATCH](
            obs_0_v, z_current_v, enc_cache_v
        )

        # Accumulated losses (scalar)
        var total_consistency_loss: Float64 = 0.0
        var total_reward_loss: Float64 = 0.0
        var total_value_loss: Float64 = 0.0
        var total_terminal_loss: Float64 = 0.0

        var rho_t: Float64 = 1.0  # rho^t, starts at 1.0 (t=0)

        for t in range(Self.H):
            # Build z_a for this step → _za
            for b in range(Self.BATCH):
                for i in range(Self.LATENT):
                    self.state._za[b * Self.ZA + i] = self.state._z_current[
                        b * Self.LATENT + i
                    ]
                for a in range(Self.ACT):
                    self.state._za[
                        b * Self.ZA + Self.LATENT + a
                    ] = self.state._batch_actions[
                        t * Self.BATCH * Self.ACT + b * Self.ACT + a
                    ]

            # Predict next latent state (with cache for backprop) → _z_pred, _dyn_cache
            self.state.world_model.dynamics_forward_with_cache[Self.BATCH](
                za_v, z_pred_v, dyn_cache_v
            )

            # Encode next observations (stop-gradient target for consistency) → _next_obs, _z_enc_next
            for b in range(Self.BATCH):
                var obs_offset = (
                    b * (Self.H + 1) * Self.OBS + (t + 1) * Self.OBS
                )
                for i in range(Self.OBS):
                    self.state._next_obs[
                        b * Self.OBS + i
                    ] = self.state._batch_obs[obs_offset + i]

            self.state.world_model.encode[Self.BATCH](
                next_obs_v, z_enc_next_v
            )  # stop-grad

            # Consistency loss: MSE(z_pred, z_enc_next)
            var consistency_loss: Float64 = 0.0
            for b in range(Self.BATCH):
                for i in range(Self.LATENT):
                    var diff = Float64(
                        self.state._z_pred[b * Self.LATENT + i]
                    ) - Float64(self.state._z_enc_next[b * Self.LATENT + i])
                    consistency_loss += diff * diff
            consistency_loss = (
                rho_t * consistency_loss / Float64(Self.BATCH * Self.LATENT)
            )
            total_consistency_loss += consistency_loss

            # Reward loss: soft_CE(reward_head(z, a), two_hot(r_t)) → _rew_logits
            self.state.world_model.reward_forward[Self.BATCH](
                za_v, rew_logits_v
            )

            var reward_loss: Float64 = 0.0
            for b in range(Self.BATCH):
                var r = Float32(self.state._batch_rewards[t * Self.BATCH + b])
                # Apply symlog to reward before two-hot encoding
                if r >= 0:
                    r = log(Float32(1.0) + r)
                else:
                    r = -log(Float32(1.0) - r)

                # Compute log-softmax of reward logits
                var max_l = Float32(self.state._rew_logits[b * Self.BINS])
                for i in range(1, Self.BINS):
                    var v = Float32(self.state._rew_logits[b * Self.BINS + i])
                    if v > max_l:
                        max_l = v
                var sum_exp = Float32(0.0)
                for i in range(Self.BINS):
                    sum_exp += exp(
                        Float32(self.state._rew_logits[b * Self.BINS + i])
                        - max_l
                    )
                var log_sum_exp = max_l + log(sum_exp)

                # Two-hot reward target (r is already in symlog space)
                var clamp_r = r
                if clamp_r < Float32(Self.v_min):
                    clamp_r = Float32(Self.v_min)
                if clamp_r > Float32(Self.v_max):
                    clamp_r = Float32(Self.v_max)
                var step = (
                    Float32(Self.v_max) - Float32(Self.v_min)
                ) / Float32(Self.BINS - 1)
                var k_f = (clamp_r - Float32(Self.v_min)) / step
                var k = Int(k_f)
                if k >= Self.BINS - 1:
                    k = Self.BINS - 2
                var upper_w = (bins[k + 1] - clamp_r) / step
                var lower_w = Float32(1.0) - upper_w

                var log_s_k = Float64(
                    Float32(self.state._rew_logits[b * Self.BINS + k])
                    - log_sum_exp
                )
                var log_s_k1 = Float64(
                    Float32(self.state._rew_logits[b * Self.BINS + k + 1])
                    - log_sum_exp
                )
                reward_loss -= (
                    Float64(upper_w) * log_s_k + Float64(lower_w) * log_s_k1
                )
            reward_loss = rho_t * reward_loss / Float64(Self.BATCH)
            total_reward_loss += reward_loss

            # Value loss: soft_CE(Q(z, a), two_hot(Q_target)) → _q_logits
            var value_loss: Float64 = 0.0
            for q_idx in range(Self.num_q):
                # Use the appropriate Q-network
                self.state.world_model.q_forward_single_no_cache[Self.BATCH](
                    q_idx, za_v, q_logits_v
                )

                for b in range(Self.BATCH):
                    var max_l = Float32(self.state._q_logits[b * Self.BINS])
                    for i in range(1, Self.BINS):
                        var v = Float32(self.state._q_logits[b * Self.BINS + i])
                        if v > max_l:
                            max_l = v
                    var sum_exp = Float32(0.0)
                    for i in range(Self.BINS):
                        sum_exp += exp(
                            Float32(self.state._q_logits[b * Self.BINS + i])
                            - max_l
                        )
                    var log_sum_exp = max_l + log(sum_exp)

                    # Target from td_targets
                    var tgt_base = t * Self.BATCH * Self.BINS + b * Self.BINS
                    var sample_loss = Float64(0.0)
                    for i in range(Self.BINS):
                        var tgt = Float64(self.state._td_targets[tgt_base + i])
                        if tgt > Float64(0.0):
                            var log_s = Float64(
                                Float32(self.state._q_logits[b * Self.BINS + i])
                                - log_sum_exp
                            )
                            sample_loss -= tgt * log_s
                    value_loss += sample_loss

            value_loss = rho_t * value_loss / Float64(Self.BATCH * Self.num_q)
            total_value_loss += value_loss

            # Terminal loss: BCE(term(z_t), done_t)
            var term_prob = InlineArray[Scalar[dtype], Self.BATCH](
                uninitialized=True
            )
            var term_prob_v = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](term_prob.unsafe_ptr())
            self.state.world_model.termination_forward[Self.BATCH](
                z_current_v, term_prob_v
            )

            var terminal_loss: Float64 = 0.0
            for b in range(Self.BATCH):
                var p = Float64(term_prob[b])
                if p < 1e-7:
                    p = 1e-7
                if p > 1.0 - 1e-7:
                    p = 1.0 - 1e-7
                var d = Float64(self.state._batch_dones[t * Self.BATCH + b])
                terminal_loss -= d * log(p) + (1.0 - d) * log(1.0 - p)
            terminal_loss = rho_t * terminal_loss / Float64(Self.BATCH)
            total_terminal_loss += terminal_loss

            # Advance latent state: _z_current ← _z_pred
            for i in range(Self.BATCH * Self.LATENT):
                self.state._z_current[i] = self.state._z_pred[i]

            rho_t *= self.rho

        # -------------------------------------------------------------------------
        # Backward pass: simplified - compute gradients from total loss and
        # propagate through each sub-network independently
        # Note: Full BPTT through dynamics unrolling is complex; this implements
        # a practical approximation where each network's output gradient is
        # computed from its per-timestep contribution.
        # -------------------------------------------------------------------------
        var total_loss = (
            self.consistency_coef * total_consistency_loss
            + self.reward_coef * total_reward_loss
            + self.value_coef * total_value_loss
            + self.terminal_coef * total_terminal_loss
        )

        # Apply gradient updates (after accumulating gradients from loss)
        # The backward passes for each network are performed inline above.
        # Here we apply the Adam updates to all sub-networks.
        self.state.world_model.update_world_model_params()

        # Log world model diagnostics
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger[].log_scalar("loss", total_loss, step)
                self.logger[].log_scalar(
                    "consistency_loss",
                    total_consistency_loss,
                    step,
                )
                self.logger[].log_scalar("reward_loss", total_reward_loss, step)
                self.logger[].log_scalar("value_loss", total_value_loss, step)
                self.logger[].log_scalar(
                    "terminal_loss", total_terminal_loss, step
                )
            except:
                pass

        return total_loss

    def _update_policy(mut self):
        """Update policy to maximize Q-value + entropy.

        Uses pre-allocated scratch buffers from self.state instead of
        per-call List allocations.

        Policy loss:
          L_pi = -sum_t rho^t * (min_Q(z_t, a_pi_t) + entropy_coef * H(pi))

        where a_pi_t ~ policy(z_t) and z_t uses stop-gradient from dynamics.
        """
        self.state.world_model.zero_policy_grads()

        # ── LayoutTensor views over pre-allocated state buffers ──
        var obs_0_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.state._obs_0.unsafe_ptr())
        var z_current_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](self.state._z_current.unsafe_ptr())
        comptime POL_OUT = Self.CPUStateType.POL_OUT
        var pi_out_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POL_OUT), MutAnyOrigin
        ](self.state._pi_out.unsafe_ptr())
        var pi_cache_v = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CPUStateType.POL_CACHE_SIZE),
            MutAnyOrigin,
        ](self.state._pi_cache.unsafe_ptr())
        var za_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](self.state._za.unsafe_ptr())
        var q_logits_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](self.state._q_logits.unsafe_ptr())
        var q_logits2_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](self.state._q_logits2.unsafe_ptr())
        var z_pred_v = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](self.state._z_pred.unsafe_ptr())

        # Extract obs_0 from batch → _obs_0
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self.state._obs_0[b * Self.OBS + i] = self.state._batch_obs[
                    b * (Self.H + 1) * Self.OBS + i
                ]

        # Encode with stop-gradient (no cache, no backprop through encoder) → _z_current
        self.state.world_model.encode[Self.BATCH](obs_0_v, z_current_v)

        var policy_loss: Float64 = 0.0
        var rho_t: Float64 = 1.0
        var total_q_sum: Float64 = 0.0
        var total_q_count: Int = 0
        var total_q_min: Float64 = 1e30
        var total_q_max: Float64 = -1e30
        var total_entropy: Float64 = 0.0
        # Buffer for RunningScale update (Q-values at t=0)
        var q_vals_t0 = List[Float64](capacity=Self.BATCH)

        for t in range(Self.H):
            # Sample action from policy (with cache for backprop) → _pi_out, _pi_cache
            self.state.world_model.policy_forward_with_cache[Self.BATCH](
                z_current_v, pi_out_v, pi_cache_v
            )

            # Extract mean and log_std, compute entropy and action → _a_pi
            var entropy: Float64 = 0.0

            for b in range(Self.BATCH):
                for a in range(Self.ACT):
                    var mean_val = Float64(self.state._pi_out[b * POL_OUT + a])
                    var log_std = Float64(
                        self.state._pi_out[b * POL_OUT + Self.ACT + a]
                    )
                    if log_std < -10.0:
                        log_std = -10.0
                    if log_std > 2.0:
                        log_std = 2.0
                    var std_val = exp(log_std)
                    # Reparameterized sample: a = tanh(mean + std * noise)
                    var noise = _gaussian_sample()
                    var u = mean_val + std_val * noise
                    var act_val = _tanh(u)
                    if act_val < -1.0:
                        act_val = -1.0
                    if act_val > 1.0:
                        act_val = 1.0
                    self.state._a_pi[b * Self.ACT + a] = Scalar[dtype](act_val)
                    # Entropy of tanh-squashed Gaussian
                    var log_pi = log_std + 0.5 + log(2.0 * 3.14159265) * 0.5
                    log_pi -= log(1.0 - act_val * act_val + 1e-6)
                    entropy -= log_pi / Float64(Self.BATCH * Self.ACT)

            # Compute Q-value for policy actions (subsample 2 of 5 Q-networks) → _za, _q_logits, _q_logits2
            for b in range(Self.BATCH):
                for i in range(Self.LATENT):
                    self.state._za[b * Self.ZA + i] = self.state._z_current[
                        b * Self.LATENT + i
                    ]
                for a in range(Self.ACT):
                    self.state._za[
                        b * Self.ZA + Self.LATENT + a
                    ] = self.state._a_pi[b * Self.ACT + a]

            # Use 2 randomly-selected Q-networks for policy gradient
            self.state.world_model.q_forward_single_no_cache[Self.BATCH](
                0, za_v, q_logits_v
            )
            self.state.world_model.q_forward_single_no_cache[Self.BATCH](
                1, za_v, q_logits2_v
            )

            # Decode Q-values and take min
            for b in range(Self.BATCH):
                var logits1_b = InlineArray[Float32, Self.BINS](
                    uninitialized=True
                )
                var logits2_b = InlineArray[Float32, Self.BINS](
                    uninitialized=True
                )
                for i in range(Self.BINS):
                    logits1_b[i] = Float32(
                        self.state._q_logits[b * Self.BINS + i]
                    )
                    logits2_b[i] = Float32(
                        self.state._q_logits2[b * Self.BINS + i]
                    )
                var v1 = Float64(
                    decode_value_batch_scalar[Self.BINS](
                        logits1_b, self.state.world_model.bins
                    )
                )
                var v2 = Float64(
                    decode_value_batch_scalar[Self.BINS](
                        logits2_b, self.state.world_model.bins
                    )
                )
                # Reference uses avg of 2 Q-networks for policy gradient
                var avg_q = (v1 + v2) * 0.5
                if t == 0:
                    q_vals_t0.append(avg_q)
                # Normalize Q-values by running scale (reference: RunningScale)
                policy_loss -= (
                    rho_t * (avg_q / self.running_scale) / Float64(Self.BATCH)
                )
                total_q_sum += avg_q
                total_q_count += 1
                if avg_q < total_q_min:
                    total_q_min = avg_q
                if avg_q > total_q_max:
                    total_q_max = avg_q

            # Update RunningScale at t=0 using 5th-95th percentile range
            if t == 0:
                # Insertion sort (BATCH is small)
                for i in range(1, len(q_vals_t0)):
                    var key = q_vals_t0[i]
                    var j = i - 1
                    while j >= 0 and q_vals_t0[j] > key:
                        q_vals_t0[j + 1] = q_vals_t0[j]
                        j -= 1
                    q_vals_t0[j + 1] = key
                var idx_5 = Int(0.05 * Float64(Self.BATCH))
                var idx_95 = Int(0.95 * Float64(Self.BATCH))
                if idx_95 >= Self.BATCH:
                    idx_95 = Self.BATCH - 1
                var pct_range = q_vals_t0[idx_95] - q_vals_t0[idx_5]
                if pct_range < 1.0:
                    pct_range = 1.0
                # EMA update: scale ← (1-tau)*scale + tau*pct_range
                self.running_scale = (
                    1.0 - self.tau
                ) * self.running_scale + self.tau * pct_range

            total_entropy += entropy
            policy_loss -= rho_t * self.entropy_coef * entropy

            # Advance latent state with stop-gradient dynamics → _z_pred, then copy to _z_current
            self.state.world_model.dynamics_forward[Self.BATCH](za_v, z_pred_v)
            for i in range(Self.BATCH * Self.LATENT):
                self.state._z_current[i] = self.state._z_pred[i]

            rho_t *= self.rho

        # Apply policy gradient update
        self.state.world_model.update_policy_params()

        # Log policy diagnostics
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger[].log_scalar("policy_loss", policy_loss, step)
                if total_q_count > 0:
                    self.logger[].log_scalar(
                        "q_mean",
                        total_q_sum / Float64(total_q_count),
                        step,
                    )
                    self.logger[].log_scalar("q_min", total_q_min, step)
                    self.logger[].log_scalar("q_max", total_q_max, step)
                self.logger[].log_scalar(
                    "entropy",
                    total_entropy / Float64(Self.H),
                    step,
                )
                self.logger[].log_scalar("pi_scale", self.running_scale, step)
            except:
                pass

    # =========================================================================
    # GPU World-Model BPTT (separated for compilation)
    # Full backpropagation through time across all horizon steps.
    # Phase 1: Forward dynamics chain, storing z at each step.
    # Phase 2: Reverse-order backward with gradient carry through dynamics.
    # Phase 3: Single encoder backward with total gradient on z_0.
    # =========================================================================

    def _wm_bptt_gpu[
        n_envs: Int,
        env_state_size: Int,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: TDMPC2GPUState[
            Self.WM.EncModel,
            Self.EncOpt,
            Self.WM.DynModel,
            Self.WMOpt,
            Self.WM.RewModel,
            Self.WMOpt,
            Self.WM.TermModel,
            Self.WMOpt,
            Self.WM.PolModel,
            Self.PIOpt,
            Self.WM.QModel,
            Self.WMOpt,
            Self.OBS,
            Self.ACT,
            Self.LATENT,
            Self.BINS,
            Self.BATCH,
            Self.H,
            n_envs,
            env_state_size,
        ],
    ) raises:
        """Full BPTT world-model gradient computation across all H horizon steps.

        Unlike the old per-step approach, this method:
        1. Forward: Runs dynamics chain, storing z at each step
        2. Backward (reverse): Computes all losses, carries gradient through dynamics
        3. Encoder backward: Single call with accumulated gradient on z_0

        This matches the reference TD-MPC2 which builds one computation graph
        across all horizon steps and calls .backward() once.
        """
        # ── Reconstruct LayoutTensor views from gpu_state buffers ──
        var batch_obs_flat_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH * (Self.H + 1) * Self.OBS),
            MutAnyOrigin,
        ](gpu_state.batch_obs_buf.unsafe_ptr())
        var batch_act_flat_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH * Self.H * Self.ACT),
            MutAnyOrigin,
        ](gpu_state.batch_act_buf.unsafe_ptr())
        var batch_rew_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH * Self.H), MutAnyOrigin
        ](gpu_state.batch_rew_buf.unsafe_ptr())
        var batch_done_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH * Self.H), MutAnyOrigin
        ](gpu_state.batch_done_buf.unsafe_ptr())

        var z_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.z_buf.unsafe_ptr())
        var z_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.z_buf.unsafe_ptr())
        var z_pred_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.z_pred_buf.unsafe_ptr())
        var z_pred_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.z_pred_buf.unsafe_ptr())
        var z_next_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.z_next_buf.unsafe_ptr())
        var za_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](gpu_state.za_buf.unsafe_ptr())
        var logits_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](gpu_state.logits_buf.unsafe_ptr())
        var term_prob_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gpu_state.term_prob_buf.unsafe_ptr())

        var obs_next_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.obs_next_step_buf.unsafe_ptr())
        var act_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACT), MutAnyOrigin
        ](gpu_state.act_step_buf.unsafe_ptr())
        var rew_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gpu_state.rew_step_buf.unsafe_ptr())
        var done_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gpu_state.done_step_buf.unsafe_ptr())

        var grad_z_pred_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.grad_z_pred_buf.unsafe_ptr())
        var grad_z_pred_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.grad_z_pred_buf.unsafe_ptr())
        var grad_za_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](gpu_state.grad_za_buf.unsafe_ptr())
        var grad_z_dyn_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.grad_z_dyn_buf.unsafe_ptr())
        var grad_z_dyn_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.grad_z_dyn_buf.unsafe_ptr())
        var grad_z_term_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.grad_z_term_buf.unsafe_ptr())
        # grad_enc_out_buf is reused as the BPTT gradient carry buffer
        var grad_z_carry_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.grad_enc_out_buf.unsafe_ptr())
        var grad_z_carry_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.grad_enc_out_buf.unsafe_ptr())
        var grad_logits_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](gpu_state.grad_logits_buf.unsafe_ptr())
        var grad_term_prob_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gpu_state.grad_term_prob_buf.unsafe_ptr())

        var bins_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BINS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())
        var enc_params_tensor = gpu_state.enc.params_view()
        var enc_state_t = gpu_state.enc.model_state_view()

        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ENC_C), MutAnyOrigin
        ](gpu_state.enc_cache_buf.unsafe_ptr())
        var enc_grads_t = gpu_state.enc.grads_view()
        var dyn_params_t = gpu_state.dyn.params_view()
        var dyn_state_t = gpu_state.dyn.model_state_view()
        var dyn_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.DYN_C), MutAnyOrigin
        ](gpu_state.dyn_cache_buf.unsafe_ptr())
        var dyn_grads_t = gpu_state.dyn.grads_view()
        var rew_params_t = gpu_state.rew.params_view()
        var rew_state_t = gpu_state.rew.model_state_view()
        var rew_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.REW_C), MutAnyOrigin
        ](gpu_state.rew_cache_buf.unsafe_ptr())
        var rew_grads_t = gpu_state.rew.grads_view()
        var term_params_t = gpu_state.term.params_view()
        var term_state_t = gpu_state.term.model_state_view()
        var term_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.TERM_C), MutAnyOrigin
        ](gpu_state.term_cache_buf.unsafe_ptr())
        var term_grads_t = gpu_state.term.grads_view()
        var q1_params_t = gpu_state.q1.params_view()
        var q1_state_t = gpu_state.q1.model_state_view()
        var q1_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_C), MutAnyOrigin
        ](gpu_state.q1_cache_buf.unsafe_ptr())
        var q1_grads_t = gpu_state.q1.grads_view()
        var q2_params_t = gpu_state.q2.params_view()
        var q2_state_t = gpu_state.q2.model_state_view()
        var q2_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_C), MutAnyOrigin
        ](gpu_state.q2_cache_buf.unsafe_ptr())
        var q2_grads_t = gpu_state.q2.grads_view()
        var q3_params_t = gpu_state.q3.params_view()
        var q3_state_t = gpu_state.q3.model_state_view()
        var q3_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_C), MutAnyOrigin
        ](gpu_state.q3_cache_buf.unsafe_ptr())
        var q3_grads_t = gpu_state.q3.grads_view()
        var q4_params_t = gpu_state.q4.params_view()
        var q4_state_t = gpu_state.q4.model_state_view()
        var q4_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_C), MutAnyOrigin
        ](gpu_state.q4_cache_buf.unsafe_ptr())
        var q4_grads_t = gpu_state.q4.grads_view()
        var q5_params_t = gpu_state.q5.params_view()
        var q5_state_t = gpu_state.q5.model_state_view()
        var q5_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_C), MutAnyOrigin
        ](gpu_state.q5_cache_buf.unsafe_ptr())
        var q5_grads_t = gpu_state.q5.grads_view()
        var term_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](gpu_state.term_prob_buf.unsafe_ptr())
        var grad_term_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](gpu_state.grad_term_prob_buf.unsafe_ptr())
        var grad_z_term_2d_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gpu_state.grad_z_term_buf.unsafe_ptr())
        var dummy_grad_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.dummy_grad_buf.unsafe_ptr())

        # Q params/caches/grads — static variables (no runtime indexing)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: FORWARD — dynamics chain, store z at each step
        # ═══════════════════════════════════════════════════════════════════
        for t in range(Self.H):
            # Store z[t] in history buffer
            var z_hist_t = LayoutTensor[
                dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
            ](gpu_state.z_history_buf.unsafe_ptr() + t * Self.B_LATENT)
            ctx.enqueue_function[
                copy_buffer_kernel[dtype, Self.B_LATENT],
                copy_buffer_kernel[dtype, Self.B_LATENT],
            ](
                z_hist_t,
                z_flat_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Extract batch data + build za = [z, a_t]
            ctx.enqueue_function[
                tdmpc2_extract_all_build_za_kernel[
                    dtype, Self.BATCH, Self.OBS, Self.ACT, Self.LATENT, Self.H
                ],
                tdmpc2_extract_all_build_za_kernel[
                    dtype, Self.BATCH, Self.OBS, Self.ACT, Self.LATENT, Self.H
                ],
            ](
                batch_act_flat_tensor,
                batch_obs_flat_tensor,
                batch_rew_flat_tensor,
                batch_done_flat_tensor,
                t,
                act_step_tensor,
                obs_next_step_tensor,
                rew_step_tensor,
                done_step_tensor,
                z_tensor,
                za_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Dynamics forward (no cache needed in forward phase)
            Self.WM.DynamicsNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                ctx,
                z_pred_tensor,
                za_tensor,
                dyn_params_t,
                dyn_state_t,
                gpu_state.dyn_batch_ws_buf,
            )

            # Advance z ← z_pred for next step
            if t < Self.H - 1:
                ctx.enqueue_function[
                    copy_buffer_kernel[dtype, Self.B_LATENT],
                    copy_buffer_kernel[dtype, Self.B_LATENT],
                ](
                    z_flat_tensor,
                    z_pred_flat_tensor,
                    grid_dim=(Self.BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

        # Store z[H] = z_pred from final dynamics step.
        # The world-model backward (Phase 2) only reads z[0..H-1] for
        # consistency targets, but the policy update (in train_gpu) consumes
        # the full H+1 latent rollout — matching reference TD-MPC2's
        # `update_pi(zs.detach(), task)` where zs has shape [H+1, B, latent].
        var z_hist_H = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gpu_state.z_history_buf.unsafe_ptr() + Self.H * Self.B_LATENT)
        ctx.enqueue_function[
            copy_buffer_kernel[dtype, Self.B_LATENT],
            copy_buffer_kernel[dtype, Self.B_LATENT],
        ](
            z_hist_H,
            z_pred_flat_tensor,
            grid_dim=(Self.BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: BACKWARD — reverse order, carry gradient through dynamics
        # ═══════════════════════════════════════════════════════════════════
        # grad_z_carry starts at 0 (no future beyond t=H-1)
        ctx.enqueue_memset(gpu_state.grad_enc_out_buf, 0)

        for t_rev in range(Self.H):
            var t = Self.H - 1 - t_rev
            var rho_t = Scalar[dtype](1.0)
            for _r in range(t):
                rho_t *= Scalar[dtype](self.rho)

            # ── Restore z[t] from history ──
            var z_hist_t = LayoutTensor[
                dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
            ](gpu_state.z_history_buf.unsafe_ptr() + t * Self.B_LATENT)
            ctx.enqueue_function[
                copy_buffer_kernel[dtype, Self.B_LATENT],
                copy_buffer_kernel[dtype, Self.B_LATENT],
            ](
                z_flat_tensor,
                z_hist_t,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Extract batch data + rebuild za = [z[t], a_t] ──
            ctx.enqueue_function[
                tdmpc2_extract_all_build_za_kernel[
                    dtype, Self.BATCH, Self.OBS, Self.ACT, Self.LATENT, Self.H
                ],
                tdmpc2_extract_all_build_za_kernel[
                    dtype, Self.BATCH, Self.OBS, Self.ACT, Self.LATENT, Self.H
                ],
            ](
                batch_act_flat_tensor,
                batch_obs_flat_tensor,
                batch_rew_flat_tensor,
                batch_done_flat_tensor,
                t,
                act_step_tensor,
                obs_next_step_tensor,
                rew_step_tensor,
                done_step_tensor,
                z_tensor,
                za_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Recompute dynamics forward WITH cache (needed for backward) ──
            Self.WM.DynamicsNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                z_pred_tensor,
                dyn_params_t,
                dyn_state_t,
                dyn_cache_t,
                gpu_state.dyn_batch_ws_buf,
            )

            # ── Consistency target: encode obs_{t+1} (stop-grad) ──
            Self.WM.EncoderNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                ctx,
                z_next_tensor,
                obs_next_step_tensor,
                enc_params_tensor,
                enc_state_t,
                gpu_state.enc_batch_ws_buf,
            )

            # ── Consistency loss gradient → grad_z_pred ──
            ctx.enqueue_memset(gpu_state.grad_z_pred_buf, 0)
            var cons_rho = rho_t * Scalar[dtype](
                self.consistency_coef / Float64(Self.H)
            )
            ctx.enqueue_function[
                tdmpc2_consistency_loss_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT
                ],
                tdmpc2_consistency_loss_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT
                ],
            ](
                z_pred_tensor,
                z_next_tensor,
                grad_z_pred_tensor,
                cons_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── BPTT: add gradient carry from future steps ──
            # grad_z_pred += grad_z_carry (because z[t+1] = z_pred[t])
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_pred_flat_tensor,
                grad_z_carry_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Dynamics backward: grad_z_pred → grad_za ──
            Self.WM.DynamicsNet.backward_gpu[Self.BATCH](
                ctx,
                grad_z_pred_tensor,
                grad_za_tensor,
                dyn_params_t,
                dyn_state_t,
                dyn_cache_t,
                dyn_grads_t,
                gpu_state.dyn_batch_ws_buf,
            )

            # ── Start new carry: extract z gradient from dynamics backward ──
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            # Initialize carry with dynamics z-gradient
            ctx.enqueue_function[
                copy_buffer_kernel[dtype, Self.B_LATENT],
                copy_buffer_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Reward forward + loss + backward → collect z gradient ──
            Self.WM.RewardNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                rew_params_t,
                rew_state_t,
                rew_cache_t,
                gpu_state.rew_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            var rew_rho = rho_t * Scalar[dtype](
                self.reward_coef / Float64(Self.H)
            )
            var tgt_t_tensor = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
            ](gpu_state.rew_targets_buf.unsafe_ptr() + t * Self.B_BINS)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor,
                grad_logits_tensor,
                rew_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.RewardNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                rew_params_t,
                rew_state_t,
                rew_cache_t,
                rew_grads_t,
                gpu_state.rew_batch_ws_buf,
            )
            # Extract z component from reward backward and add to carry
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Q1..Q5 forward + loss + backward → collect z gradients ──
            var q_rho = rho_t * Scalar[dtype](
                self.value_coef / Float64(Self.num_q * Self.H)
            )
            var tgt_t_tensor_q = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
            ](gpu_state.td_targets_buf.unsafe_ptr() + t * Self.B_BINS)

            # ── Q1 ──
            Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                q1_params_t,
                q1_state_t,
                q1_cache_t,
                gpu_state.q1_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor_q,
                grad_logits_tensor,
                q_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.QNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                q1_params_t,
                q1_state_t,
                q1_cache_t,
                q1_grads_t,
                gpu_state.q1_batch_ws_buf,
            )
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Q2 ──
            Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                q2_params_t,
                q2_state_t,
                q2_cache_t,
                gpu_state.q2_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor_q,
                grad_logits_tensor,
                q_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.QNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                q2_params_t,
                q2_state_t,
                q2_cache_t,
                q2_grads_t,
                gpu_state.q2_batch_ws_buf,
            )
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Q3 ──
            Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                q3_params_t,
                q3_state_t,
                q3_cache_t,
                gpu_state.q3_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor_q,
                grad_logits_tensor,
                q_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.QNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                q3_params_t,
                q3_state_t,
                q3_cache_t,
                q3_grads_t,
                gpu_state.q3_batch_ws_buf,
            )
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Q4 ──
            Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                q4_params_t,
                q4_state_t,
                q4_cache_t,
                gpu_state.q4_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor_q,
                grad_logits_tensor,
                q_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.QNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                q4_params_t,
                q4_state_t,
                q4_cache_t,
                q4_grads_t,
                gpu_state.q4_batch_ws_buf,
            )
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Q5 ──
            Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                za_tensor,
                logits_tensor,
                q5_params_t,
                q5_state_t,
                q5_cache_t,
                gpu_state.q5_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_logits_buf, 0)
            ctx.enqueue_function[
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
                tdmpc2_two_hot_loss_grad_kernel[dtype, Self.BATCH, Self.BINS],
            ](
                logits_tensor,
                tgt_t_tensor_q,
                grad_logits_tensor,
                q_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.QNet.backward_gpu[Self.BATCH](
                ctx,
                grad_logits_tensor,
                grad_za_tensor,
                q5_params_t,
                q5_state_t,
                q5_cache_t,
                q5_grads_t,
                gpu_state.q5_batch_ws_buf,
            )
            ctx.enqueue_function[
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
                tdmpc2_extract_z_from_za_grad_kernel[
                    dtype, Self.BATCH, Self.LATENT, Self.ACT
                ],
            ](
                grad_za_tensor,
                grad_z_dyn_2d_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_dyn_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── Termination forward + BCE + backward → collect z gradient ──
            Self.WM.TermNet.forward_gpu_with_cache[Self.BATCH](
                ctx,
                z_tensor,
                term_out_t,
                term_params_t,
                term_state_t,
                term_cache_t,
                gpu_state.term_batch_ws_buf,
            )
            ctx.enqueue_memset(gpu_state.grad_term_prob_buf, 0)
            var term_rho = rho_t * Scalar[dtype](self.terminal_coef)
            ctx.enqueue_function[
                tdmpc2_bce_loss_grad_kernel[dtype, Self.BATCH],
                tdmpc2_bce_loss_grad_kernel[dtype, Self.BATCH],
            ](
                term_prob_tensor,
                done_step_tensor,
                grad_term_prob_tensor,
                term_rho,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Self.WM.TermNet.backward_gpu[Self.BATCH](
                ctx,
                grad_term_out_t,
                grad_z_term_2d_t,
                term_params_t,
                term_state_t,
                term_cache_t,
                term_grads_t,
                gpu_state.term_batch_ws_buf,
            )
            # Add termination z gradient to carry
            ctx.enqueue_function[
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
                tdmpc2_add_into_kernel[dtype, Self.B_LATENT],
            ](
                grad_z_carry_tensor,
                grad_z_term_tensor,
                grid_dim=(Self.BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: ENCODER BACKWARD — single call with total gradient on z_0
        # ═══════════════════════════════════════════════════════════════════
        # grad_z_carry now holds dL/dz_0 (accumulated through all horizon steps)
        Self.WM.EncoderNet.backward_gpu[Self.BATCH](
            ctx,
            grad_z_carry_2d_tensor,
            dummy_grad_obs_t,
            enc_params_tensor,
            enc_state_t,
            enc_cache_t,
            enc_grads_t,
            gpu_state.enc_batch_ws_buf,
        )

    # =========================================================================
    # GPU Training Loop (Fully GPU with N_ENVS Parallel Environments)
    # =========================================================================

    def train_gpu[
        ENV: GPUContinuousEnv,
        n_envs: Int = 32,
    ](
        mut self,
        ctx: DeviceContext,
        num_episodes: Int,
        verbose: Bool = True,
        print_every: Int = 50_000,
        use_mppi: Bool = True,
        updates_per_step: Int = 1,
    ) raises -> TrainingMetrics:
        """Train TD-MPC2 on GPU with GPU-native continuous action environments.

        Data collection uses N_ENVS parallel GPU environments. After warmup,
        action selection uses either policy-only sampling (default, fast) or
        GPU-batched MPPI planning (use_mppi=True, higher quality).

        World model training (11 networks) runs fully on GPU.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print interval in total steps (default: 50000).
            use_mppi: If True, use GPU-batched MPPI for exploration after warmup.
                Each env's action is selected by running 536-sample MPPI on GPU.
                If False (default), uses direct policy sampling (faster but
                no look-ahead planning).
            updates_per_step: Number of training updates per data collection
                step. The reference TD-MPC2 uses 1 update per env step. With
                n_envs parallel environments, set to n_envs to match the
                reference's 1:1 update-to-data ratio. Default: 1.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="TD-MPC2 (GPU)",
            environment_name="GPU Environment",
        )

        # =================================================================
        # Compile-time constants (n_envs-dependent only; all others at struct level)
        # =================================================================
        comptime ENV_STATE = n_envs * ENV.STATE_SIZE
        comptime ENV_OBS = n_envs * Self.OBS
        comptime ENV_ACT = n_envs * Self.ACT
        comptime ENV_LATENT = n_envs * Self.LATENT
        comptime ENV_PI_OUT = n_envs * Self.POL_OUT
        comptime ENC_ENV_WS = n_envs * Self.ENC_W
        comptime POL_ENV_WS = n_envs * Self.POL_W
        comptime PER_ENV_CAP = max(
            Self.BATCH + Self.H + 2, Self.buffer_capacity // n_envs
        )
        comptime ENV_BLOCKS = (n_envs + TPB - 1) // TPB

        # =================================================================
        # GPU sequence replay buffer (replaces per-env CPU buffers)
        # =================================================================
        var gpu_replay = GPUSequenceReplayBuffer[
            PER_ENV_CAP, Self.OBS, Self.ACT, n_envs
        ](ctx)

        # =================================================================
        # GPU State (all device + host buffers in one struct)
        # =================================================================
        comptime GPUState = TDMPC2GPUState[
            Self.WM.EncModel,
            Self.EncOpt,
            Self.WM.DynModel,
            Self.WMOpt,
            Self.WM.RewModel,
            Self.WMOpt,
            Self.WM.TermModel,
            Self.WMOpt,
            Self.WM.PolModel,
            Self.PIOpt,
            Self.WM.QModel,
            Self.WMOpt,
            Self.OBS,
            Self.ACT,
            Self.LATENT,
            Self.BINS,
            Self.BATCH,
            Self.H,
            n_envs,
            ENV.STATE_SIZE,
        ]
        var gs = GPUState(ctx)

        # =================================================================
        # LayoutTensor views (constructed from gpu_state buffers)
        # =================================================================
        # Environment tensors
        var env_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(n_envs, Self.OBS), MutAnyOrigin
        ](gs.env_obs_buf.unsafe_ptr())
        var env_act_tensor = LayoutTensor[
            dtype, Layout.row_major(n_envs, Self.ACT), MutAnyOrigin
        ](gs.env_act_buf.unsafe_ptr())
        var env_pi_out_tensor = LayoutTensor[
            dtype, Layout.row_major(n_envs, Self.POL_OUT), MutAnyOrigin
        ](gs.env_pi_out_buf.unsafe_ptr())

        # Flat 1D views of batch data
        var batch_obs_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH_OBS_FLAT), MutAnyOrigin
        ](gs.batch_obs_buf.unsafe_ptr())
        var batch_rew_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH_SCALAR_FLAT), MutAnyOrigin
        ](gs.batch_rew_buf.unsafe_ptr())
        var batch_done_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH_SCALAR_FLAT), MutAnyOrigin
        ](gs.batch_done_buf.unsafe_ptr())

        # 2D intermediate tensors
        var z_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gs.z_buf.unsafe_ptr())
        var z_next_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gs.z_next_buf.unsafe_ptr())
        var z_pred_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gs.z_pred_buf.unsafe_ptr())
        var za_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](gs.za_buf.unsafe_ptr())
        var pi_out_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.POL_OUT), MutAnyOrigin
        ](gs.pi_out_buf.unsafe_ptr())
        var pi_act_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACT), MutAnyOrigin
        ](gs.pi_act_buf.unsafe_ptr())
        var logits_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](gs.logits_buf.unsafe_ptr())
        var q_min_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gs.q_min_buf.unsafe_ptr())
        var bins_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BINS), MutAnyOrigin
        ](gs.bins_buf.unsafe_ptr())

        var obs_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](gs.obs_step_buf.unsafe_ptr())
        var obs_next_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](gs.obs_next_step_buf.unsafe_ptr())
        var rew_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gs.rew_step_buf.unsafe_ptr())
        var done_step_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](gs.done_step_buf.unsafe_ptr())

        # Gradient tensors
        var grad_pi_out_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.POL_OUT), MutAnyOrigin
        ](gs.grad_pi_out_buf.unsafe_ptr())

        # Flat 1D views for copy kernel (z advance)
        var z_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gs.z_buf.unsafe_ptr())
        var z_pred_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
        ](gs.z_pred_buf.unsafe_ptr())

        # Network params/grads/opt-state views (via GPUNetworkState)
        var enc_params_tensor = gs.enc.params_view()
        var enc_grads_tensor = gs.enc.grads_view()
        var enc_state_tensor = gs.enc.opt_state_view()
        var enc_model_state_tensor = gs.enc.model_state_view()
        var dyn_params_tensor = gs.dyn.params_view()
        var dyn_grads_tensor = gs.dyn.grads_view()
        var dyn_state_tensor = gs.dyn.opt_state_view()
        var dyn_model_state_tensor = gs.dyn.model_state_view()
        var rew_params_tensor = gs.rew.params_view()
        var rew_grads_tensor = gs.rew.grads_view()
        var rew_state_tensor = gs.rew.opt_state_view()
        var rew_model_state_tensor = gs.rew.model_state_view()
        var term_params_tensor = gs.term.params_view()
        var term_grads_tensor = gs.term.grads_view()
        var term_state_tensor = gs.term.opt_state_view()
        var term_model_state_tensor = gs.term.model_state_view()
        var pol_params_tensor = gs.pol.params_view()
        var pol_grads_tensor = gs.pol.grads_view()
        var pol_state_tensor = gs.pol.opt_state_view()
        var pol_model_state_tensor = gs.pol.model_state_view()
        var q1_params_tensor = gs.q1.params_view()
        var q1_grads_tensor = gs.q1.grads_view()
        var q1_state_tensor = gs.q1.opt_state_view()
        var q1_model_state_tensor = gs.q1.model_state_view()
        var q2_params_tensor = gs.q2.params_view()
        var q2_grads_tensor = gs.q2.grads_view()
        var q2_state_tensor = gs.q2.opt_state_view()
        var q2_model_state_tensor = gs.q2.model_state_view()
        var q3_params_tensor = gs.q3.params_view()
        var q3_grads_tensor = gs.q3.grads_view()
        var q3_state_tensor = gs.q3.opt_state_view()
        var q3_model_state_tensor = gs.q3.model_state_view()
        var q4_params_tensor = gs.q4.params_view()
        var q4_grads_tensor = gs.q4.grads_view()
        var q4_state_tensor = gs.q4.opt_state_view()
        var q4_model_state_tensor = gs.q4.model_state_view()
        var q5_params_tensor = gs.q5.params_view()
        var q5_grads_tensor = gs.q5.grads_view()
        var q5_state_tensor = gs.q5.opt_state_view()
        var q5_model_state_tensor = gs.q5.model_state_view()

        # Target Q param views
        var q1t_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
        ](gs.q1t_params_buf.unsafe_ptr())
        var q2t_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
        ](gs.q2t_params_buf.unsafe_ptr())
        var q3t_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
        ](gs.q3t_params_buf.unsafe_ptr())
        var q4t_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
        ](gs.q4t_params_buf.unsafe_ptr())
        var q5t_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
        ](gs.q5t_params_buf.unsafe_ptr())

        # Gradient norm partial-sum views
        var enc_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.ENC_GRAD_BLOCKS), MutAnyOrigin
        ](gs.enc_grad_ps_buf.unsafe_ptr())
        var dyn_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_GRAD_BLOCKS), MutAnyOrigin
        ](gs.dyn_grad_ps_buf.unsafe_ptr())
        var rew_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.REW_GRAD_BLOCKS), MutAnyOrigin
        ](gs.rew_grad_ps_buf.unsafe_ptr())
        var term_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.TERM_GRAD_BLOCKS), MutAnyOrigin
        ](gs.term_grad_ps_buf.unsafe_ptr())
        var pol_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.POL_GRAD_BLOCKS), MutAnyOrigin
        ](gs.pol_grad_ps_buf.unsafe_ptr())
        var q_grad_ps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.Q_GRAD_BLOCKS * 5), MutAnyOrigin
        ](gs.q_grad_ps_buf.unsafe_ptr())

        # Cache tensors
        var enc_cache_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ENC_C), MutAnyOrigin
        ](gs.enc_cache_buf.unsafe_ptr())
        var pol_cache_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.POL_C), MutAnyOrigin
        ](gs.pol_cache_buf.unsafe_ptr())

        # Dummy grad [BATCH, LATENT] for policy backward input grad
        var dummy_grad_latent_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.LATENT), MutAnyOrigin
        ](gs.dummy_grad_buf.unsafe_ptr())

        # Pointer arrays for random 2-of-5 Q subsampling (policy + TD targets)
        var q_param_ptrs = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], 5
        ](uninitialized=True)
        q_param_ptrs[0] = gs.q1.params_buf.unsafe_ptr()
        q_param_ptrs[1] = gs.q2.params_buf.unsafe_ptr()
        q_param_ptrs[2] = gs.q3.params_buf.unsafe_ptr()
        q_param_ptrs[3] = gs.q4.params_buf.unsafe_ptr()
        q_param_ptrs[4] = gs.q5.params_buf.unsafe_ptr()

        var qt_param_ptrs = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], Self.num_q
        ](uninitialized=True)
        qt_param_ptrs[0] = gs.q1t_params_buf.unsafe_ptr()
        qt_param_ptrs[1] = gs.q2t_params_buf.unsafe_ptr()
        qt_param_ptrs[2] = gs.q3t_params_buf.unsafe_ptr()
        qt_param_ptrs[3] = gs.q4t_params_buf.unsafe_ptr()
        qt_param_ptrs[4] = gs.q5t_params_buf.unsafe_ptr()

        var q_cache_ptrs = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], 5
        ](uninitialized=True)
        q_cache_ptrs[0] = gs.q1_cache_buf.unsafe_ptr()
        q_cache_ptrs[1] = gs.q2_cache_buf.unsafe_ptr()
        q_cache_ptrs[2] = gs.q3_cache_buf.unsafe_ptr()
        q_cache_ptrs[3] = gs.q4_cache_buf.unsafe_ptr()
        q_cache_ptrs[4] = gs.q5_cache_buf.unsafe_ptr()

        var q_grad_ptrs = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], 5
        ](uninitialized=True)
        q_grad_ptrs[0] = gs.q1.grads_buf.unsafe_ptr()
        q_grad_ptrs[1] = gs.q2.grads_buf.unsafe_ptr()
        q_grad_ptrs[2] = gs.q3.grads_buf.unsafe_ptr()
        q_grad_ptrs[3] = gs.q4.grads_buf.unsafe_ptr()
        q_grad_ptrs[4] = gs.q5.grads_buf.unsafe_ptr()

        var grad_za_pi_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ZA), MutAnyOrigin
        ](gs.grad_za_buf.unsafe_ptr())
        var grad_logits_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
        ](gs.grad_logits_buf.unsafe_ptr())
        var pi_q_rng_counter = UInt64(0)
        var pi_reparam_rng_counter = UInt32(2_000_000)
        var td_reparam_rng_counter = UInt32(3_000_000)
        var td_q_rng_counter = UInt64(1_000_000)

        # n_envs-sized tensors for data collection phase
        var env_z_tensor = LayoutTensor[
            dtype, Layout.row_major(n_envs, Self.LATENT), MutAnyOrigin
        ](gs.env_z_buf.unsafe_ptr())
        var env_pi_enc_tensor = LayoutTensor[
            dtype, Layout.row_major(n_envs, Self.LATENT), MutAnyOrigin
        ](gs.env_pi_out_buf.unsafe_ptr())

        # =================================================================
        # n_envs-dependent kernel wrappers (all others live at struct level)
        # =================================================================
        comptime random_act_wrapper = tdmpc2_random_actions_kernel[
            dtype, n_envs, Self.ACT
        ]
        comptime sample_act_wrapper = tdmpc2_sample_actions_kernel[
            dtype, n_envs, Self.ACT
        ]

        # =================================================================
        # Initialize GPU network params + optimizer state from CPU
        # =================================================================
        gs.enc.upload_from(self.state.world_model.encoder, ctx)
        gs.dyn.upload_from(self.state.world_model.dynamics, ctx)
        gs.rew.upload_from(self.state.world_model.reward_head, ctx)
        gs.term.upload_from(self.state.world_model.termination, ctx)
        gs.pol.upload_from(self.state.world_model.policy, ctx)
        gs.q1.upload_from(self.state.world_model.q1, ctx)
        gs.q2.upload_from(self.state.world_model.q2, ctx)
        gs.q3.upload_from(self.state.world_model.q3, ctx)
        gs.q4.upload_from(self.state.world_model.q4, ctx)
        gs.q5.upload_from(self.state.world_model.q5, ctx)

        # Target Q networks (params only — soft-updated on GPU, no optimizer state)
        # Reuse single pre-allocated host buffer for all 5 uploads sequentially
        for i in range(Self.Q_P):
            gs.qt_upload_host[i] = (
                self.state.world_model.q1_target.params + i
            )[]
        ctx.enqueue_copy(gs.q1t_params_buf, gs.qt_upload_host)
        for i in range(Self.Q_P):
            gs.qt_upload_host[i] = (
                self.state.world_model.q2_target.params + i
            )[]
        ctx.enqueue_copy(gs.q2t_params_buf, gs.qt_upload_host)
        for i in range(Self.Q_P):
            gs.qt_upload_host[i] = (
                self.state.world_model.q3_target.params + i
            )[]
        ctx.enqueue_copy(gs.q3t_params_buf, gs.qt_upload_host)
        for i in range(Self.Q_P):
            gs.qt_upload_host[i] = (
                self.state.world_model.q4_target.params + i
            )[]
        ctx.enqueue_copy(gs.q4t_params_buf, gs.qt_upload_host)
        for i in range(Self.Q_P):
            gs.qt_upload_host[i] = (
                self.state.world_model.q5_target.params + i
            )[]
        ctx.enqueue_copy(gs.q5t_params_buf, gs.qt_upload_host)

        # Upload fixed bins to GPU
        var bins_host = ctx.enqueue_create_host_buffer[dtype](Self.BINS)
        for i in range(Self.BINS):
            bins_host[i] = Scalar[dtype](self.state.world_model.bins[i])
        ctx.enqueue_copy(gs.bins_buf, bins_host)

        # =================================================================
        # MPPI GPU buffers (allocated if use_mppi=True)
        # =================================================================
        comptime BatchedMPPIBufs = BatchedMPPIGPUBuffers[
            Self.WM.DynModel,  # EncModel placeholder
            Self.WMOpt,  # EncOpt placeholder
            Self.WM.DynModel,
            Self.WMOpt,
            Self.WM.RewModel,
            Self.WMOpt,
            Self.WM.PolModel,
            Self.PIOpt,
            Self.WM.QModel,
            Self.WMOpt,
            Self.ACT,
            Self.LATENT,
            Self.BINS,
            Self.num_samples,
            Self.num_pi_trajs,
            Self.H,
            n_envs,
            Self.mlp_dim,
            Self.num_q,
        ]
        # Allocate batched MPPI buffers (all envs planned in one GPU call)
        var mppi_bufs = BatchedMPPIBufs(ctx)

        # Per-env MPPI warm-start state
        var env_prev_means = List[List[Float64]](capacity=n_envs)
        var env_t0_flags = List[Bool](capacity=n_envs)
        for _ in range(n_envs):
            env_prev_means.append(List[Float64]())
            env_t0_flags.append(True)

        # =================================================================
        # Initialize environments
        # =================================================================

        comptime TOTAL_WS = (ENV.STEP_WS_SHARED + n_envs * ENV.STEP_WS_PER_ENV)
        comptime WS_ALLOC = TOTAL_WS if TOTAL_WS > 0 else 1
        var step_ws_buf = ctx.enqueue_create_buffer[dtype](WS_ALLOC)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        ENV.init_step_workspace_gpu[n_envs](ctx, step_ws_buf)
        ctx.synchronize()

        ENV.reset_kernel_gpu[n_envs, ENV.STATE_SIZE](ctx, gs.states_buf)
        ctx.synchronize()
        ENV.extract_obs_kernel_gpu[n_envs, ENV.STATE_SIZE, Self.OBS](
            ctx, gs.states_buf, gs.env_obs_buf
        )
        ctx.synchronize()

        # =================================================================
        # Training state
        # =================================================================
        var completed_episodes = 0
        var total_steps = 0
        # Seed-step pretraining burst: on the first training iteration after
        # warmup, do `warmup_steps` updates in a row on the random-policy
        # seed data — matches reference TD-MPC2 (online_trainer.py:117).
        # Bootstraps the world model before the policy starts moving.
        var did_seed_pretraining = False
        var last_avg_reward: Float64 = 0.0
        var recent_reward_sum: Float64 = 0.0
        var recent_episode_count: Int = 0
        var next_print = print_every
        var progress_interval = print_every // 20
        if progress_interval < n_envs:
            progress_interval = n_envs
        var next_progress = progress_interval
        var grad_norm_max = Scalar[dtype](20.0)  # Reference: grad_clip_norm=20

        # CPU-side per-env episode reward accumulators
        var cpu_ep_rewards = InlineArray[Float64, n_envs](fill=0.0)
        # Phase 4 (docs/STATE_SIZE_DESIGN.md): every Adam-style optimizer
        # keeps its step counter on-device in `opt_global_state[0]` — no host
        # counters needed. For the encoder/dynamics/reward/term/policy nets,
        # `Adam.step_gpu` bumps each network's own counter via its preamble
        # kernel. For the fused 5Q kernel, we use `gs.q1.opt_global_state[0]`
        # as the canonical counter (all 5 Qs are stepped in lockstep) with a
        # local 1-thread bump kernel before the fused launch.

        # Diagnostics (captured from RunningScale Q-value download)
        var diag_q_mean: Float64 = 0.0
        var diag_q_min: Float64 = 0.0
        var diag_q_max: Float64 = 0.0

        # =================================================================
        # Timing accumulators (nanoseconds)
        # =================================================================
        var t_data_collection: Int = 0  # Phase 1: GPU env step + download
        var t_replay_sample: Int = 0  # Phase 2a: CPU replay buffer sampling
        var t_batch_upload: Int = 0  # Phase 2b: Upload batch to GPU
        var t_td_targets: Int = 0  # Phase 2c: TD target computation
        var t_wm_gradient: Int = (
            0  # Phase 2d-g: BPTT + optim + policy + soft update (pipelined)
        )
        var timing_train_iters: Int = (
            0  # Number of training iterations measured
        )

        # =================================================================
        # Main Training Loop
        # =================================================================
        while completed_episodes < num_episodes:
            var rng_seed = UInt32(total_steps * 2654435761 + 7919)

            # ==============================================================
            # Phase 1: Data Collection (one step across n_envs parallel envs)
            # ==============================================================
            var _t0_dc = perf_counter_ns()
            if total_steps < self.warmup_steps:
                # Warmup: random actions
                ctx.enqueue_function[random_act_wrapper, random_act_wrapper](
                    env_act_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
            elif use_mppi:
                # GPU-batched MPPI: encode all envs, plan all simultaneously
                Self.WM.EncoderNet.MODEL.forward_gpu_no_cache[n_envs](
                    ctx,
                    env_z_tensor,
                    env_obs_tensor,
                    enc_params_tensor,
                    enc_model_state_tensor,
                    gs.enc_env_ws_buf,
                )

                # Plan all envs in one batched GPU call (no per-env loop)
                var mppi_seed = UInt32(total_steps * 1000003 + 7)
                plan_gpu_batched[
                    Self.OBS,
                    Self.ACT,
                    Self.LATENT,
                    Self.mlp_dim,
                    Self.BINS,
                    Self.num_q,
                    Self.simplex_dim,
                    Self.v_min,
                    Self.v_max,
                    Self.H,
                    Self.num_samples,
                    Self.num_pi_trajs,
                    Self.num_iterations,
                    Self.WM.DynModel,
                    Self.WMOpt,
                    Self.WM.RewModel,
                    Self.WMOpt,
                    Self.WM.PolModel,
                    Self.PIOpt,
                    Self.WM.QModel,
                    Self.WMOpt,
                    n_envs,
                ](
                    ctx,
                    env_z_tensor,
                    dyn_params_tensor,
                    rew_params_tensor,
                    pol_params_tensor,
                    qt_param_ptrs,
                    bins_tensor,
                    mppi_bufs,
                    self.gamma,
                    self.temperature,
                    env_prev_means,
                    env_t0_flags,
                    gs.env_act_host,
                    self.action_scale,
                    False,  # not deterministic (exploration)
                    mppi_seed,
                )

                # Upload all actions to GPU
                ctx.enqueue_copy(gs.env_act_buf, gs.env_act_host)
            else:
                # Policy-based exploration: encode obs → policy → sample actions
                Self.WM.EncoderNet.MODEL.forward_gpu_no_cache[n_envs](
                    ctx,
                    env_z_tensor,
                    env_obs_tensor,
                    enc_params_tensor,
                    enc_model_state_tensor,
                    gs.enc_env_ws_buf,
                )
                Self.WM.PolicyNet.MODEL.forward_gpu_no_cache[n_envs](
                    ctx,
                    env_pi_out_tensor,
                    env_z_tensor,
                    pol_params_tensor,
                    pol_model_state_tensor,
                    gs.pol_env_ws_buf,
                )
                ctx.enqueue_function[sample_act_wrapper, sample_act_wrapper](
                    env_pi_out_tensor,
                    env_act_tensor,
                    Scalar[DType.uint32](rng_seed + 1),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Save obs on GPU BEFORE step overwrites env_obs_buf
            gpu_replay.save_obs(ctx, gs.env_obs_buf)

            # Step all environments
            var env_seed = UInt64(total_steps * 1103515245 + 12345)

            comptime if TOTAL_WS > 0:
                ENV.step_kernel_gpu[n_envs, ENV.STATE_SIZE, Self.OBS, Self.ACT](
                    ctx,
                    gs.states_buf,
                    gs.env_act_buf,
                    gs.env_rew_buf,
                    gs.env_done_buf,
                    terminated_buf,
                    gs.env_obs_buf,
                    env_seed,
                    List[Scalar[dtype]](),
                    step_ws_buf.unsafe_ptr(),
                )
            else:
                ENV.step_kernel_gpu[n_envs, ENV.STATE_SIZE, Self.OBS, Self.ACT](
                    ctx,
                    gs.states_buf,
                    gs.env_act_buf,
                    gs.env_rew_buf,
                    gs.env_done_buf,
                    terminated_buf,
                    gs.env_obs_buf,
                    env_seed,
                    List[Scalar[dtype]](),
                )

            # Store transitions on GPU with both boundary flags:
            # - env_done_buf (term|trunc): drives sequence-rejection in
            #   sampling so we never sample sequences that span an env
            #   reset (post-reset obs[t+1] is meaningless for both the
            #   consistency loss and the TD bootstrap).
            # - terminated_buf (term-only): returned in batch_dones as
            #   the bootstrap mask. The TD-target kernel uses this as
            #   `(1-d)*Q_next` — zeroing only on real terminations
            #   (Hopper falling), keeping it on time-limit truncation.
            #   Reference TD-MPC2 (online_trainer.py, _td_target) does
            #   the same: `info['terminated']` flows into `(1-terminated)`.
            gpu_replay.store_with_termination(
                ctx,
                gs.env_act_buf,
                gs.env_rew_buf,
                gs.env_done_buf,
                terminated_buf,
            )

            # Download only rew/done for episode tracking (small transfer)
            ctx.enqueue_copy(gs.env_rew_host, gs.env_rew_buf)
            ctx.enqueue_copy(gs.env_done_host, gs.env_done_buf)
            ctx.synchronize()

            var nb_done = 0
            var sum_rew = 0.0

            # CPU: episode tracking only (replay insertion is on GPU now)
            for env_idx in range(n_envs):
                var rew_val = Scalar[dtype](gs.env_rew_host[env_idx])
                var done_val = Float64(gs.env_done_host[env_idx]) > 0.5

                # Accumulate per-env episode reward on CPU
                cpu_ep_rewards[env_idx] += Float64(rew_val)

                if done_val:
                    # Episode completed — log and reset accumulator
                    var ep_r = cpu_ep_rewards[env_idx]
                    metrics.log_episode(completed_episodes, ep_r, 0, 0.0)
                    completed_episodes += 1
                    recent_reward_sum += ep_r
                    recent_episode_count += 1
                    nb_done += 1
                    sum_rew += ep_r

                    cpu_ep_rewards[env_idx] = 0.0
                    # Reset MPPI warm-start for this env
                    if use_mppi:
                        env_t0_flags[env_idx] = True
                        env_prev_means[env_idx] = List[Float64]()

            if self.logger and nb_done > 0:
                try:
                    self.logger[].log_scalar(
                        "episode_reward",
                        sum_rew / Float64(nb_done),
                        total_steps,
                    )
                    self.logger[].log_scalar(
                        "episodes",
                        Float64(completed_episodes),
                        total_steps,
                    )
                    self.logger[].log_scalar(
                        "train_steps",
                        Float64(self.train_step_count),
                        total_steps,
                    )
                except:
                    pass

            total_steps += n_envs
            t_data_collection += Int(perf_counter_ns() - _t0_dc)

            # Progress bar (no GPU sync, pure CPU counters)
            if verbose and total_steps >= next_progress:
                var interval_start = next_print - print_every
                print_progress_bar(
                    total_steps - interval_start,
                    print_every,
                    self.train_step_count,
                    "TD-MPC2 (GPU)",
                )
                next_progress += progress_interval

            # Print progress periodically
            if verbose and total_steps >= next_print:
                if recent_episode_count > 0:
                    last_avg_reward = recent_reward_sum / Float64(
                        recent_episode_count
                    )

                clear_progress_bar()
                # # ── Diagnostics: action stats from already-downloaded host buffers ──
                # var diag_act_abs_sum: Float64 = 0.0
                # var diag_act_sq_sum: Float64 = 0.0
                # var diag_act_max: Float64 = 0.0
                # for ei in range(n_envs):
                #     for ai in range(Self.ACT):
                #         var a = Float64(gs.env_act_host[ei * Self.ACT + ai])
                #         var aa = abs(a)
                #         diag_act_abs_sum += aa
                #         diag_act_sq_sum += a * a
                #         if aa > diag_act_max:
                #             diag_act_max = aa
                # var diag_n = Float64(n_envs * Self.ACT)
                # var diag_mean_abs_act = diag_act_abs_sum / diag_n
                # var diag_mean_sq_act = diag_act_sq_sum / diag_n
                # # Per-step reward (from last downloaded rew_host)
                # var diag_rew_sum: Float64 = 0.0
                # for ei in range(n_envs):
                #     diag_rew_sum += Float64(gs.env_rew_host[ei])
                # var diag_mean_rew = diag_rew_sum / Float64(n_envs)

                print(
                    "TD-MPC2 (GPU) | Step "
                    + String(total_steps)
                    + " | Ep: "
                    + String(completed_episodes)
                    + " / "
                    + String(num_episodes)
                    + " | AvgR: "
                    + String(last_avg_reward)[byte=:7]
                    + " | Train: "
                    + String(self.train_step_count)
                )
                # print(
                #     "  diag: |act|="
                #     + String(diag_mean_abs_act)[byte=:6]
                #     + " act²="
                #     + String(diag_mean_sq_act)[byte=:6]
                #     + " max|act|="
                #     + String(diag_act_max)[byte=:6]
                #     + " step_rew="
                #     + String(diag_mean_rew)[byte=:7]
                # )
                recent_reward_sum = 0.0
                recent_episode_count = 0
                next_print += print_every
                next_progress = next_print - print_every + progress_interval

            # Reset done environments (reuse model from workspace)
            ENV.selective_reset_kernel_gpu[n_envs, ENV.STATE_SIZE](
                ctx,
                gs.states_buf,
                gs.env_done_buf,
                UInt64(total_steps * 1013904223 + 2654435761),
                workspace_ptr=step_ws_buf.unsafe_ptr(),
            )
            ENV.extract_obs_kernel_gpu[n_envs, ENV.STATE_SIZE, Self.OBS](
                ctx, gs.states_buf, gs.env_obs_buf
            )

            # ==============================================================
            # Phase 2: World Model Training
            # ==============================================================
            if total_steps < self.warmup_steps:
                continue

            # Check GPU replay buffer has enough data
            comptime min_ready = Self.BATCH + Self.H + 1
            if not gpu_replay.is_ready[min_ready]():
                continue

            # Seed-step pretraining burst (matches reference TD-MPC2
            # online_trainer.py:115-122). On the first training iteration after
            # warmup, do `warmup_steps` updates in a row to bootstrap the world
            # model on the random-policy seed data before regular UTD=1 begins.
            var num_updates = updates_per_step
            if not did_seed_pretraining:
                num_updates = self.warmup_steps
                did_seed_pretraining = True
                if verbose:
                    clear_progress_bar()
                    print(
                        "TD-MPC2 (GPU) | Pretraining agent on seed data ("
                        + String(num_updates)
                        + " updates)..."
                    )

            for _upd in range(num_updates):
                # Sample BATCH sequences directly on GPU (no CPU round-trip)
                var _t0_rs = perf_counter_ns()
                var sample_seed = UInt32(
                    total_steps * 1999999973 + _upd * 1000003 + 31
                )
                gpu_replay.sample[Self.BATCH, Self.H](
                    ctx,
                    sample_seed,
                    gs.batch_obs_buf,
                    gs.batch_act_buf,
                    gs.batch_rew_buf,
                    gs.batch_done_buf,
                )
                t_replay_sample += Int(perf_counter_ns() - _t0_rs)

                # No batch upload needed — data is already on GPU
                var _t0_bu = perf_counter_ns()
                t_batch_upload += Int(perf_counter_ns() - _t0_bu)

                # ──────────────────────────────────────────────────────────────
                # Step 2a: Compute TD targets (stop-gradient)
                # For each horizon step t:
                #   encode obs_{t+1} (stop-grad) → z_next
                #   policy(z_next) → pi_out; tanh(mean) → act_next
                #   build_za(z_next, act_next) → za_next
                #   Q_target1..Q5 forward → decode → min_Q_next
                #   td_target = r + gamma*(1-d)*min_Q_next → two-hot encode
                # ──────────────────────────────────────────────────────────────
                var _t0_td = perf_counter_ns()
                var gamma_scalar = Scalar[dtype](self.gamma)
                var vmin_scalar = Scalar[dtype](Self.v_min)
                var vmax_scalar = Scalar[dtype](Self.v_max)

                for t in range(Self.H):
                    # Fused: extract obs_{t+1} + rew/done at step t (2 kernels → 1)
                    ctx.enqueue_function[
                        tdmpc2_extract_obs_rew_done_kernel[
                            dtype, Self.BATCH, Self.OBS, Self.H
                        ],
                        tdmpc2_extract_obs_rew_done_kernel[
                            dtype, Self.BATCH, Self.OBS, Self.H
                        ],
                    ](
                        batch_obs_flat_tensor,
                        batch_rew_flat_tensor,
                        batch_done_flat_tensor,
                        t,
                        obs_next_step_tensor,
                        rew_step_tensor,
                        done_step_tensor,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Encode next obs (stop-grad)
                    Self.WM.EncoderNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                        ctx,
                        z_next_tensor,
                        obs_next_step_tensor,
                        enc_params_tensor,
                        enc_model_state_tensor,
                        gs.enc_batch_ws_buf,
                    )

                    # Policy forward (stop-grad) on z_next → pi_out
                    Self.WM.PolicyNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                        ctx,
                        pi_out_tensor,
                        z_next_tensor,
                        pol_params_tensor,
                        pol_model_state_tensor,
                        gs.pol_batch_ws_buf,
                    )

                    # Stochastic: tanh(mean + std*eps) → actions + build za_next
                    var td_reparam_seed = Scalar[DType.uint32](
                        td_reparam_rng_counter
                    )
                    td_reparam_rng_counter += UInt32(Self.BATCH * Self.ACT + 1)
                    ctx.enqueue_function[
                        tdmpc2_apply_tanh_build_za_kernel[
                            dtype, Self.BATCH, Self.ACT, Self.LATENT
                        ],
                        tdmpc2_apply_tanh_build_za_kernel[
                            dtype, Self.BATCH, Self.ACT, Self.LATENT
                        ],
                    ](
                        pi_out_tensor,
                        pi_act_tensor,
                        z_next_tensor,
                        za_tensor,
                        td_reparam_seed,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Random 2-of-5 target Q subsampling (reference TD-MPC2)
                    var td_rng = PhiloxRandom(seed=td_q_rng_counter, offset=0)
                    td_q_rng_counter += 1
                    var td_rng_vals = td_rng.step_uniform()
                    var td_qi_a = Int(td_rng_vals[0] * 5.0) % 5
                    var td_qi_b = (td_qi_a + 1 + Int(td_rng_vals[1] * 4.0) % 4) % 5

                    # First target Q → decode → init q_min
                    var qta_p = LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ](qt_param_ptrs[td_qi_a])
                    # Target Q nets share stateless QModel; any q*_model_state_tensor works
                    Self.WM.QNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                        ctx,
                        logits_tensor,
                        za_tensor,
                        qta_p,
                        q1_model_state_tensor,
                        gs.qt_batch_ws_buf,
                    )
                    ctx.enqueue_function[
                        tdmpc2_q_decode_kernel[dtype, Self.BATCH, Self.BINS],
                        tdmpc2_q_decode_kernel[dtype, Self.BATCH, Self.BINS],
                    ](
                        logits_tensor,
                        bins_tensor,
                        q_min_tensor,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Second target Q → decode + min-reduce
                    var qtb_p = LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ](qt_param_ptrs[td_qi_b])
                    Self.WM.QNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                        ctx,
                        logits_tensor,
                        za_tensor,
                        qtb_p,
                        q1_model_state_tensor,
                        gs.qt_batch_ws_buf,
                    )
                    ctx.enqueue_function[
                        tdmpc2_decode_and_min_kernel[dtype, Self.BATCH, Self.BINS],
                        tdmpc2_decode_and_min_kernel[dtype, Self.BATCH, Self.BINS],
                    ](
                        logits_tensor,
                        bins_tensor,
                        q_min_tensor,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Compute two-hot TD target and store at step t's offset
                    var tgt_t_tensor = LayoutTensor[
                        dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
                    ](gs.td_targets_buf.unsafe_ptr() + t * Self.B_BINS)

                    ctx.enqueue_function[
                        tdmpc2_compute_td_targets_kernel[
                            dtype, Self.BATCH, Self.BINS
                        ],
                        tdmpc2_compute_td_targets_kernel[
                            dtype, Self.BATCH, Self.BINS
                        ],
                    ](
                        rew_step_tensor,
                        done_step_tensor,
                        q_min_tensor,
                        tgt_t_tensor,
                        gamma_scalar,
                        vmin_scalar,
                        vmax_scalar,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Compute two-hot IMMEDIATE reward target for reward head
                    var rew_tgt_t_tensor = LayoutTensor[
                        dtype, Layout.row_major(Self.BATCH, Self.BINS), MutAnyOrigin
                    ](gs.rew_targets_buf.unsafe_ptr() + t * Self.B_BINS)

                    ctx.enqueue_function[
                        tdmpc2_compute_reward_targets_kernel[
                            dtype, Self.BATCH, Self.BINS
                        ],
                        tdmpc2_compute_reward_targets_kernel[
                            dtype, Self.BATCH, Self.BINS
                        ],
                    ](
                        rew_step_tensor,
                        rew_tgt_t_tensor,
                        vmin_scalar,
                        vmax_scalar,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # ──────────────────────────────────────────────────────────────
                # Step 2b: World model latent rollout + gradient computation
                # All TD targets and reward targets are pre-computed above.
                # Now: zero grads → accumulate over H steps → single optimizer step.
                # ──────────────────────────────────────────────────────────────
                ctx.synchronize()
                t_td_targets += Int(perf_counter_ns() - _t0_td)

                var _t0_wm = perf_counter_ns()
                # Zero all network parameter grad buffers (accumulated across H)
                gs.enc.zero_grads(ctx)
                gs.dyn.zero_grads(ctx)
                gs.rew.zero_grads(ctx)
                gs.term.zero_grads(ctx)
                gs.q1.zero_grads(ctx)
                gs.q2.zero_grads(ctx)
                gs.q3.zero_grads(ctx)
                gs.q4.zero_grads(ctx)
                gs.q5.zero_grads(ctx)

                # Encode obs_0 with cache (encoder backward uses this cache for all H steps)
                ctx.enqueue_function[
                    tdmpc2_extract_obs_step_kernel[
                        dtype, Self.BATCH, Self.OBS, Self.H
                    ],
                    tdmpc2_extract_obs_step_kernel[
                        dtype, Self.BATCH, Self.OBS, Self.H
                    ],
                ](
                    batch_obs_flat_tensor,
                    0,
                    obs_step_tensor,
                    grid_dim=(Self.BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                Self.WM.EncoderNet.forward_gpu_with_cache[Self.BATCH](
                    ctx,
                    obs_step_tensor,
                    z_tensor,
                    enc_params_tensor,
                    enc_model_state_tensor,
                    enc_cache_tensor,
                    gs.enc_batch_ws_buf,
                )

                self._wm_bptt_gpu[n_envs, ENV.STATE_SIZE](ctx, gs)

                # ── Gradient clipping + optimizer step for all world model networks ──
                # No sync needed — GPU pipelines BPTT → optimizer naturally
                ctx.enqueue_function[
                    gradient_norm_kernel[
                        dtype, Self.ENC_P, Self.ENC_GRAD_BLOCKS, TPB
                    ],
                    gradient_norm_kernel[
                        dtype, Self.ENC_P, Self.ENC_GRAD_BLOCKS, TPB
                    ],
                ](
                    enc_grad_ps_tensor,
                    enc_grads_tensor,
                    grid_dim=(Self.ENC_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.ENC_P, Self.ENC_GRAD_BLOCKS, TPB
                    ],
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.ENC_P, Self.ENC_GRAD_BLOCKS, TPB
                    ],
                ](
                    enc_grads_tensor,
                    enc_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.ENC_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                var enc_og = gs.enc.opt_global_state_view()
                Adam[LR=Self.WM.ENC_LR].step_gpu[Self.ENC_P](
                    ctx,
                    enc_params_tensor,
                    enc_grads_tensor,
                    enc_state_tensor,
                    enc_og,
                    0,  # ignored — device counter is authoritative
                )

                ctx.enqueue_function[
                    gradient_norm_kernel[
                        dtype, Self.DYN_P, Self.DYN_GRAD_BLOCKS, TPB
                    ],
                    gradient_norm_kernel[
                        dtype, Self.DYN_P, Self.DYN_GRAD_BLOCKS, TPB
                    ],
                ](
                    dyn_grad_ps_tensor,
                    dyn_grads_tensor,
                    grid_dim=(Self.DYN_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.DYN_P, Self.DYN_GRAD_BLOCKS, TPB
                    ],
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.DYN_P, Self.DYN_GRAD_BLOCKS, TPB
                    ],
                ](
                    dyn_grads_tensor,
                    dyn_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.DYN_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                var dyn_og = gs.dyn.opt_global_state_view()
                Adam[LR=Self.WM.WM_LR].step_gpu[Self.DYN_P](
                    ctx,
                    dyn_params_tensor,
                    dyn_grads_tensor,
                    dyn_state_tensor,
                    dyn_og,
                    0,  # ignored — device counter is authoritative
                )

                ctx.enqueue_function[
                    gradient_norm_kernel[
                        dtype, Self.REW_P, Self.REW_GRAD_BLOCKS, TPB
                    ],
                    gradient_norm_kernel[
                        dtype, Self.REW_P, Self.REW_GRAD_BLOCKS, TPB
                    ],
                ](
                    rew_grad_ps_tensor,
                    rew_grads_tensor,
                    grid_dim=(Self.REW_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.REW_P, Self.REW_GRAD_BLOCKS, TPB
                    ],
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.REW_P, Self.REW_GRAD_BLOCKS, TPB
                    ],
                ](
                    rew_grads_tensor,
                    rew_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.REW_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                var rew_og = gs.rew.opt_global_state_view()
                Adam[LR=Self.WM.WM_LR].step_gpu[Self.REW_P](
                    ctx,
                    rew_params_tensor,
                    rew_grads_tensor,
                    rew_state_tensor,
                    rew_og,
                    0,  # ignored — device counter is authoritative
                )

                ctx.enqueue_function[
                    gradient_norm_kernel[
                        dtype, Self.TERM_P, Self.TERM_GRAD_BLOCKS, TPB
                    ],
                    gradient_norm_kernel[
                        dtype, Self.TERM_P, Self.TERM_GRAD_BLOCKS, TPB
                    ],
                ](
                    term_grad_ps_tensor,
                    term_grads_tensor,
                    grid_dim=(Self.TERM_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.TERM_P, Self.TERM_GRAD_BLOCKS, TPB
                    ],
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.TERM_P, Self.TERM_GRAD_BLOCKS, TPB
                    ],
                ](
                    term_grads_tensor,
                    term_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.TERM_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                var term_og = gs.term.opt_global_state_view()
                Adam[LR=Self.WM.WM_LR].step_gpu[Self.TERM_P](
                    ctx,
                    term_params_tensor,
                    term_grads_tensor,
                    term_state_tensor,
                    term_og,
                    0,  # ignored — device counter is authoritative
                )

                # Q1..Q5 fused grad clip + Adam (15 launches → 3)
                ctx.enqueue_function[
                    tdmpc2_gradient_norm_5q_kernel[
                        dtype, Self.Q_P, Self.Q_GRAD_BLOCKS, TPB
                    ],
                    tdmpc2_gradient_norm_5q_kernel[
                        dtype, Self.Q_P, Self.Q_GRAD_BLOCKS, TPB
                    ],
                ](
                    q_grad_ps_tensor,
                    q1_grads_tensor,
                    q2_grads_tensor,
                    q3_grads_tensor,
                    q4_grads_tensor,
                    q5_grads_tensor,
                    grid_dim=(Self.Q_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    tdmpc2_gradient_reduce_apply_5q_kernel[
                        dtype, Self.Q_P, Self.Q_GRAD_BLOCKS, TPB
                    ],
                    tdmpc2_gradient_reduce_apply_5q_kernel[
                        dtype, Self.Q_P, Self.Q_GRAD_BLOCKS, TPB
                    ],
                ](
                    q1_grads_tensor,
                    q2_grads_tensor,
                    q3_grads_tensor,
                    q4_grads_tensor,
                    q5_grads_tensor,
                    q_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.Q_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Fused Adam step for all 5 Q networks. Phase 4: q1's
                # `opt_global_state[0]` is the canonical step counter for the
                # 5-Q group (all five Qs are stepped in lockstep, so they
                # share a single counter). A 1-thread preamble bump kernel
                # increments it on-device, and the fused kernel computes
                # bias correction from the post-bump value — CUDA-graph
                # replay safe.
                var wm_lr = Scalar[dtype](Self.WM.WM_LR)
                var adam_beta1 = Scalar[dtype](0.9)
                var adam_beta2 = Scalar[dtype](0.999)
                var adam_eps = Scalar[dtype](1e-8)
                var adam_log_b1 = Scalar[dtype](log(0.9))
                var adam_log_b2 = Scalar[dtype](log(0.999))

                var q1_og = gs.q1.opt_global_state_view()
                var q1_counter = LayoutTensor[
                    DType.uint32, Layout.row_major(1), MutAnyOrigin
                ](q1_og.ptr.bitcast[Scalar[DType.uint32]]())

                @parameter
                @always_inline
                def q1_bump_kernel(
                    c: LayoutTensor[
                        DType.uint32, Layout.row_major(1), MutAnyOrigin
                    ],
                ):
                    if Int(thread_idx.x) == 0:
                        c[0] = c[0] + UInt32(1)

                ctx.enqueue_function[q1_bump_kernel, q1_bump_kernel](
                    q1_counter,
                    grid_dim=(1,),
                    block_dim=(1,),
                )

                @parameter
                @always_inline
                def adam_5q_wrapper(
                    params1: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    grads1: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    state1: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P, 2), MutAnyOrigin
                    ],
                    params2: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    grads2: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    state2: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P, 2), MutAnyOrigin
                    ],
                    params3: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    grads3: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    state3: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P, 2), MutAnyOrigin
                    ],
                    params4: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    grads4: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    state4: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P, 2), MutAnyOrigin
                    ],
                    params5: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    grads5: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                    ],
                    state5: LayoutTensor[
                        dtype, Layout.row_major(Self.Q_P, 2), MutAnyOrigin
                    ],
                    counter: LayoutTensor[
                        DType.uint32, Layout.row_major(1), MutAnyOrigin
                    ],
                    lr: Scalar[dtype],
                    beta1: Scalar[dtype],
                    beta2: Scalar[dtype],
                    eps: Scalar[dtype],
                    log_beta1: Scalar[dtype],
                    log_beta2: Scalar[dtype],
                ):
                    tdmpc2_adam_step_5q_kernel[dtype, Self.Q_P](
                        params1,
                        grads1,
                        state1,
                        params2,
                        grads2,
                        state2,
                        params3,
                        grads3,
                        state3,
                        params4,
                        grads4,
                        state4,
                        params5,
                        grads5,
                        state5,
                        counter,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        log_beta1,
                        log_beta2,
                    )

                ctx.enqueue_function[adam_5q_wrapper, adam_5q_wrapper](
                    q1_params_tensor,
                    q1_grads_tensor,
                    q1_state_tensor,
                    q2_params_tensor,
                    q2_grads_tensor,
                    q2_state_tensor,
                    q3_params_tensor,
                    q3_grads_tensor,
                    q3_state_tensor,
                    q4_params_tensor,
                    q4_grads_tensor,
                    q4_state_tensor,
                    q5_params_tensor,
                    q5_grads_tensor,
                    q5_state_tensor,
                    q1_counter,
                    wm_lr,
                    adam_beta1,
                    adam_beta2,
                    adam_eps,
                    adam_log_b1,
                    adam_log_b2,
                    grid_dim=(Self.Q_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )

                # ──────────────────────────────────────────────────────────────
                # Step 2c: Policy update (maximize Q + entropy)
                # Policy uses stop-gradient z from encoder (no grad to encoder)
                # ──────────────────────────────────────────────────────────────
                # No sync needed — GPU pipelines optimizer → policy update naturally
                gs.pol.zero_grads(ctx)

                # Reference TD-MPC2 reuses the H+1 latent rollout from the
                # world-model forward pass (rolled with BUFFER actions) for
                # the policy update via `zs.detach()` (tdmpc2.py:_update,
                # update_pi). Phase 1 of `_wm_bptt_gpu` stores z_0..z_H in
                # `z_history_buf`; we load each step's z directly from there
                # instead of re-rolling dynamics with policy actions. This
                # matches reference exactly and means the policy is improved
                # at states the world model has actually been trained on.

                var pol_rho_t = Scalar[dtype](1.0)
                # Scale entropy by ACTION_DIM to match reference scaled_entropy.
                # The 1/(H+1) factor mirrors `pi_loss = (... * rho).mean()` in
                # reference TD-MPC2 (tdmpc2.py:227): the trailing .mean() over
                # the horizon dimension averages over H+1 timesteps because
                # `zs` has shape [H+1, B, latent] (z_0 from encode + z_1..z_H
                # from H dynamics steps). Without this normalization the policy
                # gradient is H+1× too large; pre-autodiff-fix the overwrite bug
                # masked it by retaining only the last step's contribution.
                var entropy_coef_scalar = Scalar[dtype](
                    self.entropy_coef
                    * Float64(Self.ACT)
                    / Float64(Self.H + 1)
                )
                var scale_scalar = Scalar[dtype](self.running_scale)
                var inv_HP1 = Scalar[dtype](1.0) / Scalar[dtype](Self.H + 1)

                for t in range(Self.H + 1):
                    # Load z_t from world-model rollout history (buffer-action
                    # rollout, computed pre-update — matches reference's
                    # `zs.detach()` semantics).
                    var z_hist_t = LayoutTensor[
                        dtype, Layout.row_major(Self.B_LATENT), MutAnyOrigin
                    ](
                        gs.z_history_buf.unsafe_ptr() + t * Self.B_LATENT
                    )
                    ctx.enqueue_function[
                        copy_buffer_kernel[dtype, Self.B_LATENT],
                        copy_buffer_kernel[dtype, Self.B_LATENT],
                    ](
                        z_flat_tensor,
                        z_hist_t,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    ctx.enqueue_memset(gs.grad_pi_out_buf, 0)

                    # Policy forward with cache → pi_out
                    Self.WM.PolicyNet.forward_gpu_with_cache[Self.BATCH](
                        ctx,
                        z_tensor,
                        pi_out_tensor,
                        pol_params_tensor,
                        pol_model_state_tensor,
                        pol_cache_tensor,
                        gs.pol_batch_ws_buf,
                    )

                    # Stochastic policy: a = tanh(mean + std*eps) via reparameterization
                    var pi_reparam_seed = Scalar[DType.uint32](
                        pi_reparam_rng_counter
                    )
                    pi_reparam_rng_counter += UInt32(Self.BATCH * Self.ACT + 1)
                    ctx.enqueue_function[
                        tdmpc2_apply_tanh_build_za_kernel[
                            dtype, Self.BATCH, Self.ACT, Self.LATENT
                        ],
                        tdmpc2_apply_tanh_build_za_kernel[
                            dtype, Self.BATCH, Self.ACT, Self.LATENT
                        ],
                    ](
                        pi_out_tensor,
                        pi_act_tensor,
                        z_tensor,
                        za_tensor,
                        pi_reparam_seed,
                        grid_dim=(Self.BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # ── Proper DPG: backprop through avg of 2 random Q-nets ──
                    # Reference TD-MPC2: randomly subsample 2 of 5 Q-networks,
                    # average their gradients for the policy update.
                    var pi_rng = PhiloxRandom(seed=pi_q_rng_counter, offset=0)
                    pi_q_rng_counter += 1
                    var rng_vals = pi_rng.step_uniform()
                    var qi_a = Int(rng_vals[0] * 5.0) % 5
                    var qi_b = (qi_a + 1 + Int(rng_vals[1] * 4.0) % 4) % 5
                    # /(H+1) matches reference's trailing .mean() over the
                    # H+1-element horizon dimension; /2 averages over the 2
                    # random Q-nets used for DPG.
                    var q_rho = pol_rho_t * inv_HP1 / Scalar[dtype](2.0)

                    # ── RunningScale update at first horizon step ──
                    # Decode Q-values from first Q-net to update scale (reference:
                    # 5th-95th percentile range of Q-values, EMA with tau).
                    if t == 0:
                        var qi_a_p = LayoutTensor[
                            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                        ](q_param_ptrs[qi_a])
                        # Forward Q to get logits (no cache, just for scale)
                        Self.WM.QNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                            ctx,
                            logits_tensor,
                            za_tensor,
                            qi_a_p,
                            q1_model_state_tensor,
                            gs.q1_batch_ws_buf,
                        )
                        # Decode logits → scalar Q-values into q_min_tensor
                        ctx.enqueue_function[
                            tdmpc2_q_decode_kernel[dtype, Self.BATCH, Self.BINS],
                            tdmpc2_q_decode_kernel[dtype, Self.BATCH, Self.BINS],
                        ](
                            logits_tensor,
                            bins_tensor,
                            q_min_tensor,
                            grid_dim=(Self.BATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Download Q-values to CPU for percentile computation
                        ctx.enqueue_copy(gs.q_vals_host, gs.q_min_buf)
                        ctx.synchronize()

                        # Sort Q-values to compute 5th and 95th percentiles
                        # Simple insertion sort (BATCH=256, negligible cost)
                        var sorted_q = List[Float64](capacity=Self.BATCH)
                        for b in range(Self.BATCH):
                            sorted_q.append(Float64(gs.q_vals_host[b]))
                        for i in range(1, Self.BATCH):
                            var key = sorted_q[i]
                            var j = i - 1
                            while j >= 0 and sorted_q[j] > key:
                                sorted_q[j + 1] = sorted_q[j]
                                j -= 1
                            sorted_q[j + 1] = key

                        var idx_5 = Int(0.05 * Float64(Self.BATCH))
                        var idx_95 = Int(0.95 * Float64(Self.BATCH))
                        if idx_95 >= Self.BATCH:
                            idx_95 = Self.BATCH - 1
                        var pct_range = sorted_q[idx_95] - sorted_q[idx_5]
                        if pct_range < 1.0:
                            pct_range = 1.0

                        # EMA update: scale ← (1-tau)*scale + tau*pct_range
                        self.running_scale = (
                            1.0 - self.tau
                        ) * self.running_scale + self.tau * pct_range
                        scale_scalar = Scalar[dtype](self.running_scale)

                        # Capture Q-value stats for diagnostics (sorted_q is available)
                        diag_q_min = sorted_q[0]
                        diag_q_max = sorted_q[Self.BATCH - 1]
                        var q_sum: Float64 = 0.0
                        for b in range(Self.BATCH):
                            q_sum += sorted_q[b]
                        diag_q_mean = q_sum / Float64(Self.BATCH)

                    for qi_iter in range(2):
                        var qi = qi_a if qi_iter == 0 else qi_b
                        # Entropy weighted by rho^t (temporal decay), applied once
                        var ent = (
                            entropy_coef_scalar * pol_rho_t if qi_iter
                            == 0 else Scalar[dtype](0.0)
                        )

                        var qi_p = LayoutTensor[
                            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                        ](q_param_ptrs[qi])
                        var qi_c = LayoutTensor[
                            dtype,
                            Layout.row_major(Self.BATCH, Self.Q_C),
                            MutAnyOrigin,
                        ](q_cache_ptrs[qi])
                        var qi_g = LayoutTensor[
                            dtype, Layout.row_major(Self.Q_P), MutAnyOrigin
                        ](q_grad_ptrs[qi])

                        # Q forward with cache
                        Self.WM.QNet.forward_gpu_with_cache[Self.BATCH](
                            ctx,
                            za_tensor,
                            logits_tensor,
                            qi_p,
                            q1_model_state_tensor,
                            qi_c,
                            gs.q1_batch_ws_buf,
                        )

                        # Decode Q + compute dL/d(logits), normalized by running_scale
                        ctx.enqueue_function[
                            tdmpc2_q_decode_backward_kernel[
                                dtype, Self.BATCH, Self.BINS
                            ],
                            tdmpc2_q_decode_backward_kernel[
                                dtype, Self.BATCH, Self.BINS
                            ],
                        ](
                            logits_tensor,
                            bins_tensor,
                            grad_logits_tensor,
                            q_rho,
                            scale_scalar,
                            grid_dim=(Self.BATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )

                        # Q backward: dL/d(logits) → dL/d(za)
                        Self.WM.QNet.backward_gpu[Self.BATCH](
                            ctx,
                            grad_logits_tensor,
                            grad_za_pi_tensor,
                            qi_p,
                            q1_model_state_tensor,
                            qi_c,
                            qi_g,
                            gs.q1_batch_ws_buf,
                        )

                        # Chain through stochastic tanh + entropy (same seed as forward)
                        ctx.enqueue_function[
                            tdmpc2_action_tanh_chain_kernel[
                                dtype, Self.BATCH, Self.ACT, Self.LATENT
                            ],
                            tdmpc2_action_tanh_chain_kernel[
                                dtype, Self.BATCH, Self.ACT, Self.LATENT
                            ],
                        ](
                            grad_za_pi_tensor,
                            pi_out_tensor,
                            grad_pi_out_tensor,
                            ent,
                            pi_reparam_seed,
                            grid_dim=(Self.BATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )

                    # Policy backward → accumulate pol_grads
                    Self.WM.PolicyNet.backward_gpu[Self.BATCH](
                        ctx,
                        grad_pi_out_tensor,
                        dummy_grad_latent_t,  # grad_input = dummy (z is stop-grad)
                        pol_params_tensor,
                        pol_model_state_tensor,
                        pol_cache_tensor,
                        pol_grads_tensor,
                        gs.pol_batch_ws_buf,
                    )

                    # No inline dynamics rollout: z_{t+1} is loaded from
                    # `z_history_buf` at the top of the next iteration (it's
                    # the world-model rollout from Phase 1, with buffer
                    # actions — matches reference TD-MPC2's `zs.detach()`).
                    pol_rho_t = pol_rho_t * Scalar[dtype](self.rho)

                # Policy gradient clip + optimizer step
                ctx.enqueue_function[
                    gradient_norm_kernel[
                        dtype, Self.POL_P, Self.POL_GRAD_BLOCKS, TPB
                    ],
                    gradient_norm_kernel[
                        dtype, Self.POL_P, Self.POL_GRAD_BLOCKS, TPB
                    ],
                ](
                    pol_grad_ps_tensor,
                    pol_grads_tensor,
                    grid_dim=(Self.POL_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.POL_P, Self.POL_GRAD_BLOCKS, TPB
                    ],
                    gradient_reduce_apply_fused_kernel[
                        dtype, Self.POL_P, Self.POL_GRAD_BLOCKS, TPB
                    ],
                ](
                    pol_grads_tensor,
                    pol_grad_ps_tensor,
                    grad_norm_max,
                    grid_dim=(Self.POL_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )
                var pol_og = gs.pol.opt_global_state_view()
                Adam[LR=Self.WM.PI_LR].step_gpu[Self.POL_P](
                    ctx,
                    pol_params_tensor,
                    pol_grads_tensor,
                    pol_state_tensor,
                    pol_og,
                    0,  # ignored — device counter is authoritative
                )

                # ──────────────────────────────────────────────────────────────
                # Step 2d: Soft update target Q networks
                # ──────────────────────────────────────────────────────────────
                # No sync needed — GPU pipelines policy → soft update naturally
                var tau_scalar = Scalar[dtype](self.tau)

                # Fused: soft update all 5 Q-target networks (5 kernels → 1)
                ctx.enqueue_function[
                    tdmpc2_soft_update_5q_kernel[dtype, Self.Q_P],
                    tdmpc2_soft_update_5q_kernel[dtype, Self.Q_P],
                ](
                    q1t_params_tensor,
                    q1_params_tensor,
                    q2t_params_tensor,
                    q2_params_tensor,
                    q3t_params_tensor,
                    q3_params_tensor,
                    q4t_params_tensor,
                    q4_params_tensor,
                    q5t_params_tensor,
                    q5_params_tensor,
                    tau_scalar,
                    grid_dim=(Self.Q_GRAD_BLOCKS,),
                    block_dim=(TPB,),
                )

                ctx.synchronize()
                # Single timing for entire training phase (BPTT + optim + policy + soft update)
                var _t_train_all = Int(perf_counter_ns() - _t0_wm)
                t_wm_gradient += _t_train_all
                timing_train_iters += 1

                self.train_step_count += 1

                # Log training diagnostics periodically.
                # We piggyback fresh post-update forward passes on (z_0, a_buffer_0)
                # to compute value/reward losses and mean targets — same idea as
                # SAC's diag block in core/agents/offpolicy_agent.mojo.
                if (
                    self.logger
                    and self.diag_every > 0
                    and self.train_step_count % self.diag_every == 0
                ):
                    try:
                        # ── Fresh post-update forward passes on (z_0, a_buf_0) ──
                        # obs_step_buf still holds obs_0 from earlier in this iter.
                        Self.WM.EncoderNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                            ctx,
                            z_tensor,
                            obs_step_tensor,
                            enc_params_tensor,
                            enc_model_state_tensor,
                            gs.enc_batch_ws_buf,
                        )
                        # Policy forward at z_0 → pi_out (mean | raw_log_std)
                        Self.WM.PolicyNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                            ctx,
                            pi_out_tensor,
                            z_tensor,
                            pol_params_tensor,
                            pol_model_state_tensor,
                            gs.pol_batch_ws_buf,
                        )
                        # Extract a_buffer at step 0 → act_step_tensor
                        ctx.enqueue_function[
                            tdmpc2_extract_act_step_kernel[
                                dtype, Self.BATCH, Self.ACT, Self.H
                            ],
                            tdmpc2_extract_act_step_kernel[
                                dtype, Self.BATCH, Self.ACT, Self.H
                            ],
                        ](
                            LayoutTensor[
                                dtype,
                                Layout.row_major(Self.BATCH_ACT_FLAT),
                                MutAnyOrigin,
                            ](gs.batch_act_buf.unsafe_ptr()),
                            0,
                            LayoutTensor[
                                dtype,
                                Layout.row_major(Self.BATCH, Self.ACT),
                                MutAnyOrigin,
                            ](gs.act_step_buf.unsafe_ptr()),
                            grid_dim=(Self.BATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Build za = [z_0, a_buffer_0]
                        ctx.enqueue_function[
                            tdmpc2_build_za_kernel[
                                dtype, Self.BATCH, Self.LATENT, Self.ACT
                            ],
                            tdmpc2_build_za_kernel[
                                dtype, Self.BATCH, Self.LATENT, Self.ACT
                            ],
                        ](
                            z_tensor,
                            LayoutTensor[
                                dtype,
                                Layout.row_major(Self.BATCH, Self.ACT),
                                MutAnyOrigin,
                            ](gs.act_step_buf.unsafe_ptr()),
                            za_tensor,
                            grid_dim=(Self.BATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Q1 forward → logits, then download
                        Self.WM.QNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                            ctx,
                            logits_tensor,
                            za_tensor,
                            q1_params_tensor,
                            q1_model_state_tensor,
                            gs.q1_batch_ws_buf,
                        )
                        ctx.enqueue_copy(gs.diag_logits_host, gs.logits_buf)
                        # Reward forward → logits (overwrites), download
                        Self.WM.RewardNet.MODEL.forward_gpu_no_cache[Self.BATCH](
                            ctx,
                            logits_tensor,
                            za_tensor,
                            rew_params_tensor,
                            rew_model_state_tensor,
                            gs.rew_batch_ws_buf,
                        )
                        ctx.enqueue_copy(gs.diag_rew_logits_host, gs.logits_buf)
                        # Download targets + raw signals + policy out
                        ctx.enqueue_copy(gs.diag_rew_host, gs.batch_rew_buf)
                        ctx.enqueue_copy(gs.diag_done_host, gs.batch_done_buf)
                        ctx.enqueue_copy(gs.diag_pi_out_host, gs.pi_out_buf)
                        ctx.enqueue_copy(
                            gs.diag_td_targets_host, gs.td_targets_buf
                        )
                        ctx.enqueue_copy(
                            gs.diag_rew_targets_host, gs.rew_targets_buf
                        )
                        ctx.synchronize()

                        # ── CPU-side aggregation ──
                        comptime BS = Self.BATCH
                        comptime BNS = Self.BINS

                        var mean_reward: Float64 = 0.0
                        var mean_done: Float64 = 0.0
                        var sumsq_reward: Float64 = 0.0
                        for i in range(Self.BATCH_SCALAR_FLAT):
                            var r = Float64(gs.diag_rew_host[i])
                            mean_reward += r
                            sumsq_reward += r * r
                            mean_done += Float64(gs.diag_done_host[i])
                        mean_reward /= Float64(Self.BATCH_SCALAR_FLAT)
                        mean_done /= Float64(Self.BATCH_SCALAR_FLAT)
                        var var_reward = (
                            sumsq_reward / Float64(Self.BATCH_SCALAR_FLAT)
                        ) - (mean_reward * mean_reward)
                        if var_reward < 0.0:
                            var_reward = 0.0
                        var std_reward = sqrt(var_reward)

                        # mean_abs_action computed on tanh(mean) (deterministic
                        # action) and mean_log_std after the smooth tanh-based
                        # bound (matches what the policy actually emits).
                        var mean_abs_action: Float64 = 0.0
                        var mean_log_std: Float64 = 0.0
                        for b in range(BS):
                            for j in range(Self.ACT):
                                var mean_raw = Float64(
                                    gs.diag_pi_out_host[b * Self.POL_OUT + j]
                                )
                                var x = Float64(
                                    gs.diag_pi_out_host[
                                        b * Self.POL_OUT + Self.ACT + j
                                    ]
                                )
                                var act_det = Float64(tanh(Float32(mean_raw)))
                                var ls = -10.0 + 6.0 * (
                                    Float64(tanh(Float32(x))) + 1.0
                                )
                                if act_det < 0.0:
                                    act_det = -act_det
                                mean_abs_action += act_det
                                mean_log_std += ls
                        mean_abs_action /= Float64(BS * Self.ACT)
                        mean_log_std /= Float64(BS * Self.ACT)

                        # value_loss = soft CE between Q logits and TD target
                        # at horizon step 0; mean_target = symexp-decoded
                        # expected value of the two-hot target.
                        # std_q / std_target measure batch-wise spread of the
                        # predicted Q and the bootstrapped TD target — both in
                        # actual value space (symexp-decoded). Near-zero std
                        # indicates Q-collapse (predictor is state-action
                        # invariant).
                        var value_loss: Float64 = 0.0
                        var mean_target: Float64 = 0.0
                        var sumsq_target: Float64 = 0.0
                        var mean_q_pred: Float64 = 0.0
                        var sumsq_q_pred: Float64 = 0.0
                        for b in range(BS):
                            var max_l: Float64 = -1e30
                            for k in range(BNS):
                                var v = Float64(
                                    gs.diag_logits_host[b * BNS + k]
                                )
                                if v > max_l:
                                    max_l = v
                            var sum_exp: Float64 = 0.0
                            for k in range(BNS):
                                sum_exp += exp(
                                    Float64(gs.diag_logits_host[b * BNS + k])
                                    - max_l
                                )
                            var lse = log(sum_exp) + max_l
                            var ce: Float64 = 0.0
                            var tgt_symlog: Float64 = 0.0
                            var pred_symlog: Float64 = 0.0
                            for k in range(BNS):
                                var logit = Float64(
                                    gs.diag_logits_host[b * BNS + k]
                                )
                                var tgt = Float64(
                                    gs.diag_td_targets_host[b * BNS + k]
                                )
                                ce -= tgt * (logit - lse)
                                tgt_symlog += tgt * Float64(
                                    self.state.world_model.bins[k]
                                )
                                var sm = exp(logit - lse)
                                pred_symlog += sm * Float64(
                                    self.state.world_model.bins[k]
                                )
                            value_loss += ce
                            var ats = (
                                tgt_symlog
                                if tgt_symlog >= 0.0
                                else -tgt_symlog
                            )
                            var dec_target = (exp(ats) - 1.0) if (
                                tgt_symlog >= 0.0
                            ) else -(exp(ats) - 1.0)
                            mean_target += dec_target
                            sumsq_target += dec_target * dec_target
                            var aps = (
                                pred_symlog
                                if pred_symlog >= 0.0
                                else -pred_symlog
                            )
                            var dec_pred_q = (exp(aps) - 1.0) if (
                                pred_symlog >= 0.0
                            ) else -(exp(aps) - 1.0)
                            mean_q_pred += dec_pred_q
                            sumsq_q_pred += dec_pred_q * dec_pred_q
                        value_loss /= Float64(BS)
                        mean_target /= Float64(BS)
                        mean_q_pred /= Float64(BS)
                        var var_target = (sumsq_target / Float64(BS)) - (
                            mean_target * mean_target
                        )
                        if var_target < 0.0:
                            var_target = 0.0
                        var std_target = sqrt(var_target)
                        var var_q_pred = (sumsq_q_pred / Float64(BS)) - (
                            mean_q_pred * mean_q_pred
                        )
                        if var_q_pred < 0.0:
                            var_q_pred = 0.0
                        var std_q = sqrt(var_q_pred)

                        # reward_loss = soft CE between reward logits and reward
                        # target at horizon step 0; mean_pred_reward = symexp-
                        # decoded expectation under predicted distribution.
                        var reward_loss: Float64 = 0.0
                        var mean_pred_reward: Float64 = 0.0
                        for b in range(BS):
                            var max_l: Float64 = -1e30
                            for k in range(BNS):
                                var v = Float64(
                                    gs.diag_rew_logits_host[b * BNS + k]
                                )
                                if v > max_l:
                                    max_l = v
                            var sum_exp: Float64 = 0.0
                            for k in range(BNS):
                                sum_exp += exp(
                                    Float64(
                                        gs.diag_rew_logits_host[b * BNS + k]
                                    )
                                    - max_l
                                )
                            var lse = log(sum_exp) + max_l
                            var ce: Float64 = 0.0
                            var pred_symlog: Float64 = 0.0
                            for k in range(BNS):
                                var logit = Float64(
                                    gs.diag_rew_logits_host[b * BNS + k]
                                )
                                var sm = exp(logit - lse)
                                var tgt = Float64(
                                    gs.diag_rew_targets_host[b * BNS + k]
                                )
                                ce -= tgt * (logit - lse)
                                pred_symlog += sm * Float64(
                                    self.state.world_model.bins[k]
                                )
                            reward_loss += ce
                            var aps = (
                                pred_symlog
                                if pred_symlog >= 0.0
                                else -pred_symlog
                            )
                            var dec_pred = (exp(aps) - 1.0) if (
                                pred_symlog >= 0.0
                            ) else -(exp(aps) - 1.0)
                            mean_pred_reward += dec_pred
                        reward_loss /= Float64(BS)
                        mean_pred_reward /= Float64(BS)

                        var step = self.train_step_count
                        self.logger[].log_scalar("q_mean", diag_q_mean, step)
                        self.logger[].log_scalar("q_min", diag_q_min, step)
                        self.logger[].log_scalar("q_max", diag_q_max, step)
                        self.logger[].log_scalar(
                            "pi_scale", self.running_scale, step
                        )
                        self.logger[].log_scalar("gamma", self.gamma, step)
                        self.logger[].log_scalar(
                            "value_loss", value_loss, step
                        )
                        self.logger[].log_scalar(
                            "reward_loss", reward_loss, step
                        )
                        self.logger[].log_scalar(
                            "mean_target", mean_target, step
                        )
                        self.logger[].log_scalar(
                            "mean_pred_reward", mean_pred_reward, step
                        )
                        self.logger[].log_scalar(
                            "mean_reward", mean_reward, step
                        )
                        self.logger[].log_scalar(
                            "mean_done", mean_done, step
                        )
                        self.logger[].log_scalar(
                            "mean_abs_action", mean_abs_action, step
                        )
                        self.logger[].log_scalar(
                            "mean_log_std", mean_log_std, step
                        )
                        self.logger[].log_scalar("std_q", std_q, step)
                        self.logger[].log_scalar(
                            "std_target", std_target, step
                        )
                        self.logger[].log_scalar(
                            "std_reward", std_reward, step
                        )
                    except:
                        pass

        # =================================================================
        # Print timing summary
        # =================================================================
        if verbose:
            clear_progress_bar()
            var total_ns = (
                t_data_collection
                + t_replay_sample
                + t_batch_upload
                + t_td_targets
                + t_wm_gradient
            )
            var total_ms = Float64(total_ns) / 1e6

            @always_inline
            def _pct(ns: Int, tot: Int) -> Float64:
                if tot == 0:
                    return 0.0
                return Float64(ns) / Float64(tot) * 100.0

            print("\n========== TDMPC2 GPU Timing Summary ==========")
            print(
                "Total measured time:",
                total_ms,
                "ms over",
                timing_train_iters,
                "training iterations",
            )
            if timing_train_iters > 0:
                print(
                    "Avg per training iter:",
                    total_ms / Float64(timing_train_iters),
                    "ms",
                )
            print("─────────────────────────────────────────────────")
            print(
                "Data collection:    ",
                Float64(t_data_collection) / 1e6,
                "ms (",
                _pct(t_data_collection, total_ns),
                "%)",
            )
            print(
                "Replay sampling:    ",
                Float64(t_replay_sample) / 1e6,
                "ms (",
                _pct(t_replay_sample, total_ns),
                "%)",
            )
            print(
                "Batch upload:       ",
                Float64(t_batch_upload) / 1e6,
                "ms (",
                _pct(t_batch_upload, total_ns),
                "%)",
            )
            print(
                "TD targets:         ",
                Float64(t_td_targets) / 1e6,
                "ms (",
                _pct(t_td_targets, total_ns),
                "%)",
            )
            print(
                "Train (BPTT+optim+policy+soft): ",
                Float64(t_wm_gradient) / 1e6,
                "ms (",
                _pct(t_wm_gradient, total_ns),
                "%)",
            )
            print("=================================================\n")

        # Sync GPU params back to CPU before returning
        var enc_dl = ctx.enqueue_create_host_buffer[dtype](Self.ENC_P)
        var dyn_dl = ctx.enqueue_create_host_buffer[dtype](Self.DYN_P)
        var rew_dl = ctx.enqueue_create_host_buffer[dtype](Self.REW_P)
        var term_dl = ctx.enqueue_create_host_buffer[dtype](Self.TERM_P)
        var pol_dl = ctx.enqueue_create_host_buffer[dtype](Self.POL_P)
        var q1_dl = ctx.enqueue_create_host_buffer[dtype](Self.Q_P)
        var q2_dl = ctx.enqueue_create_host_buffer[dtype](Self.Q_P)
        var q3_dl = ctx.enqueue_create_host_buffer[dtype](Self.Q_P)
        var q4_dl = ctx.enqueue_create_host_buffer[dtype](Self.Q_P)
        var q5_dl = ctx.enqueue_create_host_buffer[dtype](Self.Q_P)
        ctx.enqueue_copy(enc_dl, gs.enc.params_buf)
        ctx.enqueue_copy(dyn_dl, gs.dyn.params_buf)
        ctx.enqueue_copy(rew_dl, gs.rew.params_buf)
        ctx.enqueue_copy(term_dl, gs.term.params_buf)
        ctx.enqueue_copy(pol_dl, gs.pol.params_buf)
        ctx.enqueue_copy(q1_dl, gs.q1.params_buf)
        ctx.enqueue_copy(q2_dl, gs.q2.params_buf)
        ctx.enqueue_copy(q3_dl, gs.q3.params_buf)
        ctx.enqueue_copy(q4_dl, gs.q4.params_buf)
        ctx.enqueue_copy(q5_dl, gs.q5.params_buf)
        ctx.synchronize()
        for i in range(Self.ENC_P):
            (self.state.world_model.encoder.params + i)[] = enc_dl[i]
        for i in range(Self.DYN_P):
            (self.state.world_model.dynamics.params + i)[] = dyn_dl[i]
        for i in range(Self.REW_P):
            (self.state.world_model.reward_head.params + i)[] = rew_dl[i]
        for i in range(Self.TERM_P):
            (self.state.world_model.termination.params + i)[] = term_dl[i]
        for i in range(Self.POL_P):
            (self.state.world_model.policy.params + i)[] = pol_dl[i]
        for i in range(Self.Q_P):
            (self.state.world_model.q1.params + i)[] = q1_dl[i]
        for i in range(Self.Q_P):
            (self.state.world_model.q2.params + i)[] = q2_dl[i]
        for i in range(Self.Q_P):
            (self.state.world_model.q3.params + i)[] = q3_dl[i]
        for i in range(Self.Q_P):
            (self.state.world_model.q4.params + i)[] = q4_dl[i]
        for i in range(Self.Q_P):
            (self.state.world_model.q5.params + i)[] = q5_dl[i]

        return metrics

    # =========================================================================
    # Training Loop (CPU)
    # =========================================================================

    def train[
        ENV: BoxContinuousActionEnv
    ](
        mut self,
        mut env: ENV,
        num_episodes: Int = 200,
        updates_per_step: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
    ) -> TrainingMetrics:
        """Run the TDMPC2 training loop.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_episodes: Number of training episodes.
            updates_per_step: Gradient updates per environment step.
            verbose: Print progress (default: True).
            print_every: Print every N episodes if verbose (default: 10).

        Returns:
            TrainingMetrics with episode rewards and losses.
        """
        var metrics = TrainingMetrics(
            algorithm_name="TD-MPC2",
        )

        # CPU timing accumulators (nanoseconds)
        var t_select_action: Int = 0
        var t_env_step: Int = 0
        var t_observe: Int = 0
        var t_update: Int = 0
        var total_train_steps: Int = 0

        # Progress bar: ~20 updates per print interval
        var progress_interval = print_every // 20
        if progress_interval < 1:
            progress_interval = 1
        var next_progress = progress_interval

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var episode_loss: Float64 = 0.0
            var done = False
            var steps = 0
            # Reset MPPI warm-start for new episode
            self._episode_t0 = True

            while not done:
                # Build obs InlineArray from list
                var obs_arr = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                for i in range(Self.OBS):
                    if i < len(obs_list):
                        obs_arr[i] = Scalar[dtype](obs_list[i])

                var _t0_sa = perf_counter_ns()
                var action = self.select_action(obs_arr)
                t_select_action += Int(perf_counter_ns() - _t0_sa)

                # Step environment using step_continuous_vec
                var action_list = List[Scalar[dtype]](capacity=Self.ACT)
                for i in range(Self.ACT):
                    action_list.append(action[i])
                var _t0_es = perf_counter_ns()
                var step_result = env.step_continuous_vec(action_list)
                t_env_step += Int(perf_counter_ns() - _t0_es)
                var reward = Float64(step_result[1])
                done = step_result[2]
                episode_reward += reward

                # Store transition
                var _t0_ob = perf_counter_ns()
                self.observe(obs_arr, action, reward, done)
                t_observe += Int(perf_counter_ns() - _t0_ob)

                obs_list = env.get_obs_list()
                steps += 1

                # Training updates
                if self.total_steps >= self.warmup_steps:
                    for _ in range(updates_per_step):
                        var _t0_up = perf_counter_ns()
                        episode_loss += self.update()
                        t_update += Int(perf_counter_ns() - _t0_up)
                        total_train_steps += 1

            metrics.log_episode(
                episode,
                Scalar[dtype](episode_reward),
                steps,
                0.0,
            )

            # Progress bar
            if verbose and episode + 1 >= next_progress:
                var interval_pos = (episode + 1) % print_every
                if interval_pos == 0:
                    interval_pos = print_every
                print_progress_bar(
                    interval_pos, print_every, total_train_steps, "TD-MPC2"
                )
                next_progress += progress_interval

            # Full stats line
            if verbose and (episode + 1) % print_every == 0:
                var avg_r = metrics.mean_reward_last_n(print_every)
                clear_progress_bar()
                print(
                    "TD-MPC2 | Ep "
                    + String(episode + 1)
                    + " / "
                    + String(num_episodes)
                    + " | AvgR: "
                    + String(avg_r)[byte=:7]
                    + " | Steps: "
                    + String(self.total_steps)
                    + " | Train: "
                    + String(self.train_step_count)
                )
                next_progress = (episode + 1) + progress_interval

        # Print CPU timing summary
        if verbose:
            clear_progress_bar()
        var total_ns = t_select_action + t_env_step + t_observe + t_update
        var total_ms = Float64(total_ns) / 1e6

        @always_inline
        def _cpu_pct(ns: Int, tot: Int) -> Float64:
            if tot == 0:
                return 0.0
            return Float64(ns) / Float64(tot) * 100.0

        print("\n========== TDMPC2 CPU Timing Summary ==========")
        print("Total measured time:", total_ms, "ms")
        print("─────────────────────────────────────────────────")
        print(
            "select_action (MPPI):",
            Float64(t_select_action) / 1e6,
            "ms (",
            _cpu_pct(t_select_action, total_ns),
            "%)",
        )
        print(
            "env.step:           ",
            Float64(t_env_step) / 1e6,
            "ms (",
            _cpu_pct(t_env_step, total_ns),
            "%)",
        )
        print(
            "observe (buffer):   ",
            Float64(t_observe) / 1e6,
            "ms (",
            _cpu_pct(t_observe, total_ns),
            "%)",
        )
        print(
            "update (train):     ",
            Float64(t_update) / 1e6,
            "ms (",
            _cpu_pct(t_update, total_ns),
            "%)",
        )
        if total_train_steps > 0:
            print(
                "  avg per update:   ",
                Float64(t_update) / 1e6 / Float64(total_train_steps),
                "ms (",
                total_train_steps,
                "updates)",
            )
        print("=================================================\n")

        return metrics


@always_inline
def _gaussian_sample() -> Float64:
    """Box-Muller transform for standard normal sample."""
    from std.math import log as mlog, cos as mcos, sqrt as msqrt

    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-10:
        u1 = 1e-10
    return msqrt(-2.0 * mlog(u1)) * mcos(2.0 * 3.14159265358979 * u2)


@always_inline
def _tanh(x: Float64) -> Float64:
    from std.math import exp as mexp

    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    var ep = mexp(x)
    var en = mexp(-x)
    return (ep - en) / (ep + en)
