"""DreamerV3 Agent — World model-based RL with RSSM and latent imagination.

Learns an RSSM world model from environment interactions, then trains
an actor-critic purely in imagined latent trajectories. Uses distributional
value estimation (two-hot / symlog bins), KL balancing, and percentile-based
return normalization.

Architecture:
  World Model (RSSM): encoder, GRU dynamics, posterior, prior, decoder,
                       reward head (distributional), continue head
  Actor: feat -> tanh-normal (mean, log_std) via Parallel output heads
  Critic: feat -> NUM_BINS logits (distributional, with slow EMA target)

Reference: Hafner et al., 2023 — Mastering Diverse Domains through
World Models (DreamerV3)
"""

from std.math import exp, log, sqrt
from std.random import random_float64
from std.time import perf_counter_ns
from std.memory import alloc, memset
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.loss.two_hot import (
    compute_symlog_bins,
    two_hot_encode,
    decode_value,
    symlog,
    symexp,
)
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from .rssm import RSSM, categorical_sample, kl_divergence
from .state import DreamerV3CPUState, DreamerV3GPUState
from .imagination import (
    compute_lambda_returns,
    normalize_returns,
    sample_tanh_normal,
    log_prob_tanh_normal,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from .kernels import (
    symlog_kernel,
    symexp_kernel,
    gru_gate_kernel,
    concat_feat_kernel,
    concat_gru_input_kernel,
    concat_deter_embed_kernel,
    action_normalize_kernel,
    categorical_sample_kernel,
    kl_divergence_kernel,
    kl_categorical_gradient_kernel,
    gru_gate_backward_kernel,
    accumulate_kernel,
    concat_feat_backward_kernel,
    concat_deter_embed_backward_kernel,
    concat_gru_input_backward_kernel,
    clamp_kernel,
    min_max_reduce_kernel,
    normalize_advantages_kernel,
    reparam_tanh_backward_kernel,
    lambda_returns_kernel,
    normalize_returns_elementwise_kernel,
    two_hot_ce_grad_kernel,
    mse_grad_kernel,
    bce_grad_kernel,
    tanh_normal_sample_kernel,
    reinforce_grad_kernel,
    decode_value_kernel,
    two_hot_encode_kernel,
    sigmoid_kernel,
    copy_kernel,
    zero_kernel,
    advantage_kernel,
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
    straight_through_softmax_vjp_kernel,
    deinterleave_kernel,
    interleave_kernel,
    one_minus_kernel,
    TPB,
)
from mojo_rl.nn.autodiff.composite_params import CompositeParams


# =============================================================================
# DreamerV3 Agent
# =============================================================================


struct DreamerV3Agent[
    obs_dim: Int,
    action_dim: Int,
    deter_dim: Int = 512,
    hidden: Int = 128,
    stoch_dim: Int = 8,
    classes: Int = 8,
    units: Int = 128,
    num_bins: Int = 255,
    blocks: Int = 4,
    batch_size: Int = 16,
    batch_length: Int = 64,
    imagine_horizon: Int = 15,
    buffer_capacity: Int = 1000000,
    wm_lr: Float64 = 1e-4,
    actor_lr: Float64 = 3e-5,
    critic_lr: Float64 = 3e-5,
    free_nats: Float64 = 1.0,
    L: Logger = NoOpLogger,
](Movable):
    """DreamerV3 agent for continuous control.

    Learns an RSSM world model from environment data, then optimizes
    an actor-critic in purely imagined latent trajectories. The critic
    uses distributional value estimation with symlog two-hot encoding.

    Parameters:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        deter_dim: GRU deterministic state dimension (default: 512).
        hidden: GRU projection hidden dimension (default: 128).
        stoch_dim: Number of categorical stochastic variables (default: 8).
        classes: Number of classes per stochastic variable (default: 8).
        units: Actor/critic hidden width (default: 128).
        num_bins: Distributional critic bins (default: 255).
        blocks: RSSM blocks (default: 4).
        batch_size: Training batch size (default: 16).
        batch_length: BPTT sequence length (default: 64).
        imagine_horizon: Imagination rollout length (default: 15).
        buffer_capacity: Replay buffer capacity (default: 1M).
        wm_lr: World model learning rate (default: 1e-4).
        actor_lr: Actor learning rate (default: 3e-5).
        critic_lr: Critic learning rate (default: 3e-5).
        free_nats: Free nats threshold for KL loss (default: 1.0).
        L: Logger type for diagnostics (default: NoOpLogger).
    """

    # ── Derived compile-time constants ────────────────────────────────────
    comptime STOCH_FLAT: Int = Self.stoch_dim * Self.classes
    comptime FEAT_DIM: Int = Self.deter_dim + Self.STOCH_FLAT
    comptime IMAG_BATCH: Int = Self.batch_size * Self.batch_length

    # ── State type aliases ────────────────────────────────────────────────
    comptime StateType = DreamerV3CPUState[
        Self.obs_dim,
        Self.action_dim,
        Self.deter_dim,
        Self.hidden,
        Self.stoch_dim,
        Self.classes,
        Self.units,
        Self.num_bins,
        Self.blocks,
        WM_LR=Self.wm_lr,
        ACTOR_LR=Self.actor_lr,
        CRITIC_LR=Self.critic_lr,
        FREE_NATS=Self.free_nats,
        BUFFER_CAPACITY=Self.buffer_capacity,
        BATCH_SIZE=Self.batch_size,
        BATCH_LENGTH=Self.batch_length,
        IMAGINE_HORIZON=Self.imagine_horizon,
    ]

    comptime GPUStateType = DreamerV3GPUState[
        Self.obs_dim,
        Self.action_dim,
        Self.deter_dim,
        Self.hidden,
        Self.stoch_dim,
        Self.classes,
        Self.units,
        Self.num_bins,
        Self.blocks,
        WM_LR=Self.wm_lr,
        ACTOR_LR=Self.actor_lr,
        CRITIC_LR=Self.critic_lr,
        FREE_NATS=Self.free_nats,
        BATCH_SIZE=Self.batch_size,
        BATCH_LENGTH=Self.batch_length,
        IMAGINE_HORIZON=Self.imagine_horizon,
    ]

    # ── Actor/Critic Network aliases (matching state.mojo definitions) ───
    comptime ActorNet = Network[Self.StateType.ActorModel, Adam[LR=Self.actor_lr]]
    comptime CriticNet = Network[Self.StateType.CriticModel, Adam[LR=Self.critic_lr]]

    # ── Core state ────────────────────────────────────────────────────────
    var state: Self.StateType

    # Hyperparameters
    var gamma: Float64
    var lambda_: Float64
    var kl_balance: Float64
    var actor_entropy: Float64
    var slow_critic_tau: Float64
    var return_norm_rate: Float64
    var max_grad_norm: Float64

    # Running state for inference (single env)
    var _current_deter: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var _current_stoch: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var _prev_action: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # Diagnostics
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    # Step counters
    var total_steps: Int
    var train_step_count: Int
    var warmup_steps: Int

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    fn __init__(
        out self,
        gamma: Float64 = 0.997,
        lambda_: Float64 = 0.95,
        kl_balance: Float64 = 0.8,
        actor_entropy: Float64 = 3e-4,
        slow_critic_tau: Float64 = 0.02,
        return_norm_rate: Float64 = 0.01,
        warmup_steps: Int = 1000,
        max_grad_norm: Float64 = 100.0,
        diag_every: Int = 0,
    ):
        """Initialize DreamerV3 agent with all sub-networks and buffers.

        Args:
            gamma: Discount factor (default: 0.997).
            lambda_: Lambda for generalized lambda returns (default: 0.95).
            kl_balance: KL balancing coefficient (default: 0.8).
            actor_entropy: Entropy regularization weight (default: 3e-4).
            slow_critic_tau: Slow critic EMA coefficient (default: 0.02).
            return_norm_rate: Return normalization EMA rate (default: 0.01).
            warmup_steps: Random exploration steps before training (default: 1000).
            max_grad_norm: Maximum gradient norm for clipping (default: 1000.0).
            diag_every: Log diagnostics every N steps (0 = every step).
        """
        self.state = Self.StateType()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_balance = kl_balance
        self.actor_entropy = actor_entropy
        self.slow_critic_tau = slow_critic_tau
        self.return_norm_rate = return_norm_rate
        self.max_grad_norm = max_grad_norm
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = diag_every
        self.total_steps = 0
        self.train_step_count = 0
        self.warmup_steps = warmup_steps

        # Allocate and zero-initialize running state for single-env inference
        self._current_deter = alloc[Scalar[dtype]](Self.deter_dim)
        memset(self._current_deter, 0, Self.deter_dim)
        self._current_stoch = alloc[Scalar[dtype]](Self.STOCH_FLAT)
        memset(self._current_stoch, 0, Self.STOCH_FLAT)
        self._prev_action = alloc[Scalar[dtype]](Self.action_dim)
        memset(self._prev_action, 0, Self.action_dim)

    fn __init__(out self, *, deinit take: Self):
        """Move constructor — transfer ownership of all fields."""
        self.state = take.state^
        self.gamma = take.gamma
        self.lambda_ = take.lambda_
        self.kl_balance = take.kl_balance
        self.actor_entropy = take.actor_entropy
        self.slow_critic_tau = take.slow_critic_tau
        self.return_norm_rate = take.return_norm_rate
        self.max_grad_norm = take.max_grad_norm
        self.logger = take.logger
        self.diag_every = take.diag_every
        self.total_steps = take.total_steps
        self.train_step_count = take.train_step_count
        self.warmup_steps = take.warmup_steps
        self._current_deter = take._current_deter
        self._current_stoch = take._current_stoch
        self._prev_action = take._prev_action

    # ══════════════════════════════════════════════════════════════════════
    # Episode Management
    # ══════════════════════════════════════════════════════════════════════

    fn reset_episode(mut self):
        """Reset the running RSSM state for a new episode.

        Zeros out the deterministic state, stochastic state, and previous
        action buffers used during single-environment inference.
        """
        memset(self._current_deter, 0, Self.deter_dim)
        memset(self._current_stoch, 0, Self.STOCH_FLAT)
        memset(self._prev_action, 0, Self.action_dim)

    # ══════════════════════════════════════════════════════════════════════
    # Data Collection
    # ══════════════════════════════════════════════════════════════════════

    fn observe(
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        done: Bool,
    ):
        """Store a transition in the replay buffer.

        Args:
            obs: Current observation [obs_dim].
            action: Action taken [action_dim].
            reward: Reward received.
            done: Whether the episode ended.
        """
        # Convert to InlineArray for buffer API
        var obs_arr = InlineArray[Scalar[DType.float32], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            if i < len(obs):
                obs_arr[i] = Scalar[DType.float32](obs[i])
            else:
                obs_arr[i] = Scalar[DType.float32](0.0)

        var act_arr = InlineArray[Scalar[DType.float32], Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.action_dim):
            if i < len(action):
                act_arr[i] = Scalar[DType.float32](action[i])
            else:
                act_arr[i] = Scalar[DType.float32](0.0)

        self.state.buffer.add(
            obs_arr, act_arr, Scalar[DType.float32](reward), done
        )

        if done:
            self.reset_episode()

    # ══════════════════════════════════════════════════════════════════════
    # Action Selection
    # ══════════════════════════════════════════════════════════════════════

    fn select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
    ) -> List[Scalar[dtype]]:
        """Select an action using the current RSSM state and actor network.

        Pipeline (BATCH=1):
        1. Symlog-preprocess observation, encode to embedding
        2. GRU core forward: (deter, stoch, prev_action) -> new_deter
        3. Posterior: concat(new_deter, embed) -> sample new stoch
        4. feat = concat(new_deter, new_stoch)
        5. Actor forward: feat -> (mean, log_std) -> sample tanh-normal

        Args:
            obs: Current observation [obs_dim].
            training: If True, sample stochastically; if False, use mode.

        Returns:
            Action as List[Scalar[dtype]], each element in (-1, 1).
        """
        comptime B: Int = 1

        # ── Prepare observation tensor ──────────────────────────────────
        var obs_ptr = alloc[Scalar[dtype]](B * Self.obs_dim)
        for i in range(Self.obs_dim):
            if i < len(obs):
                obs_ptr[i] = obs[i]
            else:
                obs_ptr[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.obs_dim), MutAnyOrigin
        ](obs_ptr)

        # ── Wrap running state as LayoutTensors ─────────────────────────
        var deter_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.deter_dim), MutAnyOrigin
        ](self._current_deter)
        var stoch_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.STOCH_FLAT), MutAnyOrigin
        ](self._current_stoch)
        var action_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.action_dim), MutAnyOrigin
        ](self._prev_action)

        # ── Allocate output buffers ─────────────────────────────────────
        var new_deter_ptr = alloc[Scalar[dtype]](B * Self.deter_dim)
        memset(new_deter_ptr, 0, B * Self.deter_dim)
        var new_stoch_ptr = alloc[Scalar[dtype]](B * Self.STOCH_FLAT)
        memset(new_stoch_ptr, 0, B * Self.STOCH_FLAT)
        var post_probs_ptr = alloc[Scalar[dtype]](B * Self.STOCH_FLAT)
        memset(post_probs_ptr, 0, B * Self.STOCH_FLAT)
        var prior_probs_ptr = alloc[Scalar[dtype]](B * Self.STOCH_FLAT)
        memset(prior_probs_ptr, 0, B * Self.STOCH_FLAT)
        var feat_ptr = alloc[Scalar[dtype]](B * Self.FEAT_DIM)
        memset(feat_ptr, 0, B * Self.FEAT_DIM)

        var new_deter_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.deter_dim), MutAnyOrigin
        ](new_deter_ptr)
        var new_stoch_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.STOCH_FLAT), MutAnyOrigin
        ](new_stoch_ptr)
        var post_probs_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.STOCH_FLAT), MutAnyOrigin
        ](post_probs_ptr)
        var prior_probs_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.STOCH_FLAT), MutAnyOrigin
        ](prior_probs_ptr)
        var feat_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.FEAT_DIM), MutAnyOrigin
        ](feat_ptr)

        # ── RSSM observe step ──────────────────────────────────────────
        self.state.rssm.observe_step[B](
            obs_t,
            deter_t,
            stoch_t,
            action_t,
            new_deter_t,
            new_stoch_t,
            post_probs_t,
            prior_probs_t,
            feat_t,
            training,
        )

        # ── Actor forward: feat -> (mean, log_std) ─────────────────────
        comptime ACTOR_OUT_DIM = Self.StateType.ActorModel.OUT_DIM
        var actor_out_ptr = alloc[Scalar[dtype]](B * ACTOR_OUT_DIM)
        memset(actor_out_ptr, 0, B * ACTOR_OUT_DIM)
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(B, ACTOR_OUT_DIM), MutAnyOrigin
        ](actor_out_ptr)

        Self.ActorNet.forward[B](
            feat_t, actor_out_t, self.state.actor.params_view()
        )

        # ── Sample action from tanh-normal ─────────────────────────────
        var result = List[Scalar[dtype]](capacity=Self.action_dim)
        for a in range(Self.action_dim):
            var mean_val = Float64(rebind[Scalar[dtype]](actor_out_t[0, a]))
            var log_std_val = Float64(
                rebind[Scalar[dtype]](actor_out_t[0, Self.action_dim + a])
            )
            # Clamp log_std to [-5, 2] for stability
            if log_std_val < -5.0:
                log_std_val = -5.0
            if log_std_val > 2.0:
                log_std_val = 2.0

            var action_val: Float64
            if training:
                # Box-Muller approximation: use uniform -> rough normal
                var u1 = random_float64(0.0001, 0.9999)
                var u2 = random_float64(0.0001, 0.9999)
                var z = sqrt(-2.0 * log(u1)) * (
                    exp(Float64(0.0)) * (2.0 * u2 - 1.0)
                )
                # Approximate standard normal via CLT
                action_val = sample_tanh_normal(mean_val, log_std_val, z)
            else:
                # Mode: tanh(mean) when not training
                var ep = exp(mean_val)
                var en = exp(-mean_val)
                action_val = (ep - en) / (ep + en)

            # Clamp to [-1, 1]
            if action_val > 1.0:
                action_val = 1.0
            if action_val < -1.0:
                action_val = -1.0
            result.append(Scalar[dtype](action_val))

        # ── Update running state ───────────────────────────────────────
        for i in range(Self.deter_dim):
            (self._current_deter + i)[] = (new_deter_ptr + i)[]
        for i in range(Self.STOCH_FLAT):
            (self._current_stoch + i)[] = (new_stoch_ptr + i)[]
        for a in range(Self.action_dim):
            (self._prev_action + a)[] = result[a]

        # ── Free temporary buffers ─────────────────────────────────────
        obs_ptr.free()
        new_deter_ptr.free()
        new_stoch_ptr.free()
        post_probs_ptr.free()
        prior_probs_ptr.free()
        feat_ptr.free()
        actor_out_ptr.free()

        return result^

    # ══════════════════════════════════════════════════════════════════════
    # Training Step
    # ══════════════════════════════════════════════════════════════════════

    fn update(mut self) -> Float64:
        """Full DreamerV3 training step.

        1. Sample sequences from replay buffer
        2. RSSM observe loop (posterior, fills _all_* buffers)
        3. Full BPTT world model backward (autodiff):
           - Prediction heads fan-out via ComputeGraph (decoder + reward + continue)
           - Straight-through categorical → posterior + encoder backward
           - Dual KL balancing → prior + posterior backward
           - GRU core backward with gradient carry across timesteps
        4. World model optimizer step (all 11 sub-networks)
        5. Imagination rollout from observed states
        6. Lambda returns + normalization
        7. Actor loss (reinforce + entropy) + backward + step
        8. Critic loss (two-hot cross-entropy) + backward + step
        9. Slow critic EMA update

        Returns:
            Total training loss (sum of world model, actor, critic losses).
        """
        comptime B = Self.batch_size
        comptime BL = Self.batch_length
        comptime DETER = Self.deter_dim
        comptime STOCH = Self.STOCH_FLAT
        comptime FEAT = Self.FEAT_DIM
        comptime OBS = Self.obs_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime HORIZON = Self.imagine_horizon
        comptime IB = Self.IMAG_BATCH

        # ── 1. Sample sequences from replay buffer ──────────────────────
        var batch_obs = List[Scalar[DType.float32]](capacity=B * (BL + 1) * OBS)
        var batch_actions = List[Scalar[DType.float32]](capacity=B * BL * ACT)
        var batch_rewards = List[Scalar[DType.float32]](capacity=B * BL)
        var batch_dones = List[Scalar[DType.float32]](capacity=B * BL)

        # Pre-fill lists to required size
        for _ in range(B * (BL + 1) * OBS):
            batch_obs.append(Scalar[DType.float32](0))
        for _ in range(B * BL * ACT):
            batch_actions.append(Scalar[DType.float32](0))
        for _ in range(B * BL):
            batch_rewards.append(Scalar[DType.float32](0))
            batch_dones.append(Scalar[DType.float32](0))

        self.state.buffer.sample_sequences[B, BL](
            batch_obs, batch_actions, batch_rewards, batch_dones
        )

        # ── 2. RSSM Observe Loop ────────────────────────────────────────
        # Initialize deter/stoch to zeros for each batch element
        var deter_ptr = alloc[Scalar[dtype]](B * DETER)
        memset(deter_ptr, 0, B * DETER)
        var stoch_ptr = alloc[Scalar[dtype]](B * STOCH)
        memset(stoch_ptr, 0, B * STOCH)

        # Scratch for observe step outputs
        var new_deter_ptr = alloc[Scalar[dtype]](B * DETER)
        var new_stoch_ptr = alloc[Scalar[dtype]](B * STOCH)
        var post_probs_ptr = alloc[Scalar[dtype]](B * STOCH)
        var prior_probs_ptr = alloc[Scalar[dtype]](B * STOCH)
        var feat_ptr = alloc[Scalar[dtype]](B * FEAT)
        var obs_step_ptr = alloc[Scalar[dtype]](B * OBS)
        var act_step_ptr = alloc[Scalar[dtype]](B * ACT)

        for t in range(BL):
            # ── Extract obs[t] and action[t] for all batch elements ─────
            for b in range(B):
                for i in range(OBS):
                    var idx = b * (BL + 1) * OBS + t * OBS + i
                    (obs_step_ptr + b * OBS + i)[] = Scalar[dtype](
                        batch_obs[idx]
                    )
                for i in range(ACT):
                    if t == 0:
                        (act_step_ptr + b * ACT + i)[] = Scalar[dtype](0.0)
                    else:
                        var idx = b * BL * ACT + (t - 1) * ACT + i
                        (act_step_ptr + b * ACT + i)[] = Scalar[dtype](
                            batch_actions[idx]
                        )

            # Create LayoutTensor views
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](obs_step_ptr)
            var deter_t = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](deter_ptr)
            var stoch_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](stoch_ptr)
            var act_t = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](act_step_ptr)

            memset(new_deter_ptr, 0, B * DETER)
            memset(new_stoch_ptr, 0, B * STOCH)
            memset(post_probs_ptr, 0, B * STOCH)
            memset(prior_probs_ptr, 0, B * STOCH)
            memset(feat_ptr, 0, B * FEAT)

            var new_deter_t = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](new_deter_ptr)
            var new_stoch_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](new_stoch_ptr)
            var post_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](post_probs_ptr)
            var prior_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](prior_probs_ptr)
            var feat_t = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](feat_ptr)

            # ── RSSM observe step ──────────────────────────────────────
            self.state.rssm.observe_step[B](
                obs_t,
                deter_t,
                stoch_t,
                act_t,
                new_deter_t,
                new_stoch_t,
                post_probs_t,
                prior_probs_t,
                feat_t,
                True,
            )

            # Store in all_* buffers for backward pass
            for b in range(B):
                for i in range(DETER):
                    (
                        self.state._all_deter + t * B * DETER + b * DETER + i
                    )[] = (new_deter_ptr + b * DETER + i)[]
                for i in range(STOCH):
                    (
                        self.state._all_stoch + t * B * STOCH + b * STOCH + i
                    )[] = (new_stoch_ptr + b * STOCH + i)[]
                    (
                        self.state._all_post_probs
                        + t * B * STOCH
                        + b * STOCH
                        + i
                    )[] = (post_probs_ptr + b * STOCH + i)[]
                    (
                        self.state._all_prior_probs
                        + t * B * STOCH
                        + b * STOCH
                        + i
                    )[] = (prior_probs_ptr + b * STOCH + i)[]
                for i in range(FEAT):
                    (self.state._all_feats + t * B * FEAT + b * FEAT + i)[] = (
                        feat_ptr + b * FEAT + i
                    )[]

            # Update deter/stoch for next timestep
            for b in range(B):
                for i in range(DETER):
                    (deter_ptr + b * DETER + i)[] = (
                        new_deter_ptr + b * DETER + i
                    )[]
                for i in range(STOCH):
                    (stoch_ptr + b * STOCH + i)[] = (
                        new_stoch_ptr + b * STOCH + i
                    )[]

        # ── 3. Full BPTT backward (autodiff) ─────────────────────────────
        # Computes losses + gradients for ALL 11 RSSM sub-networks via
        # ComputeGraph fan-out + straight-through + GRU backward
        var wm_result = self._backward_world_model_autodiff[B](
            batch_obs, batch_actions, batch_rewards, batch_dones
        )
        var total_wm_loss = wm_result[0]
        var obs_loss = wm_result[1]
        var rew_loss = wm_result[2]
        var cont_loss = wm_result[3]
        var dyn_kl_total = wm_result[4]
        var rep_kl_total = wm_result[5]

        # ── 4. World model optimizer step ──────────────────────────────
        self.state.rssm.update_all_params()

        # ── 5. Imagination rollout ─────────────────────────────────────
        # Initialize imagination from all observed (deter, stoch) pairs
        # Flatten BL*B states into IMAG_BATCH
        for t in range(BL):
            for b in range(B):
                var ib_idx = t * B + b
                for i in range(DETER):
                    (self.state._imag_deter + ib_idx * DETER + i)[] = (
                        self.state._all_deter + t * B * DETER + b * DETER + i
                    )[]
                for i in range(STOCH):
                    (self.state._imag_stoch + ib_idx * STOCH + i)[] = (
                        self.state._all_stoch + t * B * STOCH + b * STOCH + i
                    )[]
                for i in range(FEAT):
                    (self.state._imag_feat + ib_idx * FEAT + i)[] = (
                        self.state._all_feats + t * B * FEAT + b * FEAT + i
                    )[]

        # Zero actor/critic grads
        self.state.actor.zero_grads()
        self.state.critic.zero_grads()

        var actor_loss = Float64(0.0)
        var critic_loss = Float64(0.0)

        # Imagination rollout: HORIZON steps
        for h in range(HORIZON):
            # ── Actor: select actions from imagined features ───────────
            var imag_feat_h = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](self.state._imag_feat + h * IB * FEAT)

            var actor_out_h = LayoutTensor[
                dtype,
                Layout.row_major(IB, Self.StateType.ActorModel.OUT_DIM),
                MutAnyOrigin,
            ](self.state._actor_out)

            Self.ActorNet.forward[IB](
                imag_feat_h, actor_out_h, self.state.actor.params_view()
            )

            # Sample actions and compute log probs
            for ib in range(IB):
                for a in range(ACT):
                    var mean_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, a])
                    )
                    var log_std_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, ACT + a])
                    )
                    if log_std_val < -5.0:
                        log_std_val = -5.0
                    if log_std_val > 2.0:
                        log_std_val = 2.0

                    var u1 = random_float64(0.0001, 0.9999)
                    var u2 = random_float64(0.0001, 0.9999)
                    var z = sqrt(-2.0 * log(u1)) * (2.0 * u2 - 1.0)
                    var action_val = sample_tanh_normal(
                        mean_val, log_std_val, z
                    )
                    if action_val > 1.0:
                        action_val = 1.0
                    if action_val < -1.0:
                        action_val = -1.0
                    (
                        self.state._imag_actions + h * IB * ACT + ib * ACT + a
                    )[] = Scalar[dtype](action_val)

                # Accumulate log prob over action dimensions
                var total_lp = Float64(0.0)
                for a in range(ACT):
                    var action_val = Float64(
                        (
                            self.state._imag_actions
                            + h * IB * ACT
                            + ib * ACT
                            + a
                        )[]
                    )
                    var mean_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, a])
                    )
                    var log_std_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, ACT + a])
                    )
                    if log_std_val < -5.0:
                        log_std_val = -5.0
                    if log_std_val > 2.0:
                        log_std_val = 2.0
                    total_lp += log_prob_tanh_normal(
                        action_val, mean_val, log_std_val
                    )
                (self.state._imag_log_probs + h * IB + ib)[] = Scalar[dtype](
                    total_lp
                )

            # ── RSSM imagine step ─────────────────────────────────────
            if h < HORIZON - 1:
                var curr_deter_h = LayoutTensor[
                    dtype, Layout.row_major(IB, DETER), MutAnyOrigin
                ](self.state._imag_deter + h * IB * DETER)
                var curr_stoch_h = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](self.state._imag_stoch + h * IB * STOCH)
                var actions_h = LayoutTensor[
                    dtype, Layout.row_major(IB, ACT), MutAnyOrigin
                ](self.state._imag_actions + h * IB * ACT)
                var next_deter_h = LayoutTensor[
                    dtype, Layout.row_major(IB, DETER), MutAnyOrigin
                ](self.state._imag_deter + (h + 1) * IB * DETER)
                var next_stoch_h = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](self.state._imag_stoch + (h + 1) * IB * STOCH)
                var next_feat_h = LayoutTensor[
                    dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
                ](self.state._imag_feat + (h + 1) * IB * FEAT)

                self.state.rssm.imagine_step[IB](
                    curr_deter_h,
                    curr_stoch_h,
                    actions_h,
                    next_deter_h,
                    next_stoch_h,
                    next_feat_h,
                    True,
                )

            # ── Predict reward and continue from imagined features ────
            var rew_logits_ptr = alloc[Scalar[dtype]](IB * BINS)
            memset(rew_logits_ptr, 0, IB * BINS)
            var rew_logits_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](rew_logits_ptr)
            self.state.rssm.predict_reward[IB](imag_feat_h, rew_logits_h)

            # Decode reward from distributional logits
            for ib in range(IB):
                var logits_arr = InlineArray[Float32, Self.num_bins](
                    uninitialized=True
                )
                for k in range(BINS):
                    logits_arr[k] = Float32(
                        rebind[Scalar[dtype]](rew_logits_h[ib, k])
                    )
                var reward_symlog = decode_value[Self.num_bins](
                    logits_arr, self.state.rssm.bins
                )
                # Decode from symlog space to actual value
                var reward_val = symexp(reward_symlog)
                (self.state._imag_rewards + h * IB + ib)[] = Scalar[dtype](
                    reward_val
                )

            rew_logits_ptr.free()

            # Predict continuation
            var cont_out_ptr = alloc[Scalar[dtype]](IB * 1)
            memset(cont_out_ptr, 0, IB * 1)
            var cont_out_h = LayoutTensor[
                dtype, Layout.row_major(IB, 1), MutAnyOrigin
            ](cont_out_ptr)
            self.state.rssm.predict_continue[IB](imag_feat_h, cont_out_h)

            for ib in range(IB):
                (self.state._imag_continues + h * IB + ib)[] = rebind[
                    Scalar[dtype]
                ](cont_out_h[ib, 0])

            cont_out_ptr.free()

            # ── Critic value prediction ───────────────────────────────
            var critic_logits_ptr = alloc[Scalar[dtype]](IB * BINS)
            memset(critic_logits_ptr, 0, IB * BINS)
            var critic_logits_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](critic_logits_ptr)
            Self.CriticNet.forward[IB](
                imag_feat_h,
                critic_logits_h,
                self.state.critic.params_view(),
            )

            for ib in range(IB):
                var logits_arr = InlineArray[Float32, Self.num_bins](
                    uninitialized=True
                )
                for k in range(BINS):
                    logits_arr[k] = Float32(
                        rebind[Scalar[dtype]](critic_logits_h[ib, k])
                    )
                var value_symlog = decode_value[Self.num_bins](
                    logits_arr, self.state.rssm.bins
                )
                # Decode from symlog space to actual value
                var value_val = symexp(value_symlog)
                (self.state._imag_values + h * IB + ib)[] = Scalar[dtype](
                    value_val
                )

            critic_logits_ptr.free()

        # ── 6. Lambda returns + normalization ──────────────────────────
        compute_lambda_returns[HORIZON, IB](
            self.state._imag_rewards,
            self.state._imag_values,
            self.state._imag_continues,
            self.state._imag_returns,
            self.gamma,
            self.lambda_,
        )

        var scale = normalize_returns[HORIZON, IB](
            self.state._imag_returns,
            self.state.return_ema_lo,
            self.state.return_ema_hi,
            self.return_norm_rate,
        )

        # ── 7. Actor loss: REINFORCE + entropy ─────────────────────────
        # Actor loss = -E[sg(returns - values) * log_prob + entropy_coef * entropy]
        for h in range(HORIZON - 1):
            for ib in range(IB):
                var ret = Float64((self.state._imag_returns + h * IB + ib)[])
                var val = Float64((self.state._imag_values + h * IB + ib)[])
                var advantage = ret - val
                var log_prob = Float64(
                    (self.state._imag_log_probs + h * IB + ib)[]
                )
                # Reinforce-style: maximize advantage * log_prob
                actor_loss -= advantage * log_prob
                # Entropy bonus (approximate: -log_prob as proxy)
                actor_loss -= self.actor_entropy * (-log_prob)

        actor_loss /= Float64((HORIZON - 1) * IB)

        # ── 8. Critic loss: two-hot cross-entropy ──────────────────────
        # Train critic to predict symlog(return) using two-hot encoding
        for h in range(HORIZON - 1):
            var imag_feat_h = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](self.state._imag_feat + h * IB * FEAT)

            var critic_logits_ptr = alloc[Scalar[dtype]](IB * BINS)
            memset(critic_logits_ptr, 0, IB * BINS)
            var critic_logits_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](critic_logits_ptr)

            # Forward with cache for backward
            comptime CACHE_SIZE = Self.StateType.CriticModel.CACHE_SIZE
            var cache_ptr = alloc[Scalar[dtype]](IB * CACHE_SIZE)
            memset(cache_ptr, 0, IB * CACHE_SIZE)
            var cache_t = LayoutTensor[
                dtype, Layout.row_major(IB, CACHE_SIZE), MutAnyOrigin
            ](cache_ptr)

            Self.CriticNet.forward_with_cache[IB](
                imag_feat_h,
                critic_logits_h,
                self.state.critic.params_view(),
                cache_t,
            )

            # Compute two-hot cross-entropy gradient
            var grad_out_ptr = alloc[Scalar[dtype]](IB * BINS)
            memset(grad_out_ptr, 0, IB * BINS)

            for ib in range(IB):
                # Target: symlog(return)
                var ret_raw = Float64(
                    (self.state._imag_returns + h * IB + ib)[]
                )
                # Undo normalization for the target
                var ret_actual = ret_raw * scale + self.state.return_ema_lo
                var ret_symlog = Float32(symlog(Float32(ret_actual)))

                var target_dist = InlineArray[Float32, Self.num_bins](
                    uninitialized=True
                )
                two_hot_encode[Self.num_bins](
                    ret_symlog, self.state.rssm.bins, target_dist
                )

                # Softmax of logits
                var max_logit = Float64(
                    rebind[Scalar[dtype]](critic_logits_h[ib, 0])
                )
                for k in range(1, BINS):
                    var v = Float64(
                        rebind[Scalar[dtype]](critic_logits_h[ib, k])
                    )
                    if v > max_logit:
                        max_logit = v
                var sum_exp = Float64(0.0)
                for k in range(BINS):
                    sum_exp += exp(
                        Float64(rebind[Scalar[dtype]](critic_logits_h[ib, k]))
                        - max_logit
                    )

                # Gradient of cross-entropy w.r.t. logits: softmax(logit) - target
                for k in range(BINS):
                    var softmax_k = (
                        exp(
                            Float64(
                                rebind[Scalar[dtype]](critic_logits_h[ib, k])
                            )
                            - max_logit
                        )
                        / sum_exp
                    )
                    var target_k = Float64(target_dist[k])
                    (grad_out_ptr + ib * BINS + k)[] = Scalar[dtype](
                        (softmax_k - target_k) / Float64(IB)
                    )

                # Accumulate critic loss for reporting
                for k in range(BINS):
                    var t_k = Float64(target_dist[k])
                    if t_k > 1e-8:
                        var logit_k = Float64(
                            rebind[Scalar[dtype]](critic_logits_h[ib, k])
                        )
                        var log_softmax_k = logit_k - log(sum_exp) - max_logit
                        critic_loss -= t_k * log_softmax_k

            var grad_out_t = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](grad_out_ptr)
            var grad_in_ptr = alloc[Scalar[dtype]](IB * FEAT)
            memset(grad_in_ptr, 0, IB * FEAT)
            var grad_in_t = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](grad_in_ptr)

            # Backward through critic
            var critic_grads = self.state.critic.grads_view()
            Self.CriticNet.backward[IB](
                grad_out_t,
                grad_in_t,
                self.state.critic.params_view(),
                cache_t,
                critic_grads,
            )

            critic_logits_ptr.free()
            cache_ptr.free()
            grad_out_ptr.free()
            grad_in_ptr.free()

        critic_loss /= Float64((HORIZON - 1) * IB)

        # ── Critic optimizer step ──────────────────────────────────────
        self.state.critic.optimizer_step()

        # ── Actor backward (simplified: per-horizon-step gradient) ─────
        # For each imagination step, backprop the REINFORCE gradient
        # through the actor network
        for h in range(HORIZON - 1):
            var imag_feat_h = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](self.state._imag_feat + h * IB * FEAT)

            comptime ACTOR_OUT = Self.StateType.ActorModel.OUT_DIM
            var actor_out_ptr = alloc[Scalar[dtype]](IB * ACTOR_OUT)
            memset(actor_out_ptr, 0, IB * ACTOR_OUT)
            var actor_out_h = LayoutTensor[
                dtype, Layout.row_major(IB, ACTOR_OUT), MutAnyOrigin
            ](actor_out_ptr)

            comptime ACTOR_CACHE = Self.StateType.ActorModel.CACHE_SIZE
            var actor_cache_ptr = alloc[Scalar[dtype]](IB * ACTOR_CACHE)
            memset(actor_cache_ptr, 0, IB * ACTOR_CACHE)
            var actor_cache_t = LayoutTensor[
                dtype, Layout.row_major(IB, ACTOR_CACHE), MutAnyOrigin
            ](actor_cache_ptr)

            Self.ActorNet.forward_with_cache[IB](
                imag_feat_h,
                actor_out_h,
                self.state.actor.params_view(),
                actor_cache_t,
            )

            # Compute actor gradient: d(-advantage * log_prob) / d(actor_output)
            # This is a simplified gradient — we approximate the gradient of
            # log_prob w.r.t. actor outputs (mean, log_std)
            var actor_grad_ptr = alloc[Scalar[dtype]](IB * ACTOR_OUT)
            memset(actor_grad_ptr, 0, IB * ACTOR_OUT)

            for ib in range(IB):
                var ret = Float64((self.state._imag_returns + h * IB + ib)[])
                var val = Float64((self.state._imag_values + h * IB + ib)[])
                var advantage = ret - val
                var inv_ib = 1.0 / Float64(IB * (HORIZON - 1))

                for a in range(ACT):
                    var action_val = Float64(
                        (
                            self.state._imag_actions
                            + h * IB * ACT
                            + ib * ACT
                            + a
                        )[]
                    )
                    var mean_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, a])
                    )
                    var log_std_val = Float64(
                        rebind[Scalar[dtype]](actor_out_h[ib, ACT + a])
                    )
                    if log_std_val < -5.0:
                        log_std_val = -5.0
                    if log_std_val > 2.0:
                        log_std_val = 2.0

                    var std_val = exp(log_std_val)
                    if std_val < 1e-6:
                        std_val = 1e-6

                    # Clamp action for atanh
                    var a_clamped = action_val
                    if a_clamped > 0.999999:
                        a_clamped = 0.999999
                    if a_clamped < -0.999999:
                        a_clamped = -0.999999
                    var pre_tanh = 0.5 * log(
                        (1.0 + a_clamped) / (1.0 - a_clamped)
                    )
                    var z = (pre_tanh - mean_val) / std_val

                    # d(log_prob)/d(mean) = z / std
                    var grad_mean = z / std_val
                    # d(log_prob)/d(log_std) = z^2 - 1
                    var grad_log_std = z * z - 1.0

                    # We want to maximize advantage * log_prob, so gradient
                    # is advantage * d(log_prob)/d(params). For the output
                    # gradient going into backward, we negate (minimizing loss).
                    var weight = -advantage * inv_ib
                    (actor_grad_ptr + ib * ACTOR_OUT + a)[] = Scalar[dtype](
                        weight * grad_mean
                    )
                    (actor_grad_ptr + ib * ACTOR_OUT + ACT + a)[] = Scalar[
                        dtype
                    ](weight * grad_log_std)

            var actor_grad_t = LayoutTensor[
                dtype, Layout.row_major(IB, ACTOR_OUT), MutAnyOrigin
            ](actor_grad_ptr)
            var actor_grad_in_ptr = alloc[Scalar[dtype]](IB * FEAT)
            memset(actor_grad_in_ptr, 0, IB * FEAT)
            var actor_grad_in_t = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](actor_grad_in_ptr)

            var actor_grads = self.state.actor.grads_view()
            Self.ActorNet.backward[IB](
                actor_grad_t,
                actor_grad_in_t,
                self.state.actor.params_view(),
                actor_cache_t,
                actor_grads,
            )

            actor_out_ptr.free()
            actor_cache_ptr.free()
            actor_grad_ptr.free()
            actor_grad_in_ptr.free()

        # ── Actor optimizer step ───────────────────────────────────────
        self.state.actor.optimizer_step()

        # ── 9. Slow critic EMA update ──────────────────────────────────
        self.state.slow_critic_update(self.slow_critic_tau)

        # ── Cleanup ────────────────────────────────────────────────────
        deter_ptr.free()
        stoch_ptr.free()
        new_deter_ptr.free()
        new_stoch_ptr.free()
        post_probs_ptr.free()
        prior_probs_ptr.free()
        feat_ptr.free()
        obs_step_ptr.free()
        act_step_ptr.free()

        self.train_step_count += 1

        # Log DreamerV3 diagnostics
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                # World model losses
                self.logger[].log_scalar(
                    "loss",
                    total_wm_loss + actor_loss + critic_loss,
                    step,
                )
                self.logger[].log_scalar("obs_loss", obs_loss, step)
                self.logger[].log_scalar("reward_loss", rew_loss, step)
                self.logger[].log_scalar("continue_loss", cont_loss, step)
                self.logger[].log_scalar("dyn_kl", dyn_kl_total, step)
                self.logger[].log_scalar("rep_kl", rep_kl_total, step)
                # Actor-critic
                self.logger[].log_scalar("policy_loss", actor_loss, step)
                self.logger[].log_scalar("value_loss", critic_loss, step)
                # Return normalization
                self.logger[].log_scalar(
                    "return_scale",
                    Float64(self.state.return_ema_hi)
                    - Float64(self.state.return_ema_lo),
                    step,
                )
                # Mean imagined reward
                var imag_rew_sum: Float64 = 0.0
                for i in range(HORIZON * IB):
                    imag_rew_sum += Float64((self.state._imag_rewards + i)[])
                self.logger[].log_scalar(
                    "imagined_reward_mean",
                    imag_rew_sum / Float64(HORIZON * IB),
                    step,
                )
                # Entropy (mean negative log_prob across imagination)
                var entropy_sum: Float64 = 0.0
                for i in range((HORIZON - 1) * IB):
                    entropy_sum -= Float64((self.state._imag_log_probs + i)[])
                self.logger[].log_scalar(
                    "entropy",
                    entropy_sum / Float64((HORIZON - 1) * IB),
                    step,
                )
            except:
                pass

        return total_wm_loss + actor_loss + critic_loss

    # ══════════════════════════════════════════════════════════════════════
    # World Model Backward (Full BPTT via Autodiff)
    # ══════════════════════════════════════════════════════════════════════

    fn _backward_world_model_autodiff[
        B: Int
    ](
        mut self,
        batch_obs: List[Scalar[DType.float32]],
        batch_actions: List[Scalar[DType.float32]],
        batch_rewards: List[Scalar[DType.float32]],
        batch_dones: List[Scalar[DType.float32]],
    ) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
        """Full BPTT backward for world model using autodiff.

        Replaces the forward-only loss computation + partial decoder backward
        with proper gradient flow through ALL 11 RSSM networks:
        - Prediction heads (decoder, reward, continue) via ComputeGraph fan-out
        - Encoder + posterior via straight-through categorical
        - Prior via KL loss
        - GRU core (deter_proj, stoch_proj, action_proj, gru_hidden, gru_gates)

        Uses the cached _all_* buffers from the forward observe loop.

        Returns the total world model loss.
        """
        comptime BL = Self.batch_length
        comptime DETER = Self.deter_dim
        comptime STOCH = Self.STOCH_FLAT
        comptime FEAT = Self.FEAT_DIM
        comptime OBS = Self.obs_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime HEADS_OUT = Self.StateType.RSSMType.HEADS_OUT_DIM
        comptime HEADS_CS = Self.StateType.RSSMType.HEADS_CACHE_SIZE

        var obs_loss = Float64(0.0)
        var rew_loss = Float64(0.0)
        var cont_loss = Float64(0.0)
        var dyn_kl_total = Float64(0.0)
        var rep_kl_total = Float64(0.0)

        # Zero all world model gradients
        self.state.rssm.zero_all_grads()

        # Gradient carry for BPTT
        var grad_deter_carry = alloc[Scalar[dtype]](B * DETER)
        memset(grad_deter_carry, 0, B * DETER)

        # Process timesteps in REVERSE for BPTT
        for _ri in range(BL):
            var t = BL - 1 - _ri

            # ── Load cached states for timestep t ────────────────────────
            var feat_t = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](self.state._all_feats + t * B * FEAT)
            var deter_t = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](self.state._all_deter + t * B * DETER)
            var post_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](self.state._all_post_probs + t * B * STOCH)
            var prior_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](self.state._all_prior_probs + t * B * STOCH)

            # ── Load observation for this timestep ───────────────────────
            var obs_buf = alloc[Scalar[dtype]](B * OBS)
            for b in range(B):
                for i in range(OBS):
                    var idx = b * (BL + 1) * OBS + t * OBS + i
                    (obs_buf + b * OBS + i)[] = Scalar[dtype](batch_obs[idx])
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](obs_buf)

            # ── 1. Forward prediction heads ──────────────────────────────
            var heads_out = alloc[Scalar[dtype]](B * HEADS_OUT)
            memset(heads_out, 0, B * HEADS_OUT)
            var heads_t = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_OUT), MutAnyOrigin
            ](heads_out)
            var heads_cache = alloc[Scalar[dtype]](B * HEADS_CS)
            memset(heads_cache, 0, B * HEADS_CS)
            var heads_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_CS), MutAnyOrigin
            ](heads_cache)

            self.state.rssm.predict_all_heads[B](feat_t, heads_t, heads_cache_t)

            # ── 2. Compute loss gradients ────────────────────────────────
            # Output layout: [obs_hat(OBS), rew_logits(BINS), cont_logit(1)]
            var grad_heads = alloc[Scalar[dtype]](B * HEADS_OUT)
            memset(grad_heads, 0, B * HEADS_OUT)

            # Decoder loss: MSE(obs_hat, symlog(obs_{t+1}))
            var scale_obs = 2.0 / Float64(B * OBS)
            for b in range(B):
                for i in range(OBS):
                    var obs_next_idx = b * (BL + 1) * OBS + (t + 1) * OBS + i
                    var target = Float64(
                        symlog(Float32(batch_obs[obs_next_idx]))
                    )
                    var pred = Float64((heads_out + b * HEADS_OUT + i)[])
                    var diff = pred - target
                    obs_loss += diff * diff
                    (grad_heads + b * HEADS_OUT + i)[] = Scalar[dtype](
                        diff * scale_obs
                    )

            # Reward loss: two-hot CE (t > 0 only)
            if t > 0:
                for b in range(B):
                    var rew_val = Float32(batch_rewards[b * BL + t])
                    var rew_symlog = symlog(rew_val)
                    var target_dist = InlineArray[Float32, Self.num_bins](
                        uninitialized=True
                    )
                    two_hot_encode[Self.num_bins](
                        rew_symlog, self.state.rssm.bins, target_dist
                    )

                    # Softmax
                    var max_logit = Float64((heads_out + b * HEADS_OUT + OBS)[])
                    for k in range(1, BINS):
                        var v = Float64((heads_out + b * HEADS_OUT + OBS + k)[])
                        if v > max_logit:
                            max_logit = v
                    var sum_exp = Float64(0.0)
                    var softmax_vals = InlineArray[Float64, Self.num_bins](
                        uninitialized=True
                    )
                    for k in range(BINS):
                        var e = exp(
                            Float64((heads_out + b * HEADS_OUT + OBS + k)[])
                            - max_logit
                        )
                        softmax_vals[k] = e
                        sum_exp += e
                    for k in range(BINS):
                        softmax_vals[k] /= sum_exp

                    # Loss + gradient
                    var log_sum_exp = log(sum_exp) + max_logit
                    for k in range(BINS):
                        var t_k = Float64(target_dist[k])
                        if t_k > 1e-8:
                            var logit_k = Float64(
                                (heads_out + b * HEADS_OUT + OBS + k)[]
                            )
                            rew_loss -= t_k * (logit_k - log_sum_exp)
                        # CE gradient: softmax - target
                        var grad_k = (
                            softmax_vals[k] - Float64(target_dist[k])
                        ) / Float64(B)
                        (grad_heads + b * HEADS_OUT + OBS + k)[] = Scalar[
                            dtype
                        ](grad_k)

            # Continue loss: BCE (t > 0 only)
            # Note: HeadsGraph outputs raw logit (no sigmoid).
            # BCE with logit: grad = sigmoid(logit) - target
            if t > 0:
                for b in range(B):
                    var cont_target = 1.0 - Float64(batch_dones[b * BL + t])
                    var logit = Float64(
                        (heads_out + b * HEADS_OUT + OBS + BINS)[]
                    )
                    var one = 1.0
                    var prob = one / (one + exp(-logit))
                    # Clamp
                    if prob < 1e-6:
                        prob = 1e-6
                    if prob > 1.0 - 1e-6:
                        prob = 1.0 - 1e-6
                    cont_loss -= cont_target * log(prob) + (
                        one - cont_target
                    ) * log(one - prob)
                    # BCE-with-logit gradient: sigmoid(logit) - target
                    var grad_cont = (prob - cont_target) / Float64(B)
                    (grad_heads + b * HEADS_OUT + OBS + BINS)[] = Scalar[dtype](
                        grad_cont
                    )

            # ── 3. Backward prediction heads → grad_feat ─────────────────
            var grad_heads_t = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_OUT), MutAnyOrigin
            ](grad_heads)
            var grad_feat = alloc[Scalar[dtype]](B * FEAT)
            memset(grad_feat, 0, B * FEAT)
            var grad_feat_t = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](grad_feat)

            self.state.rssm.backward_all_heads[B](
                grad_heads_t, grad_feat_t, heads_cache_t
            )

            # ── 4. Backward feat → encoder ───────────────────────────────
            var grad_deter = alloc[Scalar[dtype]](B * DETER)
            memset(grad_deter, 0, B * DETER)
            var grad_deter_t = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](grad_deter)

            self.state.rssm.backward_feat_to_encoder[B](
                grad_feat_t, obs_t, deter_t, post_probs_t, grad_deter_t
            )

            # ── 5. KL loss backward ──────────────────────────────────────
            var kl_val = kl_divergence[B, Self.stoch_dim, Self.classes](
                post_probs_t, prior_probs_t
            )
            dyn_kl_total += kl_val
            rep_kl_total += kl_val

            self.state.rssm.backward_kl_loss[B](
                obs_t,
                deter_t,
                post_probs_t,
                prior_probs_t,
                0.5,
                0.1,
                grad_deter_t,
            )

            # ── 6. Add BPTT carry from future timesteps ──────────────────
            for i in range(B * DETER):
                (grad_deter + i)[] = (grad_deter + i)[] + grad_deter_carry[i]

            # ── 7. GRU backward → grad_prev_deter, grad_prev_stoch ──────
            if t > 0:
                # Load prev_deter, prev_stoch, prev_action for timestep t
                var prev_deter_t = LayoutTensor[
                    dtype, Layout.row_major(B, DETER), MutAnyOrigin
                ](self.state._all_deter + (t - 1) * B * DETER)
                var prev_stoch_t = LayoutTensor[
                    dtype, Layout.row_major(B, STOCH), MutAnyOrigin
                ](self.state._all_stoch + (t - 1) * B * STOCH)

                # Load action at t-1
                var act_buf = alloc[Scalar[dtype]](B * ACT)
                for b in range(B):
                    for i in range(ACT):
                        if t == 1:
                            (act_buf + b * ACT + i)[] = Scalar[dtype](0.0)
                        else:
                            var idx = b * BL * ACT + (t - 2) * ACT + i
                            (act_buf + b * ACT + i)[] = Scalar[dtype](
                                batch_actions[idx]
                            )
                var act_t = LayoutTensor[
                    dtype, Layout.row_major(B, ACT), MutAnyOrigin
                ](act_buf)

                var grad_prev_stoch = alloc[Scalar[dtype]](B * STOCH)
                memset(grad_prev_stoch, 0, B * STOCH)
                var grad_prev_stoch_t = LayoutTensor[
                    dtype, Layout.row_major(B, STOCH), MutAnyOrigin
                ](grad_prev_stoch)

                # Reset carry for next iteration
                memset(grad_deter_carry, 0, B * DETER)
                var grad_carry_t = LayoutTensor[
                    dtype, Layout.row_major(B, DETER), MutAnyOrigin
                ](grad_deter_carry)

                self.state.rssm.backward_gru_core[B](
                    grad_deter_t,
                    prev_deter_t,
                    prev_stoch_t,
                    act_t,
                    grad_carry_t,
                    grad_prev_stoch_t,
                )

                # Note: grad_prev_stoch could be used to add additional
                # gradient to the stoch at t-1 (through the straight-through
                # at the previous timestep). For now we include it in the
                # deter carry as it flows through feat = concat(deter, stoch).
                # A full implementation would add it to grad_feat at t-1.

                act_buf.free()
                grad_prev_stoch.free()
            else:
                # t=0: no previous timestep, just zero the carry
                memset(grad_deter_carry, 0, B * DETER)

            # Free per-timestep buffers
            obs_buf.free()
            heads_out.free()
            heads_cache.free()
            grad_heads.free()
            grad_feat.free()
            grad_deter.free()

        grad_deter_carry.free()

        # Normalize losses
        var inv_bl = 1.0 / Float64(BL)
        var inv_bl_b = 1.0 / Float64(BL * B)
        obs_loss *= inv_bl_b
        rew_loss *= inv_bl_b
        cont_loss *= inv_bl_b
        dyn_kl_total *= inv_bl
        rep_kl_total *= inv_bl

        var total = (
            obs_loss
            + rew_loss
            + cont_loss
            + (0.5 * dyn_kl_total + 0.1 * rep_kl_total)
        )
        return (
            total,
            obs_loss,
            rew_loss,
            cont_loss,
            dyn_kl_total,
            rep_kl_total,
        )

    # ══════════════════════════════════════════════════════════════════════
    # GPU BPTT Backward via Autodiff (replaces per-timestep head backward
    # + separate BPTT backward loop with a single unified reverse pass)
    # ══════════════════════════════════════════════════════════════════════

    fn _gpu_bptt_autodiff(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        batch_obs: List[Scalar[DType.float32]],
        batch_actions: List[Scalar[DType.float32]],
        batch_rewards: List[Scalar[DType.float32]],
        batch_dones: List[Scalar[DType.float32]],
    ) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
        """Full GPU BPTT backward for world model using autodiff.

        Replaces both the per-timestep head backward AND the BPTT backward
        loop with a single unified reverse pass. For each timestep (reverse):
        1. Re-forward prediction heads with cache
        2. Compute loss gradients (MSE/two-hot CE/BCE)
        3. Backward heads -> d_feat
        4. Split d_feat, add recurrent stoch carry
        5. Straight-through VJP + KL gradients
        6. Posterior/Prior/Encoder backward
        7. GRU backward -> recurrent carries

        Returns:
            (total_wm_loss, obs_loss, rew_loss, cont_loss, dyn_kl, rep_kl)
        """
        comptime B = Self.batch_size
        comptime BL = Self.batch_length
        comptime DETER = Self.deter_dim
        comptime STOCH = Self.STOCH_FLAT
        comptime FEAT = Self.FEAT_DIM
        comptime OBS = Self.obs_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime HID = Self.hidden

        # ── Network type aliases ─────────────────────────────────────────
        comptime EncNet = Self.StateType.RSSMType.EncNet
        comptime PostNet = Self.StateType.RSSMType.PostNet
        comptime PriorNet = Self.StateType.RSSMType.PriorNet
        comptime DecNet = Self.StateType.RSSMType.DecNet
        comptime RewNet = Self.StateType.RSSMType.RewNet
        comptime ContNet = Self.StateType.RSSMType.ContNet
        comptime DProjNet = Self.StateType.RSSMType.DeterProjNet
        comptime SProjNet = Self.StateType.RSSMType.StochProjNet
        comptime AProjNet = Self.StateType.RSSMType.ActionProjNet
        comptime GHNet = Self.StateType.RSSMType.GRUHiddenNet
        comptime GGNet = Self.StateType.RSSMType.GRUGateNet

        # ── Cache sizes ──────────────────────────────────────────────────
        comptime DEC_CACHE = Self.StateType.RSSMType.DecModel.CACHE_SIZE
        comptime REW_CACHE = Self.StateType.RSSMType.RewModel.CACHE_SIZE
        comptime CONT_CACHE = Self.StateType.RSSMType.ContModel.CACHE_SIZE
        comptime POST_CACHE = Self.StateType.RSSMType.PostModel.CACHE_SIZE
        comptime PRIOR_CACHE = Self.StateType.RSSMType.PriorModel.CACHE_SIZE
        comptime ENC_CACHE = Self.StateType.RSSMType.EncModel.CACHE_SIZE
        comptime DPROJ_CACHE = Self.StateType.RSSMType.DeterProj.CACHE_SIZE
        comptime SPROJ_CACHE = Self.StateType.RSSMType.StochProj.CACHE_SIZE
        comptime APROJ_CACHE = Self.StateType.RSSMType.ActionProj.CACHE_SIZE
        comptime GH_CACHE = Self.StateType.RSSMType.GRUHiddenModel.CACHE_SIZE
        comptime GG_CACHE = Self.StateType.RSSMType.GRUGateModel.CACHE_SIZE

        # ── ComputeGraph heads constants ─────────────────────────────────
        comptime RSSMType = Self.StateType.RSSMType
        comptime HEADS_OUT = RSSMType.HEADS_OUT_DIM
        comptime HEADS_CS = RSSMType.HEADS_CACHE_SIZE
        comptime HeadsGraph = RSSMType.HeadsGraph

        # ── Flat sizes for kernel launches ────────────────────────────────
        comptime FEAT_FLAT = B * FEAT
        comptime DETER_FLAT = B * DETER
        comptime STOCH_FLAT_SZ = B * STOCH
        comptime POST_IN = DETER + STOCH
        comptime GRU_IN = DETER + 3 * HID

        # ── Zero all 11 RSSM network gradients ───────────────────────────
        ctx.enqueue_memset(gpu_state.encoder.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.posterior.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.prior.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.decoder.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.reward_head.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.continue_head.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.deter_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.stoch_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.action_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.gru_hidden.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.gru_gates.grads_buf, 0)

        # ── Zero combined heads grads + assemble combined params ──────────
        ctx.enqueue_memset(gpu_state.heads_grads_buf, 0)
        comptime HeadsCP = Self.GPUStateType.HeadsCP
        HeadsCP.assemble_gpu(
            ctx,
            gpu_state.heads_params_buf.unsafe_ptr(),
            gpu_state.decoder.params_buf.unsafe_ptr(),
            gpu_state.reward_head.params_buf.unsafe_ptr(),
            gpu_state.continue_head.params_buf.unsafe_ptr(),
        )

        # ── Zero recurrent carry buffers ──────────────────────────────────
        ctx.enqueue_memset(gpu_state.d_recurrent_deter_buf, 0)
        ctx.enqueue_memset(gpu_state.d_recurrent_stoch_buf, 0)

        # ── Loss accumulators (computed on CPU for diagnostics) ────────────
        var obs_loss = Float64(0.0)
        var rew_loss = Float64(0.0)
        var cont_loss = Float64(0.0)
        var dyn_kl_total = Float64(0.0)
        var rep_kl_total = Float64(0.0)

        # ── Reverse loop over timesteps ───────────────────────────────────
        for t_rev in range(BL):
            var t = BL - 1 - t_rev

            # ══════════════════════════════════════════════════════════════
            # Step 1: Load saved feat from all_feats_buf[t]
            # ══════════════════════════════════════════════════════════════
            var feat_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.all_feats_buf.unsafe_ptr() + t * FEAT_FLAT)

            # ══════════════════════════════════════════════════════════════
            # Step 2: Forward all 3 prediction heads via ComputeGraph
            # ══════════════════════════════════════════════════════════════
            # HeadsGraph: feat → [obs_hat(OBS), rew_logits(BINS), cont(1)]
            var heads_out_2d = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_OUT), MutAnyOrigin
            ](gpu_state.heads_out_buf.unsafe_ptr())
            var heads_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_CS), MutAnyOrigin
            ](gpu_state.heads_cache_buf.unsafe_ptr())
            var heads_params_1d = LayoutTensor[
                dtype,
                Layout.row_major(HeadsGraph.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu_state.heads_params_buf.unsafe_ptr())
            RSSMType.predict_all_heads_gpu[B](
                ctx,
                feat_2d,
                heads_out_2d,
                heads_params_1d,
                heads_cache_2d,
                gpu_state.ws_heads,
            )

            # Deinterleave combined output → individual buffers
            # (needed for loss gradient kernels + diagnostics)
            var dec_out_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_out_buf.unsafe_ptr())
            var heads_out_1d = LayoutTensor[
                dtype, Layout.row_major(B * HEADS_OUT), MutAnyOrigin
            ](gpu_state.heads_out_buf.unsafe_ptr())

            comptime HEADS_FLAT = B * HEADS_OUT
            comptime _deint_dec = deinterleave_kernel[
                B * OBS, OBS, HEADS_OUT, 0, HEADS_FLAT
            ]
            comptime _deint_rew = deinterleave_kernel[
                B * BINS, BINS, HEADS_OUT, OBS, HEADS_FLAT
            ]
            comptime _deint_cont = deinterleave_kernel[
                B, 1, HEADS_OUT, OBS + BINS, HEADS_FLAT
            ]

            comptime DEC_DEINT_BLK = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[_deint_dec, _deint_dec](
                dec_out_1d,
                heads_out_1d,
                grid_dim=(DEC_DEINT_BLK,),
                block_dim=(TPB,),
            )

            var rew_out_1d = LayoutTensor[
                dtype, Layout.row_major(B * BINS), MutAnyOrigin
            ](gpu_state.rew_logits_buf.unsafe_ptr())
            comptime REW_DEINT_BLK = (B * BINS + TPB - 1) // TPB
            ctx.enqueue_function[_deint_rew, _deint_rew](
                rew_out_1d,
                heads_out_1d,
                grid_dim=(REW_DEINT_BLK,),
                block_dim=(TPB,),
            )

            var cont_out_1d = LayoutTensor[
                dtype, Layout.row_major(B), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            comptime CONT_DEINT_BLK = (B + TPB - 1) // TPB
            ctx.enqueue_function[_deint_cont, _deint_cont](
                cont_out_1d,
                heads_out_1d,
                grid_dim=(CONT_DEINT_BLK,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 3: Compute loss gradients (all on GPU, no syncs)
            # ══════════════════════════════════════════════════════════════

            # -- Decoder: MSE gradient against symlog(obs[t+1]) --
            # Gather obs[:,t+1,:] from batch_obs on GPU, then symlog
            comptime BPTT_OBS_STRIDE = (BL + 1) * OBS
            comptime BPTT_OBS_SRC = B * BPTT_OBS_STRIDE
            comptime BPTT_OBS_FLAT = B * OBS
            comptime run_deint_target = deinterleave_kernel[
                BPTT_OBS_FLAT, OBS, BPTT_OBS_STRIDE, 0, BPTT_OBS_SRC,
            ]
            comptime BPTT_OBS_BLK = (BPTT_OBS_FLAT + TPB - 1) // TPB
            comptime run_symlog_target = symlog_kernel[BPTT_OBS_FLAT]

            var dec_target_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_OBS_FLAT), MutAnyOrigin
            ](gpu_state.dec_target_buf.unsafe_ptr())
            var batch_obs_tp1 = LayoutTensor[
                dtype, Layout.row_major(BPTT_OBS_SRC), MutAnyOrigin
            ](gpu_state.batch_obs.unsafe_ptr() + (t + 1) * OBS)
            ctx.enqueue_function[run_deint_target, run_deint_target](
                dec_target_1d,
                batch_obs_tp1,
                grid_dim=(BPTT_OBS_BLK,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[run_symlog_target, run_symlog_target](
                dec_target_1d,
                dec_target_1d,
                grid_dim=(BPTT_OBS_BLK,),
                block_dim=(TPB,),
            )
            var dec_pred_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_out_buf.unsafe_ptr())
            var dec_grad_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_grad_out_buf.unsafe_ptr())
            var mse_scale = Scalar[dtype](2.0 / Float64(B * OBS))

            comptime ad_mse_grad = mse_grad_kernel[B * OBS]

            comptime MSE_BLOCKS = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[ad_mse_grad, ad_mse_grad](
                dec_grad_1d,
                dec_pred_1d,
                dec_target_1d,
                mse_scale,
                grid_dim=(MSE_BLOCKS,),
                block_dim=(TPB,),
            )

            # -- Reward: two-hot CE gradient (t > 0 only) --
            if t > 0:
                # Gather rewards[:,t] from batch_rewards on GPU, then symlog
                comptime BPTT_REW_SRC = B * BL
                comptime run_deint_rew = deinterleave_kernel[
                    B, 1, BL, 0, BPTT_REW_SRC,
                ]
                comptime BPTT_REW_BLK = (B + TPB - 1) // TPB
                comptime run_symlog_rew = symlog_kernel[B]

                var rew_symlog_1d = LayoutTensor[
                    dtype, Layout.row_major(B), MutAnyOrigin
                ](gpu_state.rew_symlog_buf.unsafe_ptr())
                var batch_rew_t = LayoutTensor[
                    dtype, Layout.row_major(BPTT_REW_SRC), MutAnyOrigin
                ](gpu_state.batch_rewards.unsafe_ptr() + t)
                ctx.enqueue_function[run_deint_rew, run_deint_rew](
                    rew_symlog_1d,
                    batch_rew_t,
                    grid_dim=(BPTT_REW_BLK,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[run_symlog_rew, run_symlog_rew](
                    rew_symlog_1d,
                    rew_symlog_1d,
                    grid_dim=(BPTT_REW_BLK,),
                    block_dim=(TPB,),
                )
                var rew_target_2d = LayoutTensor[
                    dtype, Layout.row_major(B, BINS), MutAnyOrigin
                ](gpu_state.rew_target_buf.unsafe_ptr())
                var bins_1d = LayoutTensor[
                    dtype, Layout.row_major(BINS), MutAnyOrigin
                ](gpu_state.bins_buf.unsafe_ptr())

                comptime ad_rew_two_hot = two_hot_encode_kernel[B, BINS]

                comptime REW_TH_BLOCKS = (B + TPB - 1) // TPB
                ctx.enqueue_function[ad_rew_two_hot, ad_rew_two_hot](
                    rew_target_2d,
                    rew_symlog_1d,
                    bins_1d,
                    grid_dim=(REW_TH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Two-hot CE gradient
                var rew_logits_2d = LayoutTensor[
                    dtype, Layout.row_major(B, BINS), MutAnyOrigin
                ](gpu_state.rew_logits_buf.unsafe_ptr())
                var rew_grad_out_2d = LayoutTensor[
                    dtype, Layout.row_major(B, BINS), MutAnyOrigin
                ](gpu_state.rew_grad_out_buf.unsafe_ptr())
                var rew_inv_batch = Scalar[dtype](1.0 / Float64(B))

                comptime ad_rew_ce_grad = two_hot_ce_grad_kernel[B, BINS]

                ctx.enqueue_function[ad_rew_ce_grad, ad_rew_ce_grad](
                    rew_grad_out_2d,
                    rew_logits_2d,
                    rew_target_2d,
                    rew_inv_batch,
                    grid_dim=(REW_TH_BLOCKS,),
                    block_dim=(TPB,),
                )
            else:
                # t == 0: zero reward gradient
                ctx.enqueue_memset(gpu_state.rew_grad_out_buf, 0)

            # -- Continue: BCE gradient (t > 0 only) --
            if t > 0:
                # Sigmoid on cont logit
                var cont_pred_1d = LayoutTensor[
                    dtype, Layout.row_major(B), MutAnyOrigin
                ](gpu_state.cont_out_buf.unsafe_ptr())

                comptime ad_cont_sigmoid = sigmoid_kernel[B]

                comptime CONT_SIG_BLOCKS = (B + TPB - 1) // TPB
                ctx.enqueue_function[ad_cont_sigmoid, ad_cont_sigmoid](
                    cont_pred_1d,
                    cont_pred_1d,
                    grid_dim=(CONT_SIG_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Gather 1.0 - dones[:,t] from batch_dones on GPU
                comptime BPTT_DONE_SRC = B * BL
                comptime run_deint_done = deinterleave_kernel[
                    B, 1, BL, 0, BPTT_DONE_SRC,
                ]
                comptime BPTT_DONE_BLK = (B + TPB - 1) // TPB
                comptime run_one_minus_done = one_minus_kernel[B]

                var cont_tgt_1d = LayoutTensor[
                    dtype, Layout.row_major(B), MutAnyOrigin
                ](gpu_state.cont_target_buf.unsafe_ptr())
                var batch_done_t = LayoutTensor[
                    dtype, Layout.row_major(BPTT_DONE_SRC), MutAnyOrigin
                ](gpu_state.batch_dones.unsafe_ptr() + t)
                ctx.enqueue_function[run_deint_done, run_deint_done](
                    cont_tgt_1d,
                    batch_done_t,
                    grid_dim=(BPTT_DONE_BLK,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[run_one_minus_done, run_one_minus_done](
                    cont_tgt_1d,
                    cont_tgt_1d,
                    grid_dim=(BPTT_DONE_BLK,),
                    block_dim=(TPB,),
                )

                # BCE gradient
                var cont_pred_2d = LayoutTensor[
                    dtype, Layout.row_major(B, 1), MutAnyOrigin
                ](gpu_state.cont_out_buf.unsafe_ptr())
                var cont_target_2d = LayoutTensor[
                    dtype, Layout.row_major(B, 1), MutAnyOrigin
                ](gpu_state.cont_target_buf.unsafe_ptr())
                var cont_grad_2d = LayoutTensor[
                    dtype, Layout.row_major(B, 1), MutAnyOrigin
                ](gpu_state.cont_grad_buf.unsafe_ptr())
                var cont_inv_batch = Scalar[dtype](1.0 / Float64(B))

                comptime ad_cont_bce_grad = bce_grad_kernel[B]

                ctx.enqueue_function[ad_cont_bce_grad, ad_cont_bce_grad](
                    cont_grad_2d,
                    cont_pred_2d,
                    cont_target_2d,
                    cont_inv_batch,
                    grid_dim=(CONT_SIG_BLOCKS,),
                    block_dim=(TPB,),
                )
            else:
                # t == 0: zero continue gradient
                ctx.enqueue_memset(gpu_state.cont_grad_buf, 0)

            # ══════════════════════════════════════════════════════════════
            # Step 4: Interleave loss gradients → combined heads_grad_out
            # ══════════════════════════════════════════════════════════════
            var hg_out_1d = LayoutTensor[
                dtype, Layout.row_major(HEADS_FLAT), MutAnyOrigin
            ](gpu_state.heads_grad_out_buf.unsafe_ptr())

            # decoder grad → heads_grad_out[:, :OBS]
            var dg_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_grad_out_buf.unsafe_ptr())

            comptime _int_dec = interleave_kernel[
                B * OBS, OBS, HEADS_OUT, 0, HEADS_FLAT
            ]

            comptime INT_DEC_BLK = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[_int_dec, _int_dec](
                hg_out_1d,
                dg_1d,
                grid_dim=(INT_DEC_BLK,),
                block_dim=(TPB,),
            )

            # reward grad → heads_grad_out[:, OBS:OBS+BINS]
            var rg_1d = LayoutTensor[
                dtype, Layout.row_major(B * BINS), MutAnyOrigin
            ](gpu_state.rew_grad_out_buf.unsafe_ptr())

            comptime _int_rew = interleave_kernel[
                B * BINS, BINS, HEADS_OUT, OBS, HEADS_FLAT
            ]

            comptime INT_REW_BLK = (B * BINS + TPB - 1) // TPB
            ctx.enqueue_function[_int_rew, _int_rew](
                hg_out_1d,
                rg_1d,
                grid_dim=(INT_REW_BLK,),
                block_dim=(TPB,),
            )

            # continue grad → heads_grad_out[:, OBS+BINS:]
            var cg_1d = LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin](
                gpu_state.cont_grad_buf.unsafe_ptr()
            )

            comptime _int_cont = interleave_kernel[
                B, 1, HEADS_OUT, OBS + BINS, HEADS_FLAT
            ]

            comptime INT_CONT_BLK = (B + TPB - 1) // TPB
            ctx.enqueue_function[_int_cont, _int_cont](
                hg_out_1d,
                cg_1d,
                grid_dim=(INT_CONT_BLK,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 5: ComputeGraph backward → d_feat + heads param grads
            # ══════════════════════════════════════════════════════════════
            # HeadsGraph.backward_gpu handles fan-out gradient accumulation
            # at the feat input automatically — no manual d_feat sum needed.
            var d_feat_2d_bwd = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.d_feat_buf.unsafe_ptr())
            var hg_grad_out_2d = LayoutTensor[
                dtype, Layout.row_major(B, HEADS_OUT), MutAnyOrigin
            ](gpu_state.heads_grad_out_buf.unsafe_ptr())
            var heads_grads_1d = LayoutTensor[
                dtype,
                Layout.row_major(HeadsGraph.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu_state.heads_grads_buf.unsafe_ptr())
            RSSMType.backward_all_heads_gpu[B](
                ctx,
                hg_grad_out_2d,
                d_feat_2d_bwd,
                heads_params_1d,
                heads_cache_2d,
                heads_grads_1d,
                gpu_state.ws_heads,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 6: Split d_feat -> d_deter, d_stoch
            # ══════════════════════════════════════════════════════════════
            var d_feat_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.d_feat_buf.unsafe_ptr())
            var d_deter = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_deter_total_buf.unsafe_ptr())
            var d_stoch = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_stoch_feat_buf.unsafe_ptr())

            comptime ad_split_feat = concat_feat_backward_kernel[
                B, DETER, STOCH
            ]

            comptime SPLIT_FEAT_BLOCKS = (B * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[ad_split_feat, ad_split_feat](
                d_deter,
                d_stoch,
                d_feat_2d,
                grid_dim=(SPLIT_FEAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 7: Add recurrent stoch carry: d_stoch += d_recurrent_stoch
            # ══════════════════════════════════════════════════════════════
            var d_stoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_stoch_feat_buf.unsafe_ptr())
            var rec_stoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_recurrent_stoch_buf.unsafe_ptr())

            comptime ad_add_rec_stoch = accumulate_kernel[STOCH_FLAT_SZ]

            comptime STOCH_BLOCKS = (STOCH_FLAT_SZ + TPB - 1) // TPB
            ctx.enqueue_function[ad_add_rec_stoch, ad_add_rec_stoch](
                d_stoch_1d,
                rec_stoch_1d,
                grid_dim=(STOCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 8: Straight-through VJP: d_stoch, post_probs -> d_post_logits
            # ══════════════════════════════════════════════════════════════
            var post_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.all_post_probs_buf.unsafe_ptr() + t * B * STOCH)
            var d_post_logits = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_post_logits_total_buf.unsafe_ptr())

            comptime ad_st_vjp = straight_through_softmax_vjp_kernel[
                B,
                Self.stoch_dim,
                Self.classes,
                Self.StateType.RSSMType.UNIMIX,
            ]

            comptime ST_BLOCKS = (B * Self.stoch_dim + TPB - 1) // TPB
            ctx.enqueue_function[ad_st_vjp, ad_st_vjp](
                d_post_logits,
                d_stoch,
                post_probs_t,
                grid_dim=(ST_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 9: KL gradient: kl_divergence + kl_categorical_gradient
            # ══════════════════════════════════════════════════════════════
            var prior_probs_t = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.all_prior_probs_buf.unsafe_ptr() + t * B * STOCH)

            # KL divergence
            var kl_val = LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin](
                gpu_state.kl_buf.unsafe_ptr()
            )

            comptime ad_kl_div = kl_divergence_kernel[
                B, Self.stoch_dim, Self.classes
            ]

            comptime KL_BLOCKS = (B + TPB - 1) // TPB
            ctx.enqueue_function[ad_kl_div, ad_kl_div](
                kl_val,
                post_probs_t,
                prior_probs_t,
                grid_dim=(KL_BLOCKS,),
                block_dim=(TPB,),
            )

            # KL gradient -> d_post_kl, d_prior_logits
            var d_post_kl = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.post_grad_out_buf.unsafe_ptr())
            var d_prior_logits = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.prior_grad_out_buf.unsafe_ptr())
            var kl_free_nats = Scalar[dtype](Self.StateType.RSSMType.FREE_NATS)
            var kl_dyn_scale = Scalar[dtype](0.5)
            var kl_rep_scale = Scalar[dtype](0.1)
            var kl_inv_batch = Scalar[dtype](1.0 / Float64(B))

            comptime ad_kl_grad = kl_categorical_gradient_kernel[
                B, Self.stoch_dim, Self.classes
            ]

            comptime KL_GRAD_BLOCKS = (B * Self.stoch_dim + TPB - 1) // TPB
            ctx.enqueue_function[ad_kl_grad, ad_kl_grad](
                d_post_kl,
                d_prior_logits,
                post_probs_t,
                prior_probs_t,
                kl_val,
                kl_free_nats,
                kl_dyn_scale,
                kl_rep_scale,
                kl_inv_batch,
                grid_dim=(KL_GRAD_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 10: d_post_total = d_post_logits + d_post_kl
            # ══════════════════════════════════════════════════════════════
            var d_post_total_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_post_logits_total_buf.unsafe_ptr())
            var d_post_kl_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.post_grad_out_buf.unsafe_ptr())

            comptime ad_add_kl_post = accumulate_kernel[STOCH_FLAT_SZ]

            ctx.enqueue_function[ad_add_kl_post, ad_add_kl_post](
                d_post_total_1d,
                d_post_kl_1d,
                grid_dim=(STOCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 11: Posterior backward (saved post_cache at offset t)
            # ══════════════════════════════════════════════════════════════
            var d_post_total_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_post_logits_total_buf.unsafe_ptr())
            var post_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, POST_CACHE), MutAnyOrigin
            ](gpu_state.all_post_cache_buf.unsafe_ptr() + t * B * POST_CACHE)
            var post_grad_in = LayoutTensor[
                dtype, Layout.row_major(B, POST_IN), MutAnyOrigin
            ](gpu_state.post_grad_in_buf.unsafe_ptr())
            var post_grads_bptt = gpu_state.posterior.grads_view()
            PostNet.backward_gpu[B](
                ctx,
                d_post_total_2d,
                post_grad_in,
                gpu_state.posterior.params_view(),
                post_cache_t,
                post_grads_bptt,
                gpu_state.ws_posterior,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 12: Split d_post_in -> d_deter_from_post, d_embed
            # ══════════════════════════════════════════════════════════════
            var d_deter_from_post = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())
            var d_embed = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_embed_bwd_buf.unsafe_ptr())

            comptime ad_split_post_in = concat_deter_embed_backward_kernel[
                B, DETER, STOCH
            ]

            comptime SPLIT_POST_BLOCKS = (B * POST_IN + TPB - 1) // TPB
            ctx.enqueue_function[ad_split_post_in, ad_split_post_in](
                d_deter_from_post,
                d_embed,
                post_grad_in,
                grid_dim=(SPLIT_POST_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 13: Prior backward (saved prior_cache at offset t)
            # ══════════════════════════════════════════════════════════════
            var prior_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, PRIOR_CACHE), MutAnyOrigin
            ](gpu_state.all_prior_cache_buf.unsafe_ptr() + t * B * PRIOR_CACHE)
            var prior_grad_in = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.prior_grad_in_buf.unsafe_ptr())
            var prior_grads_bptt = gpu_state.prior.grads_view()
            PriorNet.backward_gpu[B](
                ctx,
                d_prior_logits,
                prior_grad_in,
                gpu_state.prior.params_view(),
                prior_cache_t,
                prior_grads_bptt,
                gpu_state.ws_prior,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 14: Encoder backward (saved enc_cache at offset t)
            # ══════════════════════════════════════════════════════════════
            var enc_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, ENC_CACHE), MutAnyOrigin
            ](gpu_state.all_enc_cache_buf.unsafe_ptr() + t * B * ENC_CACHE)
            var d_symlog_obs = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](gpu_state.d_symlog_obs_bwd_buf.unsafe_ptr())
            var enc_grads_bptt = gpu_state.encoder.grads_view()
            EncNet.backward_gpu[B](
                ctx,
                d_embed,
                d_symlog_obs,
                gpu_state.encoder.params_view(),
                enc_cache_t,
                enc_grads_bptt,
                gpu_state.ws_encoder,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 15: d_deter_total += d_deter_from_post + d_deter_from_prior
            #          + d_recurrent_deter
            # ══════════════════════════════════════════════════════════════
            var dd_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_deter_total_buf.unsafe_ptr())
            var dd_post_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())
            var dd_prior_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.prior_grad_in_buf.unsafe_ptr())
            var dd_rec_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_recurrent_deter_buf.unsafe_ptr())

            comptime DD_BLOCKS = (DETER_FLAT + TPB - 1) // TPB

            # d_deter_total already has d_deter_feat from split; add the rest
            comptime ad_add_dd_post = accumulate_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_add_dd_post, ad_add_dd_post](
                dd_1d,
                dd_post_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime ad_add_dd_prior = accumulate_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_add_dd_prior, ad_add_dd_prior](
                dd_1d,
                dd_prior_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime ad_add_dd_rec = accumulate_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_add_dd_rec, ad_add_dd_rec](
                dd_1d,
                dd_rec_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 16: GRU gate backward: d_deter_total -> d_gate, d_prev_deter_gru
            # ══════════════════════════════════════════════════════════════
            var d_gate = LayoutTensor[
                dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin
            ](gpu_state.d_gate_out_bwd_buf.unsafe_ptr())
            var d_prev_deter_gru = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_prev_deter_gru_buf.unsafe_ptr())
            var prev_deter_t = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.all_prev_deter_buf.unsafe_ptr() + t * B * DETER)
            var gate_out_t = LayoutTensor[
                dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin
            ](gpu_state.all_gate_out_buf.unsafe_ptr() + t * B * 3 * DETER)

            comptime ad_gru_bwd = gru_gate_backward_kernel[B, DETER]

            comptime GRU_BLOCKS = (B * DETER + TPB - 1) // TPB
            ctx.enqueue_function[ad_gru_bwd, ad_gru_bwd](
                d_gate,
                d_prev_deter_gru,
                d_deter,
                prev_deter_t,
                gate_out_t,
                grid_dim=(GRU_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 17: GRU gates backward (saved cache) -> d_hidden
            # ══════════════════════════════════════════════════════════════
            var gg_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, GG_CACHE), MutAnyOrigin
            ](gpu_state.all_gru_gates_cache_buf.unsafe_ptr() + t * B * GG_CACHE)
            var d_hidden = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_hidden_out_bwd_buf.unsafe_ptr())
            var gg_grads_bptt = gpu_state.gru_gates.grads_view()
            GGNet.backward_gpu[B](
                ctx,
                d_gate,
                d_hidden,
                gpu_state.gru_gates.params_view(),
                gg_cache_t,
                gg_grads_bptt,
                gpu_state.ws_gru_gates,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 18: GRU hidden backward (saved cache) -> d_concat
            # ══════════════════════════════════════════════════════════════
            var gh_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, GH_CACHE), MutAnyOrigin
            ](
                gpu_state.all_gru_hidden_cache_buf.unsafe_ptr()
                + t * B * GH_CACHE
            )
            var d_concat = LayoutTensor[
                dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin
            ](gpu_state.d_concat_bwd_buf.unsafe_ptr())
            var gh_grads_bptt = gpu_state.gru_hidden.grads_view()
            GHNet.backward_gpu[B](
                ctx,
                d_hidden,
                d_concat,
                gpu_state.gru_hidden.params_view(),
                gh_cache_t,
                gh_grads_bptt,
                gpu_state.ws_gru_hidden,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 19: Split d_concat -> d_prev_deter_concat, d_proj_d/s/a
            # ══════════════════════════════════════════════════════════════
            var d_prev_deter_concat = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](
                gpu_state.d_deter_from_post_buf.unsafe_ptr()
            )  # reuse buffer
            var d_proj_d = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_d_bwd_buf.unsafe_ptr())
            var d_proj_s = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_s_bwd_buf.unsafe_ptr())
            var d_proj_a = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_a_bwd_buf.unsafe_ptr())

            comptime ad_split_concat = concat_gru_input_backward_kernel[
                B, DETER, HID
            ]

            comptime SPLIT_CONCAT_BLOCKS = (B * GRU_IN + TPB - 1) // TPB
            ctx.enqueue_function[ad_split_concat, ad_split_concat](
                d_prev_deter_concat,
                d_proj_d,
                d_proj_s,
                d_proj_a,
                d_concat,
                grid_dim=(SPLIT_CONCAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 20: DeterProj backward (saved cache) -> d_prev_deter_dproj
            # ══════════════════════════════════════════════════════════════
            var dproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, DPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_dproj_cache_buf.unsafe_ptr() + t * B * DPROJ_CACHE)
            var d_prev_deter_dproj = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_prev_deter_dproj_buf.unsafe_ptr())
            var dp_grads_bptt = gpu_state.deter_proj.grads_view()
            DProjNet.backward_gpu[B](
                ctx,
                d_proj_d,
                d_prev_deter_dproj,
                gpu_state.deter_proj.params_view(),
                dproj_cache_t,
                dp_grads_bptt,
                gpu_state.ws_deter_proj,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 21: StochProj backward (saved cache) -> d_prev_stoch
            # ══════════════════════════════════════════════════════════════
            var sproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, SPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_sproj_cache_buf.unsafe_ptr() + t * B * SPROJ_CACHE)
            var d_prev_stoch = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_prev_stoch_bwd_buf.unsafe_ptr())
            var sp_grads_bptt = gpu_state.stoch_proj.grads_view()
            SProjNet.backward_gpu[B](
                ctx,
                d_proj_s,
                d_prev_stoch,
                gpu_state.stoch_proj.params_view(),
                sproj_cache_t,
                sp_grads_bptt,
                gpu_state.ws_stoch_proj,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 22: ActionProj backward (saved cache) -> d_prev_action (discarded)
            # ══════════════════════════════════════════════════════════════
            var aproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, APROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_aproj_cache_buf.unsafe_ptr() + t * B * APROJ_CACHE)
            var d_prev_action = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.d_prev_action_bwd_buf.unsafe_ptr())
            var ap_grads_bptt = gpu_state.action_proj.grads_view()
            AProjNet.backward_gpu[B](
                ctx,
                d_proj_a,
                d_prev_action,
                gpu_state.action_proj.params_view(),
                aproj_cache_t,
                ap_grads_bptt,
                gpu_state.ws_action_proj,
            )

            # ══════════════════════════════════════════════════════════════
            # Step 23: d_recurrent_deter = d_prev_deter_gru + d_prev_deter_concat
            #          + d_prev_deter_dproj
            # ══════════════════════════════════════════════════════════════
            var rec_deter = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_recurrent_deter_buf.unsafe_ptr())
            var dpd_gru_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_prev_deter_gru_buf.unsafe_ptr())
            var dpd_concat_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](
                gpu_state.d_deter_from_post_buf.unsafe_ptr()
            )  # reused for concat split
            var dpd_dproj_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_prev_deter_dproj_buf.unsafe_ptr())

            # Copy d_prev_deter_gru -> d_recurrent_deter
            comptime ad_copy_rec_d = copy_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_copy_rec_d, ad_copy_rec_d](
                rec_deter,
                dpd_gru_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            # + d_prev_deter_concat
            comptime ad_add_concat_d = accumulate_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_add_concat_d, ad_add_concat_d](
                rec_deter,
                dpd_concat_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            # + d_prev_deter_dproj
            comptime ad_add_dproj_d = accumulate_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_add_dproj_d, ad_add_dproj_d](
                rec_deter,
                dpd_dproj_1d,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 24: d_recurrent_stoch = d_prev_stoch (copy)
            # ══════════════════════════════════════════════════════════════
            var rec_stoch_dst = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_recurrent_stoch_buf.unsafe_ptr())
            var dpstoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_prev_stoch_bwd_buf.unsafe_ptr())

            comptime ad_copy_rec_s = copy_kernel[STOCH_FLAT_SZ]

            ctx.enqueue_function[ad_copy_rec_s, ad_copy_rec_s](
                rec_stoch_dst,
                dpstoch_1d,
                grid_dim=(STOCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # ══════════════════════════════════════════════════════════════
            # Step 25: Clamp recurrent carries to prevent explosion
            # ══════════════════════════════════════════════════════════════
            var clamp_max = Scalar[dtype](1.0)

            comptime ad_clamp_rec_d = clamp_kernel[DETER_FLAT]

            ctx.enqueue_function[ad_clamp_rec_d, ad_clamp_rec_d](
                rec_deter,
                clamp_max,
                grid_dim=(DD_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime ad_clamp_rec_s = clamp_kernel[STOCH_FLAT_SZ]

            ctx.enqueue_function[ad_clamp_rec_s, ad_clamp_rec_s](
                rec_stoch_dst,
                clamp_max,
                grid_dim=(STOCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── Scatter combined heads grads → individual network grads ──────
        HeadsCP.scatter_add_gpu(
            ctx,
            gpu_state.heads_grads_buf.unsafe_ptr(),
            gpu_state.decoder.grads_buf.unsafe_ptr(),
            gpu_state.reward_head.grads_buf.unsafe_ptr(),
            gpu_state.continue_head.grads_buf.unsafe_ptr(),
        )

        # ── Normalize losses ─────────────────────────────────────────────
        var inv_bl = 1.0 / Float64(BL)
        var inv_bl_b = 1.0 / Float64(BL * B)
        obs_loss *= inv_bl_b
        rew_loss *= inv_bl_b
        cont_loss *= inv_bl_b
        dyn_kl_total *= inv_bl
        rep_kl_total *= inv_bl

        var total = (
            obs_loss
            + rew_loss
            + cont_loss
            + (0.5 * dyn_kl_total + 0.1 * rep_kl_total)
        )
        return (
            total,
            obs_loss,
            rew_loss,
            cont_loss,
            dyn_kl_total,
            rep_kl_total,
        )

    # ══════════════════════════════════════════════════════════════════════
    # GPU Methods
    # ══════════════════════════════════════════════════════════════════════

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for DreamerV3 training."""
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises:
        """Upload CPU network weights and bins to GPU."""
        # RSSM networks
        gpu_state.encoder.upload_from(self.state.rssm.encoder, ctx)
        gpu_state.posterior.upload_from(self.state.rssm.posterior, ctx)
        gpu_state.prior.upload_from(self.state.rssm.prior, ctx)
        gpu_state.decoder.upload_from(self.state.rssm.decoder, ctx)
        gpu_state.reward_head.upload_from(self.state.rssm.reward_head, ctx)
        gpu_state.continue_head.upload_from(self.state.rssm.continue_head, ctx)
        gpu_state.deter_proj.upload_from(self.state.rssm.deter_proj, ctx)
        gpu_state.stoch_proj.upload_from(self.state.rssm.stoch_proj, ctx)
        gpu_state.action_proj.upload_from(self.state.rssm.action_proj, ctx)
        gpu_state.gru_hidden.upload_from(self.state.rssm.gru_hidden, ctx)
        gpu_state.gru_gates.upload_from(self.state.rssm.gru_gates, ctx)

        # Actor-Critic
        gpu_state.actor.upload_from(self.state.actor, ctx)
        gpu_state.critic.upload_from(self.state.critic, ctx)
        # Slow critic: upload critic params as slow critic
        gpu_state.slow_critic.upload_from(self.state.critic, ctx)

        # Upload symlog bins
        var bins_host = gpu_state.host_bins_buf
        for i in range(Self.num_bins):
            bins_host[i] = Scalar[dtype](self.state.rssm.bins[i])
        ctx.enqueue_copy(gpu_state.bins_buf, bins_host)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises:
        """Download GPU trained weights to CPU."""
        gpu_state.encoder.download_to(self.state.rssm.encoder, ctx)
        gpu_state.posterior.download_to(self.state.rssm.posterior, ctx)
        gpu_state.prior.download_to(self.state.rssm.prior, ctx)
        gpu_state.decoder.download_to(self.state.rssm.decoder, ctx)
        gpu_state.reward_head.download_to(self.state.rssm.reward_head, ctx)
        gpu_state.continue_head.download_to(self.state.rssm.continue_head, ctx)
        gpu_state.deter_proj.download_to(self.state.rssm.deter_proj, ctx)
        gpu_state.stoch_proj.download_to(self.state.rssm.stoch_proj, ctx)
        gpu_state.action_proj.download_to(self.state.rssm.action_proj, ctx)
        gpu_state.gru_hidden.download_to(self.state.rssm.gru_hidden, ctx)
        gpu_state.gru_gates.download_to(self.state.rssm.gru_gates, ctx)
        gpu_state.actor.download_to(self.state.actor, ctx)
        gpu_state.critic.download_to(self.state.critic, ctx)

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        batch_obs: List[Scalar[DType.float32]],
        batch_actions: List[Scalar[DType.float32]],
        batch_rewards: List[Scalar[DType.float32]],
        batch_dones: List[Scalar[DType.float32]],
    ) raises:
        """One full DreamerV3 GPU training step.

        Sequence data is sampled on CPU and uploaded here.
        All forward/backward passes run on GPU.

        Steps:
        1. Upload batch data to GPU
        2. RSSM observe loop (sequential across BL, parallel across BATCH)
        3. World model losses + backward + optimizer step
        4. Imagination rollout (sequential across HORIZON, parallel across IB)
        5. Lambda returns + normalization
        6. Critic loss + backward + optimizer step
        7. Actor loss + backward + optimizer step
        8. Slow critic EMA update

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with all device buffers.
            batch_obs: Sampled observations [BATCH * (BL+1) * OBS].
            batch_actions: Sampled actions [BATCH * BL * ACT].
            batch_rewards: Sampled rewards [BATCH * BL].
            batch_dones: Sampled dones [BATCH * BL].
        """
        comptime B = Self.batch_size
        comptime BL = Self.batch_length
        comptime DETER = Self.deter_dim
        comptime STOCH = Self.STOCH_FLAT
        comptime FEAT = Self.FEAT_DIM
        comptime OBS = Self.obs_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime HORIZON = Self.imagine_horizon
        comptime IB = Self.IMAG_BATCH
        comptime HID = Self.hidden

        # ── Network type aliases ─────────────────────────────────────────
        comptime EncNet = Self.StateType.RSSMType.EncNet
        comptime PostNet = Self.StateType.RSSMType.PostNet
        comptime PriorNet = Self.StateType.RSSMType.PriorNet
        comptime DecNet = Self.StateType.RSSMType.DecNet
        comptime RewNet = Self.StateType.RSSMType.RewNet
        comptime ContNet = Self.StateType.RSSMType.ContNet
        comptime DProjNet = Self.StateType.RSSMType.DeterProjNet
        comptime SProjNet = Self.StateType.RSSMType.StochProjNet
        comptime AProjNet = Self.StateType.RSSMType.ActionProjNet
        comptime GHNet = Self.StateType.RSSMType.GRUHiddenNet
        comptime GGNet = Self.StateType.RSSMType.GRUGateNet

        # ── Phase timing (synced to measure actual GPU time) ─────────────
        var _pt0 = perf_counter_ns()

        # ── 1. Upload batch data to GPU ──────────────────────────────────
        comptime OBS_SIZE = B * (BL + 1) * OBS
        comptime ACT_SIZE = B * BL * ACT
        comptime SCALAR_SIZE = B * BL

        var host_obs = gpu_state.host_upload_obs_buf
        for i in range(OBS_SIZE):
            host_obs[i] = Scalar[dtype](batch_obs[i])
        ctx.enqueue_copy(gpu_state.batch_obs, host_obs)

        var host_act = gpu_state.host_upload_act_buf
        for i in range(ACT_SIZE):
            host_act[i] = Scalar[dtype](batch_actions[i])
        ctx.enqueue_copy(gpu_state.batch_actions, host_act)

        var host_rew = gpu_state.host_upload_rew_buf
        for i in range(SCALAR_SIZE):
            host_rew[i] = Scalar[dtype](batch_rewards[i])
        ctx.enqueue_copy(gpu_state.batch_rewards, host_rew)

        var host_done = gpu_state.host_upload_done_buf
        for i in range(SCALAR_SIZE):
            host_done[i] = Scalar[dtype](batch_dones[i])
        ctx.enqueue_copy(gpu_state.batch_dones, host_done)

        # ── Zero all world model gradients ───────────────────────────────
        ctx.enqueue_memset(gpu_state.encoder.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.posterior.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.prior.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.decoder.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.reward_head.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.continue_head.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.deter_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.stoch_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.action_proj.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.gru_hidden.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.gru_gates.grads_buf, 0)

        # ── Zero initial deter/stoch ─────────────────────────────────────
        ctx.enqueue_memset(gpu_state.deter_buf, 0)
        ctx.enqueue_memset(gpu_state.stoch_buf, 0)

        ctx.synchronize()
        var _pt1 = perf_counter_ns()  # end upload

        # ── 2. RSSM Observe Loop ─────────────────────────────────────────
        var total_kl = Float64(0.0)

        # ── Kernel aliases for timestep gather ───────────────────────────
        comptime OBS_FLAT = B * OBS
        comptime OBS_STRIDE = (BL + 1) * OBS
        comptime OBS_SRC_FLAT = OBS_SIZE  # B * (BL+1) * OBS
        comptime OBS_DEINT_BLK = (OBS_FLAT + TPB - 1) // TPB
        comptime run_deint_obs = deinterleave_kernel[
            OBS_FLAT, OBS, OBS_STRIDE, 0, OBS_SRC_FLAT,
        ]

        comptime ACT_FLAT = B * ACT
        comptime ACT_STRIDE = BL * ACT
        comptime ACT_SRC_FLAT = ACT_SIZE  # B * BL * ACT
        comptime ACT_DEINT_BLK = (ACT_FLAT + TPB - 1) // TPB
        comptime run_deint_act = deinterleave_kernel[
            ACT_FLAT, ACT, ACT_STRIDE, 0, ACT_SRC_FLAT,
        ]

        var obs_step_lt = LayoutTensor[
            dtype, Layout.row_major(OBS_FLAT), MutAnyOrigin
        ](gpu_state.obs_step_buf.unsafe_ptr())
        var act_step_lt = LayoutTensor[
            dtype, Layout.row_major(ACT_FLAT), MutAnyOrigin
        ](gpu_state.act_step_buf.unsafe_ptr())

        for t in range(BL):
            # Gather obs[:,t,:] from batch_obs on GPU via deinterleave
            # with pointer shifted by t*OBS so kernel reads
            # src[b*OBS_STRIDE + d] = batch_obs[b*(BL+1)*OBS + t*OBS + d]
            var batch_obs_at_t = LayoutTensor[
                dtype, Layout.row_major(OBS_SRC_FLAT), MutAnyOrigin
            ](gpu_state.batch_obs.unsafe_ptr() + t * OBS)
            ctx.enqueue_function[run_deint_obs, run_deint_obs](
                obs_step_lt,
                batch_obs_at_t,
                grid_dim=(OBS_DEINT_BLK,),
                block_dim=(TPB,),
            )

            # Gather act[:,t-1,:] from batch_actions on GPU (zero for t==0)
            if t == 0:
                ctx.enqueue_memset(gpu_state.act_step_buf, 0)
            else:
                var batch_act_at_t = LayoutTensor[
                    dtype, Layout.row_major(ACT_SRC_FLAT), MutAnyOrigin
                ](gpu_state.batch_actions.unsafe_ptr() + (t - 1) * ACT)
                ctx.enqueue_function[run_deint_act, run_deint_act](
                    act_step_lt,
                    batch_act_at_t,
                    grid_dim=(ACT_DEINT_BLK,),
                    block_dim=(TPB,),
                )

            # Symlog observations
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.obs_step_buf.unsafe_ptr())
            var symlog_t = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.symlog_obs_buf.unsafe_ptr())

            comptime run_symlog = symlog_kernel[B * OBS]

            comptime SYMLOG_BLOCKS = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[run_symlog, run_symlog](
                symlog_t,
                obs_t,
                grid_dim=(SYMLOG_BLOCKS,),
                block_dim=(TPB,),
            )

            # Encode: symlog_obs -> embed (with cache for BPTT)
            var symlog_obs_2d = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](gpu_state.symlog_obs_buf.unsafe_ptr())
            var embed_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.embed_buf.unsafe_ptr())
            comptime ENC_CACHE = Self.StateType.RSSMType.EncModel.CACHE_SIZE
            var enc_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, ENC_CACHE), MutAnyOrigin
            ](gpu_state.all_enc_cache_buf.unsafe_ptr() + t * B * ENC_CACHE)
            EncNet.forward_gpu_with_cache[B](
                ctx,
                symlog_obs_2d,
                embed_2d,
                gpu_state.encoder.params_view(),
                enc_cache_t,
                gpu_state.ws_encoder,
            )

            # Save symlog_obs per timestep for BPTT encoder backward
            comptime SYMLOG_SLICE = B * OBS
            var all_symlog_t = LayoutTensor[
                dtype, Layout.row_major(SYMLOG_SLICE), MutAnyOrigin
            ](gpu_state.all_symlog_obs_buf.unsafe_ptr() + t * SYMLOG_SLICE)
            var symlog_1d = LayoutTensor[
                dtype, Layout.row_major(SYMLOG_SLICE), MutAnyOrigin
            ](gpu_state.symlog_obs_buf.unsafe_ptr())

            comptime copy_symlog = copy_kernel[SYMLOG_SLICE]

            comptime COPY_SL_BLOCKS = (SYMLOG_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_symlog, copy_symlog](
                all_symlog_t,
                symlog_1d,
                grid_dim=(COPY_SL_BLOCKS,),
                block_dim=(TPB,),
            )

            # Action normalize
            var act_2d = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.act_step_buf.unsafe_ptr())
            var norm_act_2d = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.norm_action_buf.unsafe_ptr())

            comptime run_action_norm = action_normalize_kernel[B, ACT]

            comptime NORM_BLOCKS = (B * ACT + TPB - 1) // TPB
            ctx.enqueue_function[run_action_norm, run_action_norm](
                norm_act_2d,
                act_2d,
                grid_dim=(NORM_BLOCKS,),
                block_dim=(TPB,),
            )

            # Save norm_action per timestep for BPTT
            comptime ACT_SLICE = B * ACT
            var all_nact_t = LayoutTensor[
                dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin
            ](gpu_state.all_norm_action_buf.unsafe_ptr() + t * ACT_SLICE)
            var nact_1d = LayoutTensor[
                dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin
            ](gpu_state.norm_action_buf.unsafe_ptr())

            comptime copy_nact = copy_kernel[ACT_SLICE]

            comptime COPY_NA_BLOCKS = (ACT_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_nact, copy_nact](
                all_nact_t,
                nact_1d,
                grid_dim=(COPY_NA_BLOCKS,),
                block_dim=(TPB,),
            )

            # Save prev_deter per timestep for BPTT GRU backward
            var deter_2d = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.deter_buf.unsafe_ptr())
            comptime PREV_D_SLICE = B * DETER
            var all_prev_deter_t = LayoutTensor[
                dtype, Layout.row_major(PREV_D_SLICE), MutAnyOrigin
            ](gpu_state.all_prev_deter_buf.unsafe_ptr() + t * PREV_D_SLICE)
            var deter_1d_src = LayoutTensor[
                dtype, Layout.row_major(PREV_D_SLICE), MutAnyOrigin
            ](gpu_state.deter_buf.unsafe_ptr())

            comptime copy_prev_d = copy_kernel[PREV_D_SLICE]

            comptime COPY_PD_BLOCKS = (PREV_D_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_prev_d, copy_prev_d](
                all_prev_deter_t,
                deter_1d_src,
                grid_dim=(COPY_PD_BLOCKS,),
                block_dim=(TPB,),
            )

            # GRU input projections (with cache for BPTT)
            var stoch_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.stoch_buf.unsafe_ptr())
            var proj_d_2d = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.proj_d_buf.unsafe_ptr())
            var proj_s_2d = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.proj_s_buf.unsafe_ptr())
            var proj_a_2d = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.proj_a_buf.unsafe_ptr())

            comptime DPROJ_CACHE = Self.StateType.RSSMType.DeterProj.CACHE_SIZE
            comptime SPROJ_CACHE = Self.StateType.RSSMType.StochProj.CACHE_SIZE
            comptime APROJ_CACHE = Self.StateType.RSSMType.ActionProj.CACHE_SIZE
            var dproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, DPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_dproj_cache_buf.unsafe_ptr() + t * B * DPROJ_CACHE)
            var sproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, SPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_sproj_cache_buf.unsafe_ptr() + t * B * SPROJ_CACHE)
            var aproj_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, APROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_aproj_cache_buf.unsafe_ptr() + t * B * APROJ_CACHE)

            DProjNet.forward_gpu_with_cache[B](
                ctx,
                deter_2d,
                proj_d_2d,
                gpu_state.deter_proj.params_view(),
                dproj_cache_t,
                gpu_state.ws_deter_proj,
            )
            SProjNet.forward_gpu_with_cache[B](
                ctx,
                stoch_2d,
                proj_s_2d,
                gpu_state.stoch_proj.params_view(),
                sproj_cache_t,
                gpu_state.ws_stoch_proj,
            )
            AProjNet.forward_gpu_with_cache[B](
                ctx,
                norm_act_2d,
                proj_a_2d,
                gpu_state.action_proj.params_view(),
                aproj_cache_t,
                gpu_state.ws_action_proj,
            )

            # Concat [deter, proj_d, proj_s, proj_a]
            comptime GRU_IN = DETER + 3 * HID
            var concat_2d = LayoutTensor[
                dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin
            ](gpu_state.concat_buf.unsafe_ptr())

            comptime run_concat_gru = concat_gru_input_kernel[B, DETER, HID]

            comptime CONCAT_BLOCKS = (B * GRU_IN + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_gru, run_concat_gru](
                concat_2d,
                deter_2d,
                proj_d_2d,
                proj_s_2d,
                proj_a_2d,
                grid_dim=(CONCAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # GRU hidden layer (with cache for BPTT)
            var hidden_2d = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.hidden_out_buf.unsafe_ptr())
            comptime GH_CACHE = Self.StateType.RSSMType.GRUHiddenModel.CACHE_SIZE
            var gh_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, GH_CACHE), MutAnyOrigin
            ](
                gpu_state.all_gru_hidden_cache_buf.unsafe_ptr()
                + t * B * GH_CACHE
            )
            GHNet.forward_gpu_with_cache[B](
                ctx,
                concat_2d,
                hidden_2d,
                gpu_state.gru_hidden.params_view(),
                gh_cache_t,
                gpu_state.ws_gru_hidden,
            )

            # GRU gates (with cache for BPTT)
            var gate_2d = LayoutTensor[
                dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin
            ](gpu_state.gate_out_buf.unsafe_ptr())
            comptime GG_CACHE = Self.StateType.RSSMType.GRUGateModel.CACHE_SIZE
            var gg_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, GG_CACHE), MutAnyOrigin
            ](gpu_state.all_gru_gates_cache_buf.unsafe_ptr() + t * B * GG_CACHE)
            GGNet.forward_gpu_with_cache[B](
                ctx,
                hidden_2d,
                gate_2d,
                gpu_state.gru_gates.params_view(),
                gg_cache_t,
                gpu_state.ws_gru_gates,
            )

            # Save gate_out per timestep for BPTT GRU backward
            comptime GATE_SLICE = B * 3 * DETER
            var all_gate_t = LayoutTensor[
                dtype, Layout.row_major(GATE_SLICE), MutAnyOrigin
            ](gpu_state.all_gate_out_buf.unsafe_ptr() + t * GATE_SLICE)
            var gate_1d = LayoutTensor[
                dtype, Layout.row_major(GATE_SLICE), MutAnyOrigin
            ](gpu_state.gate_out_buf.unsafe_ptr())

            comptime copy_gate = copy_kernel[GATE_SLICE]

            comptime COPY_G_BLOCKS = (GATE_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_gate, copy_gate](
                all_gate_t,
                gate_1d,
                grid_dim=(COPY_G_BLOCKS,),
                block_dim=(TPB,),
            )

            # Apply GRU gating
            var new_deter_2d = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.new_deter_buf.unsafe_ptr())

            comptime run_gru_gate = gru_gate_kernel[B, DETER]

            comptime GATE_BLOCKS = (B * DETER + TPB - 1) // TPB
            ctx.enqueue_function[run_gru_gate, run_gru_gate](
                new_deter_2d,
                deter_2d,
                gate_2d,
                grid_dim=(GATE_BLOCKS,),
                block_dim=(TPB,),
            )

            # Posterior: concat(deter, embed) -> logits
            comptime POST_IN = DETER + STOCH
            var post_in_2d = LayoutTensor[
                dtype, Layout.row_major(B, POST_IN), MutAnyOrigin
            ](gpu_state.post_in_buf.unsafe_ptr())

            comptime run_concat_de = concat_deter_embed_kernel[B, DETER, STOCH]

            comptime DE_BLOCKS = (B * POST_IN + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_de, run_concat_de](
                post_in_2d,
                new_deter_2d,
                embed_2d,
                grid_dim=(DE_BLOCKS,),
                block_dim=(TPB,),
            )

            var post_logits_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.post_logits_buf.unsafe_ptr())
            comptime POST_CACHE = Self.StateType.RSSMType.PostModel.CACHE_SIZE
            var post_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, POST_CACHE), MutAnyOrigin
            ](gpu_state.all_post_cache_buf.unsafe_ptr() + t * B * POST_CACHE)
            PostNet.forward_gpu_with_cache[B](
                ctx,
                post_in_2d,
                post_logits_2d,
                gpu_state.posterior.params_view(),
                post_cache_2d,
                gpu_state.ws_posterior,
            )

            # Prior (with per-timestep cache for BPTT)
            var prior_logits_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.prior_logits_buf.unsafe_ptr())
            comptime PRIOR_CACHE = Self.StateType.RSSMType.PriorModel.CACHE_SIZE
            var prior_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, PRIOR_CACHE), MutAnyOrigin
            ](gpu_state.all_prior_cache_buf.unsafe_ptr() + t * B * PRIOR_CACHE)
            PriorNet.forward_gpu_with_cache[B](
                ctx,
                new_deter_2d,
                prior_logits_2d,
                gpu_state.prior.params_view(),
                prior_cache_2d,
                gpu_state.ws_prior,
            )

            # Categorical sample (posterior)
            var new_stoch_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.new_stoch_buf.unsafe_ptr())
            var post_probs_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.post_probs_buf.unsafe_ptr())

            var cat_seed = Scalar[DType.uint32](
                UInt32(self.train_step_count * BL + t)
                * UInt32(B * Self.stoch_dim * Self.classes + 1)
            )

            comptime run_cat_post = categorical_sample_kernel[
                B,
                Self.stoch_dim,
                Self.classes,
                Self.StateType.RSSMType.UNIMIX,
            ]

            comptime CAT_BLOCKS = (B * Self.stoch_dim + TPB - 1) // TPB
            ctx.enqueue_function[run_cat_post, run_cat_post](
                new_stoch_2d,
                post_probs_2d,
                post_logits_2d,
                cat_seed,
                Scalar[DType.bool](True),
                grid_dim=(CAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # Categorical sample (prior — just for probs, discard output)
            var prior_probs_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.prior_probs_buf.unsafe_ptr())
            var dummy_stoch_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.dummy_stoch_buf.unsafe_ptr())

            comptime run_cat_prior = categorical_sample_kernel[
                B,
                Self.stoch_dim,
                Self.classes,
                Self.StateType.RSSMType.UNIMIX,
            ]

            ctx.enqueue_function[run_cat_prior, run_cat_prior](
                dummy_stoch_2d,
                prior_probs_2d,
                prior_logits_2d,
                cat_seed,
                Scalar[DType.bool](False),
                grid_dim=(CAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # Build feat = concat(deter, stoch)
            var feat_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.feat_buf.unsafe_ptr())

            comptime run_concat_feat = concat_feat_kernel[B, DETER, STOCH]

            comptime FEAT_BLOCKS = (B * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_feat, run_concat_feat](
                feat_2d,
                new_deter_2d,
                new_stoch_2d,
                grid_dim=(FEAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # Store deter/stoch/probs/feat in all_* buffers for imagination
            # Copy new_deter -> all_deter[t]
            comptime DETER_SLICE = B * DETER
            var all_deter_t = LayoutTensor[
                dtype, Layout.row_major(DETER_SLICE), MutAnyOrigin
            ](gpu_state.all_deter_buf.unsafe_ptr() + t * DETER_SLICE)
            var new_deter_1d = LayoutTensor[
                dtype, Layout.row_major(DETER_SLICE), MutAnyOrigin
            ](gpu_state.new_deter_buf.unsafe_ptr())

            comptime copy_deter = copy_kernel[DETER_SLICE]

            comptime COPY_D_BLOCKS = (DETER_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_deter, copy_deter](
                all_deter_t,
                new_deter_1d,
                grid_dim=(COPY_D_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime STOCH_SLICE = B * STOCH
            var all_stoch_t = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.all_stoch_buf.unsafe_ptr() + t * STOCH_SLICE)
            var new_stoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.new_stoch_buf.unsafe_ptr())

            comptime copy_stoch = copy_kernel[STOCH_SLICE]

            comptime COPY_S_BLOCKS = (STOCH_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_stoch, copy_stoch](
                all_stoch_t,
                new_stoch_1d,
                grid_dim=(COPY_S_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime FEAT_SLICE = B * FEAT
            var all_feat_t = LayoutTensor[
                dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin
            ](gpu_state.all_feats_buf.unsafe_ptr() + t * FEAT_SLICE)
            var feat_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin
            ](gpu_state.feat_buf.unsafe_ptr())

            comptime copy_feat = copy_kernel[FEAT_SLICE]

            comptime COPY_F_BLOCKS = (FEAT_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_feat, copy_feat](
                all_feat_t,
                feat_1d,
                grid_dim=(COPY_F_BLOCKS,),
                block_dim=(TPB,),
            )

            # Copy post_probs -> all_post_probs[t]
            var all_post_probs_t = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.all_post_probs_buf.unsafe_ptr() + t * STOCH_SLICE)
            var post_probs_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.post_probs_buf.unsafe_ptr())

            comptime copy_post_probs = copy_kernel[STOCH_SLICE]

            ctx.enqueue_function[copy_post_probs, copy_post_probs](
                all_post_probs_t,
                post_probs_1d,
                grid_dim=(COPY_S_BLOCKS,),
                block_dim=(TPB,),
            )

            # Copy prior_probs -> all_prior_probs[t]
            var all_prior_probs_t = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.all_prior_probs_buf.unsafe_ptr() + t * STOCH_SLICE)
            var prior_probs_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.prior_probs_buf.unsafe_ptr())

            comptime copy_prior_probs = copy_kernel[STOCH_SLICE]

            ctx.enqueue_function[copy_prior_probs, copy_prior_probs](
                all_prior_probs_t,
                prior_probs_1d,
                grid_dim=(COPY_S_BLOCKS,),
                block_dim=(TPB,),
            )

            # Swap deter/stoch for next timestep
            ctx.enqueue_copy(gpu_state.deter_buf, gpu_state.new_deter_buf)
            ctx.enqueue_copy(gpu_state.stoch_buf, gpu_state.new_stoch_buf)

        ctx.synchronize()
        var _pt2 = perf_counter_ns()  # end RSSM observe

        # ── 3. Full BPTT Backward (autodiff) ────────────────────────────────
        # Replaces per-timestep head backward + separate BPTT loop with a
        # single unified reverse pass matching the tested CPU autodiff code.
        var wm_losses = self._gpu_bptt_autodiff(
            ctx,
            gpu_state,
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_dones,
        )
        var total_wm_loss = wm_losses[0]
        var obs_loss = wm_losses[1]
        var rew_loss = wm_losses[2]
        var cont_loss = wm_losses[3]
        var dyn_kl_total = wm_losses[4]
        var rep_kl_total = wm_losses[5]

        ctx.synchronize()
        var _pt3 = perf_counter_ns()  # end BPTT backward

        # ── 4. World model gradient clipping + optimizer step ──────────────
        var grad_norm_max = Scalar[dtype](self.max_grad_norm)
        _clip_grads_gpu[EncNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.encoder.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[PostNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.posterior.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[PriorNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.prior.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[DecNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.decoder.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[RewNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.reward_head.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[ContNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.continue_head.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[DProjNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.deter_proj.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[SProjNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.stoch_proj.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[AProjNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.action_proj.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[GHNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.gru_hidden.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        _clip_grads_gpu[GGNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.gru_gates.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        gpu_state.encoder.optimizer_step(ctx)
        gpu_state.posterior.optimizer_step(ctx)
        gpu_state.prior.optimizer_step(ctx)
        gpu_state.decoder.optimizer_step(ctx)
        gpu_state.reward_head.optimizer_step(ctx)
        gpu_state.continue_head.optimizer_step(ctx)
        gpu_state.deter_proj.optimizer_step(ctx)
        gpu_state.stoch_proj.optimizer_step(ctx)
        gpu_state.action_proj.optimizer_step(ctx)
        gpu_state.gru_hidden.optimizer_step(ctx)
        gpu_state.gru_gates.optimizer_step(ctx)

        ctx.synchronize()
        var _pt4 = perf_counter_ns()  # end WM optim

        # ── 5. Imagination rollout ───────────────────────────────────────
        # Initialize from all observed states: all_feats[BL*B] -> imag buffers
        # Copy all_deter[IB*DETER] into first half of imag_deter[2*IB*DETER]
        comptime IB_DETER = IB * DETER
        var all_deter_1d = LayoutTensor[
            dtype, Layout.row_major(IB_DETER), MutAnyOrigin
        ](gpu_state.all_deter_buf.unsafe_ptr())
        var imag_deter_init = LayoutTensor[
            dtype, Layout.row_major(IB_DETER), MutAnyOrigin
        ](gpu_state.imag_deter_buf.unsafe_ptr())

        comptime copy_all_deter = copy_kernel[IB_DETER]

        comptime INIT_D_BLOCKS = (IB_DETER + TPB - 1) // TPB
        ctx.enqueue_function[copy_all_deter, copy_all_deter](
            imag_deter_init,
            all_deter_1d,
            grid_dim=(INIT_D_BLOCKS,),
            block_dim=(TPB,),
        )

        comptime IB_STOCH = IB * STOCH
        var all_stoch_1d = LayoutTensor[
            dtype, Layout.row_major(IB_STOCH), MutAnyOrigin
        ](gpu_state.all_stoch_buf.unsafe_ptr())
        var imag_stoch_init = LayoutTensor[
            dtype, Layout.row_major(IB_STOCH), MutAnyOrigin
        ](gpu_state.imag_stoch_buf.unsafe_ptr())

        comptime copy_all_stoch = copy_kernel[IB_STOCH]

        comptime INIT_S_BLOCKS = (IB_STOCH + TPB - 1) // TPB
        ctx.enqueue_function[copy_all_stoch, copy_all_stoch](
            imag_stoch_init,
            all_stoch_1d,
            grid_dim=(INIT_S_BLOCKS,),
            block_dim=(TPB,),
        )

        # Zero actor/critic grads
        ctx.enqueue_memset(gpu_state.actor.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.critic.grads_buf, 0)

        # Imagination uses ping-pong buffers:
        # Even steps: read from offset 0, write to offset IB*DIM
        # Odd steps: read from offset IB*DIM, write to offset 0
        for h in range(HORIZON):
            var read_off = (h % 2) * IB
            var write_off = ((h + 1) % 2) * IB

            # Build feat from current deter/stoch
            var imag_deter_2d = LayoutTensor[
                dtype, Layout.row_major(IB, DETER), MutAnyOrigin
            ](gpu_state.imag_deter_buf.unsafe_ptr() + read_off * DETER)
            var imag_stoch_2d = LayoutTensor[
                dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
            ](gpu_state.imag_stoch_buf.unsafe_ptr() + read_off * STOCH)
            var imag_feat_2d = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](gpu_state.imag_feat_buf.unsafe_ptr())

            comptime run_concat_imag_feat = concat_feat_kernel[IB, DETER, STOCH]

            # Save deter/stoch for multi-step actor-critic
            comptime IMAG_D_SLICE = IB * DETER
            comptime IMAG_S_SLICE = IB * STOCH
            var save_d = LayoutTensor[
                dtype, Layout.row_major(IMAG_D_SLICE), MutAnyOrigin
            ](gpu_state.imag_all_deter_buf.unsafe_ptr() + h * IMAG_D_SLICE)
            var save_s = LayoutTensor[
                dtype, Layout.row_major(IMAG_S_SLICE), MutAnyOrigin
            ](gpu_state.imag_all_stoch_buf.unsafe_ptr() + h * IMAG_S_SLICE)
            var src_d = LayoutTensor[
                dtype, Layout.row_major(IMAG_D_SLICE), MutAnyOrigin
            ](gpu_state.imag_deter_buf.unsafe_ptr() + read_off * DETER)
            var src_s = LayoutTensor[
                dtype, Layout.row_major(IMAG_S_SLICE), MutAnyOrigin
            ](gpu_state.imag_stoch_buf.unsafe_ptr() + read_off * STOCH)

            comptime copy_imag_d = copy_kernel[IMAG_D_SLICE]

            comptime copy_imag_s = copy_kernel[IMAG_S_SLICE]

            comptime COPY_ID_BLOCKS = (IMAG_D_SLICE + TPB - 1) // TPB
            comptime COPY_IS_BLOCKS = (IMAG_S_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_imag_d, copy_imag_d](
                save_d,
                src_d,
                grid_dim=(COPY_ID_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[copy_imag_s, copy_imag_s](
                save_s,
                src_s,
                grid_dim=(COPY_IS_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime IB_FEAT_BLOCKS = (IB * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_imag_feat, run_concat_imag_feat](
                imag_feat_2d,
                imag_deter_2d,
                imag_stoch_2d,
                grid_dim=(IB_FEAT_BLOCKS,),
                block_dim=(TPB,),
            )

            # Actor forward -> sample actions
            # Actor forward with cache for dynamics backprop
            comptime ACTOR_CACHE_SZ = Self.StateType.ActorModel.CACHE_SIZE
            comptime ACTOR_OUT_DIM_I = Self.StateType.ActorModel.OUT_DIM
            var actor_out_2d = LayoutTensor[
                dtype,
                Layout.row_major(IB, ACTOR_OUT_DIM_I),
                MutAnyOrigin,
            ](gpu_state.actor_out_buf.unsafe_ptr())
            var imag_actor_cache_h = LayoutTensor[
                dtype, Layout.row_major(IB, ACTOR_CACHE_SZ), MutAnyOrigin
            ](
                gpu_state.imag_actor_cache_buf.unsafe_ptr()
                + h * IB * ACTOR_CACHE_SZ
            )
            Self.ActorNet.forward_gpu_with_cache[IB](
                ctx,
                imag_feat_2d,
                actor_out_2d,
                gpu_state.actor.params_view(),
                imag_actor_cache_h,
                gpu_state.ws_actor,
            )

            # Save actor_out per step for reparameterization backward
            comptime AO_SLICE = IB * 2 * ACT
            var save_ao = LayoutTensor[
                dtype, Layout.row_major(AO_SLICE), MutAnyOrigin
            ](gpu_state.imag_actor_out_save_buf.unsafe_ptr() + h * AO_SLICE)
            var src_ao = LayoutTensor[
                dtype, Layout.row_major(AO_SLICE), MutAnyOrigin
            ](gpu_state.actor_out_buf.unsafe_ptr())

            comptime copy_ao = copy_kernel[AO_SLICE]

            comptime COPY_AO_BLOCKS = (AO_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_ao, copy_ao](
                save_ao,
                src_ao,
                grid_dim=(COPY_AO_BLOCKS,),
                block_dim=(TPB,),
            )

            # Sample tanh-normal actions + log probs
            var actions_2d = LayoutTensor[
                dtype, Layout.row_major(IB, ACT), MutAnyOrigin
            ](gpu_state.imag_actions_buf.unsafe_ptr())
            var log_probs_1d = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_log_probs_buf.unsafe_ptr())

            var act_seed = Scalar[DType.uint32](
                UInt32(self.train_step_count * HORIZON + h)
                * UInt32(IB * ACT + 1)
            )

            comptime run_sample_actions = tanh_normal_sample_kernel[IB, ACT]

            comptime SAMPLE_BLOCKS = (IB + TPB - 1) // TPB
            ctx.enqueue_function[run_sample_actions, run_sample_actions](
                actions_2d,
                log_probs_1d,
                actor_out_2d,
                act_seed,
                grid_dim=(SAMPLE_BLOCKS,),
                block_dim=(TPB,),
            )

            # Save actions per horizon step for actor-critic training
            comptime IMAG_A_SLICE = IB * ACT
            var save_a = LayoutTensor[
                dtype, Layout.row_major(IMAG_A_SLICE), MutAnyOrigin
            ](gpu_state.imag_all_actions_buf.unsafe_ptr() + h * IMAG_A_SLICE)
            var src_a = LayoutTensor[
                dtype, Layout.row_major(IMAG_A_SLICE), MutAnyOrigin
            ](gpu_state.imag_actions_buf.unsafe_ptr())

            comptime copy_imag_a = copy_kernel[IMAG_A_SLICE]

            comptime COPY_IA_BLOCKS = (IMAG_A_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_imag_a, copy_imag_a](
                save_a,
                src_a,
                grid_dim=(COPY_IA_BLOCKS,),
                block_dim=(TPB,),
            )

            # Predict reward from feat
            var rew_logits_2d = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.rew_logits_buf.unsafe_ptr())
            comptime IMAG_REW_CACHE_SZ = Self.StateType.RSSMType.RewModel.CACHE_SIZE
            var imag_rew_cache_h = LayoutTensor[
                dtype, Layout.row_major(IB, IMAG_REW_CACHE_SZ), MutAnyOrigin
            ](
                gpu_state.imag_rew_cache_buf.unsafe_ptr()
                + h * IB * IMAG_REW_CACHE_SZ
            )
            RewNet.forward_gpu_with_cache[IB](
                ctx,
                imag_feat_2d,
                rew_logits_2d,
                gpu_state.reward_head.params_view(),
                imag_rew_cache_h,
                gpu_state.ws_reward,
            )

            # Decode reward values
            var rewards_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_rewards_buf.unsafe_ptr() + h * IB)
            var bins_1d = LayoutTensor[
                dtype, Layout.row_major(BINS), MutAnyOrigin
            ](gpu_state.bins_buf.unsafe_ptr())

            comptime run_decode_reward = decode_value_kernel[IB, BINS]

            ctx.enqueue_function[run_decode_reward, run_decode_reward](
                rewards_h,
                rew_logits_2d,
                bins_1d,
                Scalar[DType.bool](True),
                grid_dim=(SAMPLE_BLOCKS,),
                block_dim=(TPB,),
            )

            # Predict continue
            var cont_out_2d = LayoutTensor[
                dtype, Layout.row_major(IB, 1), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            ContNet.forward_gpu[IB](
                ctx,
                imag_feat_2d,
                cont_out_2d,
                gpu_state.continue_head.params_view(),
                gpu_state.ws_continue,
            )

            # Apply sigmoid to continue output
            var cont_1d_in = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            var continues_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_continues_buf.unsafe_ptr() + h * IB)

            comptime run_sigmoid = sigmoid_kernel[IB]

            ctx.enqueue_function[run_sigmoid, run_sigmoid](
                continues_h,
                cont_1d_in,
                grid_dim=(SAMPLE_BLOCKS,),
                block_dim=(TPB,),
            )

            # Critic value prediction
            var critic_logits_2d = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.critic_logits_buf.unsafe_ptr())
            Self.CriticNet.forward_gpu[IB](
                ctx,
                imag_feat_2d,
                critic_logits_2d,
                gpu_state.critic.params_view(),
                gpu_state.ws_critic,
            )

            var values_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_values_buf.unsafe_ptr() + h * IB)

            comptime run_decode_value = decode_value_kernel[IB, BINS]

            ctx.enqueue_function[run_decode_value, run_decode_value](
                values_h,
                critic_logits_2d,
                bins_1d,
                Scalar[DType.bool](True),
                grid_dim=(SAMPLE_BLOCKS,),
                block_dim=(TPB,),
            )

            # RSSM imagine step (next deter/stoch) — skip last horizon step
            if h < HORIZON - 1:
                var next_deter_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, DETER), MutAnyOrigin
                ](gpu_state.imag_deter_buf.unsafe_ptr() + write_off * DETER)
                var next_stoch_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](gpu_state.imag_stoch_buf.unsafe_ptr() + write_off * STOCH)

                # ── GRU core forward on GPU (IB-sized) ────────────────

                # Action normalize
                var imag_norm_act_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, ACT), MutAnyOrigin
                ](gpu_state.imag_norm_act_buf.unsafe_ptr())

                comptime run_imag_act_norm = action_normalize_kernel[IB, ACT]

                comptime IMAG_NORM_BLOCKS = (IB * ACT + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_act_norm, run_imag_act_norm](
                    imag_norm_act_2d,
                    actions_2d,
                    grid_dim=(IMAG_NORM_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Input projections
                var imag_proj_d_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, HID), MutAnyOrigin
                ](gpu_state.imag_proj_d_buf.unsafe_ptr())
                var imag_proj_s_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, HID), MutAnyOrigin
                ](gpu_state.imag_proj_s_buf.unsafe_ptr())
                var imag_proj_a_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, HID), MutAnyOrigin
                ](gpu_state.imag_proj_a_buf.unsafe_ptr())

                DProjNet.forward_gpu[IB](
                    ctx,
                    imag_deter_2d,
                    imag_proj_d_2d,
                    gpu_state.deter_proj.params_view(),
                    gpu_state.ws_deter_proj,
                )
                SProjNet.forward_gpu[IB](
                    ctx,
                    imag_stoch_2d,
                    imag_proj_s_2d,
                    gpu_state.stoch_proj.params_view(),
                    gpu_state.ws_stoch_proj,
                )
                comptime IMAG_APROJ_CACHE_SZ = Self.StateType.RSSMType.ActionProj.CACHE_SIZE
                var imag_aproj_cache_h = LayoutTensor[
                    dtype,
                    Layout.row_major(IB, IMAG_APROJ_CACHE_SZ),
                    MutAnyOrigin,
                ](
                    gpu_state.imag_aproj_cache_buf.unsafe_ptr()
                    + h * IB * IMAG_APROJ_CACHE_SZ
                )
                AProjNet.forward_gpu_with_cache[IB](
                    ctx,
                    imag_norm_act_2d,
                    imag_proj_a_2d,
                    gpu_state.action_proj.params_view(),
                    imag_aproj_cache_h,
                    gpu_state.ws_action_proj,
                )

                # Concat [deter, proj_d, proj_s, proj_a]
                comptime GRU_IN_IB = DETER + 3 * HID
                var imag_concat_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, GRU_IN_IB), MutAnyOrigin
                ](gpu_state.imag_concat_buf.unsafe_ptr())

                comptime run_imag_concat_gru = concat_gru_input_kernel[
                    IB, DETER, HID
                ]

                comptime IMAG_CONCAT_BLOCKS = (IB * GRU_IN_IB + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_concat_gru, run_imag_concat_gru](
                    imag_concat_2d,
                    imag_deter_2d,
                    imag_proj_d_2d,
                    imag_proj_s_2d,
                    imag_proj_a_2d,
                    grid_dim=(IMAG_CONCAT_BLOCKS,),
                    block_dim=(TPB,),
                )

                # GRU hidden layer (with cache for dynamics backprop)
                comptime IMAG_GH_CACHE_SZ = Self.StateType.RSSMType.GRUHiddenModel.CACHE_SIZE
                var imag_hidden_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, DETER), MutAnyOrigin
                ](gpu_state.imag_hidden_buf.unsafe_ptr())
                var imag_gh_cache_h = LayoutTensor[
                    dtype, Layout.row_major(IB, IMAG_GH_CACHE_SZ), MutAnyOrigin
                ](
                    gpu_state.imag_gh_cache_buf.unsafe_ptr()
                    + h * IB * IMAG_GH_CACHE_SZ
                )
                GHNet.forward_gpu_with_cache[IB](
                    ctx,
                    imag_concat_2d,
                    imag_hidden_2d,
                    gpu_state.gru_hidden.params_view(),
                    imag_gh_cache_h,
                    gpu_state.ws_gru_hidden,
                )

                # GRU gates (with cache for dynamics backprop)
                comptime IMAG_GG_CACHE_SZ = Self.StateType.RSSMType.GRUGateModel.CACHE_SIZE
                var imag_gate_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, 3 * DETER), MutAnyOrigin
                ](gpu_state.imag_gate_buf.unsafe_ptr())
                var imag_gg_cache_h = LayoutTensor[
                    dtype, Layout.row_major(IB, IMAG_GG_CACHE_SZ), MutAnyOrigin
                ](
                    gpu_state.imag_gg_cache_buf.unsafe_ptr()
                    + h * IB * IMAG_GG_CACHE_SZ
                )
                GGNet.forward_gpu_with_cache[IB](
                    ctx,
                    imag_hidden_2d,
                    imag_gate_2d,
                    gpu_state.gru_gates.params_view(),
                    imag_gg_cache_h,
                    gpu_state.ws_gru_gates,
                )

                # Save gate_out for GRU backward
                comptime IMAG_GATE_SLICE = IB * 3 * DETER
                var save_gate = LayoutTensor[
                    dtype, Layout.row_major(IMAG_GATE_SLICE), MutAnyOrigin
                ](
                    gpu_state.imag_gate_out_save_buf.unsafe_ptr()
                    + h * IMAG_GATE_SLICE
                )
                var src_gate = LayoutTensor[
                    dtype, Layout.row_major(IMAG_GATE_SLICE), MutAnyOrigin
                ](gpu_state.imag_gate_buf.unsafe_ptr())

                comptime copy_imag_gate = copy_kernel[IMAG_GATE_SLICE]

                comptime COPY_IG_BLOCKS = (IMAG_GATE_SLICE + TPB - 1) // TPB
                ctx.enqueue_function[copy_imag_gate, copy_imag_gate](
                    save_gate,
                    src_gate,
                    grid_dim=(COPY_IG_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Apply GRU gating -> next_deter
                comptime run_imag_gru_gate = gru_gate_kernel[IB, DETER]

                comptime IMAG_GATE_BLOCKS = (IB * DETER + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_gru_gate, run_imag_gru_gate](
                    next_deter_2d,
                    imag_deter_2d,
                    imag_gate_2d,
                    grid_dim=(IMAG_GATE_BLOCKS,),
                    block_dim=(TPB,),
                )

                # ── Prior: deter -> logits -> sample stoch ────────────

                var imag_prior_logits_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](gpu_state.imag_prior_logits_buf.unsafe_ptr())
                PriorNet.forward_gpu[IB](
                    ctx,
                    next_deter_2d,
                    imag_prior_logits_2d,
                    gpu_state.prior.params_view(),
                    gpu_state.ws_prior,
                )

                # Categorical sample from prior
                var imag_prior_probs_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](gpu_state.imag_prior_probs_buf.unsafe_ptr())

                var imag_cat_seed = Scalar[DType.uint32](
                    UInt32(self.train_step_count * HORIZON + h + 1000)
                    * UInt32(IB * Self.stoch_dim * Self.classes + 1)
                )

                comptime run_imag_cat_prior = categorical_sample_kernel[
                    IB,
                    Self.stoch_dim,
                    Self.classes,
                    Self.StateType.RSSMType.UNIMIX,
                ]

                comptime IMAG_CAT_BLOCKS = (
                    IB * Self.stoch_dim + TPB - 1
                ) // TPB
                ctx.enqueue_function[run_imag_cat_prior, run_imag_cat_prior](
                    next_stoch_2d,
                    imag_prior_probs_2d,
                    imag_prior_logits_2d,
                    imag_cat_seed,
                    Scalar[DType.bool](True),
                    grid_dim=(IMAG_CAT_BLOCKS,),
                    block_dim=(TPB,),
                )

        ctx.synchronize()
        var _pt5 = perf_counter_ns()  # end imagination

        # ── 6. Lambda returns (GPU) ─────────────────────────────────────
        # Compute lambda returns entirely on GPU, only download 2 scalars
        # for EMA normalization tracking.
        var returns_2d = LayoutTensor[
            dtype, Layout.row_major(HORIZON, IB), MutAnyOrigin
        ](gpu_state.imag_returns_buf.unsafe_ptr())
        var rewards_2d = LayoutTensor[
            dtype, Layout.row_major(HORIZON, IB), MutAnyOrigin
        ](gpu_state.imag_rewards_buf.unsafe_ptr())
        var values_2d = LayoutTensor[
            dtype, Layout.row_major(HORIZON, IB), MutAnyOrigin
        ](gpu_state.imag_values_buf.unsafe_ptr())
        var continues_2d = LayoutTensor[
            dtype, Layout.row_major(HORIZON, IB), MutAnyOrigin
        ](gpu_state.imag_continues_buf.unsafe_ptr())

        comptime run_lambda_returns = lambda_returns_kernel[HORIZON, IB]

        comptime LAMBDA_BLOCKS = (IB + TPB - 1) // TPB
        ctx.enqueue_function[run_lambda_returns, run_lambda_returns](
            returns_2d,
            rewards_2d,
            values_2d,
            continues_2d,
            Scalar[dtype](self.gamma),
            Scalar[dtype](self.lambda_),
            grid_dim=(LAMBDA_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 6b. Compute return scale for advantage normalization ─────────
        comptime RETURNS_SIZE = HORIZON * IB
        comptime ENCODE_BLOCKS = (IB + TPB - 1) // TPB
        var returns_flat = LayoutTensor[
            dtype, Layout.row_major(RETURNS_SIZE), MutAnyOrigin
        ](gpu_state.imag_returns_buf.unsafe_ptr())
        var minmax_2 = LayoutTensor[dtype, Layout.row_major(2), MutAnyOrigin](
            gpu_state.returns_minmax_buf.unsafe_ptr()
        )

        comptime run_minmax = min_max_reduce_kernel[RETURNS_SIZE, TPB]

        ctx.enqueue_function[run_minmax, run_minmax](
            minmax_2,
            returns_flat,
            grid_dim=(1,),
            block_dim=(TPB,),
        )

        var host_minmax = gpu_state.host_minmax_buf
        ctx.enqueue_copy(host_minmax, gpu_state.returns_minmax_buf)
        ctx.synchronize()

        var lo = Float64(host_minmax[0])
        var hi = Float64(host_minmax[1])
        self.state.return_ema_lo = (
            1.0 - self.return_norm_rate
        ) * self.state.return_ema_lo + self.return_norm_rate * lo
        self.state.return_ema_hi = (
            1.0 - self.return_norm_rate
        ) * self.state.return_ema_hi + self.return_norm_rate * hi
        var scale = self.state.return_ema_hi - self.state.return_ema_lo
        if scale < 1.0:
            scale = 1.0

        # ── 6c. Compute advantages: (return - value) / rscale ─────────
        # Reference DreamerV3: adv = (ret - tarval) / rscale
        # Both returns and values in original scale. Divide by return scale.
        comptime ADV_TOTAL = (HORIZON - 1) * IB
        var inv_rscale = Scalar[dtype](1.0 / scale)
        for h in range(HORIZON - 1):
            var ret_h = LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin](
                gpu_state.imag_returns_buf.unsafe_ptr() + h * IB
            )
            var val_h = LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin](
                gpu_state.imag_values_buf.unsafe_ptr() + h * IB
            )
            var adv_h = LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin](
                gpu_state.imag_rewards_buf.unsafe_ptr() + h * IB
            )

            comptime run_adv_precompute = advantage_kernel[IB]

            ctx.enqueue_function[run_adv_precompute, run_adv_precompute](
                adv_h,
                ret_h,
                val_h,
                grid_dim=(ENCODE_BLOCKS,),
                block_dim=(TPB,),
            )

        # Scale advantages by 1/rscale
        var all_adv = LayoutTensor[
            dtype, Layout.row_major(ADV_TOTAL), MutAnyOrigin
        ](gpu_state.imag_rewards_buf.unsafe_ptr())

        comptime run_scale_adv = normalize_returns_elementwise_kernel[ADV_TOTAL]

        # adv = (adv - 0) * inv_rscale = adv / rscale
        ctx.enqueue_function[run_scale_adv, run_scale_adv](
            all_adv,
            Scalar[dtype](0.0),  # no offset
            inv_rscale,
            grid_dim=((ADV_TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        # Normalize advantages: (adv - mean) / max(std, 1.0)
        comptime run_norm_adv = normalize_advantages_kernel[ADV_TOTAL, TPB]

        ctx.enqueue_function[run_norm_adv, run_norm_adv](
            all_adv,
            grid_dim=(1,),
            block_dim=(TPB,),
        )

        # ── 6d. Normalize returns IN-PLACE for critic training ─────────────
        # Advantages already computed from raw returns above.
        # Now normalize returns so critic trains on symlog(normalized_return).
        # This matches reference: critic sees [0,1]-scale targets.
        comptime run_norm_returns = normalize_returns_elementwise_kernel[
            RETURNS_SIZE
        ]

        ctx.enqueue_function[run_norm_returns, run_norm_returns](
            returns_flat,
            Scalar[dtype](self.state.return_ema_lo),
            Scalar[dtype](1.0 / scale),
            grid_dim=((RETURNS_SIZE + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        ctx.synchronize()
        var _pt6 = perf_counter_ns()  # end lambda returns

        # ── 7. Multi-step Critic + Actor training ──────────────────────
        ctx.enqueue_memset(gpu_state.actor.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.critic.grads_buf, 0)

        comptime ACTOR_OUT_DIM = Self.StateType.ActorModel.OUT_DIM
        comptime IB_FEAT_BLOCKS2 = (IB * FEAT + TPB - 1) // TPB
        comptime SAMPLE_BLOCKS2 = (IB + TPB - 1) // TPB
        comptime SYMLOG_RET_BLOCKS = (IB + TPB - 1) // TPB
        var inv_ib = Scalar[dtype](1.0 / Float64(IB * (HORIZON - 1)))
        var entropy_coef = Scalar[dtype](self.actor_entropy)
        var bins_1d_ac = LayoutTensor[
            dtype, Layout.row_major(BINS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())

        for h in range(HORIZON - 1):
            # Read saved deter/stoch for imagination step h
            var h_deter_2d = LayoutTensor[
                dtype, Layout.row_major(IB, DETER), MutAnyOrigin
            ](gpu_state.imag_all_deter_buf.unsafe_ptr() + h * IB * DETER)
            var h_stoch_2d = LayoutTensor[
                dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
            ](gpu_state.imag_all_stoch_buf.unsafe_ptr() + h * IB * STOCH)
            var h_feat_2d = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](gpu_state.imag_feat_buf.unsafe_ptr())

            comptime run_concat_h_feat = concat_feat_kernel[IB, DETER, STOCH]

            ctx.enqueue_function[run_concat_h_feat, run_concat_h_feat](
                h_feat_2d,
                h_deter_2d,
                h_stoch_2d,
                grid_dim=(IB_FEAT_BLOCKS2,),
                block_dim=(TPB,),
            )

            # Returns and values at step h
            var returns_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_returns_buf.unsafe_ptr() + h * IB)
            var values_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_values_buf.unsafe_ptr() + h * IB)

            # ── Critic: forward, gradient, backward (accumulate grads) ──
            var critic_logits_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.critic_logits_buf.unsafe_ptr())
            var critic_cache_h = LayoutTensor[
                dtype,
                Layout.row_major(IB, Self.StateType.CriticModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu_state.critic_cache_buf.unsafe_ptr())
            Self.CriticNet.forward_gpu_with_cache[IB](
                ctx,
                h_feat_2d,
                critic_logits_h,
                gpu_state.critic.params_view(),
                critic_cache_h,
                gpu_state.ws_critic,
            )

            # symlog(returns[h]) -> two-hot encode -> critic CE gradient
            var symlog_ret_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.symlog_returns_buf.unsafe_ptr())

            comptime run_symlog_ret_h = symlog_kernel[IB]

            ctx.enqueue_function[run_symlog_ret_h, run_symlog_ret_h](
                symlog_ret_h,
                returns_h,
                grid_dim=(SYMLOG_RET_BLOCKS,),
                block_dim=(TPB,),
            )

            var two_hot_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.two_hot_targets_buf.unsafe_ptr())

            comptime run_two_hot_h = two_hot_encode_kernel[IB, BINS]

            ctx.enqueue_function[run_two_hot_h, run_two_hot_h](
                two_hot_h,
                symlog_ret_h,
                bins_1d_ac,
                grid_dim=(ENCODE_BLOCKS,),
                block_dim=(TPB,),
            )

            var critic_grad_h = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.critic_grad_buf.unsafe_ptr())

            comptime run_critic_grad_h = two_hot_ce_grad_kernel[IB, BINS]

            ctx.enqueue_function[run_critic_grad_h, run_critic_grad_h](
                critic_grad_h,
                critic_logits_h,
                two_hot_h,
                inv_ib,
                grid_dim=(ENCODE_BLOCKS,),
                block_dim=(TPB,),
            )

            var critic_grad_in_h = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](gpu_state.critic_grad_in_buf.unsafe_ptr())
            var critic_grads = gpu_state.critic.grads_view()
            Self.CriticNet.backward_gpu[IB](
                ctx,
                critic_grad_h,
                critic_grad_in_h,
                gpu_state.critic.params_view(),
                critic_cache_h,
                critic_grads,
                gpu_state.ws_critic,
            )

            # ── Actor: REINFORCE with saved imagination actions ────────────
            # Reference DreamerV3 uses pure REINFORCE (not dynamics backprop).
            # All features/advantages are stop-gradiented. Gradient flows
            # only through log_prob(saved_action | current_policy).
            var actor_out_h = LayoutTensor[
                dtype,
                Layout.row_major(IB, ACTOR_OUT_DIM),
                MutAnyOrigin,
            ](gpu_state.actor_out_buf.unsafe_ptr())
            var actor_cache_h = LayoutTensor[
                dtype,
                Layout.row_major(IB, Self.StateType.ActorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu_state.actor_cache_buf.unsafe_ptr())
            Self.ActorNet.forward_gpu_with_cache[IB](
                ctx,
                h_feat_2d,
                actor_out_h,
                gpu_state.actor.params_view(),
                actor_cache_h,
                gpu_state.ws_actor,
            )

            # Read pre-computed normalized advantages for step h
            var adv_h = LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin](
                gpu_state.imag_rewards_buf.unsafe_ptr() + h * IB
            )

            # Use saved imagination actions (same actions that generated returns)
            var actions_h = LayoutTensor[
                dtype, Layout.row_major(IB, ACT), MutAnyOrigin
            ](gpu_state.imag_all_actions_buf.unsafe_ptr() + h * IB * ACT)

            # REINFORCE gradient: -advantage * d(log_prob)/d(actor_out)
            var actor_grad_h = LayoutTensor[
                dtype, Layout.row_major(IB, ACTOR_OUT_DIM), MutAnyOrigin
            ](gpu_state.actor_grad_buf.unsafe_ptr())

            comptime run_reinforce_h = reinforce_grad_kernel[IB, ACT]

            ctx.enqueue_function[run_reinforce_h, run_reinforce_h](
                actor_grad_h,
                actor_out_h,
                actions_h,
                adv_h,
                inv_ib,
                entropy_coef,
                grid_dim=(SAMPLE_BLOCKS2,),
                block_dim=(TPB,),
            )

            # Actor backward (accumulates into actor grads)
            var actor_grad_in_h = LayoutTensor[
                dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
            ](gpu_state.actor_grad_in_buf.unsafe_ptr())
            var actor_grads = gpu_state.actor.grads_view()
            Self.ActorNet.backward_gpu[IB](
                ctx,
                actor_grad_h,
                actor_grad_in_h,
                gpu_state.actor.params_view(),
                actor_cache_h,
                actor_grads,
                gpu_state.ws_actor,
            )

        # ── Critic gradient clipping + optimizer step ─────────────────────
        _clip_grads_gpu[Self.CriticNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.critic.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        gpu_state.critic.optimizer_step(ctx)

        # ── Actor gradient clipping + optimizer step ─────────────────────
        _clip_grads_gpu[Self.ActorNet.MODEL.PARAM_SIZE](
            ctx,
            gpu_state.actor.grads_view(),
            gpu_state.grad_partial_sums_buf,
            grad_norm_max,
        )
        gpu_state.actor.optimizer_step(ctx)

        ctx.synchronize()
        var _pt7 = perf_counter_ns()  # end critic + actor

        # ── 8. Slow critic EMA update ──────────────────────────────────
        gpu_state.slow_critic.soft_update_from_gpu(
            gpu_state.critic,
            Float64(self.slow_critic_tau),
            ctx,
        )

        ctx.synchronize()
        var _pt8 = perf_counter_ns()  # end EMA + sync

        # ── Phase timing report (every diag_every steps) ────────────────
        if self.diag_every > 0 and self.train_step_count % self.diag_every == 0:
            var _total = Float64(_pt8 - _pt0) / 1e6
            print(
                "  [timing] step="
                + String(self.train_step_count)
                + " total="
                + String(_total)[:7]
                + "ms | upload="
                + String(Float64(_pt1 - _pt0) / 1e6)[:6]
                + " observe="
                + String(Float64(_pt2 - _pt1) / 1e6)[:6]
                + " bptt="
                + String(Float64(_pt3 - _pt2) / 1e6)[:6]
                + " wm_opt="
                + String(Float64(_pt4 - _pt3) / 1e6)[:6]
                + " imagine="
                + String(Float64(_pt5 - _pt4) / 1e6)[:6]
                + " returns="
                + String(Float64(_pt6 - _pt5) / 1e6)[:6]
                + " ac_train="
                + String(Float64(_pt7 - _pt6) / 1e6)[:6]
                + " ema="
                + String(Float64(_pt8 - _pt7) / 1e6)[:6]
            )

        # ── Diagnostics ──────────────────────────────────────────────────
        if self.diag_every > 0 and self.train_step_count % self.diag_every == 0:
            # Download a few diagnostic values
            var diag_imag = gpu_state.host_diag_imag_buf

            # Download advantages (stored in imag_rewards_buf after actor step)
            ctx.enqueue_copy(
                diag_imag, gpu_state.imag_rewards_buf
            )  # now has advantages
            ctx.synchronize()

            # Compute stats across ALL advantages (not just h=0)
            comptime DIAG_ADV_N = (HORIZON - 1) * IB
            var avg_adv = Float64(0)
            var adv_min = Float64(1e30)
            var adv_max = Float64(-1e30)
            for i in range(DIAG_ADV_N):
                var v = Float64(diag_imag[i])
                avg_adv += v
                if v < adv_min:
                    adv_min = v
                if v > adv_max:
                    adv_max = v
            avg_adv /= Float64(DIAG_ADV_N)
            var adv_var = Float64(0)
            for i in range(DIAG_ADV_N):
                var d = Float64(diag_imag[i]) - avg_adv
                adv_var += d * d
            adv_var /= Float64(DIAG_ADV_N)

            # Download values and compute avg before buffer is reused
            ctx.enqueue_copy(diag_imag, gpu_state.imag_values_buf)
            ctx.synchronize()
            var avg_val = Float64(0)
            for i in range(IB):
                avg_val += Float64(diag_imag[i])
            avg_val /= Float64(IB)

            # Check actor gradient L2 norm
            comptime ACTOR_PS = Self.StateType.ActorModel.PARAM_SIZE
            var diag_actor = gpu_state.host_diag_actor_buf
            ctx.enqueue_copy(diag_actor, gpu_state.actor.grads_buf)
            ctx.synchronize()
            var actor_grad_norm = Float64(0)
            for i in range(ACTOR_PS):
                var g = Float64(diag_actor[i])
                actor_grad_norm += g * g
            actor_grad_norm = sqrt(actor_grad_norm)

            # Check actor first layer weights L2 norm (to see if they change)
            ctx.enqueue_copy(diag_actor, gpu_state.actor.params_buf)
            ctx.synchronize()
            var actor_param_norm = Float64(0)
            for i in range(min(FEAT * Self.units, ACTOR_PS)):
                var p = Float64(diag_actor[i])
                actor_param_norm += p * p
            actor_param_norm = sqrt(actor_param_norm)

            # print(
            #     "  [diag] step="
            #     + String(self.train_step_count)
            #     + " wm_loss="
            #     + String(total_wm_loss)[:7]
            #     + " obs="
            #     + String(obs_loss)[:6]
            #     + " rew="
            #     + String(rew_loss)[:6]
            #     + " cont="
            #     + String(cont_loss)[:6]
            #     + " kl="
            #     + String(dyn_kl_total)[:6]
            #     + " adv_std="
            #     + String(sqrt(adv_var))[:6]
            #     + " val="
            #     + String(avg_val)[:6]
            #     + " actor_grad="
            #     + String(actor_grad_norm)
            # )

            if self.logger:
                try:
                    var diag_step = self.train_step_count
                    self.logger[].log_scalar(
                        "wm_loss", total_wm_loss, diag_step
                    )
                    self.logger[].log_scalar("obs_loss", obs_loss, diag_step)
                    self.logger[].log_scalar("reward_loss", rew_loss, diag_step)
                    self.logger[].log_scalar(
                        "continue_loss", cont_loss, diag_step
                    )
                    self.logger[].log_scalar("dyn_kl", dyn_kl_total, diag_step)
                    self.logger[].log_scalar("rep_kl", rep_kl_total, diag_step)
                    self.logger[].log_scalar("adv_mean", avg_adv, diag_step)
                    self.logger[].log_scalar("adv_min", adv_min, diag_step)
                    self.logger[].log_scalar("adv_max", adv_max, diag_step)
                    self.logger[].log_scalar("adv_var", adv_var, diag_step)
                    self.logger[].log_scalar("val_mean", avg_val, diag_step)
                    self.logger[].log_scalar(
                        "actor_grad_norm", actor_grad_norm, diag_step
                    )
                    self.logger[].log_scalar(
                        "actor_param_norm", actor_param_norm, diag_step
                    )
                except:
                    pass

        self.train_step_count += 1

    # ══════════════════════════════════════════════════════════════════════
    # Training Loop (CPU)
    # ══════════════════════════════════════════════════════════════════════

    fn train[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int = 1000000,
        train_every: Int = 5,
        seed_episodes: Int = 5,
        print_every: Int = 10,
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) -> TrainingMetrics:
        """Train DreamerV3 on a continuous control environment (CPU).

        Alternates between collecting environment data and training
        the world model + actor-critic from replay.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            total_timesteps: Total environment steps (default: 1M).
            train_every: Steps between training updates (default: 5).
            seed_episodes: Random exploration episodes before training.
            print_every: Episodes between progress prints (default: 10).
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps (0 = every step).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        comptime ACT = Self.action_dim

        var metrics = TrainingMetrics(algorithm_name="DreamerV3")
        var episode_reward = Float64(0.0)
        var episode_steps = 0
        var episode_count = 0
        var total_env_steps = 0

        # ── Seed with random episodes ────────────────────────────────
        _ = env.reset()
        for _ in range(seed_episodes):
            var done = False
            while not done:
                var obs = _to_dtype_list(env.get_obs_list())
                var action = List[Scalar[dtype]](capacity=ACT)
                for _ in range(ACT):
                    action.append(Scalar[dtype](random_float64(-1.0, 1.0)))
                var result = env.step_continuous_vec[dtype](action)
                var reward = result[1]
                done = result[2]
                self.observe(obs, action, Float64(reward), done)
                total_env_steps += 1
                if done:
                    _ = env.reset()

        # ── Main training loop ───────────────────────────────────────
        _ = env.reset()
        self.reset_episode()

        for step in range(total_timesteps):
            var obs = _to_dtype_list(env.get_obs_list())

            # Select action
            var action: List[Scalar[dtype]]
            if total_env_steps < self.warmup_steps:
                action = List[Scalar[dtype]](capacity=ACT)
                for _ in range(ACT):
                    action.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            else:
                action = self.select_action(obs, training=True)

            # Environment step
            var result = env.step_continuous_vec[dtype](action)
            var reward = result[1]
            var done = result[2]

            self.observe(obs, action, Float64(reward), done)
            episode_reward += Float64(reward)
            episode_steps += 1
            total_env_steps += 1
            self.total_steps += 1

            if done:
                episode_count += 1
                metrics.log_episode[dtype](
                    episode_count,
                    Scalar[dtype](episode_reward),
                    episode_steps,
                    0.0,
                )

                # Log episode metrics
                if self.logger:
                    try:
                        self.logger[].log_scalar(
                            "episode_reward",
                            episode_reward,
                            total_env_steps,
                        )
                        self.logger[].log_scalar(
                            "episodes",
                            Float64(episode_count),
                            total_env_steps,
                        )
                        self.logger[].log_scalar(
                            "train_steps",
                            Float64(self.train_step_count),
                            total_env_steps,
                        )
                    except:
                        pass

                if episode_count % print_every == 0:
                    clear_progress_bar()
                    print(
                        "Episode "
                        + String(episode_count)
                        + " | Reward: "
                        + (
                            String("NaN") if episode_reward
                            != episode_reward else String(Int(episode_reward))
                        )
                        + " | Steps: "
                        + String(episode_steps)
                        + " | Train updates: "
                        + String(self.train_step_count)
                        + " | Buffer: "
                        + String(self.state.buffer.len())
                    )

                episode_reward = 0.0
                episode_steps = 0
                _ = env.reset()
                self.reset_episode()

            # Train
            if step % train_every == 0 and self.state.is_ready():
                _ = self.update()

            # Progress bar
            if step % 100 == 0:
                print_progress_bar(
                    step,
                    total_timesteps,
                    self.train_step_count,
                    "DreamerV3",
                )

        clear_progress_bar()
        print(
            "Training complete. Episodes: "
            + String(episode_count)
            + " | Total steps: "
            + String(total_env_steps)
            + " | Train updates: "
            + String(self.train_step_count)
        )
        return metrics^

    # ══════════════════════════════════════════════════════════════════════
    # Training Loop (GPU environments + GPU training)
    # ══════════════════════════════════════════════════════════════════════

    fn train_gpu[
        E: GPUContinuousEnv,
        n_envs: Int = 32,
    ](
        mut self,
        ctx: DeviceContext,
        num_episodes: Int = 1000,
        train_every: Int = 1,
        train_ratio: Int = 0,
        sync_every: Int = 50,
        verbose: Bool = True,
        print_every: Int = 50_000,
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train DreamerV3 with GPU environments and GPU training.

        N_ENVS parallel GPU environments for data collection. RSSM action
        selection runs entirely on GPU (symlog → encoder → GRU → posterior
        → actor → tanh-normal sample). World model + actor-critic training
        on GPU. Transitions downloaded to CPU per-env replay buffers.

        Parameters:
            E: GPU environment type implementing GPUContinuousEnv.
            n_envs: Number of parallel GPU environments (default: 32).

        Args:
            ctx: GPU device context.
            num_episodes: Target episodes to complete.
            train_every: Train every N env collection steps.
            train_ratio: Training updates per collection step (0 = n_envs for 1:1 ratio).
            sync_every: GPU->CPU weight sync interval (train steps).
            verbose: Print progress.
            print_every: Print interval in total env transitions.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        # Default train_ratio: n_envs for ~1:1 step:train ratio
        var actual_train_ratio = train_ratio if train_ratio > 0 else n_envs

        # ── Comptime aliases ─────────────────────────────────────────
        comptime OBS = Self.obs_dim
        comptime ACT = Self.action_dim
        comptime DETER = Self.deter_dim
        comptime STOCH = Self.STOCH_FLAT
        comptime FEAT = Self.FEAT_DIM
        comptime HID = Self.hidden
        comptime BL = Self.batch_length
        comptime B = Self.batch_size
        comptime ACTOR_OUT_DIM = Self.StateType.ActorModel.OUT_DIM
        comptime GRU_IN = DETER + 3 * HID
        comptime POST_IN = DETER + STOCH
        comptime ENV_BLOCKS = (n_envs + TPB - 1) // TPB
        comptime PER_ENV_CAP = max(B + BL + 2, Self.buffer_capacity // n_envs)

        # ── Network type aliases ─────────────────────────────────────
        comptime RSSMType = Self.StateType.RSSMType
        comptime EncNet = RSSMType.EncNet
        comptime PostNet = RSSMType.PostNet
        comptime DProjNet = RSSMType.DeterProjNet
        comptime SProjNet = RSSMType.StochProjNet
        comptime AProjNet = RSSMType.ActionProjNet
        comptime GHNet = RSSMType.GRUHiddenNet
        comptime GGNet = RSSMType.GRUGateNet

        # ── Workspace for env step ───────────────────────────────────
        comptime TOTAL_WS = (E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV)
        comptime WS_ALLOC = TOTAL_WS if TOTAL_WS > 0 else 1
        var step_ws_buf = ctx.enqueue_create_buffer[dtype](WS_ALLOC)
        E.init_step_workspace_gpu[n_envs](ctx, step_ws_buf)

        # ── GPU env buffers ──────────────────────────────────────────
        var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * OBS)
        var act_buf = ctx.enqueue_create_buffer[dtype](n_envs * ACT)
        var rew_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var done_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

        # ── Host transfer buffers (for CPU replay buffer) ────────────
        var obs_host = ctx.enqueue_create_host_buffer[dtype](n_envs * OBS)
        var act_host = ctx.enqueue_create_host_buffer[dtype](n_envs * ACT)
        var rew_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
        var done_host = ctx.enqueue_create_host_buffer[dtype](n_envs)

        # ── GPU RSSM inference state (persistent across steps) ───────
        var inf_deter = ctx.enqueue_create_buffer[dtype](n_envs * DETER)
        var inf_stoch = ctx.enqueue_create_buffer[dtype](n_envs * STOCH)
        var inf_prev_act = ctx.enqueue_create_buffer[dtype](n_envs * ACT)

        # ── GPU RSSM inference scratch ───────────────────────────────
        var inf_symlog = ctx.enqueue_create_buffer[dtype](n_envs * OBS)
        var inf_embed = ctx.enqueue_create_buffer[dtype](n_envs * STOCH)
        var inf_norm_act = ctx.enqueue_create_buffer[dtype](n_envs * ACT)
        var inf_proj_d = ctx.enqueue_create_buffer[dtype](n_envs * HID)
        var inf_proj_s = ctx.enqueue_create_buffer[dtype](n_envs * HID)
        var inf_proj_a = ctx.enqueue_create_buffer[dtype](n_envs * HID)
        var inf_gru_concat = ctx.enqueue_create_buffer[dtype](n_envs * GRU_IN)
        var inf_hidden = ctx.enqueue_create_buffer[dtype](n_envs * DETER)
        var inf_gate = ctx.enqueue_create_buffer[dtype](n_envs * 3 * DETER)
        var inf_new_deter = ctx.enqueue_create_buffer[dtype](n_envs * DETER)
        var inf_post_in = ctx.enqueue_create_buffer[dtype](n_envs * POST_IN)
        var inf_post_logits = ctx.enqueue_create_buffer[dtype](n_envs * STOCH)
        var inf_new_stoch = ctx.enqueue_create_buffer[dtype](n_envs * STOCH)
        var inf_post_probs = ctx.enqueue_create_buffer[dtype](n_envs * STOCH)
        var inf_feat = ctx.enqueue_create_buffer[dtype](n_envs * FEAT)
        var inf_actor_out = ctx.enqueue_create_buffer[dtype](
            n_envs * ACTOR_OUT_DIM
        )
        var inf_log_probs = ctx.enqueue_create_buffer[dtype](n_envs)

        # ── GPU inference workspace (shared, sequential reuse) ───────
        comptime ws1 = max(
            RSSMType.EncModel.WORKSPACE_SIZE_PER_SAMPLE,
            RSSMType.DeterProj.WORKSPACE_SIZE_PER_SAMPLE,
        )
        comptime ws2 = max(ws1, RSSMType.StochProj.WORKSPACE_SIZE_PER_SAMPLE)
        comptime ws3 = max(ws2, RSSMType.ActionProj.WORKSPACE_SIZE_PER_SAMPLE)
        comptime ws4 = max(
            ws3, RSSMType.GRUHiddenModel.WORKSPACE_SIZE_PER_SAMPLE
        )
        comptime ws5 = max(ws4, RSSMType.GRUGateModel.WORKSPACE_SIZE_PER_SAMPLE)
        comptime ws6 = max(ws5, RSSMType.PostModel.WORKSPACE_SIZE_PER_SAMPLE)
        comptime MAX_INF_WS_PER = max(
            ws6, Self.StateType.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
        )
        comptime INF_WS_SIZE = max(n_envs * MAX_INF_WS_PER, 1)
        var inf_ws = ctx.enqueue_create_buffer[dtype](INF_WS_SIZE)

        # ── Per-env replay buffers (CPU) ─────────────────────────────
        comptime PerEnvBuf = SequenceReplayBuffer[PER_ENV_CAP, OBS, ACT]
        var env_bufs = List[PerEnvBuf](capacity=n_envs)
        for _ in range(n_envs):
            env_bufs.append(PerEnvBuf())

        # ── GPU training state ───────────────────────────────────────
        var gpu_state = self.make_gpu_state(ctx)
        self.upload_to_gpu(gpu_state, ctx)
        ctx.synchronize()

        # ── Kernel aliases ──────────────────────────────────────────
        comptime run_symlog = symlog_kernel[n_envs * OBS]
        comptime run_action_norm = action_normalize_kernel[n_envs, ACT]
        comptime run_concat_gru = concat_gru_input_kernel[n_envs, DETER, HID]
        comptime run_gru_gate = gru_gate_kernel[n_envs, DETER]
        comptime run_concat_de = concat_deter_embed_kernel[n_envs, DETER, STOCH]
        comptime run_cat_sample = categorical_sample_kernel[
            n_envs, Self.stoch_dim, Self.classes, RSSMType.UNIMIX
        ]
        comptime run_concat_feat = concat_feat_kernel[n_envs, DETER, STOCH]
        comptime run_sample_actions = tanh_normal_sample_kernel[n_envs, ACT]
        comptime run_copy_deter = copy_kernel[n_envs * DETER]
        comptime run_copy_stoch = copy_kernel[n_envs * STOCH]
        comptime run_copy_act = copy_kernel[n_envs * ACT]

        @always_inline
        fn run_rssm_reset_done(
            det: LayoutTensor[
                dtype, Layout.row_major(n_envs, DETER), MutAnyOrigin
            ],
            sto: LayoutTensor[
                dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
            ],
            act: LayoutTensor[
                dtype, Layout.row_major(n_envs, ACT), MutAnyOrigin
            ],
            dones: LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin],
        ):
            var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
            if idx >= n_envs:
                return
            if dones[idx] > Scalar[dtype](0.5):
                for i in range(DETER):
                    det[idx, i] = Scalar[dtype](0)
                for i in range(STOCH):
                    sto[idx, i] = Scalar[dtype](0)
                for i in range(ACT):
                    act[idx, i] = Scalar[dtype](0)

        # ── Pre-create LayoutTensor views ────────────────────────────
        var obs_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var sym_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * OBS), MutAnyOrigin
        ](inf_symlog.unsafe_ptr())
        var sym_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, OBS), MutAnyOrigin
        ](inf_symlog.unsafe_ptr())
        var emb_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
        ](inf_embed.unsafe_ptr())
        var deter_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, DETER), MutAnyOrigin
        ](inf_deter.unsafe_ptr())
        var stoch_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
        ](inf_stoch.unsafe_ptr())
        var prev_act_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, ACT), MutAnyOrigin
        ](inf_prev_act.unsafe_ptr())
        var norm_act_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, ACT), MutAnyOrigin
        ](inf_norm_act.unsafe_ptr())
        var proj_d_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, HID), MutAnyOrigin
        ](inf_proj_d.unsafe_ptr())
        var proj_s_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, HID), MutAnyOrigin
        ](inf_proj_s.unsafe_ptr())
        var proj_a_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, HID), MutAnyOrigin
        ](inf_proj_a.unsafe_ptr())
        var concat_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, GRU_IN), MutAnyOrigin
        ](inf_gru_concat.unsafe_ptr())
        var hidden_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, DETER), MutAnyOrigin
        ](inf_hidden.unsafe_ptr())
        var gate_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, 3 * DETER), MutAnyOrigin
        ](inf_gate.unsafe_ptr())
        var new_deter_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, DETER), MutAnyOrigin
        ](inf_new_deter.unsafe_ptr())
        var post_in_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, POST_IN), MutAnyOrigin
        ](inf_post_in.unsafe_ptr())
        var post_logits_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
        ](inf_post_logits.unsafe_ptr())
        var new_stoch_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
        ](inf_new_stoch.unsafe_ptr())
        var post_probs_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, STOCH), MutAnyOrigin
        ](inf_post_probs.unsafe_ptr())
        var feat_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, FEAT), MutAnyOrigin
        ](inf_feat.unsafe_ptr())
        var actor_out_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, ACTOR_OUT_DIM), MutAnyOrigin
        ](inf_actor_out.unsafe_ptr())
        var act_2d = LayoutTensor[
            dtype, Layout.row_major(n_envs, ACT), MutAnyOrigin
        ](act_buf.unsafe_ptr())
        var log_probs_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](inf_log_probs.unsafe_ptr())
        var done_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](done_buf.unsafe_ptr())

        # 1D views for copy kernels
        var inf_deter_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * DETER), MutAnyOrigin
        ](inf_deter.unsafe_ptr())
        var inf_stoch_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * STOCH), MutAnyOrigin
        ](inf_stoch.unsafe_ptr())
        var inf_prev_act_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * ACT), MutAnyOrigin
        ](inf_prev_act.unsafe_ptr())
        var new_deter_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * DETER), MutAnyOrigin
        ](inf_new_deter.unsafe_ptr())
        var new_stoch_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * STOCH), MutAnyOrigin
        ](inf_new_stoch.unsafe_ptr())
        var act_1d = LayoutTensor[
            dtype, Layout.row_major(n_envs * ACT), MutAnyOrigin
        ](act_buf.unsafe_ptr())

        # ── Grid dimensions ──────────────────────────────────────────
        comptime SYM_BLOCKS = (n_envs * OBS + TPB - 1) // TPB
        comptime NORM_BLOCKS = (n_envs * ACT + TPB - 1) // TPB
        comptime CONCAT_GRU_BLOCKS = (n_envs * GRU_IN + TPB - 1) // TPB
        comptime GATE_BLOCKS = (n_envs * DETER + TPB - 1) // TPB
        comptime DE_BLOCKS = (n_envs * POST_IN + TPB - 1) // TPB
        comptime CAT_BLOCKS = (n_envs * Self.stoch_dim + TPB - 1) // TPB
        comptime FEAT_BLOCKS = (n_envs * FEAT + TPB - 1) // TPB
        comptime SAMPLE_BLOCKS = (n_envs + TPB - 1) // TPB
        comptime COPY_D_BLOCKS = (n_envs * DETER + TPB - 1) // TPB
        comptime COPY_S_BLOCKS = (n_envs * STOCH + TPB - 1) // TPB
        comptime COPY_A_BLOCKS = (n_envs * ACT + TPB - 1) // TPB

        # ── Tracking ─────────────────────────────────────────────────
        var metrics = TrainingMetrics(algorithm_name="DreamerV3-GPU")
        var cpu_ep_rewards = List[Float64](capacity=n_envs)
        for _ in range(n_envs):
            cpu_ep_rewards.append(0.0)
        var completed_episodes = 0
        var total_steps = 0
        var recent_reward_sum: Float64 = 0.0
        var recent_ep_count = 0
        var next_print = print_every
        var collection_step = 0

        # ── Init: reset envs, zero RSSM state ───────────────────────
        E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf)
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, OBS](
            ctx, states_buf, obs_buf
        )
        # Zero persistent RSSM state
        var z_d = LayoutTensor[
            dtype, Layout.row_major(n_envs * DETER), MutAnyOrigin
        ](inf_deter.unsafe_ptr())
        var z_s = LayoutTensor[
            dtype, Layout.row_major(n_envs * STOCH), MutAnyOrigin
        ](inf_stoch.unsafe_ptr())
        var z_a = LayoutTensor[
            dtype, Layout.row_major(n_envs * ACT), MutAnyOrigin
        ](inf_prev_act.unsafe_ptr())

        comptime run_zero_d = zero_kernel[n_envs * DETER]
        comptime run_zero_s = zero_kernel[n_envs * STOCH]
        comptime run_zero_a = zero_kernel[n_envs * ACT]

        ctx.enqueue_function[run_zero_d, run_zero_d](
            z_d, grid_dim=(COPY_D_BLOCKS,), block_dim=(TPB,)
        )
        ctx.enqueue_function[run_zero_s, run_zero_s](
            z_s, grid_dim=(COPY_S_BLOCKS,), block_dim=(TPB,)
        )
        ctx.enqueue_function[run_zero_a, run_zero_a](
            z_a, grid_dim=(COPY_A_BLOCKS,), block_dim=(TPB,)
        )
        ctx.synchronize()

        # ── Pre-allocated batch buffers (reused across training iters) ─
        comptime BATCH_OBS_SIZE = B * (BL + 1) * OBS
        comptime BATCH_ACT_SIZE = B * BL * ACT
        comptime BATCH_SCALAR_SIZE = B * BL
        var batch_obs = List[Scalar[DType.float32]](capacity=BATCH_OBS_SIZE)
        var batch_acts = List[Scalar[DType.float32]](capacity=BATCH_ACT_SIZE)
        var batch_rews = List[Scalar[DType.float32]](capacity=BATCH_SCALAR_SIZE)
        var batch_dones = List[Scalar[DType.float32]](
            capacity=BATCH_SCALAR_SIZE
        )
        for _ in range(BATCH_OBS_SIZE):
            batch_obs.append(Scalar[DType.float32](0))
        for _ in range(BATCH_ACT_SIZE):
            batch_acts.append(Scalar[DType.float32](0))
        for _ in range(BATCH_SCALAR_SIZE):
            batch_rews.append(Scalar[DType.float32](0))
            batch_dones.append(Scalar[DType.float32](0))

        # ── Pre-allocated per-sample buffers (reused across samples) ──
        comptime SEQ_OBS_SIZE = (BL + 1) * OBS
        comptime SEQ_ACT_SIZE = BL * ACT
        var s_obs = List[Scalar[DType.float32]](capacity=SEQ_OBS_SIZE)
        var s_act = List[Scalar[DType.float32]](capacity=SEQ_ACT_SIZE)
        var s_rew = List[Scalar[DType.float32]](capacity=BL)
        var s_don = List[Scalar[DType.float32]](capacity=BL)
        for _ in range(SEQ_OBS_SIZE):
            s_obs.append(Scalar[DType.float32](0))
        for _ in range(SEQ_ACT_SIZE):
            s_act.append(Scalar[DType.float32](0))
        for _ in range(BL):
            s_rew.append(Scalar[DType.float32](0))
            s_don.append(Scalar[DType.float32](0))

        # ── Timing accumulators (nanoseconds) ─────────────────────────
        var t_action_select: Int = 0   # GPU RSSM inference + action sampling
        var t_env_step: Int = 0        # GPU env step + download + reset
        var t_replay_add: Int = 0      # CPU replay buffer add + episode bookkeeping
        var t_replay_sample: Int = 0   # CPU replay buffer sampling + batch build
        var t_gpu_train: Int = 0       # GPU training step (upload + fwd/bwd)
        var t_sync: Int = 0            # GPU→CPU weight sync
        var timing_train_iters: Int = 0

        # ── Main training loop ───────────────────────────────────────
        while completed_episodes < num_episodes:
            # ── 1. Download current obs (for replay buffer) ──────────
            ctx.enqueue_copy(obs_host, obs_buf)

            # ── 2. Action selection (all on GPU) ─────────────────────
            var _t0_act = perf_counter_ns()
            if total_steps < self.warmup_steps * n_envs:
                # Warmup: random actions on CPU, upload
                ctx.synchronize()  # wait for obs_host
                for i in range(n_envs * ACT):
                    act_host[i] = Scalar[dtype](random_float64(-1.0, 1.0))
                ctx.enqueue_copy(act_buf, act_host)
            else:
                # Full GPU RSSM observe + actor + sample
                # 1. Symlog obs
                ctx.enqueue_function[run_symlog, run_symlog](
                    sym_1d,
                    obs_1d,
                    grid_dim=(SYM_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 2. Encoder forward
                EncNet.forward_gpu[n_envs](
                    ctx,
                    sym_2d,
                    emb_2d,
                    gpu_state.encoder.params_view(),
                    inf_ws,
                )
                # 3. Action normalize
                ctx.enqueue_function[run_action_norm, run_action_norm](
                    norm_act_2d,
                    prev_act_2d,
                    grid_dim=(NORM_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 4-6. Projections (sequential, shared workspace)
                DProjNet.forward_gpu[n_envs](
                    ctx,
                    deter_2d,
                    proj_d_2d,
                    gpu_state.deter_proj.params_view(),
                    inf_ws,
                )
                SProjNet.forward_gpu[n_envs](
                    ctx,
                    stoch_2d,
                    proj_s_2d,
                    gpu_state.stoch_proj.params_view(),
                    inf_ws,
                )
                AProjNet.forward_gpu[n_envs](
                    ctx,
                    norm_act_2d,
                    proj_a_2d,
                    gpu_state.action_proj.params_view(),
                    inf_ws,
                )
                # 7. Concat GRU input
                ctx.enqueue_function[run_concat_gru, run_concat_gru](
                    concat_2d,
                    deter_2d,
                    proj_d_2d,
                    proj_s_2d,
                    proj_a_2d,
                    grid_dim=(CONCAT_GRU_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 8. GRU hidden forward
                GHNet.forward_gpu[n_envs](
                    ctx,
                    concat_2d,
                    hidden_2d,
                    gpu_state.gru_hidden.params_view(),
                    inf_ws,
                )
                # 9. GRU gate forward
                GGNet.forward_gpu[n_envs](
                    ctx,
                    hidden_2d,
                    gate_2d,
                    gpu_state.gru_gates.params_view(),
                    inf_ws,
                )
                # 10. GRU gate application → new_deter
                ctx.enqueue_function[run_gru_gate, run_gru_gate](
                    new_deter_2d,
                    deter_2d,
                    gate_2d,
                    grid_dim=(GATE_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 11. Concat deter + embed → posterior input
                ctx.enqueue_function[run_concat_de, run_concat_de](
                    post_in_2d,
                    new_deter_2d,
                    emb_2d,
                    grid_dim=(DE_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 12. Posterior forward
                PostNet.forward_gpu[n_envs](
                    ctx,
                    post_in_2d,
                    post_logits_2d,
                    gpu_state.posterior.params_view(),
                    inf_ws,
                )
                # 13. Categorical sample → new_stoch
                var cat_seed = Scalar[DType.uint32](
                    UInt32(total_steps + 1)
                    * UInt32(n_envs * Self.stoch_dim * Self.classes + 1)
                )
                ctx.enqueue_function[run_cat_sample, run_cat_sample](
                    new_stoch_2d,
                    post_probs_2d,
                    post_logits_2d,
                    cat_seed,
                    Scalar[DType.bool](True),
                    grid_dim=(CAT_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 14. Concat feat
                ctx.enqueue_function[run_concat_feat, run_concat_feat](
                    feat_2d,
                    new_deter_2d,
                    new_stoch_2d,
                    grid_dim=(FEAT_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 15. Actor forward
                Self.ActorNet.forward_gpu[n_envs](
                    ctx,
                    feat_2d,
                    actor_out_2d,
                    gpu_state.actor.params_view(),
                    inf_ws,
                )
                # 16. Sample tanh-normal actions → act_buf
                var act_seed = Scalar[DType.uint32](
                    UInt32(total_steps + 2) * UInt32(n_envs * ACT + 1)
                )
                ctx.enqueue_function[run_sample_actions, run_sample_actions](
                    act_2d,
                    log_probs_1d,
                    actor_out_2d,
                    act_seed,
                    grid_dim=(SAMPLE_BLOCKS,),
                    block_dim=(TPB,),
                )
                # 17. Update persistent RSSM state
                ctx.enqueue_function[run_copy_deter, run_copy_deter](
                    inf_deter_1d,
                    new_deter_1d,
                    grid_dim=(COPY_D_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[run_copy_stoch, run_copy_stoch](
                    inf_stoch_1d,
                    new_stoch_1d,
                    grid_dim=(COPY_S_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[run_copy_act, run_copy_act](
                    inf_prev_act_1d,
                    act_1d,
                    grid_dim=(COPY_A_BLOCKS,),
                    block_dim=(TPB,),
                )

            t_action_select += Int(perf_counter_ns() - _t0_act)

            # ── 3. Step GPU envs ─────────────────────────────────────
            var _t0_env = perf_counter_ns()
            comptime if TOTAL_WS > 0:
                E.step_kernel_gpu[n_envs, E.STATE_SIZE, OBS, ACT](
                    ctx,
                    states_buf,
                    act_buf,
                    rew_buf,
                    done_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(total_steps * 1103515245 + 12345),
                    List[Scalar[dtype]](),
                    step_ws_buf.unsafe_ptr(),
                )
            else:
                E.step_kernel_gpu[n_envs, E.STATE_SIZE, OBS, ACT](
                    ctx,
                    states_buf,
                    act_buf,
                    rew_buf,
                    done_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(total_steps * 1103515245 + 12345),
                    List[Scalar[dtype]](),
                )

            # ── 4. Download transitions BEFORE reset clears done flags ──
            ctx.enqueue_copy(act_host, act_buf)
            ctx.enqueue_copy(rew_host, rew_buf)
            ctx.enqueue_copy(done_host, done_buf)
            ctx.synchronize()  # also completes obs_host from step 1

            # ── 5. Reset done envs (physics + RSSM) ──────────────────
            # Reset RSSM state for done envs (before done_buf is cleared)
            ctx.enqueue_function[run_rssm_reset_done, run_rssm_reset_done](
                deter_2d,
                stoch_2d,
                prev_act_2d,
                done_1d,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            # Reset physics state for done envs (clears done_buf to 0)
            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                done_buf,
                UInt64(total_steps * 2654435761 + 1),
                workspace_ptr=step_ws_buf.unsafe_ptr(),
            )
            # Re-extract obs (reset obs for done envs, next obs for others)
            E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, OBS](
                ctx, states_buf, obs_buf
            )

            t_env_step += Int(perf_counter_ns() - _t0_env)

            # ── 6. Process transitions (CPU, done_host already downloaded) ─
            var _t0_add = perf_counter_ns()
            for e in range(n_envs):
                var rew_val = Scalar[DType.float32](rew_host[e])
                var done_val = Float64(done_host[e]) > 0.5

                cpu_ep_rewards[e] += Float64(rew_val)

                var obs_arr = InlineArray[Scalar[DType.float32], OBS](fill=0)
                var act_arr = InlineArray[Scalar[DType.float32], ACT](fill=0)
                for k in range(OBS):
                    obs_arr[k] = Scalar[DType.float32](obs_host[e * OBS + k])
                for k in range(ACT):
                    act_arr[k] = Scalar[DType.float32](act_host[e * ACT + k])
                env_bufs[e].add(obs_arr, act_arr, rew_val, done_val)

                if done_val:
                    var ep_r = cpu_ep_rewards[e]
                    completed_episodes += 1
                    metrics.log_episode(completed_episodes, ep_r, 0, 0.0)
                    recent_reward_sum += ep_r
                    recent_ep_count += 1

                    if self.logger:
                        try:
                            self.logger[].log_scalar(
                                "episode_reward",
                                ep_r,
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

                    cpu_ep_rewards[e] = 0.0
                    if completed_episodes >= num_episodes:
                        break

            t_replay_add += Int(perf_counter_ns() - _t0_add)
            total_steps += n_envs
            collection_step += 1

            # ── 7. Training (train_ratio updates per collection step) ──
            if total_steps >= self.warmup_steps * n_envs:
                if collection_step % train_every == 0:
                    comptime min_ready = B + BL + 1
                    var ready = True
                    for e in range(n_envs):
                        if not env_bufs[e].is_ready[min_ready]():
                            ready = False
                            break

                    if ready:
                        for _tr in range(actual_train_ratio):
                            var _t0_samp = perf_counter_ns()
                            var b_per_env = B // n_envs
                            var b_rem = B % n_envs
                            var b_offset = 0
                            for e in range(n_envs):
                                var n_seqs = b_per_env + (1 if e < b_rem else 0)
                                for _ in range(n_seqs):
                                    env_bufs[e].sample_sequences[1, BL](
                                        s_obs, s_act, s_rew, s_don
                                    )

                                    var b = b_offset
                                    for k in range(SEQ_OBS_SIZE):
                                        batch_obs[
                                            b * SEQ_OBS_SIZE + k
                                        ] = s_obs[k]
                                    for k in range(SEQ_ACT_SIZE):
                                        batch_acts[
                                            b * SEQ_ACT_SIZE + k
                                        ] = s_act[k]
                                    for k in range(BL):
                                        batch_rews[b * BL + k] = s_rew[k]
                                        batch_dones[b * BL + k] = s_don[k]
                                    b_offset += 1
                            t_replay_sample += Int(perf_counter_ns() - _t0_samp)

                            var _t0_train = perf_counter_ns()
                            self.do_gpu_train_step(
                                ctx,
                                gpu_state,
                                batch_obs,
                                batch_acts,
                                batch_rews,
                                batch_dones,
                            )
                            t_gpu_train += Int(perf_counter_ns() - _t0_train)

                            if self.train_step_count % sync_every == 0:
                                var _t0_sync = perf_counter_ns()
                                self.download_from_gpu(gpu_state, ctx)
                                ctx.synchronize()
                                t_sync += Int(perf_counter_ns() - _t0_sync)
                            timing_train_iters += 1

            # ── 8. Progress ──────────────────────────────────────────
            if verbose and total_steps >= next_print:
                if recent_ep_count > 0:
                    var avg = recent_reward_sum / Float64(recent_ep_count)
                    clear_progress_bar()
                    print(
                        "Steps: "
                        + String(total_steps)
                        + " | Episodes: "
                        + String(completed_episodes)
                        + " | Avg reward: "
                        + String(avg)[:8]
                        + " | Train: "
                        + String(self.train_step_count)
                    )
                recent_reward_sum = 0.0
                recent_ep_count = 0
                next_print += print_every

            if verbose and total_steps % (n_envs * 100) == 0:
                print_progress_bar(
                    completed_episodes,
                    num_episodes,
                    self.train_step_count,
                    "DreamerV3-GPU",
                )

        # ── Final sync ───────────────────────────────────────────────
        self.download_from_gpu(gpu_state, ctx)
        ctx.synchronize()

        clear_progress_bar()
        print(
            "GPU Training complete. Episodes: "
            + String(completed_episodes)
            + " | Total steps: "
            + String(total_steps)
            + " | Train updates: "
            + String(self.train_step_count)
        )

        # ── Timing summary ──────────────────────────────────────────
        if verbose:
            var total_ns = (
                t_action_select
                + t_env_step
                + t_replay_add
                + t_replay_sample
                + t_gpu_train
                + t_sync
            )
            var total_ms = Float64(total_ns) / 1e6

            @always_inline
            fn _pct(ns: Int, tot: Int) -> Float64:
                if tot == 0:
                    return 0.0
                return Float64(ns) / Float64(tot) * 100.0

            print("\n======== DreamerV3 GPU Timing Summary ========")
            print(
                "Total measured time:",
                String(total_ms)[:9],
                "ms over",
                timing_train_iters,
                "training iterations",
            )
            if timing_train_iters > 0:
                print(
                    "Avg per training iter:",
                    String(total_ms / Float64(timing_train_iters))[:7],
                    "ms",
                )
            print("──────────────────────────────────────────────")
            print(
                "Action selection:   ",
                String(Float64(t_action_select) / 1e6)[:9],
                "ms (",
                String(_pct(t_action_select, total_ns))[:5],
                "%)",
            )
            print(
                "Env step+download:  ",
                String(Float64(t_env_step) / 1e6)[:9],
                "ms (",
                String(_pct(t_env_step, total_ns))[:5],
                "%)",
            )
            print(
                "Replay buffer add:  ",
                String(Float64(t_replay_add) / 1e6)[:9],
                "ms (",
                String(_pct(t_replay_add, total_ns))[:5],
                "%)",
            )
            print(
                "Replay sampling:    ",
                String(Float64(t_replay_sample) / 1e6)[:9],
                "ms (",
                String(_pct(t_replay_sample, total_ns))[:5],
                "%)",
            )
            print(
                "GPU training:       ",
                String(Float64(t_gpu_train) / 1e6)[:9],
                "ms (",
                String(_pct(t_gpu_train, total_ns))[:5],
                "%)",
            )
            print(
                "GPU→CPU sync:       ",
                String(Float64(t_sync) / 1e6)[:9],
                "ms (",
                String(_pct(t_sync, total_ns))[:5],
                "%)",
            )
            print("==============================================")

        return metrics^


# =============================================================================
# Helpers
# =============================================================================


@always_inline
fn _to_dtype_list[SRC: DType](src: List[Scalar[SRC]]) -> List[Scalar[dtype]]:
    """Convert a list of scalars from any floating-point DType to dtype."""
    var out = List[Scalar[dtype]](capacity=len(src))
    for i in range(len(src)):
        out.append(Scalar[dtype](src[i]))
    return out^


fn _clip_grads_gpu[
    PARAM_SIZE: Int,
](
    ctx: DeviceContext,
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    partial_sums_buf: DeviceBuffer[dtype],
    max_grad_norm: Scalar[dtype],
) raises:
    """Clip gradients of a single network on GPU.

    Two-kernel approach:
    1. gradient_norm_kernel: compute partial sums of squared grads
    2. gradient_reduce_apply_fused_kernel: reduce + clip in one pass
    """
    comptime GRAD_BLOCKS = (PARAM_SIZE + TPB - 1) // TPB
    var ps = LayoutTensor[dtype, Layout.row_major(GRAD_BLOCKS), MutAnyOrigin](
        partial_sums_buf.unsafe_ptr()
    )

    comptime run_norm = gradient_norm_kernel[
        dtype, PARAM_SIZE, GRAD_BLOCKS, TPB
    ]

    ctx.enqueue_function[run_norm, run_norm](
        ps,
        grads,
        grid_dim=(GRAD_BLOCKS,),
        block_dim=(TPB,),
    )

    comptime run_clip = gradient_reduce_apply_fused_kernel[
        dtype, PARAM_SIZE, GRAD_BLOCKS, TPB
    ]

    ctx.enqueue_function[run_clip, run_clip](
        grads,
        ps,
        max_grad_norm,
        grid_dim=(GRAD_BLOCKS,),
        block_dim=(TPB,),
    )
