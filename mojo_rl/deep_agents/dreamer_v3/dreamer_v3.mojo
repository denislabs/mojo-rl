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
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv
from .rssm import RSSM, categorical_sample, kl_divergence
from .state import DreamerV3CPUState, DreamerV3GPUState
from .imagination import (
    compute_lambda_returns,
    normalize_returns,
    sample_tanh_normal,
    log_prob_tanh_normal,
)
from mojo_rl.core.logger import LoggerPtr, _log, _log_flush
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
    TPB,
)


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
        BATCH_SIZE = Self.batch_size,
        BATCH_LENGTH = Self.batch_length,
        IMAGINE_HORIZON = Self.imagine_horizon,
    ]

    # ── Actor/Critic Network aliases (matching state.mojo definitions) ───
    comptime ActorNet = Network[Self.StateType.ActorModel, Adam[LR=3e-5]]
    comptime CriticNet = Network[Self.StateType.CriticModel, Adam[LR=3e-5]]

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
    var logger: LoggerPtr
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
        max_grad_norm: Float64 = 1000.0,
        logger: LoggerPtr = LoggerPtr(),
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
        """
        self.state = Self.StateType()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_balance = kl_balance
        self.actor_entropy = actor_entropy
        self.slow_critic_tau = slow_critic_tau
        self.return_norm_rate = return_norm_rate
        self.max_grad_norm = max_grad_norm
        self.logger = logger
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
        2. RSSM observe loop (posterior, per-timestep — no full BPTT)
        3. World model losses (decoder, reward, continue, KL)
        4. World model backward + optimizer step
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

        # Accumulated losses
        var obs_loss = Float64(0.0)
        var rew_loss = Float64(0.0)
        var cont_loss = Float64(0.0)
        var dyn_kl_total = Float64(0.0)
        var rep_kl_total = Float64(0.0)

        # Zero all world model gradients before the sequence
        self.state.rssm.zero_all_grads()

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

            # Store in all_* buffers for later use
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

            # ── Per-timestep world model losses ────────────────────────
            # Decoder loss: MSE(decoder(feat), symlog(obs_t+1))
            var dec_out_ptr = alloc[Scalar[dtype]](B * OBS)
            memset(dec_out_ptr, 0, B * OBS)
            var dec_out_t = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](dec_out_ptr)
            self.state.rssm.decode[B](feat_t, dec_out_t)

            for b in range(B):
                # Target: symlog(obs[t+1])
                for i in range(OBS):
                    var obs_next_idx = b * (BL + 1) * OBS + (t + 1) * OBS + i
                    var target = symlog(Float32(batch_obs[obs_next_idx]))
                    var pred = Float64(rebind[Scalar[dtype]](dec_out_t[b, i]))
                    var diff = pred - Float64(target)
                    obs_loss += diff * diff

            # Reward loss (skip t=0: no reward for initial observation)
            if t > 0:
                var rew_logits_ptr = alloc[Scalar[dtype]](B * BINS)
                memset(rew_logits_ptr, 0, B * BINS)
                var rew_logits_t = LayoutTensor[
                    dtype, Layout.row_major(B, BINS), MutAnyOrigin
                ](rew_logits_ptr)
                self.state.rssm.predict_reward[B](feat_t, rew_logits_t)

                for b in range(B):
                    var rew_val = Float32(batch_rewards[b * BL + t])
                    var rew_symlog = symlog(rew_val)

                    # Two-hot cross-entropy loss
                    var target_dist = InlineArray[Float32, Self.num_bins](
                        uninitialized=True
                    )
                    two_hot_encode[Self.num_bins](
                        rew_symlog, self.state.rssm.bins, target_dist
                    )

                    # Softmax + cross-entropy
                    var max_logit = Float64(
                        rebind[Scalar[dtype]](rew_logits_t[b, 0])
                    )
                    for k in range(1, BINS):
                        var v = Float64(
                            rebind[Scalar[dtype]](rew_logits_t[b, k])
                        )
                        if v > max_logit:
                            max_logit = v
                    var sum_exp = Float64(0.0)
                    for k in range(BINS):
                        sum_exp += exp(
                            Float64(rebind[Scalar[dtype]](rew_logits_t[b, k]))
                            - max_logit
                        )
                    var log_sum_exp = log(sum_exp) + max_logit

                    for k in range(BINS):
                        var t_k = Float64(target_dist[k])
                        if t_k > 1e-8:
                            var logit_k = Float64(
                                rebind[Scalar[dtype]](rew_logits_t[b, k])
                            )
                            rew_loss -= t_k * (logit_k - log_sum_exp)

                rew_logits_ptr.free()

            # Continue loss (skip t=0)
            if t > 0:
                var cont_out_ptr = alloc[Scalar[dtype]](B * 1)
                memset(cont_out_ptr, 0, B * 1)
                var cont_out_t = LayoutTensor[
                    dtype, Layout.row_major(B, 1), MutAnyOrigin
                ](cont_out_ptr)
                self.state.rssm.predict_continue[B](feat_t, cont_out_t)

                for b in range(B):
                    var cont_target = 1.0 - Float64(batch_dones[b * BL + t])
                    var cont_prob = Float64(
                        rebind[Scalar[dtype]](cont_out_t[b, 0])
                    )
                    # Clamp for numerical stability
                    if cont_prob < 1e-6:
                        cont_prob = 1e-6
                    if cont_prob > 1.0 - 1e-6:
                        cont_prob = 1.0 - 1e-6
                    # BCE loss
                    cont_loss -= cont_target * log(cont_prob) + (
                        1.0 - cont_target
                    ) * log(1.0 - cont_prob)

                cont_out_ptr.free()

            # KL losses (dual KL balancing)
            var dyn_kl = kl_divergence[B, Self.stoch_dim, Self.classes](
                post_probs_t, prior_probs_t
            )
            var rep_kl = kl_divergence[B, Self.stoch_dim, Self.classes](
                post_probs_t, prior_probs_t
            )
            # Apply free nats
            comptime FREE_NATS = 1.0
            if dyn_kl < FREE_NATS:
                dyn_kl = FREE_NATS
            if rep_kl < FREE_NATS:
                rep_kl = FREE_NATS
            dyn_kl_total += dyn_kl
            rep_kl_total += rep_kl

            # ── Per-timestep backward for decoder ──────────────────────
            # Compute gradient of decoder loss and backprop
            self._backward_decoder_step[B](feat_t, dec_out_t, batch_obs, t)

            dec_out_ptr.free()

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

        # ── Normalize losses by timesteps ──────────────────────────────
        var inv_bl = 1.0 / Float64(BL)
        var inv_bl_b = 1.0 / Float64(BL * B)
        obs_loss *= inv_bl_b
        rew_loss *= inv_bl_b
        cont_loss *= inv_bl_b
        dyn_kl_total *= inv_bl
        rep_kl_total *= inv_bl

        var total_wm_loss = (
            obs_loss
            + rew_loss
            + cont_loss
            + (0.5 * dyn_kl_total + 0.1 * rep_kl_total)
        )

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
            self.diag_every <= 0
            or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                # World model losses
                _log(
                    self.logger,
                    "loss",
                    total_wm_loss + actor_loss + critic_loss,
                    step,
                )
                _log(self.logger, "obs_loss", obs_loss, step)
                _log(self.logger, "reward_loss", rew_loss, step)
                _log(self.logger, "continue_loss", cont_loss, step)
                _log(self.logger, "dyn_kl", dyn_kl_total, step)
                _log(self.logger, "rep_kl", rep_kl_total, step)
                # Actor-critic
                _log(self.logger, "policy_loss", actor_loss, step)
                _log(self.logger, "value_loss", critic_loss, step)
                # Return normalization
                _log(
                    self.logger,
                    "return_scale",
                    Float64(self.state.return_ema_hi)
                    - Float64(self.state.return_ema_lo),
                    step,
                )
                # Mean imagined reward
                var imag_rew_sum: Float64 = 0.0
                for i in range(HORIZON * IB):
                    imag_rew_sum += Float64(
                        (self.state._imag_rewards + i)[]
                    )
                _log(
                    self.logger,
                    "imagined_reward_mean",
                    imag_rew_sum / Float64(HORIZON * IB),
                    step,
                )
                # Entropy (mean negative log_prob across imagination)
                var entropy_sum: Float64 = 0.0
                for i in range((HORIZON - 1) * IB):
                    entropy_sum -= Float64(
                        (self.state._imag_log_probs + i)[]
                    )
                _log(
                    self.logger,
                    "entropy",
                    entropy_sum / Float64((HORIZON - 1) * IB),
                    step,
                )
            except:
                pass

        return total_wm_loss + actor_loss + critic_loss

    # ══════════════════════════════════════════════════════════════════════
    # Private Helpers
    # ══════════════════════════════════════════════════════════════════════

    fn _backward_decoder_step[
        B: Int
    ](
        mut self,
        feat: LayoutTensor[
            dtype, Layout.row_major(B, Self.FEAT_DIM), MutAnyOrigin
        ],
        dec_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.obs_dim), MutAnyOrigin
        ],
        batch_obs: List[Scalar[DType.float32]],
        t: Int,
    ):
        """Backward pass for the decoder at a single timestep.

        Computes MSE gradient d/d(output) = 2 * (pred - target) / (B * obs_dim),
        then runs decoder backward to accumulate parameter gradients.

        Args:
            feat: Input features [B, FEAT_DIM].
            dec_out: Decoder output [B, OBS_DIM] (already computed by forward).
            batch_obs: Full batch observations for target extraction.
            t: Current timestep index.
        """
        comptime OBS = Self.obs_dim
        comptime FEAT = Self.FEAT_DIM
        comptime BL = Self.batch_length

        # Decoder cache for backward
        comptime DEC_CACHE = Self.StateType.RSSMType.DecModel.CACHE_SIZE
        var dec_cache_ptr = alloc[Scalar[dtype]](B * DEC_CACHE)
        memset(dec_cache_ptr, 0, B * DEC_CACHE)
        var dec_cache_t = LayoutTensor[
            dtype, Layout.row_major(B, DEC_CACHE), MutAnyOrigin
        ](dec_cache_ptr)

        # Re-run forward with cache
        var dec_out2_ptr = alloc[Scalar[dtype]](B * OBS)
        memset(dec_out2_ptr, 0, B * OBS)
        var dec_out2_t = LayoutTensor[
            dtype, Layout.row_major(B, OBS), MutAnyOrigin
        ](dec_out2_ptr)

        Self.StateType.RSSMType.DecNet.forward_with_cache[B](
            feat,
            dec_out2_t,
            self.state.rssm.decoder.params_view(),
            dec_cache_t,
        )

        # Compute gradient: 2 * (pred - symlog(target)) / (B * OBS)
        var grad_out_ptr = alloc[Scalar[dtype]](B * OBS)
        var scale_factor = 2.0 / Float64(B * OBS)
        for b in range(B):
            for i in range(OBS):
                var obs_idx = b * (BL + 1) * OBS + (t + 1) * OBS + i
                var target = Float64(symlog(Float32(batch_obs[obs_idx])))
                var pred = Float64(rebind[Scalar[dtype]](dec_out2_t[b, i]))
                (grad_out_ptr + b * OBS + i)[] = Scalar[dtype](
                    (pred - target) * scale_factor
                )

        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(B, OBS), MutAnyOrigin
        ](grad_out_ptr)
        var grad_in_ptr = alloc[Scalar[dtype]](B * FEAT)
        memset(grad_in_ptr, 0, B * FEAT)
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(B, FEAT), MutAnyOrigin
        ](grad_in_ptr)

        var dec_grads = self.state.rssm.decoder.grads_view()
        Self.StateType.RSSMType.DecNet.backward[B](
            grad_out_t,
            grad_in_t,
            self.state.rssm.decoder.params_view(),
            dec_cache_t,
            dec_grads,
        )

        dec_cache_ptr.free()
        dec_out2_ptr.free()
        grad_out_ptr.free()
        grad_in_ptr.free()

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
        var bins_host = ctx.enqueue_create_host_buffer[dtype](Self.num_bins)
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

        # ── 1. Upload batch data to GPU ──────────────────────────────────
        comptime OBS_SIZE = B * (BL + 1) * OBS
        comptime ACT_SIZE = B * BL * ACT
        comptime SCALAR_SIZE = B * BL

        var host_obs = ctx.enqueue_create_host_buffer[dtype](OBS_SIZE)
        for i in range(OBS_SIZE):
            host_obs[i] = Scalar[dtype](batch_obs[i])
        ctx.enqueue_copy(gpu_state.batch_obs, host_obs)

        var host_act = ctx.enqueue_create_host_buffer[dtype](ACT_SIZE)
        for i in range(ACT_SIZE):
            host_act[i] = Scalar[dtype](batch_actions[i])
        ctx.enqueue_copy(gpu_state.batch_actions, host_act)

        var host_rew = ctx.enqueue_create_host_buffer[dtype](SCALAR_SIZE)
        for i in range(SCALAR_SIZE):
            host_rew[i] = Scalar[dtype](batch_rewards[i])
        ctx.enqueue_copy(gpu_state.batch_rewards, host_rew)

        var host_done = ctx.enqueue_create_host_buffer[dtype](SCALAR_SIZE)
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

        # ── 2. RSSM Observe Loop ─────────────────────────────────────────
        var total_kl = Float64(0.0)

        for t in range(BL):
            # Extract obs[t] and action[t] from batch buffers on GPU
            # Copy obs slice: batch_obs[b*(BL+1)*OBS + t*OBS : ... + (t+1)*OBS]
            # For simplicity, we do this on CPU and upload per timestep.
            # A more optimized version would use gather kernels.
            var host_obs_step = ctx.enqueue_create_host_buffer[dtype](B * OBS)
            var host_act_step = ctx.enqueue_create_host_buffer[dtype](B * ACT)
            for b in range(B):
                for i in range(OBS):
                    host_obs_step[b * OBS + i] = Scalar[dtype](
                        batch_obs[b * (BL + 1) * OBS + t * OBS + i]
                    )
                for i in range(ACT):
                    if t == 0:
                        host_act_step[b * ACT + i] = Scalar[dtype](0.0)
                    else:
                        host_act_step[b * ACT + i] = Scalar[dtype](
                            batch_actions[b * BL * ACT + (t - 1) * ACT + i]
                        )
            ctx.enqueue_copy(gpu_state.obs_step_buf, host_obs_step)
            ctx.enqueue_copy(gpu_state.act_step_buf, host_act_step)

            # Symlog observations
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.obs_step_buf.unsafe_ptr())
            var symlog_t = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.symlog_obs_buf.unsafe_ptr())

            @always_inline
            fn run_symlog(
                o: LayoutTensor[dtype, Layout.row_major(B * OBS), MutAnyOrigin],
                inp: LayoutTensor[dtype, Layout.row_major(B * OBS), MutAnyOrigin],
            ):
                symlog_kernel[B * OBS](o, inp)

            comptime SYMLOG_BLOCKS = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[run_symlog, run_symlog](
                symlog_t, obs_t,
                grid_dim=(SYMLOG_BLOCKS,), block_dim=(TPB,),
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
                ctx, symlog_obs_2d, embed_2d,
                gpu_state.encoder.params_view(), enc_cache_t,
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

            @always_inline
            fn copy_symlog(
                d: LayoutTensor[dtype, Layout.row_major(SYMLOG_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(SYMLOG_SLICE), MutAnyOrigin],
            ):
                copy_kernel[SYMLOG_SLICE](d, s)

            comptime COPY_SL_BLOCKS = (SYMLOG_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_symlog, copy_symlog](
                all_symlog_t, symlog_1d,
                grid_dim=(COPY_SL_BLOCKS,), block_dim=(TPB,),
            )

            # Action normalize
            var act_2d = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.act_step_buf.unsafe_ptr())
            var norm_act_2d = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.norm_action_buf.unsafe_ptr())

            @always_inline
            fn run_action_norm(
                o: LayoutTensor[dtype, Layout.row_major(B, ACT), MutAnyOrigin],
                inp: LayoutTensor[dtype, Layout.row_major(B, ACT), MutAnyOrigin],
            ):
                action_normalize_kernel[B, ACT](o, inp)

            comptime NORM_BLOCKS = (B * ACT + TPB - 1) // TPB
            ctx.enqueue_function[run_action_norm, run_action_norm](
                norm_act_2d, act_2d,
                grid_dim=(NORM_BLOCKS,), block_dim=(TPB,),
            )

            # Save norm_action per timestep for BPTT
            comptime ACT_SLICE = B * ACT
            var all_nact_t = LayoutTensor[
                dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin
            ](gpu_state.all_norm_action_buf.unsafe_ptr() + t * ACT_SLICE)
            var nact_1d = LayoutTensor[
                dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin
            ](gpu_state.norm_action_buf.unsafe_ptr())

            @always_inline
            fn copy_nact(
                d: LayoutTensor[dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(ACT_SLICE), MutAnyOrigin],
            ):
                copy_kernel[ACT_SLICE](d, s)

            comptime COPY_NA_BLOCKS = (ACT_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_nact, copy_nact](
                all_nact_t, nact_1d,
                grid_dim=(COPY_NA_BLOCKS,), block_dim=(TPB,),
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

            @always_inline
            fn copy_prev_d(
                d: LayoutTensor[dtype, Layout.row_major(PREV_D_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(PREV_D_SLICE), MutAnyOrigin],
            ):
                copy_kernel[PREV_D_SLICE](d, s)

            comptime COPY_PD_BLOCKS = (PREV_D_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_prev_d, copy_prev_d](
                all_prev_deter_t, deter_1d_src,
                grid_dim=(COPY_PD_BLOCKS,), block_dim=(TPB,),
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
                ctx, deter_2d, proj_d_2d,
                gpu_state.deter_proj.params_view(), dproj_cache_t,
                gpu_state.ws_deter_proj,
            )
            SProjNet.forward_gpu_with_cache[B](
                ctx, stoch_2d, proj_s_2d,
                gpu_state.stoch_proj.params_view(), sproj_cache_t,
                gpu_state.ws_stoch_proj,
            )
            AProjNet.forward_gpu_with_cache[B](
                ctx, norm_act_2d, proj_a_2d,
                gpu_state.action_proj.params_view(), aproj_cache_t,
                gpu_state.ws_action_proj,
            )

            # Concat [deter, proj_d, proj_s, proj_a]
            comptime GRU_IN = DETER + 3 * HID
            var concat_2d = LayoutTensor[
                dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin
            ](gpu_state.concat_buf.unsafe_ptr())

            @always_inline
            fn run_concat_gru(
                co: LayoutTensor[dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                pd: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
                ps: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
                pa: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
            ):
                concat_gru_input_kernel[B, DETER, HID](co, d, pd, ps, pa)

            comptime CONCAT_BLOCKS = (B * GRU_IN + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_gru, run_concat_gru](
                concat_2d, deter_2d, proj_d_2d, proj_s_2d, proj_a_2d,
                grid_dim=(CONCAT_BLOCKS,), block_dim=(TPB,),
            )

            # GRU hidden layer (with cache for BPTT)
            var hidden_2d = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.hidden_out_buf.unsafe_ptr())
            comptime GH_CACHE = Self.StateType.RSSMType.GRUHiddenModel.CACHE_SIZE
            var gh_cache_t = LayoutTensor[
                dtype, Layout.row_major(B, GH_CACHE), MutAnyOrigin
            ](gpu_state.all_gru_hidden_cache_buf.unsafe_ptr() + t * B * GH_CACHE)
            GHNet.forward_gpu_with_cache[B](
                ctx, concat_2d, hidden_2d,
                gpu_state.gru_hidden.params_view(), gh_cache_t,
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
                ctx, hidden_2d, gate_2d,
                gpu_state.gru_gates.params_view(), gg_cache_t,
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

            @always_inline
            fn copy_gate(
                d: LayoutTensor[dtype, Layout.row_major(GATE_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(GATE_SLICE), MutAnyOrigin],
            ):
                copy_kernel[GATE_SLICE](d, s)

            comptime COPY_G_BLOCKS = (GATE_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_gate, copy_gate](
                all_gate_t, gate_1d,
                grid_dim=(COPY_G_BLOCKS,), block_dim=(TPB,),
            )

            # Apply GRU gating
            var new_deter_2d = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.new_deter_buf.unsafe_ptr())

            @always_inline
            fn run_gru_gate(
                nd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                pd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                go: LayoutTensor[dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin],
            ):
                gru_gate_kernel[B, DETER](nd, pd, go)

            comptime GATE_BLOCKS = (B * DETER + TPB - 1) // TPB
            ctx.enqueue_function[run_gru_gate, run_gru_gate](
                new_deter_2d, deter_2d, gate_2d,
                grid_dim=(GATE_BLOCKS,), block_dim=(TPB,),
            )

            # Posterior: concat(deter, embed) -> logits
            comptime POST_IN = DETER + STOCH
            var post_in_2d = LayoutTensor[
                dtype, Layout.row_major(B, POST_IN), MutAnyOrigin
            ](gpu_state.post_in_buf.unsafe_ptr())

            @always_inline
            fn run_concat_de(
                co: LayoutTensor[dtype, Layout.row_major(B, POST_IN), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                e: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
            ):
                concat_deter_embed_kernel[B, DETER, STOCH](co, d, e)

            comptime DE_BLOCKS = (B * POST_IN + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_de, run_concat_de](
                post_in_2d, new_deter_2d, embed_2d,
                grid_dim=(DE_BLOCKS,), block_dim=(TPB,),
            )

            var post_logits_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.post_logits_buf.unsafe_ptr())
            comptime POST_CACHE = Self.StateType.RSSMType.PostModel.CACHE_SIZE
            var post_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, POST_CACHE), MutAnyOrigin
            ](gpu_state.all_post_cache_buf.unsafe_ptr() + t * B * POST_CACHE)
            PostNet.forward_gpu_with_cache[B](
                ctx, post_in_2d, post_logits_2d,
                gpu_state.posterior.params_view(), post_cache_2d,
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
                ctx, new_deter_2d, prior_logits_2d,
                gpu_state.prior.params_view(), prior_cache_2d,
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
                UInt32(self.train_step_count * BL + t) * UInt32(B * Self.stoch_dim * Self.classes + 1)
            )

            @always_inline
            fn run_cat_post(
                o: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                p: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                l: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                s: Scalar[DType.uint32],
                tr: Scalar[DType.bool],
            ):
                categorical_sample_kernel[B, Self.stoch_dim, Self.classes, Self.StateType.RSSMType.UNIMIX](o, p, l, s, tr)

            comptime CAT_BLOCKS = (B * Self.stoch_dim + TPB - 1) // TPB
            ctx.enqueue_function[run_cat_post, run_cat_post](
                new_stoch_2d, post_probs_2d, post_logits_2d,
                cat_seed, Scalar[DType.bool](True),
                grid_dim=(CAT_BLOCKS,), block_dim=(TPB,),
            )

            # Categorical sample (prior — just for probs, discard output)
            var prior_probs_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.prior_probs_buf.unsafe_ptr())
            var dummy_stoch_2d = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.dummy_stoch_buf.unsafe_ptr())

            @always_inline
            fn run_cat_prior(
                o: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                p: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                l: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                s: Scalar[DType.uint32],
                tr: Scalar[DType.bool],
            ):
                categorical_sample_kernel[B, Self.stoch_dim, Self.classes, Self.StateType.RSSMType.UNIMIX](o, p, l, s, tr)

            ctx.enqueue_function[run_cat_prior, run_cat_prior](
                dummy_stoch_2d, prior_probs_2d, prior_logits_2d,
                cat_seed, Scalar[DType.bool](False),
                grid_dim=(CAT_BLOCKS,), block_dim=(TPB,),
            )

            # Build feat = concat(deter, stoch)
            var feat_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.feat_buf.unsafe_ptr())

            @always_inline
            fn run_concat_feat(
                f: LayoutTensor[dtype, Layout.row_major(B, FEAT), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
            ):
                concat_feat_kernel[B, DETER, STOCH](f, d, s)

            comptime FEAT_BLOCKS = (B * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_feat, run_concat_feat](
                feat_2d, new_deter_2d, new_stoch_2d,
                grid_dim=(FEAT_BLOCKS,), block_dim=(TPB,),
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

            @always_inline
            fn copy_deter(
                d: LayoutTensor[dtype, Layout.row_major(DETER_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(DETER_SLICE), MutAnyOrigin],
            ):
                copy_kernel[DETER_SLICE](d, s)

            comptime COPY_D_BLOCKS = (DETER_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_deter, copy_deter](
                all_deter_t, new_deter_1d,
                grid_dim=(COPY_D_BLOCKS,), block_dim=(TPB,),
            )

            comptime STOCH_SLICE = B * STOCH
            var all_stoch_t = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.all_stoch_buf.unsafe_ptr() + t * STOCH_SLICE)
            var new_stoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin
            ](gpu_state.new_stoch_buf.unsafe_ptr())

            @always_inline
            fn copy_stoch(
                d: LayoutTensor[dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(STOCH_SLICE), MutAnyOrigin],
            ):
                copy_kernel[STOCH_SLICE](d, s)

            comptime COPY_S_BLOCKS = (STOCH_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_stoch, copy_stoch](
                all_stoch_t, new_stoch_1d,
                grid_dim=(COPY_S_BLOCKS,), block_dim=(TPB,),
            )

            comptime FEAT_SLICE = B * FEAT
            var all_feat_t = LayoutTensor[
                dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin
            ](gpu_state.all_feats_buf.unsafe_ptr() + t * FEAT_SLICE)
            var feat_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin
            ](gpu_state.feat_buf.unsafe_ptr())

            @always_inline
            fn copy_feat(
                d: LayoutTensor[dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(FEAT_SLICE), MutAnyOrigin],
            ):
                copy_kernel[FEAT_SLICE](d, s)

            comptime COPY_F_BLOCKS = (FEAT_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_feat, copy_feat](
                all_feat_t, feat_1d,
                grid_dim=(COPY_F_BLOCKS,), block_dim=(TPB,),
            )

            # Decoder forward + backward (MSE loss against symlog(obs[t+1]))
            var dec_out_2d = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](gpu_state.dec_out_buf.unsafe_ptr())
            comptime DEC_CACHE = Self.StateType.RSSMType.DecModel.CACHE_SIZE
            var dec_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, DEC_CACHE), MutAnyOrigin
            ](gpu_state.dec_cache_buf.unsafe_ptr())
            DecNet.forward_gpu_with_cache[B](
                ctx, feat_2d, dec_out_2d,
                gpu_state.decoder.params_view(), dec_cache_2d,
                gpu_state.ws_decoder,
            )

            # Upload symlog(obs[t+1]) target to GPU
            var host_target = ctx.enqueue_create_host_buffer[dtype](B * OBS)
            for b in range(B):
                for i in range(OBS):
                    var idx = b * (BL + 1) * OBS + (t + 1) * OBS + i
                    host_target[b * OBS + i] = Scalar[dtype](
                        symlog(Float32(batch_obs[idx]))
                    )
            ctx.enqueue_copy(gpu_state.dec_target_buf, host_target)

            # MSE gradient
            var dec_target_2d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_target_buf.unsafe_ptr())
            var dec_pred_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_out_buf.unsafe_ptr())
            var dec_grad_1d = LayoutTensor[
                dtype, Layout.row_major(B * OBS), MutAnyOrigin
            ](gpu_state.dec_grad_out_buf.unsafe_ptr())
            var mse_scale = Scalar[dtype](2.0 / Float64(B * OBS))

            @always_inline
            fn run_mse_grad(
                g: LayoutTensor[dtype, Layout.row_major(B * OBS), MutAnyOrigin],
                p: LayoutTensor[dtype, Layout.row_major(B * OBS), MutAnyOrigin],
                tgt: LayoutTensor[dtype, Layout.row_major(B * OBS), MutAnyOrigin],
                s: Scalar[dtype],
            ):
                mse_grad_kernel[B * OBS](g, p, tgt, s)

            comptime MSE_BLOCKS = (B * OBS + TPB - 1) // TPB
            ctx.enqueue_function[run_mse_grad, run_mse_grad](
                dec_grad_1d, dec_pred_1d, dec_target_2d, mse_scale,
                grid_dim=(MSE_BLOCKS,), block_dim=(TPB,),
            )

            # Decoder backward
            var dec_grad_out_2d = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](gpu_state.dec_grad_out_buf.unsafe_ptr())
            var dec_grad_in_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.dec_grad_in_buf.unsafe_ptr())
            var dec_grads = gpu_state.decoder.grads_view()
            DecNet.backward_gpu[B](
                ctx, dec_grad_out_2d, dec_grad_in_2d,
                gpu_state.decoder.params_view(), dec_cache_2d, dec_grads,
                gpu_state.ws_decoder,
            )

            # ── Reward head backward ─────────────────────────────────────
            # Forward with cache
            comptime REW_CACHE = Self.StateType.RSSMType.RewModel.CACHE_SIZE
            var rew_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, REW_CACHE), MutAnyOrigin
            ](gpu_state.rew_cache_buf.unsafe_ptr())
            var rew_logits_2d = LayoutTensor[
                dtype, Layout.row_major(B, BINS), MutAnyOrigin
            ](gpu_state.rew_logits_buf.unsafe_ptr())
            RewNet.forward_gpu_with_cache[B](
                ctx, feat_2d, rew_logits_2d,
                gpu_state.reward_head.params_view(), rew_cache_2d,
                gpu_state.ws_reward,
            )

            # Upload symlog(reward[t]) target
            var host_rew_symlog = ctx.enqueue_create_host_buffer[dtype](B)
            for b in range(B):
                var r = batch_rewards[b * BL + t]
                host_rew_symlog[b] = Scalar[dtype](symlog(Float32(r)))
            ctx.enqueue_copy(gpu_state.rew_symlog_buf, host_rew_symlog)

            # Two-hot encode symlog reward
            var rew_symlog_1d = LayoutTensor[
                dtype, Layout.row_major(B), MutAnyOrigin
            ](gpu_state.rew_symlog_buf.unsafe_ptr())
            var rew_target_2d = LayoutTensor[
                dtype, Layout.row_major(B, BINS), MutAnyOrigin
            ](gpu_state.rew_target_buf.unsafe_ptr())
            var bins_1d = LayoutTensor[
                dtype, Layout.row_major(BINS), MutAnyOrigin
            ](gpu_state.bins_buf.unsafe_ptr())

            @always_inline
            fn run_rew_two_hot(
                tgt: LayoutTensor[dtype, Layout.row_major(B, BINS), MutAnyOrigin],
                vals: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
                b_: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
            ):
                two_hot_encode_kernel[B, BINS](tgt, vals, b_)

            comptime REW_TH_BLOCKS = (B + TPB - 1) // TPB
            ctx.enqueue_function[run_rew_two_hot, run_rew_two_hot](
                rew_target_2d, rew_symlog_1d, bins_1d,
                grid_dim=(REW_TH_BLOCKS,), block_dim=(TPB,),
            )

            # Two-hot CE gradient
            var rew_grad_out_2d = LayoutTensor[
                dtype, Layout.row_major(B, BINS), MutAnyOrigin
            ](gpu_state.rew_grad_out_buf.unsafe_ptr())
            var rew_inv_batch = Scalar[dtype](1.0 / Float64(B))

            @always_inline
            fn run_rew_ce_grad(
                g: LayoutTensor[dtype, Layout.row_major(B, BINS), MutAnyOrigin],
                l: LayoutTensor[dtype, Layout.row_major(B, BINS), MutAnyOrigin],
                tgt: LayoutTensor[dtype, Layout.row_major(B, BINS), MutAnyOrigin],
                ib: Scalar[dtype],
            ):
                two_hot_ce_grad_kernel[B, BINS](g, l, tgt, ib)

            ctx.enqueue_function[run_rew_ce_grad, run_rew_ce_grad](
                rew_grad_out_2d, rew_logits_2d, rew_target_2d, rew_inv_batch,
                grid_dim=(REW_TH_BLOCKS,), block_dim=(TPB,),
            )

            # Reward backward
            var rew_grad_in_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.rew_grad_in_buf.unsafe_ptr())
            var rew_grads = gpu_state.reward_head.grads_view()
            RewNet.backward_gpu[B](
                ctx, rew_grad_out_2d, rew_grad_in_2d,
                gpu_state.reward_head.params_view(), rew_cache_2d, rew_grads,
                gpu_state.ws_reward,
            )

            # ── Continue head backward ───────────────────────────────────
            # Forward with cache
            comptime CONT_CACHE = Self.StateType.RSSMType.ContModel.CACHE_SIZE
            var cont_cache_2d = LayoutTensor[
                dtype, Layout.row_major(B, CONT_CACHE), MutAnyOrigin
            ](gpu_state.cont_cache_buf.unsafe_ptr())
            var cont_logit_2d = LayoutTensor[
                dtype, Layout.row_major(B, 1), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            ContNet.forward_gpu_with_cache[B](
                ctx, feat_2d, cont_logit_2d,
                gpu_state.continue_head.params_view(), cont_cache_2d,
                gpu_state.ws_continue,
            )

            # Sigmoid
            var cont_pred_1d = LayoutTensor[
                dtype, Layout.row_major(B), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())

            @always_inline
            fn run_cont_sigmoid(
                o: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
                inp: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
            ):
                sigmoid_kernel[B](o, inp)

            comptime CONT_SIG_BLOCKS = (B + TPB - 1) // TPB
            ctx.enqueue_function[run_cont_sigmoid, run_cont_sigmoid](
                cont_pred_1d, cont_pred_1d,
                grid_dim=(CONT_SIG_BLOCKS,), block_dim=(TPB,),
            )

            # Upload 1.0 - done[t] as target
            var host_cont_target = ctx.enqueue_create_host_buffer[dtype](B)
            for b in range(B):
                host_cont_target[b] = Scalar[dtype](1.0 - Float64(batch_dones[b * BL + t]))
            ctx.enqueue_copy(gpu_state.cont_target_buf, host_cont_target)

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

            @always_inline
            fn run_cont_bce_grad(
                g: LayoutTensor[dtype, Layout.row_major(B, 1), MutAnyOrigin],
                p: LayoutTensor[dtype, Layout.row_major(B, 1), MutAnyOrigin],
                tgt: LayoutTensor[dtype, Layout.row_major(B, 1), MutAnyOrigin],
                ib: Scalar[dtype],
            ):
                bce_grad_kernel[B](g, p, tgt, ib)

            ctx.enqueue_function[run_cont_bce_grad, run_cont_bce_grad](
                cont_grad_2d, cont_pred_2d, cont_target_2d, cont_inv_batch,
                grid_dim=(CONT_SIG_BLOCKS,), block_dim=(TPB,),
            )

            # Continue backward
            var cont_grad_in_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.cont_grad_in_buf.unsafe_ptr())
            var cont_grads = gpu_state.continue_head.grads_view()
            ContNet.backward_gpu[B](
                ctx, cont_grad_2d, cont_grad_in_2d,
                gpu_state.continue_head.params_view(), cont_cache_2d, cont_grads,
                gpu_state.ws_continue,
            )

            # ── Accumulate d_feat = dec_grad_in + rew_grad_in + cont_grad_in ─
            # Start with dec_grad_in, then add rew_grad_in and cont_grad_in
            comptime FEAT_FLAT = B * FEAT
            var d_feat_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin
            ](gpu_state.d_feat_buf.unsafe_ptr())
            var dec_gi_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin
            ](gpu_state.dec_grad_in_buf.unsafe_ptr())
            var rew_gi_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin
            ](gpu_state.rew_grad_in_buf.unsafe_ptr())
            var cont_gi_1d = LayoutTensor[
                dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin
            ](gpu_state.cont_grad_in_buf.unsafe_ptr())

            # Copy dec_grad_in -> d_feat
            @always_inline
            fn copy_dfeat(
                d: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
            ):
                copy_kernel[FEAT_FLAT](d, s)

            comptime DFEAT_BLOCKS = (FEAT_FLAT + TPB - 1) // TPB
            ctx.enqueue_function[copy_dfeat, copy_dfeat](
                d_feat_1d, dec_gi_1d,
                grid_dim=(DFEAT_BLOCKS,), block_dim=(TPB,),
            )

            # d_feat += rew_grad_in
            @always_inline
            fn accum_rew(
                d: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[FEAT_FLAT](d, s)

            ctx.enqueue_function[accum_rew, accum_rew](
                d_feat_1d, rew_gi_1d,
                grid_dim=(DFEAT_BLOCKS,), block_dim=(TPB,),
            )

            # d_feat += cont_grad_in
            @always_inline
            fn accum_cont(
                d: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[FEAT_FLAT](d, s)

            ctx.enqueue_function[accum_cont, accum_cont](
                d_feat_1d, cont_gi_1d,
                grid_dim=(DFEAT_BLOCKS,), block_dim=(TPB,),
            )

            # Save d_feat per timestep
            var all_dfeat_t = LayoutTensor[
                dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin
            ](gpu_state.all_d_feat_buf.unsafe_ptr() + t * FEAT_FLAT)

            @always_inline
            fn copy_dfeat_save(
                d: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(FEAT_FLAT), MutAnyOrigin],
            ):
                copy_kernel[FEAT_FLAT](d, s)

            ctx.enqueue_function[copy_dfeat_save, copy_dfeat_save](
                all_dfeat_t, d_feat_1d,
                grid_dim=(DFEAT_BLOCKS,), block_dim=(TPB,),
            )

            # Save post_probs and prior_probs per timestep (already saved in all_post_probs_buf/all_prior_probs_buf below)

            # Store post/prior probs for BPTT KL gradient recomputation
            comptime PROBS_SLICE = B * STOCH
            var all_pp_t = LayoutTensor[
                dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin
            ](gpu_state.all_post_probs_buf.unsafe_ptr() + t * PROBS_SLICE)
            var pp_1d = LayoutTensor[
                dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin
            ](gpu_state.post_probs_buf.unsafe_ptr())

            @always_inline
            fn copy_pp(
                d: LayoutTensor[dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin],
            ):
                copy_kernel[PROBS_SLICE](d, s)

            comptime COPY_PP_BLOCKS = (PROBS_SLICE + TPB - 1) // TPB
            ctx.enqueue_function[copy_pp, copy_pp](
                all_pp_t, pp_1d,
                grid_dim=(COPY_PP_BLOCKS,), block_dim=(TPB,),
            )

            var all_prp_t = LayoutTensor[
                dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin
            ](gpu_state.all_prior_probs_buf.unsafe_ptr() + t * PROBS_SLICE)
            var prp_1d = LayoutTensor[
                dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin
            ](gpu_state.prior_probs_buf.unsafe_ptr())

            @always_inline
            fn copy_prp(
                d: LayoutTensor[dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(PROBS_SLICE), MutAnyOrigin],
            ):
                copy_kernel[PROBS_SLICE](d, s)

            ctx.enqueue_function[copy_prp, copy_prp](
                all_prp_t, prp_1d,
                grid_dim=(COPY_PP_BLOCKS,), block_dim=(TPB,),
            )

            # Swap deter/stoch for next timestep
            ctx.enqueue_copy(gpu_state.deter_buf, gpu_state.new_deter_buf)
            ctx.enqueue_copy(gpu_state.stoch_buf, gpu_state.new_stoch_buf)

        # ── 3. BPTT Backward Loop ─────────────────────────────────────────
        # Propagate gradients backward through time to train encoder,
        # projections, and GRU networks.
        ctx.enqueue_memset(gpu_state.d_recurrent_deter_buf, 0)
        ctx.enqueue_memset(gpu_state.d_recurrent_stoch_buf, 0)

        for t_rev in range(BL):
            var t = BL - 1 - t_rev

            # Load saved d_feat[t] -> d_feat_buf
            comptime BPTT_FEAT_FLAT = B * FEAT
            var bptt_d_feat = LayoutTensor[
                dtype, Layout.row_major(BPTT_FEAT_FLAT), MutAnyOrigin
            ](gpu_state.all_d_feat_buf.unsafe_ptr() + t * BPTT_FEAT_FLAT)
            var bptt_d_feat_cur = LayoutTensor[
                dtype, Layout.row_major(BPTT_FEAT_FLAT), MutAnyOrigin
            ](gpu_state.d_feat_buf.unsafe_ptr())

            @always_inline
            fn bptt_copy_dfeat(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_FEAT_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_FEAT_FLAT), MutAnyOrigin],
            ):
                copy_kernel[BPTT_FEAT_FLAT](d, s)

            comptime BPTT_DFEAT_BLOCKS = (BPTT_FEAT_FLAT + TPB - 1) // TPB
            ctx.enqueue_function[bptt_copy_dfeat, bptt_copy_dfeat](
                bptt_d_feat_cur, bptt_d_feat,
                grid_dim=(BPTT_DFEAT_BLOCKS,), block_dim=(TPB,),
            )

            # Split d_feat -> d_deter_feat, d_stoch_feat
            var bptt_d_feat_2d = LayoutTensor[
                dtype, Layout.row_major(B, FEAT), MutAnyOrigin
            ](gpu_state.d_feat_buf.unsafe_ptr())
            var bptt_d_deter = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_deter_total_buf.unsafe_ptr())
            var bptt_d_stoch = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_stoch_feat_buf.unsafe_ptr())

            @always_inline
            fn bptt_split_feat(
                dd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                ds: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                df: LayoutTensor[dtype, Layout.row_major(B, FEAT), MutAnyOrigin],
            ):
                concat_feat_backward_kernel[B, DETER, STOCH](dd, ds, df)

            comptime BPTT_SPLIT_BLOCKS = (B * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[bptt_split_feat, bptt_split_feat](
                bptt_d_deter, bptt_d_stoch, bptt_d_feat_2d,
                grid_dim=(BPTT_SPLIT_BLOCKS,), block_dim=(TPB,),
            )

            # Add recurrent stoch gradient from next timestep
            comptime STOCH_FLAT_SZ = B * STOCH
            var bptt_d_stoch_1d = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_stoch_feat_buf.unsafe_ptr())
            var bptt_rec_stoch = LayoutTensor[
                dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin
            ](gpu_state.d_recurrent_stoch_buf.unsafe_ptr())

            @always_inline
            fn bptt_add_rec_stoch(
                d: LayoutTensor[dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(STOCH_FLAT_SZ), MutAnyOrigin],
            ):
                accumulate_kernel[STOCH_FLAT_SZ](d, s)

            comptime BPTT_STOCH_BLOCKS = (STOCH_FLAT_SZ + TPB - 1) // TPB
            ctx.enqueue_function[bptt_add_rec_stoch, bptt_add_rec_stoch](
                bptt_d_stoch_1d, bptt_rec_stoch,
                grid_dim=(BPTT_STOCH_BLOCKS,), block_dim=(TPB,),
            )

            # Compute KL gradients (recompute from saved probs)
            var bptt_post_probs = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.all_post_probs_buf.unsafe_ptr() + t * B * STOCH)
            var bptt_prior_probs = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.all_prior_probs_buf.unsafe_ptr() + t * B * STOCH)

            # KL divergence
            var bptt_kl = LayoutTensor[
                dtype, Layout.row_major(B), MutAnyOrigin
            ](gpu_state.kl_buf.unsafe_ptr())

            @always_inline
            fn bptt_kl_div(
                kl: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
                pp: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                prp: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
            ):
                kl_divergence_kernel[B, Self.stoch_dim, Self.classes](kl, pp, prp)

            comptime BPTT_KL_BLOCKS = (B + TPB - 1) // TPB
            ctx.enqueue_function[bptt_kl_div, bptt_kl_div](
                bptt_kl, bptt_post_probs, bptt_prior_probs,
                grid_dim=(BPTT_KL_BLOCKS,), block_dim=(TPB,),
            )

            # KL gradient -> d_post_logits_kl, d_prior_logits
            var bptt_d_post_kl = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.post_grad_out_buf.unsafe_ptr())
            var bptt_d_prior_logits = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.prior_grad_out_buf.unsafe_ptr())
            var kl_free_nats = Scalar[dtype](Self.StateType.RSSMType.FREE_NATS)
            var kl_dyn_scale = Scalar[dtype](0.5)
            var kl_rep_scale = Scalar[dtype](0.1)
            var kl_inv_batch = Scalar[dtype](1.0 / Float64(B))

            @always_inline
            fn bptt_kl_grad(
                gp: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                gpr: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                pp: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                prp: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                kl: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
                fn_: Scalar[dtype],
                ds: Scalar[dtype],
                rs: Scalar[dtype],
                ib: Scalar[dtype],
            ):
                kl_categorical_gradient_kernel[B, Self.stoch_dim, Self.classes](
                    gp, gpr, pp, prp, kl, fn_, ds, rs, ib,
                )

            comptime BPTT_KL_GRAD_BLOCKS = (B * Self.stoch_dim + TPB - 1) // TPB
            ctx.enqueue_function[bptt_kl_grad, bptt_kl_grad](
                bptt_d_post_kl, bptt_d_prior_logits,
                bptt_post_probs, bptt_prior_probs,
                bptt_kl, kl_free_nats, kl_dyn_scale, kl_rep_scale, kl_inv_batch,
                grid_dim=(BPTT_KL_GRAD_BLOCKS,), block_dim=(TPB,),
            )

            # d_post_logits_total = d_stoch_feat (straight-through) + d_post_logits_kl
            var bptt_d_post_total = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_post_logits_total_buf.unsafe_ptr())

            # Copy d_stoch_feat -> d_post_logits_total (straight-through estimator)
            comptime BPTT_STOCH_SZ = B * STOCH
            var bptt_dpt_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin
            ](gpu_state.d_post_logits_total_buf.unsafe_ptr())
            var bptt_ds_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin
            ](gpu_state.d_stoch_feat_buf.unsafe_ptr())

            @always_inline
            fn bptt_copy_st(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
            ):
                copy_kernel[BPTT_STOCH_SZ](d, s)

            ctx.enqueue_function[bptt_copy_st, bptt_copy_st](
                bptt_dpt_1d, bptt_ds_1d,
                grid_dim=(BPTT_STOCH_BLOCKS,), block_dim=(TPB,),
            )

            # Add KL posterior gradient
            var bptt_kl_post_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin
            ](gpu_state.post_grad_out_buf.unsafe_ptr())

            @always_inline
            fn bptt_add_kl_post(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_STOCH_SZ](d, s)

            ctx.enqueue_function[bptt_add_kl_post, bptt_add_kl_post](
                bptt_dpt_1d, bptt_kl_post_1d,
                grid_dim=(BPTT_STOCH_BLOCKS,), block_dim=(TPB,),
            )

            # Posterior backward: d_post_logits_total -> d_post_in
            comptime BPTT_POST_CACHE = Self.StateType.RSSMType.PostModel.CACHE_SIZE
            var bptt_post_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_POST_CACHE), MutAnyOrigin
            ](gpu_state.all_post_cache_buf.unsafe_ptr() + t * B * BPTT_POST_CACHE)
            var bptt_post_grad_in = LayoutTensor[
                dtype, Layout.row_major(B, DETER + STOCH), MutAnyOrigin
            ](gpu_state.post_grad_in_buf.unsafe_ptr())
            var bptt_post_grads = gpu_state.posterior.grads_view()
            PostNet.backward_gpu[B](
                ctx, bptt_d_post_total, bptt_post_grad_in,
                gpu_state.posterior.params_view(), bptt_post_cache, bptt_post_grads,
                gpu_state.ws_posterior,
            )

            # Split d_post_in -> d_deter_from_post, d_embed
            comptime BPTT_POST_IN = DETER + STOCH
            var bptt_d_deter_from_post = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())
            var bptt_d_embed = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_embed_bwd_buf.unsafe_ptr())

            @always_inline
            fn bptt_split_post_in(
                dd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                de: LayoutTensor[dtype, Layout.row_major(B, STOCH), MutAnyOrigin],
                dc: LayoutTensor[dtype, Layout.row_major(B, BPTT_POST_IN), MutAnyOrigin],
            ):
                concat_deter_embed_backward_kernel[B, DETER, STOCH](dd, de, dc)

            comptime BPTT_SPLIT_POST_BLOCKS = (B * BPTT_POST_IN + TPB - 1) // TPB
            ctx.enqueue_function[bptt_split_post_in, bptt_split_post_in](
                bptt_d_deter_from_post, bptt_d_embed, bptt_post_grad_in,
                grid_dim=(BPTT_SPLIT_POST_BLOCKS,), block_dim=(TPB,),
            )

            # Prior backward: d_prior_logits -> d_deter_from_prior
            comptime BPTT_PRIOR_CACHE = Self.StateType.RSSMType.PriorModel.CACHE_SIZE
            var bptt_prior_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_PRIOR_CACHE), MutAnyOrigin
            ](gpu_state.all_prior_cache_buf.unsafe_ptr() + t * B * BPTT_PRIOR_CACHE)
            var bptt_prior_grad_in = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.prior_grad_in_buf.unsafe_ptr())
            var bptt_prior_grads = gpu_state.prior.grads_view()
            PriorNet.backward_gpu[B](
                ctx, bptt_d_prior_logits, bptt_prior_grad_in,
                gpu_state.prior.params_view(), bptt_prior_cache, bptt_prior_grads,
                gpu_state.ws_prior,
            )

            # Encoder backward: d_embed -> d_symlog_obs (discarded)
            comptime BPTT_ENC_CACHE = Self.StateType.RSSMType.EncModel.CACHE_SIZE
            var bptt_enc_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_ENC_CACHE), MutAnyOrigin
            ](gpu_state.all_enc_cache_buf.unsafe_ptr() + t * B * BPTT_ENC_CACHE)
            var bptt_d_symlog = LayoutTensor[
                dtype, Layout.row_major(B, OBS), MutAnyOrigin
            ](gpu_state.d_symlog_obs_bwd_buf.unsafe_ptr())
            var bptt_enc_grads = gpu_state.encoder.grads_view()
            EncNet.backward_gpu[B](
                ctx, bptt_d_embed, bptt_d_symlog,
                gpu_state.encoder.params_view(), bptt_enc_cache, bptt_enc_grads,
                gpu_state.ws_encoder,
            )

            # Accumulate d_deter_total = d_deter_feat + d_deter_from_post + d_deter_from_prior + d_recurrent
            comptime BPTT_DETER_FLAT = B * DETER
            var bptt_dd_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_deter_total_buf.unsafe_ptr())
            var bptt_dd_post_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())
            var bptt_dd_prior_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.prior_grad_in_buf.unsafe_ptr())
            var bptt_dd_rec_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_recurrent_deter_buf.unsafe_ptr())

            comptime BPTT_DD_BLOCKS = (BPTT_DETER_FLAT + TPB - 1) // TPB

            # d_deter_total already has d_deter_feat from split; add the rest
            @always_inline
            fn bptt_add_dd_post(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_add_dd_post, bptt_add_dd_post](
                bptt_dd_1d, bptt_dd_post_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            @always_inline
            fn bptt_add_dd_prior(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_add_dd_prior, bptt_add_dd_prior](
                bptt_dd_1d, bptt_dd_prior_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            @always_inline
            fn bptt_add_dd_rec(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_add_dd_rec, bptt_add_dd_rec](
                bptt_dd_1d, bptt_dd_rec_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            # GRU gate backward: d_deter_total -> d_gate_out, d_prev_deter_gru
            var bptt_d_gate = LayoutTensor[
                dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin
            ](gpu_state.d_gate_out_bwd_buf.unsafe_ptr())
            var bptt_d_prev_deter_gru = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_prev_deter_gru_buf.unsafe_ptr())
            var bptt_prev_deter = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.all_prev_deter_buf.unsafe_ptr() + t * B * DETER)
            var bptt_gate_out = LayoutTensor[
                dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin
            ](gpu_state.all_gate_out_buf.unsafe_ptr() + t * B * 3 * DETER)

            @always_inline
            fn bptt_gru_bwd(
                dg: LayoutTensor[dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin],
                dpd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                dnd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                pd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                go: LayoutTensor[dtype, Layout.row_major(B, 3 * DETER), MutAnyOrigin],
            ):
                gru_gate_backward_kernel[B, DETER](dg, dpd, dnd, pd, go)

            comptime BPTT_GRU_BLOCKS = (B * DETER + TPB - 1) // TPB
            ctx.enqueue_function[bptt_gru_bwd, bptt_gru_bwd](
                bptt_d_gate, bptt_d_prev_deter_gru, bptt_d_deter, bptt_prev_deter, bptt_gate_out,
                grid_dim=(BPTT_GRU_BLOCKS,), block_dim=(TPB,),
            )

            # GGNet backward: d_gate_out -> d_hidden_out
            comptime BPTT_GG_CACHE = Self.StateType.RSSMType.GRUGateModel.CACHE_SIZE
            var bptt_gg_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_GG_CACHE), MutAnyOrigin
            ](gpu_state.all_gru_gates_cache_buf.unsafe_ptr() + t * B * BPTT_GG_CACHE)
            var bptt_d_hidden = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_hidden_out_bwd_buf.unsafe_ptr())
            var bptt_gg_grads = gpu_state.gru_gates.grads_view()
            GGNet.backward_gpu[B](
                ctx, bptt_d_gate, bptt_d_hidden,
                gpu_state.gru_gates.params_view(), bptt_gg_cache, bptt_gg_grads,
                gpu_state.ws_gru_gates,
            )

            # GHNet backward: d_hidden_out -> d_concat
            comptime BPTT_GH_CACHE = Self.StateType.RSSMType.GRUHiddenModel.CACHE_SIZE
            comptime GRU_IN = DETER + 3 * HID
            var bptt_gh_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_GH_CACHE), MutAnyOrigin
            ](gpu_state.all_gru_hidden_cache_buf.unsafe_ptr() + t * B * BPTT_GH_CACHE)
            var bptt_d_concat = LayoutTensor[
                dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin
            ](gpu_state.d_concat_bwd_buf.unsafe_ptr())
            var bptt_gh_grads = gpu_state.gru_hidden.grads_view()
            GHNet.backward_gpu[B](
                ctx, bptt_d_hidden, bptt_d_concat,
                gpu_state.gru_hidden.params_view(), bptt_gh_cache, bptt_gh_grads,
                gpu_state.ws_gru_hidden,
            )

            # Split d_concat -> d_prev_deter_concat, d_proj_d, d_proj_s, d_proj_a
            var bptt_d_prev_deter_concat = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())  # reuse buffer
            var bptt_d_proj_d = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_d_bwd_buf.unsafe_ptr())
            var bptt_d_proj_s = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_s_bwd_buf.unsafe_ptr())
            var bptt_d_proj_a = LayoutTensor[
                dtype, Layout.row_major(B, HID), MutAnyOrigin
            ](gpu_state.d_proj_a_bwd_buf.unsafe_ptr())

            @always_inline
            fn bptt_split_concat(
                dd: LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin],
                dpd: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
                dps: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
                dpa: LayoutTensor[dtype, Layout.row_major(B, HID), MutAnyOrigin],
                dc: LayoutTensor[dtype, Layout.row_major(B, GRU_IN), MutAnyOrigin],
            ):
                concat_gru_input_backward_kernel[B, DETER, HID](dd, dpd, dps, dpa, dc)

            comptime BPTT_SPLIT_CONCAT_BLOCKS = (B * GRU_IN + TPB - 1) // TPB
            ctx.enqueue_function[bptt_split_concat, bptt_split_concat](
                bptt_d_prev_deter_concat, bptt_d_proj_d, bptt_d_proj_s, bptt_d_proj_a, bptt_d_concat,
                grid_dim=(BPTT_SPLIT_CONCAT_BLOCKS,), block_dim=(TPB,),
            )

            # DeterProj backward: d_proj_d -> d_prev_deter_dproj
            comptime BPTT_DPROJ_CACHE = Self.StateType.RSSMType.DeterProj.CACHE_SIZE
            var bptt_dproj_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_DPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_dproj_cache_buf.unsafe_ptr() + t * B * BPTT_DPROJ_CACHE)
            var bptt_d_prev_deter_dproj = LayoutTensor[
                dtype, Layout.row_major(B, DETER), MutAnyOrigin
            ](gpu_state.d_prev_deter_dproj_buf.unsafe_ptr())
            var bptt_dproj_grads = gpu_state.deter_proj.grads_view()
            DProjNet.backward_gpu[B](
                ctx, bptt_d_proj_d, bptt_d_prev_deter_dproj,
                gpu_state.deter_proj.params_view(), bptt_dproj_cache, bptt_dproj_grads,
                gpu_state.ws_deter_proj,
            )

            # StochProj backward: d_proj_s -> d_prev_stoch
            comptime BPTT_SPROJ_CACHE = Self.StateType.RSSMType.StochProj.CACHE_SIZE
            var bptt_sproj_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_SPROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_sproj_cache_buf.unsafe_ptr() + t * B * BPTT_SPROJ_CACHE)
            var bptt_d_prev_stoch = LayoutTensor[
                dtype, Layout.row_major(B, STOCH), MutAnyOrigin
            ](gpu_state.d_prev_stoch_bwd_buf.unsafe_ptr())
            var bptt_sproj_grads = gpu_state.stoch_proj.grads_view()
            SProjNet.backward_gpu[B](
                ctx, bptt_d_proj_s, bptt_d_prev_stoch,
                gpu_state.stoch_proj.params_view(), bptt_sproj_cache, bptt_sproj_grads,
                gpu_state.ws_stoch_proj,
            )

            # ActionProj backward: d_proj_a -> d_prev_action (discarded)
            comptime BPTT_APROJ_CACHE = Self.StateType.RSSMType.ActionProj.CACHE_SIZE
            var bptt_aproj_cache = LayoutTensor[
                dtype, Layout.row_major(B, BPTT_APROJ_CACHE), MutAnyOrigin
            ](gpu_state.all_aproj_cache_buf.unsafe_ptr() + t * B * BPTT_APROJ_CACHE)
            var bptt_d_prev_action = LayoutTensor[
                dtype, Layout.row_major(B, ACT), MutAnyOrigin
            ](gpu_state.d_prev_action_bwd_buf.unsafe_ptr())
            var bptt_aproj_grads = gpu_state.action_proj.grads_view()
            AProjNet.backward_gpu[B](
                ctx, bptt_d_proj_a, bptt_d_prev_action,
                gpu_state.action_proj.params_view(), bptt_aproj_cache, bptt_aproj_grads,
                gpu_state.ws_action_proj,
            )

            # Compute recurrent gradients for next iteration (t-1)
            # d_recurrent_deter = d_prev_deter_gru + d_prev_deter_concat + d_prev_deter_dproj
            var bptt_rec_deter = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_recurrent_deter_buf.unsafe_ptr())
            var bptt_dpd_gru_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_prev_deter_gru_buf.unsafe_ptr())
            var bptt_dpd_concat_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_deter_from_post_buf.unsafe_ptr())  # reused for concat split
            var bptt_dpd_dproj_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin
            ](gpu_state.d_prev_deter_dproj_buf.unsafe_ptr())

            # Copy d_prev_deter_gru -> d_recurrent_deter
            @always_inline
            fn bptt_copy_rec_d(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                copy_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_copy_rec_d, bptt_copy_rec_d](
                bptt_rec_deter, bptt_dpd_gru_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            # + d_prev_deter_concat
            @always_inline
            fn bptt_add_concat_d(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_add_concat_d, bptt_add_concat_d](
                bptt_rec_deter, bptt_dpd_concat_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            # + d_prev_deter_dproj
            @always_inline
            fn bptt_add_dproj_d(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
            ):
                accumulate_kernel[BPTT_DETER_FLAT](d, s)

            ctx.enqueue_function[bptt_add_dproj_d, bptt_add_dproj_d](
                bptt_rec_deter, bptt_dpd_dproj_1d,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            # d_recurrent_stoch = d_prev_stoch (from StochProj backward)
            var bptt_rec_stoch_dst = LayoutTensor[
                dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin
            ](gpu_state.d_recurrent_stoch_buf.unsafe_ptr())
            var bptt_dpstoch_1d = LayoutTensor[
                dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin
            ](gpu_state.d_prev_stoch_bwd_buf.unsafe_ptr())

            @always_inline
            fn bptt_copy_rec_s(
                d: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
            ):
                copy_kernel[BPTT_STOCH_SZ](d, s)

            ctx.enqueue_function[bptt_copy_rec_s, bptt_copy_rec_s](
                bptt_rec_stoch_dst, bptt_dpstoch_1d,
                grid_dim=(BPTT_STOCH_BLOCKS,), block_dim=(TPB,),
            )

            # Clamp recurrent gradients to prevent explosion across timesteps
            var bptt_clamp_max = Scalar[dtype](1.0)

            @always_inline
            fn bptt_clamp_rec_d(
                b: LayoutTensor[dtype, Layout.row_major(BPTT_DETER_FLAT), MutAnyOrigin],
                m: Scalar[dtype],
            ):
                clamp_kernel[BPTT_DETER_FLAT](b, m)

            ctx.enqueue_function[bptt_clamp_rec_d, bptt_clamp_rec_d](
                bptt_rec_deter, bptt_clamp_max,
                grid_dim=(BPTT_DD_BLOCKS,), block_dim=(TPB,),
            )

            @always_inline
            fn bptt_clamp_rec_s(
                b: LayoutTensor[dtype, Layout.row_major(BPTT_STOCH_SZ), MutAnyOrigin],
                m: Scalar[dtype],
            ):
                clamp_kernel[BPTT_STOCH_SZ](b, m)

            ctx.enqueue_function[bptt_clamp_rec_s, bptt_clamp_rec_s](
                bptt_rec_stoch_dst, bptt_clamp_max,
                grid_dim=(BPTT_STOCH_BLOCKS,), block_dim=(TPB,),
            )

        # ── 4. World model gradient clipping + optimizer step ──────────────
        var grad_norm_max = Scalar[dtype](self.max_grad_norm)
        _clip_grads_gpu[EncNet.MODEL.PARAM_SIZE](ctx, gpu_state.encoder.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[PostNet.MODEL.PARAM_SIZE](ctx, gpu_state.posterior.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[PriorNet.MODEL.PARAM_SIZE](ctx, gpu_state.prior.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[DecNet.MODEL.PARAM_SIZE](ctx, gpu_state.decoder.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[RewNet.MODEL.PARAM_SIZE](ctx, gpu_state.reward_head.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[ContNet.MODEL.PARAM_SIZE](ctx, gpu_state.continue_head.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[DProjNet.MODEL.PARAM_SIZE](ctx, gpu_state.deter_proj.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[SProjNet.MODEL.PARAM_SIZE](ctx, gpu_state.stoch_proj.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[AProjNet.MODEL.PARAM_SIZE](ctx, gpu_state.action_proj.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[GHNet.MODEL.PARAM_SIZE](ctx, gpu_state.gru_hidden.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        _clip_grads_gpu[GGNet.MODEL.PARAM_SIZE](ctx, gpu_state.gru_gates.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
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

        @always_inline
        fn copy_all_deter(
            d: LayoutTensor[dtype, Layout.row_major(IB_DETER), MutAnyOrigin],
            s: LayoutTensor[dtype, Layout.row_major(IB_DETER), MutAnyOrigin],
        ):
            copy_kernel[IB_DETER](d, s)

        comptime INIT_D_BLOCKS = (IB_DETER + TPB - 1) // TPB
        ctx.enqueue_function[copy_all_deter, copy_all_deter](
            imag_deter_init, all_deter_1d,
            grid_dim=(INIT_D_BLOCKS,), block_dim=(TPB,),
        )

        comptime IB_STOCH = IB * STOCH
        var all_stoch_1d = LayoutTensor[
            dtype, Layout.row_major(IB_STOCH), MutAnyOrigin
        ](gpu_state.all_stoch_buf.unsafe_ptr())
        var imag_stoch_init = LayoutTensor[
            dtype, Layout.row_major(IB_STOCH), MutAnyOrigin
        ](gpu_state.imag_stoch_buf.unsafe_ptr())

        @always_inline
        fn copy_all_stoch(
            d: LayoutTensor[dtype, Layout.row_major(IB_STOCH), MutAnyOrigin],
            s: LayoutTensor[dtype, Layout.row_major(IB_STOCH), MutAnyOrigin],
        ):
            copy_kernel[IB_STOCH](d, s)

        comptime INIT_S_BLOCKS = (IB_STOCH + TPB - 1) // TPB
        ctx.enqueue_function[copy_all_stoch, copy_all_stoch](
            imag_stoch_init, all_stoch_1d,
            grid_dim=(INIT_S_BLOCKS,), block_dim=(TPB,),
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

            @always_inline
            fn run_concat_imag_feat(
                f: LayoutTensor[dtype, Layout.row_major(IB, FEAT), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(IB, DETER), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(IB, STOCH), MutAnyOrigin],
            ):
                concat_feat_kernel[IB, DETER, STOCH](f, d, s)

            comptime IB_FEAT_BLOCKS = (IB * FEAT + TPB - 1) // TPB
            ctx.enqueue_function[run_concat_imag_feat, run_concat_imag_feat](
                imag_feat_2d, imag_deter_2d, imag_stoch_2d,
                grid_dim=(IB_FEAT_BLOCKS,), block_dim=(TPB,),
            )

            # Actor forward -> sample actions
            var actor_out_2d = LayoutTensor[
                dtype, Layout.row_major(IB, Self.StateType.ActorModel.OUT_DIM), MutAnyOrigin,
            ](gpu_state.actor_out_buf.unsafe_ptr())
            Self.ActorNet.forward_gpu[IB](
                ctx, imag_feat_2d, actor_out_2d,
                gpu_state.actor.params_view(), gpu_state.ws_actor,
            )

            # Sample tanh-normal actions + log probs
            var actions_2d = LayoutTensor[
                dtype, Layout.row_major(IB, ACT), MutAnyOrigin
            ](gpu_state.imag_actions_buf.unsafe_ptr())
            var log_probs_1d = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_log_probs_buf.unsafe_ptr())

            var act_seed = Scalar[DType.uint32](
                UInt32(self.train_step_count * HORIZON + h) * UInt32(IB * ACT + 1)
            )

            @always_inline
            fn run_sample_actions(
                a: LayoutTensor[dtype, Layout.row_major(IB, ACT), MutAnyOrigin],
                lp: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
                ao: LayoutTensor[dtype, Layout.row_major(IB, 2 * ACT), MutAnyOrigin],
                s: Scalar[DType.uint32],
            ):
                tanh_normal_sample_kernel[IB, ACT](a, lp, ao, s)

            comptime SAMPLE_BLOCKS = (IB + TPB - 1) // TPB
            ctx.enqueue_function[run_sample_actions, run_sample_actions](
                actions_2d, log_probs_1d, actor_out_2d, act_seed,
                grid_dim=(SAMPLE_BLOCKS,), block_dim=(TPB,),
            )

            # Predict reward from feat
            var rew_logits_2d = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.rew_logits_buf.unsafe_ptr())
            RewNet.forward_gpu[IB](
                ctx, imag_feat_2d, rew_logits_2d,
                gpu_state.reward_head.params_view(), gpu_state.ws_reward,
            )

            # Decode reward values
            var rewards_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_rewards_buf.unsafe_ptr() + h * IB)
            var bins_1d = LayoutTensor[
                dtype, Layout.row_major(BINS), MutAnyOrigin
            ](gpu_state.bins_buf.unsafe_ptr())

            @always_inline
            fn run_decode_reward(
                v: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
                l: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
                b: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
                se: Scalar[DType.bool],
            ):
                decode_value_kernel[IB, BINS](v, l, b, se)

            ctx.enqueue_function[run_decode_reward, run_decode_reward](
                rewards_h, rew_logits_2d, bins_1d, Scalar[DType.bool](True),
                grid_dim=(SAMPLE_BLOCKS,), block_dim=(TPB,),
            )

            # Predict continue
            var cont_out_2d = LayoutTensor[
                dtype, Layout.row_major(IB, 1), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            ContNet.forward_gpu[IB](
                ctx, imag_feat_2d, cont_out_2d,
                gpu_state.continue_head.params_view(), gpu_state.ws_continue,
            )

            # Apply sigmoid to continue output
            var cont_1d_in = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.cont_out_buf.unsafe_ptr())
            var continues_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_continues_buf.unsafe_ptr() + h * IB)

            @always_inline
            fn run_sigmoid(
                o: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
                i: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            ):
                sigmoid_kernel[IB](o, i)

            ctx.enqueue_function[run_sigmoid, run_sigmoid](
                continues_h, cont_1d_in,
                grid_dim=(SAMPLE_BLOCKS,), block_dim=(TPB,),
            )

            # Critic value prediction
            var critic_logits_2d = LayoutTensor[
                dtype, Layout.row_major(IB, BINS), MutAnyOrigin
            ](gpu_state.critic_logits_buf.unsafe_ptr())
            Self.CriticNet.forward_gpu[IB](
                ctx, imag_feat_2d, critic_logits_2d,
                gpu_state.critic.params_view(), gpu_state.ws_critic,
            )

            var values_h = LayoutTensor[
                dtype, Layout.row_major(IB), MutAnyOrigin
            ](gpu_state.imag_values_buf.unsafe_ptr() + h * IB)

            @always_inline
            fn run_decode_value(
                v: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
                l: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
                b: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
                se: Scalar[DType.bool],
            ):
                decode_value_kernel[IB, BINS](v, l, b, se)

            ctx.enqueue_function[run_decode_value, run_decode_value](
                values_h, critic_logits_2d, bins_1d, Scalar[DType.bool](True),
                grid_dim=(SAMPLE_BLOCKS,), block_dim=(TPB,),
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

                @always_inline
                fn run_imag_act_norm(
                    o: LayoutTensor[dtype, Layout.row_major(IB, ACT), MutAnyOrigin],
                    inp: LayoutTensor[dtype, Layout.row_major(IB, ACT), MutAnyOrigin],
                ):
                    action_normalize_kernel[IB, ACT](o, inp)

                comptime IMAG_NORM_BLOCKS = (IB * ACT + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_act_norm, run_imag_act_norm](
                    imag_norm_act_2d, actions_2d,
                    grid_dim=(IMAG_NORM_BLOCKS,), block_dim=(TPB,),
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
                    ctx, imag_deter_2d, imag_proj_d_2d,
                    gpu_state.deter_proj.params_view(), gpu_state.ws_deter_proj,
                )
                SProjNet.forward_gpu[IB](
                    ctx, imag_stoch_2d, imag_proj_s_2d,
                    gpu_state.stoch_proj.params_view(), gpu_state.ws_stoch_proj,
                )
                AProjNet.forward_gpu[IB](
                    ctx, imag_norm_act_2d, imag_proj_a_2d,
                    gpu_state.action_proj.params_view(), gpu_state.ws_action_proj,
                )

                # Concat [deter, proj_d, proj_s, proj_a]
                comptime GRU_IN_IB = DETER + 3 * HID
                var imag_concat_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, GRU_IN_IB), MutAnyOrigin
                ](gpu_state.imag_concat_buf.unsafe_ptr())

                @always_inline
                fn run_imag_concat_gru(
                    co: LayoutTensor[dtype, Layout.row_major(IB, GRU_IN_IB), MutAnyOrigin],
                    d: LayoutTensor[dtype, Layout.row_major(IB, DETER), MutAnyOrigin],
                    pd: LayoutTensor[dtype, Layout.row_major(IB, HID), MutAnyOrigin],
                    ps: LayoutTensor[dtype, Layout.row_major(IB, HID), MutAnyOrigin],
                    pa: LayoutTensor[dtype, Layout.row_major(IB, HID), MutAnyOrigin],
                ):
                    concat_gru_input_kernel[IB, DETER, HID](co, d, pd, ps, pa)

                comptime IMAG_CONCAT_BLOCKS = (IB * GRU_IN_IB + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_concat_gru, run_imag_concat_gru](
                    imag_concat_2d, imag_deter_2d,
                    imag_proj_d_2d, imag_proj_s_2d, imag_proj_a_2d,
                    grid_dim=(IMAG_CONCAT_BLOCKS,), block_dim=(TPB,),
                )

                # GRU hidden layer
                var imag_hidden_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, DETER), MutAnyOrigin
                ](gpu_state.imag_hidden_buf.unsafe_ptr())
                GHNet.forward_gpu[IB](
                    ctx, imag_concat_2d, imag_hidden_2d,
                    gpu_state.gru_hidden.params_view(), gpu_state.ws_gru_hidden,
                )

                # GRU gates
                var imag_gate_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, 3 * DETER), MutAnyOrigin
                ](gpu_state.imag_gate_buf.unsafe_ptr())
                GGNet.forward_gpu[IB](
                    ctx, imag_hidden_2d, imag_gate_2d,
                    gpu_state.gru_gates.params_view(), gpu_state.ws_gru_gates,
                )

                # Apply GRU gating -> next_deter
                @always_inline
                fn run_imag_gru_gate(
                    nd: LayoutTensor[dtype, Layout.row_major(IB, DETER), MutAnyOrigin],
                    pd: LayoutTensor[dtype, Layout.row_major(IB, DETER), MutAnyOrigin],
                    go: LayoutTensor[dtype, Layout.row_major(IB, 3 * DETER), MutAnyOrigin],
                ):
                    gru_gate_kernel[IB, DETER](nd, pd, go)

                comptime IMAG_GATE_BLOCKS = (IB * DETER + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_gru_gate, run_imag_gru_gate](
                    next_deter_2d, imag_deter_2d, imag_gate_2d,
                    grid_dim=(IMAG_GATE_BLOCKS,), block_dim=(TPB,),
                )

                # ── Prior: deter -> logits -> sample stoch ────────────

                var imag_prior_logits_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](gpu_state.imag_prior_logits_buf.unsafe_ptr())
                PriorNet.forward_gpu[IB](
                    ctx, next_deter_2d, imag_prior_logits_2d,
                    gpu_state.prior.params_view(), gpu_state.ws_prior,
                )

                # Categorical sample from prior
                var imag_prior_probs_2d = LayoutTensor[
                    dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
                ](gpu_state.imag_prior_probs_buf.unsafe_ptr())

                var imag_cat_seed = Scalar[DType.uint32](
                    UInt32(self.train_step_count * HORIZON + h + 1000) * UInt32(IB * Self.stoch_dim * Self.classes + 1)
                )

                @always_inline
                fn run_imag_cat_prior(
                    o: LayoutTensor[dtype, Layout.row_major(IB, STOCH), MutAnyOrigin],
                    p: LayoutTensor[dtype, Layout.row_major(IB, STOCH), MutAnyOrigin],
                    l: LayoutTensor[dtype, Layout.row_major(IB, STOCH), MutAnyOrigin],
                    s: Scalar[DType.uint32],
                    tr: Scalar[DType.bool],
                ):
                    categorical_sample_kernel[IB, Self.stoch_dim, Self.classes, Self.StateType.RSSMType.UNIMIX](o, p, l, s, tr)

                comptime IMAG_CAT_BLOCKS = (IB * Self.stoch_dim + TPB - 1) // TPB
                ctx.enqueue_function[run_imag_cat_prior, run_imag_cat_prior](
                    next_stoch_2d, imag_prior_probs_2d, imag_prior_logits_2d,
                    imag_cat_seed, Scalar[DType.bool](True),
                    grid_dim=(IMAG_CAT_BLOCKS,), block_dim=(TPB,),
                )

        # ── 6. Lambda returns ────────────────────────────────────────────
        # Download imagination scalars to CPU for lambda return computation
        ctx.synchronize()
        var host_rewards = ctx.enqueue_create_host_buffer[dtype](HORIZON * IB)
        var host_values = ctx.enqueue_create_host_buffer[dtype](HORIZON * IB)
        var host_continues = ctx.enqueue_create_host_buffer[dtype](HORIZON * IB)
        ctx.enqueue_copy(host_rewards, gpu_state.imag_rewards_buf)
        ctx.enqueue_copy(host_values, gpu_state.imag_values_buf)
        ctx.enqueue_copy(host_continues, gpu_state.imag_continues_buf)
        ctx.synchronize()

        # Compute lambda returns on CPU
        var returns_raw = alloc[Scalar[dtype]](HORIZON * IB)
        var rewards_raw = alloc[Scalar[dtype]](HORIZON * IB)
        var values_raw = alloc[Scalar[dtype]](HORIZON * IB)
        var continues_raw = alloc[Scalar[dtype]](HORIZON * IB)
        for i in range(HORIZON * IB):
            rewards_raw[i] = host_rewards[i]
            values_raw[i] = host_values[i]
            continues_raw[i] = host_continues[i]

        # Rebind to MutAnyOrigin for compute_lambda_returns signature
        var returns_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](returns_raw)
        var rewards_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](rewards_raw)
        var values_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](values_raw)
        var continues_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](continues_raw)

        compute_lambda_returns[HORIZON, IB](
            rewards_ptr, values_ptr, continues_ptr, returns_ptr,
            self.gamma, self.lambda_,
        )

        var scale = normalize_returns[HORIZON, IB](
            returns_ptr,
            self.state.return_ema_lo,
            self.state.return_ema_hi,
            self.return_norm_rate,
        )

        # Upload normalized returns and values back to GPU
        var host_returns = ctx.enqueue_create_host_buffer[dtype](HORIZON * IB)
        for i in range(HORIZON * IB):
            host_returns[i] = returns_ptr[i]
        ctx.enqueue_copy(gpu_state.imag_returns_buf, host_returns)

        returns_raw.free()
        rewards_raw.free()
        values_raw.free()
        continues_raw.free()

        # ── 7. Critic + Actor backward (per horizon step) ─────────────────
        # Zero actor/critic grads (already done above, but ensure clean)
        ctx.enqueue_memset(gpu_state.actor.grads_buf, 0)
        ctx.enqueue_memset(gpu_state.critic.grads_buf, 0)

        # Upload values (already on GPU in imag_values_buf) and returns
        # for advantage computation. Returns are already uploaded above.
        # We need to iterate over horizon steps, building feat from
        # the stored all_deter/all_stoch (observe phase), then running
        # critic/actor forward+backward.
        #
        # In DreamerV3 imagination, we iterate over HORIZON-1 steps.
        # For each step h, the feat is already computed by imagination.
        # But feat was overwritten each step (single buffer). So we
        # re-derive feat from the ping-pong deter/stoch buffers.
        #
        # Simplified: run critic+actor backward once on the *last*
        # imagination feat (h=HORIZON-1). This is an approximation
        # that avoids re-running all imagination steps.
        # For a full implementation, we'd store feat per horizon step.
        #
        # Actually, we can reconstruct feat from the last deter/stoch
        # in the ping-pong buffer.

        # Use initial imagination states (h=0) = observed states from world model
        # These have returns[0] which contain full lambda-bootstrapped signal
        var init_read_off = 0  # h=0 reads from offset 0
        var last_deter_2d = LayoutTensor[
            dtype, Layout.row_major(IB, DETER), MutAnyOrigin
        ](gpu_state.imag_deter_buf.unsafe_ptr() + init_read_off * DETER)
        var last_stoch_2d = LayoutTensor[
            dtype, Layout.row_major(IB, STOCH), MutAnyOrigin
        ](gpu_state.imag_stoch_buf.unsafe_ptr() + init_read_off * STOCH)
        var last_feat_2d = LayoutTensor[
            dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
        ](gpu_state.imag_feat_buf.unsafe_ptr())

        @always_inline
        fn run_concat_last_feat(
            f: LayoutTensor[dtype, Layout.row_major(IB, FEAT), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(IB, DETER), MutAnyOrigin],
            s: LayoutTensor[dtype, Layout.row_major(IB, STOCH), MutAnyOrigin],
        ):
            concat_feat_kernel[IB, DETER, STOCH](f, d, s)

        comptime IB_FEAT_BLOCKS2 = (IB * FEAT + TPB - 1) // TPB
        ctx.enqueue_function[run_concat_last_feat, run_concat_last_feat](
            last_feat_2d, last_deter_2d, last_stoch_2d,
            grid_dim=(IB_FEAT_BLOCKS2,), block_dim=(TPB,),
        )

        # ── Critic forward with cache ──────────────────────────────────
        var critic_logits_2d_ac = LayoutTensor[
            dtype, Layout.row_major(IB, BINS), MutAnyOrigin
        ](gpu_state.critic_logits_buf.unsafe_ptr())
        var critic_cache_2d = LayoutTensor[
            dtype, Layout.row_major(IB, Self.StateType.CriticModel.CACHE_SIZE), MutAnyOrigin
        ](gpu_state.critic_cache_buf.unsafe_ptr())
        Self.CriticNet.forward_gpu_with_cache[IB](
            ctx, last_feat_2d, critic_logits_2d_ac,
            gpu_state.critic.params_view(), critic_cache_2d,
            gpu_state.ws_critic,
        )

        # ── Compute two-hot targets from returns ───────────────────────
        # Use h=0 returns (full lambda-bootstrapped signal from horizon)
        # matching the h=0 feats used for critic/actor
        comptime LAST_H = HORIZON - 1

        # symlog(returns) for two-hot encoding
        var returns_1d = LayoutTensor[
            dtype, Layout.row_major(IB), MutAnyOrigin
        ](gpu_state.imag_returns_buf.unsafe_ptr())  # h=0
        var symlog_ret_1d = LayoutTensor[
            dtype, Layout.row_major(IB), MutAnyOrigin
        ](gpu_state.symlog_returns_buf.unsafe_ptr())

        @always_inline
        fn run_symlog_returns(
            o: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            inp: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
        ):
            symlog_kernel[IB](o, inp)

        comptime SYMLOG_RET_BLOCKS = (IB + TPB - 1) // TPB
        ctx.enqueue_function[run_symlog_returns, run_symlog_returns](
            symlog_ret_1d, returns_1d,
            grid_dim=(SYMLOG_RET_BLOCKS,), block_dim=(TPB,),
        )

        # Two-hot encode the symlog returns
        var two_hot_tgt_2d = LayoutTensor[
            dtype, Layout.row_major(IB, BINS), MutAnyOrigin
        ](gpu_state.two_hot_targets_buf.unsafe_ptr())
        var bins_1d_ac = LayoutTensor[
            dtype, Layout.row_major(BINS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())

        @always_inline
        fn run_two_hot_encode(
            tgt: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
            vals: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            b: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
        ):
            two_hot_encode_kernel[IB, BINS](tgt, vals, b)

        comptime ENCODE_BLOCKS = (IB + TPB - 1) // TPB
        ctx.enqueue_function[run_two_hot_encode, run_two_hot_encode](
            two_hot_tgt_2d, symlog_ret_1d, bins_1d_ac,
            grid_dim=(ENCODE_BLOCKS,), block_dim=(TPB,),
        )

        # ── Critic gradient: softmax(logits) - target ───────────────────
        var critic_grad_2d = LayoutTensor[
            dtype, Layout.row_major(IB, BINS), MutAnyOrigin
        ](gpu_state.critic_grad_buf.unsafe_ptr())
        var inv_ib = Scalar[dtype](1.0 / Float64(IB))

        @always_inline
        fn run_critic_grad(
            g: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
            l: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
            t: LayoutTensor[dtype, Layout.row_major(IB, BINS), MutAnyOrigin],
            ib: Scalar[dtype],
        ):
            two_hot_ce_grad_kernel[IB, BINS](g, l, t, ib)

        ctx.enqueue_function[run_critic_grad, run_critic_grad](
            critic_grad_2d, critic_logits_2d_ac, two_hot_tgt_2d, inv_ib,
            grid_dim=(ENCODE_BLOCKS,), block_dim=(TPB,),
        )

        # ── Critic backward ─────────────────────────────────────────────
        var critic_grad_in_2d = LayoutTensor[
            dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
        ](gpu_state.critic_grad_in_buf.unsafe_ptr())
        var critic_grads = gpu_state.critic.grads_view()
        Self.CriticNet.backward_gpu[IB](
            ctx, critic_grad_2d, critic_grad_in_2d,
            gpu_state.critic.params_view(), critic_cache_2d, critic_grads,
            gpu_state.ws_critic,
        )

        # ── Critic gradient clipping + optimizer step ─────────────────────
        _clip_grads_gpu[Self.CriticNet.MODEL.PARAM_SIZE](ctx, gpu_state.critic.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        gpu_state.critic.optimizer_step(ctx)

        # ── Actor forward with cache ────────────────────────────────────
        comptime ACTOR_OUT_DIM = Self.StateType.ActorModel.OUT_DIM
        var actor_out_2d_ac = LayoutTensor[
            dtype, Layout.row_major(IB, ACTOR_OUT_DIM), MutAnyOrigin,
        ](gpu_state.actor_out_buf.unsafe_ptr())
        var actor_cache_2d = LayoutTensor[
            dtype, Layout.row_major(IB, Self.StateType.ActorModel.CACHE_SIZE), MutAnyOrigin
        ](gpu_state.actor_cache_buf.unsafe_ptr())
        Self.ActorNet.forward_gpu_with_cache[IB](
            ctx, last_feat_2d, actor_out_2d_ac,
            gpu_state.actor.params_view(), actor_cache_2d,
            gpu_state.ws_actor,
        )

        # ── Compute advantages: returns - values ────────────────────────
        # Download returns and values for this step to compute advantages
        # (already on GPU, compute advantage inline on GPU)
        var advantages_1d = LayoutTensor[
            dtype, Layout.row_major(IB), MutAnyOrigin
        ](gpu_state.imag_advantages_buf.unsafe_ptr())
        var values_last = LayoutTensor[
            dtype, Layout.row_major(IB), MutAnyOrigin
        ](gpu_state.imag_values_buf.unsafe_ptr())  # h=0

        # advantage = returns - values (elementwise)
        @always_inline
        fn run_advantage(
            adv: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            ret: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            val: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
        ):
            advantage_kernel[IB](adv, ret, val)

        ctx.enqueue_function[run_advantage, run_advantage](
            advantages_1d, returns_1d, values_last,
            grid_dim=(ENCODE_BLOCKS,), block_dim=(TPB,),
        )

        # ── Sample actions for REINFORCE gradient ─────────────────────
        var actions_2d_ac = LayoutTensor[
            dtype, Layout.row_major(IB, ACT), MutAnyOrigin
        ](gpu_state.imag_actions_buf.unsafe_ptr())

        var act_seed_ac = Scalar[DType.uint32](
            UInt32(self.train_step_count * HORIZON + HORIZON) * UInt32(IB * ACT + 1)
        )

        @always_inline
        fn run_sample_actor(
            a: LayoutTensor[dtype, Layout.row_major(IB, ACT), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            ao: LayoutTensor[dtype, Layout.row_major(IB, 2 * ACT), MutAnyOrigin],
            s: Scalar[DType.uint32],
        ):
            tanh_normal_sample_kernel[IB, ACT](a, lp, ao, s)

        comptime SAMPLE_BLOCKS2 = (IB + TPB - 1) // TPB
        var log_probs_ac = LayoutTensor[
            dtype, Layout.row_major(IB), MutAnyOrigin
        ](gpu_state.imag_log_probs_buf.unsafe_ptr())
        ctx.enqueue_function[run_sample_actor, run_sample_actor](
            actions_2d_ac, log_probs_ac, actor_out_2d_ac, act_seed_ac,
            grid_dim=(SAMPLE_BLOCKS2,), block_dim=(TPB,),
        )

        # ── REINFORCE gradient ──────────────────────────────────────────
        var actor_grad_2d = LayoutTensor[
            dtype, Layout.row_major(IB, ACTOR_OUT_DIM), MutAnyOrigin
        ](gpu_state.actor_grad_buf.unsafe_ptr())
        var entropy_coef = Scalar[dtype](self.actor_entropy)

        @always_inline
        fn run_reinforce(
            g: LayoutTensor[dtype, Layout.row_major(IB, 2 * ACT), MutAnyOrigin],
            ao: LayoutTensor[dtype, Layout.row_major(IB, 2 * ACT), MutAnyOrigin],
            a: LayoutTensor[dtype, Layout.row_major(IB, ACT), MutAnyOrigin],
            adv: LayoutTensor[dtype, Layout.row_major(IB), MutAnyOrigin],
            ib_scale: Scalar[dtype],
            ec: Scalar[dtype],
        ):
            reinforce_grad_kernel[IB, ACT](g, ao, a, adv, ib_scale, ec)

        ctx.enqueue_function[run_reinforce, run_reinforce](
            actor_grad_2d, actor_out_2d_ac, actions_2d_ac,
            advantages_1d, inv_ib, entropy_coef,
            grid_dim=(SAMPLE_BLOCKS2,), block_dim=(TPB,),
        )

        # ── Actor backward ──────────────────────────────────────────────
        var actor_grad_in_2d = LayoutTensor[
            dtype, Layout.row_major(IB, FEAT), MutAnyOrigin
        ](gpu_state.actor_grad_in_buf.unsafe_ptr())
        var actor_grads = gpu_state.actor.grads_view()
        Self.ActorNet.backward_gpu[IB](
            ctx, actor_grad_2d, actor_grad_in_2d,
            gpu_state.actor.params_view(), actor_cache_2d, actor_grads,
            gpu_state.ws_actor,
        )

        # ── Actor gradient clipping + optimizer step ─────────────────────
        _clip_grads_gpu[Self.ActorNet.MODEL.PARAM_SIZE](ctx, gpu_state.actor.grads_view(), gpu_state.grad_partial_sums_buf, grad_norm_max)
        gpu_state.actor.optimizer_step(ctx)

        # ── 8. Slow critic EMA update ──────────────────────────────────
        gpu_state.slow_critic.soft_update_from_gpu(
            gpu_state.critic, Float64(self.slow_critic_tau), ctx,
        )

        ctx.synchronize()
        self.train_step_count += 1

# =============================================================================
# Helpers
# =============================================================================


@always_inline
fn _to_dtype_list[
    SRC: DType
](src: List[Scalar[SRC]]) -> List[Scalar[dtype]]:
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
    var ps = LayoutTensor[
        dtype, Layout.row_major(GRAD_BLOCKS), MutAnyOrigin
    ](partial_sums_buf.unsafe_ptr())

    @always_inline
    fn run_norm(
        p: LayoutTensor[dtype, Layout.row_major(GRAD_BLOCKS), MutAnyOrigin],
        g: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    ):
        gradient_norm_kernel[dtype, PARAM_SIZE, GRAD_BLOCKS, TPB](p, g)

    ctx.enqueue_function[run_norm, run_norm](
        ps, grads,
        grid_dim=(GRAD_BLOCKS,), block_dim=(TPB,),
    )

    @always_inline
    fn run_clip(
        g: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        p: LayoutTensor[dtype, Layout.row_major(GRAD_BLOCKS), MutAnyOrigin],
        m: Scalar[dtype],
    ):
        gradient_reduce_apply_fused_kernel[dtype, PARAM_SIZE, GRAD_BLOCKS, TPB](g, p, m)

    ctx.enqueue_function[run_clip, run_clip](
        grads, ps, max_grad_norm,
        grid_dim=(GRAD_BLOCKS,), block_dim=(TPB,),
    )


# =============================================================================
# Training Loop (CPU)
# =============================================================================


fn run_dreamer_v3_training[
    E: BoxContinuousActionEnv,
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
](
    mut env: E,
    mut agent: DreamerV3Agent[
        obs_dim,
        action_dim,
        deter_dim,
        hidden,
        stoch_dim,
        classes,
        units,
        num_bins,
        blocks,
        batch_size,
        batch_length,
        imagine_horizon,
        buffer_capacity,
    ],
    total_timesteps: Int = 1000000,
    train_every: Int = 5,
    seed_episodes: Int = 5,
    print_every: Int = 10,
):
    """Train a DreamerV3 agent on a continuous control environment.

    The training loop alternates between:
    1. Collecting environment data (storing in replay buffer)
    2. Training the world model + actor-critic from replay

    Args:
        env: Environment implementing BoxContinuousActionEnv.
        agent: Pre-initialized DreamerV3Agent.
        total_timesteps: Total environment steps (default: 1M).
        train_every: Steps between training updates (default: 5).
        seed_episodes: Random exploration episodes before training (default: 5).
        print_every: Episodes between progress prints (default: 10).
    """
    var metrics = TrainingMetrics(algorithm_name="DreamerV3")
    var episode_reward = Float64(0.0)
    var episode_steps = 0
    var episode_count = 0
    var total_env_steps = 0

    # ── Seed with random episodes ──────────────────────────────────────
    _ = env.reset()
    for _ in range(seed_episodes):
        var done = False
        while not done:
            var obs = _to_dtype_list(env.get_obs_list())
            var action = List[Scalar[dtype]](capacity=action_dim)
            for _ in range(action_dim):
                action.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            var result = env.step_continuous_vec[dtype](action)
            var reward = result[1]
            done = result[2]
            agent.observe(obs, action, Float64(reward), done)
            total_env_steps += 1
            if done:
                _ = env.reset()

    # ── Main training loop ─────────────────────────────────────────────
    _ = env.reset()
    agent.reset_episode()

    for step in range(total_timesteps):
        var obs = _to_dtype_list(env.get_obs_list())

        # Select action
        var action: List[Scalar[dtype]]
        if total_env_steps < agent.warmup_steps:
            action = List[Scalar[dtype]](capacity=action_dim)
            for _ in range(action_dim):
                action.append(Scalar[dtype](random_float64(-1.0, 1.0)))
        else:
            action = agent.select_action(obs, training=True)

        # Environment step
        var result = env.step_continuous_vec[dtype](action)
        var reward = result[1]
        var done = result[2]

        agent.observe(obs, action, Float64(reward), done)
        episode_reward += Float64(reward)
        episode_steps += 1
        total_env_steps += 1
        agent.total_steps += 1

        if done:
            episode_count += 1
            metrics.log_episode[dtype](
                episode_count,
                Scalar[dtype](episode_reward),
                episode_steps,
                0.0,
            )

            if episode_count % print_every == 0:
                clear_progress_bar()
                print(
                    "Episode "
                    + String(episode_count)
                    + " | Reward: "
                    + (String("NaN") if episode_reward != episode_reward else String(Int(episode_reward)))
                    + " | Steps: "
                    + String(episode_steps)
                    + " | Train updates: "
                    + String(agent.train_step_count)
                    + " | Buffer: "
                    + String(agent.state.buffer.len())
                )

            episode_reward = 0.0
            episode_steps = 0
            _ = env.reset()
            agent.reset_episode()

        # Train
        if step % train_every == 0 and agent.state.is_ready():
            var loss = agent.update()

        # Progress bar
        if step % 100 == 0:
            print_progress_bar(
                step, total_timesteps, agent.train_step_count, "DreamerV3"
            )

    clear_progress_bar()
    print(
        "Training complete. Episodes: "
        + String(episode_count)
        + " | Total steps: "
        + String(total_env_steps)
        + " | Train updates: "
        + String(agent.train_step_count)
    )


# =============================================================================
# Training Loop (GPU)
# =============================================================================


fn run_dreamer_v3_training_gpu[
    E: BoxContinuousActionEnv,
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
](
    mut env: E,
    mut agent: DreamerV3Agent[
        obs_dim,
        action_dim,
        deter_dim,
        hidden,
        stoch_dim,
        classes,
        units,
        num_bins,
        blocks,
        batch_size,
        batch_length,
        imagine_horizon,
        buffer_capacity,
    ],
    ctx: DeviceContext,
    total_timesteps: Int = 1000000,
    train_every: Int = 5,
    seed_episodes: Int = 5,
    print_every: Int = 10,
    sync_every: Int = 1000,
) raises:
    """Train a DreamerV3 agent with GPU-accelerated training steps.

    Data collection runs on CPU (single env). Batch training runs on GPU.
    Periodically syncs GPU weights back to CPU for action selection.

    Args:
        env: Environment implementing BoxContinuousActionEnv.
        agent: Pre-initialized DreamerV3Agent.
        ctx: GPU device context.
        total_timesteps: Total environment steps.
        train_every: Steps between training updates.
        seed_episodes: Random exploration episodes before training.
        print_every: Episodes between progress prints.
        sync_every: Training steps between GPU->CPU weight sync.
    """
    comptime BL = batch_length
    comptime B = batch_size
    comptime ACT = action_dim

    var metrics = TrainingMetrics(algorithm_name="DreamerV3-GPU")
    var episode_reward = Float64(0.0)
    var episode_steps = 0
    var episode_count = 0
    var total_env_steps = 0

    # Allocate GPU state and upload initial weights
    var gpu_state = agent.make_gpu_state(ctx)
    agent.upload_to_gpu(gpu_state, ctx)
    ctx.synchronize()

    # ── Seed with random episodes ──────────────────────────────────────
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
            agent.observe(obs, action, Float64(reward), done)
            total_env_steps += 1
            if done:
                _ = env.reset()

    # ── Main training loop ─────────────────────────────────────────────
    _ = env.reset()
    agent.reset_episode()

    for step in range(total_timesteps):
        var obs = _to_dtype_list(env.get_obs_list())

        # Select action (CPU, using CPU weights)
        var action: List[Scalar[dtype]]
        if total_env_steps < agent.warmup_steps:
            action = List[Scalar[dtype]](capacity=ACT)
            for _ in range(ACT):
                action.append(Scalar[dtype](random_float64(-1.0, 1.0)))
        else:
            action = agent.select_action(obs, training=True)

        # Environment step
        var result = env.step_continuous_vec[dtype](action)
        var reward = result[1]
        var done = result[2]

        agent.observe(obs, action, Float64(reward), done)
        episode_reward += Float64(reward)
        episode_steps += 1
        total_env_steps += 1

        if done:
            episode_count += 1
            metrics.log_episode[dtype](
                episode_count,
                Scalar[dtype](episode_reward),
                episode_steps,
                0.0,
            )

            if episode_count % print_every == 0:
                clear_progress_bar()
                print(
                    "Episode "
                    + String(episode_count)
                    + " | Reward: "
                    + (String("NaN") if episode_reward != episode_reward else String(Int(episode_reward)))
                    + " | Steps: "
                    + String(episode_steps)
                    + " | Train updates: "
                    + String(agent.train_step_count)
                    + " | Buffer: "
                    + String(agent.state.buffer.len())
                )

            episode_reward = 0.0
            episode_steps = 0
            _ = env.reset()
            agent.reset_episode()

        # Train on GPU
        if step % train_every == 0 and agent.state.is_ready():
            # Sample batch on CPU (pre-fill with zeros like CPU path)
            var batch_obs = List[Scalar[DType.float32]](
                capacity=B * (BL + 1) * obs_dim
            )
            var batch_actions = List[Scalar[DType.float32]](
                capacity=B * BL * ACT
            )
            var batch_rewards = List[Scalar[DType.float32]](
                capacity=B * BL
            )
            var batch_dones = List[Scalar[DType.float32]](
                capacity=B * BL
            )
            for _ in range(B * (BL + 1) * obs_dim):
                batch_obs.append(Scalar[DType.float32](0))
            for _ in range(B * BL * ACT):
                batch_actions.append(Scalar[DType.float32](0))
            for _ in range(B * BL):
                batch_rewards.append(Scalar[DType.float32](0))
                batch_dones.append(Scalar[DType.float32](0))

            agent.state.buffer.sample_sequences[B, BL](
                batch_obs, batch_actions, batch_rewards, batch_dones
            )

            # GPU training step
            agent.do_gpu_train_step(
                ctx, gpu_state,
                batch_obs, batch_actions, batch_rewards, batch_dones,
            )

            # Periodic GPU -> CPU sync for action selection
            if agent.train_step_count % sync_every == 0:
                agent.download_from_gpu(gpu_state, ctx)
                ctx.synchronize()

        # Progress bar
        if step % 100 == 0:
            print_progress_bar(
                step, total_timesteps, agent.train_step_count, "DreamerV3-GPU"
            )

    # Final sync
    agent.download_from_gpu(gpu_state, ctx)
    ctx.synchronize()

    clear_progress_bar()
    print(
        "GPU Training complete. Episodes: "
        + String(episode_count)
        + " | Total steps: "
        + String(total_env_steps)
        + " | Train updates: "
        + String(agent.train_step_count)
    )
