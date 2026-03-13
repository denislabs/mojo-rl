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
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.loss.two_hot import (
    compute_symlog_bins,
    two_hot_encode,
    decode_value,
    symlog,
    symexp,
)
from mojo_rl.deep_agents.core.utils import print_progress_bar, clear_progress_bar
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv
from .rssm import RSSM, categorical_sample, kl_divergence
from .state import DreamerV3CPUState
from .imagination import (
    compute_lambda_returns,
    normalize_returns,
    sample_tanh_normal,
    log_prob_tanh_normal,
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

    # ── State type alias ─────────────────────────────────────────────────
    comptime StateType = DreamerV3CPUState[
        Self.obs_dim, Self.action_dim, Self.deter_dim, Self.hidden,
        Self.stoch_dim, Self.classes, Self.units, Self.num_bins, Self.blocks,
    ]

    # ── Actor/Critic Network aliases (matching state.mojo definitions) ───
    comptime ActorNet = Network[
        Self.StateType.ActorModel, Adam[LR=3e-5]
    ]
    comptime CriticNet = Network[
        Self.StateType.CriticModel, Adam[LR=3e-5]
    ]

    # ── Core state ────────────────────────────────────────────────────────
    var state: Self.StateType

    # Hyperparameters
    var gamma: Float64
    var lambda_: Float64
    var kl_balance: Float64
    var actor_entropy: Float64
    var slow_critic_tau: Float64
    var return_norm_rate: Float64

    # Running state for inference (single env)
    var _current_deter: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var _current_stoch: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var _prev_action: UnsafePointer[Scalar[dtype], MutAnyOrigin]

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
        """
        self.state = Self.StateType()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_balance = kl_balance
        self.actor_entropy = actor_entropy
        self.slow_critic_tau = slow_critic_tau
        self.return_norm_rate = return_norm_rate
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
            obs_t, deter_t, stoch_t, action_t,
            new_deter_t, new_stoch_t, post_probs_t, prior_probs_t, feat_t,
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
        var batch_obs = List[Scalar[DType.float32]](
            capacity=B * (BL + 1) * OBS
        )
        var batch_actions = List[Scalar[DType.float32]](
            capacity=B * BL * ACT
        )
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
                obs_t, deter_t, stoch_t, act_t,
                new_deter_t, new_stoch_t, post_probs_t, prior_probs_t, feat_t,
                True,
            )

            # Store in all_* buffers for later use
            for b in range(B):
                for i in range(DETER):
                    (self.state._all_deter + t * B * DETER + b * DETER + i
                    )[] = (new_deter_ptr + b * DETER + i)[]
                for i in range(STOCH):
                    (self.state._all_stoch + t * B * STOCH + b * STOCH + i
                    )[] = (new_stoch_ptr + b * STOCH + i)[]
                    (self.state._all_post_probs + t * B * STOCH + b * STOCH
                     + i)[] = (post_probs_ptr + b * STOCH + i)[]
                    (self.state._all_prior_probs + t * B * STOCH + b * STOCH
                     + i)[] = (prior_probs_ptr + b * STOCH + i)[]
                for i in range(FEAT):
                    (self.state._all_feats + t * B * FEAT + b * FEAT + i
                    )[] = (feat_ptr + b * FEAT + i)[]

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
                            Float64(
                                rebind[Scalar[dtype]](rew_logits_t[b, k])
                            ) - max_logit
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
                    var cont_target = 1.0 - Float64(
                        batch_dones[b * BL + t]
                    )
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

        var total_wm_loss = obs_loss + rew_loss + cont_loss + (
            0.5 * dyn_kl_total + 0.1 * rep_kl_total
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
                dtype, Layout.row_major(IB, Self.StateType.ActorModel.OUT_DIM),
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
                    (self.state._imag_actions + h * IB * ACT + ib * ACT + a
                    )[] = Scalar[dtype](action_val)

                # Accumulate log prob over action dimensions
                var total_lp = Float64(0.0)
                for a in range(ACT):
                    var action_val = Float64(
                        (self.state._imag_actions + h * IB * ACT + ib * ACT
                         + a)[]
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
                    curr_deter_h, curr_stoch_h, actions_h,
                    next_deter_h, next_stoch_h, next_feat_h,
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
                imag_feat_h, critic_logits_h,
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
                var ret = Float64(
                    (self.state._imag_returns + h * IB + ib)[]
                )
                var val = Float64(
                    (self.state._imag_values + h * IB + ib)[]
                )
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
                imag_feat_h, critic_logits_h,
                self.state.critic.params_view(), cache_t,
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
                        Float64(
                            rebind[Scalar[dtype]](critic_logits_h[ib, k])
                        ) - max_logit
                    )

                # Gradient of cross-entropy w.r.t. logits: softmax(logit) - target
                for k in range(BINS):
                    var softmax_k = exp(
                        Float64(
                            rebind[Scalar[dtype]](critic_logits_h[ib, k])
                        ) - max_logit
                    ) / sum_exp
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
                grad_out_t, grad_in_t,
                self.state.critic.params_view(), cache_t,
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
                imag_feat_h, actor_out_h,
                self.state.actor.params_view(), actor_cache_t,
            )

            # Compute actor gradient: d(-advantage * log_prob) / d(actor_output)
            # This is a simplified gradient — we approximate the gradient of
            # log_prob w.r.t. actor outputs (mean, log_std)
            var actor_grad_ptr = alloc[Scalar[dtype]](IB * ACTOR_OUT)
            memset(actor_grad_ptr, 0, IB * ACTOR_OUT)

            for ib in range(IB):
                var ret = Float64(
                    (self.state._imag_returns + h * IB + ib)[]
                )
                var val = Float64(
                    (self.state._imag_values + h * IB + ib)[]
                )
                var advantage = ret - val
                var inv_ib = 1.0 / Float64(IB * (HORIZON - 1))

                for a in range(ACT):
                    var action_val = Float64(
                        (self.state._imag_actions + h * IB * ACT + ib * ACT
                         + a)[]
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
                    (actor_grad_ptr + ib * ACTOR_OUT + ACT + a
                    )[] = Scalar[dtype](weight * grad_log_std)

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
                actor_grad_t, actor_grad_in_t,
                self.state.actor.params_view(), actor_cache_t,
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
            feat, dec_out2_t, self.state.rssm.decoder.params_view(),
            dec_cache_t,
        )

        # Compute gradient: 2 * (pred - symlog(target)) / (B * OBS)
        var grad_out_ptr = alloc[Scalar[dtype]](B * OBS)
        var scale_factor = 2.0 / Float64(B * OBS)
        for b in range(B):
            for i in range(OBS):
                var obs_idx = b * (BL + 1) * OBS + (t + 1) * OBS + i
                var target = Float64(symlog(Float32(batch_obs[obs_idx])))
                var pred = Float64(
                    rebind[Scalar[dtype]](dec_out2_t[b, i])
                )
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
            grad_out_t, grad_in_t,
            self.state.rssm.decoder.params_view(), dec_cache_t,
            dec_grads,
        )

        dec_cache_ptr.free()
        dec_out2_ptr.free()
        grad_out_ptr.free()
        grad_in_ptr.free()


# =============================================================================
# Training Loop
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
        obs_dim, action_dim, deter_dim, hidden, stoch_dim, classes,
        units, num_bins, blocks, batch_size, batch_length, imagine_horizon,
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
    for ep in range(seed_episodes):
        var done = False
        while not done:
            var obs = env.get_obs_list()
            var action = List[Scalar[dtype]](capacity=action_dim)
            for _ in range(action_dim):
                action.append(
                    Scalar[dtype](random_float64(-1.0, 1.0))
                )
            var result = env.step_continuous_vec[dtype](action)
            var next_obs = result[0]
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
        var obs = env.get_obs_list()

        # Select action
        var action: List[Scalar[dtype]]
        if total_env_steps < agent.warmup_steps:
            action = List[Scalar[dtype]](capacity=action_dim)
            for _ in range(action_dim):
                action.append(
                    Scalar[dtype](random_float64(-1.0, 1.0))
                )
        else:
            action = agent.select_action(obs, training=True)

        # Environment step
        var result = env.step_continuous_vec[dtype](action)
        var next_obs = result[0]
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
                    "Episode " + String(episode_count)
                    + " | Reward: " + String(int(episode_reward))
                    + " | Steps: " + String(episode_steps)
                    + " | Train updates: "
                    + String(agent.train_step_count)
                    + " | Buffer: " + String(agent.state.buffer.len())
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
        "Training complete. Episodes: " + String(episode_count)
        + " | Total steps: " + String(total_env_steps)
        + " | Train updates: " + String(agent.train_step_count)
    )
