"""MuZero State Containers — CPU and GPU state for MuZero training.

CPU state (MuZeroCPUState):
  - Three networks: representation h, dynamics g, prediction f
  - SequenceReplayBuffer for streaming obs/act/rew/done
  - Additional storage for MCTS policies and values alongside replay data
  - Pre-allocated scratch buffers for K-step unrolled training

GPU state (MuZeroGPUState):
  - GPUNetworkState for all 3 networks
  - DeviceBuffer scratch for K-step unrolled training
  - Host buffers for CPU<->GPU data transfer

Created once in MuZeroAgent.__init__ / make_gpu_state.
"""

from std.memory import alloc, memset
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from mojo_rl.deep_agents.core.replay.gpu_sequence_replay_buffer import (
    GPUSequenceReplayBuffer,
)
from .configs import MuZeroConfig


struct MuZeroCPUState[
    Config: MuZeroConfig,
    _CAP: Int = 100000,
](Movable):
    """CPU-resident state for MuZero training.

    Holds all heap-allocated data needed for the MuZero training loop:
      - Representation network h: obs -> hidden state
      - Dynamics network g: (hidden, action) -> (next_hidden, reward_logits)
      - Prediction network f: hidden -> (policy_logits, value_logits)
      - Replay buffer with MCTS policy/value targets
      - Pre-allocated scratch for K-step unrolled training

    Parameters:
        Config: MuZeroConfig trait providing all dimensions, network types,
                and training hyperparameters.
        _CAP: Capacity of the replay buffer (default: 100000).
    """

    # ── Shorthand compile-time constants ─────────────────────────────────
    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime LATENT: Int = Self.Config.latent_dim
    comptime BINS: Int = Self.Config.num_bins
    comptime BATCH: Int = Self.Config.batch_size
    comptime K: Int = Self.Config.unroll_steps
    comptime N: Int = Self.Config.td_steps
    comptime DYN_IN: Int = Self.Config.DYN_IN
    comptime DYN_OUT: Int = Self.Config.DYN_OUT
    comptime PRED_OUT: Int = Self.Config.PRED_OUT
    # ── Network type aliases ─────────────────────────────────────────────
    comptime RepModel = Self.Config.RepModel
    comptime DynModel = Self.Config.DynModel
    comptime PredModel = Self.Config.PredModel
    comptime OptType = Self.Config.OptType

    # ── Core state ───────────────────────────────────────────────────────

    # Networks
    var representation: NetworkState[Self.RepModel, Self.OptType]
    var dynamics: NetworkState[Self.DynModel, Self.OptType]
    var prediction: NetworkState[Self.PredModel, Self.OptType]

    # Replay buffer (obs, actions, rewards, dones)
    var buffer: SequenceReplayBuffer[Self._CAP, Self.OBS, Self.ACT]

    # MCTS target storage — parallel arrays alongside replay buffer
    # Stores MCTS visit count policies and root values for training targets
    var mcts_policies: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [CAPACITY * ACT]
    var mcts_values: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [CAPACITY]

    # ── Batch sampling scratch ───────────────────────────────────────────
    # For K-step unroll training, we sample positions and extract windows
    var _batch_obs: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * (K+1) * OBS]
    var _batch_actions: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * K]  (discrete action indices)
    var _batch_rewards: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * K]
    var _batch_dones: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH * K]
    var _batch_policies: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * (K+1) * ACT]
    var _batch_values: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * (K+1)]

    # ── K-step unroll scratch ────────────────────────────────────────────
    # Hidden states through the unroll
    var _hidden_states: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * LATENT]

    # Prediction outputs at each step
    var _pred_outputs: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * PRED_OUT]

    # Dynamics outputs (reward logits) at each step
    var _dyn_reward_logits: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * BINS]

    # ── Network cache scratch (for backward pass) ────────────────────────
    var _rep_cache: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * RepModel.CACHE_SIZE]
    var _dyn_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * DynModel.CACHE_SIZE]
    var _pred_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * PredModel.CACHE_SIZE]

    # ── Gradient scratch ─────────────────────────────────────────────────
    var _grad_hidden: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * LATENT]
    var _grad_pred_out: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * PRED_OUT]
    var _grad_dyn_out: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * DYN_OUT]
    var _grad_dyn_in: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * DYN_IN]
    var _grad_rep_out: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * LATENT]
    var _grad_rep_in: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * OBS]

    # ── Value/Reward target scratch ──────────────────────────────────────
    var _value_targets: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH]
    var _reward_targets: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH]

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    def __init__(out self):
        """Allocate networks, replay buffer, and all scratch buffers."""

        # ── Networks ─────────────────────────────────────────────────────
        self.representation = NetworkState[Self.RepModel, Self.OptType]()
        self.representation.initialize[Kaiming[]]()

        self.dynamics = NetworkState[Self.DynModel, Self.OptType]()
        self.dynamics.initialize[Kaiming[]]()

        self.prediction = NetworkState[Self.PredModel, Self.OptType]()
        self.prediction.initialize[Kaiming[]]()

        # ── Replay buffer ────────────────────────────────────────────────
        self.buffer = SequenceReplayBuffer[Self._CAP, Self.OBS, Self.ACT]()

        # MCTS target parallel storage
        comptime POLICY_SIZE = Self._CAP * Self.ACT
        self.mcts_policies = alloc[Scalar[dtype]](POLICY_SIZE)
        memset(self.mcts_policies, 0, POLICY_SIZE)

        self.mcts_values = alloc[Scalar[dtype]](Self._CAP)
        memset(self.mcts_values, 0, Self._CAP)

        # ── Batch scratch ────────────────────────────────────────────────
        comptime BATCH_OBS_SIZE = Self.BATCH * (Self.K + 1) * Self.OBS
        self._batch_obs = alloc[Scalar[dtype]](BATCH_OBS_SIZE)
        memset(self._batch_obs, 0, BATCH_OBS_SIZE)

        comptime BATCH_ACT_SIZE = Self.BATCH * Self.K
        self._batch_actions = alloc[Scalar[dtype]](BATCH_ACT_SIZE)
        memset(self._batch_actions, 0, BATCH_ACT_SIZE)

        comptime BATCH_SCALAR_SIZE = Self.BATCH * Self.K
        self._batch_rewards = alloc[Scalar[dtype]](BATCH_SCALAR_SIZE)
        memset(self._batch_rewards, 0, BATCH_SCALAR_SIZE)

        self._batch_dones = alloc[Scalar[dtype]](BATCH_SCALAR_SIZE)
        memset(self._batch_dones, 0, BATCH_SCALAR_SIZE)

        comptime BATCH_POLICY_SIZE = Self.BATCH * (Self.K + 1) * Self.ACT
        self._batch_policies = alloc[Scalar[dtype]](BATCH_POLICY_SIZE)
        memset(self._batch_policies, 0, BATCH_POLICY_SIZE)

        comptime BATCH_VALUE_SIZE = Self.BATCH * (Self.K + 1)
        self._batch_values = alloc[Scalar[dtype]](BATCH_VALUE_SIZE)
        memset(self._batch_values, 0, BATCH_VALUE_SIZE)

        # ── Unroll scratch ───────────────────────────────────────────────
        comptime HIDDEN_SIZE = (Self.K + 1) * Self.BATCH * Self.LATENT
        self._hidden_states = alloc[Scalar[dtype]](HIDDEN_SIZE)
        memset(self._hidden_states, 0, HIDDEN_SIZE)

        comptime PRED_SIZE = (Self.K + 1) * Self.BATCH * Self.PRED_OUT
        self._pred_outputs = alloc[Scalar[dtype]](PRED_SIZE)
        memset(self._pred_outputs, 0, PRED_SIZE)

        comptime DYN_REW_SIZE = Self.K * Self.BATCH * Self.BINS
        self._dyn_reward_logits = alloc[Scalar[dtype]](DYN_REW_SIZE)
        memset(self._dyn_reward_logits, 0, DYN_REW_SIZE)

        # ── Cache scratch ────────────────────────────────────────────────
        comptime REP_CS = Self.RepModel.CACHE_SIZE
        comptime REP_CACHE_SIZE = Self.BATCH * REP_CS
        self._rep_cache = alloc[Scalar[dtype]](REP_CACHE_SIZE)
        memset(self._rep_cache, 0, REP_CACHE_SIZE)

        comptime DYN_CS = Self.DynModel.CACHE_SIZE
        comptime DYN_CACHE_SIZE = Self.K * Self.BATCH * DYN_CS
        self._dyn_caches = alloc[Scalar[dtype]](DYN_CACHE_SIZE)
        memset(self._dyn_caches, 0, DYN_CACHE_SIZE)

        comptime PRED_CS = Self.PredModel.CACHE_SIZE
        comptime PRED_CACHE_SIZE = (Self.K + 1) * Self.BATCH * PRED_CS
        self._pred_caches = alloc[Scalar[dtype]](PRED_CACHE_SIZE)
        memset(self._pred_caches, 0, PRED_CACHE_SIZE)

        # ── Gradient scratch ─────────────────────────────────────────────
        comptime GRAD_HIDDEN_SIZE = Self.BATCH * Self.LATENT
        self._grad_hidden = alloc[Scalar[dtype]](GRAD_HIDDEN_SIZE)
        memset(self._grad_hidden, 0, GRAD_HIDDEN_SIZE)

        comptime GRAD_PRED_SIZE = Self.BATCH * Self.PRED_OUT
        self._grad_pred_out = alloc[Scalar[dtype]](GRAD_PRED_SIZE)
        memset(self._grad_pred_out, 0, GRAD_PRED_SIZE)

        comptime GRAD_DYN_OUT_SIZE = Self.BATCH * Self.DYN_OUT
        self._grad_dyn_out = alloc[Scalar[dtype]](GRAD_DYN_OUT_SIZE)
        memset(self._grad_dyn_out, 0, GRAD_DYN_OUT_SIZE)

        comptime GRAD_DYN_IN_SIZE = Self.BATCH * Self.DYN_IN
        self._grad_dyn_in = alloc[Scalar[dtype]](GRAD_DYN_IN_SIZE)
        memset(self._grad_dyn_in, 0, GRAD_DYN_IN_SIZE)

        comptime GRAD_REP_SIZE = Self.BATCH * Self.LATENT
        self._grad_rep_out = alloc[Scalar[dtype]](GRAD_REP_SIZE)
        memset(self._grad_rep_out, 0, GRAD_REP_SIZE)

        comptime GRAD_REP_IN_SIZE = Self.BATCH * Self.OBS
        self._grad_rep_in = alloc[Scalar[dtype]](GRAD_REP_IN_SIZE)
        memset(self._grad_rep_in, 0, GRAD_REP_IN_SIZE)

        # ── Target scratch ───────────────────────────────────────────────
        comptime VAL_TARGET_SIZE = (Self.K + 1) * Self.BATCH
        self._value_targets = alloc[Scalar[dtype]](VAL_TARGET_SIZE)
        memset(self._value_targets, 0, VAL_TARGET_SIZE)

        comptime REW_TARGET_SIZE = Self.K * Self.BATCH
        self._reward_targets = alloc[Scalar[dtype]](REW_TARGET_SIZE)
        memset(self._reward_targets, 0, REW_TARGET_SIZE)

    def __init__(out self, *, deinit take: Self):
        """Move constructor — transfer ownership of all fields."""
        self.representation = take.representation^
        self.dynamics = take.dynamics^
        self.prediction = take.prediction^
        self.buffer = take.buffer^
        self.mcts_policies = take.mcts_policies
        self.mcts_values = take.mcts_values
        self._batch_obs = take._batch_obs
        self._batch_actions = take._batch_actions
        self._batch_rewards = take._batch_rewards
        self._batch_dones = take._batch_dones
        self._batch_policies = take._batch_policies
        self._batch_values = take._batch_values
        self._hidden_states = take._hidden_states
        self._pred_outputs = take._pred_outputs
        self._dyn_reward_logits = take._dyn_reward_logits
        self._rep_cache = take._rep_cache
        self._dyn_caches = take._dyn_caches
        self._pred_caches = take._pred_caches
        self._grad_hidden = take._grad_hidden
        self._grad_pred_out = take._grad_pred_out
        self._grad_dyn_out = take._grad_dyn_out
        self._grad_dyn_in = take._grad_dyn_in
        self._grad_rep_out = take._grad_rep_out
        self._grad_rep_in = take._grad_rep_in
        self._value_targets = take._value_targets
        self._reward_targets = take._reward_targets

    def __del__(deinit self):
        """Free all heap-allocated buffers."""
        self.mcts_policies.free()
        self.mcts_values.free()
        self._batch_obs.free()
        self._batch_actions.free()
        self._batch_rewards.free()
        self._batch_dones.free()
        self._batch_policies.free()
        self._batch_values.free()
        self._hidden_states.free()
        self._pred_outputs.free()
        self._dyn_reward_logits.free()
        self._rep_cache.free()
        self._dyn_caches.free()
        self._pred_caches.free()
        self._grad_hidden.free()
        self._grad_pred_out.free()
        self._grad_dyn_out.free()
        self._grad_dyn_in.free()
        self._grad_rep_out.free()
        self._grad_rep_in.free()
        self._value_targets.free()
        self._reward_targets.free()

    # ══════════════════════════════════════════════════════════════════════
    # Buffer Readiness
    # ══════════════════════════════════════════════════════════════════════

    def is_ready(self) -> Bool:
        """Check if the replay buffer has enough data for training.

        Returns:
            True if buffer has more samples than BATCH_SIZE and enough
            for K-step unrolling.
        """
        return (
            self.buffer.len() > Self.BATCH * 2
            and self.buffer.len() > Self.K + Self.N + 1
        )

    # ══════════════════════════════════════════════════════════════════════
    # Sampling with MCTS Targets
    # ══════════════════════════════════════════════════════════════════════

    def sample_batch_with_targets(
        mut self,
        gamma: Float64,
    ):
        """Sample BATCH sequences and extract MCTS policy/value targets.

        Fills all _batch_* scratch buffers including _batch_policies and
        _batch_values. Also computes n-step bootstrapped value targets
        and reward targets (scalar-transformed).

        This replaces the separate sample_sequences + _compute_targets flow
        with a single method that correctly aligns MCTS targets with
        sampled buffer positions.

        Args:
            gamma: Discount factor for n-step returns.
        """
        comptime BATCH = Self.BATCH
        comptime K = Self.K
        comptime N = Self.N
        comptime OBS = Self.OBS
        comptime ACT = Self.ACT
        comptime CAPACITY = Self._CAP

        var sampled = 0
        var max_attempts = BATCH * 100
        var attempts = 0

        while sampled < BATCH and attempts < max_attempts:
            attempts += 1

            # Random starting position in the valid range
            var start = (
                Int(random_float64() * Float64(self.buffer.size))
                % self.buffer.size
            )
            var actual_start = (
                self.buffer.ptr - self.buffer.size + start
            ) % CAPACITY
            if actual_start < 0:
                actual_start += CAPACITY

            # Need K+1 observations (K steps) minimum
            if self.buffer.size < K + 1:
                continue

            # Check for episode boundaries within the K-step window
            if not self.buffer._is_valid_sequence_start(actual_start, K):
                continue

            # Verify end index is within recorded data
            var end_idx = (actual_start + K) % CAPACITY
            var end_age = (self.buffer.ptr - end_idx - 1 + CAPACITY) % CAPACITY
            if end_age >= self.buffer.size:
                continue

            # ── Copy sequence data ───────────────────────────────────
            var b = sampled
            var obs_off = b * (K + 1) * OBS
            var act_off = b * K * ACT
            var rew_off = b * K
            var don_off = b * K
            var pol_off = b * (K + 1) * ACT
            var val_off = b * (K + 1)

            # Copy K+1 observations and their MCTS targets
            for t in range(K + 1):
                var idx = (actual_start + t) % CAPACITY

                # Observations
                var obs_start = obs_off + t * OBS
                for i in range(OBS):
                    self._batch_obs[obs_start + i] = Scalar[dtype](
                        self.buffer.obs[idx * OBS + i]
                    )

                # MCTS policy targets (K+1 policies)
                var pol_start = pol_off + t * ACT
                for a in range(ACT):
                    self._batch_policies[pol_start + a] = self.mcts_policies[
                        idx * ACT + a
                    ]

                # MCTS value targets (K+1 values)
                self._batch_values[val_off + t] = self.mcts_values[idx]

            # Copy K actions, rewards, dones
            for t in range(K):
                var idx = (actual_start + t) % CAPACITY

                # Actions (one-hot encoded in buffer)
                var act_start = act_off + t * ACT
                for a in range(ACT):
                    self._batch_actions[act_start + a] = Scalar[dtype](
                        self.buffer.actions[idx * ACT + a]
                    )

                # Rewards and dones
                self._batch_rewards[rew_off + t] = Scalar[dtype](
                    self.buffer.rewards[idx]
                )
                self._batch_dones[don_off + t] = Scalar[dtype](
                    self.buffer.dones[idx]
                )

            # ── Compute n-step value targets for this sample ─────────
            # z_value(t+k) = sum_{i=0}^{min(n,T-t-k)-1} gamma^i * r_{t+k+i}
            #                + gamma^n * v_{t+k+n}  (if not terminal)
            for k in range(K + 1):
                var base_idx = (actual_start + k) % CAPACITY

                var n_step_return = Float64(0.0)
                var gamma_power = Float64(1.0)
                var steps_used = 0
                var hit_terminal = False

                for i in range(N):
                    var step_idx = (base_idx + i) % CAPACITY
                    # Check if this step is within valid buffer range
                    var step_age = (
                        self.buffer.ptr - step_idx - 1 + CAPACITY
                    ) % CAPACITY
                    if step_age >= self.buffer.size:
                        break

                    n_step_return += gamma_power * Float64(
                        self.buffer.rewards[step_idx]
                    )
                    gamma_power *= gamma
                    steps_used += 1

                    # Check for terminal
                    if Float64(self.buffer.dones[step_idx]) > 0.5:
                        hit_terminal = True
                        break

                # Bootstrap with MCTS root value if not terminal
                if not hit_terminal and steps_used == N:
                    var bootstrap_idx = (base_idx + N) % CAPACITY
                    var boot_age = (
                        self.buffer.ptr - bootstrap_idx - 1 + CAPACITY
                    ) % CAPACITY
                    if boot_age < self.buffer.size:
                        n_step_return += gamma_power * Float64(
                            self.mcts_values[bootstrap_idx]
                        )

                self._value_targets[k * BATCH + b] = Scalar[dtype](
                    n_step_return
                )

            # ── Reward targets (actual rewards, no transform yet) ────
            for k in range(K):
                self._reward_targets[k * BATCH + b] = self._batch_rewards[
                    rew_off + k
                ]

            sampled += 1

        # Zero-fill any unsampled batch slots
        for b in range(sampled, BATCH):
            var obs_off = b * (K + 1) * OBS
            for i in range((K + 1) * OBS):
                self._batch_obs[obs_off + i] = Scalar[dtype](0.0)
            var act_off = b * K * ACT
            for i in range(K * ACT):
                self._batch_actions[act_off + i] = Scalar[dtype](0.0)
            for i in range(K):
                self._batch_rewards[b * K + i] = Scalar[dtype](0.0)
                self._batch_dones[b * K + i] = Scalar[dtype](0.0)
            var pol_off = b * (K + 1) * ACT
            for i in range((K + 1) * ACT):
                self._batch_policies[pol_off + i] = Scalar[dtype](
                    1.0 / Float64(ACT)
                )
            for i in range(K + 1):
                self._batch_values[b * (K + 1) + i] = Scalar[dtype](0.0)
                self._value_targets[i * BATCH + b] = Scalar[dtype](0.0)
            for i in range(K):
                self._reward_targets[i * BATCH + b] = Scalar[dtype](0.0)


# ══════════════════════════════════════════════════════════════════════════
# GPU State
# ══════════════════════════════════════════════════════════════════════════


struct MuZeroGPUState[
    Config: MuZeroConfig, N_ENVS: Int = 64, PER_ENV_CAP: Int = 1000
](Movable):
    """GPU-resident state for MuZero training.

    Holds GPUNetworkState for all three networks and DeviceBuffers
    for all training scratch. Created by MuZeroAgent.make_gpu_state().
    """

    # ── Shorthand compile-time constants ─────────────────────────────
    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime LATENT: Int = Self.Config.latent_dim
    comptime BINS: Int = Self.Config.num_bins
    comptime BATCH: Int = Self.Config.batch_size
    comptime K: Int = Self.Config.unroll_steps
    comptime DYN_IN: Int = Self.Config.DYN_IN
    comptime DYN_OUT: Int = Self.Config.DYN_OUT
    comptime PRED_OUT: Int = Self.Config.PRED_OUT

    # Network type aliases (from Config)
    comptime RepModel = Self.Config.RepModel
    comptime DynModel = Self.Config.DynModel
    comptime PredModel = Self.Config.PredModel
    comptime OptType = Self.Config.OptType

    # ── GPU Network States ───────────────────────────────────────────
    var representation: GPUNetworkState[Self.RepModel, Self.OptType]
    var dynamics: GPUNetworkState[Self.DynModel, Self.OptType]
    var prediction: GPUNetworkState[Self.PredModel, Self.OptType]

    # ── GPU Sequence Replay Buffer ───────────────────────────────────
    var replay: GPUSequenceReplayBuffer[
        Self.PER_ENV_CAP, Self.OBS, Self.ACT, Self.N_ENVS
    ]

    # ── MCTS Target Buffers (parallel to replay, same per-env layout) ─
    var mcts_policy_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP * ACT]
    var mcts_value_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP]

    # Per-step MCTS target staging buffers (for CPU→GPU upload per step)
    var mcts_step_policy_buf: DeviceBuffer[dtype]  # [N_ENVS * ACT]
    var mcts_step_value_buf: DeviceBuffer[dtype]  # [N_ENVS]

    # Host buffers for uploading MCTS targets from CPU
    var mcts_policy_host: HostBuffer[dtype]  # [N_ENVS * ACT]
    var mcts_value_host: HostBuffer[dtype]  # [N_ENVS]

    # ── Batch data (sampled on GPU) ──────────────────────────────────
    var batch_obs_buf: DeviceBuffer[dtype]  # [BATCH * (K+1) * OBS]
    var batch_actions_buf: DeviceBuffer[dtype]  # [BATCH * K * ACT]
    var batch_policies_buf: DeviceBuffer[dtype]  # [BATCH * (K+1) * ACT]

    # ── Value/Reward target scratch ──────────────────────────────────
    var value_targets_buf: DeviceBuffer[
        dtype
    ]  # [(K+1) * BATCH]  (scalar-transformed)
    var reward_targets_buf: DeviceBuffer[
        dtype
    ]  # [K * BATCH]      (scalar-transformed)
    var value_target_dist_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * BINS] (two-hot encoded)
    var reward_target_dist_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * BINS] (two-hot encoded)

    # ── K-step unroll scratch ────────────────────────────────────────
    var hidden_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LATENT]
    var pred_out_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * PRED_OUT] (reused per step)
    var dyn_input_buf: DeviceBuffer[dtype]  # [BATCH * DYN_IN]
    var dyn_output_buf: DeviceBuffer[dtype]  # [BATCH * DYN_OUT]

    # ── Network cache (for backward) ────────────────────────────────
    var rep_cache_buf: DeviceBuffer[dtype]  # [BATCH * RepModel.CACHE_SIZE]
    var dyn_cache_buf: DeviceBuffer[dtype]  # [K * BATCH * DynModel.CACHE_SIZE]
    var pred_cache_buf: DeviceBuffer[
        dtype
    ]  # [(K+1) * BATCH * PredModel.CACHE_SIZE]

    # ── Gradient scratch ─────────────────────────────────────────────
    var grad_pred_out_buf: DeviceBuffer[dtype]  # [BATCH * PRED_OUT]
    var grad_pred_in_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_dyn_out_buf: DeviceBuffer[dtype]  # [BATCH * DYN_OUT]
    var grad_dyn_in_buf: DeviceBuffer[dtype]  # [BATCH * DYN_IN]
    var grad_hidden_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_rep_out_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_rep_in_buf: DeviceBuffer[dtype]  # [BATCH * OBS]

    # ── Network workspace (for forward/backward GPU) ─────────────────
    var workspace_buf: DeviceBuffer[dtype]  # max workspace across all networks

    # ── Host transfer buffers ────────────────────────────────────────
    var batch_obs_host: HostBuffer[dtype]
    var batch_actions_host: HostBuffer[dtype]
    var batch_policies_host: HostBuffer[dtype]
    var value_targets_host: HostBuffer[dtype]
    var reward_targets_host: HostBuffer[dtype]

    # ══════════════════════════════════════════════════════════════════
    # Constructor
    # ══════════════════════════════════════════════════════════════════

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers and network states."""

        # ── Networks ─────────────────────────────────────────────────
        self.representation = GPUNetworkState[Self.RepModel, Self.OptType](ctx)
        self.dynamics = GPUNetworkState[Self.DynModel, Self.OptType](ctx)
        self.prediction = GPUNetworkState[Self.PredModel, Self.OptType](ctx)

        # ── GPU Replay Buffer ───────────────────────────────────────
        self.replay = GPUSequenceReplayBuffer[
            Self.PER_ENV_CAP, Self.OBS, Self.ACT, Self.N_ENVS
        ](ctx)

        # ── MCTS Target Buffers ─────────────────────────────────────
        self.mcts_policy_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP * Self.ACT
        )
        self.mcts_value_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP
        )
        ctx.enqueue_memset(self.mcts_policy_buf, 0)
        ctx.enqueue_memset(self.mcts_value_buf, 0)

        self.mcts_step_policy_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.mcts_step_value_buf = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        self.mcts_policy_host = ctx.enqueue_create_host_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.mcts_value_host = ctx.enqueue_create_host_buffer[dtype](
            Self.N_ENVS
        )

        # ── Batch data ───────────────────────────────────────────────
        self.batch_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.OBS
        )
        self.batch_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.K * Self.ACT
        )
        self.batch_policies_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.ACT
        )

        # ── Targets ──────────────────────────────────────────────────
        self.value_targets_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH
        )
        self.reward_targets_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH
        )
        self.value_target_dist_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.BINS
        )
        self.reward_target_dist_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.BINS
        )

        # ── Unroll scratch ───────────────────────────────────────────
        self.hidden_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * Self.LATENT
        )
        self.pred_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PRED_OUT
        )
        self.dyn_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_IN
        )
        self.dyn_output_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_OUT
        )

        # ── Cache ────────────────────────────────────────────────────
        comptime REP_CS = Self.RepModel.CACHE_SIZE
        self.rep_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * REP_CS
        )
        comptime DYN_CS = Self.DynModel.CACHE_SIZE
        self.dyn_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * DYN_CS
        )
        comptime PRED_CS = Self.PredModel.CACHE_SIZE
        self.pred_cache_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * PRED_CS
        )

        # ── Gradient scratch ─────────────────────────────────────────
        self.grad_pred_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PRED_OUT
        )
        self.grad_pred_in_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_dyn_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_OUT
        )
        self.grad_dyn_in_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_IN
        )
        self.grad_hidden_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_rep_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_rep_in_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.OBS
        )

        # ── Workspace (max across all networks) ─────────────────────
        comptime WS_REP = Self.RepModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_DYN = Self.DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_PRED = Self.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime MAX_WS = WS_REP if WS_REP > WS_DYN else WS_DYN
        comptime MAX_WS2 = MAX_WS if MAX_WS > WS_PRED else WS_PRED
        comptime WS_TOTAL = Self.BATCH * MAX_WS2 if MAX_WS2 > 0 else 1
        self.workspace_buf = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

        # ── Host transfer buffers ────────────────────────────────────
        self.batch_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.OBS
        )
        self.batch_actions_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.K * Self.ACT
        )
        self.batch_policies_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.ACT
        )
        self.value_targets_host = ctx.enqueue_create_host_buffer[dtype](
            (Self.K + 1) * Self.BATCH
        )
        self.reward_targets_host = ctx.enqueue_create_host_buffer[dtype](
            Self.K * Self.BATCH
        )

    def __init__(out self, *, deinit take: Self):
        """Move constructor."""
        self.representation = take.representation^
        self.dynamics = take.dynamics^
        self.prediction = take.prediction^
        self.replay = take.replay^
        self.mcts_policy_buf = take.mcts_policy_buf^
        self.mcts_value_buf = take.mcts_value_buf^
        self.mcts_step_policy_buf = take.mcts_step_policy_buf^
        self.mcts_step_value_buf = take.mcts_step_value_buf^
        self.mcts_policy_host = take.mcts_policy_host^
        self.mcts_value_host = take.mcts_value_host^
        self.batch_obs_buf = take.batch_obs_buf^
        self.batch_actions_buf = take.batch_actions_buf^
        self.batch_policies_buf = take.batch_policies_buf^
        self.value_targets_buf = take.value_targets_buf^
        self.reward_targets_buf = take.reward_targets_buf^
        self.value_target_dist_buf = take.value_target_dist_buf^
        self.reward_target_dist_buf = take.reward_target_dist_buf^
        self.hidden_buf = take.hidden_buf^
        self.pred_out_buf = take.pred_out_buf^
        self.dyn_input_buf = take.dyn_input_buf^
        self.dyn_output_buf = take.dyn_output_buf^
        self.rep_cache_buf = take.rep_cache_buf^
        self.dyn_cache_buf = take.dyn_cache_buf^
        self.pred_cache_buf = take.pred_cache_buf^
        self.grad_pred_out_buf = take.grad_pred_out_buf^
        self.grad_pred_in_buf = take.grad_pred_in_buf^
        self.grad_dyn_out_buf = take.grad_dyn_out_buf^
        self.grad_dyn_in_buf = take.grad_dyn_in_buf^
        self.grad_hidden_buf = take.grad_hidden_buf^
        self.grad_rep_out_buf = take.grad_rep_out_buf^
        self.grad_rep_in_buf = take.grad_rep_in_buf^
        self.workspace_buf = take.workspace_buf^
        self.batch_obs_host = take.batch_obs_host^
        self.batch_actions_host = take.batch_actions_host^
        self.batch_policies_host = take.batch_policies_host^
        self.value_targets_host = take.value_targets_host^
        self.reward_targets_host = take.reward_targets_host^

    def upload_from[
        _C: Int
    ](
        mut self,
        cpu: MuZeroCPUState[Self.Config, _C],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU network params to GPU."""
        self.representation.upload_from(cpu.representation, ctx)
        self.dynamics.upload_from(cpu.dynamics, ctx)
        self.prediction.upload_from(cpu.prediction, ctx)

    def download_to[
        _C: Int
    ](
        mut self,
        mut cpu: MuZeroCPUState[Self.Config, _C],
        ctx: DeviceContext,
    ) raises:
        """Download GPU network params to CPU."""
        self.representation.download_to(cpu.representation, ctx)
        self.dynamics.download_to(cpu.dynamics, ctx)
        self.prediction.download_to(cpu.prediction, ctx)
