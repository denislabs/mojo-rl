"""GenericEfficientZeroV2Agent — discrete-action variant.

Phase 2 / Step 5a: inference + episode-buffer management only.
`train()` is a stub that prints a "not implemented" message; the K-step
BPTT + composite loss assembly lands in step 5b. This intermediate stage
lets us confirm that the agent's search → action → episode-buffer →
replay-flush plumbing is correct end-to-end, with the existing GumbelMCTS
machinery from Phase 1, before adding the (substantial) backward pass.

Action sampling, in line with Gumbel-MuZero / EZ-V2:
    π̂ = softmax(logits + σ(completed_Q))           [from GumbelMCTS]
    action ~ π̂^{1/T} / norm                        [training, temperature T]
    action = argmax π̂                              [eval, T → 0]

The "root value" returned alongside the policy is the SVE estimate
(Σ total_value / Σ visit_count at the root) — this is what the value
target for `obs[t]` will be once `train()` lands. We stash it now for
later replay sampling.
"""

from std.math import exp, log, sqrt
from std.memory import alloc, memset
from std.random import random_float64
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import LSTMCell
from mojo_rl.nn.training import Network
from mojo_rl.deep_agents.efficient_zero_v2.configs import (
    EZV2DiscreteConfig,
    VALUE_TARGET_SEARCH,
    VALUE_TARGET_SARSA,
    VALUE_TARGET_MIXED,
)
from mojo_rl.deep_agents.efficient_zero_v2.networks import RewardPrefixHeadMLP
from mojo_rl.deep_agents.efficient_zero_v2.state import (
    EZV2DiscreteCPUState,
    EZV2DiscreteGPUState,
)
from mojo_rl.deep_agents.efficient_zero_v2.mcts import GumbelMCTS
from mojo_rl.deep_agents.efficient_zero_v2.strategies import (
    compute_sve,
    MixedValueTarget,
)
from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_copy_obs_at_step_kernel,
    ezv2_build_dyn_input_kernel,
    ezv2_extract_hidden_after_dyn_kernel,
    ezv2_policy_loss_grad_kernel,
    ezv2_value_loss_grad_kernel,
    ezv2_reward_loss_grad_kernel,
    ezv2_cosine_loss_grad_kernel,
    ezv2_reduce_add_kernel,
    ezv2_add_kernel,
    ezv2_assemble_grad_dyn_step_kernel,
    ezv2_accumulate_dyn_grad_in_kernel,
    ezv2_gather_reward_at_step_kernel,
    ezv2_gather_value_target_kernel,
    ezv2_gather_policy_target_kernel,
    ezv2_priority_from_v_loss_kernel,
    ezv2_copy_lstm_input_kernel,
    ezv2_reward_prefix_loss_grad_kernel,
)
from mojo_rl.deep_agents.muzero.utils import (
    scalar_transform,
    encode_categorical,
    decode_categorical,
    inverse_scalar_transform,
    cross_entropy_with_softmax,
)
from mojo_rl.deep_agents.muzero.kernels import (
    scalar_transform_kernel,
    two_hot_encode_kernel,
)
from mojo_rl.deep_agents.core.kernels import (
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
)


def _clip_grads_inplace(
    grads: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    max_norm: Float64,
):
    """Clip gradient L2-norm in-place to `max_norm`.

    Computes `norm = sqrt(Σ grads[i]^2)`, then if `norm > max_norm`
    multiplies every entry by `max_norm / (norm + 1e-8)`. No-op when
    `norm <= max_norm`. Matches PyTorch's `clip_grad_norm_` per-parameter-
    group semantics.
    """
    var sq_sum = Float64(0.0)
    for i in range(n):
        var g = Float64(grads[i])
        sq_sum += g * g
    var norm = sqrt(sq_sum)
    if norm <= max_norm:
        return
    var scale = max_norm / (norm + Float64(1e-8))
    for i in range(n):
        grads[i] = Scalar[dtype](Float64(grads[i]) * scale)


struct GenericEfficientZeroV2Agent[Config: EZV2DiscreteConfig](Movable):
    """EZ-V2 agent (CPU, discrete actions).

    Holds the five-network state + a single-tree GumbelMCTS engine + the
    per-episode buffers needed to populate the replay buffer with MCTS
    targets at episode-end.

    Parameters:
        Config: `EZV2DiscreteConfig` providing dimensions, networks,
            optimizer, training hyperparams, and EZ-V2 loss weights.
    """

    comptime ACT: Int = Self.Config.action_dim

    # Networks + replay live in this state. We accept the
    # `EZV2DiscreteCPUState` default `_CAP=50000` rather than threading
    # `Self.Config.buffer_capacity` through — Mojo nightly's type checker
    # treats `Config.buffer_capacity` as a distinct type from its
    # numerical equivalent, which breaks downstream alias unification.
    var state: EZV2DiscreteCPUState[Self.Config]

    # GumbelMCTS engine (re-used across calls; resets internally each search).
    var mcts: GumbelMCTS[
        Self.ACT,
        Self.Config.latent_dim,
        Self.Config.num_bins,
        Self.Config.num_simulations,
        Self.Config.num_root_candidates,
        Self.Config.max_nodes,
    ]

    # Hyperparameters
    var gamma: Float64
    var v_min: Float64
    var v_max: Float64
    var temperature: Float64
    var temperature_decay_steps: Int
    var max_grad_norm: Float64

    # Counters
    var total_steps: Int
    var train_step_count: Int

    # Running max of per-transition priority. Used as the default for
    # newly-stored transitions (paper App. A: "fresh transitions get
    # max-seen priority so they're guaranteed to be sampled at least
    # once") and bumped after each `train_step` whenever a sampled
    # window's per-sample value-CE loss exceeds the current max.
    var max_priority: Float64

    # Number of parallel envs serviced by this agent. Single-env runs
    # use n_envs=1 (default) and pass `env_id=0` everywhere — identical
    # behavior to the pre-multi-env code path.
    var n_envs: Int

    # Per-episode buffers, parallelized across envs — flushed to
    # replay at done. Outer index = env_id, inner = transition index.
    var _episode_obs: List[List[List[Scalar[dtype]]]]
    var _episode_actions: List[List[Int]]
    var _episode_rewards: List[List[Float64]]
    var _episode_policies: List[List[InlineArray[Float64, Self.ACT]]]
    var _episode_values: List[List[Float64]]

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    def __init__(
        out self,
        gamma: Float64 = 0.997,
        v_min: Float64 = -50.0,
        v_max: Float64 = 50.0,
        temperature: Float64 = 1.0,
        temperature_decay_steps: Int = 50000,
        max_grad_norm: Float64 = 5.0,
        n_envs: Int = 1,
    ):
        self.state = EZV2DiscreteCPUState[Self.Config]()
        self.mcts = GumbelMCTS[
            Self.ACT,
            Self.Config.latent_dim,
            Self.Config.num_bins,
            Self.Config.num_simulations,
            Self.Config.num_root_candidates,
            Self.Config.max_nodes,
        ](gamma=gamma)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.temperature = temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.max_grad_norm = max_grad_norm
        self.total_steps = 0
        self.train_step_count = 0
        # Initial value matches a typical untrained value-CE loss
        # magnitude on a 51-bin support — high enough that fresh
        # transitions are competitive with already-seen ones in the
        # priority distribution. Updated dynamically by train_step.
        self.max_priority = 1.0

        # Multi-env Phase A: outer list indexed by env_id, sized at
        # construction. n_envs=1 reproduces single-env behavior.
        self.n_envs = n_envs if n_envs > 0 else 1
        self._episode_obs = List[List[List[Scalar[dtype]]]]()
        self._episode_actions = List[List[Int]]()
        self._episode_rewards = List[List[Float64]]()
        self._episode_policies = List[
            List[InlineArray[Float64, Self.ACT]]
        ]()
        self._episode_values = List[List[Float64]]()
        for _ in range(self.n_envs):
            self._episode_obs.append(List[List[Scalar[dtype]]]())
            self._episode_actions.append(List[Int]())
            self._episode_rewards.append(List[Float64]())
            self._episode_policies.append(
                List[InlineArray[Float64, Self.ACT]]()
            )
            self._episode_values.append(List[Float64]())

        # Sync target networks to online at startup so the first reanalyze
        # uses a meaningful (if still untrained) target rather than an
        # independently-initialized random network.
        self.update_target_networks(tau=1.0)

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.mcts = take.mcts^
        self.gamma = take.gamma
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.temperature = take.temperature
        self.temperature_decay_steps = take.temperature_decay_steps
        self.max_grad_norm = take.max_grad_norm
        self.total_steps = take.total_steps
        self.train_step_count = take.train_step_count
        self.max_priority = take.max_priority
        self.n_envs = take.n_envs
        self._episode_obs = take._episode_obs^
        self._episode_actions = take._episode_actions^
        self._episode_rewards = take._episode_rewards^
        self._episode_policies = take._episode_policies^
        self._episode_values = take._episode_values^

    # ══════════════════════════════════════════════════════════════════════
    # Action selection
    # ══════════════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
        legal_mask: List[Bool] = List[Bool](),
    ) -> Tuple[Int, InlineArray[Float64, Self.ACT], Float64]:
        """Run GumbelMCTS, return (action, improved_policy, SVE root value).

        Action sampling:
            • eval (training=False) or temperature < 0.01 → argmax.
            • otherwise → multinomial sample from `π̂^{1/T} / norm` where
              π̂ is the GumbelMCTS improved policy.

        Returns:
            (action, π̂, SVE).
        """
        # GumbelMCTS owns the alias-bound network types; pass through directly.
        var policy = self.mcts.search(
            obs,
            self.state.representation,
            self.state.dynamics,
            self.state.prediction,
            self.v_min,
            self.v_max,
            legal_mask,
        )

        # SVE = Σ_a total_value(root, a) / Σ_a visit_count(root, a).
        var sum_value = Float64(0.0)
        var sum_visits = 0
        if len(self.mcts.nodes) > 0:
            var root = self.mcts.nodes[0]
            for a in range(Self.ACT):
                sum_value += root.total_value[a]
                sum_visits += root.visit_count[a]
        var root_value = compute_sve(sum_value, sum_visits)

        # Action sampling.
        var action: Int
        if not training or self.temperature < 0.01:
            action = 0
            var best = policy[0]
            for a in range(1, Self.ACT):
                if policy[a] > best:
                    best = policy[a]
                    action = a
        else:
            var temp_policy = InlineArray[Float64, Self.ACT](
                uninitialized=True
            )
            var inv_t = 1.0 / self.temperature
            var sum_p = Float64(0.0)
            for a in range(Self.ACT):
                if policy[a] > 0.0:
                    # exp((1/T) · ln(p)) for numerical stability over T<<1.
                    temp_policy[a] = exp(inv_t * log(policy[a]))
                else:
                    temp_policy[a] = Float64(0.0)
                sum_p += temp_policy[a]
            if sum_p > 0.0:
                for a in range(Self.ACT):
                    temp_policy[a] /= sum_p
            else:
                # Pathological — fall back to uniform over all actions.
                for a in range(Self.ACT):
                    temp_policy[a] = 1.0 / Float64(Self.ACT)

            var u = random_float64(0.0, 1.0)
            var cumsum = Float64(0.0)
            action = Self.ACT - 1
            for a in range(Self.ACT):
                cumsum += temp_policy[a]
                if u <= cumsum:
                    action = a
                    break

        return (action, policy, root_value)

    def decay_temperature(mut self):
        """Linearly decay the action-sampling temperature toward 0 over
        `temperature_decay_steps`."""
        if self.temperature_decay_steps <= 0:
            return
        var frac = Float64(self.total_steps) / Float64(
            self.temperature_decay_steps
        )
        if frac > 1.0:
            frac = 1.0
        # Linear decay from initial T to 0; caller can re-init temperature
        # on each call to override.
        var new_t = (1.0 - frac) * 1.0
        if new_t < 0.0:
            new_t = 0.0
        self.temperature = new_t

    # ══════════════════════════════════════════════════════════════════════
    # Episode management
    # ══════════════════════════════════════════════════════════════════════

    def reset_episode(mut self, env_id: Int = 0):
        """Clear `env_id`'s episode buffer. Default env_id=0 preserves
        single-env behavior."""
        self._episode_obs[env_id].clear()
        self._episode_actions[env_id].clear()
        self._episode_rewards[env_id].clear()
        self._episode_policies[env_id].clear()
        self._episode_values[env_id].clear()

    def store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        policy: InlineArray[Float64, Self.ACT],
        value: Float64,
        done: Bool,
        env_id: Int = 0,
    ):
        """Append a transition to env_id's episode buffer. Flushes that
        env's episode to the replay buffer at `done`. Default env_id=0
        preserves single-env behavior."""
        self._episode_obs[env_id].append(obs.copy())
        self._episode_actions[env_id].append(action)
        self._episode_rewards[env_id].append(reward)
        self._episode_policies[env_id].append(policy)
        self._episode_values[env_id].append(value)
        self.total_steps += 1

        if done:
            self._flush_episode(env_id)

    def _flush_episode(mut self, env_id: Int = 0):
        """Write env_id's accumulated episode to the SequenceReplayBuffer
        plus the parallel MCTS-target arrays."""
        var ep_len = len(self._episode_obs[env_id])

        for t in range(ep_len):
            var obs_arr = InlineArray[
                Scalar[DType.float32], Self.Config.obs_dim
            ](uninitialized=True)
            for i in range(Self.Config.obs_dim):
                if i < len(self._episode_obs[env_id][t]):
                    obs_arr[i] = Scalar[DType.float32](
                        self._episode_obs[env_id][t][i]
                    )
                else:
                    obs_arr[i] = Scalar[DType.float32](0.0)

            # One-hot action for SequenceReplayBuffer's ACT-wide action slot.
            var act_arr = InlineArray[
                Scalar[DType.float32], Self.ACT
            ](uninitialized=True)
            for i in range(Self.ACT):
                act_arr[i] = Scalar[DType.float32](0.0)
            act_arr[
                self._episode_actions[env_id][t]
            ] = Scalar[DType.float32](1.0)

            var is_done = t == ep_len - 1

            self.state.buffer.add(
                obs_arr,
                act_arr,
                Scalar[DType.float32](
                    self._episode_rewards[env_id][t]
                ),
                is_done,
            )

            # Mirror the MCTS targets into the parallel storage at the
            # buffer's just-written index.
            comptime CAP = 50000
            var buf_idx = (
                self.state.buffer.ptr - 1 + CAP
            ) % CAP
            for a in range(Self.ACT):
                self.state.mcts_policies[
                    buf_idx * Self.ACT + a
                ] = Scalar[dtype](
                    self._episode_policies[env_id][t][a]
                )
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[env_id][t]
            )
            # Stamp the transition with current train-step count so the
            # mixed-value-target blend can compute per-sample data age.
            self.state.step_at_write[buf_idx] = Scalar[DType.uint32](
                self.train_step_count
            )
            # Default priority = max-seen so fresh transitions compete
            # for sampling with previously-seen high-priority windows.
            self.state.priorities[buf_idx] = Scalar[dtype](
                self.max_priority
            )

        self.reset_episode(env_id)

    # ══════════════════════════════════════════════════════════════════════
    # Reanalyze (paper App. A) + target-network sync
    # ══════════════════════════════════════════════════════════════════════

    def update_target_networks(mut self, tau: Float64 = 1.0):
        """Polyak-update target networks from online networks.
            target ← τ · online + (1 − τ) · target
        `tau=1.0` is a hard copy (used after `__init__` to sync targets
        and at coarse intervals during training); `tau ≪ 1` slowly
        tracks the online network the way a SAC target net does.

        Only rep / dyn / pred get target copies — the SimSiam projector
        and predictor are training-only and aren't used during reanalyze
        search.
        """
        for i in range(Self.Config.RepModel.PARAM_SIZE):
            var src = Float64(self.state.representation.params[i])
            var tgt = Float64(self.state.representation_target.params[i])
            self.state.representation_target.params[i] = Scalar[dtype](
                tau * src + (1.0 - tau) * tgt
            )
        for i in range(Self.Config.DynModel.PARAM_SIZE):
            var src = Float64(self.state.dynamics.params[i])
            var tgt = Float64(self.state.dynamics_target.params[i])
            self.state.dynamics_target.params[i] = Scalar[dtype](
                tau * src + (1.0 - tau) * tgt
            )
        for i in range(Self.Config.PredModel.PARAM_SIZE):
            var src = Float64(self.state.prediction.params[i])
            var tgt = Float64(self.state.prediction_target.params[i])
            self.state.prediction_target.params[i] = Scalar[dtype](
                tau * src + (1.0 - tau) * tgt
            )

    def reanalyze(mut self, num_samples: Int = 16) -> Int:
        """Re-run Gumbel search on `num_samples` random replay-buffer
        positions using the **target** networks. Overwrites the stored
        MCTS policies + root values + age stamp at those indices so
        they reflect the current target model rather than whatever the
        online model looked like at collection time.

        Skips work if the buffer isn't ready (returns 0). Otherwise
        returns the number of indices it actually refreshed.

        Convention: callers should run this every
        `target_network_updating_interval` train steps (paper default
        400), typically right after a target-network sync, so the
        targets reflect the latest policy.
        """
        if not self.state.is_ready():
            return 0

        comptime CAP = 50000
        var buf_size = self.state.buffer.size
        var buf_ptr = self.state.buffer.ptr
        var oldest = (buf_ptr - buf_size + CAP) % CAP
        var n_refreshed = 0

        for _ in range(num_samples):
            var rand_offset = Int(random_float64() * Float64(buf_size))
            if rand_offset >= buf_size:
                rand_offset = buf_size - 1
            if rand_offset < 0:
                rand_offset = 0
            var idx = (oldest + rand_offset) % CAP

            # Build obs from buffer at this index.
            var obs = List[Scalar[dtype]](capacity=Self.Config.obs_dim)
            for d in range(Self.Config.obs_dim):
                obs.append(
                    self.state.buffer.obs[
                        idx * Self.Config.obs_dim + d
                    ]
                )

            # Run Gumbel search with target networks.
            var policy = self.mcts.search(
                obs,
                self.state.representation_target,
                self.state.dynamics_target,
                self.state.prediction_target,
                self.v_min,
                self.v_max,
                List[Bool](),
            )

            # Fresh SVE root value from the search.
            var sum_value = Float64(0.0)
            var sum_visits = 0
            if len(self.mcts.nodes) > 0:
                var root = self.mcts.nodes[0]
                for a in range(Self.ACT):
                    sum_value += root.total_value[a]
                    sum_visits += root.visit_count[a]
            var sve = compute_sve(sum_value, sum_visits)

            # Overwrite the stored targets at this index.
            for a in range(Self.ACT):
                self.state.mcts_policies[
                    idx * Self.ACT + a
                ] = Scalar[dtype](policy[a])
            self.state.mcts_values[idx] = Scalar[dtype](sve)
            # Treat this as fresh data — the mixed-value-target's age
            # term should now blend toward SVE since the stored value
            # was just produced by the current target net.
            self.state.step_at_write[idx] = Scalar[DType.uint32](
                self.train_step_count
            )

            n_refreshed += 1

        return n_refreshed

    # ══════════════════════════════════════════════════════════════════════
    # Training (stub — landing in step 5b)
    # ══════════════════════════════════════════════════════════════════════

    def train(mut self) -> Float64:
        """Convenience: one full train step. Returns L_total. See
        `train_step` for component-wise losses."""
        var t = self.train_step()
        return t[0]

    # ══════════════════════════════════════════════════════════════════════
    # Full training step (forward + K-step BPTT + optimizer)
    # ══════════════════════════════════════════════════════════════════════

    def train_step(
        mut self,
    ) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """K-step BPTT through all five networks + Adam optimizer step.

        Returns (L_total, L_R, L_P, L_V, L_G), where
            L_total = λ_R·L_R + λ_P·L_P + λ_V·L_V + λ_G·L_G

        Pipeline:

            1. Sample BATCH (K+1)-step windows + matching MCTS targets.
            2. Forward (with cache):
                 rep(o[0]) → z[0]
                 for k=1..K:  dyn(z[k-1] ‖ a[k-1]) → z[k] ‖ rew_logits[k-1]
                 for k=0..K:  pred(z[k]) → policy ‖ value
                 for k=1..K:  projector(z[k]) → proj_dyn[k]
                              predictor(proj_dyn[k]) → pred_dyn[k]
                              rep(o[k])             — no cache (stop-grad)
                              projector(rep_obs[k]) — no cache (stop-grad)
            3. Compute upstream gradients on every cached output using the
               composite loss weights (paper Eq. 3).
            4. Zero all five `grads` buffers, then backward in topological
               order so per-step grad_hidden accumulates from pred + projector
               + dyn-of-next-step before each dyn.backward consumes it.
            5. Adam.step on every network.

        Returns scalar means of every loss component (matches the
        forward-only `compute_loss_components` numerics for unit tests
        that compare them).
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime N_TD = Self.Config.td_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime LATENT = Self.Config.latent_dim
        comptime PROJ = Self.Config.proj_dim
        comptime BINS = Self.Config.num_bins
        comptime DYN_IN = LATENT + ACT
        comptime DYN_OUT = LATENT + BINS
        comptime PRED_OUT = ACT + BINS
        comptime CAP = 50000

        if not self.state.is_ready():
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        # ── 1. Sample (priority-weighted; matches compute_loss_components)
        var batch_obs = alloc[Scalar[dtype]](BATCH * (K + 1) * OBS)
        var batch_actions = alloc[Scalar[dtype]](BATCH * K * ACT)
        var batch_rewards = alloc[Scalar[dtype]](BATCH * K)
        var batch_mcts_pol = alloc[Scalar[dtype]](BATCH * (K + 1) * ACT)
        var batch_mcts_val = alloc[Scalar[dtype]](BATCH * (K + 1))
        var batch_age = alloc[Scalar[DType.int32]](BATCH * (K + 1))
        var batch_start_idx = alloc[Int](BATCH)
        memset(batch_obs, 0, BATCH * (K + 1) * OBS)
        memset(batch_actions, 0, BATCH * K * ACT)
        memset(batch_rewards, 0, BATCH * K)
        memset(batch_mcts_pol, 0, BATCH * (K + 1) * ACT)
        memset(batch_mcts_val, 0, BATCH * (K + 1))
        memset(batch_age, 0, BATCH * (K + 1))

        var buf_size = self.state.buffer.size
        var buf_ptr = self.state.buffer.ptr
        var current_train_step = self.train_step_count
        var oldest = (buf_ptr - buf_size + CAP) % CAP

        # Build cumulative-priority array over valid window starts. A
        # window of length K is invalid if any of its K transitions has
        # the episode-boundary `dones` flag set.
        var n_cands_alloc = buf_size - K if buf_size > K else 1
        var cum_prio = alloc[Float64](n_cands_alloc)
        var cand_starts = alloc[Int](n_cands_alloc)
        var n_valid = 0
        var total_prio = Float64(0.0)
        if buf_size > K:
            for offset in range(buf_size - K):
                var idx = (oldest + offset) % CAP
                var valid = True
                for k in range(K):
                    var iidx = (idx + k) % CAP
                    if Float64(self.state.buffer.dones[iidx]) > 0.5:
                        valid = False
                        break
                if not valid:
                    continue
                var p = Float64(self.state.priorities[idx])
                if p < 1e-8:
                    p = 1e-8
                total_prio += p
                cum_prio[n_valid] = total_prio
                cand_starts[n_valid] = idx
                n_valid += 1

        if n_valid < BATCH or total_prio < 1e-8:
            batch_obs.free()
            batch_actions.free()
            batch_rewards.free()
            batch_mcts_pol.free()
            batch_mcts_val.free()
            batch_age.free()
            batch_start_idx.free()
            cum_prio.free()
            cand_starts.free()
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        # Sample BATCH window starts proportional to priority.
        for sampled in range(BATCH):
            var u = random_float64() * total_prio
            var picked = 0
            for i in range(n_valid):
                if cum_prio[i] >= u:
                    picked = i
                    break
            var start = cand_starts[picked]
            batch_start_idx[sampled] = start

            for k in range(K + 1):
                var idx = (start + k) % CAP
                for d in range(OBS):
                    batch_obs[
                        (sampled * (K + 1) + k) * OBS + d
                    ] = self.state.buffer.obs[idx * OBS + d]
                for a in range(ACT):
                    batch_mcts_pol[
                        (sampled * (K + 1) + k) * ACT + a
                    ] = self.state.mcts_policies[idx * ACT + a]
                batch_mcts_val[
                    sampled * (K + 1) + k
                ] = self.state.mcts_values[idx]
                var age = current_train_step - Int(
                    self.state.step_at_write[idx]
                )
                if age < 0:
                    age = 0
                batch_age[sampled * (K + 1) + k] = Scalar[DType.int32](age)
            for k in range(K):
                var idx = (start + k) % CAP
                for a in range(ACT):
                    batch_actions[
                        (sampled * K + k) * ACT + a
                    ] = self.state.buffer.actions[idx * ACT + a]
                batch_rewards[
                    sampled * K + k
                ] = self.state.buffer.rewards[(start + k) % CAP]

        cum_prio.free()
        cand_starts.free()

        # ── Param/state LayoutTensor views (bypass `.params_view()`) ────
        var rep_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.representation.params)
        var rep_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.RepModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.representation.model_state)
        var rep_grads = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.representation.grads)
        var dyn_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.DynModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.params)
        var dyn_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.DynModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.model_state)
        var dyn_grads = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.DynModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.grads)
        var pred_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.params)
        var pred_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.model_state)
        var pred_grads = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.grads)
        var proj_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ProjectorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.projector.params)
        var proj_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ProjectorModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.projector.model_state)
        var proj_grads = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ProjectorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.projector.grads)
        var predr_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredictorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.params)
        var predr_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredictorModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.model_state)
        var predr_grads = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredictorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.grads)

        # Optimizer state views
        var rep_opt_state = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.Config.RepModel.PARAM_SIZE,
                Self.Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](self.state.representation.optimizer_state)
        var dyn_opt_state = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.Config.DynModel.PARAM_SIZE,
                Self.Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](self.state.dynamics.optimizer_state)
        var pred_opt_state = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.Config.PredModel.PARAM_SIZE,
                Self.Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](self.state.prediction.optimizer_state)
        var proj_opt_state = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.Config.ProjectorModel.PARAM_SIZE,
                Self.Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](self.state.projector.optimizer_state)
        var predr_opt_state = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.Config.PredictorModel.PARAM_SIZE,
                Self.Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](self.state.predictor.optimizer_state)

        # Optimizer global state (step counter etc — usually empty for
        # CPU Adam but the trait requires it).
        var rep_opt_global = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.representation.opt_global_state)
        var dyn_opt_global = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.opt_global_state)
        var pred_opt_global = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.opt_global_state)
        var proj_opt_global = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.projector.opt_global_state)
        var predr_opt_global = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.opt_global_state)

        # ── 2. Forward pass with cache. ──────────────────────────────────
        # Reuse pre-allocated state caches — they're sized exactly for
        # this purpose.
        var hidden = self.state._hidden_states  # [(K+1) * BATCH * LATENT]
        var pred_out = self.state._pred_outputs  # [(K+1) * BATCH * PRED_OUT]
        memset(hidden, 0, (K + 1) * BATCH * LATENT)
        memset(pred_out, 0, (K + 1) * BATCH * PRED_OUT)

        var dyn_out_buf = alloc[Scalar[dtype]](K * BATCH * DYN_OUT)
        memset(dyn_out_buf, 0, K * BATCH * DYN_OUT)

        var rep_input = alloc[Scalar[dtype]](BATCH * OBS)
        for b in range(BATCH):
            for d in range(OBS):
                rep_input[b * OBS + d] = batch_obs[
                    (b * (K + 1) + 0) * OBS + d
                ]

        # rep(o[0]) → hidden[0]
        var rep_input_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](rep_input)
        var hidden0_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](hidden)
        var rep_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self.state._rep_cache)
        Network[
            Self.Config.RepModel, Self.Config.OptType
        ].forward_with_cache[BATCH](
            rep_input_t,
            hidden0_t,
            rep_params,
            rep_state_buf,
            rep_cache_t,
        )

        # K dynamics steps
        var dyn_input = alloc[Scalar[dtype]](BATCH * DYN_IN)
        comptime DYN_CS = Self.Config.DynModel.CACHE_SIZE
        for k in range(K):
            for b in range(BATCH):
                for d in range(LATENT):
                    dyn_input[b * DYN_IN + d] = hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ]
                for a in range(ACT):
                    dyn_input[b * DYN_IN + LATENT + a] = batch_actions[
                        (b * K + k) * ACT + a
                    ]
            var dyn_input_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.IN_DIM),
                MutAnyOrigin,
            ](dyn_input)
            var dyn_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.OUT_DIM),
                MutAnyOrigin,
            ](dyn_out_buf + k * BATCH * DYN_OUT)
            var dyn_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.CACHE_SIZE),
                MutAnyOrigin,
            ](self.state._dyn_caches + k * BATCH * DYN_CS)
            Network[
                Self.Config.DynModel, Self.Config.OptType
            ].forward_with_cache[BATCH](
                dyn_input_t,
                dyn_out_t,
                dyn_params,
                dyn_state_buf,
                dyn_cache_t,
            )
            for b in range(BATCH):
                for d in range(LATENT):
                    hidden[
                        (k + 1) * BATCH * LATENT + b * LATENT + d
                    ] = dyn_out_buf[
                        k * BATCH * DYN_OUT + b * DYN_OUT + d
                    ]
        dyn_input.free()

        # Pred at k = 0..K
        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        for k in range(K + 1):
            var pred_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](hidden + k * BATCH * LATENT)
            var pred_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](pred_out + k * BATCH * PRED_OUT)
            var pred_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.CACHE_SIZE),
                MutAnyOrigin,
            ](self.state._pred_caches + k * BATCH * PRED_CS)
            Network[
                Self.Config.PredModel, Self.Config.OptType
            ].forward_with_cache[BATCH](
                pred_in_t,
                pred_out_t,
                pred_params,
                pred_state_buf,
                pred_cache_t,
            )

        # SimSiam forward — cached only on the dynamics branch (gradient
        # flows). Obs branch uses no-cache forward (stop-grad target).
        comptime PROJ_CS = Self.Config.ProjectorModel.CACHE_SIZE
        comptime PREDR_CS = Self.Config.PredictorModel.CACHE_SIZE
        var proj_dyn_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        var pred_dyn_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        var proj_obs_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        memset(proj_dyn_buf, 0, K * BATCH * PROJ)
        memset(pred_dyn_buf, 0, K * BATCH * PROJ)
        memset(proj_obs_buf, 0, K * BATCH * PROJ)

        var rep_obs_step = alloc[Scalar[dtype]](BATCH * LATENT)
        var obs_input_step = alloc[Scalar[dtype]](BATCH * OBS)

        for k_offset in range(K):
            var k = k_offset + 1

            var proj_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](hidden + k * BATCH * LATENT)
            var proj_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](proj_dyn_buf + k_offset * BATCH * PROJ)
            var proj_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                self.state._proj_dyn_caches
                + k_offset * BATCH * PROJ_CS
            )
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward_with_cache[BATCH](
                proj_in_t,
                proj_out_t,
                proj_params,
                proj_state_buf,
                proj_cache_t,
            )

            var pred_in2_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.IN_DIM),
                MutAnyOrigin,
            ](proj_dyn_buf + k_offset * BATCH * PROJ)
            var pred_out2_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.Config.PredictorModel.OUT_DIM
                ),
                MutAnyOrigin,
            ](pred_dyn_buf + k_offset * BATCH * PROJ)
            var predr_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                self.state._pred_dyn_caches
                + k_offset * BATCH * PREDR_CS
            )
            Network[
                Self.Config.PredictorModel, Self.Config.OptType
            ].forward_with_cache[BATCH](
                pred_in2_t,
                pred_out2_t,
                predr_params,
                predr_state_buf,
                predr_cache_t,
            )

            # Target branch (no cache, no gradient).
            for b in range(BATCH):
                for d in range(OBS):
                    obs_input_step[b * OBS + d] = batch_obs[
                        (b * (K + 1) + k) * OBS + d
                    ]
            var obs_step_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                MutAnyOrigin,
            ](obs_input_step)
            var rep_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                MutAnyOrigin,
            ](rep_obs_step)
            Network[
                Self.Config.RepModel, Self.Config.OptType
            ].forward[BATCH](
                obs_step_t, rep_obs_t, rep_params, rep_state_buf
            )

            var rep_obs_for_proj_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](rep_obs_step)
            var proj_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.Config.ProjectorModel.OUT_DIM
                ),
                MutAnyOrigin,
            ](proj_obs_buf + k_offset * BATCH * PROJ)
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward[BATCH](
                rep_obs_for_proj_t,
                proj_obs_t,
                proj_params,
                proj_state_buf,
            )

        rep_obs_step.free()
        obs_input_step.free()

        # ── 2.5 Reward-prefix LSTM forward (paper App. G) ────────────────
        # When `use_reward_prefix=True`, we replace the per-step reward
        # CE through the dyn-network's reward output with a CE on
        #     reward_prefix_logits[k] = MLP_head(LSTM(hidden[k+1]))
        # against `two_hot(scalar_transform(Σ_{j≤k} reward[j]))`. The LSTM
        # state resets to zero every `lstm_horizon_len` unroll steps to
        # cap BPTT depth.
        comptime LSTM_HIDDEN = Self.Config.lstm_hidden
        comptime LSTM_HORIZON = Self.Config.lstm_horizon_len
        comptime _LSTMHead = LSTMCell[LATENT, LSTM_HIDDEN]
        comptime _RewardPrefixMLP = RewardPrefixHeadMLP[
            LSTM_HIDDEN,
            Self.Config.lstm_mlp_hidden,
            BINS,
        ]

        var rew_pref_logits = alloc[Scalar[dtype]](K * BATCH * BINS)
        var grad_rew_pref_logits = alloc[Scalar[dtype]](K * BATCH * BINS)
        memset(rew_pref_logits, 0, K * BATCH * BINS)
        memset(grad_rew_pref_logits, 0, K * BATCH * BINS)

        # Per-step zero scratch + mutable input slot for reset boundaries.
        # We can't memset h_lstm[k] / c_lstm[k] directly (we still need
        # those values for backward), so we use this tiny slot to feed
        # the LSTM a zeroed h_prev / c_prev at horizon boundaries while
        # leaving the time-major arrays intact.
        var lstm_h_input = alloc[Scalar[dtype]](BATCH * LSTM_HIDDEN)
        var lstm_c_input = alloc[Scalar[dtype]](BATCH * LSTM_HIDDEN)

        comptime if Self.Config.use_reward_prefix:
            # Reset h_lstm[0], c_lstm[0] = 0 at the start of every batch
            # — h_lstm/c_lstm storage is allocated zeroed but train_step
            # is called many times so we must clear at every entry.
            memset(
                self.state._lstm_h_states,
                0,
                (K + 1) * BATCH * LSTM_HIDDEN,
            )
            memset(
                self.state._lstm_c_states,
                0,
                (K + 1) * BATCH * LSTM_HIDDEN,
            )

            var lstm_params_v = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.lstm_params)

            var mlp_head_params = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.params)
            var mlp_head_state = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.model_state)

            for k in range(K):
                # Decide LSTM input (h_prev, c_prev): either previous
                # output or zeros at horizon boundary (where the chain
                # is reset BEFORE step k).
                var reset_now = (k > 0) and (k % LSTM_HORIZON == 0)
                if reset_now:
                    memset(lstm_h_input, 0, BATCH * LSTM_HIDDEN)
                    memset(lstm_c_input, 0, BATCH * LSTM_HIDDEN)
                else:
                    # Copy h_lstm[k], c_lstm[k] into the input slots.
                    for i in range(BATCH * LSTM_HIDDEN):
                        lstm_h_input[i] = self.state._lstm_h_states[
                            k * BATCH * LSTM_HIDDEN + i
                        ]
                        lstm_c_input[i] = self.state._lstm_c_states[
                            k * BATCH * LSTM_HIDDEN + i
                        ]

                # LSTM step k: input is hidden[k+1] (the post-dyn-step-k
                # latent — same alignment as the dyn-network's reward
                # logits, which predicted reward[k]).
                var z_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](hidden + (k + 1) * BATCH * LATENT)
                var h_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](lstm_h_input)
                var c_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](lstm_c_input)
                var h_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    self.state._lstm_h_states
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var c_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    self.state._lstm_c_states
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var lstm_cache_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _LSTMHead.CACHE_SIZE),
                    MutAnyOrigin,
                ](
                    self.state._lstm_caches
                    + k * BATCH * _LSTMHead.CACHE_SIZE
                )
                _LSTMHead.step_forward[BATCH](
                    z_t,
                    h_prev_t,
                    c_prev_t,
                    lstm_params_v,
                    h_t,
                    c_t,
                    lstm_cache_t,
                )

                # MLP head forward on h_lstm[k+1] → reward_prefix_logits[k]
                var mlp_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                    MutAnyOrigin,
                ](
                    self.state._lstm_h_states
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var mlp_out_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                    MutAnyOrigin,
                ](rew_pref_logits + k * BATCH * BINS)
                var mlp_cache_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.CACHE_SIZE),
                    MutAnyOrigin,
                ](
                    self.state._mlp_head_caches
                    + k * BATCH * _RewardPrefixMLP.CACHE_SIZE
                )
                Network[
                    _RewardPrefixMLP, Self.Config.OptType
                ].forward_with_cache[BATCH](
                    mlp_in_t,
                    mlp_out_t,
                    mlp_head_params,
                    mlp_head_state,
                    mlp_cache_t,
                )

            # Build cumulative-reward target on host: cum_rew[b, k] =
            # Σ_{j=0..k} batch_rewards[b, j].
            for b in range(BATCH):
                var cum = Float64(0.0)
                for k in range(K):
                    cum += Float64(batch_rewards[b * K + k])
                    self.state._cum_rewards[b * K + k] = Scalar[dtype](cum)

        # ── 3. Compute scalar losses for return + per-output upstream
        #       gradients for backward. We compute loss values alongside
        #       the gradient for L_R, L_P, L_V; for L_G we use the
        #       cosine-loss formula (forward + backward in one pass). ──
        var L_R = Float64(0.0)
        var L_P = Float64(0.0)
        var L_V = Float64(0.0)
        var L_G = Float64(0.0)

        # Allocate per-output gradient buffers. These are dense, time-major.
        var grad_pred_out = alloc[Scalar[dtype]]((K + 1) * BATCH * PRED_OUT)
        var grad_dyn_out = alloc[Scalar[dtype]](K * BATCH * DYN_OUT)
        var grad_pred_dyn = alloc[Scalar[dtype]](K * BATCH * PROJ)
        memset(grad_pred_out, 0, (K + 1) * BATCH * PRED_OUT)
        memset(grad_dyn_out, 0, K * BATCH * DYN_OUT)
        memset(grad_pred_dyn, 0, K * BATCH * PROJ)

        var two_hot_target = alloc[Float64](BINS)
        var logits_dbl = alloc[Float64](BINS)
        var pol_logits_dbl = alloc[Float64](ACT)
        var pol_target_dbl = alloc[Float64](ACT)

        # ── L_P + grad: policy CE at each k=0..K ────────────────────────
        # CE = -Σ p_t log softmax(logits)_i;  d/d logits = (softmax - p_t).
        var n_P = Float64(BATCH * (K + 1))
        var lp_scale = self.Config.lambda_policy / n_P
        for k in range(K + 1):
            for b in range(BATCH):
                var off = k * BATCH * PRED_OUT + b * PRED_OUT
                # Stable log-softmax over ACT
                for i in range(ACT):
                    pol_logits_dbl[i] = Float64(pred_out[off + i])
                    pol_target_dbl[i] = Float64(
                        batch_mcts_pol[(b * (K + 1) + k) * ACT + i]
                    )
                L_P += cross_entropy_with_softmax[ACT](
                    pol_logits_dbl, pol_target_dbl
                )
                # Compute softmax probs
                var max_l = pol_logits_dbl[0]
                for i in range(1, ACT):
                    if pol_logits_dbl[i] > max_l:
                        max_l = pol_logits_dbl[i]
                var sum_e = Float64(0.0)
                var probs = InlineArray[Float64, ACT](uninitialized=True)
                for i in range(ACT):
                    probs[i] = exp(pol_logits_dbl[i] - max_l)
                    sum_e += probs[i]
                if sum_e <= 0.0:
                    sum_e = 1.0
                for i in range(ACT):
                    probs[i] /= sum_e
                for i in range(ACT):
                    grad_pred_out[off + i] = Scalar[dtype](
                        lp_scale * (probs[i] - pol_target_dbl[i])
                    )

        # ── Fresh bootstrap values from target nets (Lever 1). ───────────
        # Only computed when the value-target mode actually consumes
        # `boot_v` — SARSA always does, MIXED can (when age > t_fresh).
        # SEARCH ignores it entirely so we skip the (K+1) target-net
        # forwards, saving ~K+1 small CPU rep+pred calls per train_step.
        var boot_v = alloc[Scalar[dtype]](BATCH * (K + 1))
        memset(boot_v, 0, BATCH * (K + 1))
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
            var tgt_rep_input = alloc[Scalar[dtype]](BATCH * OBS)
            var tgt_z = alloc[Scalar[dtype]](BATCH * LATENT)
            var tgt_pred_out = alloc[Scalar[dtype]](BATCH * PRED_OUT)
            var tgt_logits_dbl = alloc[Float64](BINS)

            var tgt_rep_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.params)
            var tgt_rep_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.model_state)
            var tgt_pred_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.params)
            var tgt_pred_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.model_state)

            for k in range(K + 1):
                for b in range(BATCH):
                    for d in range(OBS):
                        tgt_rep_input[b * OBS + d] = batch_obs[
                            (b * (K + 1) + k) * OBS + d
                        ]
                var tgt_rep_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_rep_input)
                var tgt_z_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                Network[
                    Self.Config.RepModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_rep_in_t,
                    tgt_z_t,
                    tgt_rep_params,
                    tgt_rep_state,
                )
                var tgt_pred_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                var tgt_pred_out_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_pred_out)
                Network[
                    Self.Config.PredModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_pred_in_t,
                    tgt_pred_out_t,
                    tgt_pred_params,
                    tgt_pred_state,
                )
                for b in range(BATCH):
                    var off = b * PRED_OUT + ACT
                    for i in range(BINS):
                        tgt_logits_dbl[i] = Float64(tgt_pred_out[off + i])
                    var v_raw = decode_categorical[BINS](
                        tgt_logits_dbl, self.v_min, self.v_max
                    )
                    boot_v[b * (K + 1) + k] = Scalar[dtype](
                        inverse_scalar_transform(v_raw)
                    )

            tgt_rep_input.free()
            tgt_z.free()
            tgt_pred_out.free()
            tgt_logits_dbl.free()

        # ── L_V + grad: value CE under one of three modes ────────────────
        #   SEARCH: target = stored MCTS root value `sve`.
        #   SARSA:  target = n-step TD with fresh target-net bootstrap.
        #   MIXED:  target = MixedValueTarget(sve, td, age) blend.
        var n_V = Float64(BATCH * (K + 1))
        var lv_scale = self.Config.lambda_value / n_V
        for k in range(K + 1):
            var n_eff = N_TD if N_TD < K - k else K - k
            for b in range(BATCH):
                var sve = Float64(batch_mcts_val[b * (K + 1) + k])
                var td = Float64(0.0)
                comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
                    var disc = Float64(1.0)
                    for j in range(n_eff):
                        td += disc * Float64(
                            batch_rewards[b * K + k + j]
                        )
                        disc *= self.gamma
                    td += disc * Float64(
                        boot_v[b * (K + 1) + k + n_eff]
                    )
                var age = Int(batch_age[b * (K + 1) + k])
                var v_target = Float64(0.0)
                comptime if Self.Config.value_target_mode == VALUE_TARGET_SEARCH:
                    v_target = sve
                elif Self.Config.value_target_mode == VALUE_TARGET_SARSA:
                    v_target = td
                else:  # VALUE_TARGET_MIXED
                    v_target = MixedValueTarget[
                        Self.Config.t_fresh, Self.Config.t_stale
                    ].compute(sve, td, age)
                encode_categorical[BINS](
                    scalar_transform(v_target),
                    self.v_min,
                    self.v_max,
                    two_hot_target,
                )
                var off = k * BATCH * PRED_OUT + b * PRED_OUT + ACT
                for i in range(BINS):
                    logits_dbl[i] = Float64(pred_out[off + i])
                var per_sample_v_loss = cross_entropy_with_softmax[BINS](
                    logits_dbl, two_hot_target
                )
                L_V += per_sample_v_loss
                # Update per-transition priority at unroll position k=0
                # (the window's root). |TD error| is approximated by the
                # cross-entropy loss in scalar-transformed log-bin space —
                # it dominates the absolute error and avoids re-decoding
                # the value here.
                if k == 0:
                    var new_p = per_sample_v_loss + 1e-3
                    self.state.priorities[
                        batch_start_idx[b]
                    ] = Scalar[dtype](new_p)
                    if new_p > self.max_priority:
                        self.max_priority = new_p
                # softmax + grad
                var max_l = logits_dbl[0]
                for i in range(1, BINS):
                    if logits_dbl[i] > max_l:
                        max_l = logits_dbl[i]
                var sum_e = Float64(0.0)
                var probs_v = InlineArray[Float64, BINS](uninitialized=True)
                for i in range(BINS):
                    probs_v[i] = exp(logits_dbl[i] - max_l)
                    sum_e += probs_v[i]
                if sum_e <= 0.0:
                    sum_e = 1.0
                for i in range(BINS):
                    probs_v[i] /= sum_e
                for i in range(BINS):
                    grad_pred_out[off + i] = Scalar[dtype](
                        lv_scale * (probs_v[i] - two_hot_target[i])
                    )

        # ── L_R + grad: reward CE at each k=0..K-1 ───────────────────────
        # Two paths:
        #   • `use_reward_prefix=False`: classic per-step reward CE through
        #     the dyn-network's reward output slice. Default.
        #   • `use_reward_prefix=True`: cumulative-reward CE through the
        #     reward-prefix LSTM head; dyn-output's reward grad stays zero
        #     so no gradient flows through that branch.
        var n_R = Float64(BATCH * K)
        var lr_scale = self.Config.lambda_reward / n_R

        comptime if not Self.Config.use_reward_prefix:
            for k in range(K):
                for b in range(BATCH):
                    var rew = Float64(batch_rewards[b * K + k])
                    encode_categorical[BINS](
                        scalar_transform(rew),
                        self.v_min,
                        self.v_max,
                        two_hot_target,
                    )
                    var off = k * BATCH * DYN_OUT + b * DYN_OUT + LATENT
                    for i in range(BINS):
                        logits_dbl[i] = Float64(dyn_out_buf[off + i])
                    L_R += cross_entropy_with_softmax[BINS](
                        logits_dbl, two_hot_target
                    )
                    var max_l = logits_dbl[0]
                    for i in range(1, BINS):
                        if logits_dbl[i] > max_l:
                            max_l = logits_dbl[i]
                    var sum_e = Float64(0.0)
                    var probs_r = InlineArray[Float64, BINS](
                        uninitialized=True
                    )
                    for i in range(BINS):
                        probs_r[i] = exp(logits_dbl[i] - max_l)
                        sum_e += probs_r[i]
                    if sum_e <= 0.0:
                        sum_e = 1.0
                    for i in range(BINS):
                        probs_r[i] /= sum_e
                    for i in range(BINS):
                        grad_dyn_out[off + i] = Scalar[dtype](
                            lr_scale * (probs_r[i] - two_hot_target[i])
                        )
        else:
            # Reward-prefix CE on `rew_pref_logits[k][b]` against
            # two_hot(scalar_transform(cum_rewards[b, k])).
            for k in range(K):
                for b in range(BATCH):
                    var cum = Float64(self.state._cum_rewards[b * K + k])
                    encode_categorical[BINS](
                        scalar_transform(cum),
                        self.v_min,
                        self.v_max,
                        two_hot_target,
                    )
                    var off = (k * BATCH + b) * BINS
                    for i in range(BINS):
                        logits_dbl[i] = Float64(rew_pref_logits[off + i])
                    L_R += cross_entropy_with_softmax[BINS](
                        logits_dbl, two_hot_target
                    )
                    var max_l = logits_dbl[0]
                    for i in range(1, BINS):
                        if logits_dbl[i] > max_l:
                            max_l = logits_dbl[i]
                    var sum_e = Float64(0.0)
                    var probs_r = InlineArray[Float64, BINS](
                        uninitialized=True
                    )
                    for i in range(BINS):
                        probs_r[i] = exp(logits_dbl[i] - max_l)
                        sum_e += probs_r[i]
                    if sum_e <= 0.0:
                        sum_e = 1.0
                    for i in range(BINS):
                        probs_r[i] /= sum_e
                    for i in range(BINS):
                        grad_rew_pref_logits[off + i] = Scalar[dtype](
                            lr_scale * (probs_r[i] - two_hot_target[i])
                        )

        # ── L_G + grad: cosine consistency at each k=1..K ────────────────
        var n_G = Float64(BATCH * K)
        var lg_scale = self.Config.lambda_consistency / n_G
        for k_offset in range(K):
            for b in range(BATCH):
                var p_off = (k_offset * BATCH + b) * PROJ
                var t_off = (k_offset * BATCH + b) * PROJ
                var dot = Float64(0.0)
                var na2 = Float64(0.0)
                var nb2 = Float64(0.0)
                for i in range(PROJ):
                    var pv = Float64(pred_dyn_buf[p_off + i])
                    var tv = Float64(proj_obs_buf[t_off + i])
                    dot += pv * tv
                    na2 += pv * pv
                    nb2 += tv * tv
                var na = sqrt(na2 + 1e-12)
                var nb = sqrt(nb2 + 1e-12)
                var c = dot / (na * nb)
                L_G += -c
                var inv_na2 = 1.0 / (na * na)
                var inv_na_nb = 1.0 / (na * nb)
                for i in range(PROJ):
                    var pv = Float64(pred_dyn_buf[p_off + i])
                    var tv = Float64(proj_obs_buf[t_off + i])
                    grad_pred_dyn[p_off + i] = Scalar[dtype](
                        lg_scale
                        * (c * pv * inv_na2 - tv * inv_na_nb)
                    )

        L_R = L_R / n_R if n_R > 0.0 else 0.0
        L_P = L_P / n_P if n_P > 0.0 else 0.0
        L_V = L_V / n_V if n_V > 0.0 else 0.0
        L_G = L_G / n_G if n_G > 0.0 else 0.0

        two_hot_target.free()
        logits_dbl.free()
        pol_logits_dbl.free()
        pol_target_dbl.free()
        boot_v.free()

        # ── 4. Backward ──────────────────────────────────────────────────
        # Zero all 5 networks' grad buffers.
        memset(
            self.state.representation.grads,
            0,
            Self.Config.RepModel.PARAM_SIZE,
        )
        memset(
            self.state.dynamics.grads, 0, Self.Config.DynModel.PARAM_SIZE
        )
        memset(
            self.state.prediction.grads, 0, Self.Config.PredModel.PARAM_SIZE
        )
        memset(
            self.state.projector.grads,
            0,
            Self.Config.ProjectorModel.PARAM_SIZE,
        )
        memset(
            self.state.predictor.grads,
            0,
            Self.Config.PredictorModel.PARAM_SIZE,
        )

        # grad_hidden[(K+1) * BATCH * LATENT] — accumulator from pred + projector + dyn-of-next.
        var grad_hidden = alloc[Scalar[dtype]]((K + 1) * BATCH * LATENT)
        memset(grad_hidden, 0, (K + 1) * BATCH * LATENT)

        # 4a. pred backward at k=0..K → adds into grad_hidden[k]
        var grad_pred_in = alloc[Scalar[dtype]](BATCH * LATENT)
        for k in range(K + 1):
            var grad_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](grad_pred_out + k * BATCH * PRED_OUT)
            var grad_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](grad_pred_in)
            var pred_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.CACHE_SIZE),
                MutAnyOrigin,
            ](self.state._pred_caches + k * BATCH * PRED_CS)
            Network[
                Self.Config.PredModel, Self.Config.OptType
            ].backward[BATCH](
                grad_out_t,
                grad_in_t,
                pred_params,
                pred_state_buf,
                pred_cache_t,
                pred_grads,
            )
            for b in range(BATCH):
                for d in range(LATENT):
                    grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] = grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] + grad_pred_in[b * LATENT + d]
        grad_pred_in.free()

        # 4b. SimSiam backward — predictor + projector (online branch) at k=1..K.
        var grad_proj_dyn = alloc[Scalar[dtype]](BATCH * PROJ)
        var grad_proj_in = alloc[Scalar[dtype]](BATCH * LATENT)
        for k_offset in range(K):
            var k = k_offset + 1
            # predictor.backward(grad_pred_dyn[k_offset], → grad_proj_dyn)
            var grad_predr_out_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.Config.PredictorModel.OUT_DIM
                ),
                MutAnyOrigin,
            ](grad_pred_dyn + k_offset * BATCH * PROJ)
            var grad_predr_in_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.Config.PredictorModel.IN_DIM
                ),
                MutAnyOrigin,
            ](grad_proj_dyn)
            var predr_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                self.state._pred_dyn_caches
                + k_offset * BATCH * PREDR_CS
            )
            Network[
                Self.Config.PredictorModel, Self.Config.OptType
            ].backward[BATCH](
                grad_predr_out_t,
                grad_predr_in_t,
                predr_params,
                predr_state_buf,
                predr_cache_t,
                predr_grads,
            )

            # projector.backward(grad_proj_dyn → grad on hidden[k])
            var grad_proj_out_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.Config.ProjectorModel.OUT_DIM
                ),
                MutAnyOrigin,
            ](grad_proj_dyn)
            var grad_proj_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](grad_proj_in)
            var proj_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                self.state._proj_dyn_caches
                + k_offset * BATCH * PROJ_CS
            )
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].backward[BATCH](
                grad_proj_out_t,
                grad_proj_in_t,
                proj_params,
                proj_state_buf,
                proj_cache_t,
                proj_grads,
            )
            for b in range(BATCH):
                for d in range(LATENT):
                    grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] = grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] + grad_proj_in[b * LATENT + d]
        grad_proj_dyn.free()
        grad_proj_in.free()

        # 4b'. Reward-prefix LSTM head backward (when use_reward_prefix=True)
        # — adds into grad_hidden[k+1] for k=0..K-1.
        #
        # Order: MLP-head backward at every k first → fills grad_h_lstm[k+1]
        # initially. Then LSTM backward in REVERSE time order (k = K-1..0):
        # at step k we read grad_h_lstm[k+1] / grad_c_lstm[k+1] (already
        # holding MLP contribution + any later step's dh_prev/dc_prev),
        # the cell's `step_backward` outputs grad_x = ∂L/∂hidden[k+1]
        # (which we add to `grad_hidden[k+1]`) plus grad_h_prev /
        # grad_c_prev contributions to step k's input. Reset boundaries
        # break the chain: dh_prev/dc_prev at a reset step are discarded
        # (the LSTM saw zero input there).
        comptime if Self.Config.use_reward_prefix:
            memset(self.state.lstm_grads, 0, _LSTMHead.PARAM_SIZE)
            memset(
                self.state.reward_prefix_mlp.grads,
                0,
                _RewardPrefixMLP.PARAM_SIZE,
            )

            var grad_h_lstm = alloc[Scalar[dtype]](
                (K + 1) * BATCH * LSTM_HIDDEN
            )
            var grad_c_lstm = alloc[Scalar[dtype]](
                (K + 1) * BATCH * LSTM_HIDDEN
            )
            memset(grad_h_lstm, 0, (K + 1) * BATCH * LSTM_HIDDEN)
            memset(grad_c_lstm, 0, (K + 1) * BATCH * LSTM_HIDDEN)

            var lstm_params_v_b = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.lstm_params)
            var lstm_grads_v_b = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.lstm_grads)
            var mlp_head_params_b = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.params)
            var mlp_head_state_b = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.model_state)
            var mlp_head_grads_b = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.grads)

            # Pass 1 — MLP-head backward at k = 0..K-1 → adds into grad_h_lstm[k+1]
            var grad_mlp_in_step = alloc[Scalar[dtype]](
                BATCH * LSTM_HIDDEN
            )
            for k in range(K):
                var grad_logits_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                    MutAnyOrigin,
                ](grad_rew_pref_logits + k * BATCH * BINS)
                var grad_mlp_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                    MutAnyOrigin,
                ](grad_mlp_in_step)
                var mlp_cache_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.CACHE_SIZE),
                    MutAnyOrigin,
                ](
                    self.state._mlp_head_caches
                    + k * BATCH * _RewardPrefixMLP.CACHE_SIZE
                )
                Network[
                    _RewardPrefixMLP, Self.Config.OptType
                ].backward[BATCH](
                    grad_logits_t,
                    grad_mlp_in_t,
                    mlp_head_params_b,
                    mlp_head_state_b,
                    mlp_cache_t_b,
                    mlp_head_grads_b,
                )
                # Accumulate grad_mlp_in_step → grad_h_lstm[k+1]
                for b in range(BATCH):
                    for d in range(LSTM_HIDDEN):
                        grad_h_lstm[
                            (k + 1) * BATCH * LSTM_HIDDEN
                            + b * LSTM_HIDDEN
                            + d
                        ] = grad_h_lstm[
                            (k + 1) * BATCH * LSTM_HIDDEN
                            + b * LSTM_HIDDEN
                            + d
                        ] + grad_mlp_in_step[b * LSTM_HIDDEN + d]
            grad_mlp_in_step.free()

            # Pass 2 — LSTM backward in REVERSE time. At step k we need
            # the original h_prev / c_prev forward inputs (zeroed at reset
            # boundaries, otherwise = h_lstm[k] / c_lstm[k]) — re-build
            # in `lstm_h_input` / `lstm_c_input` scratch.
            var grad_x_lstm = alloc[Scalar[dtype]](BATCH * LATENT)
            var grad_h_prev_lstm = alloc[Scalar[dtype]](
                BATCH * LSTM_HIDDEN
            )
            var grad_c_prev_lstm = alloc[Scalar[dtype]](
                BATCH * LSTM_HIDDEN
            )
            for kk in range(K):
                var k = K - 1 - kk
                var reset_now = (k > 0) and (k % LSTM_HORIZON == 0)
                if reset_now:
                    memset(lstm_h_input, 0, BATCH * LSTM_HIDDEN)
                    memset(lstm_c_input, 0, BATCH * LSTM_HIDDEN)
                else:
                    for i in range(BATCH * LSTM_HIDDEN):
                        lstm_h_input[i] = self.state._lstm_h_states[
                            k * BATCH * LSTM_HIDDEN + i
                        ]
                        lstm_c_input[i] = self.state._lstm_c_states[
                            k * BATCH * LSTM_HIDDEN + i
                        ]

                var dh_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    grad_h_lstm + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var dc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    grad_c_lstm + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var z_t_b = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](hidden + (k + 1) * BATCH * LATENT)
                var h_prev_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](lstm_h_input)
                var c_prev_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](lstm_c_input)
                var lstm_cache_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _LSTMHead.CACHE_SIZE),
                    MutAnyOrigin,
                ](
                    self.state._lstm_caches
                    + k * BATCH * _LSTMHead.CACHE_SIZE
                )
                var grad_x_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](grad_x_lstm)
                var grad_h_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](grad_h_prev_lstm)
                var grad_c_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](grad_c_prev_lstm)
                _LSTMHead.step_backward[BATCH](
                    dh_t,
                    dc_t,
                    z_t_b,
                    h_prev_t_b,
                    c_prev_t_b,
                    lstm_params_v_b,
                    lstm_cache_t_b,
                    grad_x_t,
                    grad_h_prev_t,
                    grad_c_prev_t,
                    lstm_grads_v_b,
                )

                # Accumulate grad_x = ∂L/∂hidden[k+1] into grad_hidden[k+1].
                for b in range(BATCH):
                    for d in range(LATENT):
                        grad_hidden[
                            (k + 1) * BATCH * LATENT + b * LATENT + d
                        ] = grad_hidden[
                            (k + 1) * BATCH * LATENT + b * LATENT + d
                        ] + grad_x_lstm[b * LATENT + d]

                # Thread dh_prev / dc_prev back to step k UNLESS this
                # step was a reset boundary (input was zeros, so the
                # gradient w.r.t. those zeros is meaningless and the
                # chain breaks here).
                if not reset_now:
                    for i in range(BATCH * LSTM_HIDDEN):
                        grad_h_lstm[
                            k * BATCH * LSTM_HIDDEN + i
                        ] = grad_h_lstm[
                            k * BATCH * LSTM_HIDDEN + i
                        ] + grad_h_prev_lstm[i]
                        grad_c_lstm[
                            k * BATCH * LSTM_HIDDEN + i
                        ] = grad_c_lstm[
                            k * BATCH * LSTM_HIDDEN + i
                        ] + grad_c_prev_lstm[i]

            grad_x_lstm.free()
            grad_h_prev_lstm.free()
            grad_c_prev_lstm.free()
            grad_h_lstm.free()
            grad_c_lstm.free()

        # 4c. dyn backward k=K-1..0. Walks BACKWARD in time.
        var grad_dyn_in_t_buf = alloc[Scalar[dtype]](BATCH * DYN_IN)
        for kk in range(K):
            var k = K - 1 - kk
            # Build grad_dyn_out = [grad_hidden[k+1] || grad_reward_logits[k]]
            var grad_dyn_out_step = alloc[Scalar[dtype]](BATCH * DYN_OUT)
            for b in range(BATCH):
                for d in range(LATENT):
                    grad_dyn_out_step[
                        b * DYN_OUT + d
                    ] = grad_hidden[
                        (k + 1) * BATCH * LATENT + b * LATENT + d
                    ]
                for i in range(BINS):
                    grad_dyn_out_step[
                        b * DYN_OUT + LATENT + i
                    ] = grad_dyn_out[
                        k * BATCH * DYN_OUT + b * DYN_OUT + LATENT + i
                    ]
            var grad_dyn_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.OUT_DIM),
                MutAnyOrigin,
            ](grad_dyn_out_step)
            var grad_dyn_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.IN_DIM),
                MutAnyOrigin,
            ](grad_dyn_in_t_buf)
            var dyn_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.CACHE_SIZE),
                MutAnyOrigin,
            ](self.state._dyn_caches + k * BATCH * DYN_CS)
            Network[
                Self.Config.DynModel, Self.Config.OptType
            ].backward[BATCH](
                grad_dyn_out_t,
                grad_dyn_in_t,
                dyn_params,
                dyn_state_buf,
                dyn_cache_t,
                dyn_grads,
            )
            for b in range(BATCH):
                for d in range(LATENT):
                    grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] = grad_hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ] + grad_dyn_in_t_buf[b * DYN_IN + d]
            grad_dyn_out_step.free()
        grad_dyn_in_t_buf.free()

        # 4d. rep backward at k=0 — grad input is observation, discarded.
        var grad_rep_in = alloc[Scalar[dtype]](BATCH * OBS)
        var grad_rep_out_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](grad_hidden)  # grad_hidden[0]
        var grad_rep_in_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](grad_rep_in)
        Network[
            Self.Config.RepModel, Self.Config.OptType
        ].backward[BATCH](
            grad_rep_out_t,
            grad_rep_in_t,
            rep_params,
            rep_state_buf,
            rep_cache_t,
            rep_grads,
        )
        grad_rep_in.free()

        # ── 5. Per-network gradient clipping + optimizer step. ──────────
        # Paper applies a single global clip-norm 5.0 across all params;
        # we approximate with per-network clips at the same threshold,
        # which gives a similar magnitude bound and matches the pattern
        # used in `offpolicy_agent`. Skipped when `max_grad_norm <= 0`.
        if self.max_grad_norm > 0.0:
            _clip_grads_inplace(
                self.state.representation.grads,
                Self.Config.RepModel.PARAM_SIZE,
                self.max_grad_norm,
            )
            _clip_grads_inplace(
                self.state.dynamics.grads,
                Self.Config.DynModel.PARAM_SIZE,
                self.max_grad_norm,
            )
            _clip_grads_inplace(
                self.state.prediction.grads,
                Self.Config.PredModel.PARAM_SIZE,
                self.max_grad_norm,
            )
            _clip_grads_inplace(
                self.state.projector.grads,
                Self.Config.ProjectorModel.PARAM_SIZE,
                self.max_grad_norm,
            )
            _clip_grads_inplace(
                self.state.predictor.grads,
                Self.Config.PredictorModel.PARAM_SIZE,
                self.max_grad_norm,
            )

        self.train_step_count += 1
        var step_num = self.train_step_count

        Self.Config.OptType.step[Self.Config.RepModel.PARAM_SIZE](
            rep_params,
            rep_grads,
            rep_opt_state,
            rep_opt_global,
            step_num,
            1.0,
        )
        Self.Config.OptType.step[Self.Config.DynModel.PARAM_SIZE](
            dyn_params,
            dyn_grads,
            dyn_opt_state,
            dyn_opt_global,
            step_num,
            1.0,
        )
        Self.Config.OptType.step[Self.Config.PredModel.PARAM_SIZE](
            pred_params,
            pred_grads,
            pred_opt_state,
            pred_opt_global,
            step_num,
            1.0,
        )
        Self.Config.OptType.step[Self.Config.ProjectorModel.PARAM_SIZE](
            proj_params,
            proj_grads,
            proj_opt_state,
            proj_opt_global,
            step_num,
            1.0,
        )
        Self.Config.OptType.step[Self.Config.PredictorModel.PARAM_SIZE](
            predr_params,
            predr_grads,
            predr_opt_state,
            predr_opt_global,
            step_num,
            1.0,
        )

        # Reward-prefix LSTM + MLP head Adam step (only when wired in).
        comptime if Self.Config.use_reward_prefix:
            if self.max_grad_norm > 0.0:
                _clip_grads_inplace(
                    self.state.lstm_grads,
                    _LSTMHead.PARAM_SIZE,
                    self.max_grad_norm,
                )
                _clip_grads_inplace(
                    self.state.reward_prefix_mlp.grads,
                    _RewardPrefixMLP.PARAM_SIZE,
                    self.max_grad_norm,
                )
            var lstm_params_v_o = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.lstm_params)
            var lstm_grads_v_o = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.lstm_grads)
            var lstm_opt_state_v = LayoutTensor[
                dtype,
                Layout.row_major(
                    _LSTMHead.PARAM_SIZE,
                    Self.Config.OptType.STATE_PER_PARAM,
                ),
                MutAnyOrigin,
            ](self.state.lstm_opt_state)
            # The LSTM's optimizer-global state is a private one-element
            # scratch; matches the trait shape but stays empty / unused
            # on the CPU step path.
            var _lstm_opt_global_arr = InlineArray[
                Scalar[dtype], Self.Config.OptType.GLOBAL_STATE_SIZE
            ](uninitialized=True)
            for _gi in range(Self.Config.OptType.GLOBAL_STATE_SIZE):
                _lstm_opt_global_arr[_gi] = Scalar[dtype](0.0)
            var lstm_opt_global_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
                MutAnyOrigin,
            ](_lstm_opt_global_arr.unsafe_ptr())
            Self.Config.OptType.step[_LSTMHead.PARAM_SIZE](
                lstm_params_v_o,
                lstm_grads_v_o,
                lstm_opt_state_v,
                lstm_opt_global_v,
                step_num,
                1.0,
            )

            var mlp_head_params_o = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.params)
            var mlp_head_grads_o = LayoutTensor[
                dtype,
                Layout.row_major(_RewardPrefixMLP.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.grads)
            var mlp_head_opt_state_o = LayoutTensor[
                dtype,
                Layout.row_major(
                    _RewardPrefixMLP.PARAM_SIZE,
                    Self.Config.OptType.STATE_PER_PARAM,
                ),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.optimizer_state)
            var mlp_head_opt_global_o = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
                MutAnyOrigin,
            ](self.state.reward_prefix_mlp.opt_global_state)
            Self.Config.OptType.step[_RewardPrefixMLP.PARAM_SIZE](
                mlp_head_params_o,
                mlp_head_grads_o,
                mlp_head_opt_state_o,
                mlp_head_opt_global_o,
                step_num,
                1.0,
            )

        # ── Free scratch ─────────────────────────────────────────────────
        batch_obs.free()
        batch_actions.free()
        batch_rewards.free()
        batch_mcts_pol.free()
        batch_mcts_val.free()
        batch_age.free()
        batch_start_idx.free()
        dyn_out_buf.free()
        rep_input.free()
        proj_dyn_buf.free()
        pred_dyn_buf.free()
        proj_obs_buf.free()
        grad_pred_out.free()
        grad_dyn_out.free()
        grad_pred_dyn.free()
        grad_hidden.free()
        rew_pref_logits.free()
        grad_rew_pref_logits.free()
        lstm_h_input.free()
        lstm_c_input.free()

        var L_total = (
            Self.Config.lambda_reward * L_R
            + Self.Config.lambda_policy * L_P
            + Self.Config.lambda_value * L_V
            + Self.Config.lambda_consistency * L_G
        )
        return (L_total, L_R, L_P, L_V, L_G)

    # ══════════════════════════════════════════════════════════════════════
    # GPU train step (item #7 — work-units priority list)
    # ══════════════════════════════════════════════════════════════════════
    #
    # Hybrid CPU/GPU layout:
    #   * Sampling (priority-weighted, done-flag-aware) and the
    #     mixed-value-target (paper Eq. 16) computation stay on host —
    #     both need the per-transition `priorities` / `step_at_write` /
    #     `dones` arrays that already live in `EZV2DiscreteCPUState`.
    #   * Sampled batch + mixed-value-target are uploaded ONCE per train
    #     step into pinned host buffers → device buffers.
    #   * Forward / backward / optimizer step / priority update all run
    #     on GPU.
    #   * Loss-component scalars + per-sample priority deltas come back
    #     to host so the reported `(L_total, L_R, L_P, L_V, L_G)` and
    #     the host-side `state.priorities[]` array stay in sync with
    #     the CPU `train_step` implementation.

    def train_step_gpu(
        mut self,
        mut gpu: EZV2DiscreteGPUState[Self.Config],
        ctx: DeviceContext,
    ) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """GPU mirror of `train_step`. Returns `(L_total, L_R, L_P, L_V, L_G)`.

        Caller owns `gpu` (a `EZV2DiscreteGPUState[Self.Config]` created
        once at training start) and the `DeviceContext`. The GPU state's
        network params are assumed to already reflect the agent's CPU
        state — call `gpu.upload_from(self.state, ctx)` once at training
        start, and `gpu.download_to(self.state, ctx)` whenever the host
        path (Gumbel search at action-selection, reanalyze) needs fresh
        weights.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime N_TD = Self.Config.td_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime LATENT = Self.Config.latent_dim
        comptime PROJ = Self.Config.proj_dim
        comptime BINS = Self.Config.num_bins
        comptime DYN_IN = LATENT + ACT
        comptime DYN_OUT = LATENT + BINS
        comptime PRED_OUT = ACT + BINS
        comptime CAP = 50000

        comptime TPB: Int = 256
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime BATCH_BINS_BLOCKS = (BATCH * BINS + TPB - 1) // TPB
        comptime LATENT_BLOCKS = (BATCH * LATENT + TPB - 1) // TPB

        if not self.state.is_ready():
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        # ── 1. CPU-side sampling (priority-weighted, with mixed value target) ─
        # Build cumulative-priority array over valid window starts.
        # Mirrors `train_step`'s sampling block + the mixed-value-target
        # block, but emits the result into pinned host buffers.
        var buf_size = self.state.buffer.size
        var buf_ptr = self.state.buffer.ptr
        var current_train_step = self.train_step_count
        var oldest = (buf_ptr - buf_size + CAP) % CAP

        var n_cands_alloc = buf_size - K if buf_size > K else 1
        var cum_prio = alloc[Float64](n_cands_alloc)
        var cand_starts = alloc[Int](n_cands_alloc)
        var n_valid = 0
        var total_prio = Float64(0.0)
        if buf_size > K:
            for offset in range(buf_size - K):
                var idx = (oldest + offset) % CAP
                var valid = True
                for k in range(K):
                    var iidx = (idx + k) % CAP
                    if Float64(self.state.buffer.dones[iidx]) > 0.5:
                        valid = False
                        break
                if not valid:
                    continue
                var p = Float64(self.state.priorities[idx])
                if p < 1e-8:
                    p = 1e-8
                total_prio += p
                cum_prio[n_valid] = total_prio
                cand_starts[n_valid] = idx
                n_valid += 1

        if n_valid < BATCH or total_prio < 1e-8:
            cum_prio.free()
            cand_starts.free()
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        var batch_start_idx = alloc[Int](BATCH)
        # Fill the upload host buffers in per-sample-time-major layout.
        for sampled in range(BATCH):
            var u = random_float64() * total_prio
            var picked = 0
            for i in range(n_valid):
                if cum_prio[i] >= u:
                    picked = i
                    break
            var start = cand_starts[picked]
            batch_start_idx[sampled] = start

            for k in range(K + 1):
                var idx = (start + k) % CAP
                for d in range(OBS):
                    gpu.batch_obs_host[
                        (sampled * (K + 1) + k) * OBS + d
                    ] = self.state.buffer.obs[idx * OBS + d]
                for a in range(ACT):
                    gpu.batch_mcts_pol_host[
                        (sampled * (K + 1) + k) * ACT + a
                    ] = self.state.mcts_policies[idx * ACT + a]
                gpu.batch_mcts_val_host[
                    sampled * (K + 1) + k
                ] = self.state.mcts_values[idx]
                var age = current_train_step - Int(
                    self.state.step_at_write[idx]
                )
                if age < 0:
                    age = 0
                gpu.batch_age_host[sampled * (K + 1) + k] = Scalar[
                    DType.int32
                ](age)
            for k in range(K):
                var idx = (start + k) % CAP
                for a in range(ACT):
                    gpu.batch_actions_host[
                        (sampled * K + k) * ACT + a
                    ] = self.state.buffer.actions[idx * ACT + a]
                gpu.batch_rewards_host[
                    sampled * K + k
                ] = self.state.buffer.rewards[(start + k) % CAP]

            # Cumulative-reward target for the reward-prefix LSTM head
            # (paper App. G). Computed even when use_reward_prefix=False
            # so the upload path stays the same; the kernel just doesn't
            # consume it then.
            var cum = Float64(0.0)
            for k in range(K):
                cum += Float64(
                    gpu.batch_rewards_host[sampled * K + k]
                )
                gpu.cum_rewards_host[sampled * K + k] = Scalar[dtype](cum)

        cum_prio.free()
        cand_starts.free()

        # ── Fresh bootstrap values from target nets (Lever 1). ───────────
        # Computed on host using the CPU target nets — only when the
        # value-target mode actually consumes them. SEARCH skips the
        # forward and zeroes `boot_v_host`; SARSA always uses it; MIXED
        # may use it depending on age.
        var boot_v_host = alloc[Scalar[dtype]](BATCH * (K + 1))
        memset(boot_v_host, 0, BATCH * (K + 1))
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
            var tgt_rep_input = alloc[Scalar[dtype]](BATCH * OBS)
            var tgt_z = alloc[Scalar[dtype]](BATCH * LATENT)
            var tgt_pred_out = alloc[Scalar[dtype]](BATCH * PRED_OUT)
            var tgt_logits_dbl = alloc[Float64](BINS)

            var tgt_rep_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.params)
            var tgt_rep_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.model_state)
            var tgt_pred_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.params)
            var tgt_pred_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.model_state)

            for k in range(K + 1):
                for b in range(BATCH):
                    for d in range(OBS):
                        tgt_rep_input[b * OBS + d] = gpu.batch_obs_host[
                            (b * (K + 1) + k) * OBS + d
                        ]
                var tgt_rep_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_rep_input)
                var tgt_z_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                Network[
                    Self.Config.RepModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_rep_in_t,
                    tgt_z_t,
                    tgt_rep_params,
                    tgt_rep_state,
                )
                var tgt_pred_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                var tgt_pred_out_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_pred_out)
                Network[
                    Self.Config.PredModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_pred_in_t,
                    tgt_pred_out_t,
                    tgt_pred_params,
                    tgt_pred_state,
                )
                for b in range(BATCH):
                    var off = b * PRED_OUT + ACT
                    for i in range(BINS):
                        tgt_logits_dbl[i] = Float64(tgt_pred_out[off + i])
                    var v_raw = decode_categorical[BINS](
                        tgt_logits_dbl, self.v_min, self.v_max
                    )
                    boot_v_host[b * (K + 1) + k] = Scalar[dtype](
                        inverse_scalar_transform(v_raw)
                    )

            tgt_rep_input.free()
            tgt_z.free()
            tgt_pred_out.free()
            tgt_logits_dbl.free()

        # ── Value target (SEARCH / SARSA / MIXED) precomputed on host
        # so the GPU side just sees a plain scalar tensor it can scalar-
        # transform + two-hot-encode per k.
        for sampled in range(BATCH):
            for k in range(K + 1):
                var sve = Float64(
                    gpu.batch_mcts_val_host[sampled * (K + 1) + k]
                )
                var n_eff = N_TD if N_TD < K - k else K - k
                var td = Float64(0.0)
                comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
                    var disc = Float64(1.0)
                    for j in range(n_eff):
                        td += disc * Float64(
                            gpu.batch_rewards_host[sampled * K + k + j]
                        )
                        disc *= self.gamma
                    td += disc * Float64(
                        boot_v_host[sampled * (K + 1) + k + n_eff]
                    )
                var age = Int(gpu.batch_age_host[sampled * (K + 1) + k])
                var v_target = Float64(0.0)
                comptime if Self.Config.value_target_mode == VALUE_TARGET_SEARCH:
                    v_target = sve
                elif Self.Config.value_target_mode == VALUE_TARGET_SARSA:
                    v_target = td
                else:  # VALUE_TARGET_MIXED
                    v_target = MixedValueTarget[
                        Self.Config.t_fresh, Self.Config.t_stale
                    ].compute(sve, td, age)
                gpu.value_target_full_host[
                    sampled * (K + 1) + k
                ] = Scalar[dtype](v_target)

        boot_v_host.free()

        # ── 2. Upload host → device ─────────────────────────────────────
        ctx.enqueue_copy(gpu.batch_obs_buf, gpu.batch_obs_host)
        ctx.enqueue_copy(gpu.batch_actions_buf, gpu.batch_actions_host)
        ctx.enqueue_copy(gpu.batch_rewards_buf, gpu.batch_rewards_host)
        ctx.enqueue_copy(gpu.batch_mcts_pol_buf, gpu.batch_mcts_pol_host)
        ctx.enqueue_copy(gpu.batch_mcts_val_buf, gpu.batch_mcts_val_host)
        ctx.enqueue_copy(gpu.batch_age_buf, gpu.batch_age_host)
        ctx.enqueue_copy(
            gpu.value_target_full_buf, gpu.value_target_full_host
        )
        comptime if Self.Config.use_reward_prefix:
            ctx.enqueue_copy(gpu.cum_rewards_buf, gpu.cum_rewards_host)

        # ── 3. Zero loss accumulators + per-network grads ───────────────
        ctx.enqueue_memset(gpu.L_R_buf, 0)
        ctx.enqueue_memset(gpu.L_P_buf, 0)
        ctx.enqueue_memset(gpu.L_V_buf, 0)
        ctx.enqueue_memset(gpu.L_G_buf, 0)
        ctx.enqueue_memset(gpu.grad_hidden_buf, 0)
        # The CE/cosine kernels overwrite (not accumulate into) the
        # per-output grad slices they're responsible for, so we don't
        # need to zero the grad_pred_out / grad_dyn_out / grad_pred_dyn
        # buffers across train steps — every position they care about
        # gets re-written before being read.
        gpu.representation.zero_grads(ctx)
        gpu.dynamics.zero_grads(ctx)
        gpu.prediction.zero_grads(ctx)
        gpu.projector.zero_grads(ctx)
        gpu.predictor.zero_grads(ctx)

        # ── 4. Forward pass ─────────────────────────────────────────────

        # 4.0 Pre-built LayoutTensor views that we reuse across calls.
        var batch_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * (K + 1) * OBS),
            MutAnyOrigin,
        ](gpu.batch_obs_buf.unsafe_ptr())
        var batch_actions_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * K * ACT),
            MutAnyOrigin,
        ](gpu.batch_actions_buf.unsafe_ptr())
        var batch_rewards_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * K),
            MutAnyOrigin,
        ](gpu.batch_rewards_buf.unsafe_ptr())
        var batch_mcts_pol_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * (K + 1) * ACT),
            MutAnyOrigin,
        ](gpu.batch_mcts_pol_buf.unsafe_ptr())
        var value_target_full_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * (K + 1)),
            MutAnyOrigin,
        ](gpu.value_target_full_buf.unsafe_ptr())

        var rep_input_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.rep_input_buf.unsafe_ptr())
        var rep_input_t_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * OBS),
            MutAnyOrigin,
        ](gpu.rep_input_buf.unsafe_ptr())
        var dyn_input_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.DynModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.dyn_input_buf.unsafe_ptr())
        var dyn_input_t_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * DYN_IN),
            MutAnyOrigin,
        ](gpu.dyn_input_buf.unsafe_ptr())
        var obs_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.obs_step_buf.unsafe_ptr())
        var obs_step_t_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * OBS),
            MutAnyOrigin,
        ](gpu.obs_step_buf.unsafe_ptr())
        var rep_obs_t_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.rep_obs_buf.unsafe_ptr())
        var rep_obs_t_in_proj = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.rep_obs_buf.unsafe_ptr())

        # 4.1 rep(o[0]) → hidden[0]
        # Gather batch_obs[:, 0, :] into rep_input_buf.
        comptime gather_obs = ezv2_copy_obs_at_step_kernel[
            BATCH, K + 1, OBS, dtype
        ]
        ctx.enqueue_function[gather_obs, gather_obs](
            batch_obs_t,
            rep_input_t_flat,
            0,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        var hidden0_t_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.hidden_buf.unsafe_ptr())
        var rep_cache_t_w = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu.rep_cache_buf.unsafe_ptr())
        Network[Self.Config.RepModel, Self.Config.OptType].forward_gpu_with_cache[
            BATCH
        ](
            ctx,
            rep_input_t_in,
            hidden0_t_out,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            rep_cache_t_w,
            gpu.workspace_buf,
        )

        # 4.2 K dynamics steps: for k=0..K-1 build dyn_input from hidden[k]
        # + actions[:, k, :], call dyn.forward_with_cache → dyn_out[k],
        # extract hidden[k+1].
        comptime build_dyn_in = ezv2_build_dyn_input_kernel[
            BATCH, LATENT, ACT, K, dtype
        ]
        comptime extract_hidden = ezv2_extract_hidden_after_dyn_kernel[
            BATCH, LATENT, BINS, dtype
        ]
        comptime DYN_CS = Self.Config.DynModel.CACHE_SIZE

        for k in range(K):
            # hidden[k] view (single time-slice).
            var hidden_k_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LATENT),
                MutAnyOrigin,
            ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)

            ctx.enqueue_function[build_dyn_in, build_dyn_in](
                hidden_k_flat,
                batch_actions_t,
                dyn_input_t_flat,
                k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            var dyn_out_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
            var dyn_cache_k_w = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu.dyn_caches_buf.unsafe_ptr() + k * BATCH * DYN_CS)
            Network[
                Self.Config.DynModel, Self.Config.OptType
            ].forward_gpu_with_cache[BATCH](
                ctx,
                dyn_input_t_in,
                dyn_out_k_in,
                gpu.dynamics.params_view(),
                gpu.dynamics.model_state_view(),
                dyn_cache_k_w,
                gpu.workspace_buf,
            )

            # Extract hidden part: dyn_out_buf[k][:, :LATENT] → hidden[k+1].
            var dyn_out_k_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * DYN_OUT),
                MutAnyOrigin,
            ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
            var hidden_kp1_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LATENT),
                MutAnyOrigin,
            ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
            ctx.enqueue_function[extract_hidden, extract_hidden](
                dyn_out_k_flat,
                hidden_kp1_flat,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # 4.3 Pred at k = 0..K
        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        for k in range(K + 1):
            var hidden_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
            var pred_out_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            var pred_cache_k_w = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu.pred_caches_buf.unsafe_ptr() + k * BATCH * PRED_CS)
            Network[
                Self.Config.PredModel, Self.Config.OptType
            ].forward_gpu_with_cache[BATCH](
                ctx,
                hidden_k_in,
                pred_out_k_in,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                pred_cache_k_w,
                gpu.workspace_buf,
            )

        # 4.4 SimSiam branches for k_offset = 0..K-1 (k = k_offset + 1).
        comptime PROJ_CS = Self.Config.ProjectorModel.CACHE_SIZE
        comptime PREDR_CS = Self.Config.PredictorModel.CACHE_SIZE

        for k_offset in range(K):
            var k = k_offset + 1

            # Online projector: projector(hidden[k]) → proj_dyn[k_offset]
            var hidden_k_in_proj = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
            var proj_dyn_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.proj_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var proj_dyn_cache_k_w = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                gpu.proj_dyn_caches_buf.unsafe_ptr()
                + k_offset * BATCH * PROJ_CS
            )
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward_gpu_with_cache[BATCH](
                ctx,
                hidden_k_in_proj,
                proj_dyn_k_in,
                gpu.projector.params_view(),
                gpu.projector.model_state_view(),
                proj_dyn_cache_k_w,
                gpu.workspace_buf,
            )

            # Predictor: predictor(proj_dyn[k_offset]) → pred_dyn[k_offset]
            var proj_dyn_k_input = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.proj_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var pred_dyn_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var pred_dyn_cache_k_w = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                gpu.pred_dyn_caches_buf.unsafe_ptr()
                + k_offset * BATCH * PREDR_CS
            )
            Network[
                Self.Config.PredictorModel, Self.Config.OptType
            ].forward_gpu_with_cache[BATCH](
                ctx,
                proj_dyn_k_input,
                pred_dyn_k_in,
                gpu.predictor.params_view(),
                gpu.predictor.model_state_view(),
                pred_dyn_cache_k_w,
                gpu.workspace_buf,
            )

            # Target branch (no cache, no gradient).
            # Gather batch_obs[:, k, :] → obs_step.
            ctx.enqueue_function[gather_obs, gather_obs](
                batch_obs_t,
                obs_step_t_flat,
                k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # rep(obs_step) → rep_obs (no cache; stop-grad target)
            Network[
                Self.Config.RepModel, Self.Config.OptType
            ].forward_gpu[BATCH](
                ctx,
                obs_step_t_in,
                rep_obs_t_out,
                gpu.representation.params_view(),
                gpu.representation.model_state_view(),
                gpu.workspace_buf,
            )

            # projector(rep_obs) → proj_obs[k_offset] (no cache)
            var proj_obs_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.proj_obs_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward_gpu[BATCH](
                ctx,
                rep_obs_t_in_proj,
                proj_obs_k_in,
                gpu.projector.params_view(),
                gpu.projector.model_state_view(),
                gpu.workspace_buf,
            )

        # ── 4.5 Reward-prefix LSTM forward (paper App. G, when enabled) ─
        comptime LSTM_HIDDEN = Self.Config.lstm_hidden
        comptime LSTM_HORIZON = Self.Config.lstm_horizon_len
        comptime _LSTMHead = LSTMCell[LATENT, LSTM_HIDDEN]
        comptime _RewardPrefixMLP = RewardPrefixHeadMLP[
            LSTM_HIDDEN,
            Self.Config.lstm_mlp_hidden,
            BINS,
        ]
        comptime _LSTM_CS = _LSTMHead.CACHE_SIZE
        comptime _MLP_HEAD_CS = _RewardPrefixMLP.CACHE_SIZE
        comptime LSTM_HIDDEN_BLOCKS = (
            BATCH * LSTM_HIDDEN + TPB - 1
        ) // TPB
        comptime copy_lstm_input = ezv2_copy_lstm_input_kernel[
            BATCH, LSTM_HIDDEN, dtype
        ]
        comptime add_kernel_lstm_h = ezv2_add_kernel[
            BATCH * LSTM_HIDDEN, dtype
        ]
        comptime add_kernel_latent = ezv2_add_kernel[BATCH * LATENT, dtype]

        comptime if Self.Config.use_reward_prefix:
            # Reset h_lstm[0], c_lstm[0] = 0 at the start of every batch.
            ctx.enqueue_memset(gpu.lstm_h_states_buf, 0)
            ctx.enqueue_memset(gpu.lstm_c_states_buf, 0)

            var lstm_params_v_f = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_params_buf.unsafe_ptr())

            for k in range(K):
                var reset_now = (k > 0) and (k % LSTM_HORIZON == 0)
                if reset_now:
                    ctx.enqueue_memset(gpu.lstm_h_input_buf, 0)
                    ctx.enqueue_memset(gpu.lstm_c_input_buf, 0)
                else:
                    # Copy lstm_h_states[k], lstm_c_states[k] → input scratch.
                    var hs_k_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.lstm_h_states_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var cs_k_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.lstm_c_states_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var h_in_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.lstm_h_input_buf.unsafe_ptr())
                    var c_in_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.lstm_c_input_buf.unsafe_ptr())
                    ctx.enqueue_function[copy_lstm_input, copy_lstm_input](
                        hs_k_flat,
                        cs_k_flat,
                        h_in_flat,
                        c_in_flat,
                        grid_dim=(LSTM_HIDDEN_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # LSTM step k: input is hidden[k+1] (post-dyn-step-k latent).
                var z_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
                var h_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_h_input_buf.unsafe_ptr())
                var c_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_c_input_buf.unsafe_ptr())
                var h_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_h_states_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var c_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_c_states_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var lstm_cache_t_f = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _LSTM_CS),
                    MutAnyOrigin,
                ](gpu.lstm_caches_buf.unsafe_ptr() + k * BATCH * _LSTM_CS)
                _LSTMHead.step_forward_gpu[BATCH](
                    ctx,
                    z_t,
                    h_prev_t,
                    c_prev_t,
                    lstm_params_v_f,
                    h_t,
                    c_t,
                    lstm_cache_t_f,
                )

                # MLP head forward(h_lstm[k+1]) → rew_pref_logits[k]
                var mlp_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                    MutAnyOrigin,
                ](
                    gpu.lstm_h_states_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var mlp_out_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                    MutAnyOrigin,
                ](
                    gpu.rew_pref_logits_buf.unsafe_ptr()
                    + k * BATCH * BINS
                )
                var mlp_cache_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _MLP_HEAD_CS),
                    MutAnyOrigin,
                ](
                    gpu.mlp_head_caches_buf.unsafe_ptr()
                    + k * BATCH * _MLP_HEAD_CS
                )
                Network[
                    _RewardPrefixMLP, Self.Config.OptType
                ].forward_gpu_with_cache[BATCH](
                    ctx,
                    mlp_in_t,
                    mlp_out_t,
                    gpu.reward_prefix_mlp_gpu.params_view(),
                    gpu.reward_prefix_mlp_gpu.model_state_view(),
                    mlp_cache_t,
                    gpu.workspace_buf,
                )

        # ── 5. Per-output upstream gradients + per-sample loss reductions ─

        # 5.1 Policy CE at every k = 0..K
        comptime gather_pol = ezv2_gather_policy_target_kernel[
            BATCH, K + 1, ACT, dtype
        ]
        comptime policy_grad = ezv2_policy_loss_grad_kernel[
            BATCH, ACT, PRED_OUT, dtype
        ]
        comptime reduce_one = ezv2_reduce_add_kernel[BATCH, dtype]
        var policy_target_step_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
        ](gpu.policy_target_step_buf.unsafe_ptr())
        var per_sample_loss_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.per_sample_loss_scratch_buf.unsafe_ptr())
        var L_P_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu.L_P_buf.unsafe_ptr())

        var n_P = Float64(BATCH * (K + 1))
        var lp_scale = self.Config.lambda_policy / n_P
        for k in range(K + 1):
            ctx.enqueue_function[gather_pol, gather_pol](
                batch_mcts_pol_t,
                policy_target_step_t,
                k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            var pred_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            var grad_pred_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            ctx.enqueue_function[policy_grad, policy_grad](
                pred_out_k_flat,
                policy_target_step_t,
                grad_pred_out_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lp_scale),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[reduce_one, reduce_one](
                per_sample_loss_t,
                L_P_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

        # 5.2 Value CE at every k = 0..K, using pre-uploaded mixed value
        # target. Build two-hot dist by gathering scalar → scalar_transform
        # → two_hot_encode, then run value_loss_grad_kernel. At k = 0,
        # additionally copy per_sample_loss to per_sample_v_loss_k0 for
        # the priority refresh.
        comptime gather_val = ezv2_gather_value_target_kernel[
            BATCH, K + 1, dtype
        ]
        comptime st_kernel = scalar_transform_kernel[BATCH, dtype]
        comptime th_kernel = two_hot_encode_kernel[BATCH, BINS, dtype]
        comptime value_grad = ezv2_value_loss_grad_kernel[
            BATCH, BINS, ACT, PRED_OUT, dtype
        ]
        comptime BATCH_BLOCKS_BS = (BATCH + TPB - 1) // TPB
        comptime BINS_BLOCKS = (BATCH + TPB - 1) // TPB

        var value_target_scalar_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.value_target_scalar_buf.unsafe_ptr())
        var value_target_dist_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
        ](gpu.value_target_dist_buf.unsafe_ptr())
        var L_V_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu.L_V_buf.unsafe_ptr())
        var per_sample_v_loss_k0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.per_sample_v_loss_k0_buf.unsafe_ptr())

        var n_V = Float64(BATCH * (K + 1))
        var lv_scale = self.Config.lambda_value / n_V
        comptime copy_kernel_b = ezv2_add_kernel[BATCH, dtype]
        # NOTE: at k=0 we want a *copy* of per_sample_loss into
        # per_sample_v_loss_k0_buf, not an add. Easiest: zero
        # per_sample_v_loss_k0_buf via memset before the loop, then add.
        ctx.enqueue_memset(gpu.per_sample_v_loss_k0_buf, 0)

        for k in range(K + 1):
            ctx.enqueue_function[gather_val, gather_val](
                value_target_full_t,
                value_target_scalar_t,
                k,
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[st_kernel, st_kernel](
                value_target_scalar_t,
                Scalar[dtype](0.001),
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[th_kernel, th_kernel](
                value_target_dist_t,
                value_target_scalar_t,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                grid_dim=(BINS_BLOCKS,),
                block_dim=(TPB,),
            )
            var pred_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            var grad_pred_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            ctx.enqueue_function[value_grad, value_grad](
                pred_out_k_flat,
                value_target_dist_t,
                grad_pred_out_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lv_scale),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[reduce_one, reduce_one](
                per_sample_loss_t,
                L_V_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
            if k == 0:
                ctx.enqueue_function[copy_kernel_b, copy_kernel_b](
                    per_sample_v_loss_k0_t,
                    per_sample_loss_t,
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

        # 5.3 Reward CE at every k = 0..K-1.
        # Two paths (matches CPU):
        #   • use_reward_prefix=False: classic per-step reward CE through
        #     the dyn-network's reward output slice.
        #   • use_reward_prefix=True: cumulative-reward CE through the
        #     reward-prefix LSTM head (rew_pref_logits_buf). The dyn-
        #     output's reward grad slice MUST be zeroed so dyn-backward
        #     sees no gradient through that path.
        comptime gather_rew = ezv2_gather_reward_at_step_kernel[
            BATCH, K, dtype
        ]
        comptime reward_grad = ezv2_reward_loss_grad_kernel[
            BATCH, BINS, LATENT, DYN_OUT, dtype
        ]
        comptime rew_pref_grad = ezv2_reward_prefix_loss_grad_kernel[
            BATCH, BINS, dtype
        ]
        var reward_target_scalar_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.reward_target_scalar_buf.unsafe_ptr())
        var reward_target_dist_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
        ](gpu.reward_target_dist_buf.unsafe_ptr())
        var L_R_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu.L_R_buf.unsafe_ptr())
        var cum_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * K), MutAnyOrigin
        ](gpu.cum_rewards_buf.unsafe_ptr())
        var rew_pref_target_dist_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
        ](gpu.rew_pref_target_dist_buf.unsafe_ptr())

        var n_R = Float64(BATCH * K)
        var lr_scale = self.Config.lambda_reward / n_R

        comptime if not Self.Config.use_reward_prefix:
            for k in range(K):
                ctx.enqueue_function[gather_rew, gather_rew](
                    batch_rewards_t,
                    reward_target_scalar_t,
                    k,
                    grid_dim=(BATCH_BLOCKS_BS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[st_kernel, st_kernel](
                    reward_target_scalar_t,
                    Scalar[dtype](0.001),
                    grid_dim=(BATCH_BLOCKS_BS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[th_kernel, th_kernel](
                    reward_target_dist_t,
                    reward_target_scalar_t,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    grid_dim=(BINS_BLOCKS,),
                    block_dim=(TPB,),
                )
                var dyn_out_k_flat = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
                ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
                var grad_dyn_out_k_flat = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
                ](gpu.grad_dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
                ctx.enqueue_function[reward_grad, reward_grad](
                    dyn_out_k_flat,
                    reward_target_dist_t,
                    grad_dyn_out_k_flat,
                    per_sample_loss_t,
                    Scalar[dtype](lr_scale),
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[reduce_one, reduce_one](
                    per_sample_loss_t,
                    L_R_t,
                    grid_dim=(1,),
                    block_dim=(1,),
                )
        else:
            # Zero the dyn-output reward grad slice so dyn-backward sees
            # no gradient through that path. (Without this, stale data
            # from previous train_step calls would flow through.) The
            # whole grad_dyn_out_buf is fine to zero — the pred-out
            # grad and the K-1 hidden-slice piece get rebuilt anyway.
            ctx.enqueue_memset(gpu.grad_dyn_out_buf, 0)

            for k in range(K):
                # Gather cum_rewards[:, k] → reward_target_scalar.
                ctx.enqueue_function[gather_rew, gather_rew](
                    cum_rewards_t,
                    reward_target_scalar_t,
                    k,
                    grid_dim=(BATCH_BLOCKS_BS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[st_kernel, st_kernel](
                    reward_target_scalar_t,
                    Scalar[dtype](0.001),
                    grid_dim=(BATCH_BLOCKS_BS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[th_kernel, th_kernel](
                    rew_pref_target_dist_t,
                    reward_target_scalar_t,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    grid_dim=(BINS_BLOCKS,),
                    block_dim=(TPB,),
                )
                var rpl_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * BINS),
                    MutAnyOrigin,
                ](
                    gpu.rew_pref_logits_buf.unsafe_ptr()
                    + k * BATCH * BINS
                )
                var grad_rpl_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * BINS),
                    MutAnyOrigin,
                ](
                    gpu.grad_rew_pref_logits_buf.unsafe_ptr()
                    + k * BATCH * BINS
                )
                ctx.enqueue_function[rew_pref_grad, rew_pref_grad](
                    rpl_k_flat,
                    rew_pref_target_dist_t,
                    grad_rpl_k_flat,
                    per_sample_loss_t,
                    Scalar[dtype](lr_scale),
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[reduce_one, reduce_one](
                    per_sample_loss_t,
                    L_R_t,
                    grid_dim=(1,),
                    block_dim=(1,),
                )

        # 5.4 Cosine consistency at every k_offset = 0..K-1.
        comptime cosine_grad = ezv2_cosine_loss_grad_kernel[
            BATCH, PROJ, dtype
        ]
        var L_G_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu.L_G_buf.unsafe_ptr())

        var n_G = Float64(BATCH * K)
        var lg_scale = self.Config.lambda_consistency / n_G
        for k_offset in range(K):
            var pred_dyn_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
            ](gpu.pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var proj_obs_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
            ](gpu.proj_obs_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var grad_pred_dyn_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
            ](gpu.grad_pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            ctx.enqueue_function[cosine_grad, cosine_grad](
                pred_dyn_k_flat,
                proj_obs_k_flat,
                grad_pred_dyn_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lg_scale),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[reduce_one, reduce_one](
                per_sample_loss_t,
                L_G_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

        # ── 6. Backward pass ────────────────────────────────────────────

        # 6.1 pred backward at k = 0..K → adds into grad_hidden[k].
        comptime add_kernel_lat = ezv2_add_kernel[BATCH * LATENT, dtype]
        for k in range(K + 1):
            var grad_pred_out_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
            var grad_pred_in_step_t_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.grad_pred_in_step_buf.unsafe_ptr())
            var pred_cache_k_r = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu.pred_caches_buf.unsafe_ptr() + k * BATCH * PRED_CS)
            var pred_grads_v = gpu.prediction.grads_view()
            Network[
                Self.Config.PredModel, Self.Config.OptType
            ].backward_gpu[BATCH](
                ctx,
                grad_pred_out_k_in,
                grad_pred_in_step_t_in,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                pred_cache_k_r,
                pred_grads_v,
                gpu.workspace_buf,
            )
            # grad_hidden[k] += grad_pred_in_step
            var grad_hidden_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
            var grad_pred_in_step_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](gpu.grad_pred_in_step_buf.unsafe_ptr())
            ctx.enqueue_function[add_kernel_lat, add_kernel_lat](
                grad_hidden_k_flat,
                grad_pred_in_step_flat,
                grid_dim=(LATENT_BLOCKS,),
                block_dim=(TPB,),
            )

        # 6.2 SimSiam backward (predictor + projector online branch) at
        # k_offset = 0..K-1 → adds into grad_hidden[k_offset + 1].
        for k_offset in range(K):
            var k = k_offset + 1
            # predictor.backward(grad_pred_dyn[k_offset]) → grad_predr_in_step
            var grad_pred_dyn_k_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.grad_pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
            var grad_predr_in_step_t_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.grad_predr_in_step_buf.unsafe_ptr())
            var pred_dyn_cache_k_r = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                gpu.pred_dyn_caches_buf.unsafe_ptr()
                + k_offset * BATCH * PREDR_CS
            )
            var predr_grads_v = gpu.predictor.grads_view()
            Network[
                Self.Config.PredictorModel, Self.Config.OptType
            ].backward_gpu[BATCH](
                ctx,
                grad_pred_dyn_k_in,
                grad_predr_in_step_t_in,
                gpu.predictor.params_view(),
                gpu.predictor.model_state_view(),
                pred_dyn_cache_k_r,
                predr_grads_v,
                gpu.workspace_buf,
            )

            # projector.backward(grad_predr_in_step → grad_proj_in_step)
            var grad_predr_in_step_in_proj = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.grad_predr_in_step_buf.unsafe_ptr())
            var grad_proj_in_step_t_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.grad_proj_in_step_buf.unsafe_ptr())
            var proj_dyn_cache_k_r = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](
                gpu.proj_dyn_caches_buf.unsafe_ptr()
                + k_offset * BATCH * PROJ_CS
            )
            var proj_grads_v = gpu.projector.grads_view()
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].backward_gpu[BATCH](
                ctx,
                grad_predr_in_step_in_proj,
                grad_proj_in_step_t_in,
                gpu.projector.params_view(),
                gpu.projector.model_state_view(),
                proj_dyn_cache_k_r,
                proj_grads_v,
                gpu.workspace_buf,
            )

            # grad_hidden[k] += grad_proj_in_step
            var grad_hidden_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
            var grad_proj_in_step_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](gpu.grad_proj_in_step_buf.unsafe_ptr())
            ctx.enqueue_function[add_kernel_lat, add_kernel_lat](
                grad_hidden_k_flat,
                grad_proj_in_step_flat,
                grid_dim=(LATENT_BLOCKS,),
                block_dim=(TPB,),
            )

        # 6.2'. Reward-prefix LSTM head backward (when use_reward_prefix=True)
        # — pass 1: MLP head backward at every k → adds to grad_h_lstm[k+1].
        # pass 2: LSTM step_backward in REVERSE time, threading dh/dc back to
        # the previous step (or discarding at horizon-reset boundary), and
        # accumulating ∂L/∂hidden[k+1] into the existing grad_hidden buffer.
        comptime if Self.Config.use_reward_prefix:
            ctx.enqueue_memset(gpu.lstm_grads_buf, 0)
            gpu.reward_prefix_mlp_gpu.zero_grads(ctx)
            ctx.enqueue_memset(gpu.grad_h_lstm_buf, 0)
            ctx.enqueue_memset(gpu.grad_c_lstm_buf, 0)

            # Pass 1: MLP head backward k = 0..K-1.
            for k in range(K):
                var grad_logits_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                    MutAnyOrigin,
                ](
                    gpu.grad_rew_pref_logits_buf.unsafe_ptr()
                    + k * BATCH * BINS
                )
                var grad_mlp_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                    MutAnyOrigin,
                ](gpu.grad_mlp_in_step_buf.unsafe_ptr())
                var mlp_cache_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _MLP_HEAD_CS),
                    MutAnyOrigin,
                ](
                    gpu.mlp_head_caches_buf.unsafe_ptr()
                    + k * BATCH * _MLP_HEAD_CS
                )
                var mlp_grads_v = gpu.reward_prefix_mlp_gpu.grads_view()
                Network[
                    _RewardPrefixMLP, Self.Config.OptType
                ].backward_gpu[BATCH](
                    ctx,
                    grad_logits_t,
                    grad_mlp_in_t,
                    gpu.reward_prefix_mlp_gpu.params_view(),
                    gpu.reward_prefix_mlp_gpu.model_state_view(),
                    mlp_cache_t_b,
                    mlp_grads_v,
                    gpu.workspace_buf,
                )

                # Accumulate grad_mlp_in_step → grad_h_lstm[k+1]
                var grad_h_lstm_kp1_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.grad_h_lstm_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var grad_mlp_in_step_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.grad_mlp_in_step_buf.unsafe_ptr())
                ctx.enqueue_function[
                    add_kernel_lstm_h, add_kernel_lstm_h
                ](
                    grad_h_lstm_kp1_flat,
                    grad_mlp_in_step_flat,
                    grid_dim=(LSTM_HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Pass 2: LSTM backward in REVERSE time, kk = 0..K-1, k = K-1..0.
            var lstm_params_v_b = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_params_buf.unsafe_ptr())
            var lstm_grads_v_b = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_grads_buf.unsafe_ptr())

            for kk in range(K):
                var k = K - 1 - kk
                var reset_now_b = (k > 0) and (k % LSTM_HORIZON == 0)
                if reset_now_b:
                    ctx.enqueue_memset(gpu.lstm_h_input_buf, 0)
                    ctx.enqueue_memset(gpu.lstm_c_input_buf, 0)
                else:
                    # Re-fill the input scratch with h_lstm[k] / c_lstm[k]
                    # (same values used during forward at this step).
                    var hs_k_flat_b = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.lstm_h_states_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var cs_k_flat_b = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.lstm_c_states_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var h_in_flat_b = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.lstm_h_input_buf.unsafe_ptr())
                    var c_in_flat_b = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.lstm_c_input_buf.unsafe_ptr())
                    ctx.enqueue_function[copy_lstm_input, copy_lstm_input](
                        hs_k_flat_b,
                        cs_k_flat_b,
                        h_in_flat_b,
                        c_in_flat_b,
                        grid_dim=(LSTM_HIDDEN_BLOCKS,),
                        block_dim=(TPB,),
                    )

                var dh_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.grad_h_lstm_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var dc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.grad_c_lstm_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LSTM_HIDDEN
                )
                var z_t_b = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
                var h_prev_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_h_input_buf.unsafe_ptr())
                var c_prev_t_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_c_input_buf.unsafe_ptr())
                var lstm_cache_t_bw = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, _LSTM_CS),
                    MutAnyOrigin,
                ](gpu.lstm_caches_buf.unsafe_ptr() + k * BATCH * _LSTM_CS)
                var grad_x_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
                ](gpu.grad_x_lstm_buf.unsafe_ptr())
                var grad_h_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.grad_h_prev_lstm_buf.unsafe_ptr())
                var grad_c_prev_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.grad_c_prev_lstm_buf.unsafe_ptr())
                var d_combined_ws_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, 4 * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_d_combined_ws_buf.unsafe_ptr())
                _LSTMHead.step_backward_gpu[BATCH](
                    ctx,
                    dh_t,
                    dc_t,
                    z_t_b,
                    h_prev_t_b,
                    c_prev_t_b,
                    lstm_params_v_b,
                    lstm_cache_t_bw,
                    grad_x_t,
                    grad_h_prev_t,
                    grad_c_prev_t,
                    lstm_grads_v_b,
                    d_combined_ws_t,
                )

                # Accumulate grad_x → grad_hidden[k+1].
                var grad_hidden_kp1_flat_lstm = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LATENT),
                    MutAnyOrigin,
                ](
                    gpu.grad_hidden_buf.unsafe_ptr()
                    + (k + 1) * BATCH * LATENT
                )
                var grad_x_lstm_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LATENT),
                    MutAnyOrigin,
                ](gpu.grad_x_lstm_buf.unsafe_ptr())
                ctx.enqueue_function[
                    add_kernel_latent, add_kernel_latent
                ](
                    grad_hidden_kp1_flat_lstm,
                    grad_x_lstm_flat,
                    grid_dim=(LATENT_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Thread dh_prev / dc_prev → grad_h_lstm[k] / grad_c_lstm[k]
                # unless this step was a reset boundary (input was zero).
                if not reset_now_b:
                    var grad_h_lstm_k_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.grad_h_lstm_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var grad_c_lstm_k_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](
                        gpu.grad_c_lstm_buf.unsafe_ptr()
                        + k * BATCH * LSTM_HIDDEN
                    )
                    var grad_h_prev_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.grad_h_prev_lstm_buf.unsafe_ptr())
                    var grad_c_prev_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH * LSTM_HIDDEN),
                        MutAnyOrigin,
                    ](gpu.grad_c_prev_lstm_buf.unsafe_ptr())
                    ctx.enqueue_function[
                        add_kernel_lstm_h, add_kernel_lstm_h
                    ](
                        grad_h_lstm_k_flat,
                        grad_h_prev_flat,
                        grid_dim=(LSTM_HIDDEN_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        add_kernel_lstm_h, add_kernel_lstm_h
                    ](
                        grad_c_lstm_k_flat,
                        grad_c_prev_flat,
                        grid_dim=(LSTM_HIDDEN_BLOCKS,),
                        block_dim=(TPB,),
                    )

        # 6.3 dyn backward in REVERSE topological order: kk = K-1..0.
        # Walks BACKWARD in time so grad_hidden[k+1] is fully accumulated
        # (pred + projector + later dyn-backwards) before we consume it
        # at the current step.
        comptime assemble_dyn_grad = (
            ezv2_assemble_grad_dyn_step_kernel[BATCH, LATENT, BINS, dtype]
        )
        comptime accum_dyn_grad_in = (
            ezv2_accumulate_dyn_grad_in_kernel[BATCH, LATENT, ACT, dtype]
        )
        for kk in range(K):
            var k = K - 1 - kk
            # Build grad_dyn_out_step[b] = [grad_hidden[k+1, b] || grad_dyn_out[k, b, LATENT:]]
            var grad_hidden_kp1_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](
                gpu.grad_hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT
            )
            var grad_dyn_out_k_full = LayoutTensor[
                dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
            ](gpu.grad_dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
            var grad_dyn_out_step_t = LayoutTensor[
                dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
            ](gpu.grad_dyn_out_step_buf.unsafe_ptr())
            ctx.enqueue_function[assemble_dyn_grad, assemble_dyn_grad](
                grad_hidden_kp1_flat,
                grad_dyn_out_k_full,
                grad_dyn_out_step_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # dyn.backward(grad_dyn_out_step → grad_dyn_in_step)
            var grad_dyn_out_step_t_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.grad_dyn_out_step_buf.unsafe_ptr())
            var grad_dyn_in_step_t_in = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.grad_dyn_in_step_buf.unsafe_ptr())
            var dyn_cache_k_r = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.CACHE_SIZE),
                MutAnyOrigin,
            ](gpu.dyn_caches_buf.unsafe_ptr() + k * BATCH * DYN_CS)
            var dyn_grads_v = gpu.dynamics.grads_view()
            Network[
                Self.Config.DynModel, Self.Config.OptType
            ].backward_gpu[BATCH](
                ctx,
                grad_dyn_out_step_t_in,
                grad_dyn_in_step_t_in,
                gpu.dynamics.params_view(),
                gpu.dynamics.model_state_view(),
                dyn_cache_k_r,
                dyn_grads_v,
                gpu.workspace_buf,
            )

            # grad_hidden[k] += grad_dyn_in_step[:, :LATENT]
            var grad_dyn_in_step_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
            ](gpu.grad_dyn_in_step_buf.unsafe_ptr())
            var grad_hidden_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
            ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
            ctx.enqueue_function[accum_dyn_grad_in, accum_dyn_grad_in](
                grad_dyn_in_step_flat,
                grad_hidden_k_flat,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # 6.4 rep backward at k=0 (only — the target-branch rep forwards
        # at k=1..K were stop-grad / no-cache, so no gradients flow
        # through them).
        var grad_hidden_0_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.grad_hidden_buf.unsafe_ptr())
        var grad_rep_in_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.grad_rep_in_buf.unsafe_ptr())
        var rep_grads_v = gpu.representation.grads_view()
        Network[
            Self.Config.RepModel, Self.Config.OptType
        ].backward_gpu[BATCH](
            ctx,
            grad_hidden_0_in,
            grad_rep_in_t,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            rep_cache_t_w,
            rep_grads_v,
            gpu.workspace_buf,
        )

        # ── 7. Optimizer step on every network. ─────────────────────────
        # Per-network gradient clipping (paper: max_grad_norm=5.0). Same
        # threshold applied independently per network — same pattern as
        # `offpolicy_agent`. Reuses `gpu.grad_clip_ps` sequentially.
        if self.max_grad_norm > 0.0:
            comptime _CLIP_TPB: Int = TPB

            comptime _R_PS = Self.Config.RepModel.PARAM_SIZE
            comptime _R_BLK = (_R_PS + _CLIP_TPB - 1) // _CLIP_TPB
            comptime _r_norm_k = gradient_norm_kernel[
                dtype, _R_PS, _R_BLK, _CLIP_TPB
            ]
            comptime _r_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, _R_PS, _R_BLK, _CLIP_TPB
            ]
            var _r_ps_t = LayoutTensor[
                dtype, Layout.row_major(_R_BLK), MutAnyOrigin
            ](gpu.grad_clip_ps.unsafe_ptr())
            var _r_g = gpu.representation.grads_view()
            ctx.enqueue_function[_r_norm_k, _r_norm_k](
                _r_ps_t, _r_g, grid_dim=(_R_BLK,), block_dim=(_CLIP_TPB,)
            )
            ctx.enqueue_function[_r_clip_k, _r_clip_k](
                _r_g,
                _r_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(_R_BLK,),
                block_dim=(_CLIP_TPB,),
            )

            comptime _D_PS = Self.Config.DynModel.PARAM_SIZE
            comptime _D_BLK = (_D_PS + _CLIP_TPB - 1) // _CLIP_TPB
            comptime _d_norm_k = gradient_norm_kernel[
                dtype, _D_PS, _D_BLK, _CLIP_TPB
            ]
            comptime _d_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, _D_PS, _D_BLK, _CLIP_TPB
            ]
            var _d_ps_t = LayoutTensor[
                dtype, Layout.row_major(_D_BLK), MutAnyOrigin
            ](gpu.grad_clip_ps.unsafe_ptr())
            var _d_g = gpu.dynamics.grads_view()
            ctx.enqueue_function[_d_norm_k, _d_norm_k](
                _d_ps_t, _d_g, grid_dim=(_D_BLK,), block_dim=(_CLIP_TPB,)
            )
            ctx.enqueue_function[_d_clip_k, _d_clip_k](
                _d_g,
                _d_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(_D_BLK,),
                block_dim=(_CLIP_TPB,),
            )

            comptime _P_PS = Self.Config.PredModel.PARAM_SIZE
            comptime _P_BLK = (_P_PS + _CLIP_TPB - 1) // _CLIP_TPB
            comptime _p_norm_k = gradient_norm_kernel[
                dtype, _P_PS, _P_BLK, _CLIP_TPB
            ]
            comptime _p_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, _P_PS, _P_BLK, _CLIP_TPB
            ]
            var _p_ps_t = LayoutTensor[
                dtype, Layout.row_major(_P_BLK), MutAnyOrigin
            ](gpu.grad_clip_ps.unsafe_ptr())
            var _p_g = gpu.prediction.grads_view()
            ctx.enqueue_function[_p_norm_k, _p_norm_k](
                _p_ps_t, _p_g, grid_dim=(_P_BLK,), block_dim=(_CLIP_TPB,)
            )
            ctx.enqueue_function[_p_clip_k, _p_clip_k](
                _p_g,
                _p_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(_P_BLK,),
                block_dim=(_CLIP_TPB,),
            )

            comptime _PJ_PS = Self.Config.ProjectorModel.PARAM_SIZE
            comptime _PJ_BLK = (_PJ_PS + _CLIP_TPB - 1) // _CLIP_TPB
            comptime _pj_norm_k = gradient_norm_kernel[
                dtype, _PJ_PS, _PJ_BLK, _CLIP_TPB
            ]
            comptime _pj_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, _PJ_PS, _PJ_BLK, _CLIP_TPB
            ]
            var _pj_ps_t = LayoutTensor[
                dtype, Layout.row_major(_PJ_BLK), MutAnyOrigin
            ](gpu.grad_clip_ps.unsafe_ptr())
            var _pj_g = gpu.projector.grads_view()
            ctx.enqueue_function[_pj_norm_k, _pj_norm_k](
                _pj_ps_t, _pj_g, grid_dim=(_PJ_BLK,), block_dim=(_CLIP_TPB,)
            )
            ctx.enqueue_function[_pj_clip_k, _pj_clip_k](
                _pj_g,
                _pj_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(_PJ_BLK,),
                block_dim=(_CLIP_TPB,),
            )

            comptime _PR_PS = Self.Config.PredictorModel.PARAM_SIZE
            comptime _PR_BLK = (_PR_PS + _CLIP_TPB - 1) // _CLIP_TPB
            comptime _pr_norm_k = gradient_norm_kernel[
                dtype, _PR_PS, _PR_BLK, _CLIP_TPB
            ]
            comptime _pr_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, _PR_PS, _PR_BLK, _CLIP_TPB
            ]
            var _pr_ps_t = LayoutTensor[
                dtype, Layout.row_major(_PR_BLK), MutAnyOrigin
            ](gpu.grad_clip_ps.unsafe_ptr())
            var _pr_g = gpu.predictor.grads_view()
            ctx.enqueue_function[_pr_norm_k, _pr_norm_k](
                _pr_ps_t, _pr_g, grid_dim=(_PR_BLK,), block_dim=(_CLIP_TPB,)
            )
            ctx.enqueue_function[_pr_clip_k, _pr_clip_k](
                _pr_g,
                _pr_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(_PR_BLK,),
                block_dim=(_CLIP_TPB,),
            )

        gpu.representation.optimizer_step(ctx)
        gpu.dynamics.optimizer_step(ctx)
        gpu.prediction.optimizer_step(ctx)
        gpu.projector.optimizer_step(ctx)
        gpu.predictor.optimizer_step(ctx)

        comptime if Self.Config.use_reward_prefix:
            # LSTM Adam step (the cell isn't in a GPUNetworkState so we
            # call OptType.step_gpu directly with the device buffers).
            var lstm_params_v_o = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_params_buf.unsafe_ptr())
            var lstm_grads_v_o = LayoutTensor[
                dtype,
                Layout.row_major(_LSTMHead.PARAM_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_grads_buf.unsafe_ptr())
            var lstm_opt_state_v_o = LayoutTensor[
                dtype,
                Layout.row_major(
                    _LSTMHead.PARAM_SIZE,
                    Self.Config.OptType.STATE_PER_PARAM,
                ),
                MutAnyOrigin,
            ](gpu.lstm_opt_state_buf.unsafe_ptr())
            var lstm_opt_global_v_o = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.OptType.GLOBAL_STATE_SIZE),
                MutAnyOrigin,
            ](gpu.lstm_opt_global_buf.unsafe_ptr())
            gpu.lstm_step_num += 1
            Self.Config.OptType.step_gpu[_LSTMHead.PARAM_SIZE](
                ctx,
                lstm_params_v_o,
                lstm_grads_v_o,
                lstm_opt_state_v_o,
                lstm_opt_global_v_o,
                gpu.lstm_step_num,
            )

            # MLP head Adam step.
            gpu.reward_prefix_mlp_gpu.optimizer_step(ctx)

        self.train_step_count += 1

        # ── 8. Priority refresh — kernel writes priorities_out_buf[b] =
        #       per_sample_v_loss_k0[b] + 1e-3, downloaded below.
        comptime prio_kernel = ezv2_priority_from_v_loss_kernel[
            BATCH, dtype
        ]
        var priorities_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.priorities_out_buf.unsafe_ptr())
        ctx.enqueue_function[prio_kernel, prio_kernel](
            per_sample_v_loss_k0_t,
            priorities_out_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 9. Download losses + per-sample priorities ───────────────────
        ctx.enqueue_copy(gpu.L_R_host, gpu.L_R_buf)
        ctx.enqueue_copy(gpu.L_P_host, gpu.L_P_buf)
        ctx.enqueue_copy(gpu.L_V_host, gpu.L_V_buf)
        ctx.enqueue_copy(gpu.L_G_host, gpu.L_G_buf)
        ctx.enqueue_copy(gpu.priorities_out_host, gpu.priorities_out_buf)
        ctx.synchronize()

        var L_R_sum = Float64(gpu.L_R_host[0])
        var L_P_sum = Float64(gpu.L_P_host[0])
        var L_V_sum = Float64(gpu.L_V_host[0])
        var L_G_sum = Float64(gpu.L_G_host[0])

        var L_R = L_R_sum / n_R if n_R > 0.0 else 0.0
        var L_P = L_P_sum / n_P if n_P > 0.0 else 0.0
        var L_V = L_V_sum / n_V if n_V > 0.0 else 0.0
        var L_G = L_G_sum / n_G if n_G > 0.0 else 0.0

        # ── 10. Update CPU-side priorities array at the matching slot ───
        for b in range(BATCH):
            var new_p = Float64(gpu.priorities_out_host[b])
            self.state.priorities[batch_start_idx[b]] = Scalar[dtype](
                new_p
            )
            if new_p > self.max_priority:
                self.max_priority = new_p

        batch_start_idx.free()

        var L_total = (
            Self.Config.lambda_reward * L_R
            + Self.Config.lambda_policy * L_P
            + Self.Config.lambda_value * L_V
            + Self.Config.lambda_consistency * L_G
        )
        return (L_total, L_R, L_P, L_V, L_G)

    # ══════════════════════════════════════════════════════════════════════
    # Forward-only loss assembly (step 5b)
    # ══════════════════════════════════════════════════════════════════════

    def compute_loss_components(
        mut self
    ) -> Tuple[Float64, Float64, Float64, Float64]:
        """Sample a batch, run the full K-step forward (rep + dyn × K +
        pred at every k + SimSiam projector/predictor), and return the
        four mean loss components

            (L_R, L_P, L_V, L_G)

        all computed forward-only (no gradients written). Caller can
        combine via paper Eq. 3:

            L = λ_R·L_R + λ_P·L_P + λ_V·L_V + λ_G·L_G

        Returns four zeros if the replay buffer doesn't yet hold a full
        (K+N+1)-step window or no valid sequence start exists.

        This is the building block on top of which step 5c will add the
        backward pass + optimizer step.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime N_TD = Self.Config.td_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime LATENT = Self.Config.latent_dim
        comptime PROJ = Self.Config.proj_dim
        comptime BINS = Self.Config.num_bins
        comptime DYN_IN = LATENT + ACT
        comptime DYN_OUT = LATENT + BINS
        comptime PRED_OUT = ACT + BINS
        comptime CAP = 50000  # matches EZV2DiscreteCPUState's default

        if not self.state.is_ready():
            return (Float64(0.0), Float64(0.0), Float64(0.0), Float64(0.0))

        # ── 1. Inline priority-weighted batch sampling ───────────────────
        # We can't use buffer.sample_sequences directly — it doesn't return
        # the buffer indices, but we need them to look up MCTS targets in
        # `self.state.mcts_policies` / `mcts_values`. Re-implement
        # sampling inline; both buffer and target arrays use the same
        # indexing. Sampling is proportional to per-transition priority
        # (paper App. A "Priority Precalculation").
        var batch_obs = alloc[Scalar[dtype]](BATCH * (K + 1) * OBS)
        var batch_actions = alloc[Scalar[dtype]](BATCH * K * ACT)
        var batch_rewards = alloc[Scalar[dtype]](BATCH * K)
        var batch_mcts_pol = alloc[Scalar[dtype]](BATCH * (K + 1) * ACT)
        var batch_mcts_val = alloc[Scalar[dtype]](BATCH * (K + 1))
        # Per-sample data age (in train-steps) used for the mixed value
        # target's SVE→TD blend.
        var batch_age = alloc[Scalar[DType.int32]](BATCH * (K + 1))
        memset(batch_obs, 0, BATCH * (K + 1) * OBS)
        memset(batch_actions, 0, BATCH * K * ACT)
        memset(batch_rewards, 0, BATCH * K)
        memset(batch_mcts_pol, 0, BATCH * (K + 1) * ACT)
        memset(batch_mcts_val, 0, BATCH * (K + 1))
        memset(batch_age, 0, BATCH * (K + 1))

        var buf_size = self.state.buffer.size
        var buf_ptr = self.state.buffer.ptr
        var current_train_step = self.train_step_count
        var oldest = (buf_ptr - buf_size + CAP) % CAP

        # Build cumulative-priority array over valid window starts.
        var n_cands_alloc = buf_size - K if buf_size > K else 1
        var cum_prio = alloc[Float64](n_cands_alloc)
        var cand_starts = alloc[Int](n_cands_alloc)
        var n_valid = 0
        var total_prio = Float64(0.0)
        if buf_size > K:
            for offset in range(buf_size - K):
                var idx = (oldest + offset) % CAP
                var valid = True
                for k in range(K):
                    var iidx = (idx + k) % CAP
                    if Float64(self.state.buffer.dones[iidx]) > 0.5:
                        valid = False
                        break
                if not valid:
                    continue
                var p = Float64(self.state.priorities[idx])
                if p < 1e-8:
                    p = 1e-8
                total_prio += p
                cum_prio[n_valid] = total_prio
                cand_starts[n_valid] = idx
                n_valid += 1

        if n_valid < BATCH or total_prio < 1e-8:
            batch_obs.free()
            batch_actions.free()
            batch_rewards.free()
            batch_mcts_pol.free()
            batch_mcts_val.free()
            batch_age.free()
            cum_prio.free()
            cand_starts.free()
            return (Float64(0.0), Float64(0.0), Float64(0.0), Float64(0.0))

        for sampled in range(BATCH):
            var u = random_float64() * total_prio
            var picked = 0
            for i in range(n_valid):
                if cum_prio[i] >= u:
                    picked = i
                    break
            var start = cand_starts[picked]

            for k in range(K + 1):
                var idx = (start + k) % CAP
                for d in range(OBS):
                    batch_obs[
                        (sampled * (K + 1) + k) * OBS + d
                    ] = self.state.buffer.obs[idx * OBS + d]
                for a in range(ACT):
                    batch_mcts_pol[
                        (sampled * (K + 1) + k) * ACT + a
                    ] = self.state.mcts_policies[idx * ACT + a]
                batch_mcts_val[
                    sampled * (K + 1) + k
                ] = self.state.mcts_values[idx]
                var age = current_train_step - Int(
                    self.state.step_at_write[idx]
                )
                if age < 0:
                    age = 0
                batch_age[sampled * (K + 1) + k] = Scalar[DType.int32](age)
            for k in range(K):
                var idx = (start + k) % CAP
                for a in range(ACT):
                    batch_actions[
                        (sampled * K + k) * ACT + a
                    ] = self.state.buffer.actions[idx * ACT + a]
                batch_rewards[
                    sampled * K + k
                ] = self.state.buffer.rewards[(start + k) % CAP]

        cum_prio.free()
        cand_starts.free()

        # ── 2. K-step forward through rep + dyn + pred ───────────────────
        # Time-major scratch [(K+1) * BATCH * LATENT] / [(K+1) * BATCH * PRED_OUT]
        var hidden = alloc[Scalar[dtype]]((K + 1) * BATCH * LATENT)
        var pred_out = alloc[Scalar[dtype]]((K + 1) * BATCH * PRED_OUT)
        var dyn_out = alloc[Scalar[dtype]](K * BATCH * DYN_OUT)
        memset(hidden, 0, (K + 1) * BATCH * LATENT)
        memset(pred_out, 0, (K + 1) * BATCH * PRED_OUT)
        memset(dyn_out, 0, K * BATCH * DYN_OUT)

        # Build BATCH × OBS rep input from batch_obs[:, k=0, :]
        var rep_input = alloc[Scalar[dtype]](BATCH * OBS)
        for b in range(BATCH):
            for d in range(OBS):
                rep_input[b * OBS + d] = batch_obs[
                    (b * (K + 1) + 0) * OBS + d
                ]

        # Network.forward expects LayoutTensor params/state. Build views
        # from the raw `.params` / `.model_state` UnsafePointer fields —
        # bypasses the params_view() alias-resolution bug.
        var rep_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.representation.params)
        var rep_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.RepModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.representation.model_state)
        var dyn_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.DynModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.params)
        var dyn_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.DynModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.dynamics.model_state)
        var pred_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.params)
        var pred_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.prediction.model_state)
        var proj_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ProjectorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.projector.params)
        var proj_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ProjectorModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.projector.model_state)
        var predr_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredictorModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.params)
        var predr_state_buf = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.PredictorModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.predictor.model_state)

        # rep(o[0]) → hidden[0]
        var rep_input_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
            MutAnyOrigin,
        ](rep_input)
        var hidden0_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
            MutAnyOrigin,
        ](hidden)
        Network[Self.Config.RepModel, Self.Config.OptType].forward[BATCH](
            rep_input_t, hidden0_t, rep_params, rep_state_buf
        )

        # K dynamics steps. Each step: build [hidden[k] || one_hot a[k]],
        # call dyn → dyn_out[k] = [hidden[k+1] || reward_logits[k]],
        # copy hidden[k+1] into the time-major slot.
        var dyn_input = alloc[Scalar[dtype]](BATCH * DYN_IN)
        for k in range(K):
            for b in range(BATCH):
                for d in range(LATENT):
                    dyn_input[b * DYN_IN + d] = hidden[
                        k * BATCH * LATENT + b * LATENT + d
                    ]
                for a in range(ACT):
                    dyn_input[b * DYN_IN + LATENT + a] = batch_actions[
                        (b * K + k) * ACT + a
                    ]
            var dyn_input_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.IN_DIM),
                MutAnyOrigin,
            ](dyn_input)
            var dyn_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.DynModel.OUT_DIM),
                MutAnyOrigin,
            ](dyn_out + k * BATCH * DYN_OUT)
            Network[
                Self.Config.DynModel, Self.Config.OptType
            ].forward[BATCH](
                dyn_input_t, dyn_out_t, dyn_params, dyn_state_buf
            )
            # Copy hidden prefix into hidden[k+1].
            for b in range(BATCH):
                for d in range(LATENT):
                    hidden[
                        (k + 1) * BATCH * LATENT + b * LATENT + d
                    ] = dyn_out[k * BATCH * DYN_OUT + b * DYN_OUT + d]
        dyn_input.free()

        # Pred at every k = 0..K
        for k in range(K + 1):
            var pred_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](hidden + k * BATCH * LATENT)
            var pred_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](pred_out + k * BATCH * PRED_OUT)
            Network[
                Self.Config.PredModel, Self.Config.OptType
            ].forward[BATCH](
                pred_in_t, pred_out_t, pred_params, pred_state_buf
            )

        # ── 3. SimSiam consistency: projector + predictor on dyn branch
        #       and projector ∘ rep on obs branch (target, "detached"). ──
        var proj_dyn_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        var pred_dyn_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        var proj_obs_buf = alloc[Scalar[dtype]](K * BATCH * PROJ)
        memset(proj_dyn_buf, 0, K * BATCH * PROJ)
        memset(pred_dyn_buf, 0, K * BATCH * PROJ)
        memset(proj_obs_buf, 0, K * BATCH * PROJ)

        var rep_obs_step = alloc[Scalar[dtype]](BATCH * LATENT)
        var obs_input_step = alloc[Scalar[dtype]](BATCH * OBS)

        for k_offset in range(K):
            var k = k_offset + 1  # consistency loss applies at unroll step k=1..K

            # (a) projector(z_dyn[k]) → proj_dyn[k_offset]
            var proj_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](hidden + k * BATCH * LATENT)
            var proj_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](proj_dyn_buf + k_offset * BATCH * PROJ)
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward[BATCH](
                proj_in_t, proj_out_t, proj_params, proj_state_buf
            )

            # (b) predictor(proj_dyn) → pred_dyn[k_offset]
            var pred_in2_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.IN_DIM),
                MutAnyOrigin,
            ](proj_dyn_buf + k_offset * BATCH * PROJ)
            var pred_out2_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredictorModel.OUT_DIM),
                MutAnyOrigin,
            ](pred_dyn_buf + k_offset * BATCH * PROJ)
            Network[
                Self.Config.PredictorModel, Self.Config.OptType
            ].forward[BATCH](
                pred_in2_t, pred_out2_t, predr_params, predr_state_buf
            )

            # (c) Target branch: rep(o[k]) → projector → proj_obs[k_offset]
            for b in range(BATCH):
                for d in range(OBS):
                    obs_input_step[b * OBS + d] = batch_obs[
                        (b * (K + 1) + k) * OBS + d
                    ]
            var obs_step_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                MutAnyOrigin,
            ](obs_input_step)
            var rep_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                MutAnyOrigin,
            ](rep_obs_step)
            Network[
                Self.Config.RepModel, Self.Config.OptType
            ].forward[BATCH](
                obs_step_t, rep_obs_t, rep_params, rep_state_buf
            )

            var rep_obs_for_proj_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.IN_DIM),
                MutAnyOrigin,
            ](rep_obs_step)
            var proj_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.ProjectorModel.OUT_DIM),
                MutAnyOrigin,
            ](proj_obs_buf + k_offset * BATCH * PROJ)
            Network[
                Self.Config.ProjectorModel, Self.Config.OptType
            ].forward[BATCH](
                rep_obs_for_proj_t, proj_obs_t, proj_params, proj_state_buf
            )

        rep_obs_step.free()
        obs_input_step.free()

        # ── 4. Loss assembly ─────────────────────────────────────────────
        # L_R: reward CE — targets are scalar-transformed rewards two-hot'd
        #      onto BINS support [v_min, v_max].
        var L_R = Float64(0.0)
        var two_hot_target = alloc[Float64](BINS)
        var logits_dbl = alloc[Float64](BINS)
        for k in range(K):
            for b in range(BATCH):
                var rew = Float64(batch_rewards[b * K + k])
                encode_categorical[BINS](
                    scalar_transform(rew),
                    self.v_min,
                    self.v_max,
                    two_hot_target,
                )
                # Reward logits live at dyn_out[k, b, LATENT:LATENT+BINS]
                var off = k * BATCH * DYN_OUT + b * DYN_OUT + LATENT
                for i in range(BINS):
                    logits_dbl[i] = Float64(dyn_out[off + i])
                L_R += cross_entropy_with_softmax[BINS](
                    logits_dbl, two_hot_target
                )

        # L_P: policy CE — target is the stored MCTS visit-derived policy.
        var L_P = Float64(0.0)
        var pol_logits_dbl = alloc[Float64](ACT)
        var pol_target_dbl = alloc[Float64](ACT)
        for k in range(K + 1):
            for b in range(BATCH):
                var off = k * BATCH * PRED_OUT + b * PRED_OUT
                for i in range(ACT):
                    pol_logits_dbl[i] = Float64(pred_out[off + i])
                    pol_target_dbl[i] = Float64(
                        batch_mcts_pol[(b * (K + 1) + k) * ACT + i]
                    )
                L_P += cross_entropy_with_softmax[ACT](
                    pol_logits_dbl, pol_target_dbl
                )
        pol_logits_dbl.free()
        pol_target_dbl.free()

        # L_V: value CE under the configured value-target mode. Same
        # SEARCH/SARSA/MIXED branching as `train_step` so the diagnostic
        # is a faithful snapshot of the training-time loss.
        var boot_v_diag = alloc[Scalar[dtype]](BATCH * (K + 1))
        memset(boot_v_diag, 0, BATCH * (K + 1))
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
            var tgt_rep_input = alloc[Scalar[dtype]](BATCH * OBS)
            var tgt_z = alloc[Scalar[dtype]](BATCH * LATENT)
            var tgt_pred_out = alloc[Scalar[dtype]](BATCH * PRED_OUT)
            var tgt_logits_dbl = alloc[Float64](BINS)

            var tgt_rep_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.params)
            var tgt_rep_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.RepModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.representation_target.model_state)
            var tgt_pred_params = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.PARAM_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.params)
            var tgt_pred_state = LayoutTensor[
                dtype,
                Layout.row_major(Self.Config.PredModel.STATE_SIZE),
                MutAnyOrigin,
            ](self.state.prediction_target.model_state)

            for k in range(K + 1):
                for b in range(BATCH):
                    for d in range(OBS):
                        tgt_rep_input[b * OBS + d] = batch_obs[
                            (b * (K + 1) + k) * OBS + d
                        ]
                var tgt_rep_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_rep_input)
                var tgt_z_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                Network[
                    Self.Config.RepModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_rep_in_t,
                    tgt_z_t,
                    tgt_rep_params,
                    tgt_rep_state,
                )
                var tgt_pred_in_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                    MutAnyOrigin,
                ](tgt_z)
                var tgt_pred_out_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                    MutAnyOrigin,
                ](tgt_pred_out)
                Network[
                    Self.Config.PredModel, Self.Config.OptType
                ].forward[BATCH](
                    tgt_pred_in_t,
                    tgt_pred_out_t,
                    tgt_pred_params,
                    tgt_pred_state,
                )
                for b in range(BATCH):
                    var off = b * PRED_OUT + ACT
                    for i in range(BINS):
                        tgt_logits_dbl[i] = Float64(tgt_pred_out[off + i])
                    var v_raw = decode_categorical[BINS](
                        tgt_logits_dbl, self.v_min, self.v_max
                    )
                    boot_v_diag[b * (K + 1) + k] = Scalar[dtype](
                        inverse_scalar_transform(v_raw)
                    )

            tgt_rep_input.free()
            tgt_z.free()
            tgt_pred_out.free()
            tgt_logits_dbl.free()

        var L_V = Float64(0.0)
        for k in range(K + 1):
            var n_eff = N_TD if N_TD < K - k else K - k
            for b in range(BATCH):
                var sve = Float64(batch_mcts_val[b * (K + 1) + k])
                var td = Float64(0.0)
                comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
                    var disc = Float64(1.0)
                    for j in range(n_eff):
                        td += disc * Float64(
                            batch_rewards[b * K + k + j]
                        )
                        disc *= self.gamma
                    td += disc * Float64(
                        boot_v_diag[b * (K + 1) + k + n_eff]
                    )
                var age = Int(batch_age[b * (K + 1) + k])
                var v_target = Float64(0.0)
                comptime if Self.Config.value_target_mode == VALUE_TARGET_SEARCH:
                    v_target = sve
                elif Self.Config.value_target_mode == VALUE_TARGET_SARSA:
                    v_target = td
                else:  # VALUE_TARGET_MIXED
                    v_target = MixedValueTarget[
                        Self.Config.t_fresh, Self.Config.t_stale
                    ].compute(sve, td, age)
                encode_categorical[BINS](
                    scalar_transform(v_target),
                    self.v_min,
                    self.v_max,
                    two_hot_target,
                )
                var off = k * BATCH * PRED_OUT + b * PRED_OUT + ACT
                for i in range(BINS):
                    logits_dbl[i] = Float64(pred_out[off + i])
                L_V += cross_entropy_with_softmax[BINS](
                    logits_dbl, two_hot_target
                )
        boot_v_diag.free()
        two_hot_target.free()
        logits_dbl.free()

        # L_G: cosine consistency, mean over k=1..K and batch.
        var L_G = Float64(0.0)
        for k_offset in range(K):
            for b in range(BATCH):
                var p_off = (k_offset * BATCH + b) * PROJ
                var t_off = (k_offset * BATCH + b) * PROJ
                var dot = Float64(0.0)
                var na2 = Float64(0.0)
                var nb2 = Float64(0.0)
                for i in range(PROJ):
                    var pv = Float64(pred_dyn_buf[p_off + i])
                    var tv = Float64(proj_obs_buf[t_off + i])
                    dot += pv * tv
                    na2 += pv * pv
                    nb2 += tv * tv
                var na = sqrt(na2 + 1e-12)
                var nb = sqrt(nb2 + 1e-12)
                L_G += -(dot / (na * nb))

        # ── Means ────────────────────────────────────────────────────────
        var n_R = Float64(BATCH * K)
        var n_P = Float64(BATCH * (K + 1))
        var n_V = Float64(BATCH * (K + 1))
        var n_G = Float64(BATCH * K)
        L_R = L_R / n_R if n_R > 0.0 else 0.0
        L_P = L_P / n_P if n_P > 0.0 else 0.0
        L_V = L_V / n_V if n_V > 0.0 else 0.0
        L_G = L_G / n_G if n_G > 0.0 else 0.0

        # ── Free scratch ─────────────────────────────────────────────────
        batch_obs.free()
        batch_actions.free()
        batch_rewards.free()
        batch_mcts_pol.free()
        batch_mcts_val.free()
        batch_age.free()
        hidden.free()
        pred_out.free()
        dyn_out.free()
        rep_input.free()
        proj_dyn_buf.free()
        pred_dyn_buf.free()
        proj_obs_buf.free()

        return (L_R, L_P, L_V, L_G)
