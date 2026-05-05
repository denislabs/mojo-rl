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
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig
from mojo_rl.deep_agents.efficient_zero_v2.state import EZV2DiscreteCPUState
from mojo_rl.deep_agents.efficient_zero_v2.mcts import GumbelMCTS
from mojo_rl.deep_agents.efficient_zero_v2.strategies import (
    compute_sve,
    MixedValueTarget,
)
from mojo_rl.deep_agents.muzero.utils import (
    scalar_transform,
    encode_categorical,
    cross_entropy_with_softmax,
)


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

    # Per-episode buffers — flushed to replay at done.
    var _episode_obs: List[List[Scalar[dtype]]]
    var _episode_actions: List[Int]
    var _episode_rewards: List[Float64]
    var _episode_policies: List[InlineArray[Float64, Self.ACT]]
    var _episode_values: List[Float64]

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

        self._episode_obs = List[List[Scalar[dtype]]]()
        self._episode_actions = List[Int]()
        self._episode_rewards = List[Float64]()
        self._episode_policies = List[InlineArray[Float64, Self.ACT]]()
        self._episode_values = List[Float64]()

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

    def reset_episode(mut self):
        self._episode_obs.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_policies.clear()
        self._episode_values.clear()

    def store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        policy: InlineArray[Float64, Self.ACT],
        value: Float64,
        done: Bool,
    ):
        """Append a transition to the episode buffer. Flushes the entire
        episode to the replay buffer at `done`."""
        self._episode_obs.append(obs.copy())
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._episode_policies.append(policy)
        self._episode_values.append(value)
        self.total_steps += 1

        if done:
            self._flush_episode()

    def _flush_episode(mut self):
        """Write the accumulated episode to the SequenceReplayBuffer plus
        the parallel MCTS-target arrays."""
        var ep_len = len(self._episode_obs)

        for t in range(ep_len):
            var obs_arr = InlineArray[
                Scalar[DType.float32], Self.Config.obs_dim
            ](uninitialized=True)
            for i in range(Self.Config.obs_dim):
                if i < len(self._episode_obs[t]):
                    obs_arr[i] = Scalar[DType.float32](
                        self._episode_obs[t][i]
                    )
                else:
                    obs_arr[i] = Scalar[DType.float32](0.0)

            # One-hot action for SequenceReplayBuffer's ACT-wide action slot.
            var act_arr = InlineArray[
                Scalar[DType.float32], Self.ACT
            ](uninitialized=True)
            for i in range(Self.ACT):
                act_arr[i] = Scalar[DType.float32](0.0)
            act_arr[self._episode_actions[t]] = Scalar[DType.float32](1.0)

            var is_done = t == ep_len - 1

            self.state.buffer.add(
                obs_arr,
                act_arr,
                Scalar[DType.float32](self._episode_rewards[t]),
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
                ] = Scalar[dtype](self._episode_policies[t][a])
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[t]
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

        self.reset_episode()

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

        # ── L_V + grad: value CE with mixed value target (paper Eq. 16) ─
        # blend SVE (= stored MCTS root value) and the n-step TD return
        # computed from the K-window rewards + bootstrap from the stored
        # MCTS value at position min(N_TD, K-k) ahead.
        var n_V = Float64(BATCH * (K + 1))
        var lv_scale = self.Config.lambda_value / n_V
        for k in range(K + 1):
            var n_eff = N_TD if N_TD < K - k else K - k
            for b in range(BATCH):
                var sve = Float64(batch_mcts_val[b * (K + 1) + k])
                var td = Float64(0.0)
                var disc = Float64(1.0)
                for j in range(n_eff):
                    td += disc * Float64(
                        batch_rewards[b * K + k + j]
                    )
                    disc *= self.gamma
                td += disc * Float64(
                    batch_mcts_val[b * (K + 1) + k + n_eff]
                )
                var age = Int(batch_age[b * (K + 1) + k])
                var v_target = MixedValueTarget[
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
        var n_R = Float64(BATCH * K)
        var lr_scale = self.Config.lambda_reward / n_R
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
                var probs_r = InlineArray[Float64, BINS](uninitialized=True)
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

        # ── 5. Optimizer step on every network. ─────────────────────────
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

        # L_V: value CE with the EZ-V2 mixed value target (paper Eq. 16) —
        #      blend SVE (= stored MCTS root value at this position) with
        #      n-step TD return computed from the K-window rewards +
        #      bootstrap from the stored MCTS value at position
        #      `min(N_TD, K-k)` ahead.
        var L_V = Float64(0.0)
        for k in range(K + 1):
            var n_eff = N_TD if N_TD < K - k else K - k
            for b in range(BATCH):
                var sve = Float64(batch_mcts_val[b * (K + 1) + k])
                # n-step TD return from position k of this sample.
                var td = Float64(0.0)
                var disc = Float64(1.0)
                for j in range(n_eff):
                    td += disc * Float64(
                        batch_rewards[b * K + k + j]
                    )
                    disc *= self.gamma
                td += disc * Float64(
                    batch_mcts_val[b * (K + 1) + k + n_eff]
                )
                # Blend by data age (paper Eq. 16).
                var age = Int(batch_age[b * (K + 1) + k])
                var v_target = MixedValueTarget[
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
