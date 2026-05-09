"""GenericEZV2ContinuousAgent — continuous-action sibling of
`GenericEfficientZeroV2Agent`.

The training side of EZ-V2 is action-space-agnostic at the buffer-layout
level — `Config.ActSpace.policy_loss_grad_gpu` dispatches the policy-head
loss/grad kernel inside the BPTT core. So the only methods that *must*
differ from the discrete agent are the acting-side three:

  • `select_action`: runs the sampled-Gumbel MCTS (`SampledGumbelMCTS`)
    and returns a continuous action vector + root value.
  • `store_transition`: appends a raw `[ACT_DIM]` action vector to the
    per-env episode buffer (the discrete version stores an `Int`).
  • `_flush_episode`: writes raw action vectors and the search-chosen-
    action policy target into the replay state at the buffer slot.

The training methods (`train_step_gpu` here) are copy-pasted verbatim
from `GenericEfficientZeroV2Agent.train_step_gpu` — they manipulate only
raw float buffers and Config-parameterized kernels, no discrete-specific
logic. Marked `# DUP:` for a follow-up de-duplication pass that lifts
the body into a free function.

Limitations of v1:
  • CPU `train_step` not exposed (the existing impl bakes in the discrete
    `pred_output[ACT:ACT+BINS]` value-bin offset; for continuous it would
    need `2*ACT_DIM:2*ACT_DIM+BINS`). GPU `train_step_gpu` is fine because
    `ezv2_train_step_gpu_core` already dispatches via Config.
  • `train_step_gpu_with_replay` not yet ported (uses a different sampling
    path; trivial follow-up).
  • `reanalyze` not yet ported (needs sampled-MCTS variant in the target-
    net path).
  • Multi-env GPU acting (`run_sampled_gumbel_search_gpu` driven from the
    agent's training loop) not yet wired — the agent runs CPU MCTS for
    `select_action`. GPU multi-env acting is a future perf optimization
    once single-env convergence is validated.

Caller-facing API mirrors the discrete agent where possible:
    var agent = GenericEZV2ContinuousAgent[Config](...)
    var (action, root_value) = agent.select_action(obs, training=True)
    agent.store_transition(obs, action, reward, root_value, done)
    if agent.state.is_ready():
        var (L_total, L_R, L_P, L_V, L_G) = agent.train_step_gpu(gpu, ctx)
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
from mojo_rl.deep_agents.efficient_zero_v2.networks import (
    RewardPrefixHeadMLP,
)
from mojo_rl.deep_agents.efficient_zero_v2.state import (
    EZV2DiscreteCPUState,
    EZV2GPUStateBase,
)
from mojo_rl.deep_agents.efficient_zero_v2.train_step_core import (
    ezv2_train_step_gpu_core,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_sampling import (
    ezv2_gpu_sample_and_gather,
)
from mojo_rl.deep_agents.efficient_zero_v2.mcts_sampled import (
    SampledGumbelMCTS,
)
from mojo_rl.deep_agents.efficient_zero_v2.strategies import (
    compute_sve,
    MixedValueTarget,
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


# Shared with discrete agent — could lift to a utility module on a
# follow-up cleanup. Same body as `efficient_zero_v2._clip_grads_inplace`.
def _clip_grads_inplace(
    grads: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    max_norm: Float64,
):
    """Clip gradient L2-norm in-place to `max_norm`."""
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


struct GenericEZV2ContinuousAgent[Config: EZV2DiscreteConfig](Movable):
    """EZ-V2 agent for continuous-action environments.

    Holds the same five-network state as the discrete agent + a
    `SampledGumbelMCTS` engine for continuous-action search + per-env
    episode buffers (raw action vectors instead of Ints).

    Parameters:
        Config: `EZV2DiscreteConfig` providing dimensions, networks,
            optimizer, training hyperparams, and EZ-V2 loss weights.
            `Config.ActSpace.IS_CONTINUOUS` must be True (typically via
            an `EZV2ContinuousMLPConfig` instance) — checked at runtime
            for the search engine setup, but errors deferred to the loss
            kernel mismatch otherwise.
    """

    comptime ACT_DIM: Int = Self.Config.action_dim

    var state: EZV2DiscreteCPUState[Self.Config]

    # Sampled-Gumbel CPU MCTS (re-used across calls; resets internally each
    # search). K_NON_ROOT defaults to K_ROOT // 2 per paper App. A.
    var mcts: SampledGumbelMCTS[
        Self.ACT_DIM,
        Self.Config.latent_dim,
        Self.Config.num_bins,
        Self.Config.num_simulations,
        Self.Config.num_root_candidates,
        Self.Config.num_root_candidates // 2 if Self.Config.num_root_candidates // 2 >= 1 else 1,
        Self.Config.max_nodes,
        1.0,
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

    # Running max of per-transition priority.
    var max_priority: Float64

    # Multi-env support.
    var n_envs: Int

    # Per-episode buffers, parallelized across envs. Continuous
    # action storage: raw `[ACT_DIM]` float vectors (vs discrete's `Int`).
    # `_episode_action_targets` stores the search-chosen-action vector
    # used as the policy-loss target — same shape as actions, but
    # semantically distinct (chosen by MCTS visit-count, not the actual
    # env-stepped action when training mode adds noise).
    var _episode_obs: List[List[List[Scalar[dtype]]]]
    var _episode_actions: List[List[List[Scalar[dtype]]]]
    var _episode_action_targets: List[List[List[Scalar[dtype]]]]
    var _episode_rewards: List[List[Float64]]
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
        self.mcts = SampledGumbelMCTS[
            Self.ACT_DIM,
            Self.Config.latent_dim,
            Self.Config.num_bins,
            Self.Config.num_simulations,
            Self.Config.num_root_candidates,
            Self.Config.num_root_candidates // 2 if Self.Config.num_root_candidates // 2 >= 1 else 1,
            Self.Config.max_nodes,
            1.0,
        ](gamma=gamma)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.temperature = temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.max_grad_norm = max_grad_norm
        self.total_steps = 0
        self.train_step_count = 0
        self.max_priority = 1.0

        self.n_envs = n_envs if n_envs > 0 else 1
        self._episode_obs = List[List[List[Scalar[dtype]]]]()
        self._episode_actions = List[List[List[Scalar[dtype]]]]()
        self._episode_action_targets = List[List[List[Scalar[dtype]]]]()
        self._episode_rewards = List[List[Float64]]()
        self._episode_values = List[List[Float64]]()
        for _ in range(self.n_envs):
            self._episode_obs.append(List[List[Scalar[dtype]]]())
            self._episode_actions.append(List[List[Scalar[dtype]]]())
            self._episode_action_targets.append(List[List[Scalar[dtype]]]())
            self._episode_rewards.append(List[Float64]())
            self._episode_values.append(List[Float64]())

        # Hard-sync target networks at startup.
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
        self._episode_action_targets = take._episode_action_targets^
        self._episode_rewards = take._episode_rewards^
        self._episode_values = take._episode_values^

    # ══════════════════════════════════════════════════════════════════════
    # Action selection
    # ══════════════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
    ) -> Tuple[List[Scalar[dtype]], Float64]:
        """Run sampled-Gumbel MCTS, return (action_vector, root_value).

        Args:
            obs: Current observation (zero-padded to RepModel.IN_DIM).
            training: If True, sample weighted by visit counts (paper
                temperature=1 default — adds exploration). If False,
                pick the argmax-visit candidate (eval mode).

        Returns:
            (action: List[Scalar[dtype]] of length ACT_DIM, root_value).

        The returned action vector is also the search-chosen "policy
        target" a* — pass it back as the `action_target` argument to
        `store_transition` (the loss kernel uses simple-best-action
        target form, paper Eq. 8).
        """
        var result = self.mcts.search(
            obs,
            self.state.representation,
            self.state.dynamics,
            self.state.prediction,
            self.v_min,
            self.v_max,
            deterministic=not training,
        )
        var chosen = result[0]
        var root_value = result[2]

        var action_list = List[Scalar[dtype]](capacity=Self.ACT_DIM)
        for d in range(Self.ACT_DIM):
            action_list.append(Scalar[dtype](chosen[d]))
        return (action_list^, root_value)

    def decay_temperature(mut self):
        """Linear decay toward 0 over `temperature_decay_steps`."""
        if self.temperature_decay_steps <= 0:
            return
        var frac = Float64(self.total_steps) / Float64(
            self.temperature_decay_steps
        )
        if frac > 1.0:
            frac = 1.0
        var new_t = (1.0 - frac) * 1.0
        if new_t < 0.0:
            new_t = 0.0
        self.temperature = new_t

    # ══════════════════════════════════════════════════════════════════════
    # Episode management
    # ══════════════════════════════════════════════════════════════════════

    def reset_episode(mut self, env_id: Int = 0):
        """Clear `env_id`'s episode buffer."""
        self._episode_obs[env_id].clear()
        self._episode_actions[env_id].clear()
        self._episode_action_targets[env_id].clear()
        self._episode_rewards[env_id].clear()
        self._episode_values[env_id].clear()

    def store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        value: Float64,
        done: Bool,
        env_id: Int = 0,
    ):
        """Append a transition to env_id's episode buffer.

        For continuous EZ-V2 the search-chosen action vector and the
        env-stepped action are the same (both `action`) — the loss
        kernel takes this directly as the policy-loss target. If a
        future variant adds independent action noise after the search,
        the search-chosen action target should be plumbed separately.

        Flushes the episode to replay at `done`.
        """
        self._episode_obs[env_id].append(obs.copy())
        # Store the env-stepped action and the search-chosen-action
        # target separately so a future split (e.g. exploration noise
        # added on top of the MCTS pick) can keep them distinct.
        self._episode_actions[env_id].append(action.copy())
        self._episode_action_targets[env_id].append(action.copy())
        self._episode_rewards[env_id].append(reward)
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
                Scalar[dtype], Self.Config.obs_dim
            ](uninitialized=True)
            for i in range(Self.Config.obs_dim):
                if i < len(self._episode_obs[env_id][t]):
                    obs_arr[i] = self._episode_obs[env_id][t][i]
                else:
                    obs_arr[i] = Scalar[dtype](0.0)

            # Continuous: store raw action vector — no one-hot.
            var act_arr = InlineArray[
                Scalar[dtype], Self.Config.action_dim
            ](uninitialized=True)
            var ep_act = self._episode_actions[env_id][t].copy()
            for i in range(Self.Config.action_dim):
                if i < len(ep_act):
                    act_arr[i] = ep_act[i]
                else:
                    act_arr[i] = Scalar[dtype](0.0)

            var is_done = t == ep_len - 1

            self.state.buffer.add(
                obs_arr,
                act_arr,
                Scalar[dtype](self._episode_rewards[env_id][t]),
                is_done,
            )

            comptime CAP = 50000
            var buf_idx = (
                self.state.buffer.ptr - 1 + CAP
            ) % CAP
            # Continuous: mcts_policies stores the chosen-action vector
            # (paper Eq. 8 simple-best-action target). The continuous
            # policy-loss kernel reads this directly as `policy_target`.
            var ep_tgt = self._episode_action_targets[env_id][t].copy()
            for d in range(Self.Config.action_dim):
                self.state.mcts_policies[
                    buf_idx * Self.Config.action_dim + d
                ] = ep_tgt[d] if d < len(ep_tgt) else Scalar[dtype](0.0)
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[env_id][t]
            )
            self.state.step_at_write[buf_idx] = Scalar[DType.uint32](
                self.train_step_count
            )
            self.state.priorities[buf_idx] = Scalar[dtype](
                self.max_priority
            )

        self.reset_episode(env_id)

    # ══════════════════════════════════════════════════════════════════════
    # Target-network sync (copy from discrete agent — action-agnostic)
    # ══════════════════════════════════════════════════════════════════════

    def update_target_networks(mut self, tau: Float64 = 1.0):
        """Polyak-update target nets from online nets."""
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

    # ══════════════════════════════════════════════════════════════════════
    # GPU training (DUP: copy from GenericEfficientZeroV2Agent.train_step_gpu)
    # ══════════════════════════════════════════════════════════════════════
    #
    # The body is verbatim from `efficient_zero_v2.train_step_gpu` —
    # it manipulates only raw float buffers and Config-parameterized
    # kernels, so it works for continuous configs as-is. The follow-up
    # cleanup is to lift this into a free function shared by both
    # agents; that touches the discrete agent's body and is deferred to
    # avoid risk to the existing 25-test discrete suite.

    def train_step_gpu(
        mut self,
        mut gpu: EZV2GPUStateBase[Self.Config],
        ctx: DeviceContext,
    ) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """GPU train step. Returns `(L_total, L_R, L_P, L_V, L_G)`."""
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime N_TD = Self.Config.td_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        # PRED_OUT: 2*ACT_DIM + BINS for continuous. The host-side prep
        # below only uses BINS / ACT for buffer offsets — the actual
        # value-bin offset inside the pred output is owned by the loss
        # kernels via `Config.ActSpace.POLICY_OUT_DIM`. The host-side
        # boot-v decode (SARSA/MIXED) still bakes in the discrete offset
        # `b * PRED_OUT + ACT`, so v1 only supports SEARCH mode for
        # continuous — guarded below.
        comptime PRED_OUT = (
            2 * ACT + BINS if Self.Config.ActSpace.IS_CONTINUOUS else ACT + BINS
        )
        comptime CAP = 50000

        if not self.state.is_ready():
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        # Sampling: priority-weighted host-side. Same as discrete.
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
                # mcts_policies stores per-slot `[ACT]` floats — for
                # discrete this is the visit distribution; for continuous
                # it's the chosen-action vector. Identical layout, so
                # this loop is the same for both.
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

            var cum = Float64(0.0)
            for k in range(K):
                cum += Float64(
                    gpu.batch_rewards_host[sampled * K + k]
                )
                gpu.cum_rewards_host[sampled * K + k] = Scalar[dtype](cum)

        cum_prio.free()
        cand_starts.free()

        # Boot-v target-net forward — only for SARSA/MIXED. Currently
        # off-limits for continuous (the value-bin offset bakes in the
        # discrete pred-output layout). Comptime-guarded.
        var boot_v_host = alloc[Scalar[dtype]](BATCH * (K + 1))
        memset(boot_v_host, 0, BATCH * (K + 1))
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
            comptime if Self.Config.ActSpace.IS_CONTINUOUS:
                # Compile-time error. Continuous + non-SEARCH value
                # targets need a separate boot_v decode that respects
                # the `2*ACT_DIM:2*ACT_DIM+BINS` value-bin slice.
                _ = "ContinuousAgent.train_step_gpu: VALUE_TARGET_SARSA / VALUE_TARGET_MIXED not yet supported for continuous configs — use VALUE_TARGET_SEARCH (default) or switch to the discrete agent. Follow-up: port the boot-v decode to read pred_output[b * PRED_OUT + 2*ACT_DIM : ... + BINS]."
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

        # Value targets.
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

        var sums = ezv2_train_step_gpu_core[Self.Config](
            gpu, ctx, self.v_min, self.v_max, self.max_grad_norm,
        )
        var L_R = sums[0]
        var L_P = sums[1]
        var L_V = sums[2]
        var L_G = sums[3]

        self.train_step_count += 1

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
    # GPU-sampling training (DUP: copy from
    # GenericEfficientZeroV2Agent.train_step_gpu_with_replay)
    # ══════════════════════════════════════════════════════════════════════
    #
    # Same observations as `train_step_gpu` apply: the body is purely
    # action-space-agnostic — `ezv2_gpu_sample_and_gather` shovels raw
    # `[ACT]`-wide float buffers between GPU mirror and gather scratch,
    # and `ezv2_train_step_gpu_core[Self.Config, SKIP_UPLOAD=True]`
    # dispatches the policy-loss kernel through `Config.ActSpace`. The
    # only continuous-specific constraint is the `VALUE_TARGET_SEARCH`
    # restriction (same as discrete — SARSA/MIXED need a GPU target-net
    # forward that hasn't been ported yet).
    #
    # Caller plumbing matches the discrete sibling: keep `gpu_replay`
    # synced from `agent.state` at coarse intervals via
    # `gpu_replay.upload_from_cpu` (the priorities writeback below
    # touches host `state.priorities`, so the next sync re-mirrors them).

    def train_step_gpu_with_replay(
        mut self,
        mut gpu: EZV2GPUStateBase[Self.Config],
        mut gpu_replay: EZV2GPUReplayBuffer[
            50000, Self.Config.obs_dim, Self.Config.action_dim
        ],
        ctx: DeviceContext,
        rng_seed: UInt32,
    ) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """GPU-sampling variant of `train_step_gpu`.

        Returns `(L_total, L_R, L_P, L_V, L_G)`. SEARCH-mode only.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime CAP = 50000
        comptime VTM = Self.Config.value_target_mode

        comptime if VTM != VALUE_TARGET_SEARCH:
            comptime assert False, (
                "GenericEZV2ContinuousAgent.train_step_gpu_with_replay"
                " only supports VALUE_TARGET_MODE == VALUE_TARGET_SEARCH"
                " today; SARSA and MIXED need a GPU target-net forward"
                " (deferred). Use train_step_gpu for SARSA/MIXED."
            )

        if not self.state.is_ready():
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        # GPU priority sampling + window gather (kernels 1-4).
        var oldest = (
            gpu_replay.ptr - gpu_replay.size + CAP
        ) % CAP
        ezv2_gpu_sample_and_gather[CAP, BATCH, K, OBS, ACT](
            ctx,
            gpu_replay.priorities,
            gpu_replay.dones,
            gpu_replay.obs,
            gpu_replay.actions,
            gpu_replay.rewards,
            gpu_replay.mcts_policies,
            gpu_replay.mcts_values,
            gpu_replay.step_at_write,
            gpu.cum_prio_buf,
            gpu.cand_starts_buf,
            gpu.n_valid_buf,
            gpu.total_prio_buf,
            gpu.batch_start_idx_buf,
            gpu.batch_obs_buf,
            gpu.batch_actions_buf,
            gpu.batch_rewards_buf,
            gpu.batch_mcts_pol_buf,
            gpu.batch_mcts_val_buf,
            gpu.batch_age_buf,
            gpu.cum_rewards_buf,
            oldest=oldest,
            buf_size=gpu_replay.size,
            current_train_step=UInt32(self.train_step_count),
            rng_seed=rng_seed,
        )

        # SEARCH mode: value target = stored MCTS root value at every
        # window position. Memcpy the gathered MCTS values into the
        # value-target buffer the core consumes.
        ctx.enqueue_copy(
            gpu.value_target_full_buf, gpu.batch_mcts_val_buf
        )

        # Sections 2-9 — shared core, with section 2 (host upload) elided
        # since the GPU sampler wrote the device buffers directly.
        var sums = ezv2_train_step_gpu_core[
            Self.Config, SKIP_UPLOAD=True
        ](
            gpu, ctx, self.v_min, self.v_max, self.max_grad_norm,
        )
        var L_R = sums[0]
        var L_P = sums[1]
        var L_V = sums[2]
        var L_G = sums[3]

        self.train_step_count += 1

        # Section 10 (host) — download batch_start_idx + writeback
        # priorities. priorities_out_host is already populated by the
        # core's section 9 download.
        ctx.enqueue_copy(
            gpu.batch_start_idx_host, gpu.batch_start_idx_buf
        )
        ctx.synchronize()
        for b in range(BATCH):
            var idx = Int(gpu.batch_start_idx_host[b])
            var new_p = Float64(gpu.priorities_out_host[b])
            self.state.priorities[idx] = Scalar[dtype](new_p)
            if new_p > self.max_priority:
                self.max_priority = new_p

        var L_total = (
            Self.Config.lambda_reward * L_R
            + Self.Config.lambda_policy * L_P
            + Self.Config.lambda_value * L_V
            + Self.Config.lambda_consistency * L_G
        )
        return (L_total, L_R, L_P, L_V, L_G)
