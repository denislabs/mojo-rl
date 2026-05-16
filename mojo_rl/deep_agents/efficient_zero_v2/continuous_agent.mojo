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
    copy_window_dtype,
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
from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_copy_obs_at_step_kernel,
    ezv2_decode_boot_v_kernel,
    ezv2_compute_value_target_kernel,
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
        Self.Config.num_root_candidates
        // 2 if Self.Config.num_root_candidates
        // 2
        >= 1 else 1,
        Self.Config.max_nodes,
        Self.Config.ActSpace.MAX_ACTION,
        Self.Config.ActSpace.MIN_STD,
        Self.Config.ActSpace.STD_MAGNIFICATION,
        Self.Config.ActSpace.N_POLICY_AT_ROOT,
        Self.Config.ActSpace.SOFT_CLAMP,
        Self.Config.ActSpace.INIT_STD,
    ]

    # Hyperparameters
    var gamma: Float64
    var v_min: Float64
    var v_max: Float64
    # Reward-head two-hot support, in TRANSFORMED scalar space (after
    # `scalar_transform`). Reference DMC: raw `[-2, 2]` → transformed
    # `[h(-2), h(2)] ≈ [-0.732, +0.732]`, ~100× finer step than re-using
    # the value support. Decoupled from `v_min/v_max` 2026-05-14.
    var reward_min: Float64
    var reward_max: Float64
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
    # used as the simple-best policy-loss target (paper Eq. 7) — same
    # shape as actions, but semantically distinct (chosen by MCTS visit-
    # count, not the actual env-stepped action when training mode adds
    # noise).
    # `_episode_sampled_actions`: per timestep, the K root-sampled
    # candidate action vectors as a flat [K * ACT_DIM] list.
    # `_episode_improved_policy`: per timestep, the K improved-policy
    # weights (softmax over completed_Q + log_prior + gumbel).
    # Both used by the full-π loss (paper Eq. 6) when ACT_DIM==1.
    var _episode_obs: List[List[List[Scalar[dtype]]]]
    var _episode_actions: List[List[List[Scalar[dtype]]]]
    var _episode_action_targets: List[List[List[Scalar[dtype]]]]
    var _episode_sampled_actions: List[List[List[Scalar[dtype]]]]
    var _episode_improved_policy: List[List[List[Scalar[dtype]]]]
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
        # Reward-head support in transformed space; defaults to reference
        # DMC: `h(±2) ≈ ±0.732`. Pass narrower for envs whose per-step
        # reward never exceeds a tighter range.
        reward_min: Float64 = -0.732_050_807_568_877_3,
        reward_max: Float64 = 0.732_050_807_568_877_3,
        temperature: Float64 = 1.0,
        temperature_decay_steps: Int = 50000,
        max_grad_norm: Float64 = 5.0,
        n_envs: Int = 1,
        init_zero_heads: Bool = True,
    ):
        self.state = EZV2DiscreteCPUState[Self.Config]()
        self.mcts = SampledGumbelMCTS[
            Self.ACT_DIM,
            Self.Config.latent_dim,
            Self.Config.num_bins,
            Self.Config.num_simulations,
            Self.Config.num_root_candidates,
            Self.Config.num_root_candidates
            // 2 if Self.Config.num_root_candidates
            // 2
            >= 1 else 1,
            Self.Config.max_nodes,
            Self.Config.ActSpace.MAX_ACTION,
            Self.Config.ActSpace.MIN_STD,
            Self.Config.ActSpace.STD_MAGNIFICATION,
            Self.Config.ActSpace.N_POLICY_AT_ROOT,
            Self.Config.ActSpace.SOFT_CLAMP,
            Self.Config.ActSpace.INIT_STD,
        ](gamma=gamma, c_scale=1.0)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.reward_min = reward_min
        self.reward_max = reward_max
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
        self._episode_sampled_actions = List[List[List[Scalar[dtype]]]]()
        self._episode_improved_policy = List[List[List[Scalar[dtype]]]]()
        self._episode_rewards = List[List[Float64]]()
        self._episode_values = List[List[Float64]]()
        for _ in range(self.n_envs):
            self._episode_obs.append(List[List[Scalar[dtype]]]())
            self._episode_actions.append(List[List[Scalar[dtype]]]())
            self._episode_action_targets.append(List[List[Scalar[dtype]]]())
            self._episode_sampled_actions.append(List[List[Scalar[dtype]]]())
            self._episode_improved_policy.append(List[List[Scalar[dtype]]]())
            self._episode_rewards.append(List[Float64]())
            self._episode_values.append(List[Float64]())

        # Paper init_zero on output heads (reference `dmc_state.yaml:120`).
        # See `_init_zero_output_heads` docstring for details.
        if init_zero_heads:
            self._init_zero_output_heads()

        # Hard-sync target networks at startup — must run AFTER any head
        # zeroing so the target nets mirror the zeroed online nets.
        self.update_target_networks(tau=1.0)

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.mcts = take.mcts^
        self.gamma = take.gamma
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.reward_min = take.reward_min
        self.reward_max = take.reward_max
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
        self._episode_sampled_actions = take._episode_sampled_actions^
        self._episode_improved_policy = take._episode_improved_policy^
        self._episode_rewards = take._episode_rewards^
        self._episode_values = take._episode_values^

    # ══════════════════════════════════════════════════════════════════════
    # Action selection
    # ══════════════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
    ) -> Tuple[
        List[Scalar[dtype]],
        Float64,
        List[Scalar[dtype]],
        List[Scalar[dtype]],
    ]:
        """Run sampled-Gumbel MCTS, return search products.

        Args:
            obs: Current observation (zero-padded to RepModel.IN_DIM).
            training: If True, sample weighted by visit counts (paper
                temperature=1 default — adds exploration). If False,
                pick the argmax-visit candidate (eval mode).

        Returns 4-tuple:
            - `action`: ACT_DIM-vector — chosen action to play.
            - `root_value`: SVE backup at the root (used as value target).
            - `sampled_actions`: K_ROOT * ACT_DIM flat — the K root
              candidate action vectors. Used as `target_actions` in the
              full-π loss (paper Eq. 6) when ACT_DIM==1.
            - `improved_policy`: K_ROOT — softmax(completed_Q +
              log_prior + gumbel) over the K candidates. Used as
              `target_policy` in the full-π loss.

        Caller must thread `sampled_actions` and `improved_policy`
        through to `store_transition` so the replay buffer can keep
        them for training.
        """
        var result = self.mcts.search(
            obs,
            self.state.representation,
            self.state.dynamics,
            self.state.prediction,
            self.v_min,
            self.v_max,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            deterministic=not training,
        )
        var chosen = result[0]

        # SVE (paper Eq. 13): visit-weighted backup of the rollout-bootstrapped
        # values from the K_ROOT root candidates. Used as the stored value
        # target instead of the search's returned `result[2]` (= prediction
        # net's raw V(s) decode at the root, not refined by simulations).
        # This matches the discrete agent's `select_action` and the
        # `reanalyze` path so all `mcts_values` entries carry the same kind
        # of value (search-refined, not raw network output).
        var sum_value = Float64(0.0)
        var sum_visits = 0
        comptime K_ROOT = Self.Config.num_root_candidates
        var sampled_actions = List[Scalar[dtype]](
            capacity=K_ROOT * Self.ACT_DIM
        )
        var improved_policy = List[Scalar[dtype]](capacity=K_ROOT)
        for _ in range(K_ROOT * Self.ACT_DIM):
            sampled_actions.append(Scalar[dtype](0.0))
        for _ in range(K_ROOT):
            improved_policy.append(Scalar[dtype](0.0))
        if len(self.mcts.nodes) > 0:
            var root = self.mcts.nodes[0]
            for i in range(K_ROOT):
                sum_value += root.total_value[i]
                sum_visits += root.visit_count[i]
            # Copy K candidate action vectors from the root node.
            for i in range(K_ROOT):
                for d in range(Self.ACT_DIM):
                    sampled_actions[i * Self.ACT_DIM + d] = Scalar[dtype](
                        root.actions[i * Self.ACT_DIM + d]
                    )
            # Compute improved-policy distribution over K candidates.
            var probs = self.mcts._improved_policy_at(0)
            for i in range(K_ROOT):
                improved_policy[i] = Scalar[dtype](probs[i])
        var root_value = compute_sve(sum_value, sum_visits)

        var action_list = List[Scalar[dtype]](capacity=Self.ACT_DIM)
        for d in range(Self.ACT_DIM):
            action_list.append(Scalar[dtype](chosen[d]))
        return (action_list^, root_value, sampled_actions^, improved_policy^)

    def inspect_root(self, tag: String = ""):
        """Print MCTS root stats after the most recent `select_action`
        call. For instrumentation only — used to verify whether MCTS
        Q-values differentiate candidates or stay near-uniform.
        Prints per-candidate: action vector, log_prior, visit_count,
        mean_value, and the resulting improved_policy probability.
        """
        comptime K_ROOT = Self.Config.num_root_candidates
        if len(self.mcts.nodes) == 0:
            print("[inspect_root", tag, "] empty tree")
            return
        var root = self.mcts.nodes[0]
        var probs = self.mcts._improved_policy_at(0)
        var total_visits = 0
        for i in range(K_ROOT):
            total_visits += root.visit_count[i]
        print(
            "[inspect_root",
            tag,
            "] total_visits=",
            total_visits,
            " value_estimate=",
            root.value_estimate,
        )
        # Mirror of the GPU `inspect_root_gpu` instrumentation
        # (`gpu_continuous_train.mojo`): print the load-bearing scalars
        # for the improved-policy reconstruction so CPU vs GPU runs are
        # directly comparable. Added 2026-05-16 to confirm whether
        # global-tree min_q/max_q tracking compresses Q-signal on GPU.
        var mn_cpu = self.mcts.min_max.minimum
        var mx_cpu = self.mcts.min_max.maximum
        var v_mix_cpu = self.mcts._v_mix(0)
        var max_visit_cpu = 0
        for i in range(K_ROOT):
            if root.visit_count[i] > max_visit_cpu:
                max_visit_cpu = root.visit_count[i]
        var sigma_scale_cpu = (
            self.mcts.c_visit + Float64(max_visit_cpu)
        ) * self.mcts.c_scale
        print(
            "       min_q=",
            mn_cpu,
            " max_q=",
            mx_cpu,
            " q_range=",
            mx_cpu - mn_cpu,
            " v_self=",
            root.value_estimate,
            " v_mix=",
            v_mix_cpu,
            " sigma_scale=",
            sigma_scale_cpu,
        )
        # Compute pi entropy as a uniform-ness scalar.
        var H = Float64(0.0)
        var max_pi = Float64(0.0)
        for i in range(K_ROOT):
            var p = Float64(probs[i])
            if p > 1e-12:
                H -= p * log(p)
            if p > max_pi:
                max_pi = p
        var H_unif = log(Float64(K_ROOT))
        print(
            "       pi entropy =",
            H,
            "/ log(K)=",
            H_unif,
            "  (ratio=",
            H / H_unif,
            ")  max_pi=",
            max_pi,
        )
        for i in range(K_ROOT):
            var a_str = String("a=[")
            for d in range(Self.ACT_DIM):
                a_str += String(root.actions[i * Self.ACT_DIM + d])
                if d + 1 < Self.ACT_DIM:
                    a_str += String(",")
            a_str += String("]")
            print(
                "       i=",
                i,
                a_str,
                " log_prior=",
                root.log_prior[i],
                " visits=",
                root.visit_count[i],
                " mean_v=",
                root.mean_value(i),
                " pi=",
                probs[i],
            )

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
        self._episode_sampled_actions[env_id].clear()
        self._episode_improved_policy[env_id].clear()
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
        terminated: Bool = False,
    ):
        """Backward-compat 5-arg overload. Fills the K-candidate slots
        with one-hot at the chosen action so the full-π kernel reduces
        exactly to the simple-best NLL on the played action — old
        training scripts keep their pre-full-π behavior.

        `terminated` defaults to False (truncation semantics). Callers
        that distinguish natural termination from time-limit truncation
        should pass it explicitly; see the 7-arg overload below."""
        comptime K_ROOT = Self.Config.num_root_candidates
        var sampled_actions = List[Scalar[dtype]](
            capacity=K_ROOT * Self.ACT_DIM
        )
        var improved_policy = List[Scalar[dtype]](capacity=K_ROOT)
        for _ in range(K_ROOT * Self.ACT_DIM):
            sampled_actions.append(Scalar[dtype](0.0))
        for _ in range(K_ROOT):
            improved_policy.append(Scalar[dtype](0.0))
        for d in range(Self.ACT_DIM):
            sampled_actions[d] = action[d] if d < len(action) else Scalar[
                dtype
            ](0.0)
        improved_policy[0] = Scalar[dtype](1.0)
        self.store_transition(
            obs,
            action,
            reward,
            value,
            sampled_actions^,
            improved_policy^,
            done,
            env_id,
            terminated,
        )

    def store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        value: Float64,
        sampled_actions: List[Scalar[dtype]],
        improved_policy: List[Scalar[dtype]],
        done: Bool,
        env_id: Int = 0,
        terminated: Bool = False,
    ):
        """Append a transition to env_id's episode buffer.

        `sampled_actions` should be the K_ROOT × ACT_DIM flat list of
        root candidate actions from `select_action`'s 3rd return slot.
        `improved_policy` should be the K_ROOT-length improved-policy
        distribution from `select_action`'s 4th return slot. Both feed
        the full-π policy loss (paper Eq. 6) when ACT_DIM==1.

        For continuous EZ-V2 the search-chosen action vector and the
        env-stepped action are the same (both `action`) — the simple-
        best loss path still uses this. If a future variant adds
        independent action noise after the search, the search-chosen
        action target should be plumbed separately.

        `done` is the boundary flag (terminated OR truncated) for replay
        sequence boundaries — the buffer's `dones` field. `terminated`
        is the bootstrap flag (terminated-only, never truncated) for
        N-step TD target gating — the buffer's `terminations` field. If
        you pass `done=True` and `terminated=True` you assert a natural
        terminal state (V_next clamped to 0 in the target); if you pass
        `done=True` and `terminated=False` you assert a time-limit
        truncation (V_next kept in the target, just as if the episode
        continued). For envs that always truncate (Pendulum, HalfCheetah
        with TERMINATE_ON_UNHEALTHY=False) callers must pass
        `terminated=False` at every episode end — otherwise V is biased
        upward by the missing γ^n·V_next contribution on truncation, and
        the policy fits MCTS targets refined under that biased V.

        Flushes the episode to replay at `done`.
        """
        self._episode_obs[env_id].append(obs.copy())
        # Store the env-stepped action and the search-chosen-action
        # target separately so a future split (e.g. exploration noise
        # added on top of the MCTS pick) can keep them distinct.
        self._episode_actions[env_id].append(action.copy())
        self._episode_action_targets[env_id].append(action.copy())
        self._episode_sampled_actions[env_id].append(sampled_actions.copy())
        self._episode_improved_policy[env_id].append(improved_policy.copy())
        self._episode_rewards[env_id].append(reward)
        self._episode_values[env_id].append(value)
        self.total_steps += 1

        if done:
            self._flush_episode(env_id, terminated_at_end=terminated)

    def _flush_episode(
        mut self, env_id: Int = 0, terminated_at_end: Bool = False
    ):
        """Write env_id's accumulated episode to the SequenceReplayBuffer
        plus the parallel MCTS-target arrays."""
        var ep_len = len(self._episode_obs[env_id])

        for t in range(ep_len):
            var obs_arr = InlineArray[Scalar[dtype], Self.Config.obs_dim](
                uninitialized=True
            )
            for i in range(Self.Config.obs_dim):
                if i < len(self._episode_obs[env_id][t]):
                    obs_arr[i] = self._episode_obs[env_id][t][i]
                else:
                    obs_arr[i] = Scalar[dtype](0.0)

            # Continuous: store raw action vector — no one-hot.
            var act_arr = InlineArray[Scalar[dtype], Self.Config.action_dim](
                uninitialized=True
            )
            var ep_act = self._episode_actions[env_id][t].copy()
            for i in range(Self.Config.action_dim):
                if i < len(ep_act):
                    act_arr[i] = ep_act[i]
                else:
                    act_arr[i] = Scalar[dtype](0.0)

            var is_done = t == ep_len - 1
            # Bootstrap mask: only flip to terminated at the FINAL step of
            # a *naturally* terminated episode. Time-limit truncations
            # propagate `terminated_at_end=False` so the N-step TD target
            # keeps γ^n·V_next at the truncation point (paper's Reanalyze
            # bootstrap semantics; without this V is biased upward when
            # V_next < 0 and downward when V_next > 0).
            var is_terminated = is_done and terminated_at_end

            self.state.buffer.add_with_termination(
                obs_arr,
                act_arr,
                Scalar[dtype](self._episode_rewards[env_id][t]),
                is_done,
                is_terminated,
            )

            comptime CAP = 50000
            comptime K_ROOT = Self.Config.num_root_candidates
            var buf_idx = (self.state.buffer.ptr - 1 + CAP) % CAP
            # Continuous: mcts_policies stores the chosen-action vector
            # (paper Eq. 7 simple-best-action target). Still used by the
            # legacy simple-best loss path; full-π uses the K-candidate
            # buffers below.
            var ep_tgt = self._episode_action_targets[env_id][t].copy()
            for d in range(Self.Config.action_dim):
                self.state.mcts_policies[
                    buf_idx * Self.Config.action_dim + d
                ] = ep_tgt[d] if d < len(ep_tgt) else Scalar[dtype](0.0)
            # Full-π targets (paper Eq. 6): K candidate actions + their
            # improved-policy weights.
            var ep_samp = self._episode_sampled_actions[env_id][t].copy()
            var ep_pi = self._episode_improved_policy[env_id][t].copy()
            for j in range(K_ROOT * Self.Config.action_dim):
                self.state.mcts_sampled_actions[
                    buf_idx * K_ROOT * Self.Config.action_dim + j
                ] = ep_samp[j] if j < len(ep_samp) else Scalar[dtype](0.0)
            for i in range(K_ROOT):
                self.state.mcts_improved_policy[buf_idx * K_ROOT + i] = ep_pi[
                    i
                ] if i < len(ep_pi) else Scalar[dtype](0.0)
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[env_id][t]
            )
            self.state.step_at_write[buf_idx] = Scalar[DType.uint32](
                self.train_step_count
            )
            self.state.priorities[buf_idx] = Scalar[dtype](self.max_priority)
            # Phase 1: maintain priority_tree alongside priorities. Marks
            # slot buf_idx as "leading edge" (tree weight 0) and matures
            # slot (buf_idx - K + 1) if its window is now complete and
            # done-free. Replaces the O(buf_size * BATCH) linear scan in
            # `train_step_gpu` with O(BATCH * log _CAP) sum-tree sampling.
            self.state.on_flush_write(buf_idx)

        self.reset_episode(env_id)

    # ══════════════════════════════════════════════════════════════════════
    # Output-head init_zero (paper Eq. 3 + continuous carve-out)
    # ══════════════════════════════════════════════════════════════════════

    def _init_zero_output_heads(mut self):
        """Zero the policy head and dynamics reward head — paper default.

        Reference (`EfficientZeroV2-main/ez/config/exp/dmc_state.yaml:120`)
        sets `init_zero=True` for the prediction and reward heads. With
        W=b=0 the head's pre-activation is exactly 0 → softmax over BINS is
        uniform → expected V/reward = mid-bin in transformed space. Stops
        the multi-thousand-batch overestimation-correction window that
        would otherwise land the policy in a bad local mode.

        Continuous carve-out (`base_model.py:181`):
            `init_zero=False if is_continuous else init_zero`
        for the value head. With weights=0 on **both** pred heads, gradient
        through `Linear` w.r.t. its input is `grad_out @ W^T = 0`, so the
        encoder receives zero gradient from L_V and L_P at training start.
        The only signal feeding the encoder is then SimSiam consistency —
        which collapses to its trivial all-same-direction solution in
        ~250 train steps (cos → +0.999, L_V/L_R pinned at log(2)=0.69, σ
        collapses to MIN_STD). Keeping the value head random-initialized
        lets L_V immediately pull the encoder toward state-discriminative
        latents, defending against the collapse attractor. Found
        2026-05-13 audit (`docs/EZV2_CONTINUOUS_OPEN_ISSUES.md`).

        Network shapes (continuous, from `EZV2ContinuousMLPConfig`):
            PredModel = Sequential[LinearMish, Parallel[PolicyHead, ValueHead]]
                → zero only branch 0 (policy head) of the trailing Parallel.
            DynModel  = Sequential[SplitApply, LinearMish, LinearMish,
                                   Parallel[NextLatent, RewardHead]]
                → zero only branch 1 (reward head) of the trailing Parallel.
                  Zeroing NextLatent would collapse the latent representation.
        """
        # Offsets/sizes pre-computed by the concrete config (bypasses
        # trait-erasure that hides `Sequential.model_types` / `_param_offset`
        # when `PredModel` / `DynModel` are accessed through the
        # `EZV2DiscreteConfig` trait constraint).
        comptime PP_START = Self.Config.pred_policy_head_param_start
        comptime PP_SIZE = Self.Config.pred_policy_head_param_size
        for i in range(PP_START, PP_START + PP_SIZE):
            self.state.prediction.params[i] = Scalar[dtype](0.0)

        comptime DR_START = Self.Config.dyn_reward_head_param_start
        comptime DR_SIZE = Self.Config.dyn_reward_head_param_size
        for i in range(DR_START, DR_START + DR_SIZE):
            self.state.dynamics.params[i] = Scalar[dtype](0.0)

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
    # Reanalyze — refresh stale stored MCTS targets with current target nets
    # ══════════════════════════════════════════════════════════════════════

    def reanalyze(mut self, num_samples: Int = 16) -> Int:
        """Re-run sampled-Gumbel search on `num_samples` random replay-buffer
        positions using the **target** networks. Overwrites the stored
        chosen-action policy target + root value + age stamp at those
        indices so they reflect the current target model rather than the
        (stale) online model state at collection time.

        Mirrors the discrete agent's `reanalyze` — same sampling logic,
        same target-net plumbing, same buffer fields touched. Two
        differences for continuous:

          - The stored policy target is the **chosen action vector** (paper
            Eq. 8 simple-best-action target) rather than a visit
            distribution, so we copy `chosen` from the search return into
            `mcts_policies[idx*ACT_DIM:idx*ACT_DIM+ACT_DIM]`.
          - We use `deterministic=True` so reanalyzed targets are the
            argmax-visit candidate. Soft-pick at acting time keeps
            exploration; refreshed targets should be the cleanest
            best-action estimate under the current target net.

        SVE (search-based value estimation) is computed from the post-
        search root visit_count + total_value arrays, matching the
        discrete agent. This is a sharper bootstrap target than the
        network's V(s) decode that `select_action` stores at acting time.

        Skips work and returns 0 if the buffer isn't ready. Otherwise
        returns the number of indices it actually refreshed. Callers
        should run this at a coarse interval (paper default 400 train
        steps), typically right after a target-network sync.
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
                obs.append(self.state.buffer.obs[idx * Self.Config.obs_dim + d])

            # Run sampled-Gumbel search with target networks.
            # `deterministic=True` → chosen = argmax-visit candidate (a
            # stable best-action target for the simple-best-action
            # policy NLL).
            var result = self.mcts.search(
                obs,
                self.state.representation_target,
                self.state.dynamics_target,
                self.state.prediction_target,
                self.v_min,
                self.v_max,
                reward_min=self.reward_min,
                reward_max=self.reward_max,
                deterministic=True,
            )
            var chosen = result[0]

            # SVE from the freshly-built tree. Same algebra as the
            # discrete agent: Σ_i total_value(s_0, i) / Σ_i visits(s_0, i),
            # iterated over root candidate slots.
            var sum_value = Float64(0.0)
            var sum_visits = 0
            comptime K_ROOT = Self.Config.num_root_candidates
            if len(self.mcts.nodes) > 0:
                var root = self.mcts.nodes[0]
                for i in range(K_ROOT):
                    sum_value += root.total_value[i]
                    sum_visits += root.visit_count[i]
            var sve = compute_sve(sum_value, sum_visits)

            # Overwrite stored targets at this index — both the simple-
            # best chosen-action and the full-π K candidates + weights.
            for d in range(Self.Config.action_dim):
                self.state.mcts_policies[
                    idx * Self.Config.action_dim + d
                ] = Scalar[dtype](chosen[d])
            if len(self.mcts.nodes) > 0:
                var root = self.mcts.nodes[0]
                for i in range(K_ROOT):
                    for d in range(Self.Config.action_dim):
                        self.state.mcts_sampled_actions[
                            idx * K_ROOT * Self.Config.action_dim
                            + i * Self.Config.action_dim
                            + d
                        ] = Scalar[dtype](
                            root.actions[i * Self.Config.action_dim + d]
                        )
                var probs = self.mcts._improved_policy_at(0)
                for i in range(K_ROOT):
                    self.state.mcts_improved_policy[idx * K_ROOT + i] = Scalar[
                        dtype
                    ](probs[i])
            self.state.mcts_values[idx] = Scalar[dtype](sve)
            self.state.step_at_write[idx] = Scalar[DType.uint32](
                self.train_step_count
            )

            n_refreshed += 1

        return n_refreshed

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

        # Sampling: priority-weighted host-side via sum-tree (Phase 1).
        # `priority_tree` mirrors `priorities` with validity gating —
        # `tree.total_sum()` is the sum over slots whose K-step window is
        # complete and done-free, matching the OLD linear-scan candidate
        # pool. Each `tree.sample(u)` returns a slot in O(log _CAP).
        var buf_size = self.state.buffer.size
        var current_train_step = self.train_step_count

        var total_prio = Float64(self.state.priority_tree.total_sum())
        if buf_size <= K or total_prio < 1e-8:
            return (
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
                Float64(0.0),
            )

        comptime K_ROOT_C = Self.Config.num_root_candidates
        var batch_start_idx = alloc[Int](BATCH)
        for sampled in range(BATCH):
            var u = random_float64() * total_prio
            var start = self.state.priority_tree.sample(Scalar[dtype](u))
            batch_start_idx[sampled] = start

            # Phase 2: bulk memcpys (vs OLD per-(sample,k,d) scalar loops).
            # Each `copy_window_dtype[N]` copies (K+1) or K contiguous
            # ELEM-wide rows from the circular CPU buffer into the host
            # upload buffer, splitting at the CAP wrap when needed.
            copy_window_dtype[OBS](
                gpu.batch_obs_host.unsafe_ptr() + sampled * (K + 1) * OBS,
                self.state.buffer.obs.unsafe_ptr(),
                start, K + 1, CAP,
            )
            # mcts_policies: per-slot `[ACT]` (chosen-action vector for
            # continuous; visit distribution for discrete — same layout).
            copy_window_dtype[ACT](
                gpu.batch_mcts_pol_host.unsafe_ptr() + sampled * (K + 1) * ACT,
                self.state.mcts_policies,
                start, K + 1, CAP,
            )
            # Full-π targets (paper Eq. 6): K candidate actions +
            # improved-policy weights per slot.
            copy_window_dtype[K_ROOT_C * ACT](
                gpu.batch_mcts_samp_act_host.unsafe_ptr()
                + sampled * (K + 1) * K_ROOT_C * ACT,
                self.state.mcts_sampled_actions,
                start, K + 1, CAP,
            )
            copy_window_dtype[K_ROOT_C](
                gpu.batch_mcts_imp_pi_host.unsafe_ptr()
                + sampled * (K + 1) * K_ROOT_C,
                self.state.mcts_improved_policy,
                start, K + 1, CAP,
            )
            copy_window_dtype[1](
                gpu.batch_mcts_val_host.unsafe_ptr() + sampled * (K + 1),
                self.state.mcts_values,
                start, K + 1, CAP,
            )
            # Age: needs current_train_step - step_at_write[idx], so it
            # stays scalar (uint32 → int32 with non-negative clamp).
            for k in range(K + 1):
                var idx = (start + k) % CAP
                var age = current_train_step - Int(
                    self.state.step_at_write[idx]
                )
                if age < 0:
                    age = 0
                gpu.batch_age_host[sampled * (K + 1) + k] = Scalar[DType.int32](
                    age
                )
            # K-wide rows: actions + rewards.
            copy_window_dtype[ACT](
                gpu.batch_actions_host.unsafe_ptr() + sampled * K * ACT,
                self.state.buffer.actions.unsafe_ptr(),
                start, K, CAP,
            )
            copy_window_dtype[1](
                gpu.batch_rewards_host.unsafe_ptr() + sampled * K,
                self.state.buffer.rewards.unsafe_ptr(),
                start, K, CAP,
            )

            # Cumulative rewards: sequential dependency, stays scalar.
            var cum = Float64(0.0)
            for k in range(K):
                cum += Float64(gpu.batch_rewards_host[sampled * K + k])
                gpu.cum_rewards_host[sampled * K + k] = Scalar[dtype](cum)

        # Phase 3c (2026-05-13): all uploads happen here, then the
        # value-target compute lives entirely on GPU (boot-V forward +
        # decode kernel). The downstream `ezv2_train_step_gpu_core` is
        # called with `SKIP_UPLOAD=True`.
        ctx.enqueue_copy(gpu.batch_obs_buf, gpu.batch_obs_host)
        ctx.enqueue_copy(gpu.batch_actions_buf, gpu.batch_actions_host)
        ctx.enqueue_copy(gpu.batch_rewards_buf, gpu.batch_rewards_host)
        ctx.enqueue_copy(gpu.batch_mcts_pol_buf, gpu.batch_mcts_pol_host)
        ctx.enqueue_copy(gpu.batch_mcts_val_buf, gpu.batch_mcts_val_host)
        ctx.enqueue_copy(gpu.batch_age_buf, gpu.batch_age_host)
        ctx.enqueue_copy(
            gpu.batch_mcts_samp_act_buf, gpu.batch_mcts_samp_act_host
        )
        ctx.enqueue_copy(
            gpu.batch_mcts_imp_pi_buf, gpu.batch_mcts_imp_pi_host
        )
        comptime if Self.Config.use_reward_prefix:
            ctx.enqueue_copy(gpu.cum_rewards_buf, gpu.cum_rewards_host)

        # Boot-v target-net forward — only for SARSA/MIXED. Phase 3b
        # runs `rep_target` + `pred_target` × (K+1) timesteps on GPU
        # → `boot_v_buf` on device. No download — Phase 3c's kernel
        # reads `boot_v_buf` directly.
        #
        # Value-bin offset inside the pred-net output: `ACT` for discrete
        # (action logits then BINS); `2*ACT_DIM` for continuous (μ_raw ‖
        # σ_raw then BINS). `POLICY_OUT_DIM` abstracts both.
        comptime VALUE_OFF = Self.Config.ActSpace.POLICY_OUT_DIM
        comptime TPB: Int = 256
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime BATCH_K_BLOCKS = (BATCH * (K + 1) + TPB - 1) // TPB
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:

            # LayoutTensor views into the device-resident scratch
            # buffers + target-net params (allocated in Phase 3a).
            var batch_obs_full_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * (K + 1) * OBS),
                MutAnyOrigin,
            ](gpu.batch_obs_buf.unsafe_ptr())
            var tgt_rep_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * OBS),
                MutAnyOrigin,
            ](gpu.tgt_rep_input_buf.unsafe_ptr())
            var tgt_rep_in_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.tgt_rep_input_buf.unsafe_ptr())
            var tgt_z_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.tgt_z_buf.unsafe_ptr())
            var tgt_pred_in_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.tgt_z_buf.unsafe_ptr())
            var tgt_pred_out_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.tgt_pred_out_buf.unsafe_ptr())
            var tgt_pred_out_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * PRED_OUT),
                MutAnyOrigin,
            ](gpu.tgt_pred_out_buf.unsafe_ptr())
            var boot_v_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * (K + 1)),
                MutAnyOrigin,
            ](gpu.boot_v_buf.unsafe_ptr())

            comptime gather_obs = ezv2_copy_obs_at_step_kernel[
                BATCH, K + 1, OBS, dtype
            ]
            comptime decode_boot_v = ezv2_decode_boot_v_kernel[
                BATCH, K + 1, PRED_OUT, BINS, VALUE_OFF, dtype
            ]

            for k in range(K + 1):
                # Gather batch_obs[*, k, *] → tgt_rep_input.
                ctx.enqueue_function[gather_obs](
                    batch_obs_full_t,
                    tgt_rep_in_t,
                    k,
                    grid_dim=BATCH_BLOCKS,
                    block_dim=TPB,
                )
                # rep_target(obs) → tgt_z.
                Network[
                    Self.Config.RepModel, Self.Config.OptType
                ].forward_gpu[BATCH](
                    ctx,
                    tgt_rep_in_2d,
                    tgt_z_2d,
                    gpu.representation_target.params_view(),
                    gpu.representation_target.model_state_view(),
                    gpu.workspace_buf,
                )
                # pred_target(z) → tgt_pred_out.
                Network[
                    Self.Config.PredModel, Self.Config.OptType
                ].forward_gpu[BATCH](
                    ctx,
                    tgt_pred_in_2d,
                    tgt_pred_out_2d,
                    gpu.prediction_target.params_view(),
                    gpu.prediction_target.model_state_view(),
                    gpu.workspace_buf,
                )
                # Decode value bins → boot_v[*, k].
                ctx.enqueue_function[decode_boot_v](
                    tgt_pred_out_flat,
                    boot_v_t,
                    k,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    Scalar[dtype](0.001),
                    grid_dim=BATCH_BLOCKS,
                    block_dim=TPB,
                )

        # No download — `boot_v_buf` stays on device for Phase 3c.

        # Phase 3c: compute V-target on device. Replaces the host loop
        # that used to consume `boot_v_host` + `batch_rewards_host` +
        # `batch_age_host` + `batch_mcts_val_host`. Writes
        # `value_target_full_buf` directly — the downstream
        # `ezv2_train_step_gpu_core` (called with SKIP_UPLOAD=True)
        # consumes it as-is.
        var batch_mcts_val_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.batch_mcts_val_buf.unsafe_ptr())
        var batch_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * K), MutAnyOrigin
        ](gpu.batch_rewards_buf.unsafe_ptr())
        var batch_age_t = LayoutTensor[
            DType.int32, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.batch_age_buf.unsafe_ptr())
        var boot_v_t_3c = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.boot_v_buf.unsafe_ptr())
        var value_target_full_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.value_target_full_buf.unsafe_ptr())

        comptime compute_v_target = ezv2_compute_value_target_kernel[
            BATCH,
            K + 1,
            K,
            N_TD,
            Self.Config.value_target_mode,
            Self.Config.t_fresh,
            dtype,
        ]
        ctx.enqueue_function[compute_v_target](
            batch_mcts_val_t,
            batch_rewards_t,
            batch_age_t,
            boot_v_t_3c,
            value_target_full_t,
            Scalar[dtype](self.gamma),
            # Continuous: no train-step gate (matches existing host path).
            Scalar[DType.int32](1),
            grid_dim=BATCH_K_BLOCKS,
            block_dim=TPB,
        )

        var sums = ezv2_train_step_gpu_core[
            Self.Config, SKIP_UPLOAD=True
        ](
            gpu,
            ctx,
            self.v_min,
            self.v_max,
            self.max_grad_norm,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            rng_seed=UInt64(self.train_step_count),
        )
        var L_R = sums[0]
        var L_P = sums[1]
        var L_V = sums[2]
        var L_G = sums[3]

        self.train_step_count += 1

        for b in range(BATCH):
            var new_p = Float64(gpu.priorities_out_host[b])
            self.state.priorities[batch_start_idx[b]] = Scalar[dtype](new_p)
            if new_p > self.max_priority:
                self.max_priority = new_p
            # Phase 1: mirror onto priority_tree.
            self.state.on_priority_writeback(batch_start_idx[b], new_p)

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
            50000,
            Self.Config.obs_dim,
            Self.Config.action_dim,
            Self.Config.num_root_candidates,
        ],
        ctx: DeviceContext,
        rng_seed: UInt32,
    ) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """GPU-sampling variant of `train_step_gpu`. Supports SEARCH /
        SARSA / MIXED after Phase 3a+3b+3c (2026-05-13).

        Returns `(L_total, L_R, L_P, L_V, L_G)`.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime N_TD = Self.Config.td_steps
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime PRED_OUT = (
            2 * ACT + BINS
            if Self.Config.ActSpace.IS_CONTINUOUS
            else ACT + BINS
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

        # GPU priority sampling + window gather (kernels 1-4).
        comptime _K_ROOT = Self.Config.num_root_candidates
        var oldest = (gpu_replay.ptr - gpu_replay.size + CAP) % CAP
        ezv2_gpu_sample_and_gather[CAP, BATCH, K, OBS, ACT, _K_ROOT](
            ctx,
            gpu_replay.priorities,
            gpu_replay.dones,
            gpu_replay.obs,
            gpu_replay.actions,
            gpu_replay.rewards,
            gpu_replay.mcts_policies,
            gpu_replay.mcts_values,
            gpu_replay.step_at_write,
            gpu_replay.mcts_sampled_actions,
            gpu_replay.mcts_improved_policy,
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
            gpu.batch_mcts_samp_act_buf,
            gpu.batch_mcts_imp_pi_buf,
            oldest=oldest,
            buf_size=gpu_replay.size,
            current_train_step=UInt32(self.train_step_count),
            rng_seed=rng_seed,
        )

        # Phase 3b: GPU target-net forward → `boot_v_buf` (SARSA/MIXED only).
        # Value-bin offset: `POLICY_OUT_DIM` (= ACT for discrete, 2*ACT_DIM
        # for continuous).
        comptime VALUE_OFF = Self.Config.ActSpace.POLICY_OUT_DIM
        comptime TPB: Int = 256
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime BATCH_K_BLOCKS = (BATCH * (K + 1) + TPB - 1) // TPB
        comptime if Self.Config.value_target_mode != VALUE_TARGET_SEARCH:
            var batch_obs_full_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * (K + 1) * OBS),
                MutAnyOrigin,
            ](gpu.batch_obs_buf.unsafe_ptr())
            var tgt_rep_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH * OBS), MutAnyOrigin
            ](gpu.tgt_rep_input_buf.unsafe_ptr())
            var tgt_rep_in_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.tgt_rep_input_buf.unsafe_ptr())
            var tgt_z_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.RepModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.tgt_z_buf.unsafe_ptr())
            var tgt_pred_in_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.IN_DIM),
                MutAnyOrigin,
            ](gpu.tgt_z_buf.unsafe_ptr())
            var tgt_pred_out_2d = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Config.PredModel.OUT_DIM),
                MutAnyOrigin,
            ](gpu.tgt_pred_out_buf.unsafe_ptr())
            var tgt_pred_out_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * PRED_OUT),
                MutAnyOrigin,
            ](gpu.tgt_pred_out_buf.unsafe_ptr())
            var boot_v_t = LayoutTensor[
                dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
            ](gpu.boot_v_buf.unsafe_ptr())

            comptime gather_obs = ezv2_copy_obs_at_step_kernel[
                BATCH, K + 1, OBS, dtype
            ]
            comptime decode_boot_v = ezv2_decode_boot_v_kernel[
                BATCH, K + 1, PRED_OUT, BINS, VALUE_OFF, dtype
            ]

            for k in range(K + 1):
                ctx.enqueue_function[gather_obs](
                    batch_obs_full_t,
                    tgt_rep_in_t,
                    k,
                    grid_dim=BATCH_BLOCKS,
                    block_dim=TPB,
                )
                Network[
                    Self.Config.RepModel, Self.Config.OptType
                ].forward_gpu[BATCH](
                    ctx,
                    tgt_rep_in_2d,
                    tgt_z_2d,
                    gpu.representation_target.params_view(),
                    gpu.representation_target.model_state_view(),
                    gpu.workspace_buf,
                )
                Network[
                    Self.Config.PredModel, Self.Config.OptType
                ].forward_gpu[BATCH](
                    ctx,
                    tgt_pred_in_2d,
                    tgt_pred_out_2d,
                    gpu.prediction_target.params_view(),
                    gpu.prediction_target.model_state_view(),
                    gpu.workspace_buf,
                )
                ctx.enqueue_function[decode_boot_v](
                    tgt_pred_out_flat,
                    boot_v_t,
                    k,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    Scalar[dtype](0.001),
                    grid_dim=BATCH_BLOCKS,
                    block_dim=TPB,
                )

        # Phase 3c: compute_value_target on device. Writes
        # `value_target_full_buf` for all modes (SEARCH: trivial copy).
        var batch_mcts_val_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.batch_mcts_val_buf.unsafe_ptr())
        var batch_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * K), MutAnyOrigin
        ](gpu.batch_rewards_buf.unsafe_ptr())
        var batch_age_t = LayoutTensor[
            DType.int32, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.batch_age_buf.unsafe_ptr())
        var boot_v_t_3c = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.boot_v_buf.unsafe_ptr())
        var value_target_full_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ](gpu.value_target_full_buf.unsafe_ptr())

        comptime compute_v_target = ezv2_compute_value_target_kernel[
            BATCH,
            K + 1,
            K,
            N_TD,
            Self.Config.value_target_mode,
            Self.Config.t_fresh,
            dtype,
        ]
        ctx.enqueue_function[compute_v_target](
            batch_mcts_val_t,
            batch_rewards_t,
            batch_age_t,
            boot_v_t_3c,
            value_target_full_t,
            Scalar[dtype](self.gamma),
            # Continuous: no train-step gate (matches host path).
            Scalar[DType.int32](1),
            grid_dim=BATCH_K_BLOCKS,
            block_dim=TPB,
        )

        # Sections 2-9 — shared core, with section 2 (host upload) elided
        # since the GPU sampler wrote the device buffers directly.
        var sums = ezv2_train_step_gpu_core[Self.Config, SKIP_UPLOAD=True](
            gpu,
            ctx,
            self.v_min,
            self.v_max,
            self.max_grad_norm,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            rng_seed=UInt64(self.train_step_count),
        )
        var L_R = sums[0]
        var L_P = sums[1]
        var L_V = sums[2]
        var L_G = sums[3]

        self.train_step_count += 1

        # Section 10 (host) — download batch_start_idx + writeback
        # priorities. priorities_out_host is already populated by the
        # core's section 9 download.
        ctx.enqueue_copy(gpu.batch_start_idx_host, gpu.batch_start_idx_buf)
        ctx.synchronize()
        for b in range(BATCH):
            var idx = Int(gpu.batch_start_idx_host[b])
            var new_p = Float64(gpu.priorities_out_host[b])
            self.state.priorities[idx] = Scalar[dtype](new_p)
            if new_p > self.max_priority:
                self.max_priority = new_p
            # Phase 1: mirror onto priority_tree.
            self.state.on_priority_writeback(idx, new_p)

        var L_total = (
            Self.Config.lambda_reward * L_R
            + Self.Config.lambda_policy * L_P
            + Self.Config.lambda_value * L_V
            + Self.Config.lambda_consistency * L_G
        )
        return (L_total, L_R, L_P, L_V, L_G)
