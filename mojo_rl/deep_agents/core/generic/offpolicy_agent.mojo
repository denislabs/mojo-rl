"""Generic off-policy agent parameterized by OffPolicyConfig.

Follows the ModelDef pattern from physics3d: the agent takes components as
trait-bounded type parameters, derives all dimensions from them, and uses
Self.* consistently to avoid compile-time type unification issues.
"""

from std.random import random_float64
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState, NetworkPair
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise

from mojo_rl.deep_agents.core import (
    OffPolicyState,
    OffPolicyContinuousAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline, concat_obs_action_batch
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv

from .offpolicy_config import OffPolicyConfig
from .exploration import GaussianNoise
from .update_schedule import EveryStep


# =============================================================================
# GenericCPUState — workspace folded into agent's comptime constants
# =============================================================================


struct GenericCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    # Dimensions derived from Model types (NOT passed separately)
    # These must match the agent's Self.OBS, Self.ACTIONS, etc.
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
](Movable, OffPolicyState):
    """CPU state with workspace. Dimensions from Model types for consistency."""

    comptime BUFFER_DTYPE = dtype

    # Networks
    var actor: NetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Single workspace allocation — offsets computed inline
    # Total size depends on actor/critic cache sizes
    var ws: List[Scalar[dtype]]

    fn __init__(out self):
        self.actor = NetworkPair[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Kaiming[]]()
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Compute workspace size
        comptime OBS = Self.ActorModel.IN_DIM
        comptime ACT = Self.ActorModel.OUT_DIM
        comptime BS = Self.batch_size
        comptime CI = Self.CriticModel.IN_DIM
        comptime ACS = Self.ActorModel.CACHE_SIZE
        comptime CCS = Self.CriticModel.CACHE_SIZE
        # Regions: next_act, next_ci, next_q, targets, ci, q_out, q_cache,
        #          q_grad, d_ci, actor_act, actor_cache, new_ci, new_q,
        #          dq, d_new_ci, d_act, d_obs
        comptime WS_SIZE = (
            BS * ACT  # next_act
            + BS * CI  # next_ci
            + BS  # next_q
            + BS  # targets
            + BS * CI  # ci
            + BS  # q_out
            + BS * CCS  # q_cache
            + BS  # q_grad
            + BS * CI  # d_ci
            + BS * ACT  # actor_act
            + BS * ACS  # actor_cache
            + BS * CI  # new_ci
            + BS  # new_q
            + BS  # dq
            + BS * CI  # d_new_ci
            + BS * ACT  # d_act
            + BS * OBS  # d_obs
        )
        self.ws = List[Scalar[dtype]](capacity=WS_SIZE)
        for _ in range(WS_SIZE):
            self.ws.append(Scalar[dtype](0))

    # OffPolicyState trait
    fn store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        for i in range(Self.action_dim):
            act_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(action[i]))
        self.buffer.add(
            obs_arr, act_arr, Scalar[Self.BUFFER_DTYPE](reward), next_arr, done
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# GenericOffPolicyAgent[Config: OffPolicyConfig]
# =============================================================================


struct GenericOffPolicyAgent[
    Config: OffPolicyConfig,
](OffPolicyContinuousAgent & Checkpointable):
    """Generic off-policy agent. Config provides network types + flags.

    All dimensions derived from Config's Model types. The state container
    is parameterized with Config's types directly so Self.* comptime
    expressions are consistent everywhere (no unification issues).
    """

    # Derive ALL dimensions from Config's Model types
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.ActorModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    # Workspace offsets — all derived from Self.* for consistency
    comptime _O_NEXT_ACT: Int = 0
    comptime _O_NEXT_CI: Int = Self._O_NEXT_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEXT_Q: Int = Self._O_NEXT_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_TARGETS: Int = Self._O_NEXT_Q + Self.BATCH * Self.CRITIC_OUT
    comptime _O_CI: Int = Self._O_TARGETS + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_OUT: Int = Self._O_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_Q_CACHE: Int = Self._O_Q_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_GRAD: Int = Self._O_Q_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_D_CI: Int = Self._O_Q_GRAD + Self.BATCH * Self.CRITIC_OUT
    comptime _O_ACTOR_ACT: Int = Self._O_D_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_ACTOR_CACHE: Int = Self._O_ACTOR_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEW_CI: Int = Self._O_ACTOR_CACHE + Self.BATCH * Self.ACTOR_CS
    comptime _O_NEW_Q: Int = Self._O_NEW_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_DQ: Int = Self._O_NEW_Q + Self.BATCH * Self.CRITIC_OUT
    comptime _O_D_NEW_CI: Int = Self._O_DQ + Self.BATCH * Self.CRITIC_OUT
    comptime _O_D_ACT: Int = Self._O_D_NEW_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_D_OBS: Int = Self._O_D_ACT + Self.BATCH * Self.ACTIONS

    # CPU state type — pass Model types directly
    comptime CPUStateType = GenericCPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.ActorModel.OUT_DIM,
        Self.Config.batch_size,
    ]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var explore: GaussianNoise
    var schedule: EveryStep

    # Training state
    var total_steps: Int
    var train_step_count: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.explore = GaussianNoise(noise_std, noise_std_min, noise_decay)
        self.schedule = EveryStep()
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # OffPolicyContinuousAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](
        mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]
    ) -> List[Scalar[d]]:
        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())

        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, act_t, p)

        var raw = List[Scalar[d]](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            raw.append(Scalar[d](Float64(act_arr[i]) * self.action_scale))
        return self.explore.explore[d](raw, self.action_scale)

    fn store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var normalized = List[Scalar[d]](capacity=len(action))
        for i in range(len(action)):
            normalized.append(
                Scalar[d](Float64(action[i]) / self.action_scale)
            )
        cpu_state.store[d](obs, normalized, reward, next_obs, done)
        self.total_steps += 1

    fn do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Phase 1: Sample batch (InlineArrays — required by HeapReplayBuffer)
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act, b_rew, b_next, b_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](b_act.unsafe_ptr())

        # Workspace pointer
        var ws = cpu_state.ws.unsafe_ptr()

        # Phase 2: TD targets — actor_target(next_obs), then critic_target
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_NEXT_ACT)
        var p_at = cpu_state.actor.target.params_view()
        Self.ActorNet.forward[Self.BATCH](next_obs_t, next_act_t, p_at)

        # Concat next_obs + next_act → next_ci (manual, avoids unification)
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_NEXT_CI)
        for row in range(Self.BATCH):
            for c in range(Self.OBS):
                (ws + Self._O_NEXT_CI)[row * Self.CRITIC_IN + c] = b_next.unsafe_ptr()[row * Self.OBS + c]
            for c in range(Self.ACTIONS):
                (ws + Self._O_NEXT_CI)[row * Self.CRITIC_IN + Self.OBS + c] = (ws + Self._O_NEXT_ACT)[row * Self.ACTIONS + c]

        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NEXT_Q)
        var p_ct = cpu_state.critic.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, next_q_t, p_ct)

        # Compute targets
        var tgt_p = ws + Self._O_TARGETS
        var nq_p = ws + Self._O_NEXT_Q
        for b in range(Self.BATCH):
            var q = Float64(nq_p[b])
            if q != q:
                q = 0.0
            var dm = 1.0 - Float64(b_done[b])
            var tgt = Float64(b_rew[b]) + self.gamma * q * dm
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            tgt_p[b] = Scalar[dtype](tgt)

        # Phase 3: Critic update
        # Concat obs + act → ci
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_CI)
        for row in range(Self.BATCH):
            for c in range(Self.OBS):
                (ws + Self._O_CI)[row * Self.CRITIC_IN + c] = b_obs.unsafe_ptr()[row * Self.OBS + c]
            for c in range(Self.ACTIONS):
                (ws + Self._O_CI)[row * Self.CRITIC_IN + Self.OBS + c] = b_act.unsafe_ptr()[row * Self.ACTIONS + c]

        var q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q_OUT)
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q_CACHE)

        var p_c = cpu_state.critic.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q_t, p_c, q_cache_t
        )

        # MSE loss
        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q_GRAD)
        var qo_p = ws + Self._O_Q_OUT
        var qg_p = ws + Self._O_Q_GRAD
        var critic_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = qo_p[b] - tgt_p[b]
            critic_loss += Float64(td_err * td_err)
            qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
        critic_loss /= Float64(Self.BATCH)

        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_D_CI)
        var g_c = cpu_state.critic.grads_view()
        cpu_state.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q_grad_t, d_ci_t, p_c, q_cache_t, g_c
        )
        cpu_state.critic.optimizer_step()

        # Phase 4: Actor update (DPG)
        var aa_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_ACTOR_ACT)
        var ac_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_CS), MutAnyOrigin
        ](ws + Self._O_ACTOR_CACHE)

        var p_a = cpu_state.actor.params_view()
        Self.ActorNet.forward_with_cache[Self.BATCH](
            obs_t, aa_t, p_a, ac_t
        )

        # Concat obs + actor_act → new_ci
        var nci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_NEW_CI)
        for row in range(Self.BATCH):
            for c in range(Self.OBS):
                (ws + Self._O_NEW_CI)[row * Self.CRITIC_IN + c] = b_obs.unsafe_ptr()[row * Self.OBS + c]
            for c in range(Self.ACTIONS):
                (ws + Self._O_NEW_CI)[row * Self.CRITIC_IN + Self.OBS + c] = (ws + Self._O_ACTOR_ACT)[row * Self.ACTIONS + c]

        var nq_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NEW_Q)
        # Reuse q_cache for new forward
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q_CACHE)
        Self.CriticNet.forward_with_cache[Self.BATCH](
            nci_t, nq_t, p_c, nc_t
        )

        # dQ = -1/BATCH
        var dq_p = ws + Self._O_DQ
        for b in range(Self.BATCH):
            dq_p[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](dq_p)

        var dnci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_D_NEW_CI)
        cpu_state.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](dq_t, dnci_t, p_c, nc_t, g_c)

        # Extract action grads from critic input grads
        var da_p = ws + Self._O_D_ACT
        var dnci_p = ws + Self._O_D_NEW_CI
        for b in range(Self.BATCH):
            for i in range(Self.ACTIONS):
                da_p[b * Self.ACTIONS + i] = dnci_p[
                    b * Self.CRITIC_IN + Self.OBS + i
                ]

        var da_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](da_p)
        var do_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](ws + Self._O_D_OBS)

        var g_a = cpu_state.actor.grads_view()
        cpu_state.actor.zero_grads()
        Self.ActorNet.backward[Self.BATCH](da_t, do_t, p_a, ac_t, g_a)
        cpu_state.actor.optimizer_step()

        # Phase 5: Soft update
        cpu_state.actor.soft_update(self.tau)
        cpu_state.critic.soft_update(self.tau)

        self.train_step_count += 1
        return critic_loss

    fn decay_explore(mut self) -> None:
        self.explore.decay()

    fn get_explore_rate(self) -> Float64:
        return self.explore.get_rate()

    fn random_action[d: DType](self) -> List[Scalar[d]]:
        var result = List[Scalar[d]](capacity=Self.ACTIONS)
        for _ in range(Self.ACTIONS):
            result.append(
                Scalar[d]((random_float64() * 2.0 - 1.0) * self.action_scale)
            )
        return result^

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())

        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, act_t, p)

        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(a)
        return result^

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        pass

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

    # Convenience
    fn train[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E, num_episodes: Int = 300) raises -> TrainingMetrics:
        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        return run_offpolicy_continuous_train(
            self, cpu_state, env, num_episodes,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
        )
