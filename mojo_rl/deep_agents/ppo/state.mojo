"""PPO CPU state containers.

Separates heap-allocated rollout state (actor/critic networks + rollout
buffers) from algorithm logic (hyperparameters + update rules) in
DeepPPOAgent and DeepPPOContinuousAgent.

Mirrors the DDPG/TD3/SAC pattern where DDPGCPUState holds all mutable data
and DeepDDPGAgent holds only hyperparameters and update logic.

Two state structs:
    PPODiscreteState  — for DeepPPOAgent (discrete actions)
    PPOContinuousState — for DeepPPOContinuousAgent (continuous actions)

Both implement the corresponding OnPolicyDiscreteState / OnPolicyContinuousState
traits defined in deep_agents/core/onpolicy_train.mojo.

Usage:
    # Discrete PPO
    var state = agent.make_cpu_state()
    for _ in range(num_updates):
        agent.collect_rollout(state, env)
        agent.compute_advantages(state)
        var loss = agent.update_epochs(state)

    # or shorthand (backward compat)
    var metrics = agent.train(env, num_episodes=500)
"""

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import NetworkState, Network, GPUNetworkState
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.deep_agents.core.onpolicy_train import (
    OnPolicyDiscreteState,
    OnPolicyContinuousState,
)
from mojo_rl.deep_agents.core.gpu_onpolicy_train import GPUOnPolicyState
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer


# =============================================================================
# PPODiscreteState — CPU state container for discrete-action PPO
# =============================================================================


struct PPODiscreteState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
](Movable, OnPolicyDiscreteState):
    """CPU-resident state for discrete-action PPO training.

    Holds all heap-allocated data needed for one PPO training loop:
      - Actor and critic NetworkStates (weights + optimizer state)
      - Rollout buffers for obs, actions, rewards, values, log_probs, dones
      - Advantage and return buffers (filled by compute_advantages)
      - Current observation for between-rollout bootstrapping
      - Shuffled index buffer for minibatch sampling

    Created via agent.make_cpu_state() or PPODiscreteState[...]() directly.

    Parameters:
        ActorModel: Actor network model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        rollout_len: Steps per rollout (compile-time constant).
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime ROLLOUT = Self.rollout_len
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]

    # Networks: actor + critic (online weights + optimizer state)
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers (heap-allocated Lists to avoid stack overflow)
    var buffer_obs: List[Scalar[dtype]]  # [rollout_len * obs_dim]
    var buffer_actions: List[Int]  # [rollout_len]
    var buffer_rewards: List[Scalar[dtype]]  # [rollout_len]
    var buffer_values: List[Scalar[dtype]]  # [rollout_len]
    var buffer_log_probs: List[Scalar[dtype]]  # [rollout_len]
    var buffer_dones: List[Bool]  # [rollout_len]
    var buffer_idx: Int  # current fill position

    # Computed by compute_advantages (filled after collect_rollout)
    var _advantages: List[Scalar[dtype]]  # [rollout_len]
    var _returns: List[Scalar[dtype]]  # [rollout_len]

    # Last observation after rollout ends (used to bootstrap value)
    var _current_obs: List[Scalar[dtype]]  # [obs_dim]
    var _env_initialized: Bool

    # Shuffled index buffer (reused across epochs in update_epochs)
    var _indices: List[Int]  # [rollout_len]

    fn __init__(out self):
        """Allocate networks (Kaiming init + small actor output), rollout buffers, and scratch."""
        # Initialize actor with Xavier (agent __init__ applies small output init after)
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        # NOTE: Do NOT initialize critic here — agent.__init__ does it after
        # shrinking actor output, to match continuous PPO's RNG ordering.
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()

        # Allocate and zero-fill rollout buffers
        self.buffer_obs = List[Scalar[dtype]](capacity=Self.ROLLOUT * Self.OBS)
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))

        self.buffer_actions = List[Int](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_actions.append(0)

        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_rewards.append(Scalar[dtype](0))

        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_values.append(Scalar[dtype](0))

        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_log_probs.append(Scalar[dtype](0))

        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_dones.append(False)

        self.buffer_idx = 0

        # Advantage and return scratch
        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))

        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._returns.append(Scalar[dtype](0))

        # Current obs scratch
        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))

        self._env_initialized = False

        # Index buffer for minibatch shuffling
        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # -------------------------------------------------------------------------
    # OnPolicyDiscreteState trait methods
    # -------------------------------------------------------------------------

    fn store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        """Store one step (obs, action, reward, value, log_prob, done) in the rollout buffer.
        """
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    fn is_full(self) -> Bool:
        """Return True when rollout_len steps have been collected."""
        return self.buffer_idx >= Self.ROLLOUT

    fn clear(mut self) -> None:
        """Reset the rollout buffer write pointer."""
        self.buffer_idx = 0


# =============================================================================
# PPOContinuousState — CPU state container for continuous-action PPO
# =============================================================================


struct PPOContinuousState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    action_dim: Int,
    rollout_len: Int,
](Movable, OnPolicyContinuousState):
    """CPU-resident state for continuous-action PPO training.

    Same layout as PPODiscreteState but with continuous action buffers:
      buffer_actions: List[Scalar[dtype]] of length rollout_len * action_dim

    Parameters:
        ActorModel: Actor network model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        action_dim: Continuous action space dimension.
        rollout_len: Steps per rollout (compile-time constant).
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime ROLLOUT = Self.rollout_len
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]

    # Networks
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers
    var buffer_obs: List[Scalar[dtype]]  # [rollout_len * obs_dim]
    var buffer_actions: List[Scalar[dtype]]  # [rollout_len * action_dim]
    var buffer_rewards: List[Scalar[dtype]]  # [rollout_len]
    var buffer_values: List[Scalar[dtype]]  # [rollout_len]
    var buffer_log_probs: List[Scalar[dtype]]  # [rollout_len]
    var buffer_dones: List[Bool]  # [rollout_len]
    var buffer_idx: Int

    # Advantage and return scratch
    var _advantages: List[Scalar[dtype]]  # [rollout_len]
    var _returns: List[Scalar[dtype]]  # [rollout_len]

    # Current obs for bootstrapping
    var _current_obs: List[Scalar[dtype]]  # [obs_dim]
    var _env_initialized: Bool

    # Shuffled index buffer
    var _indices: List[Int]  # [rollout_len]

    fn __init__(out self):
        """Allocate networks (actor Kaiming init only), rollout buffers, scratch.

        IMPORTANT: Only the actor is initialized here. The critic is left
        uninitialized (raw allocation only) so that the owning agent can
        initialize it AFTER calling init_params_small on the actor head.
        This preserves the correct RNG ordering:
            actor_kaiming → init_params_small → critic_kaiming
        which matches DeepPPOContinuousAgentOld and produces stable initial
        gradient magnitudes. Do NOT call critic.initialize[Kaiming]() here.
        """
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Kaiming[]]()
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()
        # NOTE: critic intentionally NOT initialized here. See docstring above.

        self.buffer_obs = List[Scalar[dtype]](capacity=Self.ROLLOUT * Self.OBS)
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))

        self.buffer_actions = List[Scalar[dtype]](
            capacity=Self.ROLLOUT * Self.ACTIONS
        )
        for _ in range(Self.ROLLOUT * Self.ACTIONS):
            self.buffer_actions.append(Scalar[dtype](0))

        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_rewards.append(Scalar[dtype](0))

        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_values.append(Scalar[dtype](0))

        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_log_probs.append(Scalar[dtype](0))

        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_dones.append(False)

        self.buffer_idx = 0

        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))

        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._returns.append(Scalar[dtype](0))

        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))

        self._env_initialized = False

        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # -------------------------------------------------------------------------
    # OnPolicyContinuousState trait methods
    # -------------------------------------------------------------------------

    fn store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        """Store one step (obs, action, reward, value, log_prob, done) in the rollout buffer.
        """
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        for i in range(Self.ACTIONS):
            self.buffer_actions[idx * Self.ACTIONS + i] = action[i]
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    fn is_full(self) -> Bool:
        """Return True when rollout_len steps have been collected."""
        return self.buffer_idx >= Self.ROLLOUT

    fn clear(mut self) -> None:
        """Reset the rollout buffer write pointer."""
        self.buffer_idx = 0


# =============================================================================
# PPODiscreteGPUState — GPU state container for discrete-action PPO
# =============================================================================


struct PPODiscreteGPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
    n_envs: Int,
    gpu_minibatch: Int,
](GPUOnPolicyState, Movable):
    """GPU-resident state for discrete-action PPO training.

    Holds all DeviceBuffers needed for one GPU PPO training loop:
      - Actor and critic GPUNetworkStates (params + grads + optimizer state)
      - Rollout buffers (obs, actions, log_probs, rewards, values, dones)
      - Advantage and return buffers (computed by compute_advantages_gpu)
      - Pinned host buffers for GAE computation and episode tracking
      - Minibatch scratch buffers for GPU update epochs
      - Training workspace buffers (logits, cache, grad output, KL, etc.)

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        rollout_len: Steps per rollout per environment.
        n_envs: Number of parallel environments (sizes rollout buffers).
        gpu_minibatch: Minibatch size for update epochs.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime ROLLOUT = Self.rollout_len
    comptime N = Self.n_envs
    comptime MB = Self.gpu_minibatch
    comptime ROLLOUT_TOTAL = Self.ROLLOUT * Self.N

    comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
    comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
    comptime ACTOR_GRAD_BLOCKS = (Self.ACTOR_PARAMS + TPB - 1) // TPB
    comptime CRITIC_GRAD_BLOCKS = (Self.CRITIC_PARAMS + TPB - 1) // TPB

    comptime ACTOR_WS_ENV = Self.N * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime ACTOR_WS_MB = Self.MB * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_ENV = Self.N * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_MB = Self.MB * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE

    # GPU networks (params + grads + optimizer state)
    var gpu_actor: GPUNetworkState[Self.ActorModel, Self.ActorOpt]
    var gpu_critic: GPUNetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers (ROLLOUT_LEN * N_ENVS elements)
    var rollout_obs_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL * OBS_DIM]
    var rollout_actions_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_log_probs_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_values_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_rewards_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_dones_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_step: Int

    # Advantage and return buffers (filled by compute_advantages_gpu)
    var advantages_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var returns_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]

    # Pinned host buffers for GAE computation
    var rollout_rewards_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_values_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_dones_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var advantages_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var returns_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var bootstrap_values_host: HostBuffer[dtype]  # [N_ENVS]

    # Minibatch scratch buffers
    var mb_obs_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var mb_actions_buf: DeviceBuffer[dtype]  # [MB]
    var mb_advantages_buf: DeviceBuffer[dtype]  # [MB]
    var mb_returns_buf: DeviceBuffer[dtype]  # [MB]
    var mb_old_log_probs_buf: DeviceBuffer[dtype]  # [MB]
    var mb_old_values_buf: DeviceBuffer[dtype]  # [MB]
    var mb_indices_buf: DeviceBuffer[DType.int32]  # [MB]
    var mb_indices_host: HostBuffer[DType.int32]  # [MB]

    # Training workspace buffers
    var logits_buf: DeviceBuffer[dtype]  # [N_ENVS * NUM_ACTIONS]
    var actor_logits_buf: DeviceBuffer[dtype]  # [MB * NUM_ACTIONS]
    var actor_cache_buf: DeviceBuffer[dtype]  # [MB * ActorModel.CACHE_SIZE]
    var actor_grad_output_buf: DeviceBuffer[dtype]  # [MB * NUM_ACTIONS]
    var actor_grad_input_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var critic_values_buf: DeviceBuffer[dtype]  # [MB]
    var critic_cache_buf: DeviceBuffer[dtype]  # [MB * CriticModel.CACHE_SIZE]
    var critic_grad_output_buf: DeviceBuffer[dtype]  # [MB]
    var critic_grad_input_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var kl_divergences_buf: DeviceBuffer[dtype]  # [MB]
    var kl_divergences_host: HostBuffer[dtype]  # [MB]
    var mb_advantages_host: HostBuffer[dtype]  # [MB]
    var actor_grad_partial_sums_buf: DeviceBuffer[dtype]  # [ACTOR_GRAD_BLOCKS]
    var critic_grad_partial_sums_buf: DeviceBuffer[
        dtype
    ]  # [CRITIC_GRAD_BLOCKS]
    var actor_scale_buf: DeviceBuffer[dtype]  # [1]
    var critic_scale_buf: DeviceBuffer[dtype]  # [1]
    var actor_env_workspace_buf: DeviceBuffer[
        dtype
    ]  # [N_ENVS * WORKSPACE_PER_SAMPLE]
    var actor_mb_workspace_buf: DeviceBuffer[
        dtype
    ]  # [MB * WORKSPACE_PER_SAMPLE]
    var critic_env_workspace_buf: DeviceBuffer[
        dtype
    ]  # [N_ENVS * WORKSPACE_PER_SAMPLE]
    var critic_mb_workspace_buf: DeviceBuffer[
        dtype
    ]  # [MB * WORKSPACE_PER_SAMPLE]

    # Env-step scratch buffers (values and log_probs for select_actions_with_meta_gpu)
    var values_env_buf: DeviceBuffer[dtype]  # [N_ENVS]
    var log_probs_env_buf: DeviceBuffer[dtype]  # [N_ENVS]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU device and pinned host buffers."""
        self.gpu_actor = GPUNetworkState[Self.ActorModel, Self.ActorOpt](ctx)
        self.gpu_critic = GPUNetworkState[Self.CriticModel, Self.CriticOpt](ctx)

        self.rollout_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL * Self.OBS
        )
        self.rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_values_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_dones_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_step = 0

        self.advantages_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.returns_buf = ctx.enqueue_create_buffer[dtype](Self.ROLLOUT_TOTAL)

        self.rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.advantages_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.returns_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.N
        )

        self.mb_obs_buf = ctx.enqueue_create_buffer[dtype](Self.MB * Self.OBS)
        self.mb_actions_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_advantages_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_returns_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_old_values_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](Self.MB)
        self.mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.MB
        )

        self.logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.N * Self.ACTIONS
        )
        self.actor_logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ACTIONS
        )
        self.actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ActorModel.CACHE_SIZE
        )
        self.actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ACTIONS
        )
        self.actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.OBS
        )
        self.critic_values_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.CriticModel.CACHE_SIZE
        )
        self.critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.OBS
        )
        self.kl_divergences_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MB
        )
        self.mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)

        self.actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.ACTOR_GRAD_BLOCKS
        )
        self.critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.CRITIC_GRAD_BLOCKS
        )
        self.actor_scale_buf = ctx.enqueue_create_buffer[dtype](1)
        self.critic_scale_buf = ctx.enqueue_create_buffer[dtype](1)

        comptime actor_ws_size = Self.ACTOR_WS_ENV if Self.ACTOR_WS_ENV > 0 else 1
        comptime actor_mb_ws_size = Self.ACTOR_WS_MB if Self.ACTOR_WS_MB > 0 else 1
        comptime critic_ws_size = Self.CRITIC_WS_ENV if Self.CRITIC_WS_ENV > 0 else 1
        comptime critic_mb_ws_size = Self.CRITIC_WS_MB if Self.CRITIC_WS_MB > 0 else 1

        self.actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_ws_size
        )
        self.actor_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_mb_ws_size
        )
        self.critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_ws_size
        )
        self.critic_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_mb_ws_size
        )

        self.values_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)
        self.log_probs_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)

    # -------------------------------------------------------------------------
    # GPUOnPolicyState trait methods
    # -------------------------------------------------------------------------

    fn gpu_store_pre_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        log_probs_buf: DeviceBuffer[dtype],
        values_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store pre-step data (obs, actions, log_probs, values) into rollout buffers.
        """
        from mojo_rl.deep_agents.ppo.kernels import _store_pre_step_kernel

        var t_offset = self.rollout_step * N_ENVS
        var r_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](self.rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
        var r_actions = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_actions_buf.unsafe_ptr() + t_offset)
        var r_log_probs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_log_probs_buf.unsafe_ptr() + t_offset)
        var r_values = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_values_buf.unsafe_ptr() + t_offset)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](values_buf.unsafe_ptr())

        comptime store_wrapper = _store_pre_step_kernel[dtype, N_ENVS, Self.OBS]
        comptime blocks = (N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[store_wrapper, store_wrapper](
            r_obs,
            r_actions,
            r_log_probs,
            r_values,
            obs_t,
            actions_t,
            log_probs_t,
            values_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    fn gpu_store_post_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        rewards_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store post-step data (rewards, dones) into rollout buffers, advance pointer.
        """
        from mojo_rl.deep_agents.ppo.kernels import _store_post_step_kernel

        var t_offset = self.rollout_step * N_ENVS
        var r_rewards = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_rewards_buf.unsafe_ptr() + t_offset)
        var r_dones = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_dones_buf.unsafe_ptr() + t_offset)

        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime store_wrapper = _store_post_step_kernel[dtype, N_ENVS]
        comptime blocks = (N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[store_wrapper, store_wrapper](
            r_rewards,
            r_dones,
            rewards_t,
            dones_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )
        self.rollout_step += 1

    fn gpu_rollout_is_full(self) -> Bool:
        """Return True when rollout_len steps have been stored."""
        return self.rollout_step >= Self.ROLLOUT

    fn gpu_rollout_reset(mut self) -> None:
        """Reset rollout write pointer to 0 for the next update cycle."""
        self.rollout_step = 0


# =============================================================================
# PPOContinuousGPUState — GPU state container for continuous-action PPO
# =============================================================================


struct PPOContinuousGPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    action_dim: Int,
    rollout_len: Int,
    n_envs: Int,
    gpu_minibatch: Int,
](GPUOnPolicyState, Movable):
    """GPU-resident state for continuous-action PPO training.

    Same layout as PPODiscreteGPUState but rollout_actions_buf is sized
    ROLLOUT_TOTAL * ACTION_DIM (one float per action dimension).

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        action_dim: Continuous action space dimension.
        rollout_len: Steps per rollout per environment.
        n_envs: Number of parallel environments.
        gpu_minibatch: Minibatch size for update epochs.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime ROLLOUT = Self.rollout_len
    comptime N = Self.n_envs
    comptime MB = Self.gpu_minibatch
    comptime ROLLOUT_TOTAL = Self.ROLLOUT * Self.N

    comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
    comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
    comptime ACTOR_GRAD_BLOCKS = (Self.ACTOR_PARAMS + TPB - 1) // TPB
    comptime CRITIC_GRAD_BLOCKS = (Self.CRITIC_PARAMS + TPB - 1) // TPB

    comptime ACTOR_WS_ENV = Self.N * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime ACTOR_WS_MB = Self.MB * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_ENV = Self.N * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_MB = Self.MB * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE

    # GPU networks
    var gpu_actor: GPUNetworkState[Self.ActorModel, Self.ActorOpt]
    var gpu_critic: GPUNetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers
    var rollout_obs_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL * OBS_DIM]
    var rollout_actions_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL * ACTION_DIM]
    var rollout_log_probs_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_values_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_rewards_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_dones_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_step: Int

    # Advantage and return buffers
    var advantages_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]
    var returns_buf: DeviceBuffer[dtype]  # [ROLLOUT_TOTAL]

    # Pinned host buffers for GAE
    var rollout_rewards_host: HostBuffer[dtype]
    var rollout_values_host: HostBuffer[dtype]
    var rollout_dones_host: HostBuffer[dtype]
    var advantages_host: HostBuffer[dtype]
    var returns_host: HostBuffer[dtype]
    var bootstrap_values_host: HostBuffer[dtype]  # [N_ENVS]

    # Minibatch scratch
    var mb_obs_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var mb_actions_buf: DeviceBuffer[dtype]  # [MB * ACTION_DIM]
    var mb_advantages_buf: DeviceBuffer[dtype]  # [MB]
    var mb_returns_buf: DeviceBuffer[dtype]  # [MB]
    var mb_old_log_probs_buf: DeviceBuffer[dtype]  # [MB]
    var mb_old_values_buf: DeviceBuffer[dtype]  # [MB]
    var mb_indices_buf: DeviceBuffer[DType.int32]  # [MB]
    var mb_indices_host: HostBuffer[DType.int32]  # [MB]

    # Training workspace
    var actor_means_buf: DeviceBuffer[dtype]  # [N_ENVS * ACTION_DIM]
    var actor_logstds_buf: DeviceBuffer[dtype]  # [N_ENVS * ACTION_DIM]
    var actor_logits_buf: DeviceBuffer[dtype]  # [MB * ACTION_DIM * 2]
    var actor_cache_buf: DeviceBuffer[dtype]  # [MB * ActorModel.CACHE_SIZE]
    var actor_grad_output_buf: DeviceBuffer[dtype]  # [MB * ACTION_DIM * 2]
    var actor_grad_input_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var critic_values_buf: DeviceBuffer[dtype]  # [MB]
    var critic_cache_buf: DeviceBuffer[dtype]  # [MB * CriticModel.CACHE_SIZE]
    var critic_grad_output_buf: DeviceBuffer[dtype]  # [MB]
    var critic_grad_input_buf: DeviceBuffer[dtype]  # [MB * OBS_DIM]
    var kl_divergences_buf: DeviceBuffer[dtype]  # [MB]
    var kl_divergences_host: HostBuffer[dtype]  # [MB]
    var mb_advantages_host: HostBuffer[dtype]  # [MB]
    var actor_grad_partial_sums_buf: DeviceBuffer[dtype]  # [ACTOR_GRAD_BLOCKS]
    var critic_grad_partial_sums_buf: DeviceBuffer[
        dtype
    ]  # [CRITIC_GRAD_BLOCKS]
    var actor_scale_buf: DeviceBuffer[dtype]  # [1]
    var critic_scale_buf: DeviceBuffer[dtype]  # [1]
    var actor_env_workspace_buf: DeviceBuffer[dtype]
    var actor_mb_workspace_buf: DeviceBuffer[dtype]
    var critic_env_workspace_buf: DeviceBuffer[dtype]
    var critic_mb_workspace_buf: DeviceBuffer[dtype]

    # Env-step scratch
    var values_env_buf: DeviceBuffer[dtype]  # [N_ENVS]
    var log_probs_env_buf: DeviceBuffer[dtype]  # [N_ENVS]
    var sampled_actions_buf: DeviceBuffer[dtype]  # [N_ENVS * ACTION_DIM]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU device and pinned host buffers."""
        self.gpu_actor = GPUNetworkState[Self.ActorModel, Self.ActorOpt](ctx)
        self.gpu_critic = GPUNetworkState[Self.CriticModel, Self.CriticOpt](ctx)

        self.rollout_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL * Self.OBS
        )
        self.rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL * Self.ACTIONS
        )
        self.rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_values_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_dones_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_step = 0

        self.advantages_buf = ctx.enqueue_create_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.returns_buf = ctx.enqueue_create_buffer[dtype](Self.ROLLOUT_TOTAL)

        self.rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.advantages_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.returns_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.N
        )

        self.mb_obs_buf = ctx.enqueue_create_buffer[dtype](Self.MB * Self.OBS)
        self.mb_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ACTIONS
        )
        self.mb_advantages_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_returns_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_old_values_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](Self.MB)
        self.mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.MB
        )

        self.actor_means_buf = ctx.enqueue_create_buffer[dtype](
            Self.N * Self.ACTIONS
        )
        self.actor_logstds_buf = ctx.enqueue_create_buffer[dtype](
            Self.N * Self.ACTIONS
        )
        self.actor_logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ACTIONS * 2
        )
        self.actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ActorModel.CACHE_SIZE
        )
        self.actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.ACTIONS * 2
        )
        self.actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.OBS
        )
        self.critic_values_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.CriticModel.CACHE_SIZE
        )
        self.critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.MB * Self.OBS
        )
        self.kl_divergences_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MB
        )
        self.mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)

        self.actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.ACTOR_GRAD_BLOCKS
        )
        self.critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.CRITIC_GRAD_BLOCKS
        )
        self.actor_scale_buf = ctx.enqueue_create_buffer[dtype](1)
        self.critic_scale_buf = ctx.enqueue_create_buffer[dtype](1)

        comptime actor_ws_size = Self.ACTOR_WS_ENV if Self.ACTOR_WS_ENV > 0 else 1
        comptime actor_mb_ws_size = Self.ACTOR_WS_MB if Self.ACTOR_WS_MB > 0 else 1
        comptime critic_ws_size = Self.CRITIC_WS_ENV if Self.CRITIC_WS_ENV > 0 else 1
        comptime critic_mb_ws_size = Self.CRITIC_WS_MB if Self.CRITIC_WS_MB > 0 else 1

        self.actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_ws_size
        )
        self.actor_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_mb_ws_size
        )
        self.critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_ws_size
        )
        self.critic_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_mb_ws_size
        )

        self.values_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)
        self.log_probs_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)
        self.sampled_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.N * Self.ACTIONS
        )

    # -------------------------------------------------------------------------
    # GPUOnPolicyState trait methods
    # -------------------------------------------------------------------------

    fn gpu_store_pre_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        log_probs_buf: DeviceBuffer[dtype],
        values_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store pre-step data (obs, actions, log_probs, values) into rollout buffers.
        """
        from mojo_rl.deep_agents.ppo.kernels import (
            _store_continuous_pre_step_kernel,
        )

        var t_offset = self.rollout_step * N_ENVS
        var r_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](self.rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
        var r_actions = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](self.rollout_actions_buf.unsafe_ptr() + t_offset * Self.ACTIONS)
        var r_log_probs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_log_probs_buf.unsafe_ptr() + t_offset)
        var r_values = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_values_buf.unsafe_ptr() + t_offset)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](values_buf.unsafe_ptr())

        comptime store_wrapper = _store_continuous_pre_step_kernel[
            dtype, N_ENVS, Self.OBS, Self.ACTIONS
        ]
        comptime blocks = (N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[store_wrapper, store_wrapper](
            r_obs,
            r_actions,
            r_log_probs,
            r_values,
            obs_t,
            actions_t,
            log_probs_t,
            values_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    fn gpu_store_post_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        rewards_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store post-step data (rewards, dones) into rollout buffers, advance pointer.
        """
        from mojo_rl.deep_agents.ppo.kernels import _store_post_step_kernel

        var t_offset = self.rollout_step * N_ENVS
        var r_rewards = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_rewards_buf.unsafe_ptr() + t_offset)
        var r_dones = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](self.rollout_dones_buf.unsafe_ptr() + t_offset)

        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime store_wrapper = _store_post_step_kernel[dtype, N_ENVS]
        comptime blocks = (N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[store_wrapper, store_wrapper](
            r_rewards,
            r_dones,
            rewards_t,
            dones_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )
        self.rollout_step += 1

    fn gpu_rollout_is_full(self) -> Bool:
        """Return True when rollout_len steps have been stored."""
        return self.rollout_step >= Self.ROLLOUT

    fn gpu_rollout_reset(mut self) -> None:
        """Reset rollout write pointer to 0 for the next update cycle."""
        self.rollout_step = 0
