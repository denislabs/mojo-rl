"""Shared CPU helper functions for deterministic off-policy agents (DDPG, TD3).

These free functions extract the byte-identical method bodies shared between
DDPG and TD3 — select_action_list, store_list_transition, random_action_list,
and select_greedy_action_list — into a single place.

SAC does NOT use these because:
  - select_action_list: SAC uses stochastic policy (reparameterization trick)
  - select_greedy_action_list: SAC applies tanh(mean) instead of raw actor output

Usage in DDPG/TD3 (inside the struct methods):
    fn select_action_list(mut self, obs) -> List[Float64]:
        return deterministic_select_action[Self.ActorModel, Self.ActorOpt](
            self.actor.online, obs, self.action_scale, self.noise_std
        )

    fn store_list_transition(mut self, obs, action, reward, next_obs, done):
        store_continuous_transition[Self.OBS, Self.ACTIONS, Self.BUFFER_CAPACITY, dtype](
            self.buffer, obs, action, reward, next_obs, done,
            self.action_scale, self.total_steps
        )
"""

from std.random import random_float64
from layout import Layout, LayoutTensor

from nn.constants import dtype
from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import Network, NetworkState
from .utils import obs_to_inline
from deep_agents.core.replay import HeapReplayBuffer
from nn.gpu.random import gaussian_noise


fn deterministic_select_action[
    DTYPE: DType,
    ActorModel: Model,
    ActorOpt: Optimizer,
](
    actor_online: NetworkState[ActorModel, ActorOpt],
    obs: List[Scalar[DTYPE]],
    action_scale: Float64,
    noise_std: Float64,
) -> List[Scalar[DTYPE]]:
    """Select action with Gaussian exploration noise (deterministic policy).

    Shared implementation for DDPG and TD3. Runs a forward pass through the
    online actor, then adds Gaussian noise and clips to [-action_scale, action_scale].

    OBS and ACTIONS are derived from ActorModel.IN_DIM and ActorModel.OUT_DIM.

    Parameters:
        DTYPE: Data type (float32 or float64).
        ActorModel: Actor model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).

    Args:
        actor_online: Online actor NetworkState.
        obs: Observation as List[Float64].
        action_scale: Action scaling factor.
        noise_std: Exploration noise standard deviation (pre-scaled by action_scale).

    Returns:
        Action list of length ActorModel.OUT_DIM, clipped to [-action_scale, action_scale].
    """
    comptime OBS = ActorModel.IN_DIM
    comptime ACTIONS = ActorModel.OUT_DIM
    comptime ActorNet = Network[ActorModel, ActorOpt]

    var obs_arr = obs_to_inline[OBS, DTYPE](obs)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(1, ActorModel.IN_DIM), MutAnyOrigin
    ](obs_arr.unsafe_ptr())
    var act_arr = InlineArray[Scalar[dtype], ActorModel.OUT_DIM](
        uninitialized=True
    )
    var act_t = LayoutTensor[
        dtype, Layout.row_major(1, ActorModel.OUT_DIM), MutAnyOrigin
    ](act_arr.unsafe_ptr())

    var p = actor_online.params_view()
    ActorNet.forward[1](obs_t, act_t, p)

    var result = List[Scalar[DTYPE]](capacity=ACTIONS)
    for i in range(ACTIONS):
        var a = Float64(act_arr[i]) * action_scale
        a += noise_std * action_scale * gaussian_noise()
        if a > action_scale:
            a = action_scale
        elif a < -action_scale:
            a = -action_scale
        result.append(Scalar[DTYPE](a))
    return result^


fn greedy_continuous_action[
    ActorModel: Model,
    ActorOpt: Optimizer,
](
    actor_online: NetworkState[ActorModel, ActorOpt],
    obs: List[Float64],
    action_scale: Float64,
) -> List[Float64]:
    """Select deterministic action without exploration noise (evaluation).

    Shared implementation for DDPG and TD3 greedy/evaluation action selection.
    Pure actor forward pass with clamping — no Gaussian noise added.

    OBS and ACTIONS are derived from ActorModel.IN_DIM and ActorModel.OUT_DIM.

    Parameters:
        ActorModel: Actor model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).

    Args:
        actor_online: Online actor NetworkState.
        obs: Observation as List[Float64].
        action_scale: Action scaling factor.

    Returns:
        Deterministic action list of length ActorModel.OUT_DIM, clipped to [-action_scale, action_scale].
    """
    comptime OBS = ActorModel.IN_DIM
    comptime ACTIONS = ActorModel.OUT_DIM
    comptime ActorNet = Network[ActorModel, ActorOpt]

    var obs_arr = obs_to_inline[OBS, DType.float64](obs)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(1, ActorModel.IN_DIM), MutAnyOrigin
    ](obs_arr.unsafe_ptr())
    var act_arr = InlineArray[Scalar[dtype], ActorModel.OUT_DIM](
        uninitialized=True
    )
    var act_t = LayoutTensor[
        dtype, Layout.row_major(1, ActorModel.OUT_DIM), MutAnyOrigin
    ](act_arr.unsafe_ptr())

    var p = actor_online.params_view()
    ActorNet.forward[1](obs_t, act_t, p)

    var result = List[Float64](capacity=ACTIONS)
    for i in range(ACTIONS):
        var a = Float64(act_arr[i]) * action_scale
        if a > action_scale:
            a = action_scale
        elif a < -action_scale:
            a = -action_scale
        result.append(a)
    return result^


fn store_continuous_transition[
    DTYPE: DType,
    OBS: Int,
    ACTIONS: Int,
    CAPACITY: Int,
](
    mut buffer: HeapReplayBuffer[CAPACITY, OBS, ACTIONS, dtype],
    obs: List[Scalar[DTYPE]],
    action: List[Scalar[DTYPE]],
    reward: Float64,
    next_obs: List[Scalar[DTYPE]],
    done: Bool,
    action_scale: Float64,
    mut total_steps: Int,
) -> None:
    """Store a continuous-action transition in the replay buffer.

    Shared for DDPG, TD3, and SAC. Actions are normalized by dividing by
    action_scale before storage (consistent with actor output range [-1, 1]).
    Buffer dtype is always the module-level `dtype` (float32).

    Parameters:
        DTYPE: Data type (float32 or float64).
        OBS: Observation dimension (compile-time).
        ACTIONS: Action dimension (compile-time).
        CAPACITY: Replay buffer capacity (compile-time).

    Args:
        buffer: Replay buffer to store into (mutated).
        obs: Current observation.
        action: Action taken (in [-action_scale, action_scale]).
        reward: Observed reward.
        next_obs: Next observation.
        done: Episode termination flag.
        action_scale: Action scaling factor (used to normalize stored action).
        total_steps: Step counter (incremented in-place).
    """
    var obs_arr = obs_to_inline[OBS, DTYPE](obs)
    var next_arr = obs_to_inline[OBS, DTYPE](next_obs)

    var act_arr = InlineArray[Scalar[dtype], ACTIONS](uninitialized=True)
    for i in range(ACTIONS):
        act_arr[i] = Scalar[dtype](Float64(action[i]) / action_scale)

    buffer.add(obs_arr, act_arr, Scalar[dtype](reward), next_arr, done)
    total_steps += 1


fn random_continuous_action[
    DTYPE: DType
](action_dim: Int, action_scale: Float64) -> List[Scalar[DTYPE]]:
    """Return a uniformly random action in [-action_scale, action_scale].

    Shared implementation for DDPG, TD3, and SAC random exploration phase
    (before the replay buffer has enough samples to begin learning).

    Args:
        action_dim: Number of action dimensions.
        action_scale: Half-range of the action space.

    Returns:
        Random action list of length action_dim.
    """
    var result = List[Scalar[DTYPE]](capacity=action_dim)
    for _ in range(action_dim):
        result.append(
            Scalar[DTYPE]((random_float64() * 2.0 - 1.0) * action_scale)
        )
    return result^
