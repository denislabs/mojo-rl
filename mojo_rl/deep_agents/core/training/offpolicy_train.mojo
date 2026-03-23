"""Shared off-policy training infrastructure.

Provides two trait hierarchies and shared training-loop functions:

OffPolicyAgent (discrete + continuous, used by DQN family):
    Agents own all state internally; the loop calls agent methods directly.

OffPolicyContinuousAgent (continuous only, used by DDPG / TD3 / SAC):
    Mirrors GPUOffPolicyAgent — the loop holds a CPUStateType buffer container
    and calls agent methods with it explicitly.  This creates symmetry:

        CPU:  run_offpolicy_continuous_train(agent, cpu_state, env, ...)
        GPU:  run_offpolicy_continuous_train_gpu(agent, gpu_state, ctx, env, ...)

OffPolicyState (CPU buffer container trait, parallel to GPUOffPolicyState):
    Holds networks + replay buffer + scratch; exposes store() and is_ready().

Usage — OffPolicyContinuousAgent style (DDPG / TD3 / SAC):
    struct MyAgent[...](OffPolicyContinuousAgent):
        comptime CPUStateType = MyCPUState[...]
        def make_cpu_state(self) -> Self.CPUStateType: ...
        def select_action[dtype](mut self, mut cpu_state, obs) -> ...: ...
        def store_transition[dtype](mut self, mut cpu_state, obs, ...) -> None: ...
        def do_cpu_train_step(mut self, mut cpu_state) -> Float64: ...
        def decay_explore(mut self) -> None: ...
        def get_explore_rate(self) -> Float64: ...
        def random_action[dtype](self) -> List[Scalar[dtype]]: ...
        def select_greedy_action(self, cpu_state, obs) -> List[Float64]: ...

    var agent = MyAgent[...]()
    var cpu_state = agent.make_cpu_state()
    var metrics = run_offpolicy_continuous_train(agent, cpu_state, env, ...)
"""

from std.math import exp
from std.random import random_float64, seed
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from ..checkpoint_trait import Checkpointable


# =============================================================================
# OffPolicyState Trait  (CPU mirror of GPUOffPolicyState)
# =============================================================================


trait OffPolicyState:
    """CPU-side buffer container for continuous off-policy agents.

    Holds all heap-allocated state: network weights (online + target),
    replay buffer, and algorithm-specific scratch buffers.

    Mirrors GPUOffPolicyState: the training loop holds and passes this
    explicitly rather than having the agent own it internally.

    Required methods:
        store[dtype](): Push one transition (with normalized action) into
            the replay buffer.
        is_ready(): True when the buffer has enough samples to train.
    """

    def store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push one transition into the CPU replay buffer.

        Args:
            obs: Current observation.
            action: Action taken, normalized to the actor output range.
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    def is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        ...


# =============================================================================
# OffPolicyContinuousAgent Trait  (CPU mirror of GPUOffPolicyAgent)
# =============================================================================


trait OffPolicyContinuousAgent:
    """Continuous off-policy agent with explicit CPU state management.

    The agent (CPU struct) owns only hyperparameters and algorithm logic.
    All heap-allocated state (networks, replay buffer, scratch) lives in
    CPUStateType, which is created via make_cpu_state() and held by the caller
    (training loop or user code).

    This mirrors GPUOffPolicyAgent, creating a symmetric pair:

        CPU training:  run_offpolicy_continuous_train(agent, cpu_state, env)
        GPU training:  run_offpolicy_continuous_train_gpu(agent, gpu_state, ctx)

    Compile-time constants (must be set on the concrete struct):
        CPUStateType: Concrete OffPolicyState implementation.
    """

    comptime CPUStateType: OffPolicyState
    """Concrete CPU state type holding all networks, buffer, and scratch."""

    def make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType (networks + replay buffer + scratch).

        Called once before training. The returned state is owned by the caller.
        """
        ...

    def select_action[
        dtype: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dtype]],
    ) -> List[Scalar[dtype]]:
        """Select an action given the current observation (with exploration).

        Args:
            cpu_state: CPU state holding online network weights.
            obs: Current observation as List[Scalar[dtype]].

        Returns:
            Action list scaled by action_scale (length = action_dim).
        """
        ...

    def store_transition[
        dtype: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Normalize action and push transition into the replay buffer.

        Handles action normalization (dividing by action_scale) then
        delegates to cpu_state.store().

        Args:
            cpu_state: CPU state holding the replay buffer.
            obs: Current observation.
            action: Action taken (scaled by action_scale).
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    def do_cpu_train_step(
        mut self,
        mut cpu_state: Self.CPUStateType,
    ) -> Float64:
        """Perform one gradient update step using cpu_state buffers.

        Mirror of GPUOffPolicyAgent.do_gpu_train_step(ctx, gpu_state).

        Args:
            cpu_state: CPU state with replay buffer and network weights.

        Returns:
            Loss value (critic loss or similar).
        """
        ...

    def decay_explore(mut self) -> None:
        """Decay exploration rate (noise_std, epsilon, etc.)."""
        ...

    def get_explore_rate(self) -> Float64:
        """Return current exploration rate (for logging)."""
        ...

    def random_action[dtype: DType](self) -> List[Scalar[dtype]]:
        """Return a uniformly random action (used during warmup).

        Returns:
            Random action list of length action_dim, scaled by action_scale.
        """
        ...

    def select_greedy_action(
        self,
        cpu_state: Self.CPUStateType,
        obs: List[Float64],
    ) -> List[Float64]:
        """Select action without exploration (for evaluation).

        DDPG/TD3: deterministic actor forward, no noise.
        SAC: tanh(mean), no reparameterization sampling.

        Args:
            cpu_state: CPU state holding online network weights.
            obs: Current observation as List[Float64].

        Returns:
            Greedy action list scaled by action_scale.
        """
        ...


# =============================================================================
# OffPolicyDiscreteState Trait  (CPU buffer container for DQN family)
# =============================================================================


trait OffPolicyDiscreteState:
    """CPU-side buffer container for discrete off-policy agents (DQN family).

    Holds all heap-allocated state: network weights (online + target) and
    replay buffer.

    Mirrors OffPolicyState: the training loop holds and passes this
    explicitly rather than having the agent own it internally.

    Required methods:
        store[dtype](): Push one transition into the replay buffer.
        is_ready(): True when the buffer has enough samples to train.
    """

    def store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push one transition into the replay buffer.

        Args:
            obs: Current observation.
            action: Discrete action index taken.
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    def is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        ...


# =============================================================================
# OffPolicyDiscreteAgent Trait  (explicit state design for DQN family)
# =============================================================================


trait OffPolicyDiscreteAgent:
    """Discrete off-policy agent with explicit CPU state management.

    The agent struct owns only hyperparameters and algorithm logic.
    All heap-allocated state (networks, replay buffer) lives in CPUStateType,
    which is created via make_cpu_state() and held by the caller.

    This mirrors OffPolicyContinuousAgent for the discrete action setting:

        CPU training:  run_offpolicy_discrete_train(agent, cpu_state, env)

    Compile-time constants (must be set on the concrete struct):
        CPUStateType: Concrete OffPolicyDiscreteState implementation.
    """

    comptime CPUStateType: OffPolicyDiscreteState
    """Concrete CPU state type holding networks and replay buffer."""

    def make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType (networks + replay buffer).

        Called once before training. The returned state is owned by the caller.
        """
        ...

    def select_action[
        dtype: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dtype]],
    ) -> Int:
        """Select an action with exploration (epsilon-greedy).

        Args:
            cpu_state: CPU state holding online network weights.
            obs: Current observation as List[Scalar[dtype]].

        Returns:
            Selected action index.
        """
        ...

    def store_transition[
        dtype: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push transition into the replay buffer via cpu_state.store().

        Args:
            cpu_state: CPU state holding the replay buffer.
            obs: Current observation.
            action: Discrete action index taken.
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    def do_cpu_train_step(
        mut self,
        mut cpu_state: Self.CPUStateType,
    ) -> Float64:
        """Perform one gradient update step using cpu_state buffers.

        Args:
            cpu_state: CPU state with replay buffer and network weights.

        Returns:
            Loss value (critic loss or similar).
        """
        ...

    def decay_explore(mut self) -> None:
        """Decay exploration rate (epsilon, noise_std, etc.)."""
        ...

    def get_explore_rate(self) -> Float64:
        """Return current exploration rate (for logging)."""
        ...

    def random_action(self) -> Int:
        """Return a uniformly random action index (used during warmup).

        Returns:
            Random action index in [0, num_actions).
        """
        ...

    def select_greedy_action(
        self,
        cpu_state: Self.CPUStateType,
        obs: List[Float64],
    ) -> Int:
        """Select action without exploration (for evaluation).

        Args:
            cpu_state: CPU state holding online network weights.
            obs: Current observation as List[Float64].

        Returns:
            Greedy action index.
        """
        ...


# =============================================================================
# OffPolicyAgent Trait  (kept for DQN family — discrete + continuous)
# =============================================================================


trait OffPolicyAgent:
    """Common interface for all off-policy deep RL agents.

    Agents implement this trait so that run_offpolicy_discrete_train and
    run_offpolicy_continuous_train can drive any agent without knowing
    agent-specific internals.

    Action representation:
        Discrete agents: List with one element = action index as Float64.
        Continuous agents: List with action_dim elements = raw action values.
    """

    def select_action_list[
        dtype: DType
    ](mut self, obs: List[Scalar[dtype]]) -> List[Scalar[dtype]]:
        """Select an action given the current observation.

        Applies epsilon-greedy / noise internally.

        Args:
            obs: Current observation as List[Scalar[dtype]].

        Returns:
            Action list (length 1 for discrete, action_dim for continuous).
        """
        ...

    def store_list_transition[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: Current observation.
            action: Action taken (see trait docstring for encoding).
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    def is_ready(self) -> Bool:
        """Return True if the buffer holds enough samples to train."""
        ...

    def do_train_step(mut self) -> Float64:
        """Perform one gradient update step.

        Returns:
            Loss value (may be 0.0 if not ready).
        """
        ...

    def decay_explore(mut self) -> None:
        """Decay exploration rate (epsilon, noise_std, etc.)."""
        ...

    def get_explore_rate(self) -> Float64:
        """Return current exploration rate (for logging)."""
        ...

    def random_action_list[dtype: DType](self) -> List[Scalar[dtype]]:
        """Return a uniformly random action (used during warmup).

        Returns:
            Random action list using same encoding as select_action_list.
        """
        ...

    def select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        """Select action without exploration noise (for evaluation).

        DQN: pure argmax (epsilon=0). DDPG/TD3: actor forward, no Gaussian
        noise. SAC: deterministic mean (no reparameterization sampling).

        Args:
            obs: Current observation as List[Float64].

        Returns:
            Greedy action list (length 1 for discrete, action_dim for continuous).
        """
        ...


# =============================================================================
# Shared Training Loop — Discrete Actions
# =============================================================================


def run_offpolicy_discrete_train[
    E: BoxDiscreteActionEnv,
    A: OffPolicyAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 500,
    warmup_steps: Int = 1000,
    train_every: Int = 4,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """Warmup + episode training loop for discrete-action off-policy agents.

    Shared implementation used by DQN, DQN+PER, and DuelingDQN to eliminate
    the duplicated warmup/train loop.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Off-policy agent (updated in-place).
        env: Discrete-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 500).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_train_step every N steps (default: 4).
        checkpoint_every: Save checkpoint every N episodes (default: 0).
        checkpoint_path: Path to save checkpoint (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()
    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action_list[E.dtype]()
        var action_int = Int(Float64(action[0]))
        var result = env.step_obs(action_int)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_list_transition(warmup_obs, action, reward, next_obs, done)
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action_list(obs)
            var action_int = Int(Float64(action[0]))
            var result = env.step_obs(action_int)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_list_transition(obs, action, reward, next_obs, done)

            if agent.is_ready() and total_steps % train_every == 0:
                _ = agent.do_train_step()

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        # Logger: per-episode reward
        if logger:
            logger[].log_scalar("episode_reward", episode_reward, total_steps)
            logger[].log_scalar(
                "explore_rate", agent.get_explore_rate(), total_steps
            )

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_episode_" + String(episode + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            episode + 1
        ) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_reward", avg_reward, total_steps)

            if verbose:
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# Shared Training Loop — Discrete Actions (OffPolicyDiscreteAgent)
# =============================================================================


def run_offpolicy_discrete_train[
    E: BoxDiscreteActionEnv,
    A: OffPolicyDiscreteAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 500,
    warmup_steps: Int = 1000,
    train_every: Int = 4,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """Warmup + episode training loop for OffPolicyDiscreteAgent (DQN family).

    Symmetric with run_offpolicy_continuous_train (OffPolicyContinuousAgent):
        - cpu_state is passed explicitly (not owned by the agent).
        - The loop calls cpu_state.is_ready() directly.
        - The loop calls agent.do_cpu_train_step(cpu_state).

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OffPolicyDiscreteAgent.

    Args:
        agent: Off-policy agent (hyperparameters + algorithm only).
        cpu_state: CPU state buffer (networks + replay buffer).
        env: Discrete-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 500).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_cpu_train_step every N steps (default: 4).
        checkpoint_every: Save checkpoint every N episodes (default: 0).
        checkpoint_path: Path to save checkpoint (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()
    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action()
        var result = env.step_obs(action)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_transition[E.dtype](
            cpu_state, warmup_obs, action, reward, next_obs, done
        )
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action[E.dtype](cpu_state, obs)
            var result = env.step_obs(action)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_transition[E.dtype](
                cpu_state, obs, action, reward, next_obs, done
            )

            if cpu_state.is_ready() and total_steps % train_every == 0:
                _ = agent.do_cpu_train_step(cpu_state)

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        # Logger: per-episode reward
        if logger:
            logger[].log_scalar("episode_reward", episode_reward, total_steps)
            logger[].log_scalar(
                "explore_rate", agent.get_explore_rate(), total_steps
            )

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_episode_" + String(episode + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            episode + 1
        ) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_reward", avg_reward, total_steps)

            if verbose:
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# Shared Training Loop — Continuous Actions (OffPolicyAgent, legacy DQN path)
# =============================================================================


def run_offpolicy_continuous_train[
    E: BoxContinuousActionEnv,
    A: OffPolicyAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 1000,
    warmup_steps: Int = 1000,
    train_every: Int = 1,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """Warmup + episode training loop for continuous-action off-policy agents.

    Shared implementation used by DDPG, TD3, and SAC.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Off-policy agent (updated in-place).
        env: Continuous-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 1000).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_train_step every N steps (default: 1).
        checkpoint_every: Save checkpoint every N episodes (default: 0).
        checkpoint_path: Path to save checkpoint (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()

    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action_list[E.dtype]()
        var result = env.step_continuous_vec(action)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_list_transition(warmup_obs, action, reward, next_obs, done)
        warmup_count += 1
        if done:
            var reset_raw = env.reset_obs_list()
            warmup_obs = reset_raw^
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action_list(obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_list_transition(obs, action, reward, next_obs, done)

            if agent.is_ready() and total_steps % train_every == 0:
                _ = agent.do_train_step()

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        # Logger: per-episode reward
        if logger:
            logger[].log_scalar("episode_reward", episode_reward, total_steps)
            logger[].log_scalar(
                "explore_rate", agent.get_explore_rate(), total_steps
            )

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_episode_" + String(episode + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            episode + 1
        ) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_reward", avg_reward, total_steps)

            if verbose:
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# Shared Training Loop — Continuous Actions (OffPolicyContinuousAgent)
# =============================================================================


def run_offpolicy_continuous_train[
    E: BoxContinuousActionEnv,
    A: OffPolicyContinuousAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 1000,
    warmup_steps: Int = 1000,
    train_every: Int = 1,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """Warmup + episode training loop for OffPolicyContinuousAgent (DDPG/TD3/SAC).

    Symmetric with run_offpolicy_continuous_train_gpu:
        - cpu_state is passed explicitly (not owned by the agent).
        - The loop calls cpu_state.is_ready() directly.
        - The loop calls agent.do_cpu_train_step(cpu_state).

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OffPolicyContinuousAgent.

    Args:
        agent: Off-policy agent (hyperparameters + algorithm only).
        cpu_state: CPU state buffer (networks + replay + scratch).
        env: Continuous-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 1000).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_cpu_train_step every N steps (default: 1).
        checkpoint_every: Save checkpoint every N episodes (default: 0).
        checkpoint_path: Path to save checkpoint (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()
    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action[E.dtype]()
        var result = env.step_continuous_vec(action)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_transition(
            cpu_state, warmup_obs, action, reward, next_obs, done
        )
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action(cpu_state, obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_transition(
                cpu_state, obs, action, reward, next_obs, done
            )

            if cpu_state.is_ready() and total_steps % train_every == 0:
                _ = agent.do_cpu_train_step(cpu_state)

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        # Logger: per-episode reward
        if logger:
            logger[].log_scalar("episode_reward", episode_reward, total_steps)
            logger[].log_scalar(
                "explore_rate", agent.get_explore_rate(), total_steps
            )

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_episode_" + String(episode + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            episode + 1
        ) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_reward", avg_reward, total_steps)

            if verbose:
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

    if logger:
        logger[].flush()
    return metrics^
