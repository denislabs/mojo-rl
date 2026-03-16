"""Shared on-policy training infrastructure.

Provides two trait hierarchies and shared training-loop functions:

OnPolicyAgent (original, used by A2C):
    Agent owns all state internally; loop calls agent.collect_rollout(env),
    agent.compute_advantages(), agent.update_epochs() without a separate state.

OnPolicyDiscreteAgent / OnPolicyContinuousAgent (new, used by PPO):
    Mirrors OffPolicyContinuousAgent — the loop holds a CPUStateType buffer
    container and calls agent methods with it explicitly. This creates symmetry:

        CPU discrete:   run_onpolicy_discrete_train(agent, cpu_state, env, ...)
        CPU continuous: run_onpolicy_continuous_train(agent, cpu_state, env, ...)

OnPolicyDiscreteState / OnPolicyContinuousState (CPU buffer container traits):
    Hold networks + rollout buffers; expose store_step(), is_full(), clear().

Usage — new OnPolicyDiscreteAgent style (PPO):
    struct MyAgent[...](OnPolicyDiscreteAgent):
        comptime CPUStateType = PPODiscreteState[...]
        fn make_cpu_state(self) -> Self.CPUStateType: ...
        fn collect_rollout[E](mut self, mut cpu_state, mut env: E) -> None: ...
        fn compute_advantages(mut self, mut cpu_state) -> None: ...
        fn update_epochs(mut self, mut cpu_state) -> Float64: ...
        fn select_greedy_action(self, cpu_state, obs) -> List[Float64]: ...
        fn get_explore_rate(self) -> Float64: ...

    var agent = MyAgent[...]()
    var cpu_state = agent.make_cpu_state()
    var metrics = run_onpolicy_discrete_train(agent, cpu_state, env, num_updates=1000)

Usage — original OnPolicyAgent style (A2C):
    var metrics = run_onpolicy_discrete_train(agent, env, num_updates=1000)
    var metrics = run_onpolicy_continuous_train(agent, env, num_updates=1000)
"""

from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from .checkpoint_trait import Checkpointable


# =============================================================================
# OnPolicyAgent Trait
# =============================================================================


trait OnPolicyAgent:
    """Interface for on-policy agents (PPO, A2C).

    Data flow: collect_rollout → compute_advantages → update_epochs → discard.
    The shared training loop calls this sequence num_updates times and handles
    logging, episode tracking, and exploration rate reporting.

    Compile-time constants:
        ROLLOUT_LEN: Number of steps per rollout (baked into buffer sizes).
    """

    comptime ROLLOUT_LEN: Int
    """Number of steps per rollout buffer (compile-time constant)."""

    fn collect_rollout[E: BoxDiscreteActionEnv](mut self, mut env: E) -> None:
        """Collect exactly ROLLOUT_LEN steps in a discrete-action environment.

        Must handle episode resets internally when done=True.
        Stores (obs, action, reward, log_prob, value, done) in internal buffers.

        Args:
            env: Discrete-action environment.
        """
        ...

    fn collect_rollout_continuous[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E) -> None:
        """Collect exactly ROLLOUT_LEN steps in a continuous-action environment.

        Same semantics as collect_rollout but for continuous action spaces.

        Args:
            env: Continuous-action environment.
        """
        ...

    fn compute_advantages(mut self) -> None:
        """Compute GAE advantages and returns from the collected rollout.

        Called after collect_rollout, before update_epochs.
        Uses stored values, rewards, and dones with gamma and gae_lambda.
        """
        ...

    fn update_epochs(mut self) -> Float64:
        """Update policy and value function using the collected rollout.

        For PPO: multiple epochs over minibatches with clipped surrogate loss.
        For A2C: single pass over full rollout.

        Returns:
            Mean policy loss for the update (used for logging).
        """
        ...

    fn select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        """Select action without exploration for evaluation.

        Args:
            obs: Current observation.

        Returns:
            Greedy action list (length 1 discrete, action_dim continuous).
        """
        ...

    fn get_explore_rate(self) -> Float64:
        """Return current exploration rate (entropy coef, policy std, etc.)."""
        ...


# =============================================================================
# Shared Training Loop — Discrete Actions
# =============================================================================


fn run_onpolicy_discrete_train[
    E: BoxDiscreteActionEnv,
    A: OnPolicyAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut env: E,
    num_updates: Int,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
        L, MutAnyOrigin
    ](),
) raises -> TrainingMetrics:
    """Shared on-policy discrete loop: collect → advantages → update × num_updates.

    Eliminates the boilerplate outer loop in A2C.train() and PPO.train()
    (discrete variant). Each agent implements the 6 trait methods; this loop
    handles logging and episode tracking.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OnPolicyAgent and Checkpointable.

    Args:
        agent: On-policy agent (updated in-place).
        env: Discrete-action environment.
        num_updates: Number of collect+update cycles.
        checkpoint_every: Save checkpoint every N updates (default: 0 = disabled).
        checkpoint_path: Base path for checkpoint files (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with one entry per update (reward = policy loss).
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for update in range(num_updates):
        agent.collect_rollout(env)
        agent.compute_advantages()
        var loss = agent.update_epochs()

        # Log update as pseudo-episode (value = policy loss for monitoring)
        metrics.log_episode(
            update,
            Scalar[DType.float64](loss),
            A.ROLLOUT_LEN,
            agent.get_explore_rate(),
        )

        if logger:
            logger[].log_scalar("loss", loss, update)

        if checkpoint_every > 0 and (update + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_update_" + String(update + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            update + 1
        ) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_loss", avg_loss, update)

            if verbose:
                print(
                    "Update "
                    + String(update + 1)
                    + " | Loss: "
                    + String(avg_loss)[:8]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# Shared Training Loop — Continuous Actions
# =============================================================================


fn run_onpolicy_continuous_train[
    E: BoxContinuousActionEnv,
    A: OnPolicyAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut env: E,
    num_updates: Int,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
        L, MutAnyOrigin
    ](),
) raises -> TrainingMetrics:
    """Shared on-policy continuous loop: collect → advantages → update × num_updates.

    Continuous-action variant (PPO with Gaussian policy, SAC on-policy variant).

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OnPolicyAgent and Checkpointable.

    Args:
        agent: On-policy agent (updated in-place).
        env: Continuous-action environment.
        num_updates: Number of collect+update cycles.
        checkpoint_every: Save checkpoint every N updates (default: 0 = disabled).
        checkpoint_path: Base path for checkpoint files (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with one entry per update (reward = policy loss).
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for update in range(num_updates):
        agent.collect_rollout_continuous(env)
        agent.compute_advantages()
        var loss = agent.update_epochs()

        metrics.log_episode(
            update,
            Scalar[DType.float64](loss),
            A.ROLLOUT_LEN,
            agent.get_explore_rate(),
        )

        if logger:
            logger[].log_scalar("loss", loss, update)

        if checkpoint_every > 0 and (update + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_update_" + String(update + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            update + 1
        ) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_loss", avg_loss, update)

            if verbose:
                print(
                    "Update "
                    + String(update + 1)
                    + " | Loss: "
                    + String(avg_loss)[:8]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# OnPolicyDiscreteState Trait
# =============================================================================


trait OnPolicyDiscreteState:
    """CPU-side buffer container for discrete on-policy agents.

    Holds heap-allocated rollout state: actor/critic network weights,
    rollout buffers (obs, actions, rewards, values, log_probs, dones),
    advantage/return scratch, and current-obs for bootstrapping.

    Exposed to training loops via store_step / is_full / clear.
    """

    fn store_step(
        mut self,
        obs: List[Scalar[DType.float32]],
        action: Int,
        reward: Float64,
        value: Scalar[DType.float32],
        log_prob: Scalar[DType.float32],
        done: Bool,
    ) -> None:
        """Store one (obs, action, reward, value, log_prob, done) step."""
        ...

    fn is_full(self) -> Bool:
        """Return True when the rollout buffer is at capacity."""
        ...

    fn clear(mut self) -> None:
        """Reset the write pointer (does not zero the buffer)."""
        ...


# =============================================================================
# OnPolicyContinuousState Trait
# =============================================================================


trait OnPolicyContinuousState:
    """CPU-side buffer container for continuous on-policy agents.

    Same as OnPolicyDiscreteState but action is List[Scalar[float32]]
    (one float per action dimension) instead of Int.
    """

    fn store_step(
        mut self,
        obs: List[Scalar[DType.float32]],
        action: List[Scalar[DType.float32]],
        reward: Float64,
        value: Scalar[DType.float32],
        log_prob: Scalar[DType.float32],
        done: Bool,
    ) -> None:
        """Store one (obs, action, reward, value, log_prob, done) step."""
        ...

    fn is_full(self) -> Bool:
        """Return True when the rollout buffer is at capacity."""
        ...

    fn clear(mut self) -> None:
        """Reset the write pointer (does not zero the buffer)."""
        ...


# =============================================================================
# OnPolicyDiscreteAgent Trait
# =============================================================================


trait OnPolicyDiscreteAgent:
    """Discrete on-policy agent with explicit CPU state management.

    The agent struct owns only hyperparameters and algorithm logic.
    All heap-allocated state (networks, rollout buffers, scratch) lives in
    CPUStateType, which is created via make_cpu_state() and held by the caller.

    CPU training: run_onpolicy_discrete_train(agent, cpu_state, env, ...)

    Compile-time constants:
        CPUStateType: Concrete OnPolicyDiscreteState implementation.
    """

    comptime CPUStateType: OnPolicyDiscreteState

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType. Networks initialized with Xavier."""
        ...

    fn collect_rollout[
        E: BoxDiscreteActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        """Collect exactly rollout_len steps into cpu_state rollout buffers."""
        ...

    fn compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        """Compute GAE advantages and returns from the collected rollout."""
        ...

    fn update_epochs(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Update actor/critic over multiple epochs. Returns mean policy loss.
        """
        ...

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select a greedy action for evaluation."""
        ...

    fn get_explore_rate(self) -> Float64:
        """Return current exploration coefficient."""
        ...


# =============================================================================
# OnPolicyContinuousAgent Trait
# =============================================================================


trait OnPolicyContinuousAgent:
    """Continuous on-policy agent with explicit CPU state management.

    Symmetric to OnPolicyDiscreteAgent but for continuous action spaces.
    """

    comptime CPUStateType: OnPolicyContinuousState

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType."""
        ...

    fn collect_rollout[
        E: BoxContinuousActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        """Collect exactly rollout_len steps into cpu_state rollout buffers."""
        ...

    fn compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        """Compute GAE advantages and returns from the collected rollout."""
        ...

    fn update_epochs(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Update actor/critic over multiple epochs. Returns mean policy loss.
        """
        ...

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select a deterministic action for evaluation."""
        ...

    fn get_explore_rate(self) -> Float64:
        """Return current exploration coefficient."""
        ...


# =============================================================================
# New Training Loop — Discrete Actions (OnPolicyDiscreteAgent)
# =============================================================================


fn run_onpolicy_discrete_train[
    E: BoxDiscreteActionEnv,
    A: OnPolicyDiscreteAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut cpu_state: A.CPUStateType,
    mut env: E,
    num_updates: Int,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
        L, MutAnyOrigin
    ](),
) raises -> TrainingMetrics:
    """Shared on-policy discrete loop with explicit state: collect → advantages → update.

    The caller creates state via agent.make_cpu_state() and holds it across
    updates, enabling persistent env state and weight continuity between rollouts.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OnPolicyDiscreteAgent and Checkpointable.

    Args:
        agent: On-policy agent (hyperparameters + update logic).
        cpu_state: CPU buffer container (networks + rollout buffers).
        env: Discrete-action environment.
        num_updates: Number of collect+update cycles.
        checkpoint_every: Save checkpoint every N updates (default: 0 = disabled).
        checkpoint_path: Base path for checkpoint files (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with one entry per update (value = policy loss).
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for update in range(num_updates):
        agent.collect_rollout(cpu_state, env)
        agent.compute_advantages(cpu_state)
        var loss = agent.update_epochs(cpu_state)

        metrics.log_episode(
            update,
            Scalar[DType.float64](loss),
            0,
            agent.get_explore_rate(),
        )

        if logger:
            logger[].log_scalar("loss", loss, update)

        if checkpoint_every > 0 and (update + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_update_" + String(update + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            update + 1
        ) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_loss", avg_loss, update)

            if verbose:
                print(
                    "Update "
                    + String(update + 1)
                    + " | Loss: "
                    + String(avg_loss)[:8]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# New Training Loop — Continuous Actions (OnPolicyContinuousAgent)
# =============================================================================


fn run_onpolicy_continuous_train[
    E: BoxContinuousActionEnv,
    A: OnPolicyContinuousAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    mut cpu_state: A.CPUStateType,
    mut env: E,
    num_updates: Int,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
        L, MutAnyOrigin
    ](),
) raises -> TrainingMetrics:
    """Shared on-policy continuous loop with explicit state: collect → advantages → update.

    Continuous-action variant (PPO Gaussian policy). Same structure as the
    discrete overload but env type bound is BoxContinuousActionEnv.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OnPolicyContinuousAgent and Checkpointable.

    Args:
        agent: On-policy agent (hyperparameters + update logic).
        cpu_state: CPU buffer container (networks + rollout buffers).
        env: Continuous-action environment.
        num_updates: Number of collect+update cycles.
        checkpoint_every: Save checkpoint every N updates (default: 0 = disabled).
        checkpoint_path: Base path for checkpoint files (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        logger: Optional metrics logger pointer (default: null = no logging).

    Returns:
        TrainingMetrics with one entry per update (value = policy loss).
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for update in range(num_updates):
        agent.collect_rollout(cpu_state, env)
        agent.compute_advantages(cpu_state)
        var loss = agent.update_epochs(cpu_state)

        metrics.log_episode(
            update,
            Scalar[DType.float64](loss),
            0,
            agent.get_explore_rate(),
        )

        if logger:
            logger[].log_scalar("loss", loss, update)

        if checkpoint_every > 0 and (update + 1) % checkpoint_every == 0:
            agent.save_checkpoint(
                checkpoint_path + "_update_" + String(update + 1) + ".ckpt"
            )

        if (verbose or (logger and logger[].is_active())) and (
            update + 1
        ) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            if logger:
                logger[].log_scalar("avg_loss", avg_loss, update)

            if verbose:
                print(
                    "Update "
                    + String(update + 1)
                    + " | Loss: "
                    + String(avg_loss)[:8]
                    + " | Explore: "
                    + String(agent.get_explore_rate())[:5]
                )

    if logger:
        logger[].flush()
    return metrics^
