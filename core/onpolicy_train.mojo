"""Shared on-policy training infrastructure.

Provides the OnPolicyAgent trait and free training loop functions for
PPO, A2C, and other on-policy algorithms.

Data flow per update: collect_rollout → compute_advantages → update_epochs → discard.
Agents implement these 6 methods and the shared loop handles the outer
update count, logging, and episode tracking.

Usage:
    struct MyAgent[...](OnPolicyAgent):
        comptime ROLLOUT_LEN: Int = 128

        fn collect_rollout[E: BoxDiscreteActionEnv](mut self, mut env: E): ...
        fn collect_rollout_continuous[E: BoxContinuousActionEnv](mut self, mut env: E): ...
        fn compute_advantages(mut self): ...
        fn update_epochs(mut self) -> Float64: ...
        fn select_greedy_action_list(self, obs: List[Scalar[Float64]]) -> ...: ...
        fn get_explore_rate(self) -> Float64: ...

    var metrics = run_onpolicy_discrete_train(agent, env, num_updates=1000)
    var metrics = run_onpolicy_continuous_train(agent, env, num_updates=1000)
"""

from .metrics import TrainingMetrics
from .env_traits import BoxDiscreteActionEnv, BoxContinuousActionEnv


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

    fn collect_rollout[
        E: BoxDiscreteActionEnv
    ](mut self, mut env: E) -> None:
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

    fn select_greedy_action_list(
        self, obs: List[Scalar[Float64]]
    ) -> List[Scalar[Float64]]:
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
    E: BoxDiscreteActionEnv, A: OnPolicyAgent
](
    mut agent: A,
    mut env: E,
    num_updates: Int,
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
) -> TrainingMetrics:
    """Shared on-policy discrete loop: collect → advantages → update × num_updates.

    Eliminates the boilerplate outer loop in A2C.train() and PPO.train()
    (discrete variant). Each agent implements the 6 trait methods; this loop
    handles logging and episode tracking.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OnPolicyAgent.

    Args:
        agent: On-policy agent (updated in-place).
        env: Discrete-action environment.
        num_updates: Number of collect+update cycles.
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

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

        if verbose and (update + 1) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            print(
                "Update "
                + String(update + 1)
                + " | Loss: "
                + String(avg_loss)[:8]
                + " | Explore: "
                + String(agent.get_explore_rate())[:5]
            )

    return metrics^


# =============================================================================
# Shared Training Loop — Continuous Actions
# =============================================================================


fn run_onpolicy_continuous_train[
    E: BoxContinuousActionEnv, A: OnPolicyAgent
](
    mut agent: A,
    mut env: E,
    num_updates: Int,
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OnPolicy",
) -> TrainingMetrics:
    """Shared on-policy continuous loop: collect → advantages → update × num_updates.

    Continuous-action variant (PPO with Gaussian policy, SAC on-policy variant).

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OnPolicyAgent.

    Args:
        agent: On-policy agent (updated in-place).
        env: Continuous-action environment.
        num_updates: Number of collect+update cycles.
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

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

        if verbose and (update + 1) % print_every == 0:
            var avg_loss = metrics.mean_reward_last_n(print_every)
            print(
                "Update "
                + String(update + 1)
                + " | Loss: "
                + String(avg_loss)[:8]
                + " | Explore: "
                + String(agent.get_explore_rate())[:5]
            )

    return metrics^
