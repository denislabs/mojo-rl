"""Shared evaluation loops for off-policy deep RL agents.

Provides run_offpolicy_{discrete,continuous}_eval as drop-in replacements for
the copy-pasted 95-line evaluate() methods in DQN, DDPG, TD3, and SAC.

Key differences from the per-agent evaluate():
- Returns TrainingMetrics (not a raw Float64) for richer analysis
- agent is not mut — evaluation must not modify agent state
- No render support (use agent.evaluate() with RenderableEnv for that)

Usage:
    from core.eval import run_offpolicy_continuous_eval

    var metrics = run_offpolicy_continuous_eval(
        agent, env, num_episodes=10, verbose=True
    )
    print("Mean reward:", metrics.mean_reward())

    # Each agent's evaluate() becomes a thin delegation:
    fn evaluate[E: BoxContinuousActionEnv](self, mut env: E, ...) -> Float64:
        return run_offpolicy_continuous_eval(
            self, env, num_episodes=num_episodes, verbose=verbose,
        ).mean_reward()
"""

from .metrics import TrainingMetrics
from .env_traits import BoxDiscreteActionEnv, BoxContinuousActionEnv, RenderableEnv
from .offpolicy_train import OffPolicyAgent, OffPolicyContinuousAgent, OffPolicyDiscreteAgent


# =============================================================================
# Shared Evaluation Loop — Continuous Actions
# =============================================================================


fn run_offpolicy_continuous_eval[
    E: BoxContinuousActionEnv, A: OffPolicyAgent
](
    agent: A,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) -> TrainingMetrics:
    """Evaluate an off-policy continuous-action agent (no render support).

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OffPolicyAgent.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action_list(obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    return metrics^


fn run_offpolicy_continuous_eval[
    E: BoxContinuousActionEnv & RenderableEnv, A: OffPolicyAgent
](
    agent: A,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    render: Bool = False,
    frame_delay_ms: Int = 16,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) raises -> TrainingMetrics:
    """Evaluate an off-policy continuous-action agent (deterministic policy).

    Uses select_greedy_action_list to run evaluation without exploration noise.
    Replaces the copy-pasted 95-line evaluate() in DDPG, TD3, and SAC.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv and RenderableEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Trained agent — immutable, evaluation does not update state.
        env: Continuous-action environment.
        num_episodes: Number of evaluation episodes (default: 10).
        max_steps: Maximum steps per episode (default: 1000).
        verbose: Print per-episode results (default: False).
        render: Render the environment (default: False).
        frame_delay_ms: Delay between frames in milliseconds (default: 16).
        algorithm_name: Name for metrics labeling (default: "Eval").
        environment_name: Name for metrics labeling (default: "Environment").

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
        Use metrics.mean_reward() for average reward across episodes.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )
    var quit_requested = False

    if render:
        _ = env.init_renderer()

    for episode in range(num_episodes):
        if quit_requested:
            break

        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action_list(obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if render:
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
                if env.check_renderer_quit():
                    quit_requested = True
                    break

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    if render:
        env.close_renderer()

    return metrics^


# =============================================================================
# Shared Evaluation Loop — Discrete Actions
# =============================================================================


fn run_offpolicy_discrete_eval[
    E: BoxDiscreteActionEnv, A: OffPolicyAgent
](
    agent: A,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 500,
    verbose: Bool = False,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) -> TrainingMetrics:
    """Evaluate an off-policy discrete-action agent (greedy argmax policy).

    Uses select_greedy_action_list to run evaluation without epsilon-greedy.
    Replaces the copy-pasted evaluate() in DQN agents (for those implementing
    OffPolicyAgent).

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Trained agent — immutable, evaluation does not update state.
        env: Discrete-action environment.
        num_episodes: Number of evaluation episodes (default: 10).
        max_steps: Maximum steps per episode (default: 500).
        verbose: Print per-episode results (default: False).
        algorithm_name: Name for metrics labeling (default: "Eval").
        environment_name: Name for metrics labeling (default: "Environment").

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
        Use metrics.mean_reward() for average reward across episodes.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for episode in range(num_episodes):
        var obs = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action_list = agent.select_greedy_action_list(obs)
            var action_int = Int(Float64(action_list[0]))
            var result = env.step_obs(action_int)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    return metrics^

# =============================================================================
# Shared Evaluation Loop — Continuous Actions (OffPolicyContinuousAgent)
# =============================================================================


fn run_offpolicy_continuous_eval[
    E: BoxContinuousActionEnv, A: OffPolicyContinuousAgent
](
    agent: A,
    cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) -> TrainingMetrics:
    """Evaluate an OffPolicyContinuousAgent (no render support).

    Symmetric with the OffPolicyAgent overload but takes cpu_state explicitly.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OffPolicyContinuousAgent.

    Args:
        agent: Trained agent — immutable, evaluation does not update state.
        cpu_state: CPU state holding the trained network weights.
        env: Continuous-action environment.
        num_episodes: Number of evaluation episodes (default: 10).
        max_steps: Maximum steps per episode (default: 1000).
        verbose: Print per-episode results (default: False).
        algorithm_name: Name for metrics labeling.
        environment_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action(cpu_state, obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    return metrics^


fn run_offpolicy_continuous_eval[
    E: BoxContinuousActionEnv & RenderableEnv, A: OffPolicyContinuousAgent
](
    agent: A,
    cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    render: Bool = False,
    frame_delay_ms: Int = 16,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) raises -> TrainingMetrics:
    """Evaluate an OffPolicyContinuousAgent with optional rendering.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv and RenderableEnv.
        A: Agent type implementing OffPolicyContinuousAgent.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )
    var quit_requested = False

    if render:
        _ = env.init_renderer()

    for episode in range(num_episodes):
        if quit_requested:
            break

        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action(cpu_state, obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if render:
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
                if env.check_renderer_quit():
                    quit_requested = True
                    break

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    if render:
        env.close_renderer()

    return metrics^


# =============================================================================
# Shared Evaluation Loop — Discrete Actions (OffPolicyDiscreteAgent)
# =============================================================================


fn run_offpolicy_discrete_eval[
    E: BoxDiscreteActionEnv, A: OffPolicyDiscreteAgent
](
    agent: A,
    cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) -> TrainingMetrics:
    """Evaluate an OffPolicyDiscreteAgent using greedy policy (no render).

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OffPolicyDiscreteAgent.

    Args:
        agent: Trained agent — immutable, evaluation does not update state.
        cpu_state: CPU state holding the trained network weights.
        env: Discrete-action environment.
        num_episodes: Number of evaluation episodes (default: 10).
        max_steps: Maximum steps per episode (default: 1000).
        verbose: Print per-episode results (default: False).
        algorithm_name: Name for metrics labeling.
        environment_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action(cpu_state, obs)
            var result = env.step_obs(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            episode_reward += Float64(result[1])
            episode_steps += 1
            obs = next_obs^

            if result[2]:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    return metrics^


fn run_offpolicy_discrete_eval[
    E: BoxDiscreteActionEnv & RenderableEnv, A: OffPolicyDiscreteAgent
](
    agent: A,
    cpu_state: A.CPUStateType,
    mut env: E,
    num_episodes: Int = 10,
    max_steps: Int = 1000,
    verbose: Bool = False,
    render: Bool = False,
    frame_delay_ms: Int = 16,
    algorithm_name: String = "Eval",
    environment_name: String = "Environment",
) raises -> TrainingMetrics:
    """Evaluate an OffPolicyDiscreteAgent with optional rendering.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv and RenderableEnv.
        A: Agent type implementing OffPolicyDiscreteAgent.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )
    var quit_requested = False

    if render:
        _ = env.init_renderer()

    for episode in range(num_episodes):
        if quit_requested:
            break

        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps):
            var action = agent.select_greedy_action(cpu_state, obs)
            var result = env.step_obs(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            episode_reward += reward
            episode_steps += 1
            obs = next_obs^

            if render:
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
                if env.check_renderer_quit():
                    quit_requested = True
                    break

            if done:
                break

        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            0.0,
        )

        if verbose:
            print(
                "Eval Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[:10],
                "| Steps:",
                episode_steps,
            )

    if render:
        env.close_renderer()

    return metrics^
