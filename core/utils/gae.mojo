"""Generalized Advantage Estimation (GAE) utilities.

GAE provides a family of advantage estimators parameterized by λ that
trade off bias and variance in policy gradient methods. It's used by
PPO, A2C, and other actor-critic algorithms.

The GAE formula:
    A^GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD residual.

Special cases:
    - λ=0: One-step TD advantage (low variance, high bias)
    - λ=1: Monte Carlo advantage (high variance, low bias)
    - 0<λ<1: Exponentially-weighted average of n-step advantages

References:
    - Schulman et al. (2016): "High-Dimensional Continuous Control Using GAE"

Example usage:
    from core.utils.gae import compute_gae, compute_gae_inline

    # For List-based code (tile-coded PPO)
    var advantages = compute_gae(rewards, values, next_value, done, gamma, lambda_)
    var returns = compute_returns_from_advantages(advantages, values)

    # For InlineArray-based code (deep PPO)
    compute_gae_inline[2048](
        rewards, values, next_value, dones, gamma, lambda_,
        buffer_len, advantages, returns
    )
"""


fn compute_gae(
    rewards: List[Float64],
    values: List[Float64],
    next_value: Float64,
    done: Bool,
    gamma: Float64,
    gae_lambda: Float64,
) -> List[Float64]:
    """Compute Generalized Advantage Estimation.

    Computes advantages by iterating backwards through the trajectory,
    accumulating discounted TD residuals.

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}].
        values: List of value estimates [V(s_0), V(s_1), ..., V(s_{T-1})].
        next_value: Bootstrap value V(s_T) (used if not done).
        done: Whether the episode terminated at the last step.
        gamma: Discount factor γ.
        gae_lambda: GAE parameter λ.

    Returns:
        List of advantages [A_0, A_1, ..., A_{T-1}].

    Example:
        var rewards = List[Float64]()
        var values = List[Float64]()
        # ... collect trajectory ...
        var advantages = compute_gae(
            rewards, values, next_value, done, 0.99, 0.95
        )
    """
    var num_steps = len(rewards)
    var advantages = List[Float64]()

    # Initialize advantages list
    for _ in range(num_steps):
        advantages.append(0.0)

    # Bootstrap value for last step
    var last_value = next_value
    if done:
        last_value = 0.0

    # Compute GAE backwards
    var gae: Float64 = 0.0
    var gae_decay = gamma * gae_lambda

    for t in range(num_steps - 1, -1, -1):
        # Get next value
        var next_val: Float64
        if t == num_steps - 1:
            next_val = last_value
        else:
            next_val = values[t + 1]

        # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        var delta = rewards[t] + gamma * next_val - values[t]

        # GAE: A_t = δ_t + γλA_{t+1}
        gae = delta + gae_decay * gae
        advantages[t] = gae

    return advantages^


fn compute_returns_from_advantages(
    advantages: List[Float64],
    values: List[Float64],
) -> List[Float64]:
    """Compute returns from advantages and values.

    Returns = Advantages + Values

    This is used to compute targets for the value function when training
    with GAE.

    Args:
        advantages: GAE advantages.
        values: Value estimates.

    Returns:
        Returns for value function training.

    Example:
        var advantages = compute_gae(...)
        var returns = compute_returns_from_advantages(advantages, values)
        # Use returns as targets for critic: loss = (V(s) - returns)^2
    """
    var returns = List[Float64]()
    for t in range(len(advantages)):
        returns.append(advantages[t] + values[t])
    return returns^


fn compute_gae_inline[
    dtype: DType, N: Int
](
    rewards: InlineArray[Scalar[dtype], N],
    values: InlineArray[Scalar[dtype], N],
    next_value: Scalar[dtype],
    dones: InlineArray[Bool, N],
    gamma: Float64,
    gae_lambda: Float64,
    buffer_len: Int,
    mut advantages: InlineArray[Scalar[dtype], N],
    mut returns: InlineArray[Scalar[dtype], N],
):
    """Compute GAE advantages and returns for InlineArray-based agents.

    This version handles multiple episodes within a single rollout by
    resetting the GAE accumulator when `done` is True.

    Args:
        rewards: Rewards collected during rollout.
        values: Value estimates for each state.
        next_value: Bootstrap value V(s_T).
        dones: Done flags for each step (True if episode ended).
        gamma: Discount factor γ.
        gae_lambda: GAE parameter λ.
        buffer_len: Actual number of steps in buffer.
        advantages: Output buffer for advantages.
        returns: Output buffer for returns.

    Parameters:
        dtype: Data type for values.
        N: Maximum rollout length.

    Example:
        var advantages = InlineArray[Scalar[DType.float32], 2048](fill=0)
        var returns = InlineArray[Scalar[DType.float32], 2048](fill=0)
        compute_gae_inline[DType.float32, 2048](
            rewards, values, next_value, dones, 0.99, 0.95,
            buffer_len, advantages, returns
        )
    """
    var gae = Scalar[dtype](0.0)
    var gae_decay = Scalar[dtype](gamma * gae_lambda)

    # Compute GAE backwards
    for t in range(buffer_len - 1, -1, -1):
        # Get next value
        var next_val: Scalar[dtype]
        if t == buffer_len - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # Zero out next value if done (episode boundary)
        if dones[t]:
            next_val = Scalar[dtype](0.0)
            gae = Scalar[dtype](0.0)

        # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        var delta = rewards[t] + Scalar[dtype](gamma) * next_val - values[t]

        # GAE: A_t = δ_t + γλA_{t+1}
        gae = delta + gae_decay * gae

        advantages[t] = gae
        returns[t] = gae + values[t]


fn compute_nstep_returns(
    rewards: List[Float64],
    next_value: Float64,
    done: Bool,
    gamma: Float64,
) -> List[Float64]:
    """Compute n-step returns (Monte Carlo returns with bootstrap).

    Computes: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t}V(s_T)

    This is equivalent to GAE with λ=1.

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}].
        next_value: Bootstrap value V(s_T) (used if not done).
        done: Whether the episode terminated.
        gamma: Discount factor γ.

    Returns:
        List of n-step returns [G_0, G_1, ..., G_{T-1}].

    Example:
        var returns = compute_nstep_returns(rewards, next_value, done, 0.99)
    """
    var num_steps = len(rewards)
    var returns = List[Float64]()

    # Initialize returns list
    for _ in range(num_steps):
        returns.append(0.0)

    # Bootstrap value
    var g = next_value
    if done:
        g = 0.0

    # Compute returns backwards
    for t in range(num_steps - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g

    return returns^


fn compute_td_targets(
    rewards: List[Float64],
    values: List[Float64],
    next_value: Float64,
    done: Bool,
    gamma: Float64,
) -> List[Float64]:
    """Compute one-step TD targets.

    Computes: target_t = r_t + γV(s_{t+1})

    This is equivalent to GAE with λ=0.

    Args:
        rewards: List of rewards.
        values: List of current value estimates.
        next_value: Bootstrap value V(s_T).
        done: Whether the episode terminated.
        gamma: Discount factor γ.

    Returns:
        List of TD targets for value function training.

    Example:
        var targets = compute_td_targets(rewards, values, next_value, done, 0.99)
        # Use targets to update critic: loss = (V(s) - target)^2
    """
    var num_steps = len(rewards)
    var targets = List[Float64]()

    for t in range(num_steps):
        var next_val: Float64
        if t == num_steps - 1:
            next_val = next_value if not done else 0.0
        else:
            next_val = values[t + 1]

        targets.append(rewards[t] + gamma * next_val)

    return targets^
