"""Gymnasium Classic Control environments wrapper.

Classic Control environments:
- CartPole-v1: Balance pole on a cart (discrete)
- MountainCar-v0: Drive car up a mountain (discrete)
- MountainCarContinuous-v0: Drive car up mountain (continuous)
- Pendulum-v1: Swing up and balance pendulum (continuous)
- Acrobot-v1: Swing up double pendulum (discrete)

All use physics simulations with continuous observation spaces.
"""

from python import Python, PythonObject


struct GymCartPoleEnv:
    """CartPole-v1 environment via Gymnasium Python bindings.

    Observation space: Box(4,) - [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    Action space: Discrete(2) - 0: push left, 1: push right

    Episode terminates when:
    - Pole angle > ±12°
    - Cart position > ±2.4
    - Episode length > 500 steps
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 4]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        """Initialize CartPole environment from Gymnasium.

        Args:
            render_mode: "human" for visual rendering, "" for no rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            # Use Python kwargs for render_mode parameter
            self.env = self.gym.make(
                "CartPole-v1", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("CartPole-v1")
        self.current_obs = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> SIMD[DType.float64, 4]:
        """Reset environment and return initial observation."""
        var result = self.env.reset()
        var obs = result[0]  # Gymnasium returns (obs, info)

        # Convert numpy array to SIMD using Float64() constructor
        self.current_obs[0] = Float64(obs[0])
        self.current_obs[1] = Float64(obs[1])
        self.current_obs[2] = Float64(obs[2])
        self.current_obs[3] = Float64(obs[3])

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step(
        mut self, action: Int
    ) raises -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (observation, reward, done).

        Args:
            action: 0 for left, 1 for right
        """
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        # Convert numpy observation to SIMD
        self.current_obs[0] = Float64(obs[0])
        self.current_obs[1] = Float64(obs[1])
        self.current_obs[2] = Float64(obs[2])
        self.current_obs[3] = Float64(obs[3])

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current observation."""
        return self.current_obs

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn num_actions(self) -> Int:
        """Return number of discrete actions (2 for CartPole)."""
        return 2

    fn obs_dim(self) -> Int:
        """Return observation dimension (4 for CartPole)."""
        return 4

    fn close(mut self) raises:
        """Close the environment."""
        _ = self.env.close()

    fn render(self) raises:
        """Render the environment (if render_mode was set)."""
        _ = self.env.render()


fn discretize_cart_pole(obs: SIMD[DType.float64, 4], num_bins: Int = 10) -> Int:
    """Discretize continuous observation into a single state index.

    This enables using tabular methods on CartPole by binning observations.

    Args:
        obs: 4D observation [cart_pos, cart_vel, pole_angle, pole_vel]
        num_bins: Number of bins per dimension

    Returns:
        Single integer state index in [0, num_bins^4)
    """
    # Observation bounds (approximate for CartPole)
    var cart_pos_low: Float64 = -2.4
    var cart_pos_high: Float64 = 2.4
    var cart_vel_low: Float64 = -3.0
    var cart_vel_high: Float64 = 3.0
    var pole_angle_low: Float64 = -0.21  # ~12 degrees in radians
    var pole_angle_high: Float64 = 0.21
    var pole_vel_low: Float64 = -3.0
    var pole_vel_high: Float64 = 3.0

    fn bin_value(value: Float64, low: Float64, high: Float64, bins: Int) -> Int:
        """Bin a value into discrete bucket."""
        var normalized = (value - low) / (high - low)
        # Clamp to [0, 1]
        if normalized < 0.0:
            normalized = 0.0
        elif normalized > 1.0:
            normalized = 1.0
        var bucket = Int(normalized * Float64(bins - 1))
        return bucket

    var b0 = bin_value(obs[0], cart_pos_low, cart_pos_high, num_bins)
    var b1 = bin_value(obs[1], cart_vel_low, cart_vel_high, num_bins)
    var b2 = bin_value(obs[2], pole_angle_low, pole_angle_high, num_bins)
    var b3 = bin_value(obs[3], pole_vel_low, pole_vel_high, num_bins)

    # Combine into single index
    return ((b0 * num_bins + b1) * num_bins + b2) * num_bins + b3


fn get_cart_pole_num_states(num_bins: Int = 10) -> Int:
    """Return total number of discretized states."""
    return num_bins * num_bins * num_bins * num_bins


struct GymMountainCarEnv:
    """MountainCar-v0: Drive an underpowered car up a steep mountain.

    Observation: [position, velocity] - Box(2,)
        position: -1.2 to 0.6
        velocity: -0.07 to 0.07

    Actions: Discrete(3)
        0: Push left
        1: No push
        2: Push right

    Reward: -1 for each step until goal reached
    Goal: Reach position >= 0.5

    Episode ends: Position >= 0.5 or 200 steps
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 2]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "MountainCar-v0", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("MountainCar-v0")
        self.current_obs = SIMD[DType.float64, 2](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> SIMD[DType.float64, 2]:
        var result = self.env.reset()
        var obs = result[0]
        self.current_obs[0] = Float64(obs[0])
        self.current_obs[1] = Float64(obs[1])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step(
        mut self, action: Int
    ) raises -> Tuple[SIMD[DType.float64, 2], Float64, Bool]:
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.current_obs[0] = Float64(obs[0])
        self.current_obs[1] = Float64(obs[1])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_actions(self) -> Int:
        return 3

    fn obs_dim(self) -> Int:
        return 2


struct PendulumEnv:
    """Pendulum-v1: Swing up and balance an inverted pendulum.

    Observation: [cos(theta), sin(theta), theta_dot] - Box(3,)

    Actions: Continuous Box(1,) - torque in [-2, 2]

    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    Goal: Keep pendulum upright (theta = 0)

    Episode ends: After 200 steps (no early termination)
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[
        DType.float64, 4
    ]  # Using 4 for alignment, only 3 used
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "Pendulum-v1", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("Pendulum-v1")
        self.current_obs = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> SIMD[DType.float64, 4]:
        var result = self.env.reset()
        var obs = result[0]
        self.current_obs[0] = Float64(obs[0])  # cos(theta)
        self.current_obs[1] = Float64(obs[1])  # sin(theta)
        self.current_obs[2] = Float64(obs[2])  # theta_dot
        self.current_obs[3] = 0.0  # padding
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step(
        mut self, action: Float64
    ) raises -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take continuous action (torque in [-2, 2])."""
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        _ = py_list.append(action)
        var np_action = self.np.array(py_list)
        var result = self.env.step(np_action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.current_obs[0] = Float64(obs[0])
        self.current_obs[1] = Float64(obs[1])
        self.current_obs[2] = Float64(obs[2])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn obs_dim(self) -> Int:
        return 3

    fn action_dim(self) -> Int:
        return 1

    fn action_low(self) -> Float64:
        return -2.0

    fn action_high(self) -> Float64:
        return 2.0


struct AcrobotEnv:
    """Acrobot-v1: Swing up a two-link robot arm.

    Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] - Box(6,)

    Actions: Discrete(3)
        0: Apply -1 torque to joint
        1: Apply 0 torque
        2: Apply +1 torque

    Reward: -1 for each step until goal
    Goal: Swing the tip above the base (height threshold)

    Episode ends: Tip above threshold or 500 steps
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[
        DType.float64, 8
    ]  # Using 8 for alignment, only 6 used
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "Acrobot-v1", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("Acrobot-v1")
        self.current_obs = SIMD[DType.float64, 8](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> SIMD[DType.float64, 8]:
        var result = self.env.reset()
        var obs = result[0]
        for i in range(6):
            self.current_obs[i] = Float64(obs[i])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step(
        mut self, action: Int
    ) raises -> Tuple[SIMD[DType.float64, 8], Float64, Bool]:
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        for i in range(6):
            self.current_obs[i] = Float64(obs[i])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_actions(self) -> Int:
        return 3

    fn obs_dim(self) -> Int:
        return 6


# Discretization helpers for tabular methods


fn discretize_mountain_car(
    obs: SIMD[DType.float64, 2], num_bins: Int = 20
) -> Int:
    """Discretize MountainCar observation into state index."""
    var pos_low: Float64 = -1.2
    var pos_high: Float64 = 0.6
    var vel_low: Float64 = -0.07
    var vel_high: Float64 = 0.07

    fn bin_value(value: Float64, low: Float64, high: Float64, bins: Int) -> Int:
        var normalized = (value - low) / (high - low)
        if normalized < 0.0:
            normalized = 0.0
        elif normalized > 1.0:
            normalized = 1.0
        return Int(normalized * Float64(bins - 1))

    var b0 = bin_value(obs[0], pos_low, pos_high, num_bins)
    var b1 = bin_value(obs[1], vel_low, vel_high, num_bins)

    return b0 * num_bins + b1


fn discretize_acrobot(obs: SIMD[DType.float64, 8], num_bins: Int = 6) -> Int:
    """Discretize Acrobot observation into state index.

    Uses fewer bins since 6D observation space is large.
    """
    # Bounds for each dimension
    var bounds_low = SIMD[DType.float64, 8](
        -1.0, -1.0, -1.0, -1.0, -12.566, -28.274, 0.0, 0.0
    )
    var bounds_high = SIMD[DType.float64, 8](
        1.0, 1.0, 1.0, 1.0, 12.566, 28.274, 0.0, 0.0
    )

    fn bin_value(value: Float64, low: Float64, high: Float64, bins: Int) -> Int:
        var normalized = (value - low) / (high - low)
        if normalized < 0.0:
            normalized = 0.0
        elif normalized > 1.0:
            normalized = 1.0
        return Int(normalized * Float64(bins - 1))

    var index = 0
    var multiplier = 1
    for i in range(6):
        var b = bin_value(obs[i], bounds_low[i], bounds_high[i], num_bins)
        index += b * multiplier
        multiplier *= num_bins

    return index


fn get_mountain_car_num_states(num_bins: Int = 20) -> Int:
    return num_bins * num_bins


fn get_acrobot_num_states(num_bins: Int = 6) -> Int:
    return num_bins * num_bins * num_bins * num_bins * num_bins * num_bins
