"""CartPole environment wrapper using Python Gymnasium bindings."""

from python import Python, PythonObject


struct CartPoleEnv:
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
            self.env = self.gym.make("CartPole-v1", render_mode=PythonObject("human"))
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

    fn step(mut self, action: Int) raises -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
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


fn discretize_obs(obs: SIMD[DType.float64, 4], num_bins: Int = 10) -> Int:
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


fn get_num_states(num_bins: Int = 10) -> Int:
    """Return total number of discretized states."""
    return num_bins * num_bins * num_bins * num_bins
