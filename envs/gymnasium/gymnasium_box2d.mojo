"""Gymnasium Box2D environments wrapper.

Box2D environments (require `pip install gymnasium[box2d]`):
- LunarLander-v3: Land a spacecraft (discrete or continuous)
- BipedalWalker-v3: Walk with a 2D biped robot (continuous)
- CarRacing-v3: Drive a car around a track (continuous)

These use the Box2D physics engine for 2D rigid body simulation.
"""

from python import Python, PythonObject


struct GymLunarLanderEnv:
    """LunarLander-v3: Land a spacecraft on the moon.

    Observation: Box(8,)
        [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

    Actions (Discrete): Discrete(4)
        0: Do nothing
        1: Fire left engine
        2: Fire main engine
        3: Fire right engine

    Reward:
        - Moving toward landing pad: positive
        - Moving away: negative
        - Crash: -100
        - Rest: +100
        - Leg contact: +10 each
        - Fuel usage: small negative

    Episode ends: Landed, crashed, or 1000 steps
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 8]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var is_continuous: Bool

    fn __init__(
        out self, continuous: Bool = False, render_mode: String = ""
    ) raises:
        """Initialize LunarLander.

        Args:
            continuous: If True, use continuous action space
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        self.is_continuous = continuous

        var env_name = "LunarLander-v3"
        if render_mode == "human":
            if continuous:
                self.env = self.gym.make(
                    env_name,
                    continuous=PythonObject(True),
                    render_mode=PythonObject("human"),
                )
            else:
                self.env = self.gym.make(
                    env_name, render_mode=PythonObject("human")
                )
        else:
            if continuous:
                self.env = self.gym.make(
                    env_name, continuous=PythonObject(True)
                )
            else:
                self.env = self.gym.make(env_name)

        self.current_obs = SIMD[DType.float64, 8](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> SIMD[DType.float64, 8]:
        var result = self.env.reset()
        var obs = result[0]
        for i in range(8):
            self.current_obs[i] = Float64(obs[i])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step_discrete(
        mut self, action: Int
    ) raises -> Tuple[SIMD[DType.float64, 8], Float64, Bool]:
        """Take discrete action (0-3)."""
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        for i in range(8):
            self.current_obs[i] = Float64(obs[i])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn step_continuous(
        mut self, main_throttle: Float64, lateral_throttle: Float64
    ) raises -> Tuple[SIMD[DType.float64, 8], Float64, Bool]:
        """Take continuous action [main_engine, lateral_engine]."""
        var action = self.np.array([main_throttle, lateral_throttle])
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        for i in range(8):
            self.current_obs[i] = Float64(obs[i])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_actions(self) -> Int:
        return 4 if not self.is_continuous else 2

    fn obs_dim(self) -> Int:
        return 8


struct GymBipedalWalkerEnv:
    """BipedalWalker-v3: Walk with a 2D biped robot.

    Observation: Box(24,)
        Hull angle, angular velocity, horizontal/vertical speed,
        joint positions and velocities, leg contact with ground, lidar readings

    Actions: Continuous Box(4,) - torques for hip and knee joints
        Each in range [-1, 1]

    Reward:
        - Moving forward: positive
        - Falling: -100
        - Standing still: small negative

    Episode ends: Body touches ground or reaches end of terrain
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: List[Float64]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var hardcore: Bool

    fn __init__(
        out self, hardcore: Bool = False, render_mode: String = ""
    ) raises:
        """Initialize BipedalWalker.

        Args:
            hardcore: If True, use hardcore mode with obstacles
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        self.hardcore = hardcore

        var env_name = (
            "BipedalWalkerHardcore-v3" if hardcore else "BipedalWalker-v3"
        )
        if render_mode == "human":
            self.env = self.gym.make(
                env_name, render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make(env_name)

        self.current_obs = List[Float64]()
        for _ in range(24):
            self.current_obs.append(0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> List[Float64]:
        var result = self.env.reset()
        var obs = result[0]
        for i in range(24):
            self.current_obs[i] = Float64(obs[i])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_obs

    fn step(
        mut self, hip1: Float64, knee1: Float64, hip2: Float64, knee2: Float64
    ) raises -> Tuple[List[Float64], Float64, Bool]:
        """Take continuous action [hip1, knee1, hip2, knee2]."""
        var action = self.np.array([hip1, knee1, hip2, knee2])
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        for i in range(24):
            self.current_obs[i] = Float64(obs[i])
        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn obs_dim(self) -> Int:
        return 24

    fn action_dim(self) -> Int:
        return 4


struct GymCarRacingEnv:
    """CarRacing-v3: Drive a car around a randomly generated track.

    Observation: Box(96, 96, 3) - RGB image from top-down view

    Actions:
        Discrete(5): [do nothing, left, right, gas, brake]
        or Continuous Box(3,): [steering, gas, brake]

    Reward:
        - -0.1 for each frame
        - +1000/N for each tile visited (N = total tiles)

    Episode ends: All tiles visited or 1000 frames

    Note: This env returns image observations, needs CNN for deep RL.
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var is_continuous: Bool

    fn __init__(
        out self, continuous: Bool = True, render_mode: String = ""
    ) raises:
        """Initialize CarRacing.

        Args:
            continuous: If True, use continuous action space
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        self.is_continuous = continuous

        if render_mode == "human":
            if continuous:
                self.env = self.gym.make(
                    "CarRacing-v3",
                    continuous=PythonObject(True),
                    render_mode=PythonObject("human"),
                )
            else:
                self.env = self.gym.make(
                    "CarRacing-v3",
                    continuous=PythonObject(False),
                    render_mode=PythonObject("human"),
                )
        else:
            if continuous:
                self.env = self.gym.make(
                    "CarRacing-v3", continuous=PythonObject(True)
                )
            else:
                self.env = self.gym.make(
                    "CarRacing-v3", continuous=PythonObject(False)
                )

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> PythonObject:
        """Reset and return image observation (96x96x3 numpy array)."""
        var result = self.env.reset()
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return result[0]

    fn step_discrete(
        mut self, action: Int
    ) raises -> Tuple[PythonObject, Float64, Bool]:
        """Take discrete action (0-4)."""
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (obs, reward, self.done)

    fn step_continuous(
        mut self, steering: Float64, gas: Float64, brake: Float64
    ) raises -> Tuple[PythonObject, Float64, Bool]:
        """Take continuous action [steering, gas, brake]."""
        var action = self.np.array([steering, gas, brake])
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (obs, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn obs_shape(self) -> Tuple[Int, Int, Int]:
        return (96, 96, 3)

    fn num_discrete_actions(self) -> Int:
        return 5

    fn action_dim(self) -> Int:
        return 3


# Discretization helper for LunarLander
fn discretize_lunar_lander(
    obs: SIMD[DType.float64, 8], num_bins: Int = 10
) -> Int:
    """Discretize LunarLander observation into state index.

    Note: With 8 dimensions and 10 bins, this creates 10^8 = 100M states.
    Use fewer bins for tabular methods.
    """
    # Approximate bounds for each dimension
    var bounds_low = SIMD[DType.float64, 8](
        -1.5, -1.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0
    )
    var bounds_high = SIMD[DType.float64, 8](
        1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0
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
    for i in range(8):
        var b = bin_value(obs[i], bounds_low[i], bounds_high[i], num_bins)
        index += b * multiplier
        multiplier *= num_bins

    return index
