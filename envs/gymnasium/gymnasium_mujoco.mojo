"""Gymnasium MuJoCo environments wrapper.

MuJoCo environments (require mujoco installation):
- HalfCheetah-v5: Run forward with a 2D cheetah
- Ant-v5: Run forward with a 4-legged robot
- Humanoid-v5: Walk/run with a 3D humanoid
- Walker2d-v5: Walk forward with a 2D biped
- Hopper-v5: Hop forward with a 1-legged robot
- Swimmer-v5: Swim forward with a 3-link robot
- InvertedPendulum-v5: Balance a pole on a cart
- InvertedDoublePendulum-v5: Balance a double pendulum
- Reacher-v5: Reach a target with a 2-link arm
- Pusher-v5: Push an object to a target

All MuJoCo environments have:
- Continuous observation spaces
- Continuous action spaces
- Physics-based dynamics

Note: These require pip install "gymnasium[mujoco]" or mujoco-py
"""

from python import Python, PythonObject


struct MuJoCoEnv:
    """Generic wrapper for MuJoCo environments.

    Since MuJoCo envs have varying observation/action dimensions,
    this uses dynamic Lists instead of fixed SIMD types.
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var env_name: String

    var obs_dim: Int
    var action_dim: Int
    var action_low: Float64
    var action_high: Float64

    var current_obs: List[Float64]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, env_name: String, render_mode: String = "") raises:
        """Initialize a MuJoCo environment.

        Args:
            env_name: Environment name (e.g., "HalfCheetah-v5", "Ant-v5")
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        self.env_name = env_name

        if render_mode == "human":
            self.env = self.gym.make(env_name, render_mode=PythonObject("human"))
        else:
            self.env = self.gym.make(env_name)

        # Get observation space info
        var obs_space = self.env.observation_space
        var obs_shape = obs_space.shape
        self.obs_dim = Int(obs_shape[0])

        # Get action space info
        var act_space = self.env.action_space
        var act_shape = act_space.shape
        self.action_dim = Int(act_shape[0])
        self.action_low = Float64(act_space.low[0])
        self.action_high = Float64(act_space.high[0])

        # Initialize observation storage
        self.current_obs = List[Float64]()
        for _ in range(self.obs_dim):
            self.current_obs.append(0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> List[Float64]:
        var result = self.env.reset()
        var obs = result[0]

        for i in range(self.obs_dim):
            self.current_obs[i] = Float64(obs[i])

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

        return self.current_obs

    fn step(mut self, action: List[Float64]) raises -> Tuple[List[Float64], Float64, Bool]:
        """Take continuous action.

        Args:
            action: List of action values (length = action_dim)
        """
        # Convert Mojo list to numpy array
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        for i in range(len(action)):
            _ = py_list.append(action[i])
        var np_action = self.np.array(py_list)

        var result = self.env.step(np_action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        for i in range(self.obs_dim):
            self.current_obs[i] = Float64(obs[i])

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs, reward, self.done)

    fn sample_action(self) raises -> List[Float64]:
        """Sample random action from action space."""
        var np_action = self.env.action_space.sample()
        var action = List[Float64]()
        for i in range(self.action_dim):
            action.append(Float64(np_action[i]))
        return action

    fn close(mut self) raises:
        _ = self.env.close()

    fn get_info(self) -> String:
        return (
            "MuJoCo Env: " + self.env_name +
            " | Obs dim: " + String(self.obs_dim) +
            " | Action dim: " + String(self.action_dim)
        )


# Convenience functions for common MuJoCo environments

fn make_half_cheetah(render_mode: String = "") raises -> MuJoCoEnv:
    """HalfCheetah-v5: 2D cheetah running task.

    Obs: 17 (positions and velocities)
    Act: 6 (joint torques)
    Reward: Forward velocity - control cost
    """
    return MuJoCoEnv("HalfCheetah-v5", render_mode)


fn make_ant(render_mode: String = "") raises -> MuJoCoEnv:
    """Ant-v5: 4-legged robot running task.

    Obs: 27 (positions and velocities)
    Act: 8 (joint torques)
    Reward: Forward velocity - control cost + survival bonus
    """
    return MuJoCoEnv("Ant-v5", render_mode)


fn make_humanoid(render_mode: String = "") raises -> MuJoCoEnv:
    """Humanoid-v5: 3D humanoid walking/running task.

    Obs: 376 (positions, velocities, and COM-based features)
    Act: 17 (joint torques)
    Reward: Forward velocity - control cost + survival bonus
    """
    return MuJoCoEnv("Humanoid-v5", render_mode)


fn make_walker2d(render_mode: String = "") raises -> MuJoCoEnv:
    """Walker2d-v5: 2D biped walking task.

    Obs: 17 (positions and velocities)
    Act: 6 (joint torques)
    Reward: Forward velocity - control cost + survival bonus
    """
    return MuJoCoEnv("Walker2d-v5", render_mode)


fn make_hopper(render_mode: String = "") raises -> MuJoCoEnv:
    """Hopper-v5: 1-legged hopping task.

    Obs: 11 (positions and velocities)
    Act: 3 (joint torques)
    Reward: Forward velocity - control cost + survival bonus
    """
    return MuJoCoEnv("Hopper-v5", render_mode)


fn make_swimmer(render_mode: String = "") raises -> MuJoCoEnv:
    """Swimmer-v5: 3-link swimming task.

    Obs: 8 (positions and velocities)
    Act: 2 (joint torques)
    Reward: Forward velocity - control cost
    """
    return MuJoCoEnv("Swimmer-v5", render_mode)


fn make_inverted_pendulum(render_mode: String = "") raises -> MuJoCoEnv:
    """InvertedPendulum-v5: Balance a pole on a cart.

    Obs: 4 (cart position/velocity, pole angle/velocity)
    Act: 1 (cart force)
    Reward: +1 for each step pole stays upright
    """
    return MuJoCoEnv("InvertedPendulum-v5", render_mode)


fn make_inverted_double_pendulum(render_mode: String = "") raises -> MuJoCoEnv:
    """InvertedDoublePendulum-v5: Balance a double pendulum on a cart.

    Obs: 11 (positions and velocities)
    Act: 1 (cart force)
    Reward: Based on distance from upright position
    """
    return MuJoCoEnv("InvertedDoublePendulum-v5", render_mode)


fn make_reacher(render_mode: String = "") raises -> MuJoCoEnv:
    """Reacher-v5: Reach a target with a 2-link arm.

    Obs: 11 (arm positions/velocities + target position)
    Act: 2 (joint torques)
    Reward: Negative distance to target - control cost
    """
    return MuJoCoEnv("Reacher-v5", render_mode)


fn make_pusher(render_mode: String = "") raises -> MuJoCoEnv:
    """Pusher-v5: Push an object to a target position.

    Obs: 23 (arm + object + target positions/velocities)
    Act: 7 (joint torques)
    Reward: Negative distance (object to target) - control cost
    """
    return MuJoCoEnv("Pusher-v5", render_mode)
