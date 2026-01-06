"""Generic Gymnasium environment wrapper for Mojo.

Provides a flexible wrapper that can work with any Gymnasium environment,
handling the Python-Mojo type conversions automatically.
"""

from python import Python, PythonObject


struct GymnasiumEnv:
    """Generic wrapper for any Gymnasium environment.

    Works with both discrete and continuous action/observation spaces.
    Automatically handles numpy to Mojo conversions.
    """

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var env_name: String

    # Space info
    var obs_shape: List[Int]
    var obs_dim: Int  # Flattened observation dimension
    var action_dim: Int  # Number of discrete actions or continuous action dim
    var is_discrete_action: Bool
    var is_discrete_obs: Bool

    # Episode state
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, env_name: String, render_mode: String = "") raises:
        """Initialize a Gymnasium environment.

        Args:
            env_name: Name of the environment (e.g., "CartPole-v1", "LunarLander-v3")
            render_mode: "human" for visual rendering, "" for no rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        self.env_name = env_name

        if render_mode == "human":
            self.env = self.gym.make(
                env_name, render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make(env_name)

        # Analyze observation space
        var obs_space = self.env.observation_space
        self.obs_shape = List[Int]()
        self.is_discrete_obs = False

        # Check if discrete observation space
        var space_type = Python.import_module("gymnasium.spaces")
        if (
            Python.import_module("builtins")
            .isinstance(obs_space, space_type.Discrete)
            .__bool__()
        ):
            self.is_discrete_obs = True
            self.obs_dim = 1
            self.obs_shape.append(Int(obs_space.n))
        else:
            # Box or MultiDiscrete - get shape
            var shape = obs_space.shape
            self.obs_dim = 1
            for i in range(Int(Python.import_module("builtins").len(shape))):
                var dim = Int(shape[i])
                self.obs_shape.append(dim)
                self.obs_dim *= dim

        # Analyze action space
        var act_space = self.env.action_space
        if (
            Python.import_module("builtins")
            .isinstance(act_space, space_type.Discrete)
            .__bool__()
        ):
            self.is_discrete_action = True
            self.action_dim = Int(act_space.n)
        else:
            self.is_discrete_action = False
            var act_shape = act_space.shape
            self.action_dim = (
                Int(act_shape[0]) if Int(
                    Python.import_module("builtins").len(act_shape)
                )
                > 0 else 1
            )

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> PythonObject:
        """Reset environment and return initial observation as numpy array."""
        var result = self.env.reset()
        var obs = result[0]

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

        return obs

    fn step(
        mut self, action: PythonObject
    ) raises -> Tuple[PythonObject, Float64, Bool]:
        """Take action and return (observation, reward, done).

        Args:
            action: Action to take (int for discrete, numpy array for continuous)
        """
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (obs, reward, self.done)

    fn step_discrete(
        mut self, action: Int
    ) raises -> Tuple[PythonObject, Float64, Bool]:
        """Take discrete action (convenience method)."""
        return self.step(PythonObject(action))

    fn sample_action(self) raises -> PythonObject:
        """Sample a random action from the action space."""
        return self.env.action_space.sample()

    fn get_obs_as_list(self, obs: PythonObject) raises -> List[Float64]:
        """Convert numpy observation to Mojo List[Float64]."""
        var flat = self.np.asarray(obs).flatten()
        var result = List[Float64]()
        for i in range(self.obs_dim):
            result.append(Float64(flat[i]))
        return result^

    fn close(mut self) raises:
        """Close the environment."""
        _ = self.env.close()

    fn render(self) raises:
        """Render the environment."""
        _ = self.env.render()

    fn get_info(self) -> String:
        """Return environment info string."""
        return (
            "Env: "
            + self.env_name
            + " | Obs dim: "
            + String(self.obs_dim)
            + " | Action dim: "
            + String(self.action_dim)
            + " | Discrete action: "
            + String(self.is_discrete_action)
        )
