"""Gymnasium MuJoCo environments wrapper with trait conformance.

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

These wrappers implement BoxContinuousActionEnv and RenderableEnv traits.
Note: These require pip install "gymnasium[mujoco]" or mujoco-py
"""

from python import Python, PythonObject
from core import State, Action, BoxContinuousActionEnv, RenderableEnv


# ============================================================================
# MuJoCo State and Action types
# ============================================================================


@fieldwise_init
struct GymMuJoCoState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for MuJoCo environments: placeholder index for trait conformance.
    """

    var index: Int

    fn __init__(out self, *, copy: Self):
        self.index = copy.index

    fn __init__(out self, *, deinit take: Self):
        self.index = take.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymMuJoCoAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for MuJoCo environments: placeholder index for trait conformance.
    """

    var index: Int

    fn __init__(out self, *, copy: Self):
        self.index = copy.index

    fn __init__(out self, *, deinit take: Self):
        self.index = take.index


# ============================================================================
# GymMuJoCoEnv - implements BoxContinuousActionEnv & RenderableEnv
# ============================================================================


struct GymMuJoCoEnv(BoxContinuousActionEnv & RenderableEnv):
    """Generic wrapper for MuJoCo environments.

    Since MuJoCo envs have varying observation/action dimensions,
    this uses dynamic Lists instead of fixed SIMD types.

    Implements BoxContinuousActionEnv and RenderableEnv traits.
    Rendering is delegated to Gymnasium's own renderer (pass render_mode="human"
    at construction time to enable the Gymnasium window).
    """

    # Type aliases for trait conformance
    comptime StateType = GymMuJoCoState
    comptime ActionType = GymMuJoCoAction
    comptime dtype: DType = DType.float64

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var env_name: String

    var _obs_dim: Int
    var _action_dim: Int
    var _action_low: Float64
    var _action_high: Float64

    var current_obs: List[Float64]
    var current_obs_4d: SIMD[DType.float64, 4]  # For trait conformance
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    # RenderableEnv state
    var _render_initialized: Bool

    fn __init__(out self, env_name: String, render_mode: String = "") raises:
        """Initialize a MuJoCo environment.

        Args:
            env_name: Environment name (e.g., "HalfCheetah-v5", "Ant-v5").
            render_mode: "human" for visual rendering.
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

        # Get observation space info
        var obs_space = self.env.observation_space
        var obs_shape = obs_space.shape
        self._obs_dim = Int(py=obs_shape[0])

        # Get action space info
        var act_space = self.env.action_space
        var act_shape = act_space.shape
        self._action_dim = Int(py=act_shape[0])
        self._action_low = Float64(py=act_space.low[0])
        self._action_high = Float64(py=act_space.high[0])

        # Initialize observation storage
        self.current_obs = List[Float64]()
        for _ in range(self._obs_dim):
            self.current_obs.append(0.0)
        self.current_obs_4d = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self._render_initialized = False

    # ========================================================================
    # Env base trait methods
    # ========================================================================

    fn reset(mut self) -> GymMuJoCoState:
        """Reset environment and return state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            for i in range(self._obs_dim):
                self.current_obs[i] = Float64(py=obs[i])
            # Copy first 4 elements for trait conformance
            var min_dim = 4 if self._obs_dim >= 4 else self._obs_dim
            for i in range(min_dim):
                self.current_obs_4d[i] = self.current_obs[i]
        except:
            for i in range(self._obs_dim):
                self.current_obs[i] = 0.0
            self.current_obs_4d = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymMuJoCoState(index=0)

    fn step(
        mut self, action: GymMuJoCoAction, verbose: Bool = False
    ) -> Tuple[GymMuJoCoState, Float64, Bool]:
        """Take action (placeholder - use step_continuous for actual control).
        """
        return (GymMuJoCoState(index=0), 0.0, self.done)

    fn get_state(self) -> GymMuJoCoState:
        """Return current state."""
        return GymMuJoCoState(index=0)

    # ========================================================================
    # ContinuousStateEnv trait methods
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return first 4 dims of observation for trait conformance."""
        return self.current_obs_4d

    fn get_obs_list(self) -> List[Float64]:
        """Return current continuous observation as List (trait method)."""
        var obs = List[Float64](capacity=self._obs_dim)
        for i in range(self._obs_dim):
            obs.append(self.current_obs[i])
        return obs^

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return continuous observation."""
        _ = self.reset()
        return self.current_obs_4d

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return continuous observation as List (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension."""
        return self._obs_dim

    # ========================================================================
    # ContinuousActionEnv trait methods
    # ========================================================================

    fn action_dim(self) -> Int:
        """Return action dimension."""
        return self._action_dim

    fn action_low(self) -> Float64:
        """Return action lower bound."""
        return self._action_low

    fn action_high(self) -> Float64:
        """Return action upper bound."""
        return self._action_high

    # ========================================================================
    # BoxContinuousActionEnv trait methods
    # ========================================================================

    fn step_continuous(
        mut self, action: Float64
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take 1D continuous action (trait method).

        Note: MuJoCo envs have multi-dimensional actions.
        This only uses the first action. Use step_continuous_vec for full control.
        """
        var action_list = List[Float64]()
        action_list.append(action)
        for _ in range(1, self._action_dim):
            action_list.append(0.0)
        return self.step_continuous_vec(action_list^)

    fn step_continuous_vec[
        DTYPE: DType
    ](mut self, action: List[Scalar[DTYPE]], verbose: Bool = False) -> Tuple[
        List[Scalar[DTYPE]], Scalar[DTYPE], Bool
    ]:
        """Take multi-dimensional continuous action (trait method).

        Args:
            action: List of action values (length = action_dim).
            verbose: Whether to print verbose output (default: False).

        Returns:
            Tuple of (observation_list, reward, done).
        """
        var result = self.step_with_list(action)
        var obs_list = self.get_obs_list()
        var new_obs_list = List[Scalar[DTYPE]]()
        for i in range(self._obs_dim):
            new_obs_list.append(Scalar[DTYPE](obs_list[i]))
        return (new_obs_list^, result[1], result[2])

    # ========================================================================
    # RenderableEnv trait methods
    # ========================================================================

    fn init_renderer(mut self) raises -> Bool:
        """Mark renderer as initialized (Gymnasium renders via its own window).
        """
        self._render_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        """Render via Gymnasium's built-in renderer."""
        if not self._render_initialized:
            return
        try:
            _ = self.env.render()
        except:
            pass

    fn close_renderer(mut self) raises -> None:
        """Close the Gymnasium environment (and its render window)."""
        if not self._render_initialized:
            return
        try:
            _ = self.env.close()
        except:
            pass
        self._render_initialized = False

    fn is_renderer_open(self) -> Bool:
        """Return True if renderer has been initialized."""
        return self._render_initialized

    fn check_renderer_quit(mut self) -> Bool:
        """Gymnasium manages its own window; always returns False."""
        return False

    fn renderer_delay(self, ms: Int) -> None:
        """No-op: Gymnasium controls its own frame rate."""
        pass

    fn renderer_is_paused(self) -> Bool:
        return False

    fn renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # Additional methods - continuous control
    # ========================================================================

    fn step_with_list[
        DTYPE: DType
    ](mut self, action: List[Scalar[DTYPE]]) -> Tuple[
        SIMD[DTYPE, 4], Scalar[DTYPE], Bool
    ]:
        """Take continuous action as List[Float64].

        Args:
            action: List of action values (length = action_dim).
        """
        var reward: Scalar[DTYPE] = 0.0
        try:
            # Convert Mojo list to numpy array
            var builtins = Python.import_module("builtins")
            var py_list = builtins.list()
            for i in range(len(action)):
                _ = py_list.append(action[i])
            var np_action = self.np.array(py_list)

            var result = self.env.step(np_action)
            var obs = result[0]
            reward = Scalar[DTYPE](py=result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            for i in range(self._obs_dim):
                self.current_obs[i] = Float64(py=obs[i])
            var min_dim = 4 if self._obs_dim >= 4 else self._obs_dim
            for i in range(min_dim):
                self.current_obs_4d[i] = self.current_obs[i]

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += Float64(reward)
        self.episode_length += 1

        var new_obs_4d = SIMD[DTYPE, 4](0.0)
        for i in range(4):
            new_obs_4d[i] = Scalar[DTYPE](self.current_obs_4d[i])
        return (new_obs_4d, reward, self.done)

    fn sample_action(mut self, out action: List[Float64]):
        """Sample random action from action space."""
        action = List[Float64]()
        try:
            var np_action = self.env.action_space.sample()
            for i in range(self._action_dim):
                action.append(Float64(py=np_action[i]))
        except:
            for _ in range(self._action_dim):
                action.append(0.0)

    fn get_full_obs(self, out obs: List[Float64]):
        """Copy full observation into output list."""
        obs = List[Float64]()
        for i in range(self._obs_dim):
            obs.append(self.current_obs[i])

    fn close(mut self):
        """Close the environment."""
        try:
            _ = self.env.close()
        except:
            pass

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn get_info(self) -> String:
        """Return environment info string."""
        return (
            "MuJoCo Env: "
            + self.env_name
            + " | Obs dim: "
            + String(self._obs_dim)
            + " | Action dim: "
            + String(self._action_dim)
        )


# ============================================================================
# Convenience factory functions for common MuJoCo environments
# ============================================================================


fn make_half_cheetah(render_mode: String = "") raises -> GymMuJoCoEnv:
    """HalfCheetah-v5: 2D cheetah running task.

    Obs: 17 (positions and velocities)
    Act: 6 (joint torques)
    Reward: Forward velocity - control cost.
    """
    return GymMuJoCoEnv("HalfCheetah-v5", render_mode)


fn make_ant(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Ant-v5: 4-legged robot running task.

    Obs: 27 (positions and velocities)
    Act: 8 (joint torques)
    Reward: Forward velocity - control cost + survival bonus.
    """
    return GymMuJoCoEnv("Ant-v5", render_mode)


fn make_humanoid(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Humanoid-v5: 3D humanoid walking/running task.

    Obs: 376 (positions, velocities, and COM-based features)
    Act: 17 (joint torques)
    Reward: Forward velocity - control cost + survival bonus.
    """
    return GymMuJoCoEnv("Humanoid-v5", render_mode)


fn make_walker2d(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Walker2d-v5: 2D biped walking task.

    Obs: 17 (positions and velocities)
    Act: 6 (joint torques)
    Reward: Forward velocity - control cost + survival bonus.
    """
    return GymMuJoCoEnv("Walker2d-v5", render_mode)


fn make_hopper(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Hopper-v5: 1-legged hopping task.

    Obs: 11 (positions and velocities)
    Act: 3 (joint torques)
    Reward: Forward velocity - control cost + survival bonus.
    """
    return GymMuJoCoEnv("Hopper-v5", render_mode)


fn make_swimmer(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Swimmer-v5: 3-link swimming task.

    Obs: 8 (positions and velocities)
    Act: 2 (joint torques)
    Reward: Forward velocity - control cost.
    """
    return GymMuJoCoEnv("Swimmer-v5", render_mode)


fn make_inverted_pendulum(render_mode: String = "") raises -> GymMuJoCoEnv:
    """InvertedPendulum-v5: Balance a pole on a cart.

    Obs: 4 (cart position/velocity, pole angle/velocity)
    Act: 1 (cart force)
    Reward: +1 for each step pole stays upright.
    """
    return GymMuJoCoEnv("InvertedPendulum-v5", render_mode)


fn make_inverted_double_pendulum(
    render_mode: String = "",
) raises -> GymMuJoCoEnv:
    """InvertedDoublePendulum-v5: Balance a double pendulum on a cart.

    Obs: 11 (positions and velocities)
    Act: 1 (cart force)
    Reward: Based on distance from upright position.
    """
    return GymMuJoCoEnv("InvertedDoublePendulum-v5", render_mode)


fn make_reacher(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Reacher-v5: Reach a target with a 2-link arm.

    Obs: 11 (arm positions/velocities + target position)
    Act: 2 (joint torques)
    Reward: Negative distance to target - control cost.
    """
    return GymMuJoCoEnv("Reacher-v5", render_mode)


fn make_pusher(render_mode: String = "") raises -> GymMuJoCoEnv:
    """Pusher-v5: Push an object to a target position.

    Obs: 23 (arm + object + target positions/velocities)
    Act: 7 (joint torques)
    Reward: Negative distance (object to target) - control cost.
    """
    return GymMuJoCoEnv("Pusher-v5", render_mode)
