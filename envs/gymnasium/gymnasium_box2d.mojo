"""Gymnasium Box2D environments wrapper with trait conformance.

Box2D environments (require `pip install gymnasium[box2d]`):
- LunarLander-v3: Land a spacecraft (discrete or continuous)
- BipedalWalker-v3: Walk with a 2D biped robot (continuous)
- CarRacing-v3: Drive a car around a track (continuous)

These use the Box2D physics engine for 2D rigid body simulation.
All implement appropriate environment traits for generic training.
"""

from python import Python, PythonObject
from core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    DiscreteEnv,
)


# ============================================================================
# LunarLander State and Action types
# ============================================================================


@fieldwise_init
struct GymLunarLanderState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for LunarLander: discretized state index."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymLunarLanderAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for LunarLander: 0=nothing, 1=left, 2=main, 3=right."""

    var action: Int

    fn __copyinit__(out self, existing: Self):
        self.action = existing.action

    fn __moveinit__(out self, deinit existing: Self):
        self.action = existing.action

    @staticmethod
    fn nothing() -> Self:
        return Self(action=0)

    @staticmethod
    fn fire_left() -> Self:
        return Self(action=1)

    @staticmethod
    fn fire_main() -> Self:
        return Self(action=2)

    @staticmethod
    fn fire_right() -> Self:
        return Self(action=3)


# ============================================================================
# BipedalWalker State and Action types
# ============================================================================


@fieldwise_init
struct GymBipedalWalkerState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for BipedalWalker: discretized state index (for tabular fallback).
    """

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymBipedalWalkerAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for BipedalWalker: 4 continuous torques (stored as index for trait).
    """

    var index: Int  # Placeholder for trait conformance

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index


# ============================================================================
# CarRacing State and Action types
# ============================================================================


@fieldwise_init
struct GymCarRacingState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for CarRacing: discretized state index (for tabular fallback)."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymCarRacingAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for CarRacing: steering/gas/brake (stored as index for trait)."""

    var index: Int  # Placeholder for trait conformance

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index


# ============================================================================
# GymLunarLanderEnv - implements BoxDiscreteActionEnv & DiscreteEnv
# ============================================================================


struct GymLunarLanderEnv(BoxDiscreteActionEnv & DiscreteEnv):
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

    Implements BoxDiscreteActionEnv & DiscreteEnv traits for generic training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymLunarLanderState
    comptime ActionType = GymLunarLanderAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 8]
    var current_obs_4d: SIMD[DType.float64, 4]  # For trait conformance
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    # Discretization settings
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 10, render_mode: String = "") raises:
        """Initialize LunarLander.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            render_mode: "human" for visual rendering.
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")

        if render_mode == "human":
            self.env = self.gym.make(
                "LunarLander-v3", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("LunarLander-v3")

        self.current_obs = SIMD[DType.float64, 8](0.0)
        self.current_obs_4d = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymLunarLanderState:
        """Reset environment and return discretized initial state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            for i in range(8):
                self.current_obs[i] = Float64(obs[i])
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]
        except:
            self.current_obs = SIMD[DType.float64, 8](0.0)
            self.current_obs_4d = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymLunarLanderState(index=self._discretize_obs())

    fn step(
        mut self, action: GymLunarLanderAction
    ) -> Tuple[GymLunarLanderState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.action)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            for i in range(8):
                self.current_obs[i] = Float64(obs[i])
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (
            GymLunarLanderState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn get_state(self) -> GymLunarLanderState:
        """Return current discretized state."""
        return GymLunarLanderState(index=self._discretize_obs())

    fn state_to_index(self, state: GymLunarLanderState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymLunarLanderAction:
        """Create action from index."""
        return GymLunarLanderAction(action=action_idx)

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        # Using only first 6 dimensions for discretization (not leg contacts)
        var total = 1
        for _ in range(6):
            total *= self.num_bins
        return total

    fn num_actions(self) -> Int:
        """Return number of actions (4)."""
        return 4

    # ========================================================================
    # BoxDiscreteActionEnv (ContinuousStateEnv) trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Float64]:
        """Return current observation as List for trait conformance."""
        var obs = List[Float64](capacity=8)
        for i in range(8):
            obs.append(self.current_obs[i])
        return obs^

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return continuous observation as List."""
        _ = self.reset()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension (8)."""
        return 8

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take action and return (continuous_obs, reward, done)."""
        var result = self.step(GymLunarLanderAction(action=action))
        return (self.get_obs_list(), result[1], result[2])

    # SIMD convenience methods (not required by trait)
    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return first 4 dims of observation as SIMD."""
        return self.current_obs_4d

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return SIMD observation."""
        _ = self.reset()
        return self.current_obs_4d

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (SIMD_obs, reward, done)."""
        var result = self.step(GymLunarLanderAction(action=action))
        return (self.current_obs_4d, result[1], result[2])

    # ========================================================================
    # Additional methods - full observation access
    # ========================================================================

    fn get_full_obs(self) -> SIMD[DType.float64, 8]:
        """Return full 8D observation."""
        return self.current_obs

    fn render(mut self):
        """Render the environment."""
        try:
            _ = self.env.render()
        except:
            pass

    fn close(mut self):
        """Close the environment."""
        try:
            _ = self.env.close()
        except:
            pass

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn _discretize_obs(self) -> Int:
        """Discretize current observation into a state index."""
        # Approximate bounds for each dimension (using first 6)
        var bounds_low = SIMD[DType.float64, 8](
            -1.5, -1.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0
        )
        var bounds_high = SIMD[DType.float64, 8](
            1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0
        )

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var index = 0
        var multiplier = 1
        for i in range(6):  # Only first 6 dims
            var b = bin_value(
                self.current_obs[i],
                bounds_low[i],
                bounds_high[i],
                self.num_bins,
            )
            index += b * multiplier
            multiplier *= self.num_bins

        return index


# ============================================================================
# GymBipedalWalkerEnv - implements BoxContinuousActionEnv
# ============================================================================


struct GymBipedalWalkerEnv(BoxContinuousActionEnv):
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

    Implements BoxContinuousActionEnv trait for continuous action algorithms.
    """

    # Type aliases for trait conformance
    comptime StateType = GymBipedalWalkerState
    comptime ActionType = GymBipedalWalkerAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: List[Float64]
    var current_obs_4d: SIMD[DType.float64, 4]  # For trait conformance
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var hardcore: Bool

    fn __init__(
        out self, hardcore: Bool = False, render_mode: String = ""
    ) raises:
        """Initialize BipedalWalker.

        Args:
            hardcore: If True, use hardcore mode with obstacles.
            render_mode: "human" for visual rendering.
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
        self.current_obs_4d = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    # ========================================================================
    # Env base trait methods
    # ========================================================================

    fn reset(mut self) -> GymBipedalWalkerState:
        """Reset environment and return state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            for i in range(24):
                self.current_obs[i] = Float64(obs[i])
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]
        except:
            for i in range(24):
                self.current_obs[i] = 0.0
            self.current_obs_4d = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymBipedalWalkerState(index=0)

    fn step(
        mut self, action: GymBipedalWalkerAction
    ) -> Tuple[GymBipedalWalkerState, Float64, Bool]:
        """Take action (placeholder - use step_continuous for actual control).
        """
        # This is a placeholder - real usage should use step_continuous
        return (GymBipedalWalkerState(index=0), 0.0, self.done)

    fn get_state(self) -> GymBipedalWalkerState:
        """Return current state."""
        return GymBipedalWalkerState(index=0)

    # ========================================================================
    # ContinuousStateEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Float64]:
        """Return current observation as List for trait conformance."""
        var obs = List[Float64](capacity=24)
        for i in range(24):
            obs.append(self.current_obs[i])
        return obs^

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return continuous observation as List."""
        _ = self.reset()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension (24)."""
        return 24

    # ========================================================================
    # ContinuousActionEnv trait methods
    # ========================================================================

    fn action_dim(self) -> Int:
        """Return action dimension (4)."""
        return 4

    fn action_low(self) -> Float64:
        """Return action lower bound."""
        return -1.0

    fn action_high(self) -> Float64:
        """Return action upper bound."""
        return 1.0

    # ========================================================================
    # BoxContinuousActionEnv trait methods
    # ========================================================================

    fn step_continuous(
        mut self, action: Float64
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take 1D continuous action (trait method).

        Note: BipedalWalker has 4D actions. This only controls the first joint.
        Use step_continuous_vec for full 4D control.
        """
        var result = self.step_continuous_4d(action, 0.0, 0.0, 0.0)
        return (self.get_obs_list(), result[1], result[2])

    fn step_continuous_vec(
        mut self, action: List[Float64]
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take multi-dimensional continuous action (trait method).

        Args:
            action: List of 4 action values [hip1, knee1, hip2, knee2].

        Returns:
            Tuple of (observation_list, reward, done).
        """
        var hip1 = action[0] if len(action) > 0 else 0.0
        var knee1 = action[1] if len(action) > 1 else 0.0
        var hip2 = action[2] if len(action) > 2 else 0.0
        var knee2 = action[3] if len(action) > 3 else 0.0
        var result = self.step_continuous_4d(hip1, knee1, hip2, knee2)
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # SIMD convenience methods (not required by trait)
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return first 4 dims of observation as SIMD."""
        return self.current_obs_4d

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return SIMD observation."""
        _ = self.reset()
        return self.current_obs_4d

    # ========================================================================
    # Additional methods - full 4D continuous control
    # ========================================================================

    fn step_continuous_4d(
        mut self, hip1: Float64, knee1: Float64, hip2: Float64, knee2: Float64
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take 4D continuous action [hip1, knee1, hip2, knee2]."""
        var reward: Float64 = 0.0
        try:
            var builtins = Python.import_module("builtins")
            var py_list = builtins.list()
            _ = py_list.append(hip1)
            _ = py_list.append(knee1)
            _ = py_list.append(hip2)
            _ = py_list.append(knee2)
            var action = self.np.array(py_list)
            var result = self.env.step(action)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            for i in range(24):
                self.current_obs[i] = Float64(obs[i])
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_obs_4d, reward, self.done)

    fn get_full_obs(self, out obs: List[Float64]):
        """Copy full 24D observation into output list."""
        obs = List[Float64]()
        for i in range(24):
            obs.append(self.current_obs[i])

    fn render(mut self):
        """Render the environment."""
        try:
            _ = self.env.render()
        except:
            pass

    fn close(mut self):
        """Close the environment."""
        try:
            _ = self.env.close()
        except:
            pass

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done


# ============================================================================
# GymCarRacingEnv - implements BoxContinuousActionEnv
# ============================================================================


struct GymCarRacingEnv(BoxContinuousActionEnv):
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

    Implements BoxContinuousActionEnv trait (uses continuous actions by default).
    """

    # Type aliases for trait conformance
    comptime StateType = GymCarRacingState
    comptime ActionType = GymCarRacingAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs_4d: SIMD[
        DType.float64, 4
    ]  # For trait conformance (placeholder)
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var is_continuous: Bool

    fn __init__(
        out self, continuous: Bool = True, render_mode: String = ""
    ) raises:
        """Initialize CarRacing.

        Args:
            continuous: If True, use continuous action space.
            render_mode: "human" for visual rendering.
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

        self.current_obs_4d = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    # ========================================================================
    # Env base trait methods
    # ========================================================================

    fn reset(mut self) -> GymCarRacingState:
        """Reset environment and return state."""
        try:
            _ = self.env.reset()
        except:
            pass

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymCarRacingState(index=0)

    fn step(
        mut self, action: GymCarRacingAction
    ) -> Tuple[GymCarRacingState, Float64, Bool]:
        """Take action (placeholder - use step_continuous for actual control).
        """
        return (GymCarRacingState(index=0), 0.0, self.done)

    fn get_state(self) -> GymCarRacingState:
        """Return current state."""
        return GymCarRacingState(index=0)

    # ========================================================================
    # ContinuousStateEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Float64]:
        """Return placeholder observation (CarRacing uses images).

        Note: CarRacing uses 96x96x3 RGB images. This returns a 4D placeholder.
        Use get_image_obs() for actual image observations.
        """
        var obs = List[Float64](capacity=4)
        for i in range(4):
            obs.append(self.current_obs_4d[i])
        return obs^

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return placeholder observation."""
        _ = self.reset()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension (96*96*3 = 27648 for images)."""
        return 96 * 96 * 3

    # ========================================================================
    # ContinuousActionEnv trait methods
    # ========================================================================

    fn action_dim(self) -> Int:
        """Return action dimension (3)."""
        return 3

    fn action_low(self) -> Float64:
        """Return action lower bound."""
        return -1.0

    fn action_high(self) -> Float64:
        """Return action upper bound."""
        return 1.0

    # ========================================================================
    # BoxContinuousActionEnv trait methods
    # ========================================================================

    fn step_continuous(
        mut self, action: Float64
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take 1D continuous action (trait method).

        Note: CarRacing has 3D actions. This only controls steering.
        Use step_continuous_vec for full 3D control.
        """
        var result = self.step_continuous_3d(action, 0.0, 0.0)
        return (self.get_obs_list(), result[1], result[2])

    fn step_continuous_vec(
        mut self, action: List[Float64]
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take multi-dimensional continuous action (trait method).

        Args:
            action: List of 3 action values [steering, gas, brake].

        Returns:
            Tuple of (observation_list, reward, done).
        """
        var steering = action[0] if len(action) > 0 else 0.0
        var gas = action[1] if len(action) > 1 else 0.0
        var brake = action[2] if len(action) > 2 else 0.0
        var result = self.step_continuous_3d(steering, gas, brake)
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # SIMD convenience methods (not required by trait)
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return placeholder observation as SIMD."""
        return self.current_obs_4d

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return SIMD observation."""
        _ = self.reset()
        return self.current_obs_4d

    # ========================================================================
    # Additional methods - full 3D continuous control
    # ========================================================================

    fn step_continuous_3d(
        mut self, steering: Float64, gas: Float64, brake: Float64
    ) -> Tuple[PythonObject, Float64, Bool]:
        """Take 3D continuous action [steering, gas, brake]."""
        var reward: Float64 = 0.0
        var obs: PythonObject = PythonObject()
        try:
            var builtins = Python.import_module("builtins")
            var py_list = builtins.list()
            _ = py_list.append(steering)
            _ = py_list.append(gas)
            _ = py_list.append(brake)
            var action = self.np.array(py_list)
            var result = self.env.step(action)
            obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (obs, reward, self.done)

    fn step_discrete(
        mut self, action: Int
    ) -> Tuple[PythonObject, Float64, Bool]:
        """Take discrete action (0-4)."""
        var reward: Float64 = 0.0
        var obs: PythonObject = PythonObject()
        try:
            var result = self.env.step(action)
            obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (obs, reward, self.done)

    fn reset_image(mut self) -> PythonObject:
        """Reset and return image observation (96x96x3 numpy array)."""
        var obs: PythonObject = PythonObject()
        try:
            var result = self.env.reset()
            obs = result[0]
        except:
            pass

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return obs

    fn render(mut self):
        """Render the environment."""
        try:
            _ = self.env.render()
        except:
            pass

    fn close(mut self):
        """Close the environment."""
        try:
            _ = self.env.close()
        except:
            pass

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn obs_shape(self) -> Tuple[Int, Int, Int]:
        """Return image observation shape."""
        return (96, 96, 3)

    fn num_discrete_actions(self) -> Int:
        """Return number of discrete actions (5)."""
        return 5
