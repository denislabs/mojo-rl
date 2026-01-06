"""Gymnasium Classic Control environments wrapper with trait conformance.

Classic Control environments:
- CartPole-v1: Balance pole on a cart (discrete)
- MountainCar-v0: Drive car up a mountain (discrete)
- MountainCarContinuous-v0: Drive car up mountain (continuous)
- Pendulum-v1: Swing up and balance pendulum (continuous)
- Acrobot-v1: Swing up double pendulum (discrete)

All use physics simulations with continuous observation spaces.
These wrappers conform to the core environment traits for use with generic agents.
"""

from python import Python, PythonObject
from core import (
    State,
    Action,
    DiscreteEnv,
    ClassicControlEnv,
    ContinuousControlEnv,
    TileCoding,
    PolynomialFeatures,
)


# ============================================================================
# CartPole State and Action types
# ============================================================================


@fieldwise_init
struct GymCartPoleState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for CartPole: discretized state index."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymCartPoleAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for CartPole: 0 (push left), 1 (push right)."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        return Self(direction=0)

    @staticmethod
    fn right() -> Self:
        return Self(direction=1)


# ============================================================================
# MountainCar State and Action types
# ============================================================================


@fieldwise_init
struct GymMountainCarState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for MountainCar: discretized state index."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymMountainCarAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for MountainCar: 0 (left), 1 (no push), 2 (right)."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        return Self(direction=0)

    @staticmethod
    fn neutral() -> Self:
        return Self(direction=1)

    @staticmethod
    fn right() -> Self:
        return Self(direction=2)


# ============================================================================
# Acrobot State and Action types
# ============================================================================


@fieldwise_init
struct GymAcrobotState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Acrobot: discretized state index."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymAcrobotAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Acrobot: 0 (-1 torque), 1 (0 torque), 2 (+1 torque)."""

    var torque: Int

    fn __copyinit__(out self, existing: Self):
        self.torque = existing.torque

    fn __moveinit__(out self, deinit existing: Self):
        self.torque = existing.torque


# ============================================================================
# GymCartPoleEnv - implements ClassicControlEnv & DiscreteEnv
# ============================================================================


struct GymCartPoleEnv(ClassicControlEnv & DiscreteEnv):
    """CartPole-v1 environment via Gymnasium Python bindings.

    Observation space: Box(4,) - [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    Action space: Discrete(2) - 0: push left, 1: push right

    Episode terminates when:
    - Pole angle > ±12°
    - Cart position > ±2.4
    - Episode length > 500 steps

    Implements DiscreteEnv and ClassicControlEnv traits for generic training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymCartPoleState
    comptime ActionType = GymCartPoleAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 4]
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    # Discretization settings
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 10, render_mode: String = "") raises:
        """Initialize CartPole environment from Gymnasium.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            render_mode: "human" for visual rendering, "" for no rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "CartPole-v1", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("CartPole-v1")
        self.current_obs = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymCartPoleState:
        """Reset environment and return discretized initial state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])
            self.current_obs[2] = Float64(obs[2])
            self.current_obs[3] = Float64(obs[3])
        except:
            self.current_obs = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymCartPoleState(index=self._discretize_obs())

    fn step(
        mut self, action: GymCartPoleAction
    ) -> Tuple[GymCartPoleState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.direction)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])
            self.current_obs[2] = Float64(obs[2])
            self.current_obs[3] = Float64(obs[3])

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (
            GymCartPoleState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn get_state(self) -> GymCartPoleState:
        """Return current discretized state."""
        return GymCartPoleState(index=self._discretize_obs())

    fn state_to_index(self, state: GymCartPoleState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymCartPoleAction:
        """Create action from index."""
        return GymCartPoleAction(direction=action_idx)

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins * self.num_bins * self.num_bins

    fn num_actions(self) -> Int:
        """Return number of actions (2)."""
        return 2

    # ========================================================================
    # ClassicControlEnv (ContinuousStateEnv) trait methods
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation."""
        return self.current_obs

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return continuous observation."""
        _ = self.reset()
        return self.current_obs

    fn obs_dim(self) -> Int:
        """Return observation dimension (4)."""
        return 4

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (continuous_obs, reward, done)."""
        var result = self.step(GymCartPoleAction(direction=action))
        return (self.current_obs, result[1], result[2])

    # ========================================================================
    # Additional methods
    # ========================================================================

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
        """Discretize current continuous observation into a single state index.
        """
        var cart_pos_low: Float64 = -2.4
        var cart_pos_high: Float64 = 2.4
        var cart_vel_low: Float64 = -3.0
        var cart_vel_high: Float64 = 3.0
        var pole_angle_low: Float64 = -0.21
        var pole_angle_high: Float64 = 0.21
        var pole_vel_low: Float64 = -3.0
        var pole_vel_high: Float64 = 3.0

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(
            self.current_obs[0], cart_pos_low, cart_pos_high, self.num_bins
        )
        var b1 = bin_value(
            self.current_obs[1], cart_vel_low, cart_vel_high, self.num_bins
        )
        var b2 = bin_value(
            self.current_obs[2], pole_angle_low, pole_angle_high, self.num_bins
        )
        var b3 = bin_value(
            self.current_obs[3], pole_vel_low, pole_vel_high, self.num_bins
        )

        return (
            (b0 * self.num_bins + b1) * self.num_bins + b2
        ) * self.num_bins + b3

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for CartPole environment."""
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        var state_low = List[Float64]()
        state_low.append(-2.5)
        state_low.append(-3.5)
        state_low.append(-0.25)
        state_low.append(-3.5)

        var state_high = List[Float64]()
        state_high.append(2.5)
        state_high.append(3.5)
        state_high.append(0.25)
        state_high.append(3.5)

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 2) -> PolynomialFeatures:
        """Create polynomial features for CartPole (4D state) with normalization.
        """
        var state_low = List[Float64]()
        state_low.append(-2.4)
        state_low.append(-3.0)
        state_low.append(-0.21)
        state_low.append(-3.0)

        var state_high = List[Float64]()
        state_high.append(2.4)
        state_high.append(3.0)
        state_high.append(0.21)
        state_high.append(3.0)

        return PolynomialFeatures(
            state_dim=4,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )


# ============================================================================
# GymMountainCarEnv - implements ClassicControlEnv & DiscreteEnv
# ============================================================================


struct GymMountainCarEnv(ClassicControlEnv & DiscreteEnv):
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

    Implements DiscreteEnv and ClassicControlEnv traits for generic training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymMountainCarState
    comptime ActionType = GymMountainCarAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[
        DType.float64, 4
    ]  # Using 4 for trait conformance, only 2 used
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    # Discretization settings
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 20, render_mode: String = "") raises:
        """Initialize MountainCar environment.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "MountainCar-v0", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("MountainCar-v0")
        self.current_obs = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymMountainCarState:
        """Reset environment and return discretized initial state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])
        except:
            self.current_obs = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymMountainCarState(index=self._discretize_obs())

    fn step(
        mut self, action: GymMountainCarAction
    ) -> Tuple[GymMountainCarState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.direction)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (
            GymMountainCarState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn get_state(self) -> GymMountainCarState:
        """Return current discretized state."""
        return GymMountainCarState(index=self._discretize_obs())

    fn state_to_index(self, state: GymMountainCarState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymMountainCarAction:
        """Create action from index."""
        return GymMountainCarAction(direction=action_idx)

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins

    fn num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    # ========================================================================
    # ClassicControlEnv (ContinuousStateEnv) trait methods
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation (only first 2 elements used).
        """
        return self.current_obs

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return continuous observation."""
        _ = self.reset()
        return self.current_obs

    fn obs_dim(self) -> Int:
        """Return observation dimension (2)."""
        return 2

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (continuous_obs, reward, done)."""
        var result = self.step(GymMountainCarAction(direction=action))
        return (self.current_obs, result[1], result[2])

    # ========================================================================
    # Additional methods
    # ========================================================================

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
        var pos_low: Float64 = -1.2
        var pos_high: Float64 = 0.6
        var vel_low: Float64 = -0.07
        var vel_high: Float64 = 0.07

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(
            self.current_obs[0], pos_low, pos_high, self.num_bins
        )
        var b1 = bin_value(
            self.current_obs[1], vel_low, vel_high, self.num_bins
        )

        return b0 * self.num_bins + b1

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for MountainCar environment."""
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        var state_low = List[Float64]()
        state_low.append(-1.2)
        state_low.append(-0.07)

        var state_high = List[Float64]()
        state_high.append(0.6)
        state_high.append(0.07)

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 3) -> PolynomialFeatures:
        """Create polynomial features for MountainCar (2D state)."""
        var state_low = List[Float64]()
        state_low.append(-1.2)
        state_low.append(-0.07)

        var state_high = List[Float64]()
        state_high.append(0.6)
        state_high.append(0.07)

        return PolynomialFeatures(
            state_dim=2,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )


# ============================================================================
# GymAcrobotEnv - implements ClassicControlEnv & DiscreteEnv
# ============================================================================


struct GymAcrobotEnv(ClassicControlEnv & DiscreteEnv):
    """Acrobot-v1: Swing up a two-link robot arm.

    Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] - Box(6,)

    Actions: Discrete(3)
        0: Apply -1 torque to joint
        1: Apply 0 torque
        2: Apply +1 torque

    Reward: -1 for each step until goal
    Goal: Swing the tip above the base (height threshold)

    Episode ends: Tip above threshold or 500 steps

    Implements DiscreteEnv and ClassicControlEnv traits for generic training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymAcrobotState
    comptime ActionType = GymAcrobotAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[
        DType.float64, 8
    ]  # Using 8 for alignment, only 6 used
    var current_obs_4d: SIMD[DType.float64, 4]  # For trait conformance
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    # Discretization settings
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 6, render_mode: String = "") raises:
        """Initialize Acrobot environment.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")
        self.np = Python.import_module("numpy")
        if render_mode == "human":
            self.env = self.gym.make(
                "Acrobot-v1", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("Acrobot-v1")
        self.current_obs = SIMD[DType.float64, 8](0.0)
        self.current_obs_4d = SIMD[DType.float64, 4](0.0)
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymAcrobotState:
        """Reset environment and return discretized initial state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            for i in range(6):
                self.current_obs[i] = Float64(obs[i])
            # Copy first 4 to 4d version for trait
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]
        except:
            self.current_obs = SIMD[DType.float64, 8](0.0)
            self.current_obs_4d = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymAcrobotState(index=self._discretize_obs())

    fn step(
        mut self, action: GymAcrobotAction
    ) -> Tuple[GymAcrobotState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.torque)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            for i in range(6):
                self.current_obs[i] = Float64(obs[i])
            for i in range(4):
                self.current_obs_4d[i] = self.current_obs[i]

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (
            GymAcrobotState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn get_state(self) -> GymAcrobotState:
        """Return current discretized state."""
        return GymAcrobotState(index=self._discretize_obs())

    fn state_to_index(self, state: GymAcrobotState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymAcrobotAction:
        """Create action from index."""
        return GymAcrobotAction(torque=action_idx)

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        var total = 1
        for _ in range(6):
            total *= self.num_bins
        return total

    fn num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    # ========================================================================
    # ClassicControlEnv (ContinuousStateEnv) trait methods
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return first 4 dims of observation for trait conformance."""
        return self.current_obs_4d

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return continuous observation."""
        _ = self.reset()
        return self.current_obs_4d

    fn obs_dim(self) -> Int:
        """Return observation dimension (6)."""
        return 6

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (continuous_obs, reward, done)."""
        var result = self.step(GymAcrobotAction(torque=action))
        return (self.current_obs_4d, result[1], result[2])

    # ========================================================================
    # Additional methods - full observation access
    # ========================================================================

    fn get_full_obs(self) -> SIMD[DType.float64, 8]:
        """Return full 6D observation (in 8-element SIMD)."""
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
        var bounds_low = SIMD[DType.float64, 8](
            -1.0, -1.0, -1.0, -1.0, -12.566, -28.274, 0.0, 0.0
        )
        var bounds_high = SIMD[DType.float64, 8](
            1.0, 1.0, 1.0, 1.0, 12.566, 28.274, 0.0, 0.0
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
        for i in range(6):
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
# GymPendulumEnv - implements ContinuousControlEnv (continuous actions)
# ============================================================================


@fieldwise_init
struct GymPendulumState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Pendulum: discretized state index (for tabular methods)."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymPendulumAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Pendulum: continuous torque in [-2, 2]."""

    var torque: Float64

    fn __copyinit__(out self, existing: Self):
        self.torque = existing.torque

    fn __moveinit__(out self, deinit existing: Self):
        self.torque = existing.torque


struct GymPendulumEnv(ContinuousControlEnv):
    """Pendulum-v1: Swing up and balance an inverted pendulum.

    Observation: [cos(theta), sin(theta), theta_dot] - Box(3,)

    Actions: Continuous Box(1,) - torque in [-2, 2]

    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    Goal: Keep pendulum upright (theta = 0)

    Episode ends: After 200 steps (no early termination)

    Implements ContinuousControlEnv trait for continuous action algorithms.
    """

    # Type aliases for trait conformance
    comptime StateType = GymPendulumState
    comptime ActionType = GymPendulumAction

    var env: PythonObject
    var gym: PythonObject
    var np: PythonObject
    var current_obs: SIMD[DType.float64, 4]  # Using 4 for trait, only 3 used
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        """Initialize Pendulum environment.

        Args:
            render_mode: "human" for visual rendering
        """
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

    # ========================================================================
    # Env base trait methods
    # ========================================================================

    fn reset(mut self) -> GymPendulumState:
        """Reset environment and return state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])
            self.current_obs[2] = Float64(obs[2])
            self.current_obs[3] = 0.0
        except:
            self.current_obs = SIMD[DType.float64, 4](0.0)

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymPendulumState(index=self._discretize_obs())

    fn step(
        mut self, action: GymPendulumAction
    ) -> Tuple[GymPendulumState, Float64, Bool]:
        """Take continuous action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var builtins = Python.import_module("builtins")
            var py_list = builtins.list()
            _ = py_list.append(action.torque)
            var np_action = self.np.array(py_list)

            var result = self.env.step(np_action)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            self.current_obs[0] = Float64(obs[0])
            self.current_obs[1] = Float64(obs[1])
            self.current_obs[2] = Float64(obs[2])

            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (
            GymPendulumState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn get_state(self) -> GymPendulumState:
        """Return current state."""
        return GymPendulumState(index=self._discretize_obs())

    # ========================================================================
    # ContinuousStateEnv trait methods
    # ========================================================================

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation."""
        return self.current_obs

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return continuous observation."""
        _ = self.reset()
        return self.current_obs

    fn obs_dim(self) -> Int:
        """Return observation dimension (3)."""
        return 3

    # ========================================================================
    # ContinuousActionEnv trait methods
    # ========================================================================

    fn action_dim(self) -> Int:
        """Return action dimension (1)."""
        return 1

    fn action_low(self) -> Float64:
        """Return action lower bound."""
        return -2.0

    fn action_high(self) -> Float64:
        """Return action upper bound."""
        return 2.0

    # ========================================================================
    # Additional methods
    # ========================================================================

    fn step_continuous(
        mut self, torque: Float64
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Convenience method for continuous action step."""
        var result = self.step(GymPendulumAction(torque=torque))
        return (self.current_obs, result[1], result[2])

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
        """Discretize observation for optional tabular use."""
        var num_bins = 10

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(self.current_obs[0], -1.0, 1.0, num_bins)
        var b1 = bin_value(self.current_obs[1], -1.0, 1.0, num_bins)
        var b2 = bin_value(self.current_obs[2], -8.0, 8.0, num_bins)

        return (b0 * num_bins + b1) * num_bins + b2
