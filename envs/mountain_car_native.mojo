"""Native Mojo implementation of MountainCar environment.

Physics based on OpenAI Gym / Gymnasium MountainCar-v0:
https://gymnasium.farama.org/environments/classic_control/mountain_car/

A car is on a one-dimensional track, positioned between two "mountains".
The goal is to drive up the mountain on the right; however, the car's engine
is not strong enough to climb the mountain in a single pass. Therefore,
the only way to succeed is to drive back and forth to build up momentum.
"""

from math import cos
from random import random_float64


struct MountainCarNative:
    """Native Mojo MountainCar environment.

    State: [position, velocity]
    Actions: 0 (push left), 1 (no push), 2 (push right)

    Episode terminates when:
    - Position >= 0.5 (goal reached)
    - Episode length >= 200 steps (timeout)
    """

    # Physical constants (same as Gymnasium)
    var min_position: Float64
    var max_position: Float64
    var max_speed: Float64
    var goal_position: Float64
    var goal_velocity: Float64
    var force: Float64
    var gravity: Float64

    # Current state
    var position: Float64
    var velocity: Float64

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool

    fn __init__(out self):
        """Initialize MountainCar with default physics parameters."""
        # Physics constants from Gymnasium
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0  # Not used in standard version
        self.force = 0.001
        self.gravity = 0.0025

        # State
        self.position = -0.5
        self.velocity = 0.0

        # Episode
        self.steps = 0
        self.max_steps = 200
        self.done = False

    fn reset(mut self) -> SIMD[DType.float64, 2]:
        """Reset environment to random initial state.

        Initial position is uniformly random in [-0.6, -0.4].
        Initial velocity is 0.

        Returns observation: [position, velocity].
        """
        # Random initial position in [-0.6, -0.4]
        self.position = -0.6 + random_float64() * 0.2
        self.velocity = 0.0

        self.steps = 0
        self.done = False

        return self._get_obs()

    fn step(mut self, action: Int) -> Tuple[SIMD[DType.float64, 2], Float64, Bool]:
        """Take action and return (observation, reward, done).

        Args:
            action: 0 (push left), 1 (no push), 2 (push right)

        Physics:
            velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
            position(t+1) = position(t) + velocity(t+1)

        Both are clipped to their respective ranges.
        Collisions at boundaries are inelastic (velocity set to 0).
        """
        # Convert action to force direction: 0->-1, 1->0, 2->+1
        var force_direction = Float64(action - 1)

        # Update velocity
        self.velocity = self.velocity + force_direction * self.force - cos(3.0 * self.position) * self.gravity

        # Clip velocity
        if self.velocity < -self.max_speed:
            self.velocity = -self.max_speed
        elif self.velocity > self.max_speed:
            self.velocity = self.max_speed

        # Update position
        self.position = self.position + self.velocity

        # Handle boundary collisions (inelastic)
        if self.position < self.min_position:
            self.position = self.min_position
            self.velocity = 0.0  # Inelastic collision
        elif self.position > self.max_position:
            self.position = self.max_position
            self.velocity = 0.0

        self.steps += 1

        # Check termination conditions
        var goal_reached = self.position >= self.goal_position
        var truncated = self.steps >= self.max_steps

        self.done = goal_reached or truncated

        # Reward: -1 for each step until goal
        var reward: Float64 = -1.0

        return (self._get_obs(), reward, self.done)

    fn _get_obs(self) -> SIMD[DType.float64, 2]:
        """Return current observation."""
        var obs = SIMD[DType.float64, 2]()
        obs[0] = self.position
        obs[1] = self.velocity
        return obs

    fn get_state(self) -> SIMD[DType.float64, 2]:
        """Return current state (alias for _get_obs)."""
        return self._get_obs()

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    fn obs_dim(self) -> Int:
        """Return observation dimension (2)."""
        return 2

    fn get_height(self, position: Float64) -> Float64:
        """Get the height of the car at a given position.

        Used for visualization. The mountain shape is sin(3*x).
        """
        return sin(3.0 * position) * 0.45 + 0.55

    fn close(self):
        """No-op for native env (no resources to clean up)."""
        pass


fn discretize_obs_mountain_car(obs: SIMD[DType.float64, 2], num_bins: Int = 20) -> Int:
    """Discretize continuous observation into a single state index.

    Args:
        obs: [position, velocity]
        num_bins: Number of bins per dimension (default 20 -> 400 states).

    Returns:
        Single integer state index in [0, num_bins^2).
    """
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


fn get_num_states_mountain_car(num_bins: Int = 20) -> Int:
    """Return total number of discretized states."""
    return num_bins * num_bins


fn make_mountain_car_tile_coding(
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
) -> TileCoding:
    """Create tile coding configured for MountainCar environment.

    MountainCar state: [position, velocity]

    Args:
        num_tilings: Number of tilings (default 8).
        tiles_per_dim: Tiles per dimension (default 8).

    Returns:
        TileCoding configured for MountainCar state space.
    """
    var tiles = List[Int]()
    tiles.append(tiles_per_dim)
    tiles.append(tiles_per_dim)

    # MountainCar state bounds (slightly expanded for safety)
    var state_low = List[Float64]()
    state_low.append(-1.2)   # position min
    state_low.append(-0.07)  # velocity min

    var state_high = List[Float64]()
    state_high.append(0.6)   # position max
    state_high.append(0.07)  # velocity max

    return TileCoding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles^,
        state_low=state_low^,
        state_high=state_high^,
    )


# Import for tile coding factory
from core.tile_coding import TileCoding
from math import sin
