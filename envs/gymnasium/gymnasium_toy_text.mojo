"""Gymnasium Toy Text environments wrapper with trait conformance.

Toy Text environments (simple discrete environments):
- FrozenLake-v1: Navigate frozen lake to reach goal
- Taxi-v3: Pick up and drop off passengers
- Blackjack-v1: Play blackjack card game
- CliffWalking-v0: Navigate cliff edge

These are simple tabular environments ideal for testing RL algorithms.
All implement DiscreteEnv trait for use with generic tabular agents.
"""

from python import Python, PythonObject
from core import State, Action, DiscreteEnv


# ============================================================================
# FrozenLake State and Action types
# ============================================================================


@fieldwise_init
struct GymFrozenLakeState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for FrozenLake: position on the grid."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymFrozenLakeAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for FrozenLake: 0=Left, 1=Down, 2=Right, 3=Up."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        return Self(direction=0)

    @staticmethod
    fn down() -> Self:
        return Self(direction=1)

    @staticmethod
    fn right() -> Self:
        return Self(direction=2)

    @staticmethod
    fn up() -> Self:
        return Self(direction=3)


# ============================================================================
# Taxi State and Action types
# ============================================================================


@fieldwise_init
struct GymTaxiState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Taxi: encoded state index."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymTaxiAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Taxi: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff."""

    var action: Int

    fn __copyinit__(out self, existing: Self):
        self.action = existing.action

    fn __moveinit__(out self, deinit existing: Self):
        self.action = existing.action

    @staticmethod
    fn south() -> Self:
        return Self(action=0)

    @staticmethod
    fn north() -> Self:
        return Self(action=1)

    @staticmethod
    fn east() -> Self:
        return Self(action=2)

    @staticmethod
    fn west() -> Self:
        return Self(action=3)

    @staticmethod
    fn pickup() -> Self:
        return Self(action=4)

    @staticmethod
    fn dropoff() -> Self:
        return Self(action=5)


# ============================================================================
# CliffWalking State and Action types
# ============================================================================


@fieldwise_init
struct GymCliffWalkingState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for CliffWalking: position on the grid."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymCliffWalkingAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for CliffWalking: 0=Up, 1=Right, 2=Down, 3=Left."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn up() -> Self:
        return Self(direction=0)

    @staticmethod
    fn right() -> Self:
        return Self(direction=1)

    @staticmethod
    fn down() -> Self:
        return Self(direction=2)

    @staticmethod
    fn left() -> Self:
        return Self(direction=3)


# ============================================================================
# Blackjack State and Action types
# ============================================================================


@fieldwise_init
struct GymBlackjackState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Blackjack: encoded state index from (player_sum, dealer_card, usable_ace)."""

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct GymBlackjackAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Blackjack: 0=Stick, 1=Hit."""

    var action: Int

    fn __copyinit__(out self, existing: Self):
        self.action = existing.action

    fn __moveinit__(out self, deinit existing: Self):
        self.action = existing.action

    @staticmethod
    fn stick() -> Self:
        return Self(action=0)

    @staticmethod
    fn hit() -> Self:
        return Self(action=1)


# ============================================================================
# GymFrozenLakeEnv - implements DiscreteEnv
# ============================================================================


struct GymFrozenLakeEnv(DiscreteEnv):
    """FrozenLake-v1: Navigate a frozen lake grid.

    The agent navigates a 4x4 (or 8x8) grid:
    - S: Start
    - F: Frozen (safe)
    - H: Hole (terminal, reward 0)
    - G: Goal (terminal, reward 1)

    Observation: Discrete - current position (0-15 for 4x4)
    Actions: Discrete(4) - 0:Left, 1:Down, 2:Right, 3:Up

    The ice is slippery: actions may not result in intended movement!

    Implements DiscreteEnv trait for generic tabular training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymFrozenLakeState
    comptime ActionType = GymFrozenLakeAction

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var map_size: Int
    var is_slippery: Bool

    fn __init__(
        out self,
        map_name: String = "4x4",
        is_slippery: Bool = True,
        render_mode: String = "",
    ) raises:
        """Initialize FrozenLake.

        Args:
            map_name: "4x4" or "8x8".
            is_slippery: If True, movement is stochastic.
            render_mode: "human" for visual rendering.
        """
        self.gym = Python.import_module("gymnasium")
        self.is_slippery = is_slippery

        if map_name == "8x8":
            self.map_size = 64
        else:
            self.map_size = 16

        if render_mode == "human":
            self.env = self.gym.make(
                "FrozenLake-v1",
                map_name=PythonObject(map_name),
                is_slippery=PythonObject(is_slippery),
                render_mode=PythonObject("human"),
            )
        else:
            self.env = self.gym.make(
                "FrozenLake-v1",
                map_name=PythonObject(map_name),
                is_slippery=PythonObject(is_slippery),
            )

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymFrozenLakeState:
        """Reset environment and return initial state."""
        try:
            var result = self.env.reset()
            self.current_state = Int(result[0])
        except:
            self.current_state = 0

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymFrozenLakeState(index=self.current_state)

    fn step(
        mut self, action: GymFrozenLakeAction
    ) -> Tuple[GymFrozenLakeState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.direction)
            self.current_state = Int(result[0])
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (GymFrozenLakeState(index=self.current_state), reward, self.done)

    fn get_state(self) -> GymFrozenLakeState:
        """Return current state."""
        return GymFrozenLakeState(index=self.current_state)

    fn state_to_index(self, state: GymFrozenLakeState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymFrozenLakeAction:
        """Create action from index."""
        return GymFrozenLakeAction(direction=action_idx)

    fn num_states(self) -> Int:
        """Return total number of states."""
        return self.map_size

    fn num_actions(self) -> Int:
        """Return number of actions (4)."""
        return 4

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


# ============================================================================
# GymTaxiEnv - implements DiscreteEnv
# ============================================================================


struct GymTaxiEnv(DiscreteEnv):
    """Taxi-v3: Pick up and drop off passengers.

    5x5 grid with 4 designated pickup/dropoff locations (R, G, Y, B).
    The taxi must:
    1. Navigate to passenger location
    2. Pick up passenger
    3. Navigate to destination
    4. Drop off passenger

    Observation: Discrete(500) - encodes (taxi_row, taxi_col, passenger_loc, destination)
    Actions: Discrete(6) - 0:South, 1:North, 2:East, 3:West, 4:Pickup, 5:Dropoff

    Rewards:
        - +20 for successful dropoff
        - -10 for illegal pickup/dropoff
        - -1 for each step

    Implements DiscreteEnv trait for generic tabular training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymTaxiState
    comptime ActionType = GymTaxiAction

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        """Initialize Taxi environment.

        Args:
            render_mode: "human" for visual rendering.
        """
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make("Taxi-v3", render_mode=PythonObject("human"))
        else:
            self.env = self.gym.make("Taxi-v3")

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymTaxiState:
        """Reset environment and return initial state."""
        try:
            var result = self.env.reset()
            self.current_state = Int(result[0])
        except:
            self.current_state = 0

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymTaxiState(index=self.current_state)

    fn step(mut self, action: GymTaxiAction) -> Tuple[GymTaxiState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.action)
            self.current_state = Int(result[0])
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (GymTaxiState(index=self.current_state), reward, self.done)

    fn get_state(self) -> GymTaxiState:
        """Return current state."""
        return GymTaxiState(index=self.current_state)

    fn state_to_index(self, state: GymTaxiState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymTaxiAction:
        """Create action from index."""
        return GymTaxiAction(action=action_idx)

    fn num_states(self) -> Int:
        """Return total number of states (500)."""
        return 500

    fn num_actions(self) -> Int:
        """Return number of actions (6)."""
        return 6

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


# ============================================================================
# GymBlackjackEnv - implements DiscreteEnv
# ============================================================================


struct GymBlackjackEnv(DiscreteEnv):
    """Blackjack-v1: Play simplified blackjack.

    Observation: Tuple(player_sum, dealer_card, usable_ace)
        - player_sum: 4-21
        - dealer_card: 1-10
        - usable_ace: 0 or 1

    Actions: Discrete(2)
        0: Stick (stop receiving cards)
        1: Hit (receive another card)

    Rewards:
        +1: Win
        -1: Lose
        0: Draw

    Implements DiscreteEnv trait for generic tabular training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymBlackjackState
    comptime ActionType = GymBlackjackAction

    var env: PythonObject
    var gym: PythonObject
    var player_sum: Int
    var dealer_card: Int
    var usable_ace: Bool
    var current_state: Int
    var done: Bool
    var episode_reward: Float64

    fn __init__(
        out self,
        natural: Bool = False,
        sab: Bool = False,
        render_mode: String = "",
    ) raises:
        """Initialize Blackjack.

        Args:
            natural: If True, natural blackjack gives 1.5x reward.
            sab: If True, use Sutton & Barto rules.
            render_mode: "human" for visual rendering.
        """
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make(
                "Blackjack-v1",
                natural=PythonObject(natural),
                sab=PythonObject(sab),
                render_mode=PythonObject("human"),
            )
        else:
            self.env = self.gym.make(
                "Blackjack-v1",
                natural=PythonObject(natural),
                sab=PythonObject(sab),
            )

        self.player_sum = 0
        self.dealer_card = 0
        self.usable_ace = False
        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymBlackjackState:
        """Reset environment and return initial state."""
        try:
            var result = self.env.reset()
            var obs = result[0]
            self.player_sum = Int(obs[0])
            self.dealer_card = Int(obs[1])
            self.usable_ace = obs[2].__bool__()
        except:
            self.player_sum = 4
            self.dealer_card = 1
            self.usable_ace = False

        self.current_state = self._obs_to_index()
        self.done = False
        self.episode_reward = 0.0
        return GymBlackjackState(index=self.current_state)

    fn step(
        mut self, action: GymBlackjackAction
    ) -> Tuple[GymBlackjackState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.action)
            var obs = result[0]
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()

            self.player_sum = Int(obs[0])
            self.dealer_card = Int(obs[1])
            self.usable_ace = obs[2].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.current_state = self._obs_to_index()
        self.episode_reward += reward

        return (GymBlackjackState(index=self.current_state), reward, self.done)

    fn get_state(self) -> GymBlackjackState:
        """Return current state."""
        return GymBlackjackState(index=self.current_state)

    fn state_to_index(self, state: GymBlackjackState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymBlackjackAction:
        """Create action from index."""
        return GymBlackjackAction(action=action_idx)

    fn num_states(self) -> Int:
        """Return total number of states (360)."""
        return 18 * 10 * 2  # player_sum (4-21) * dealer (1-10) * ace (0-1)

    fn num_actions(self) -> Int:
        """Return number of actions (2)."""
        return 2

    # ========================================================================
    # Additional methods
    # ========================================================================

    fn _obs_to_index(self) -> Int:
        """Convert observation to flat state index."""
        # player_sum: 4-21 (18 values), dealer: 1-10 (10 values), ace: 0-1 (2 values)
        var ps = self.player_sum - 4  # 0-17
        var dc = self.dealer_card - 1  # 0-9
        var ua = 1 if self.usable_ace else 0
        return ps * 20 + dc * 2 + ua

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

    fn get_player_sum(self) -> Int:
        """Get current player sum."""
        return self.player_sum

    fn get_dealer_card(self) -> Int:
        """Get dealer's visible card."""
        return self.dealer_card

    fn has_usable_ace(self) -> Bool:
        """Check if player has usable ace."""
        return self.usable_ace


# ============================================================================
# GymCliffWalkingEnv - implements DiscreteEnv
# ============================================================================


struct GymCliffWalkingEnv(DiscreteEnv):
    """CliffWalking-v1: Navigate along a cliff edge.

    4x12 grid. Agent starts at bottom-left, goal at bottom-right.
    Bottom edge (except start/goal) is the cliff - falling gives -100 reward.

    Observation: Discrete(48) - current position
    Actions: Discrete(4) - 0:Up, 1:Right, 2:Down, 3:Left

    Rewards:
        - -1 for each step
        - -100 for falling off cliff (resets to start)

    Implements DiscreteEnv trait for generic tabular training.
    """

    # Type aliases for trait conformance
    comptime StateType = GymCliffWalkingState
    comptime ActionType = GymCliffWalkingAction

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        """Initialize CliffWalking environment.

        Args:
            render_mode: "human" for visual rendering.
        """
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make(
                "CliffWalking-v0", render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make("CliffWalking-v0")

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> GymCliffWalkingState:
        """Reset environment and return initial state."""
        try:
            var result = self.env.reset()
            self.current_state = Int(result[0])
        except:
            self.current_state = 36  # Bottom-left start position

        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return GymCliffWalkingState(index=self.current_state)

    fn step(
        mut self, action: GymCliffWalkingAction
    ) -> Tuple[GymCliffWalkingState, Float64, Bool]:
        """Take action and return (state, reward, done)."""
        var reward: Float64 = 0.0
        try:
            var result = self.env.step(action.direction)
            self.current_state = Int(result[0])
            reward = Float64(result[1])
            var terminated = result[2].__bool__()
            var truncated = result[3].__bool__()
            self.done = terminated or truncated
        except:
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1

        return (GymCliffWalkingState(index=self.current_state), reward, self.done)

    fn get_state(self) -> GymCliffWalkingState:
        """Return current state."""
        return GymCliffWalkingState(index=self.current_state)

    fn state_to_index(self, state: GymCliffWalkingState) -> Int:
        """Convert state to index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> GymCliffWalkingAction:
        """Create action from index."""
        return GymCliffWalkingAction(direction=action_idx)

    fn num_states(self) -> Int:
        """Return total number of states (48)."""
        return 48

    fn num_actions(self) -> Int:
        """Return number of actions (4)."""
        return 4

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
