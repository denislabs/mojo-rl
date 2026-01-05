"""Gymnasium Toy Text environments wrapper.

Toy Text environments (simple discrete environments):
- FrozenLake-v1: Navigate frozen lake to reach goal
- Taxi-v3: Pick up and drop off passengers
- Blackjack-v1: Play blackjack card game
- CliffWalking-v0: Navigate cliff edge

These are simple tabular environments ideal for testing RL algorithms.
Note: We already have native Mojo implementations for FrozenLake, Taxi,
and CliffWalking in envs/. These wrappers provide Gymnasium compatibility.
"""

from python import Python, PythonObject


struct GymFrozenLakeEnv:
    """FrozenLake-v1: Navigate a frozen lake grid.

    The agent navigates a 4x4 (or 8x8) grid:
    - S: Start
    - F: Frozen (safe)
    - H: Hole (terminal, reward 0)
    - G: Goal (terminal, reward 1)

    Observation: Discrete - current position (0-15 for 4x4)
    Actions: Discrete(4) - 0:Left, 1:Down, 2:Right, 3:Up

    The ice is slippery: actions may not result in intended movement!
    """

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int
    var map_size: Int
    var is_slippery: Bool

    fn __init__(out self, map_name: String = "4x4", is_slippery: Bool = True, render_mode: String = "") raises:
        """Initialize FrozenLake.

        Args:
            map_name: "4x4" or "8x8"
            is_slippery: If True, movement is stochastic
            render_mode: "human" for visual rendering
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
                render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make(
                "FrozenLake-v1",
                map_name=PythonObject(map_name),
                is_slippery=PythonObject(is_slippery)
            )

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> Int:
        var result = self.env.reset()
        self.current_state = Int(result[0])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_state

    fn step(mut self, action: Int) raises -> Tuple[Int, Float64, Bool]:
        var result = self.env.step(action)
        self.current_state = Int(result[0])
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_state, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_states(self) -> Int:
        return self.map_size

    fn num_actions(self) -> Int:
        return 4


struct GymTaxiEnv:
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
    """

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make("Taxi-v3", render_mode=PythonObject("human"))
        else:
            self.env = self.gym.make("Taxi-v3")

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> Int:
        var result = self.env.reset()
        self.current_state = Int(result[0])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_state

    fn step(mut self, action: Int) raises -> Tuple[Int, Float64, Bool]:
        var result = self.env.step(action)
        self.current_state = Int(result[0])
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_state, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_states(self) -> Int:
        return 500

    fn num_actions(self) -> Int:
        return 6


struct BlackjackEnv:
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
    """

    var env: PythonObject
    var gym: PythonObject
    var player_sum: Int
    var dealer_card: Int
    var usable_ace: Bool
    var done: Bool
    var episode_reward: Float64

    fn __init__(out self, natural: Bool = False, sab: Bool = False, render_mode: String = "") raises:
        """Initialize Blackjack.

        Args:
            natural: If True, natural blackjack gives 1.5x reward
            sab: If True, use Sutton & Barto rules
            render_mode: "human" for visual rendering
        """
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make(
                "Blackjack-v1",
                natural=PythonObject(natural),
                sab=PythonObject(sab),
                render_mode=PythonObject("human")
            )
        else:
            self.env = self.gym.make(
                "Blackjack-v1",
                natural=PythonObject(natural),
                sab=PythonObject(sab)
            )

        self.player_sum = 0
        self.dealer_card = 0
        self.usable_ace = False
        self.done = False
        self.episode_reward = 0.0

    fn reset(mut self) raises -> Tuple[Int, Int, Bool]:
        var result = self.env.reset()
        var obs = result[0]
        self.player_sum = Int(obs[0])
        self.dealer_card = Int(obs[1])
        self.usable_ace = obs[2].__bool__()
        self.done = False
        self.episode_reward = 0.0
        return (self.player_sum, self.dealer_card, self.usable_ace)

    fn step(mut self, action: Int) raises -> Tuple[Int, Int, Bool, Float64, Bool]:
        """Returns (player_sum, dealer_card, usable_ace, reward, done)."""
        var result = self.env.step(action)
        var obs = result[0]
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.player_sum = Int(obs[0])
        self.dealer_card = Int(obs[1])
        self.usable_ace = obs[2].__bool__()
        self.done = terminated or truncated
        self.episode_reward += reward

        return (self.player_sum, self.dealer_card, self.usable_ace, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_actions(self) -> Int:
        return 2

    fn state_to_index(self, player_sum: Int, dealer_card: Int, usable_ace: Bool) -> Int:
        """Convert observation to flat state index for tabular methods."""
        # player_sum: 4-21 (18 values), dealer: 1-10 (10 values), ace: 0-1 (2 values)
        var ps = player_sum - 4  # 0-17
        var dc = dealer_card - 1  # 0-9
        var ua = 1 if usable_ace else 0
        return ps * 20 + dc * 2 + ua

    fn num_states(self) -> Int:
        return 18 * 10 * 2  # 360 states


struct GymCliffWalkingEnv:
    """CliffWalking-v1: Navigate along a cliff edge.

    4x12 grid. Agent starts at bottom-left, goal at bottom-right.
    Bottom edge (except start/goal) is the cliff - falling gives -100 reward.

    Observation: Discrete(48) - current position
    Actions: Discrete(4) - 0:Up, 1:Right, 2:Down, 3:Left

    Rewards:
        - -1 for each step
        - -100 for falling off cliff (resets to start)
    """

    var env: PythonObject
    var gym: PythonObject
    var current_state: Int
    var done: Bool
    var episode_reward: Float64
    var episode_length: Int

    fn __init__(out self, render_mode: String = "") raises:
        self.gym = Python.import_module("gymnasium")

        if render_mode == "human":
            self.env = self.gym.make("CliffWalking-v1", render_mode=PythonObject("human"))
        else:
            self.env = self.gym.make("CliffWalking-v1")

        self.current_state = 0
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0

    fn reset(mut self) raises -> Int:
        var result = self.env.reset()
        self.current_state = Int(result[0])
        self.done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.current_state

    fn step(mut self, action: Int) raises -> Tuple[Int, Float64, Bool]:
        var result = self.env.step(action)
        self.current_state = Int(result[0])
        var reward = Float64(result[1])
        var terminated = result[2].__bool__()
        var truncated = result[3].__bool__()

        self.done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1

        return (self.current_state, reward, self.done)

    fn close(mut self) raises:
        _ = self.env.close()

    fn num_states(self) -> Int:
        return 48

    fn num_actions(self) -> Int:
        return 4
