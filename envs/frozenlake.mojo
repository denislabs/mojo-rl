from core import State, Action, Env, DiscreteEnv
from random import random_float64


@fieldwise_init
struct FrozenState(State, Copyable, Movable, ImplicitlyCopyable):
    """State representing a position on the frozen lake grid."""

    var position: Int  # Flat index (0 to size*size - 1)

    fn __copyinit__(out self, existing: Self):
        self.position = existing.position

    fn __moveinit__(out self, deinit existing: Self):
        self.position = existing.position

    fn __eq__(self, other: Self) -> Bool:
        return self.position == other.position


@fieldwise_init
struct FrozenAction(Action, Copyable, Movable, ImplicitlyCopyable):
    """Action for frozen lake: 0=left, 1=down, 2=right, 3=up."""

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


struct FrozenLake(DiscreteEnv):
    """FrozenLake environment.

    The agent navigates a frozen lake grid to reach a goal while avoiding holes.
    The ice is slippery, so the agent may not always move in the intended direction.

    Grid layout (4x4 default):
        S F F F     S = Start
        F H F H     F = Frozen (safe)
        F F F H     H = Hole (terminal, reward=0)
        H F F G     G = Goal (terminal, reward=1)

    Actions: 0=left, 1=down, 2=right, 3=up
    """

    comptime StateType = FrozenState
    comptime ActionType = FrozenAction

    var size: Int
    var state: FrozenState
    var holes: List[Int]  # Positions of holes
    var goal: Int  # Position of goal
    var is_slippery: Bool

    fn __init__(out self, size: Int = 4, is_slippery: Bool = True):
        self.size = size
        self.state = FrozenState(0)  # Start at top-left
        self.goal = size * size - 1  # Goal at bottom-right
        self.is_slippery = is_slippery

        # Default 4x4 layout holes: positions 5, 7, 11, 12
        self.holes = List[Int]()
        if size == 4:
            self.holes.append(5)
            self.holes.append(7)
            self.holes.append(11)
            self.holes.append(12)
        else:
            # For other sizes, create a simple pattern
            for i in range(size * size):
                var row = i // size
                var col = i % size
                # Add holes in a pattern (not at start or goal)
                if i != 0 and i != size * size - 1:
                    if (row + col) % 3 == 2 and row > 0:
                        self.holes.append(i)

    fn state_to_index(self, state: FrozenState) -> Int:
        """Convert a FrozenState to a flat index."""
        return state.position

    fn action_from_index(self, action_idx: Int) -> FrozenAction:
        """Create a FrozenAction from an index."""
        return FrozenAction(direction=action_idx)

    fn _is_hole(self, position: Int) -> Bool:
        """Check if position is a hole."""
        for i in range(len(self.holes)):
            if self.holes[i] == position:
                return True
        return False

    fn _move(self, position: Int, action: Int) -> Int:
        """Get new position after taking action from position."""
        var row = position // self.size
        var col = position % self.size

        if action == 0:  # left
            col = max(col - 1, 0)
        elif action == 1:  # down
            row = min(row + 1, self.size - 1)
        elif action == 2:  # right
            col = min(col + 1, self.size - 1)
        elif action == 3:  # up
            row = max(row - 1, 0)

        return row * self.size + col

    fn step(mut self, action: FrozenAction) -> Tuple[FrozenState, Float64, Bool]:
        """Take an action and return (next_state, reward, done).

        If slippery, there's a 1/3 chance of moving in each of the 3 directions
        (intended direction and the two perpendicular directions).
        """
        var actual_action = action.direction

        if self.is_slippery:
            var rand = random_float64()
            if rand < 0.333333:
                # Move perpendicular (left of intended)
                actual_action = (action.direction + 3) % 4
            elif rand < 0.666666:
                # Move perpendicular (right of intended)
                actual_action = (action.direction + 1) % 4
            # else: move in intended direction

        var new_position = self._move(self.state.position, actual_action)
        self.state = FrozenState(new_position)

        var in_hole = self._is_hole(new_position)
        var at_goal = new_position == self.goal
        var done = in_hole or at_goal
        var reward: Float64 = 1.0 if at_goal else 0.0

        return (self.state, reward, done)

    fn reset(mut self) -> FrozenState:
        """Reset agent to starting position."""
        self.state = FrozenState(0)
        return self.state

    fn get_state(self) -> FrozenState:
        """Return current state."""
        return self.state

    fn render(mut self):
        """Print the grid."""
        for row in range(self.size):
            var line = String("")
            for col in range(self.size):
                var pos = row * self.size + col
                if pos == self.state.position:
                    line += "A "  # Agent
                elif pos == 0:
                    line += "S "  # Start
                elif pos == self.goal:
                    line += "G "  # Goal
                elif self._is_hole(pos):
                    line += "H "  # Hole
                else:
                    line += "F "  # Frozen
            print(line)
        print("")

    fn close(mut self):
        """No resources to clean up."""
        pass
