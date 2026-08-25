from mojo_rl.core import State, Action, Env, DiscreteEnv, RenderableEnv
from std.random import random_float64


@fieldwise_init
struct FrozenState(Copyable, ImplicitlyCopyable, Movable, State):
    """State representing a position on the frozen lake grid."""

    var position: Int  # Flat index (0 to size*size - 1)

    def __init__(out self, *, copy: Self):
        self.position = copy.position

    def __init__(out self, *, deinit move: Self):
        self.position = move.position

    def __eq__(self, other: Self) -> Bool:
        return self.position == other.position


@fieldwise_init
struct FrozenAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for frozen lake: 0=left, 1=down, 2=right, 3=up."""

    var direction: Int

    def __init__(out self, *, copy: Self):
        self.direction = copy.direction

    def __init__(out self, *, deinit move: Self):
        self.direction = move.direction

    @staticmethod
    def left() -> Self:
        return Self(direction=0)

    @staticmethod
    def down() -> Self:
        return Self(direction=1)

    @staticmethod
    def right() -> Self:
        return Self(direction=2)

    @staticmethod
    def up() -> Self:
        return Self(direction=3)


struct FrozenLakeEnv(DiscreteEnv, RenderableEnv):
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

    comptime dtype = DType.float64
    comptime StateType = FrozenState
    comptime ActionType = FrozenAction

    var size: Int
    var state: FrozenState
    var holes: List[Int]  # Positions of holes
    var goal: Int  # Position of goal
    var is_slippery: Bool
    var _renderer_initialized: Bool

    def __init__(out self, size: Int = 4, is_slippery: Bool = True):
        self.size = size
        self.state = FrozenState(0)  # Start at top-left
        self.goal = size * size - 1  # Goal at bottom-right
        self.is_slippery = is_slippery
        self._renderer_initialized = False

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

    def state_to_index(self, state: FrozenState) -> Int:
        """Convert a FrozenState to a flat index."""
        return state.position

    def action_from_index(self, action_idx: Int) -> FrozenAction:
        """Create a FrozenAction from an index."""
        return FrozenAction(direction=action_idx)

    def num_states(self) -> Int:
        """Return total number of states (size * size)."""
        return self.size * self.size

    def num_actions(self) -> Int:
        """Return number of actions (4 directions)."""
        return 4

    def _is_hole(self, position: Int) -> Bool:
        """Check if position is a hole."""
        for i in range(len(self.holes)):
            if self.holes[i] == position:
                return True
        return False

    def _move(self, position: Int, action: Int) -> Int:
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

    def step(
        mut self, action: FrozenAction, verbose: Bool = False
    ) -> Tuple[FrozenState, Scalar[Self.dtype], Bool]:
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
        var reward: Scalar[Self.dtype] = 1.0 if at_goal else 0.0

        return (self.state, reward, done)

    def reset(mut self) -> FrozenState:
        """Reset agent to starting position."""
        self.state = FrozenState(0)
        return self.state

    def get_state(mut self) -> FrozenState:
        """Return current state."""
        return self.state

    def close(mut self):
        """No resources to clean up."""
        pass

    # =========================================================================
    # RenderableEnv Trait Implementation (text-only stubs)
    # =========================================================================

    def init_renderer(mut self) raises -> Bool:
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
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

    def close_renderer(mut self) raises -> None:
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        return False

    def check_renderer_quit(mut self) -> Bool:
        return False

    def renderer_delay(self, ms: Int) -> None:
        pass

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False
