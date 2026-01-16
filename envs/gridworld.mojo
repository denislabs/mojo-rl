from core import State, Action, Env, DiscreteEnv
from render import RendererBase


@fieldwise_init
struct GridState(Copyable, ImplicitlyCopyable, Movable, State):
    """State representing a position in a 2D grid."""

    var x: Int
    var y: Int

    fn __copyinit__(out self, existing: Self):
        self.x = existing.x
        self.y = existing.y

    fn __moveinit__(out self, deinit existing: Self):
        self.x = existing.x
        self.y = existing.y

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y


@fieldwise_init
struct GridAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for grid movement: 0=up, 1=right, 2=down, 3=left."""

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


struct GridWorldEnv(DiscreteEnv):
    """A simple grid world environment.

    Agent starts at (0, 0) and must reach the goal at (width-1, height-1).
    Rewards: -1 per step, +10 for reaching goal.
    """

    comptime StateType = GridState
    comptime ActionType = GridAction

    var width: Int
    var height: Int
    var state: GridState
    var goal: GridState

    fn __init__(out self, width: Int = 5, height: Int = 5):
        self.width = width
        self.height = height
        self.state = GridState(0, 0)
        self.goal = GridState(width - 1, height - 1)

    fn state_to_index(self, state: GridState) -> Int:
        """Convert a GridState to a flat index."""
        return state.y * self.width + state.x

    fn action_from_index(self, action_idx: Int) -> GridAction:
        """Create a GridAction from an index."""
        return GridAction(direction=action_idx)

    fn num_states(self) -> Int:
        """Return total number of states (width * height)."""
        return self.width * self.height

    fn num_actions(self) -> Int:
        """Return number of actions (4 directions)."""
        return 4

    fn step(mut self, action: GridAction) -> Tuple[GridState, Float64, Bool]:
        """Take an action and return (next_state, reward, done)."""
        var new_x = self.state.x
        var new_y = self.state.y

        if action.direction == 0:  # up
            new_y = min(self.state.y + 1, self.height - 1)
        elif action.direction == 1:  # right
            new_x = min(self.state.x + 1, self.width - 1)
        elif action.direction == 2:  # down
            new_y = max(self.state.y - 1, 0)
        elif action.direction == 3:  # left
            new_x = max(self.state.x - 1, 0)

        self.state = GridState(new_x, new_y)

        var done = self.state == self.goal
        var reward: Float64 = 10.0 if done else -1.0

        return (self.state, reward, done)

    fn reset(mut self) -> GridState:
        """Reset agent to starting position."""
        self.state = GridState(0, 0)
        return self.state

    fn get_state(self) -> GridState:
        """Return current state."""
        return self.state

    fn render(mut self, mut renderer: RendererBase):
        """Print the grid with agent position (text-based, renderer argument ignored)."""
        _ = renderer
        for y in range(self.height - 1, -1, -1):
            var row = String("")
            for x in range(self.width):
                if self.state.x == x and self.state.y == y:
                    row += "A "  # Agent
                elif self.goal.x == x and self.goal.y == y:
                    row += "G "  # Goal
                else:
                    row += ". "  # Empty
            print(row)
        print("")

    fn close(mut self):
        """No resources to clean up."""
        pass
