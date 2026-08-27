from mojo_rl.core import State, Action, Env, DiscreteEnv, RenderableEnv


@fieldwise_init
struct GridState(Copyable, ImplicitlyCopyable, Movable, State):
    """State representing a position in a 2D grid."""

    var x: Int
    var y: Int

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y

    def __init__(out self, *, deinit move: Self):
        self.x = move.x
        self.y = move.y

    def __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    def __str__(self) -> String:
        return "GridState(" + String(self.x) + ", " + String(self.y) + ")"


@fieldwise_init
struct GridAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for grid movement: 0=up, 1=right, 2=down, 3=left."""

    var direction: Int

    def __init__(out self, *, copy: Self):
        self.direction = copy.direction

    def __init__(out self, *, deinit move: Self):
        self.direction = move.direction

    @staticmethod
    def up() -> Self:
        return Self(direction=0)

    @staticmethod
    def right() -> Self:
        return Self(direction=1)

    @staticmethod
    def down() -> Self:
        return Self(direction=2)

    @staticmethod
    def left() -> Self:
        return Self(direction=3)

    def __str__(self) -> String:
        if self.direction == 0:
            return "GridAction(UP)"
        elif self.direction == 1:
            return "GridAction(RIGHT)"
        elif self.direction == 2:
            return "GridAction(DOWN)"
        elif self.direction == 3:
            return "GridAction(LEFT)"
        return "GridAction(UNKNOWN)"


struct GridWorldEnv(DiscreteEnv, RenderableEnv):
    """A simple grid world environment.

    Agent starts at (0, 0) and must reach the goal at (width-1, height-1).
    Rewards: -1 per step, +10 for reaching goal.
    """

    comptime dtype = DType.float64
    comptime StateType = GridState
    comptime ActionType = GridAction

    var width: Int
    var height: Int
    var state: GridState
    var goal: GridState
    var _renderer_initialized: Bool

    def __init__(out self, width: Int = 5, height: Int = 5):
        self.width = width
        self.height = height
        self.state = GridState(0, 0)
        self.goal = GridState(width - 1, height - 1)
        self._renderer_initialized = False

    def state_to_index(self, state: GridState) -> Int:
        """Convert a GridState to a flat index."""
        return state.y * self.width + state.x

    def action_from_index(self, action_idx: Int) -> GridAction:
        """Create a GridAction from an index."""
        return GridAction(direction=action_idx)

    def num_states(self) -> Int:
        """Return total number of states (width * height)."""
        return self.width * self.height

    def num_actions(self) -> Int:
        """Return number of actions (4 directions)."""
        return 4

    def step(
        mut self, action: GridAction, verbose: Bool = False
    ) -> Tuple[GridState, Scalar[Self.dtype], Bool]:
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
        var reward: Scalar[Self.dtype] = 10.0 if done else -1.0

        return (self.state, reward, done)

    def reset(mut self) -> GridState:
        """Reset agent to starting position."""
        self.state = GridState(0, 0)
        return self.state

    def get_state(mut self) -> GridState:
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
        """Print the grid with agent position (text-based, renderer argument ignored).
        """
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
