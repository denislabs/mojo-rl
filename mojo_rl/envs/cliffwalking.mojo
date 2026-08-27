from mojo_rl.core import State, Action, Env, DiscreteEnv, RenderableEnv


@fieldwise_init
struct CliffState(Copyable, ImplicitlyCopyable, Movable, State):
    """State representing a position on the cliff grid."""

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


@fieldwise_init
struct CliffAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for cliff walking: 0=up, 1=right, 2=down, 3=left."""

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


struct CliffWalkingEnv(DiscreteEnv, RenderableEnv):
    """CliffWalking environment.

    The agent must navigate from the start to the goal along the bottom edge
    of a grid, avoiding the cliff. Falling off the cliff returns the agent
    to the start with a large negative reward.

    Grid layout (default 4x12):
        . . . . . . . . . . . .
        . . . . . . . . . . . .
        . . . . . . . . . . . .
        S C C C C C C C C C C G

    S = Start (0, 0)
    G = Goal (width-1, 0)
    C = Cliff (returns to start with -100 reward)
    . = Safe cells (-1 reward per step)

    Actions: 0=up, 1=right, 2=down, 3=left
    """

    comptime dtype = DType.float64
    comptime StateType = CliffState
    comptime ActionType = CliffAction

    var width: Int
    var height: Int
    var state: CliffState
    var start: CliffState
    var goal: CliffState
    var _renderer_initialized: Bool

    def __init__(out self, width: Int = 12, height: Int = 4):
        self.width = width
        self.height = height
        self.start = CliffState(0, 0)
        self.goal = CliffState(width - 1, 0)
        self.state = CliffState(0, 0)
        self._renderer_initialized = False

    def state_to_index(self, state: CliffState) -> Int:
        """Convert a CliffState to a flat index."""
        return state.y * self.width + state.x

    def action_from_index(self, action_idx: Int) -> CliffAction:
        """Create a CliffAction from an index."""
        return CliffAction(direction=action_idx)

    def num_states(self) -> Int:
        """Return total number of states (width * height)."""
        return self.width * self.height

    def num_actions(self) -> Int:
        """Return number of actions (4 directions)."""
        return 4

    def _is_cliff(self, x: Int, y: Int) -> Bool:
        """Check if position is on the cliff (bottom row, excluding start and goal).
        """
        return y == 0 and x > 0 and x < self.width - 1

    def step(
        mut self, action: CliffAction, verbose: Bool = False
    ) -> Tuple[CliffState, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done).

        Rewards:
            - Falling off cliff: -100 (returns to start)
            - Reaching goal: -1 (episode ends)
            - Otherwise: -1
        """
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

        # Check if fell off cliff
        if self._is_cliff(new_x, new_y):
            self.state = self.start  # Return to start
            return (self.state, Scalar[Self.dtype](-100.0), False)

        self.state = CliffState(new_x, new_y)

        # Check if reached goal
        var done = self.state == self.goal
        var reward: Scalar[Self.dtype] = -1.0

        return (self.state, reward, done)

    def reset(mut self) -> CliffState:
        """Reset agent to starting position."""
        self.state = CliffState(0, 0)
        return self.state

    def get_state(mut self) -> CliffState:
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
        for y in range(self.height - 1, -1, -1):
            var row = String("")
            for x in range(self.width):
                if self.state.x == x and self.state.y == y:
                    row += "A "  # Agent
                elif x == 0 and y == 0:
                    row += "S "  # Start
                elif x == self.width - 1 and y == 0:
                    row += "G "  # Goal
                elif self._is_cliff(x, y):
                    row += "C "  # Cliff
                else:
                    row += ". "  # Safe
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
