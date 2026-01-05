from core import State, Action, Env, DiscreteEnv


@fieldwise_init
struct CliffState(State, Copyable, Movable, ImplicitlyCopyable):
    """State representing a position on the cliff grid."""

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
struct CliffAction(Action, Copyable, Movable, ImplicitlyCopyable):
    """Action for cliff walking: 0=up, 1=right, 2=down, 3=left."""

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


struct CliffWalking(DiscreteEnv):
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

    comptime StateType = CliffState
    comptime ActionType = CliffAction

    var width: Int
    var height: Int
    var state: CliffState
    var start: CliffState
    var goal: CliffState

    fn __init__(out self, width: Int = 12, height: Int = 4):
        self.width = width
        self.height = height
        self.start = CliffState(0, 0)
        self.goal = CliffState(width - 1, 0)
        self.state = CliffState(0, 0)

    fn state_to_index(self, state: CliffState) -> Int:
        """Convert a CliffState to a flat index."""
        return state.y * self.width + state.x

    fn action_from_index(self, action_idx: Int) -> CliffAction:
        """Create a CliffAction from an index."""
        return CliffAction(direction=action_idx)

    fn _is_cliff(self, x: Int, y: Int) -> Bool:
        """Check if position is on the cliff (bottom row, excluding start and goal)."""
        return y == 0 and x > 0 and x < self.width - 1

    fn step(mut self, action: CliffAction) -> Tuple[CliffState, Float64, Bool]:
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
            return (self.state, -100.0, False)

        self.state = CliffState(new_x, new_y)

        # Check if reached goal
        var done = self.state == self.goal
        var reward: Float64 = -1.0

        return (self.state, reward, done)

    fn reset(mut self) -> CliffState:
        """Reset agent to starting position."""
        self.state = CliffState(0, 0)
        return self.state

    fn get_state(self) -> CliffState:
        """Return current state."""
        return self.state

    fn render(self):
        """Print the grid with agent position."""
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

    fn close(mut self):
        """No resources to clean up."""
        pass
