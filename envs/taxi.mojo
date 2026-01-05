from core import State, Action, Env, DiscreteEnv


@fieldwise_init
struct TaxiState(State, Copyable, Movable, ImplicitlyCopyable):
    """State for Taxi environment.

    Encodes: taxi position (row, col), passenger location, destination.
    """

    var taxi_row: Int
    var taxi_col: Int
    var passenger_loc: Int  # 0-3 for locations R/G/Y/B, 4 = in taxi
    var destination: Int  # 0-3 for locations R/G/Y/B

    fn __copyinit__(out self, existing: Self):
        self.taxi_row = existing.taxi_row
        self.taxi_col = existing.taxi_col
        self.passenger_loc = existing.passenger_loc
        self.destination = existing.destination

    fn __moveinit__(out self, deinit existing: Self):
        self.taxi_row = existing.taxi_row
        self.taxi_col = existing.taxi_col
        self.passenger_loc = existing.passenger_loc
        self.destination = existing.destination

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.taxi_row == other.taxi_row
            and self.taxi_col == other.taxi_col
            and self.passenger_loc == other.passenger_loc
            and self.destination == other.destination
        )


@fieldwise_init
struct TaxiAction(Action, Copyable, Movable, ImplicitlyCopyable):
    """Action for Taxi: 0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff."""

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


struct Taxi(DiscreteEnv):
    """Taxi environment.

    A 5x5 grid with 4 designated locations (R, G, Y, B).
    The taxi must pick up a passenger at one location and drop them off at another.

    Grid layout:
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    Walls are represented by '|' characters.

    Locations:
        R = (0, 0), G = (0, 4), Y = (4, 0), B = (4, 3)

    Actions:
        0 = south (down)
        1 = north (up)
        2 = east (right)
        3 = west (left)
        4 = pickup
        5 = dropoff

    Rewards:
        -1 per step
        +20 for successful dropoff
        -10 for illegal pickup/dropoff

    State space: 5 * 5 * 5 * 4 = 500 states
        (taxi_row, taxi_col, passenger_loc, destination)
    """

    comptime StateType = TaxiState
    comptime ActionType = TaxiAction

    var state: TaxiState
    # Location coordinates: R=(0,0), G=(0,4), Y=(4,0), B=(4,3)
    var loc_rows: List[Int]
    var loc_cols: List[Int]

    fn __init__(out self):
        # Initialize locations
        self.loc_rows = List[Int]()
        self.loc_cols = List[Int]()
        # R = 0
        self.loc_rows.append(0)
        self.loc_cols.append(0)
        # G = 1
        self.loc_rows.append(0)
        self.loc_cols.append(4)
        # Y = 2
        self.loc_rows.append(4)
        self.loc_cols.append(0)
        # B = 3
        self.loc_rows.append(4)
        self.loc_cols.append(3)

        # Initialize state (will be randomized on reset)
        self.state = TaxiState(0, 0, 0, 1)

    fn _has_wall(self, row: Int, col: Int, action: Int) -> Bool:
        """Check if there's a wall blocking movement.

        Walls in the 5x5 grid:
        - Between (0,1) and (0,2), (1,1) and (1,2)
        - Between (3,0) and (3,1), (4,0) and (4,1)
        - Between (3,2) and (3,3), (4,2) and (4,3)
        """
        if action == 2:  # east (moving right, col -> col+1)
            if col == 1 and (row == 0 or row == 1):
                return True
            if col == 0 and (row == 3 or row == 4):
                return True
            if col == 2 and (row == 3 or row == 4):
                return True
        elif action == 3:  # west (moving left, col -> col-1)
            if col == 2 and (row == 0 or row == 1):
                return True
            if col == 1 and (row == 3 or row == 4):
                return True
            if col == 3 and (row == 3 or row == 4):
                return True
        return False

    fn _get_location_at(self, row: Int, col: Int) -> Int:
        """Return location index (0-3) if taxi is at a designated location, else -1."""
        for i in range(4):
            if self.loc_rows[i] == row and self.loc_cols[i] == col:
                return i
        return -1

    fn state_to_index(self, state: TaxiState) -> Int:
        """Convert a TaxiState to a flat index.

        Index = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination
        Total states = 5 * 5 * 5 * 4 = 500
        """
        return (
            (state.taxi_row * 5 + state.taxi_col) * 5 + state.passenger_loc
        ) * 4 + state.destination

    fn action_from_index(self, action_idx: Int) -> TaxiAction:
        """Create a TaxiAction from an index."""
        return TaxiAction(action=action_idx)

    fn step(mut self, action: TaxiAction) -> Tuple[TaxiState, Float64, Bool]:
        """Take an action and return (next_state, reward, done)."""
        var new_row = self.state.taxi_row
        var new_col = self.state.taxi_col
        var new_pass_loc = self.state.passenger_loc
        var reward: Float64 = -1.0
        var done = False

        if action.action == 0:  # south
            if new_row < 4:
                new_row += 1
        elif action.action == 1:  # north
            if new_row > 0:
                new_row -= 1
        elif action.action == 2:  # east
            if new_col < 4 and not self._has_wall(new_row, new_col, 2):
                new_col += 1
        elif action.action == 3:  # west
            if new_col > 0 and not self._has_wall(new_row, new_col, 3):
                new_col -= 1
        elif action.action == 4:  # pickup
            var loc = self._get_location_at(new_row, new_col)
            if loc != -1 and self.state.passenger_loc == loc:
                new_pass_loc = 4  # Passenger in taxi
            else:
                reward = -10.0  # Illegal pickup
        elif action.action == 5:  # dropoff
            var loc = self._get_location_at(new_row, new_col)
            if self.state.passenger_loc == 4:  # Passenger in taxi
                if loc == self.state.destination:
                    new_pass_loc = loc
                    reward = 20.0
                    done = True
                else:
                    reward = -10.0  # Wrong destination
            else:
                reward = -10.0  # No passenger to drop

        self.state = TaxiState(
            new_row, new_col, new_pass_loc, self.state.destination
        )
        return (self.state, reward, done)

    fn reset(mut self) -> TaxiState:
        """Reset to a random initial state.

        For simplicity, we use a deterministic start:
        Taxi at (2,2), passenger at R, destination at G.
        """
        # Simple deterministic reset for reproducibility
        self.state = TaxiState(
            taxi_row=2,
            taxi_col=2,
            passenger_loc=0,  # R
            destination=1,  # G
        )
        return self.state

    fn get_state(self) -> TaxiState:
        """Return current state."""
        return self.state

    fn render(self):
        """Print the taxi grid."""
        var loc_chars = List[String]()
        loc_chars.append("R")
        loc_chars.append("G")
        loc_chars.append("Y")
        loc_chars.append("B")

        print("+---------+")
        for row in range(5):
            var line = String("|")
            for col in range(5):
                var cell = String(" ")
                var loc = self._get_location_at(row, col)

                if self.state.taxi_row == row and self.state.taxi_col == col:
                    if self.state.passenger_loc == 4:
                        cell = "T"  # Taxi with passenger
                    else:
                        cell = "t"  # Empty taxi
                elif loc != -1:
                    if self.state.passenger_loc == loc:
                        cell = loc_chars[loc]  # Passenger waiting
                    elif self.state.destination == loc:
                        cell = loc_chars[loc].lower()  # Destination marker
                    else:
                        cell = ":"

                # Add wall or space after cell
                if col < 4:
                    if self._has_wall(row, col, 2):
                        line += cell + "|"
                    else:
                        line += cell + ":"
                else:
                    line += cell + "|"
            print(line)
        print("+---------+")

        # Print legend
        var pass_str: String
        if self.state.passenger_loc == 4:
            pass_str = "In Taxi"
        else:
            pass_str = loc_chars[self.state.passenger_loc]
        print("Passenger:", pass_str, "| Destination:", loc_chars[self.state.destination])
        print("")

    fn close(mut self):
        """No resources to clean up."""
        pass
