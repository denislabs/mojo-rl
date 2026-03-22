"""MuZero Evaluators — Opponent strategies for measuring agent strength.

Each evaluator implements select_action given a legal mask and game state info.
Multiple evaluators can run during training to track progress against
different baselines simultaneously.

Design: Evaluators maintain their own internal game state by tracking
every move played. This avoids needing to access the env's internal state
through the generic trait.
"""

from std.random import random_float64


# ═══════════════════════════════════════════════════════════════════════════
# Evaluator Trait
# ═══════════════════════════════════════════════════════════════════════════


trait Evaluator(Movable):
    """An opponent strategy for evaluating agent strength."""

    fn name(self) -> String:
        """Human-readable name."""
        ...

    fn reset(mut self):
        """Reset internal state for a new game."""
        ...

    fn select_action(mut self, legal_mask: List[Bool], num_actions: Int) -> Int:
        """Select an action given legal mask. May update internal state."""
        ...

    fn observe_action(mut self, action: Int, player: Int):
        """Observe an action played (by either player). Updates internal state."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Random Opponent
# ═══════════════════════════════════════════════════════════════════════════


struct RandomOpponent(Evaluator):
    """Uniformly random legal action selection. Weakest baseline."""

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn name(self) -> String:
        return "Random"

    fn reset(mut self):
        pass

    fn select_action(mut self, legal_mask: List[Bool], num_actions: Int) -> Int:
        var n_legal = 0
        for a in range(num_actions):
            if a < len(legal_mask) and legal_mask[a]:
                n_legal += 1
        if n_legal == 0:
            return 0
        var pick = Int(random_float64() * Float64(n_legal))
        if pick >= n_legal:
            pick = n_legal - 1
        var count = 0
        for a in range(num_actions):
            if a < len(legal_mask) and legal_mask[a]:
                if count == pick:
                    return a
                count += 1
        return 0

    fn observe_action(mut self, action: Int, player: Int):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Minimax TicTacToe (Perfect Play)
# ═══════════════════════════════════════════════════════════════════════════


struct MinimaxTicTacToe(Evaluator):
    """Perfect minimax solver for TicTacToe.

    Maintains its own board state by observing all moves.
    Against this opponent, a perfect agent always draws.
    """

    var board: InlineArray[Int, 9]
    var current_player: Int

    fn __init__(out self):
        self.board = InlineArray[Int, 9](fill=0)
        self.current_player = 0

    fn __init__(out self, *, deinit take: Self):
        self.board = take.board
        self.current_player = take.current_player

    fn name(self) -> String:
        return "Minimax"

    fn reset(mut self):
        for i in range(9):
            self.board[i] = 0
        self.current_player = 0

    fn select_action(mut self, legal_mask: List[Bool], num_actions: Int) -> Int:
        var is_max = self.current_player == 0
        var best_action = -1
        var best_score = -2 if is_max else 2

        for a in range(9):
            if a >= len(legal_mask) or not legal_mask[a]:
                continue
            # Make move on internal board
            var child = InlineArray[Int, 9](fill=0)
            for i in range(9):
                child[i] = self.board[i]
            child[a] = self.current_player + 1

            var score = self._minimax(child, 1 - self.current_player, not is_max)

            if is_max and score > best_score:
                best_score = score
                best_action = a
            elif not is_max and score < best_score:
                best_score = score
                best_action = a

        if best_action < 0:
            for a in range(9):
                if a < len(legal_mask) and legal_mask[a]:
                    return a
        return best_action

    fn observe_action(mut self, action: Int, player: Int):
        """Track the move on internal board."""
        if action >= 0 and action < 9:
            self.board[action] = player + 1
        self.current_player = 1 - player

    fn _minimax(self, board: InlineArray[Int, 9], next_player: Int, is_maximizing: Bool) -> Int:
        var winner = self._check_winner(board)
        if winner == 1:
            return 1
        if winner == 2:
            return -1

        var has_empty = False
        for i in range(9):
            if board[i] == 0:
                has_empty = True
                break
        if not has_empty:
            return 0

        if is_maximizing:
            var best = -2
            for a in range(9):
                if board[a] != 0:
                    continue
                var child = InlineArray[Int, 9](fill=0)
                for i in range(9):
                    child[i] = board[i]
                child[a] = next_player + 1
                var score = self._minimax(child, 1 - next_player, False)
                if score > best:
                    best = score
            return best
        else:
            var best = 2
            for a in range(9):
                if board[a] != 0:
                    continue
                var child = InlineArray[Int, 9](fill=0)
                for i in range(9):
                    child[i] = board[i]
                child[a] = next_player + 1
                var score = self._minimax(child, 1 - next_player, True)
                if score < best:
                    best = score
            return best

    fn _check_winner(self, board: InlineArray[Int, 9]) -> Int:
        # Rows
        for r in range(3):
            if board[r * 3] != 0 and board[r * 3] == board[r * 3 + 1] and board[r * 3 + 1] == board[r * 3 + 2]:
                return board[r * 3]
        # Columns
        for c in range(3):
            if board[c] != 0 and board[c] == board[c + 3] and board[c + 3] == board[c + 6]:
                return board[c]
        # Diagonals
        if board[0] != 0 and board[0] == board[4] and board[4] == board[8]:
            return board[0]
        if board[2] != 0 and board[2] == board[4] and board[4] == board[6]:
            return board[2]
        return 0
