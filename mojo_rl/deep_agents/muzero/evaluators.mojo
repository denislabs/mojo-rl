"""Evaluators — Opponent strategies for measuring agent strength.

CPU evaluators: Evaluator trait — select_action called per game on CPU.
GPU evaluators: GPUEvaluator trait — select_action_gpu enqueues a GPU kernel
  for batched action selection across all environments.

Design: CPU evaluators maintain internal game state by tracking moves.
GPU evaluators are stateless (static kernels), suitable for simple strategies
like random play that don't need game history.
"""

from std.random import random_float64
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype


# ═══════════════════════════════════════════════════════════════════════════
# CPU Evaluator Trait
# ═══════════════════════════════════════════════════════════════════════════


trait Evaluator(Movable):
    """An opponent strategy for evaluating agent strength (CPU)."""

    def name(self) -> String:
        """Human-readable name."""
        ...

    def reset(mut self):
        """Reset internal state for a new game."""
        ...

    def select_action(
        mut self, legal_mask: List[Bool], num_actions: Int
    ) -> Int:
        """Select an action given legal mask. May update internal state."""
        ...

    def observe_action(mut self, action: Int, player: Int):
        """Observe an action played (by either player). Updates internal state.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# GPU Evaluator Trait
# ═══════════════════════════════════════════════════════════════════════════


trait GPUEvaluator(RegisterPassable):
    """GPU-compatible opponent for batched evaluation.

    Selects actions for all N_ENVS environments in one GPU kernel launch.
    Has access to game states for state-based evaluators (e.g., minimax).
    """

    comptime NAME: String

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[dtype],
        legal_masks: DeviceBuffer[dtype],
        game_states: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """Select actions for all envs on GPU.

        Args:
            ctx: GPU device context.
            actions_out: Output buffer [N_ENVS] for selected actions.
            legal_masks: Legal action masks [N_ENVS * ACT].
            game_states: Env state buffer [N_ENVS * STATE_SIZE].
            rng_seed: Random seed for stochastic evaluators.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Random Opponent (CPU + GPU)
# ═══════════════════════════════════════════════════════════════════════════


struct RandomOpponent(Evaluator & GPUEvaluator):
    """Uniformly random legal action selection. Weakest baseline.

    Conforms to both CPU (Evaluator) and GPU (GPUEvaluator) traits.
    """

    comptime NAME: String = "Random"

    def __init__(out self):
        pass

    def name(self) -> String:
        return "Random"

    def reset(mut self):
        pass

    def select_action(
        mut self, legal_mask: List[Bool], num_actions: Int
    ) -> Int:
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

    def observe_action(mut self, action: Int, player: Int):
        pass

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[dtype],
        legal_masks: DeviceBuffer[dtype],
        game_states: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """Pick uniform random legal actions for all envs on GPU."""
        from mojo_rl.deep_agents.core.kernels import (
            uniform_random_legal_actions_kernel,
        )

        _ = game_states  # Not used by random
        comptime TPB = 256
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime run = uniform_random_legal_actions_kernel[dtype, N_ENVS, ACT]
        var act_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            actions_out.unsafe_ptr()
        )
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
        ](legal_masks.unsafe_ptr())
        ctx.enqueue_function[run, run](
            act_t,
            lm_t,
            Scalar[DType.uint32](UInt32(rng_seed)),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )


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

    def __init__(out self):
        self.board = InlineArray[Int, 9](fill=0)
        self.current_player = 0

    def __init__(out self, *, deinit take: Self):
        self.board = take.board
        self.current_player = take.current_player

    def name(self) -> String:
        return "Minimax"

    def reset(mut self):
        for i in range(9):
            self.board[i] = 0
        self.current_player = 0

    def select_action(
        mut self, legal_mask: List[Bool], num_actions: Int
    ) -> Int:
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

            var score = self._minimax(
                child, 1 - self.current_player, not is_max
            )

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

    def observe_action(mut self, action: Int, player: Int):
        """Track the move on internal board."""
        if action >= 0 and action < 9:
            self.board[action] = player + 1
        self.current_player = 1 - player

    def _minimax(
        self, board: InlineArray[Int, 9], next_player: Int, is_maximizing: Bool
    ) -> Int:
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

    def _check_winner(self, board: InlineArray[Int, 9]) -> Int:
        # Rows
        for r in range(3):
            if (
                board[r * 3] != 0
                and board[r * 3] == board[r * 3 + 1]
                and board[r * 3 + 1] == board[r * 3 + 2]
            ):
                return board[r * 3]
        # Columns
        for c in range(3):
            if (
                board[c] != 0
                and board[c] == board[c + 3]
                and board[c + 3] == board[c + 6]
            ):
                return board[c]
        # Diagonals
        if board[0] != 0 and board[0] == board[4] and board[4] == board[8]:
            return board[0]
        if board[2] != 0 and board[2] == board[4] and board[4] == board[6]:
            return board[2]
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# GPU Minimax TicTacToe
# ═══════════════════════════════════════════════════════════════════════════


def _gpu_check_winner(board: InlineArray[Int, 9]) -> Int:
    """Check TicTacToe winner. Returns 0=none, 1=mark1 won, 2=mark2 won."""
    # Rows
    for r in range(3):
        var i = r * 3
        if board[i] != 0 and board[i] == board[i + 1] and board[i + 1] == board[i + 2]:
            return board[i]
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


def _gpu_minimax_ttt_kernel[
    N_ENVS: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * STATE_SIZE), MutAnyOrigin
    ],
):
    """GPU minimax for TicTacToe — iterative with explicit stack.

    No recursion — uses a fixed-size stack (max depth 9) to avoid
    GPU thread stack overflow on NVIDIA.
    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var s_off = e * STATE_SIZE
    var board = InlineArray[Int, 9](fill=0)
    for i in range(9):
        board[i] = Int(rebind[Scalar[dtype]](game_states[s_off + i]))

    var current_player = Int(rebind[Scalar[dtype]](game_states[s_off + 9]))
    var my_mark = current_player + 1

    # Try each legal root action, evaluate with iterative minimax
    var best_action = -1
    var best_score = -2

    for root_a in range(9):
        if board[root_a] != 0:
            continue

        board[root_a] = my_mark
        var score = _gpu_minimax_iterative(board, my_mark)
        board[root_a] = 0

        if score > best_score:
            best_score = score
            best_action = root_a

    if best_action < 0:
        best_action = 0
    actions_out[e] = Scalar[dtype](best_action)


def _gpu_minimax_iterative(
    mut board: InlineArray[Int, 9], my_mark: Int
) -> Int:
    """Iterative minimax using explicit stack. No recursion.

    Stack frame: (action_to_try, best_so_far, is_maximizing, mark_to_place)
    Max depth = 9 (max empty cells in TicTacToe).
    """
    # Explicit stack — each frame tracks:
    #   action_idx: next action to try (0-9, 9=done)
    #   best: best score found at this level
    #   is_max: True if maximizing
    #   mark: mark to place at this level
    #   placed_action: which cell we placed to get here (-1 if root)
    comptime MAX_DEPTH = 10
    var stk_action = InlineArray[Int, MAX_DEPTH](fill=0)
    var stk_best = InlineArray[Int, MAX_DEPTH](fill=0)
    var stk_is_max = InlineArray[Int, MAX_DEPTH](fill=0)  # 0=min, 1=max
    var stk_mark = InlineArray[Int, MAX_DEPTH](fill=0)
    var stk_placed = InlineArray[Int, MAX_DEPTH](fill=-1)

    # After placing root move, opponent plays (minimizing)
    var opp_mark = 3 - my_mark
    var depth = 0
    stk_action[0] = 0
    stk_best[0] = 2   # Minimizing: start with +2
    stk_is_max[0] = 0  # Opponent minimizes
    stk_mark[0] = opp_mark
    stk_placed[0] = -1

    while depth >= 0:
        # Check if current board position is terminal
        var winner = _gpu_check_winner(board)
        if winner != 0:
            var val = 1 if winner == my_mark else -1
            # Propagate up
            if depth == 0:
                return val
            # Undo the move that got us here
            board[stk_placed[depth]] = 0
            depth -= 1
            # Update parent's best
            if stk_is_max[depth] != 0:
                if val > stk_best[depth]:
                    stk_best[depth] = val
            else:
                if val < stk_best[depth]:
                    stk_best[depth] = val
            stk_action[depth] += 1
            continue

        # Check draw (no empty cells)
        var has_empty = False
        for i in range(9):
            if board[i] == 0:
                has_empty = True
                break
        if not has_empty:
            var val = 0  # Draw
            if depth == 0:
                return val
            board[stk_placed[depth]] = 0
            depth -= 1
            if stk_is_max[depth] != 0:
                if val > stk_best[depth]:
                    stk_best[depth] = val
            else:
                if val < stk_best[depth]:
                    stk_best[depth] = val
            stk_action[depth] += 1
            continue

        # Find next action to try at current depth
        var found_action = False
        while stk_action[depth] < 9:
            var a = stk_action[depth]
            if board[a] == 0:
                # Make move and descend
                board[a] = stk_mark[depth]
                found_action = True

                # Push new frame
                depth += 1
                stk_placed[depth] = a
                stk_action[depth] = 0
                if stk_is_max[depth - 1] != 0:
                    stk_best[depth] = 2  # Child minimizes: init +2
                    stk_is_max[depth] = 0
                else:
                    stk_best[depth] = -2  # Child maximizes: init -2
                    stk_is_max[depth] = 1
                stk_mark[depth] = 3 - stk_mark[depth - 1]
                break
            stk_action[depth] += 1

        if not found_action:
            # All actions tried at this depth — propagate best up
            var val = stk_best[depth]
            if depth == 0:
                return val
            board[stk_placed[depth]] = 0
            depth -= 1
            if stk_is_max[depth] != 0:
                if val > stk_best[depth]:
                    stk_best[depth] = val
            else:
                if val < stk_best[depth]:
                    stk_best[depth] = val
            stk_action[depth] += 1

    return 0  # Should not reach here


struct GPUMinimaxTicTacToe(GPUEvaluator):
    """Perfect minimax solver for TicTacToe on GPU.

    Reads the board directly from the env state buffer.
    Each thread runs full minimax search (bounded depth 9).
    """

    comptime NAME: String = "Minimax"

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[dtype],
        legal_masks: DeviceBuffer[dtype],
        game_states: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        _ = legal_masks  # Not needed — minimax computes from board state
        _ = rng_seed  # Deterministic

        comptime TPB = 256
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime run = _gpu_minimax_ttt_kernel[N_ENVS, STATE_SIZE, dtype]
        var act_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            actions_out.unsafe_ptr()
        )
        var gs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * STATE_SIZE), MutAnyOrigin
        ](game_states.unsafe_ptr())
        ctx.enqueue_function[run, run](
            act_t,
            gs_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
