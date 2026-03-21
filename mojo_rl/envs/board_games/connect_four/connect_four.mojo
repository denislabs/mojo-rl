"""ConnectFour — CPU+GPU environment for two-player self-play RL training.

7 columns × 6 rows. Pieces drop to lowest empty row. First to connect 4 wins.

State layout (STATE_SIZE = 46):
  [0..41]  board cells (column-major: col*6+row, 0=empty, 1=P0, 2=P1)
           row 0 = bottom, row 5 = top
  [42]     current_player (0 or 1)
  [43]     game_result (0=ongoing, 1=P0 wins, 2=P1 wins, 3=draw)
  [44]     step_count
  [45]     last_col (last column played, for rendering)

Canonical obs (OBS_DIM = 126 = 3 planes × 7×6):
  Plane 0 [0..41]:    my pieces
  Plane 1 [42..83]:   opponent pieces
  Plane 2 [84..125]:  legal moves (1.0 for all cells in non-full columns)

Actions: 0-6 = column index. Full column = illegal (-1.0 reward).
"""

from std.random import random_float64
from layout import LayoutTensor, Layout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    TwoPlayerDiscreteEnv,
    GPUTwoPlayerDiscreteEnv,
)
from ..core.board_env import BoardGameState, BoardGameAction, board_dtype

# Board dimensions
comptime COLS: Int = 7
comptime ROWS: Int = 6
comptime BOARD_SIZE: Int = 42  # 7 * 6

# State slot indices
comptime S_CURRENT_PLAYER: Int = 42
comptime S_GAME_RESULT: Int = 43
comptime S_STEP_COUNT: Int = 44
comptime S_LAST_COL: Int = 45

# Game result codes
comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3


@always_inline
fn _cell_idx(col: Int, row: Int) -> Int:
    """Column-major index: col * ROWS + row. Row 0 = bottom."""
    return col * ROWS + row


# ============================================================================
# ConnectFourEnv
# ============================================================================


struct ConnectFourEnv[DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv & GPUTwoPlayerDiscreteEnv
):
    """ConnectFour environment — CPU+GPU dual path."""

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    # GPUTwoPlayerDiscreteEnv constants
    comptime STATE_SIZE: Int = 46
    comptime OBS_DIM: Int = 126  # 3 planes × 7×6
    comptime NUM_ACTIONS: Int = 7

    # CPU state
    var state: InlineArray[Scalar[Self.dtype], 46]
    var done: Bool

    fn __init__(out self):
        self.state = InlineArray[Scalar[Self.dtype], 46](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    fn reset(mut self) -> BoardGameState:
        for i in range(46):
            self.state[i] = 0.0
        self.done = False
        return BoardGameState(index=0)

    fn step(
        mut self, action: BoardGameAction, verbose: Bool = False
    ) -> Tuple[BoardGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            BoardGameState(index=Int(self.state[S_STEP_COUNT])),
            result[0],
            result[1],
        )

    fn _find_drop_row(self, col: Int) -> Int:
        """Find the lowest empty row in a column. Returns -1 if full."""
        for row in range(ROWS):
            if self.state[_cell_idx(col, row)] == 0.0:
                return row
        return -1

    fn _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Place piece in column, check win/draw. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var col = action
        if col < 0 or col >= COLS:
            return (Scalar[Self.dtype](-1.0), False)

        # Find drop row
        var row = self._find_drop_row(col)
        if row < 0:
            # Column full — illegal move
            return (Scalar[Self.dtype](-1.0), False)

        var player = Int(self.state[S_CURRENT_PLAYER])
        var mark = Scalar[Self.dtype](player + 1)

        # Place piece
        self.state[_cell_idx(col, row)] = mark
        self.state[S_STEP_COUNT] += 1.0
        self.state[S_LAST_COL] = Scalar[Self.dtype](col)

        # Check win
        if self._check_win_cpu(col, row, mark):
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](player + 1)
            self.done = True
            return (Scalar[Self.dtype](1.0), True)

        # Check draw (board full = 42 pieces)
        if Int(self.state[S_STEP_COUNT]) >= BOARD_SIZE:
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
            self.done = True
            return (Scalar[Self.dtype](0.0), True)

        # Switch player
        self.state[S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
        return (Scalar[Self.dtype](0.0), False)

    fn _check_win_cpu(self, col: Int, row: Int, mark: Scalar[Self.dtype]) -> Bool:
        """Check 4-in-a-row from the last placed piece in all 4 directions."""
        # Direction pairs: horizontal, vertical, diagonal /, diagonal \
        # (dc, dr) pairs
        return (
            self._count_dir(col, row, mark, 1, 0) + self._count_dir(col, row, mark, -1, 0) >= 3
            or self._count_dir(col, row, mark, 0, 1) + self._count_dir(col, row, mark, 0, -1) >= 3
            or self._count_dir(col, row, mark, 1, 1) + self._count_dir(col, row, mark, -1, -1) >= 3
            or self._count_dir(col, row, mark, 1, -1) + self._count_dir(col, row, mark, -1, 1) >= 3
        )

    fn _count_dir(self, col: Int, row: Int, mark: Scalar[Self.dtype], dc: Int, dr: Int) -> Int:
        """Count consecutive marks in one direction (excluding the starting cell)."""
        var count = 0
        var c = col + dc
        var r = row + dr
        while c >= 0 and c < COLS and r >= 0 and r < ROWS:
            if self.state[_cell_idx(c, r)] != mark:
                break
            count += 1
            c += dc
            r += dr
        return count

    # ========================================================================
    # Env trait methods
    # ========================================================================

    fn get_state(self) -> BoardGameState:
        return BoardGameState(index=Int(self.state[S_STEP_COUNT]))

    fn close(mut self):
        pass

    fn action_from_index(self, action_idx: Int) -> BoardGameAction:
        return BoardGameAction(value=action_idx)

    fn num_actions(self) -> Int:
        return 7

    fn obs_dim(self) -> Int:
        return 126

    fn num_states(self) -> Int:
        return 1

    fn state_to_index(self, state: BoardGameState) -> Int:
        return state.index

    # ========================================================================
    # TwoPlayerDiscreteEnv trait methods
    # ========================================================================

    fn current_player(self) -> Int:
        return Int(self.state[S_CURRENT_PLAYER])

    fn legal_action_mask(self) -> List[Bool]:
        var mask = List[Bool](capacity=7)
        for col in range(COLS):
            # Column is legal if top row (row 5) is empty
            mask.append(
                self.state[_cell_idx(col, ROWS - 1)] == 0.0 and not self.done
            )
        return mask^

    fn game_result(self) -> Int:
        return Int(self.state[S_GAME_RESULT])

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=126)
        var player = Int(self.state[S_CURRENT_PLAYER])
        var my_mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Plane 0: my pieces
        for i in range(BOARD_SIZE):
            if self.state[i] == my_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 1: opponent pieces
        for i in range(BOARD_SIZE):
            if self.state[i] == opp_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 2: legal moves (broadcast: all cells in legal columns = 1.0)
        for col in range(COLS):
            var col_legal = self.state[_cell_idx(col, ROWS - 1)] == 0.0
            for row in range(ROWS):
                if col_legal:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Single-agent step: agent plays, then random opponent plays."""
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent
        var legal_cols = List[Int](capacity=7)
        for col in range(COLS):
            if self.state[_cell_idx(col, ROWS - 1)] == 0.0:
                legal_cols.append(col)

        if len(legal_cols) > 0:
            var opp_idx = Int(random_float64() * Float64(len(legal_cols)))
            if opp_idx >= len(legal_cols):
                opp_idx = len(legal_cols) - 1
            var opp_result = self._step_impl(legal_cols[opp_idx])
            done = opp_result[1]
            if done and Int(self.state[S_GAME_RESULT]) != RESULT_DRAW:
                reward = Scalar[Self.dtype](-1.0)

        return (self.get_obs_list(), reward, done)

    # ========================================================================
    # GPU: Inline step/reset kernels
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    @always_inline
    fn _gpu_count_dir[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
        i: Int,
        col: Int,
        row: Int,
        mark: Scalar[board_dtype],
        dc: Int,
        dr: Int,
    ) -> Int:
        """Count consecutive marks in one direction on GPU."""
        var count = 0
        var c = col + dc
        var r = row + dr
        while c >= 0 and c < COLS and r >= 0 and r < ROWS:
            if states[i, _cell_idx(c, r)] != mark:
                break
            count += 1
            c += dc
            r += dr
        return count

    @staticmethod
    @always_inline
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        rewards: LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Per-thread ConnectFour step kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var col = Int(actions[i])

        # Already done?
        if states[i, S_GAME_RESULT] != 0.0:
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Validate column
        if col < 0 or col >= COLS:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Find drop row
        var row = -1
        for r in range(ROWS):
            if states[i, _cell_idx(col, r)] == 0.0:
                row = r
                break

        if row < 0:
            # Column full
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        var player = Int(states[i, S_CURRENT_PLAYER])
        var mark = Scalar[board_dtype](player + 1)

        # Place piece
        states[i, _cell_idx(col, row)] = mark
        states[i, S_STEP_COUNT] = states[i, S_STEP_COUNT] + 1.0
        states[i, S_LAST_COL] = Scalar[board_dtype](col)

        # Check win (4 directions)
        var h = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, 1, 0) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, -1, 0)
        var v = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, 0, 1) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, 0, -1)
        var d1 = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, 1, 1) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, -1, -1)
        var d2 = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, 1, -1) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](states, i, col, row, mark, -1, 1)

        if h >= 3 or v >= 3 or d1 >= 3 or d2 >= 3:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](player + 1)
            rewards[i] = 1.0
            dones[i] = 1.0
            return

        # Check draw
        if Int(states[i, S_STEP_COUNT]) >= BOARD_SIZE:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](RESULT_DRAW)
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Switch player
        states[i, S_CURRENT_PLAYER] = Scalar[board_dtype](1 - player)
        rewards[i] = 0.0
        dones[i] = 0.0

    @staticmethod
    @always_inline
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        for c in range(46):
            states[i, c] = 0.0

    @staticmethod
    @always_inline
    fn selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
        dones: LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] > 0.5:
            for c in range(46):
                states[i, c] = 0.0
            dones[i] = 0.0

    @staticmethod
    @always_inline
    fn extract_obs_and_masks[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        NUM_ACTIONS: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            ImmutAnyOrigin,
        ],
        obs: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, OBS_DIM),
            MutAnyOrigin,
        ],
        legal_masks: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, NUM_ACTIONS),
            MutAnyOrigin,
        ],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var player = Int(states[i, S_CURRENT_PLAYER])
        var my_mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)
        var game_over = states[i, S_GAME_RESULT] != 0.0

        # Plane 0: my pieces, Plane 1: opp pieces
        for c in range(BOARD_SIZE):
            var cell = states[i, c]
            if cell == my_mark:
                obs[i, c] = 1.0
            else:
                obs[i, c] = 0.0
            if cell == opp_mark:
                obs[i, BOARD_SIZE + c] = 1.0
            else:
                obs[i, BOARD_SIZE + c] = 0.0

        # Plane 2: legal moves (broadcast by column) + legal_masks
        for col in range(COLS):
            var top_idx = _cell_idx(col, ROWS - 1)
            var col_legal = states[i, top_idx] == 0.0 and not game_over
            if col_legal:
                legal_masks[i, col] = 1.0
            else:
                legal_masks[i, col] = 0.0
            for row in range(ROWS):
                var obs_idx = 2 * BOARD_SIZE + _cell_idx(col, row)
                if col_legal:
                    obs[i, obs_idx] = 1.0
                else:
                    obs[i, obs_idx] = 0.0

    # ========================================================================
    # GPU Launcher Methods
    # ========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        actions_buf: DeviceBuffer[board_dtype],
        mut rewards_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        mut terminated_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var terminated_out = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](terminated_buf.unsafe_ptr())
        var obs = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var legal_masks = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, 7), MutAnyOrigin
        ](legal_masks_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn step_wrapper(
            states: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                board_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            rewards: LayoutTensor[
                board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM),
                MutAnyOrigin,
            ],
            legal_masks: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, 7),
                MutAnyOrigin,
            ],
        ):
            ConnectFourEnv.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones
            )
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

            var states_read = LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                ImmutAnyOrigin,
            ](rebind[UnsafePointer[Scalar[board_dtype], ImmutAnyOrigin]](states.ptr))
            ConnectFourEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 7
            ](states_read, obs, legal_masks)

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            ConnectFourEnv.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64,
    ) raises:
        var states = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn sel_reset_wrapper(
            states: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                board_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            ConnectFourEnv.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones
            )

        ctx.enqueue_function[sel_reset_wrapper, sel_reset_wrapper](
            states,
            dones,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
    ) raises:
        var states = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var legal_masks = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, 7), MutAnyOrigin
        ](legal_masks_buf.unsafe_ptr())
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        fn extract_wrapper(
            states: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                ImmutAnyOrigin,
            ],
            obs: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM),
                MutAnyOrigin,
            ],
            legal_masks: LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, 7),
                MutAnyOrigin,
            ],
        ):
            ConnectFourEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 7
            ](states, obs, legal_masks)

        ctx.enqueue_function[extract_wrapper, extract_wrapper](
            states,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
