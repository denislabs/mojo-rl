"""TicTacToe — CPU+GPU environment for two-player self-play RL training.

3×3 board, two players alternate placing marks. First to get 3 in a row wins.

State layout (STATE_SIZE = 12):
  [0..8]  board cells (0=empty, 1=P0, 2=P1)
  [9]     current_player (0 or 1)
  [10]    game_result (0=ongoing, 1=P0 wins, 2=P1 wins, 3=draw)
  [11]    step_count

Canonical obs (OBS_DIM = 27 = 3 planes × 3×3):
  Plane 0 [0..8]:   my pieces (1.0 where current player has a mark)
  Plane 1 [9..17]:  opponent pieces
  Plane 2 [18..26]: legal moves (1.0 where cell is empty)

Actions: 0-8 = cell index (row-major). Illegal move = pass with -1.0 reward.
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

# State slot indices
comptime S_BOARD_START: Int = 0
comptime S_BOARD_END: Int = 9
comptime S_CURRENT_PLAYER: Int = 9
comptime S_GAME_RESULT: Int = 10
comptime S_STEP_COUNT: Int = 11

# Game result codes
comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3



# ============================================================================
# TicTacToeEnv
# ============================================================================


struct TicTacToeEnv[DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv & GPUTwoPlayerDiscreteEnv
):
    """TicTacToe environment — CPU+GPU dual path.

    CPU: Instance methods for evaluation + single-agent mode.
    GPU: Static inline methods for batched self-play training.
    """

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    # GPUTwoPlayerDiscreteEnv constants
    comptime STATE_SIZE: Int = 12
    comptime OBS_DIM: Int = 27  # 3 planes × 3×3
    comptime NUM_ACTIONS: Int = 9

    # CPU state
    var state: InlineArray[Scalar[Self.dtype], 12]
    var done: Bool

    fn __init__(out self):
        self.state = InlineArray[Scalar[Self.dtype], 12](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    fn reset(mut self) -> BoardGameState:
        for i in range(12):
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

    fn _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal step: place mark, check win/draw. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var player = Int(self.state[S_CURRENT_PLAYER])
        var mark = Scalar[Self.dtype](player + 1)  # 1 or 2

        # Check legality
        if action < 0 or action >= 9 or self.state[action] != 0.0:
            # Illegal move — penalty, no board change
            return (Scalar[Self.dtype](-1.0), False)

        # Place mark
        self.state[action] = mark
        self.state[S_STEP_COUNT] += 1.0

        # Check win
        if Self._check_win_cpu(self.state, mark):
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](player + 1)
            self.done = True
            return (Scalar[Self.dtype](1.0), True)

        # Check draw (board full)
        var empty_count = 0
        for i in range(9):
            if self.state[i] == 0.0:
                empty_count += 1
        if empty_count == 0:
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
            self.done = True
            return (Scalar[Self.dtype](0.0), True)

        # Switch player
        self.state[S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
        return (Scalar[Self.dtype](0.0), False)

    @staticmethod
    fn _check_win_cpu(
        state: InlineArray[Scalar[Self.DTYPE], 12], mark: Scalar[Self.DTYPE]
    ) -> Bool:
        """Check if the given mark has won (rows, cols, diagonals)."""
        # Rows
        if state[0] == mark and state[1] == mark and state[2] == mark:
            return True
        if state[3] == mark and state[4] == mark and state[5] == mark:
            return True
        if state[6] == mark and state[7] == mark and state[8] == mark:
            return True
        # Columns
        if state[0] == mark and state[3] == mark and state[6] == mark:
            return True
        if state[1] == mark and state[4] == mark and state[7] == mark:
            return True
        if state[2] == mark and state[5] == mark and state[8] == mark:
            return True
        # Diagonals
        if state[0] == mark and state[4] == mark and state[8] == mark:
            return True
        if state[2] == mark and state[4] == mark and state[6] == mark:
            return True
        return False

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
        return 9

    fn obs_dim(self) -> Int:
        return 27

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
        var mask = List[Bool](capacity=9)
        for i in range(9):
            mask.append(self.state[i] == 0.0 and not self.done)
        return mask^

    fn game_result(self) -> Int:
        return Int(self.state[S_GAME_RESULT])

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # Canonical observation: always from current player's perspective
    # ========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=27)
        var player = Int(self.state[S_CURRENT_PLAYER])
        var my_mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Plane 0: my pieces
        for i in range(9):
            if self.state[i] == my_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 1: opponent pieces
        for i in range(9):
            if self.state[i] == opp_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 2: legal moves
        for i in range(9):
            if self.state[i] == 0.0:
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
        """Single-agent step: agent plays, then random opponent plays.

        This makes TicTacToe compatible with existing single-agent training
        loops (DQN, PPO, etc.) by embedding a random opponent inside the env.
        """
        # Agent's move
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent's move
        var legal_cells = List[Int](capacity=9)
        for i in range(9):
            if self.state[i] == 0.0:
                legal_cells.append(i)

        if len(legal_cells) > 0:
            var opp_idx = Int(random_float64() * Float64(len(legal_cells)))
            if opp_idx >= len(legal_cells):
                opp_idx = len(legal_cells) - 1
            var opp_action = legal_cells[opp_idx]
            var opp_result = self._step_impl(opp_action)
            done = opp_result[1]
            # If opponent won, agent gets -1
            if done and Int(self.state[S_GAME_RESULT]) != RESULT_DRAW:
                reward = Scalar[Self.dtype](-1.0)

        return (self.get_obs_list(), reward, done)

    # ========================================================================
    # GPU: Inline step/reset kernels (called per-thread on GPU)
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    @always_inline
    fn _check_win_from_states[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            board_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE),
            MutAnyOrigin,
        ],
        i: Int,
        mark: Scalar[board_dtype],
    ) -> Bool:
        """Check if the given mark has won, reading directly from states tensor."""
        # Rows
        if states[i, 0] == mark and states[i, 1] == mark and states[i, 2] == mark:
            return True
        if states[i, 3] == mark and states[i, 4] == mark and states[i, 5] == mark:
            return True
        if states[i, 6] == mark and states[i, 7] == mark and states[i, 8] == mark:
            return True
        # Columns
        if states[i, 0] == mark and states[i, 3] == mark and states[i, 6] == mark:
            return True
        if states[i, 1] == mark and states[i, 4] == mark and states[i, 7] == mark:
            return True
        if states[i, 2] == mark and states[i, 5] == mark and states[i, 8] == mark:
            return True
        # Diagonals
        if states[i, 0] == mark and states[i, 4] == mark and states[i, 8] == mark:
            return True
        if states[i, 2] == mark and states[i, 4] == mark and states[i, 6] == mark:
            return True
        return False

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
        """Per-thread TicTacToe step kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var action = Int(actions[i])

        var player = Int(states[i, S_CURRENT_PLAYER])
        var mark = Scalar[board_dtype](player + 1)

        # Check if game already done
        if states[i, S_GAME_RESULT] != 0.0:
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Check legality
        if action < 0 or action >= 9 or states[i, action] != 0.0:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Place mark
        states[i, action] = mark
        states[i, S_STEP_COUNT] = states[i, S_STEP_COUNT] + 1.0

        # Check win — read board directly from states tensor
        var won = Self._check_win_from_states[BATCH_SIZE, STATE_SIZE](
            states, i, mark
        )
        if won:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](player + 1)
            rewards[i] = 1.0
            dones[i] = 1.0
            return

        # Check draw
        var empty = 0
        for c in range(9):
            if states[i, c] == 0.0:
                empty += 1
        if empty == 0:
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
        """Per-thread reset: clear board, player 0 starts."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        for c in range(12):
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
        """Per-thread: reset only if done."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        if dones[i] > 0.5:
            for c in range(12):
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
        """Per-thread: extract canonical obs + legal masks from state."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var player = Int(states[i, S_CURRENT_PLAYER])
        var my_mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)
        var game_over = states[i, S_GAME_RESULT] != 0.0

        for c in range(9):
            var cell = states[i, c]
            # Plane 0: my pieces
            if cell == my_mark:
                obs[i, c] = 1.0
            else:
                obs[i, c] = 0.0
            # Plane 1: opponent pieces
            if cell == opp_mark:
                obs[i, 9 + c] = 1.0
            else:
                obs[i, 9 + c] = 0.0
            # Plane 2: legal moves
            var is_empty = cell == 0.0
            if is_empty and not game_over:
                obs[i, 18 + c] = 1.0
                legal_masks[i, c] = 1.0
            else:
                obs[i, 18 + c] = 0.0
                legal_masks[i, c] = 0.0

    # ========================================================================
    # GPU Launcher Methods (host-side, GPUTwoPlayerDiscreteEnv trait)
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
            board_dtype, Layout.row_major(BATCH_SIZE, 9), MutAnyOrigin
        ](legal_masks_buf.unsafe_ptr())
        var states_immut = LayoutTensor[
            board_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
        ](states_buf.unsafe_ptr())

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
                Layout.row_major(BATCH_SIZE, 9),
                MutAnyOrigin,
            ],
        ):
            # Step physics
            TicTacToeEnv.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones
            )

            # Copy terminated = dones for board games
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

            # Extract obs + legal masks (need immutable view of states)
            var states_read = LayoutTensor[
                board_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                ImmutAnyOrigin,
            ](rebind[UnsafePointer[Scalar[board_dtype], ImmutAnyOrigin]](states.ptr))
            TicTacToeEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 9
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
            TicTacToeEnv.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

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
            TicTacToeEnv.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
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
            board_dtype, Layout.row_major(BATCH_SIZE, 9), MutAnyOrigin
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
                Layout.row_major(BATCH_SIZE, 9),
                MutAnyOrigin,
            ],
        ):
            TicTacToeEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 9
            ](states, obs, legal_masks)

        ctx.enqueue_function[extract_wrapper, extract_wrapper](
            states,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
