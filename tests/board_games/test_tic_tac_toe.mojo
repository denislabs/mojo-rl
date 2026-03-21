"""Test TicTacToe environment — rules, win conditions, draws, obs, masks."""

from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


fn test_reset() raises:
    """Test that reset produces a clean board."""
    print("test_reset...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # All cells empty
    for i in range(9):
        if env.state[i] != 0.0:
            print(" FAIL: cell", i, "not empty after reset")
            return

    # Player 0 starts
    if env.current_player() != 0:
        print(" FAIL: player 0 should start")
        return

    # Game ongoing
    if env.game_result() != 0:
        print(" FAIL: game should be ongoing")
        return

    # All 9 actions legal
    var mask = env.legal_action_mask()
    for i in range(9):
        if not mask[i]:
            print(" FAIL: action", i, "should be legal")
            return

    print(" OK")


fn test_win_row() raises:
    """Test P0 wins with top row: 0,1,2."""
    print("test_win_row...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # P0 plays 0, P1 plays 3, P0 plays 1, P1 plays 4, P0 plays 2 (wins)
    var moves = List[Int](capacity=5)
    moves.append(0)
    moves.append(3)
    moves.append(1)
    moves.append(4)
    moves.append(2)

    for m in range(len(moves)):
        var result = env._step_impl(moves[m])
        var reward = result[0]
        var done = result[1]
        if m == 4:  # P0's winning move
            if not done:
                print(" FAIL: game should be done after P0 wins")
                return
            if Float64(reward) != 1.0:
                print(" FAIL: winner should get +1 reward, got", reward)
                return
            if env.game_result() != 1:
                print(" FAIL: game_result should be 1 (P0 wins)")
                return
        elif m < 4:
            if done:
                print(" FAIL: game shouldn't be done yet at move", m)
                return

    print(" OK")


fn test_win_col() raises:
    """Test P1 wins with left column: 0,3,6."""
    print("test_win_col...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # P0:1, P1:0, P0:4, P1:3, P0:8, P1:6 (P1 wins column 0)
    var moves = List[Int](capacity=6)
    moves.append(1)  # P0
    moves.append(0)  # P1
    moves.append(4)  # P0
    moves.append(3)  # P1
    moves.append(8)  # P0
    moves.append(6)  # P1 wins

    for m in range(len(moves)):
        var result = env._step_impl(moves[m])
        if m == 5:  # P1's winning move
            if not result[1]:
                print(" FAIL: game should be done after P1 wins")
                return
            if env.game_result() != 2:
                print(" FAIL: game_result should be 2 (P1 wins)")
                return

    print(" OK")


fn test_win_diagonal() raises:
    """Test P0 wins with diagonal: 0,4,8."""
    print("test_win_diagonal...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # P0:0, P1:1, P0:4, P1:2, P0:8 (P0 wins diagonal)
    var moves = List[Int](capacity=5)
    moves.append(0)  # P0
    moves.append(1)  # P1
    moves.append(4)  # P0
    moves.append(2)  # P1
    moves.append(8)  # P0 wins

    for m in range(len(moves)):
        var result = env._step_impl(moves[m])
        if m == 4:
            if not result[1]:
                print(" FAIL: game should be done")
                return
            if env.game_result() != 1:
                print(" FAIL: P0 should win")
                return

    print(" OK")


fn test_draw() raises:
    """Test draw when board is full with no winner."""
    print("test_draw...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # A known draw sequence:
    # P0:0, P1:1, P0:2, P1:4, P0:3, P1:6, P0:5, P1:8, P0:7
    # Board:  X O X
    #         X O X
    #         O X O
    var moves = List[Int](capacity=9)
    moves.append(0)  # P0
    moves.append(1)  # P1
    moves.append(2)  # P0
    moves.append(4)  # P1
    moves.append(3)  # P0
    moves.append(6)  # P1
    moves.append(5)  # P0
    moves.append(8)  # P1
    moves.append(7)  # P0 - draw (board full, no winner)

    var last_done = False
    for m in range(len(moves)):
        var result = env._step_impl(moves[m])
        last_done = result[1]

    if not last_done:
        print(" FAIL: game should be done")
        return
    if env.game_result() != 3:
        print(" FAIL: game_result should be 3 (draw), got", env.game_result())
        return

    print(" OK")


fn test_illegal_move() raises:
    """Test that placing on occupied cell returns -1 reward."""
    print("test_illegal_move...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # P0 plays center
    _ = env._step_impl(4)
    # P1 tries to play center again (illegal)
    var result = env._step_impl(4)
    if Float64(result[0]) != -1.0:
        print(" FAIL: illegal move should give -1 reward, got", result[0])
        return
    if result[1]:
        print(" FAIL: illegal move shouldn't end game")
        return

    print(" OK")


fn test_legal_mask_updates() raises:
    """Test that legal mask correctly reflects board state."""
    print("test_legal_mask_updates...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # Play center
    _ = env._step_impl(4)
    var mask = env.legal_action_mask()
    if mask[4]:
        print(" FAIL: center should be illegal after being played")
        return

    # Other cells should still be legal
    for i in range(9):
        if i != 4 and not mask[i]:
            print(" FAIL: cell", i, "should still be legal")
            return

    print(" OK")


fn test_canonical_obs() raises:
    """Test that observations flip correctly based on current player."""
    print("test_canonical_obs...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # P0 plays center (cell 4)
    _ = env._step_impl(4)

    # Now it's P1's turn. Obs should show:
    # Plane 0 (my pieces = P1): all zeros (P1 hasn't played)
    # Plane 1 (opp pieces = P0): 1.0 at index 4
    var obs = env.get_obs_list()

    # Plane 0: P1's pieces (none yet)
    for i in range(9):
        if Float64(obs[i]) != 0.0:
            print(" FAIL: P1 plane should be empty, but cell", i, "=", obs[i])
            return

    # Plane 1: P0's pieces (center)
    if Float64(obs[9 + 4]) != 1.0:
        print(" FAIL: P0's center mark should be in opponent plane")
        return

    # Plane 2: legal moves (8 empty cells)
    var legal_count = 0
    for i in range(9):
        if Float64(obs[18 + i]) == 1.0:
            legal_count += 1
    if legal_count != 8:
        print(" FAIL: should have 8 legal moves, got", legal_count)
        return

    print(" OK")


fn test_obs_dim() raises:
    """Test observation dimension."""
    print("test_obs_dim...", end="")
    var env = TicTacToeEnv[DType.float64]()
    if env.obs_dim() != 27:
        print(" FAIL: obs_dim should be 27, got", env.obs_dim())
        return
    _ = env.reset()
    var obs = env.get_obs_list()
    if len(obs) != 27:
        print(" FAIL: obs length should be 27, got", len(obs))
        return
    print(" OK")


fn test_step_obs_with_opponent() raises:
    """Test step_obs plays random opponent automatically."""
    print("test_step_obs_with_opponent...", end="")
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    # After step_obs, both P0 and P1 should have played (unless game ends)
    var result = env.step_obs(4)  # P0 plays center, then random opponent plays
    var obs = result[0].copy()

    # Should be P0's turn again (after P1 played)
    if not result[2]:  # if not done
        if env.current_player() != 0:
            print(" FAIL: should be P0's turn after step_obs")
            return

        # P0 should have 1 piece (center)
        # Check plane 0 (my pieces = P0) has center
        if Float64(obs[4]) != 1.0:
            print(" FAIL: P0 should see own center mark")
            return

        # P1 should have 1 piece somewhere in plane 1
        var opp_count = 0
        for i in range(9):
            if Float64(obs[9 + i]) == 1.0:
                opp_count += 1
        if opp_count != 1:
            print(" FAIL: opponent should have 1 piece, got", opp_count)
            return

    print(" OK")


fn test_random_game_completion() raises:
    """Test that random games always terminate."""
    print("test_random_game_completion...", end="")
    var env = TicTacToeEnv[DType.float64]()
    var games_completed = 0

    for game in range(100):
        _ = env.reset()
        for step in range(20):
            # Find a legal action
            var mask = env.legal_action_mask()
            var action = -1
            for a in range(9):
                if mask[a]:
                    action = a
                    break
            if action == -1:
                break  # No legal moves (shouldn't happen if not done)

            var result = env._step_impl(action)
            if result[1]:
                games_completed += 1
                break

    if games_completed != 100:
        print(
            " FAIL: all 100 games should complete, only",
            games_completed,
            "did",
        )
        return

    print(" OK")


fn main() raises:
    print("=== Testing TicTacToeEnv ===\n")

    test_reset()
    test_win_row()
    test_win_col()
    test_win_diagonal()
    test_draw()
    test_illegal_move()
    test_legal_mask_updates()
    test_canonical_obs()
    test_obs_dim()
    test_step_obs_with_opponent()
    test_random_game_completion()

    print("\n=== All TicTacToe tests passed ===")
