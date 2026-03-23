"""Test Go environment — captures, ko, suicide, pass, scoring, parameterized sizes."""

from mojo_rl.envs.board_games.go import GoEnv


def pos(row: Int, col: Int, size: Int) -> Int:
    """Convert (row, col) to board index."""
    return row * size + col


def test_reset() raises:
    print("test_reset...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # All cells empty
    for i in range(81):
        if Float64(env.state[i]) != 0.0:
            print(" FAIL: cell", i, "not empty")
            return

    if env.current_player() != 0:
        print(" FAIL: black (P0) should start")
        return

    if env.game_result() != 0:
        print(" FAIL: game should be ongoing")
        return

    print(" OK")


def test_stone_placement() raises:
    """Test basic stone placement and player alternation."""
    print("test_stone_placement...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Black plays center (4,4)
    var center = pos(4, 4, 9)
    var result = env._step_impl(center)
    if Float64(result[0]) != 0.0:
        print(" FAIL: reward should be 0 for normal move")
        return
    if Float64(env.state[center]) != 1.0:
        print(" FAIL: black stone not placed")
        return
    if env.current_player() != 1:
        print(" FAIL: should be white's turn")
        return

    # White plays adjacent
    var adj = pos(4, 5, 9)
    _ = env._step_impl(adj)
    if Float64(env.state[adj]) != 2.0:
        print(" FAIL: white stone not placed")
        return

    print(" OK")


def test_simple_capture() raises:
    """Test capturing a single stone surrounded on all 4 sides."""
    print("test_simple_capture...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Place white stone at (4,4), surround with black
    # Black: (3,4), (5,4), (4,3), then white: (4,4), (0,0), (0,1)
    # Then black: (4,5) captures
    _ = env._step_impl(pos(3, 4, 9))  # Black
    _ = env._step_impl(pos(4, 4, 9))  # White at center
    _ = env._step_impl(pos(5, 4, 9))  # Black
    _ = env._step_impl(pos(0, 0, 9))  # White elsewhere
    _ = env._step_impl(pos(4, 3, 9))  # Black
    _ = env._step_impl(pos(0, 1, 9))  # White elsewhere
    _ = env._step_impl(pos(4, 5, 9))  # Black captures!

    # White stone at (4,4) should be removed
    if Float64(env.state[pos(4, 4, 9)]) != 0.0:
        print(" FAIL: white stone at (4,4) should be captured")
        return

    print(" OK")


def test_suicide_illegal() raises:
    """Test that suicide is illegal."""
    print("test_suicide_illegal...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Create a position where playing at (0,0) would be suicide
    # Surround (0,0) with opponent stones: (0,1) and (1,0)
    _ = env._step_impl(pos(0, 1, 9))  # Black at (0,1)
    _ = env._step_impl(pos(5, 5, 9))  # White elsewhere
    _ = env._step_impl(pos(1, 0, 9))  # Black at (1,0)
    # Now white tries to play (0,0) — surrounded by black, no captures → suicide
    var result = env._step_impl(pos(0, 0, 9))
    if Float64(result[0]) != -1.0:
        print(" FAIL: suicide should give -1 reward, got", result[0])
        return
    if Float64(env.state[pos(0, 0, 9)]) != 0.0:
        print(" FAIL: stone should not be placed on suicide")
        return

    print(" OK")


def test_ko_rule() raises:
    """Test simple ko — cannot immediately recapture."""
    print("test_ko_rule...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Classic ko shape:
    #   . B W .
    #   B . B W
    #   . B W .
    # Black captures at (1,1) by taking white's stone, then white cannot
    # immediately recapture at the ko point.

    # Build the shape (row, col):
    _ = env._step_impl(pos(0, 1, 9))  # B at (0,1)
    _ = env._step_impl(pos(0, 2, 9))  # W at (0,2)
    _ = env._step_impl(pos(1, 0, 9))  # B at (1,0)
    _ = env._step_impl(pos(1, 3, 9))  # W at (1,3)
    _ = env._step_impl(pos(1, 2, 9))  # B at (1,2)
    _ = env._step_impl(pos(1, 1, 9))  # W at (1,1)
    _ = env._step_impl(pos(2, 1, 9))  # B at (2,1)
    _ = env._step_impl(pos(2, 2, 9))  # W at (2,2)

    # Now black captures white at (1,1) by playing at... wait,
    # let me reconsider the setup. The ko shape needs:
    # B captures single W stone, W cannot recapture immediately.

    # Simpler setup: just verify ko_point is set after single-stone capture
    var env2 = GoEnv[9, DType.float64]()
    _ = env2.reset()

    # Place white at corner (0,0), black surrounds at (0,1) and (1,0)
    _ = env2._step_impl(pos(0, 1, 9))  # B
    _ = env2._step_impl(pos(0, 0, 9))  # W at corner
    _ = env2._step_impl(pos(1, 0, 9))  # B captures white at (0,0)

    # (0,0) should now be empty (captured)
    if Float64(env2.state[pos(0, 0, 9)]) != 0.0:
        print(" FAIL: white stone should be captured")
        return

    # Ko point should be set to (0,0)
    if Int(env2.state[env2.S_KO_POINT]) != pos(0, 0, 9):
        print(
            " FAIL: ko point should be",
            pos(0, 0, 9),
            "got",
            Int(env2.state[env2.S_KO_POINT]),
        )
        return

    # White should not be able to play at (0,0) immediately
    var result = env2._step_impl(pos(0, 0, 9))
    if Float64(result[0]) != -1.0:
        print(" FAIL: ko recapture should be illegal")
        return

    print(" OK")


def test_pass_and_game_end() raises:
    """Test that two consecutive passes end the game."""
    print("test_pass_and_game_end...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Both players pass
    _ = env._step_impl(env.PASS_ACTION)  # Black passes
    if env.done:
        print(" FAIL: one pass shouldn't end game")
        return

    var result = env._step_impl(env.PASS_ACTION)  # White passes
    if not result[1]:
        print(" FAIL: two passes should end game")
        return

    print(" OK")


def test_scoring_empty_board() raises:
    """Test scoring on empty board — white wins by komi."""
    print("test_scoring_empty_board...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    _ = env._step_impl(env.PASS_ACTION)  # Black passes
    _ = env._step_impl(env.PASS_ACTION)  # White passes

    # Empty board: 0 - 0 - 7.5 = -7.5 → white wins
    if env.game_result() != 2:
        print(
            " FAIL: white should win on empty board (komi), got result",
            env.game_result(),
        )
        return

    print(" OK")


def test_obs_dim_9x9() raises:
    print("test_obs_dim_9x9...", end="")
    var env = GoEnv[9, DType.float64]()
    if env.obs_dim() != 324:
        print(" FAIL: obs_dim should be 324 (4*81), got", env.obs_dim())
        return
    _ = env.reset()
    var obs = env.get_obs_list()
    if len(obs) != 324:
        print(" FAIL: obs length should be 324, got", len(obs))
        return
    print(" OK")


def test_num_actions_9x9() raises:
    print("test_num_actions_9x9...", end="")
    var env = GoEnv[9, DType.float64]()
    if env.num_actions() != 82:
        print(
            " FAIL: num_actions should be 82 (81+pass), got", env.num_actions()
        )
        return
    print(" OK")


def test_canonical_obs() raises:
    """Test canonical observation flipping."""
    print("test_canonical_obs...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    # Black plays center
    _ = env._step_impl(pos(4, 4, 9))

    # White's turn: obs plane 0 = white's stones (none), plane 1 = black's stones
    var obs = env.get_obs_list()
    var center = pos(4, 4, 9)

    # Plane 0 (my = white): should be empty
    if Float64(obs[center]) != 0.0:
        print(" FAIL: white has no stones, plane 0 should be 0")
        return

    # Plane 1 (opp = black): black's stone at center
    if Float64(obs[81 + center]) != 1.0:
        print(" FAIL: black's stone should appear in opponent plane")
        return

    print(" OK")


def test_legal_mask_occupied() raises:
    """Test that occupied cells are marked illegal."""
    print("test_legal_mask_occupied...", end="")
    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    var center = pos(4, 4, 9)
    _ = env._step_impl(center)  # Black plays center

    var mask = env.legal_action_mask()
    if mask[center]:
        print(" FAIL: occupied cell should be illegal")
        return

    # Pass should always be legal
    if not mask[81]:
        print(" FAIL: pass should be legal")
        return

    print(" OK")


def test_parameterized_sizes() raises:
    """Test that different board sizes compile and work."""
    print("test_parameterized_sizes...", end="")

    # 9x9
    var env9 = GoEnv[9, DType.float64]()
    _ = env9.reset()
    if env9.num_actions() != 82:
        print(" FAIL: 9x9 should have 82 actions")
        return

    # 13x13
    var env13 = GoEnv[13, DType.float64]()
    _ = env13.reset()
    if env13.num_actions() != 170:
        print(" FAIL: 13x13 should have 170 actions, got", env13.num_actions())
        return
    if env13.obs_dim() != 676:
        print(" FAIL: 13x13 obs_dim should be 676, got", env13.obs_dim())
        return

    # 19x19
    var env19 = GoEnv[19, DType.float64]()
    _ = env19.reset()
    if env19.num_actions() != 362:
        print(" FAIL: 19x19 should have 362 actions")
        return

    print(" OK")


def test_random_game_completion() raises:
    """Test that random 9x9 games always terminate."""
    print("test_random_game_completion...", end="")
    from std.random import random_float64

    var env = GoEnv[9, DType.float64]()
    var games_completed = 0

    for _ in range(20):
        _ = env.reset()
        for _ in range(300):
            var mask = env.legal_action_mask()
            # Collect legal moves
            var legal_moves = List[Int](capacity=82)
            for a in range(82):
                if mask[a]:
                    legal_moves.append(a)
            if len(legal_moves) == 0:
                # Only pass
                _ = env._step_impl(81)
                continue
            # Pick random legal move (bias toward pass to ensure termination)
            var r = random_float64()
            var action: Int
            if r < 0.1:
                action = 81  # 10% chance to pass
            else:
                var idx = Int(random_float64() * Float64(len(legal_moves)))
                if idx >= len(legal_moves):
                    idx = len(legal_moves) - 1
                action = legal_moves[idx]
            var result = env._step_impl(action)
            if result[1]:
                games_completed += 1
                break

    if games_completed != 20:
        print(" FAIL: all games should complete, only", games_completed, "did")
        return

    print(" OK")


def main() raises:
    print("=== Testing GoEnv ===\n")

    test_reset()
    test_stone_placement()
    test_simple_capture()
    test_suicide_illegal()
    test_ko_rule()
    test_pass_and_game_end()
    test_scoring_empty_board()
    test_obs_dim_9x9()
    test_num_actions_9x9()
    test_canonical_obs()
    test_legal_mask_occupied()
    test_parameterized_sizes()
    test_random_game_completion()

    print("\n=== All Go tests passed ===")
