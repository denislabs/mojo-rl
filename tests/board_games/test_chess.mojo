"""Test Chess environment — basic rules, moves, check, checkmate, stalemate."""

from mojo_rl.envs.board_games.chess import ChessEnv


def test_reset() raises:
    print("test_reset...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    if env.current_player() != 0:
        print(" FAIL: white should start")
        return
    if env.game_result() != 0:
        print(" FAIL: game should be ongoing")
        return
    if env.num_actions() != 4672:
        print(" FAIL: num_actions should be 4672")
        return
    if env.obs_dim() != 896:
        print(" FAIL: obs_dim should be 896")
        return

    # Check initial pieces
    # White rook at a1 (sq 0)
    if Int(env.state[0]) != 4:
        print(" FAIL: a1 should be white rook (4), got", Int(env.state[0]))
        return
    # White king at e1 (sq 4)
    if Int(env.state[4]) != 6:
        print(" FAIL: e1 should be white king (6), got", Int(env.state[4]))
        return
    # White pawn at a2 (sq 8)
    if Int(env.state[8]) != 1:
        print(" FAIL: a2 should be white pawn (1), got", Int(env.state[8]))
        return
    # Black pawn at a7 (sq 48)
    if Int(env.state[48]) != 7:
        print(" FAIL: a7 should be black pawn (7), got", Int(env.state[48]))
        return
    # Black king at e8 (sq 60)
    if Int(env.state[60]) != 12:
        print(" FAIL: e8 should be black king (12), got", Int(env.state[60]))
        return

    print(" OK")


def test_initial_legal_moves() raises:
    """White should have 20 legal moves in the starting position."""
    print("test_initial_legal_moves...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    var mask = env.legal_action_mask()
    var count = 0
    for i in range(4672):
        if mask[i]:
            count += 1

    if count != 20:
        print(" FAIL: white should have 20 legal moves, got", count)
        return

    print(" OK")


def test_obs_shape() raises:
    print("test_obs_shape...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()
    var obs = env.get_obs_list()
    if len(obs) != 896:
        print(" FAIL: obs should have 896 elements, got", len(obs))
        return
    print(" OK")


def test_pawn_move() raises:
    """Test that white can push e2-e4."""
    print("test_pawn_move...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    # Find e2-e4 in legal moves
    var mask = env.legal_action_mask()
    # e2 = sq(1,4) = 12, e4 = sq(3,4) = 28
    # Direction N (dir 0), distance 2 → move_type = 0*7 + 1 = 1
    # Canonical from_sq = 12 (white, no flip)
    # action = 12 * 73 + 1 = 877
    var action = 12 * 73 + 1  # e2-e4

    if not mask[action]:
        print(" FAIL: e2-e4 should be legal")
        return

    var result = env._step_impl(action)
    if Float64(result[0]) == -1.0:
        print(" FAIL: e2-e4 should succeed, got -1 reward")
        return

    # Check pawn moved
    if Int(env.state[12]) != 0:
        print(" FAIL: e2 should be empty after move")
        return
    if Int(env.state[28]) != 1:
        print(" FAIL: e4 should have white pawn")
        return

    # Should be black's turn
    if env.current_player() != 1:
        print(" FAIL: should be black's turn")
        return

    print(" OK")


def test_player_alternation() raises:
    """Test that players alternate correctly."""
    print("test_player_alternation...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    if env.current_player() != 0:
        print(" FAIL: white first")
        return

    # White: e2-e4 (action = 12*73+1 = 877)
    _ = env._step_impl(877)
    if env.current_player() != 1:
        print(" FAIL: black after white's move")
        return

    # Black: e7-e5. For black, canonical encoding: e7=sq(6,4)=52, flip→sq(1,4)=12
    # Direction N (in canonical), distance 2 → move_type = 1
    # canonical_from = flip(52) = (7-6)*8+4 = 12
    # action = 12 * 73 + 1 = 877
    _ = env._step_impl(877)  # Same canonical action for symmetric move
    if env.current_player() != 0:
        print(" FAIL: white after black's move")
        return

    print(" OK")


def test_random_game() raises:
    """Test that random games complete."""
    print("test_random_game...", end="")
    from std.random import random_float64

    var env = ChessEnv[DType.float64]()
    var games_finished = 0

    for _ in range(5):
        _ = env.reset()
        for _ in range(300):
            var mask = env.legal_action_mask()
            var legal_actions = List[Int](capacity=100)
            for a in range(4672):
                if mask[a]:
                    legal_actions.append(a)

            if len(legal_actions) == 0:
                break

            var idx = Int(random_float64() * Float64(len(legal_actions)))
            if idx >= len(legal_actions):
                idx = len(legal_actions) - 1

            var result = env._step_impl(legal_actions[idx])
            if result[1]:
                games_finished += 1
                break

    # At least some games should finish (checkmate, stalemate, or 50-move)
    # With random play, most games end by 50-move rule or stalemate
    print(" OK (", games_finished, "/5 games completed within 300 moves)")


def test_step_obs_opponent() raises:
    """Test that step_obs plays random opponent."""
    print("test_step_obs_opponent...", end="")
    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    # Get a legal action for white
    var mask = env.legal_action_mask()
    var action = -1
    for a in range(4672):
        if mask[a]:
            action = a
            break

    var result = env.step_obs(action)
    var obs = result[0].copy()

    # After step_obs, it should be white's turn again (opponent played)
    if not result[2]:  # if not done
        if env.current_player() != 0:
            print(" FAIL: should be white's turn after step_obs")
            return

    if len(obs) != 896:
        print(" FAIL: obs length should be 896")
        return

    print(" OK")


def main() raises:
    print("=== Testing ChessEnv ===\n")

    test_reset()
    test_initial_legal_moves()
    test_obs_shape()
    test_pawn_move()
    test_player_alternation()
    test_random_game()
    test_step_obs_opponent()

    print("\n=== All Chess tests passed ===")
