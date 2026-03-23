"""Evaluate trained MuZero/AlphaZero on TicTacToe against baselines.

Tests the agent against:
  1. Random opponent — should win almost every game
  2. Minimax (perfect play) — should draw every game if optimal

Measures: win rate, draw rate, loss rate for each matchup.

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_evaluate_muzero.mojo
"""

from std.random import random_float64
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent
from mojo_rl.deep_agents.muzero.configs import AlphaZeroConfig
from mojo_rl.nn.constants import dtype


# ═══════════════════════════════════════════════════════════════════════════
# Minimax Solver (perfect play for TicTacToe)
# ═══════════════════════════════════════════════════════════════════════════


def minimax(
    mut env: TicTacToeEnv[DType.float64],
    depth: Int,
    is_maximizing: Bool,
) -> Int:
    """Minimax with perfect play. Returns: +1 (maximizer wins), -1 (minimizer wins), 0 (draw).

    TicTacToe has at most 9 moves, so full tree search is trivial.
    Player 0 (X) is maximizer, Player 1 (O) is minimizer.
    """
    var result = env.game_result()
    if result == 1:
        return 1  # P0 (X) wins
    if result == 2:
        return -1  # P1 (O) wins
    if result == 3:
        return 0  # Draw

    # Check if all cells filled (shouldn't happen if game_result catches it)
    var legal = env.legal_action_mask()
    var any_legal = False
    for a in range(9):
        if legal[a]:
            any_legal = True
            break
    if not any_legal:
        return 0

    if is_maximizing:
        var best = -2
        for a in range(9):
            if not legal[a]:
                continue
            # Make a copy, play the move
            var child = TicTacToeEnv[DType.float64]()
            for i in range(12):
                child.state[i] = env.state[i]
            child.done = env.done
            _ = child._step_impl(a)
            var score = minimax(child, depth + 1, False)
            if score > best:
                best = score
        return best
    else:
        var best = 2
        for a in range(9):
            if not legal[a]:
                continue
            var child = TicTacToeEnv[DType.float64]()
            for i in range(12):
                child.state[i] = env.state[i]
            child.done = env.done
            _ = child._step_impl(a)
            var score = minimax(child, depth + 1, True)
            if score < best:
                best = score
        return best


def minimax_best_action(mut env: TicTacToeEnv[DType.float64]) -> Int:
    """Return the best action according to minimax."""
    var legal = env.legal_action_mask()
    var player = env.current_player()
    var is_max = player == 0
    var best_action = -1
    var best_score = -2 if is_max else 2

    for a in range(9):
        if not legal[a]:
            continue
        var child = TicTacToeEnv[DType.float64]()
        for i in range(12):
            child.state[i] = env.state[i]
        child.done = env.done
        _ = child._step_impl(a)
        var score = minimax(child, 0, not is_max)
        if is_max:
            if score > best_score:
                best_score = score
                best_action = a
        else:
            if score < best_score:
                best_score = score
                best_action = a

    return best_action


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def eval_vs_random(
    mut agent: GenericMuZeroAgent[
        AlphaZeroConfig[27, 9, HIDDEN=128, LR=1e-3, BS=64, SIMS=25, NODES=64], 1
    ],
    num_games: Int,
    agent_plays_first: Bool,
) -> Tuple[Int, Int, Int]:
    """Play agent vs random. Returns (wins, draws, losses)."""
    var wins = 0
    var draws = 0
    var losses = 0

    for _ in range(num_games):
        var env = TicTacToeEnv[DType.float64]()
        _ = env.reset()

        while not env.done:
            var player = env.current_player()
            var is_agent_turn = (player == 0 and agent_plays_first) or (
                player == 1 and not agent_plays_first
            )

            if is_agent_turn:
                # Agent plays
                var obs = List[Scalar[dtype]](capacity=27)
                var obs_raw = env.get_obs_list()
                for i in range(27):
                    if i < len(obs_raw):
                        obs.append(Scalar[dtype](obs_raw[i]))
                    else:
                        obs.append(Scalar[dtype](0.0))
                var result = agent.select_action(obs, training=False)
                var action = result[0]
                var legal = env.legal_action_mask()
                if action >= 0 and action < 9 and legal[action]:
                    _ = env._step_impl(action)
                else:
                    # Fallback to first legal
                    for a in range(9):
                        if legal[a]:
                            _ = env._step_impl(a)
                            break
            else:
                # Random opponent
                var legal = env.legal_action_mask()
                var n_legal = 0
                for a in range(9):
                    if legal[a]:
                        n_legal += 1
                var pick = Int(random_float64() * Float64(n_legal))
                if pick >= n_legal:
                    pick = n_legal - 1
                var count = 0
                for a in range(9):
                    if legal[a]:
                        if count == pick:
                            _ = env._step_impl(a)
                            break
                        count += 1

        var result = env.game_result()
        var agent_is_p0 = agent_plays_first
        if result == 1:  # P0 wins
            if agent_is_p0:
                wins += 1
            else:
                losses += 1
        elif result == 2:  # P1 wins
            if agent_is_p0:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

    return (wins, draws, losses)


def eval_vs_minimax(
    mut agent: GenericMuZeroAgent[
        AlphaZeroConfig[27, 9, HIDDEN=128, LR=1e-3, BS=64, SIMS=25, NODES=64], 1
    ],
    num_games: Int,
    agent_plays_first: Bool,
) -> Tuple[Int, Int, Int]:
    """Play agent vs perfect minimax. Returns (wins, draws, losses)."""
    var wins = 0
    var draws = 0
    var losses = 0

    for _ in range(num_games):
        var env = TicTacToeEnv[DType.float64]()
        _ = env.reset()

        while not env.done:
            var player = env.current_player()
            var is_agent_turn = (player == 0 and agent_plays_first) or (
                player == 1 and not agent_plays_first
            )

            if is_agent_turn:
                var obs = List[Scalar[dtype]](capacity=27)
                var obs_raw = env.get_obs_list()
                for i in range(27):
                    if i < len(obs_raw):
                        obs.append(Scalar[dtype](obs_raw[i]))
                    else:
                        obs.append(Scalar[dtype](0.0))
                var result = agent.select_action(obs, training=False)
                var action = result[0]
                var legal = env.legal_action_mask()
                if action >= 0 and action < 9 and legal[action]:
                    _ = env._step_impl(action)
                else:
                    for a in range(9):
                        if legal[a]:
                            _ = env._step_impl(a)
                            break
            else:
                # Minimax plays perfectly
                var action = minimax_best_action(env)
                _ = env._step_impl(action)

        var result = env.game_result()
        var agent_is_p0 = agent_plays_first
        if result == 1:
            if agent_is_p0:
                wins += 1
            else:
                losses += 1
        elif result == 2:
            if agent_is_p0:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

    return (wins, draws, losses)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  Evaluate MuZero on TicTacToe                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Load agent
    comptime Config = AlphaZeroConfig[
        27, 9, HIDDEN=128, LR=1e-3, BS=64, SIMS=25, NODES=64
    ]
    var agent = GenericMuZeroAgent[Config, 1](gamma=1.0, v_min=-1.0, v_max=1.0)

    var ckpt_path = "tictactoe_muzero.ckpt"
    print("Loading:", ckpt_path)
    agent.load_checkpoint(ckpt_path)
    print("Loaded (train steps:", agent.train_step_count, ")")
    print()

    # ── 1. Agent vs Random ───────────────────────────────────────
    comptime N_GAMES = 100
    print("=== Agent vs Random ===")

    var r1 = eval_vs_random(agent, N_GAMES, agent_plays_first=True)
    print(
        "Agent as X (first): W:",
        r1[0],
        "D:",
        r1[1],
        "L:",
        r1[2],
        "| Win%:",
        r1[0] * 100 // N_GAMES,
    )

    var r2 = eval_vs_random(agent, N_GAMES, agent_plays_first=False)
    print(
        "Agent as O (second): W:",
        r2[0],
        "D:",
        r2[1],
        "L:",
        r2[2],
        "| Win%:",
        r2[0] * 100 // N_GAMES,
    )

    print()

    # ── 2. Agent vs Minimax (perfect play) ───────────────────────
    print("=== Agent vs Minimax (perfect) ===")

    var m1 = eval_vs_minimax(agent, N_GAMES, agent_plays_first=True)
    print(
        "Agent as X (first): W:",
        m1[0],
        "D:",
        m1[1],
        "L:",
        m1[2],
        "| Draw%:",
        m1[1] * 100 // N_GAMES,
    )

    var m2 = eval_vs_minimax(agent, N_GAMES, agent_plays_first=False)
    print(
        "Agent as O (second): W:",
        m2[0],
        "D:",
        m2[1],
        "L:",
        m2[2],
        "| Draw%:",
        m2[1] * 100 // N_GAMES,
    )

    print()

    # ── Interpretation ───────────────────────────────────────────
    var total_draws_vs_minimax = m1[1] + m2[1]
    var total_games_vs_minimax = 2 * N_GAMES
    print("═══════════════════════════════════════════════════")
    if total_draws_vs_minimax == total_games_vs_minimax:
        print("PERFECT: Agent draws every game against minimax!")
        print("  → Optimal TicTacToe play achieved.")
    elif total_draws_vs_minimax > Int(total_games_vs_minimax * 0.9):
        print(
            "STRONG: Agent draws",
            total_draws_vs_minimax,
            "/",
            total_games_vs_minimax,
            "games against minimax",
        )
        print("  → Near-optimal play. A few mistakes remain.")
    elif total_draws_vs_minimax > Int(total_games_vs_minimax * 0.5):
        print(
            "DECENT: Agent draws",
            total_draws_vs_minimax,
            "/",
            total_games_vs_minimax,
            "games against minimax",
        )
        print("  → Moderate strength. Needs more training.")
    else:
        print(
            "WEAK: Agent draws only",
            total_draws_vs_minimax,
            "/",
            total_games_vs_minimax,
            "games against minimax",
        )
        print("  → Agent still makes many mistakes. Try training longer.")
    print("═══════════════════════════════════════════════════")
