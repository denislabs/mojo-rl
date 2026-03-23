"""Self-play training loop for two-player board game environments.

Uses a single network (MuZero/AlphaZero) to play both sides via canonical
observations. The environment always presents observations from the
current player's perspective, so the same network can play both colors.

Key differences from standard training loops:
- legal_masks flow alongside obs from env to agent
- One network plays both sides (canonical obs)
- Reward is sparse: 0 during game, +1/-1 at terminal
- Finished games auto-reset via selective_reset

Usage (CPU, with MuZero):
    from mojo_rl.deep_agents.muzero import MuZeroAgent
    from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv

    var env = TicTacToeEnv[DType.float64]()
    var agent = MuZeroAgent[obs_dim=27, action_dim=9]()
    var metrics = self_play_train_cpu(env, agent, total_games=10000)
"""

from mojo_rl.core import TrainingMetrics, TwoPlayerDiscreteEnv
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.nn.constants import dtype
from std.random import random_float64


def self_play_train_cpu[
    E: TwoPlayerDiscreteEnv,
](
    mut env: E,
    total_games: Int = 10000,
    print_every: Int = 100,
) -> TrainingMetrics:
    """Run self-play training loop on CPU using random policy.

    This is a validation loop — it plays random games and collects statistics.
    For actual training, use MuZero's train() method with board game envs.

    Args:
        env: A TwoPlayerDiscreteEnv environment.
        total_games: Number of games to play.
        print_every: Print progress every N games.

    Returns:
        TrainingMetrics with game statistics.
    """
    var metrics = TrainingMetrics(algorithm_name="SelfPlay")
    var p0_wins = 0
    var p1_wins = 0
    var draws = 0

    for game in range(total_games):
        _ = env.reset()
        var game_steps = 0

        while True:
            # Get legal actions
            var mask = env.legal_action_mask()
            var legal_moves = List[Int](capacity=env.num_actions())
            for a in range(env.num_actions()):
                if mask[a]:
                    legal_moves.append(a)

            if len(legal_moves) == 0:
                break

            # Random action selection
            var idx = Int(random_float64() * Float64(len(legal_moves)))
            if idx >= len(legal_moves):
                idx = len(legal_moves) - 1
            var action = legal_moves[idx]

            # Step (using _step_impl for two-player mode, not step_obs)
            var result = env.step(env.action_from_index(action))
            game_steps += 1

            if result[2]:  # done
                var gr = env.game_result()
                if gr == 1:
                    p0_wins += 1
                elif gr == 2:
                    p1_wins += 1
                else:
                    draws += 1
                metrics.log_episode(game, Float64(result[1]), game_steps, 0.0)
                break

            if game_steps > 1000:
                # Safety: force game end
                draws += 1
                metrics.log_episode(game, 0.0, game_steps, 0.0)
                break

        if (game + 1) % print_every == 0:
            var total = p0_wins + p1_wins + draws
            print(
                "Game",
                game + 1,
                "/ ",
                total_games,
                " | P0 wins:",
                p0_wins,
                "P1 wins:",
                p1_wins,
                "Draws:",
                draws,
            )

    print(
        "\nFinal: P0 wins:",
        p0_wins,
        "P1 wins:",
        p1_wins,
        "Draws:",
        draws,
        "/ Total:",
        total_games,
    )
    return metrics^
