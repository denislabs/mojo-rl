"""Smoke: arena-gated Gumbel AlphaZero (train_arena_gumbel) on TicTacToe.

Exercises the full production path — Gumbel self-play (improved-policy
targets, TEMP_MOVES sampling), PUCT-MCTS arena gating, periodic MCTS eval vs
random, symmetry augmentation — at a small budget. Asserts the loop runs,
trains, and the agent improves vs random.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_gumbel_arena_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents.zero.symmetries import D4SquareAugmenter
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime Net = AZMLPNet[27, 9, 64]
    comptime Env = TicTacToeEnv[DType.float64]

    var ctx = DeviceContext()
    var agent = AlphaZeroAgent[
        "gpu", Env, Net,
        N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, lr=0.01)

    var before = eval_policy_vs_random[Env, Net, 100, 10, 9](
        ctx, agent.net, agent_player=0, seed=3
    )
    var res = agent.train_arena_gumbel[
        AUG=D4SquareAugmenter[SIDE=3, PLANES=3],
        ARENA_GAMES=16,
        RESULT_IDX=10,
        MAX_PLIES=9,
        EVAL_GAMES=16,
        TEMP_MOVES=4,
        MAX_K=4,
    ](
        iterations=800,
        learning_starts=20,
        train_per_iter=2,
        seed=7,
        arena_every=300,
        arena_open_plies=2,
        report_every=400,
        verbose=True,
    )
    var after = eval_policy_vs_random[Env, Net, 100, 10, 9](
        ctx, agent.net, agent_player=0, seed=3
    )
    print("loss:", res.last_loss, "| promotions:", res.promotions)
    print("vs random losses:", before.losses, "->", after.losses, "(/100)")
    assert_true(res.last_loss > 0.0, "training never ran")
    assert_true(after.losses < before.losses, "no improvement vs random")
    print("Gumbel AlphaZero arena smoke: OK")
