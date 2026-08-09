"""Connect4 lighthouse: arena-gated AlphaZero + horizontal-flip augmentation.

Second game for the AlphaZero arc (after TicTacToe). Validates the whole stack
is **env-agnostic** — the MCTS adapters, self-play driver, arena gating, and the
`HFlipColumnAugmenter` (Connect4's single board symmetry) all run unchanged on
the 6×7 board (126D obs, 7 column actions, 42-ply games).

The convergence signal here is the AlphaZero-native one — **does the trained
best beat an untrained net head-to-head?** — plus at least one accepted arena
promotion (the learner overtaking the frozen best). We do NOT assert a
greedy-from-start win-rate vs random: on a light compute budget (32 sims, small
MLP) Connect4 is far from solved, and a random-init net's argmax is already a
deceptively strong P0 baseline, so that metric is confounded. The vs-random
numbers are printed as diagnostics only.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_connect4_arena.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_arena import (
    run_alphazero_selfplay_arena,
)
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.deep_agents.alphazero.arena import candidate_winrate
from mojo_rl.nn.core.hard_copy import hard_copy
from mojo_rl.deep_agents.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    comptime OBS = 126
    comptime ACT = 7
    comptime H = 128
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = ConnectFourEnv[DType.float64]
    comptime Aug = HFlipColumnAugmenter[6, 7, 3]   # Connect4: identity + h-flip
    comptime N_EVAL = 128
    comptime RESULT_IDX = 43
    comptime MAX_PLIES = 42

    var ctx = DeviceContext()
    var best = Net.make["gpu", Kaiming](Optional(ctx))
    # Frozen pre-training snapshot of `best` (same initial weights) for the
    # head-to-head "did it learn?" check.
    var untrained = Net.make["gpu", Kaiming](Optional(ctx))
    hard_copy["gpu"](best, untrained, Optional(ctx))

    var before = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, best, agent_player=0, seed=12345
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")  [diagnostic]",
    )

    var res = run_alphazero_selfplay_arena[
        Env, Net, Aug, N_ENVS=32, NUM_SIMS=32, MAX_NODES=96,
        BATCH=128, CAP=49152, MAX_TRAJ=42,
        ARENA_GAMES=32, RESULT_IDX=RESULT_IDX, MAX_PLIES=MAX_PLIES,
    ](
        ctx, best, iterations=2000, learning_starts=20, train_per_iter=2,
        lr=2e-3, seed=7, arena_every=700, arena_open_plies=4,
        promote_threshold=0.55,
    )
    print("arena run  last_loss=", res.last_loss, " promotions=", res.promotions)

    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, best, agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")  [diagnostic]",
    )

    # AlphaZero-native learning signal: the trained best vs its untrained self,
    # head-to-head from both colors with diverse openings.
    var h2h = candidate_winrate[
        Env, Net, Net, 32, RESULT_IDX, MAX_PLIES
    ](ctx, best, untrained, seed=4242, open_plies=4)
    print(
        "best vs untrained (h2h)  win=", h2h.wins, " draw=", h2h.draws,
        " loss=", h2h.losses,
    )

    # The learner must overtake the frozen best at least once during the run.
    assert_true(res.promotions >= 1, "learner was never promoted over best")
    # Training must produce finite loss (the pipeline runs cleanly on Connect4).
    assert_true(
        res.last_loss == res.last_loss and res.last_loss < 1e30,
        "Connect4 training produced non-finite loss",
    )
    # The trained best must out-score its untrained self head-to-head.
    assert_true(
        h2h.wins > h2h.losses,
        "trained Connect4 net did not beat its untrained self head-to-head",
    )
    print("AZ Connect4 arena + h-flip augmentation: OK")
