"""AlphaZero strength vs a perfect minimax opponent.

Plugs the `GPUMinimaxTicTacToe` perfect-play evaluator into the eval harness.
Minimax is deterministic, so greedy-vs-minimax from the fixed start is a single
line — we randomise the first `OPEN` plies (`open_plies`) to spread the batch
across diverse openings. From positions that are still drawable, optimal
TicTacToe never loses, so a *learning* agent's loss-rate vs perfect play must
fall: we compare a fresh (random-init) net to a self-play-trained net.

Two signals:
  * Diverse-opening loss-rate vs minimax drops markedly after training.
  * From the canonical start (no random opening) the trained agent, as P0,
    draws perfect play (the textbook "optimal never loses" property).

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_az_eval_minimax.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.alphazero.selfplay import run_alphazero_selfplay
from mojo_rl.deep_agents2.alphazero.eval import (
    eval_policy_vs_opponent, eval_policy_vs_random, eval_mcts_vs_opponent,
)
from mojo_rl.deep_agents2.zero.evaluators import (
    GPUMinimaxTicTacToe, RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9
    comptime NG = 256        # diverse openings → meaningful distribution
    comptime OPEN = 2        # 2 random opening plies before greedy vs minimax
    comptime NG_RND = 200

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT=Kaiming](ctx=ctx)

    var mm_before = eval_policy_vs_opponent[
        Env, Net, GPUMinimaxTicTacToe, NG, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=99, open_plies=OPEN)
    print(
        "vs MINIMAX (diverse) before win=", mm_before.wins,
        " draw=", mm_before.draws, " loss=", mm_before.losses, " /", NG,
    )

    _ = run_alphazero_selfplay[
        Env, Net, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, net, iterations=1200, learning_starts=20, train_per_iter=2,
      lr=0.01, seed=7)

    var mm_after = eval_policy_vs_opponent[
        Env, Net, GPUMinimaxTicTacToe, NG, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=99, open_plies=OPEN)
    print(
        "vs MINIMAX (diverse) after  win=", mm_after.wins,
        " draw=", mm_after.draws, " loss=", mm_after.losses, " /", NG,
    )

    # Canonical start, no random opening: trained P0 should draw perfect play.
    var mm_line = eval_policy_vs_opponent[
        Env, Net, GPUMinimaxTicTacToe, 8, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=1, open_plies=0)
    print(
        "vs MINIMAX (canonical P0) win=", mm_line.wins,
        " draw=", mm_line.draws, " loss=", mm_line.losses,
    )

    # Same canonical-line check, but the agent plays at full MCTS strength
    # (temp=0) — the eval the production driver/telemetry uses. A net that draws
    # perfect play via the bare policy head must also draw it with MCTS on top;
    # this is the end-to-end correctness check for `eval_mcts_vs_opponent`.
    var mcts_line = eval_mcts_vs_opponent[
        Env, Net, GPUMinimaxTicTacToe, 16, 24, 64, MAX_PLIES
    ](ctx, net, agent_player=0, seed=1)
    print(
        "vs MINIMAX (MCTS canonical P0) win=", mcts_line.wins,
        " draw=", mcts_line.draws, " loss=", mcts_line.losses,
    )

    # Cross-check both random-eval paths agree the trained agent is strong.
    var rnd = eval_policy_vs_opponent[
        Env, Net, RandomOpponent, NG_RND, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=12345, open_plies=0)
    var rnd_cpu = eval_policy_vs_random[
        Env, Net, NG_RND, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=12345)
    print(
        "vs RANDOM(gpu) win=", rnd.wins, " draw=", rnd.draws,
        " loss=", rnd.losses,
    )
    print(
        "vs RANDOM(cpu) win=", rnd_cpu.wins, " draw=", rnd_cpu.draws,
        " loss=", rnd_cpu.losses,
    )

    # 1. Diverse-opening: strictly fewer losses to perfect play, and a clear
    #    rise in salvaged non-losses (win+draw). After 2 random opening plies
    #    many positions are already lost regardless of play, so the loss count
    #    stays high — the learning signal is converting drawable positions into
    #    draws/wins (non-losses), not zeroing out the losses.
    var nonloss_before = mm_before.wins + mm_before.draws
    var nonloss_after = mm_after.wins + mm_after.draws
    assert_true(
        mm_after.losses < mm_before.losses,
        "trained agent did not reduce losses vs perfect minimax",
    )
    assert_true(
        nonloss_after * 100 >= nonloss_before * 115,
        "non-loss rate vs minimax did not clearly improve (>=+15%)",
    )
    # 2. Canonical-start: trained P0 never loses to perfect play, by both the
    #    bare-policy argmax and the full-MCTS eval (the latter validates
    #    `eval_mcts_vs_opponent` end-to-end).
    assert_true(
        mm_line.losses == 0,
        "trained agent lost the canonical line to minimax as P0",
    )
    assert_true(
        mcts_line.losses == 0,
        "trained agent (MCTS) lost the canonical line to minimax as P0",
    )
    # 3. Clearly strong vs random on both eval paths.
    assert_true(
        rnd.losses < NG_RND // 4 and rnd_cpu.losses < NG_RND // 4,
        "trained agent not clearly winning vs random on both eval paths",
    )
    print("AZ eval vs minimax: OK")
