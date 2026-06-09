"""AlphaZero CPU self-play path — learns TicTacToe with no GPU.

Validates the `TARGET="cpu"` route end-to-end: `AlphaZeroAgent["cpu", ...]`
builds a host-resident net (ctx=None), `train` drives `run_alphazero_selfplay_cpu`
(single-env `GenericCPUMCTS` with the true-rules adapters + the shared nn2 AZ
loss graph on `forward/vjp["cpu"]`), and the trained net's greedy policy clearly
beats a random opponent — the CPU "did it learn?" signal. No `DeviceContext` is
ever created.

Run (no GPU needed):
    pixi run mojo run -I . tests/deep_agents2/test_az_cpu_selfplay.mojo
"""

from std.testing import assert_true

from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.alphazero.agent import AlphaZeroAgent
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime N_EVAL = 200
    comptime MAX_PLIES = 9

    # CPU agent — N_ENVS is irrelevant on the single-env CPU path (pass 1).
    var agent = AlphaZeroAgent[
        "cpu", Env, Net, 1, 24, 64, 64, 16384, 16
    ](None, lr=0.01)

    var before = agent.eval_vs_random_cpu[N_EVAL, MAX_PLIES](
        agent_player=0, seed=12345
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")",
    )

    var loss = agent.train(
        iterations=3000, learning_starts=20, train_per_iter=2, seed=7
    )
    print("CPU train  last_loss=", loss)

    var after = agent.eval_vs_random_cpu[N_EVAL, MAX_PLIES](
        agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    # Training produced a finite loss.
    assert_true(
        loss == loss and loss < 1e30, "CPU training produced non-finite loss"
    )
    # The CPU-trained greedy policy clearly improved vs random — a substantial
    # loss-rate reduction (≥25%; this single-env / 24-sim budget reaches ~26%
    # loss vs random, down from ~50%).
    assert_true(
        after.losses * 4 < before.losses * 3,
        "CPU-trained agent did not clearly reduce losses vs random (>=25%)",
    )
    assert_true(
        after.losses < N_EVAL // 3,
        "CPU-trained agent still loses too often vs random",
    )
    print("AZ CPU self-play: OK")
