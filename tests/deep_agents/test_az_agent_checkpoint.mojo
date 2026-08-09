"""AlphaZeroAgent facade + checkpoint round-trip.

Trains a short session, evaluates, saves, reloads into a FRESH agent, and
asserts the reloaded net reproduces the exact same (deterministic) eval —
i.e. save/load preserves the model byte-for-byte.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_agent_checkpoint.mojo
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.agent import AlphaZeroAgent
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Agent = AlphaZeroAgent["gpu", Env, Net, 16, 16, 64, 32, 4096, 16]
    comptime N_EVAL = 200
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9
    comptime CKPT = "/tmp/az_ttt_agent.ckpt"

    var ctx = DeviceContext()

    var agent = Agent(ctx, lr=0.01)
    _ = agent.train(
        iterations=300, learning_starts=20, train_per_iter=2, seed=7
    )
    var e1 = agent.eval_vs_random[N_EVAL, RESULT_IDX, MAX_PLIES](
        agent_player=0, seed=12345
    )
    print(
        "trained agent  win=", e1.wins, " draw=", e1.draws, " loss=", e1.losses
    )
    agent.save(CKPT)

    # Fresh agent (different random init) → load → must match e1 exactly.
    var agent2 = Agent(ctx, lr=0.01)
    var e0 = agent2.eval_vs_random[N_EVAL, RESULT_IDX, MAX_PLIES](
        agent_player=0, seed=12345
    )
    agent2.load(CKPT)
    var e2 = agent2.eval_vs_random[N_EVAL, RESULT_IDX, MAX_PLIES](
        agent_player=0, seed=12345
    )
    print(
        "reloaded agent win=", e2.wins, " draw=", e2.draws, " loss=", e2.losses
    )

    assert_equal(e1.wins, e2.wins, "checkpoint round-trip changed wins")
    assert_equal(e1.draws, e2.draws, "checkpoint round-trip changed draws")
    assert_equal(e1.losses, e2.losses, "checkpoint round-trip changed losses")
    # Sanity: the trained net is not the same as a fresh random one (load did
    # real work, not a no-op on identical weights).
    assert_true(
        e1.losses != e0.losses or e1.wins != e0.wins,
        "trained and fresh eval identical — checkpoint test is vacuous",
    )
    print("AZ agent checkpoint round-trip: OK")
