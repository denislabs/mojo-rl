"""Gumbel AlphaZero on Connect Four (GPU) — the low-sim-budget lighthouse.

The Gumbel sibling of `connect_four_alphazero_v2.mojo`. Sequential halving
gets its policy-improvement guarantee at LOW sim counts, so this runs 64 sims
per move instead of the PUCT example's 500 — ~8× less search per move, serial
sims (no batched-leaf bias by construction). Policy targets are the improved
policy. No arena gating: plain self-play in report chunks, with a greedy
policy-head eval vs random between chunks (the TTT gate showed Gumbel-AZ
beating the PUCT baseline 8-vs-24 losses at equal sims).

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_gumbel.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.alphazero.nets import AZConnectFourResNet
from mojo_rl.deep_agents2.alphazero.selfplay_gumbel import (
    run_alphazero_gumbel_selfplay,
)
from mojo_rl.deep_agents2.alphazero.eval import eval_policy_vs_random
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== Gumbel AlphaZero on Connect Four (deep_agents2 / nn2) ===")
    comptime Net = AZConnectFourResNet[F=128, NB=5, FC=128]
    comptime Env = ConnectFourEnv[DType.float64]
    comptime RESULT_IDX = 43
    comptime MAX_PLIES = 42
    comptime N_EVAL = 128
    comptime CHUNK = 2_000
    comptime CHUNKS = 20          # 40k moves total

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT=Kaiming](ctx=ctx)

    var before = eval_policy_vs_random[
        Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES
    ](ctx, net, agent_player=0, seed=3)
    print("BEFORE  win=", before.wins, " draw=", before.draws,
          " loss=", before.losses, " (/", N_EVAL, ")")

    for c in range(CHUNKS):
        var loss = run_alphazero_gumbel_selfplay[
            Env, Net,
            N_ENVS=64, NUM_SIMS=64, MAX_NODES=256, MAX_K=4,
            BATCH=128, CAP=1_000_000, MAX_TRAJ=42,
        ](ctx, net, iterations=CHUNK,
          learning_starts=200 if c == 0 else 0,
          train_per_iter=4, lr=0.002, seed=UInt64(42 + c))
        var ev = eval_policy_vs_random[
            Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES
        ](ctx, net, agent_player=0, seed=3)
        print("chunk", c + 1, "/", CHUNKS, "moves", (c + 1) * CHUNK,
              "| loss", loss, "| vs random W", ev.wins,
              "D", ev.draws, "L", ev.losses, "(/", N_EVAL, ")")

    print("=== Done ===")
