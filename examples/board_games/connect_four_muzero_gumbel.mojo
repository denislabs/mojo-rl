"""GUMBEL MuZero on Connect Four — full GPU, the learned-model sibling of
`connect_four_alphazero_gumbel.mojo`.

Same arena harness as the AlphaZero Gumbel example (best/learner gating,
horizontal-flip augmentation, periodic full-strength MCTS eval vs 5-ply minimax +
random, RemoteLogger), but the agent plans over a **learned model** instead of
the true game rules: three MLP nets h/g/f (representation / dynamics / prediction)
trained by a K-step BPTT unroll, with the self-play search swapped to two-player
Gumbel MuZero (`run_muzero_selfplay_arena_gumbel_2p`).

The MuZero value/reward heads are categorical over the board outcome support
[-1, +1] (`BINS` atoms); the two-player n-step targets carry the perspective
sign flips. Unlike the AlphaZero example's BatchNorm ResNet, the h/g/f torsos are
plain MLPs (no BatchNorm), so arena promotion is a params-only copy — there are
no running stats to leak, sidestepping the BN-promotion pitfall.

Connect Four is heavy (126D obs = 3×6×7, 7 actions, games up to 42 plies, a
learned model searched 64 sims/move) — this needs an NVIDIA GPU to train at a
useful pace.

`iterations` / `report_every` / `arena_every` are in self-play *moves* (one loop
pass advances all N_ENVS games by one move).

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_muzero_gumbel.mojo

With no `RL_MONITOR_URL` in the environment the RemoteLogger is a silent no-op;
the per-report lines still print to stdout.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.muzero.nets import (
    MZRepNet, MZRepNetC4Conv, MZDynNet, MZPredNet
)
from mojo_rl.deep_agents2.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from mojo_rl.deep_agents2.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== Gumbel MuZero on Connect Four (deep_agents2 / nn2) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="Gumbel MuZero Connect Four (nn2)",
        buffer_size=22,
        api_key=api_key,
    )
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "MZ MLP[LATENT=128,H=128,BINS=51]")
    logger.set_config("framework", "deep_agents2/nn2")

    comptime OBS = 126
    comptime ACT = 7
    comptime LATENT = 128
    comptime BINS = 51       # categorical value/reward support over [-1, 1]
    comptime H = 128         # MLP hidden width for h/g/f
    comptime NUM_SIMS = 64   # Gumbel sims/move (matches the AlphaZero example)
    comptime MAX_NODES = 256
    comptime MAX_K = 4       # Gumbel root candidates (power of two, <= ACT)
    comptime CAP = 1_000_000
    comptime B = 128         # unroll batch
    comptime K = 5           # BPTT unroll length
    comptime N = 10          # n-step value-target horizon
    comptime MAX_PLIES = 42  # full ConnectFour board

    comptime Env = ConnectFourEnv[DType.float64]
    # Representation torso: a BN-free conv ResNet over the 3×6×7 board (the
    # spatial inductive-bias upgrade from the flat-MLP `MZRepNet[OBS, LATENT, H]`,
    # which remains a drop-in A/B swap — the dynamics/prediction nets and the
    # whole driver are agnostic to the rep torso). F = conv filters.
    comptime Rep = MZRepNetC4Conv[LATENT, H, F=64]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    # ConnectFour's only board symmetry is the left↔right column flip.
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()

    # The BEST net trio — holds the final (best) weights on return. The driver
    # builds + trains a learner copy internally and promotes it on arena wins.
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)

    var res = run_muzero_selfplay_arena_gumbel_2p[
        Env, Rep, Dyn, Pred, Aug,
        N_ENVS=64,
        OBS=OBS, ACT=ACT, LATENT=LATENT, BINS=BINS,
        NUM_SIMS=NUM_SIMS, MAX_NODES=MAX_NODES, MAX_K=MAX_K,
        CAP=CAP, B=B, K=K, N=N, MAX_PLIES=MAX_PLIES,
        OPP1=GPUMinimaxConnectFour[5],
        OPP2=RandomOpponent,
        L=RemoteLogger,
        ARENA_GAMES=64,
        EVAL_GAMES=64,
        TEMP_MOVES=20,
    ](
        ctx,
        rep, dyn, pred,
        iterations=40_000,
        learning_starts=2_000,
        train_per_iter=4,
        lr=Scalar[DT](2e-3),
        gamma=Scalar[DT](1.0),
        value_coef=Scalar[DT](0.25),
        max_grad_norm=Scalar[DT](1.0),
        seed=42,
        arena_every=2_000,
        arena_open_plies=4,
        promote_threshold=0.55,
        report_every=1_000,
        diag_every=50,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=logger),
        selfplay_open_plies=2,
        eval_open_plies=4,
    )

    logger.close()

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("=== Done ===")
