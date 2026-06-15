"""GUMBEL MuZero on Connect Four with a SPATIAL latent — the value-ceiling test.

Identical arena harness to `connect_four_muzero_gumbel.mojo`, but the learned
model keeps the latent **spatial** (`[C, H, W]` feature map, flat-encoded as
LATENT = C·6·7) and uses conv h/g/f: a conv representation, a convolutional
dynamics with action-plane embedding (`MZDynNetC4Spatial`, a ComputeGraph), and
conv prediction heads. This targets the value-fit ceiling the flat-MLP runs hit
(`value_mse` plateaued ~0.38 regardless of width/sims) — the flat-latent MLP
dynamics can't model C4 tactics, so the n-step value targets stay noisy. The
spatial dynamics gives g weight-sharing over the board, which is what the
EZv2/AlphaZero spatial model buys.

All conv blocks are BatchNorm-FREE (the arena's params-only promotion can't carry
BN running stats). C = latent channels (LATENT = C·6·7; C=32 → 1344). The spatial
latent + conv dynamics are heavier than the flat path, so this starts at 64
sims/move (raise NUM_SIMS once you see the wall-clock).

Watch `value_mse`: if it drops below the flat run's ~0.38 floor, the spatial
dynamics broke the ceiling; if it stays pinned, the value target itself is the
limit (not the model).

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_muzero_gumbel_spatial.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
)
from mojo_rl.deep_agents2.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from mojo_rl.deep_agents2.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
from mojo_rl.nn2.core.checkpoint import save_state_v2_body_gpu
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== Gumbel MuZero on Connect Four — SPATIAL latent (deep_agents2) ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="Gumbel MuZero Connect Four spatial (nn2)",
        buffer_size=22,
        api_key=api_key,
    )
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "MZ spatial conv h/g/f [C=32, 6x7]")
    logger.set_config("framework", "deep_agents2/nn2")

    comptime OBS = 126
    comptime ACT = 7
    comptime CH = 32          # latent channels → LATENT = CH*6*7 = 1344
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 51        # categorical value/reward support over [-1, 1]
    comptime NUM_SIMS = 64    # spatial latent + conv dynamics are heavier; start
    #                           at 64, raise once the wall-clock is known.
    comptime MAX_NODES = 256
    comptime MAX_K = 4        # already maxed for C4 (power of two ≤ ACT=7)
    comptime CAP = 1_000_000
    comptime B = 128
    comptime K = 5
    comptime N = 10
    comptime MAX_PLIES = 42

    comptime Env = ConnectFourEnv[DType.float64]
    comptime Rep = MZRepNetC4Spatial[CH, HH, WW]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW]
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()
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
        value_coef=Scalar[DT](0.5),
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
        selfplay_open_plies=4,
        temp_min=0.35,
        eval_open_plies=4,
        reanalyze_every=4,
        reanalyze_batch=128,
        target_sync_interval=200,
    )

    logger.close()

    var body = String("")
    save_state_v2_body_gpu(rep, body, String("rep"), ctx)
    save_state_v2_body_gpu(dyn, body, String("dyn"), ctx)
    save_state_v2_body_gpu(pred, body, String("pred"), ctx)
    with open("connect_four_muzero_gumbel_spatial.ckpt", "w") as f:
        f.write(String("nn2-ckpt v2\n") + body)

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved best net → connect_four_muzero_gumbel_spatial.ckpt")
    print("=== Done ===")
