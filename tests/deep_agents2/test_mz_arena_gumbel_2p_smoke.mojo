"""Runtime smoke for the two-player Gumbel MuZero arena driver.

Tiny params so the full loop — self-play search over the learned model →
symmetry-augmented sequence replay → K-step BPTT unroll training → one Arena
gating round → one eval-vs-opponent round — runs end-to-end in seconds on Apple.
Asserts only that it completes and returns a finite loss (convergence is a
long NVIDIA run, out of scope here).

    pixi run -e apple mojo run -I . tests/deep_agents2/test_mz_arena_gumbel_2p_smoke.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.muzero.nets import (
    MZRepNetC4Conv, MZDynNet, MZPredNet
)
from mojo_rl.deep_agents2.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from mojo_rl.deep_agents2.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.deep_agents2.zero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    comptime OBS = 126
    comptime ACT = 7
    comptime LATENT = 32
    comptime BINS = 11
    comptime H = 32

    comptime Env = ConnectFourEnv[DType.float64]
    comptime Rep = MZRepNetC4Conv[LATENT, H, F=8]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)

    var res = run_muzero_selfplay_arena_gumbel_2p[
        Env, Rep, Dyn, Pred, Aug,
        N_ENVS=4,
        OBS=OBS, ACT=ACT, LATENT=LATENT, BINS=BINS,
        NUM_SIMS=4, MAX_NODES=16, MAX_K=2,
        CAP=4000, B=16, K=3, N=4, MAX_PLIES=42,
        OPP1=RandomOpponent,
        OPP2=RandomOpponent,
        ARENA_GAMES=4,
        EVAL_GAMES=4,
        TEMP_MOVES=6,
    ](
        ctx,
        rep, dyn, pred,
        iterations=60,
        learning_starts=8,
        train_per_iter=1,
        seed=0,
        arena_every=30,
        arena_open_plies=2,
        report_every=20,
        diag_every=0,
        do_eval=True,
        do_eval2=False,
        verbose=True,
        selfplay_open_plies=2,
        temp_min=0.35,
        eval_open_plies=2,
        # exercise the reanalyze path (live-learner: target_sync_interval=0)
        reanalyze_every=2,
        reanalyze_batch=4,
        target_sync_interval=0,
    )

    var ll = res.last_loss
    if ll - ll != 0.0:
        raise Error("smoke: non-finite last_loss")
    print("OK | last_loss", ll, "| promotions", res.promotions)
