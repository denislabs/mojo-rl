"""Runtime smoke for the SPATIAL-latent MuZero nets through the 2p arena driver.

Same tiny full-loop smoke as `test_mz_arena_gumbel_2p_smoke.mojo` but with the
conv spatial h/g/f (`MZRepNetC4Spatial` / `MZDynNetC4Spatial` /
`MZPredNetC4Spatial`) — exercises the ComputeGraph dynamics' forward + the
BPTT-unroll backward (vjp through the action-plane graph) end-to-end, plus the
arena's `hard_copy_params` over the graph. Asserts a finite loss.

    pixi run -e apple mojo run -I . tests/deep_agents2/test_mz_spatial_arena_smoke.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
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
    comptime CH = 8            # latent channels → LATENT = CH*6*7 = 336
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 11

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
        iterations=40,
        learning_starts=8,
        train_per_iter=1,
        seed=0,
        arena_every=20,
        report_every=20,
        do_eval=True,
        do_eval2=False,
        verbose=True,
        selfplay_open_plies=2,
        temp_min=0.35,
        eval_open_plies=2,
        reanalyze_every=2,
        reanalyze_batch=4,
        target_sync_interval=0,
    )

    if res.last_loss - res.last_loss != 0.0:
        raise Error("spatial smoke: non-finite last_loss")
    print("OK | last_loss", res.last_loss, "| promotions", res.promotions)
