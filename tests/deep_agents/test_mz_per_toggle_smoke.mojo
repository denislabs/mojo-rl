"""PER/Uniform toggle smoke for the 2p Gumbel MuZero arena driver.

Runs the SAME tiny spatial-net arena loop twice through
`run_muzero_selfplay_arena_gumbel_2p` — once with `use_per=True` (prioritized
device sampling + IS-weighted grads + value-error priority write-back) and once
with `use_per=False` (constant priorities → uniform device sampling, no IS
weighting). Both share the device-obs `PrioritizedMCTSSequenceReplay` and the
`obs_on_device=True` training path; the flag only gates the PER behaviour.
Asserts a finite loss for both — guards both branches of the kernel's optional
`is_weights`/`out_prio` plumbing.

    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_per_toggle_smoke.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
    mzc4_init_zero_pred, mzc4_init_zero_dyn,
)
from mojo_rl.deep_agents.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from mojo_rl.deep_agents.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.deep_agents.zero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def _run[use_per: Bool](ctx: DeviceContext) raises -> Float64:
    comptime OBS = 126
    comptime ACT = 7
    comptime CH = 8            # LATENT = CH*6*7 = 336
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 11

    comptime Env = ConnectFourEnv[DType.float64]
    comptime Rep = MZRepNetC4Spatial[CH, HH, WW]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW]
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    mzc4_init_zero_pred["gpu", CH, ACT, BINS, HH, WW](pred, ctx)
    mzc4_init_zero_dyn["gpu", CH, ACT, BINS, HH, WW](dyn, ctx)

    var res = run_muzero_selfplay_arena_gumbel_2p[
        Env, Rep, Dyn, Pred, Aug,
        N_ENVS=4,
        OBS=OBS, ACT=ACT, LATENT=LATENT, BINS=BINS,
        NUM_SIMS=4, MAX_NODES=16, MAX_K=2,
        CAP=4000, B=16, K=3, N=42, MAX_PLIES=42,
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
        checkpoint_every=0,
        checkpoint_path=String(""),
        use_per=use_per,
    )
    _ = rep^; _ = dyn^; _ = pred^
    return res.last_loss


def main() raises:
    var ctx = DeviceContext()

    print("=== PER ON (use_per=True) ===")
    var loss_per = _run[True](ctx)
    if loss_per - loss_per != 0.0:
        raise Error("PER-on: non-finite last_loss")

    print("=== UNIFORM (use_per=False) ===")
    var loss_uni = _run[False](ctx)
    if loss_uni - loss_uni != 0.0:
        raise Error("PER-off: non-finite last_loss")

    print(
        "OK | PER last_loss", loss_per, "| uniform last_loss", loss_uni
    )
