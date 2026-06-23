"""Runtime smoke for the SPATIAL-latent MuZero nets through the 2p arena driver.

Same tiny full-loop smoke as `test_mz_arena_gumbel_2p_smoke.mojo` but with the
conv spatial h/g/f (`MZRepNetC4Spatial` / `MZDynNetC4Spatial` /
`MZPredNetC4Spatial`) — exercises the ComputeGraph dynamics' forward + the
BPTT-unroll backward (vjp through the action-plane graph) end-to-end, plus the
arena's `hard_copy_params` over the graph. Asserts a finite loss.

    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_spatial_arena_smoke.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
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
    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    mzc4_init_zero_pred["gpu", CH, ACT, BINS, HH, WW](pred, ctx)
    mzc4_init_zero_dyn["gpu", CH, ACT, BINS, HH, WW](dyn, ctx)

    # Verify zero-init actually matched the head param names (a wrong name is a
    # SILENT no-op): forward pred on a zero latent — with the output Linear
    # zeroed the value logits must be exactly 0 (else the names didn't match).
    var z_in = Tensor.alloc(LATENT)          # host zeros
    z_in.ensure_gpu(ctx, LATENT)
    z_in.upload(ctx)                         # H2D zero latent
    var out = Tensor()
    out.ensure_gpu(ctx, ACT + BINS)
    pred.forward["gpu", 1](TensorRefs[Pred.ARITY](z_in), out, Optional(ctx))
    out.download(ctx)
    ctx.synchronize()
    var vmax = 0.0
    for i in range(ACT, ACT + BINS):
        var v = abs(Float64(out.data[i]))
        if v > vmax:
            vmax = v
    print("zero-init check: value-logit max-abs =", vmax, "(expect 0.0)")
    if vmax > 1e-6:
        raise Error(
            "zero-init did NOT apply (value head not zeroed — param name"
            " mismatch); got max-abs " + String(vmax)
        )

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
        checkpoint_every=20,
        checkpoint_path=String("/tmp/c4_mz_smoke.ckpt"),
    )

    if res.last_loss - res.last_loss != 0.0:
        raise Error("spatial smoke: non-finite last_loss")
    print("OK | last_loss", res.last_loss, "| promotions", res.promotions)
