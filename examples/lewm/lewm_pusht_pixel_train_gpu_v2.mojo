"""LeWM trainer v2 on PushT pixels — scaled config (paper-shaped).

Matches the LeWM paper config (`references/le-wm-main/config/train/lewm.yaml`)
on the shape axes that affect dynamics-prediction quality:

- IMG=224, PATCH=14, N_PATCHES=256 — paper's image_size / patch_size.
- IN_CH=3, ACT=10 = FRAMESKIP(5) * ACTION_DIM(2).
- T=4, H=3, N_PREDS=1 — paper's history_size=3 + num_preds=1.
- DEPTH=6 — paper's predictor.depth.
- SIGReg lambda=0.09, knots=17, num_proj=1024 — paper's loss.sigreg.

Paper diverges on width (we use HIDDEN=96, EMB=96, PRED_HEADS=4,
PRED_FF=256 vs paper's embed_dim=192 / heads=16 / mlp_dim=2048 / dim_head=64).
That's the next scaling lever if convergence stalls.

H7 closed-loop drift on this config will hit rollout_steps = T - H = 1
(sanity only). Bump T to 6 (and H to 3 or 4) to get a real drift curve.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_v2.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import (
    train_lewm_offline_gpu_pusht_v2,
)


def main() raises:
    train_lewm_offline_gpu_pusht_v2[
        BATCH=16, T=4, H=3, N_PREDS=1,
        IN_CH=3, IMG=224, PATCH=14, N_PATCHES=256,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=10, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=6,
        FRAMESKIP=5, ACTION_DIM=2,
    ](
        num_steps=8000,
        log_every=200,
        rng_seed=0xCAFE,
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pusht_v2.ckpt"),
        checkpoint_every=2000,
    )
