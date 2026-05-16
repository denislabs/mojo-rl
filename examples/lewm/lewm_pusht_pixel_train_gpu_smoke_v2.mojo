"""LeWM trainer v2 on PushT pixels — smoke config (paper-shaped).

Tiny config to verify wiring against the cached HF dataset. Shapes match the
LeWM paper config (`references/le-wm-main/config/train/lewm.yaml`):

- IMG=224, PATCH=14, N_PATCHES=256 (16x16 grid).
- IN_CH=3 (RGB single-frame, no temporal stack).
- ACT=10 = FRAMESKIP(5) * ACTION_DIM(2) — paper's effective_act_dim.
- T=4, H=3, N_PREDS=1 — paper's history_size=3 + num_preds=1.
- DEPTH=2 (vs paper's 6) — keep cond-block stack thin for smoke.

The encoder workload is ~7x heavier than Pong's 36-patch config; expect
this smoke run to be slower than the Pong v2 smoke at the same step count.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_smoke_v2.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_smoke_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import (
    train_lewm_offline_gpu_pusht_v2,
)


def main() raises:
    train_lewm_offline_gpu_pusht_v2[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=3, IMG=224, PATCH=14, N_PATCHES=256,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=10, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=2,
        FRAMESKIP=5, ACTION_DIM=2,
    ](
        num_steps=200,
        log_every=50,
        rng_seed=0xCAFE,
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pusht_smoke_v2.ckpt"),
    )
