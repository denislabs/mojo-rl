"""LeWM eval-only driver — PushT, scaled config.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_v2.mojo` and
runs only the eval phases. No training cost — runs in minutes vs the
multi-hour training.

This is the main lever for iterating on PushT evals (new diagnostics,
MPC variants, longer rollouts via larger T) without burning compute.

Comptime params MUST match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_v2.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import (
    eval_lewm_offline_gpu_pusht_v2,
)


def main() raises:
    eval_lewm_offline_gpu_pusht_v2[
        BATCH=16, T=4, H=3, N_PREDS=1,
        IN_CH=3, IMG=224, PATCH=14, N_PATCHES=256,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=10, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=6,
        FRAMESKIP=5, ACTION_DIM=2,
    ](
        checkpoint_path=String("/tmp/lewm_pusht_v2.ckpt"),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
    )
