"""LeWM eval-only driver — PushT, smoke config.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_smoke_v2.mojo`
and runs only the eval phases. Skips the slow encoder forward pass on
training batches — only the eval-time encoding + MPC shots, which fit
in seconds.

Comptime params MUST match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_smoke_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import (
    eval_lewm_offline_gpu_pusht_v2,
)


def main() raises:
    eval_lewm_offline_gpu_pusht_v2[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=3, IMG=224, PATCH=14, N_PATCHES=256,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=10, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=2,
        FRAMESKIP=5, ACTION_DIM=2,
    ](
        checkpoint_path=String("/tmp/lewm_pusht_smoke_v2.ckpt"),
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
    )
