"""LeWM eval-only driver — Pong, scaled config.

Loads the checkpoint written by `lewm_pong_pixel_train_gpu_v2.mojo` and
runs only the eval phases. No training cost — finishes in a few minutes
even at scaled DEPTH=6, vs ~80 min for the original train+eval run.

This lets you iterate on eval diagnostics (new H8/H9/..., different MPC
horizons, alternate CEM configs) without retraining.

Comptime params MUST match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu_v2.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import eval_lewm_offline_gpu_v2


def main() raises:
    eval_lewm_offline_gpu_v2[
        BATCH=16, T=6, H=4, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=3, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=6,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        checkpoint_path=String("/tmp/lewm_pong_v2.ckpt"),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=3,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
    )
