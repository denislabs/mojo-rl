"""Smoke for v2 (struct-based) LeWM trainer.

Same config as lewm_pong_pixel_train_gpu_smoke.mojo but routed through
`train_lewm_offline_gpu_v2` (LeWMGPUState + LeWMTrainer structs).

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_smoke_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import train_lewm_offline_gpu_v2


def main() raises:
    train_lewm_offline_gpu_v2[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
        ACT=3, SMOOTHED=16,
        PRED_HEADS=2, PRED_FF=64,
        DEPTH=2,
        SIG_NUM_PROJ=64, SIG_KNOTS=5,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        num_steps=200,
        log_every=50,
        rng_seed=0xCAFE,
        lambda_sigreg=0.09,
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=2,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pong_smoke_v2.ckpt"),
    )
