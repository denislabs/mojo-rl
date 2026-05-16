"""LeWM trainer (struct-based v2) on Pong pixels — scaled config.

Same compile-time params as `lewm_pong_pixel_train_gpu.mojo` but routes
through `train_lewm_offline_gpu_v2` (struct-based) so DEPTH=6 actually
builds without OOMing Mojo's compiler.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_v2.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_v2.mojo
"""

from mojo_rl.experimental.lewm.trainer_struct import train_lewm_offline_gpu_v2


def main() raises:
    train_lewm_offline_gpu_v2[
        BATCH=16, T=6, H=4, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=3, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=6,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        num_steps=8000,
        log_every=200,
        rng_seed=0xCAFE,
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=3,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pong_v2.ckpt"),
        checkpoint_every=2000,
    )
