"""Smoke for the LeWM GPU trainer on Pong (tiny config).

BATCH=4, T=4, DEPTH=2 — meant for sub-minute compile + a short training
run that exercises the full pipeline (encoder + cond_blocks + projector
+ all eval phases) without burning real time. Writes a checkpoint to
/tmp/lewm_pong_smoke.ckpt for the eval-only smoke
(`lewm_pong_pixel_eval_gpu_smoke.mojo`) to consume.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import train_lewm_offline_gpu


def main() raises:
    train_lewm_offline_gpu[
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
        checkpoint_path=String("/tmp/lewm_pong_smoke.ckpt"),
    )
