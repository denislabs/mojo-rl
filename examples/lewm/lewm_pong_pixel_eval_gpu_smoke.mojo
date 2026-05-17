"""LeWM eval-only driver — Pong, smoke config.

Loads the checkpoint written by `lewm_pong_pixel_train_gpu_smoke.mojo`
and runs only the eval phases (H6 action-shuffle, H7 closed-loop drift,
Phase 4b/4c MPC+CEM). No training — should finish in seconds.

Comptime params MUST match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import eval_lewm_offline_gpu


def main() raises:
    eval_lewm_offline_gpu[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
        ACT=3, SMOOTHED=16,
        PRED_HEADS=2, PRED_FF=64,
        DEPTH=2,
        SIG_NUM_PROJ=64, SIG_KNOTS=5,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        checkpoint_path=String("/tmp/lewm_pong_smoke.ckpt"),
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=2,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
    )
