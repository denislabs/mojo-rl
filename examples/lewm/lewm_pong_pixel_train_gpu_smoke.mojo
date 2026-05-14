"""Tiny GPU smoke for SIGReg-enabled LeWM trainer.

200 steps at toy config — checks the SIGReg-wired trainer runs end-to-end
and probes look sane (var_min > 0.1, gram < ~0.5) early on. Should take
~10-20s on any GPU.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.train_offline_gpu import train_lewm_offline_gpu


def main() raises:
    # Tiny SIG knobs: num_proj=64, knots=5 (vs paper 1024/17). Enough to
    # see SIGReg moving probes the right way without saturating memory at
    # toy EMB=32.
    train_lewm_offline_gpu[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
        ACT=3, SMOOTHED=16,
        PRED_HEADS=2, PRED_FF=64,
        SIG_NUM_PROJ=64, SIG_KNOTS=5,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        num_steps=200,
        log_every=20,
        rng_seed=0xCAFE,
        lambda_sigreg=0.09,
    )
