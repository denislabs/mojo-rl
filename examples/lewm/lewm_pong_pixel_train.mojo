"""LeWM Phase 3 — offline training on Pong pixel buffer (CPU smoke).

Assumes you have run `lewm_pong_collect_buffer.mojo` first to produce
`/tmp/lewm_pong_buffer.bin`. Runs `train_lewm_offline` for a small number
of steps and prints loss curves.

Run:
    pixi run mojo run -I . examples/lewm/lewm_pong_pixel_train.mojo
"""

from mojo_rl.experimental.lewm.train_offline import train_lewm_offline


def main() raises:
    # Pong scale (4×84×84 pixels, ACT=3) — small POC config.
    train_lewm_offline[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
        ACT=3, SMOOTHED=16,
        PRED_HEADS=2, PRED_FF=64,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        num_steps=200,
        log_every=20,
        rng_seed=0xCAFE,
    )
