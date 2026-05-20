"""LeWM trainer on Pong pixels — scaled config.

DEPTH=6, HIDDEN=96, EMB=96 — paper-shaped training run. ~80 min on
Apple, ~50 min on NVIDIA. Writes a checkpoint to /tmp/lewm_pong.ckpt
that `lewm_pong_pixel_eval_gpu.mojo` consumes.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import train_lewm_offline_gpu
from mojo_rl.experimental.lewm.lewm_config import LeWMPongViTConfig


def main() raises:
    train_lewm_offline_gpu[LeWMPongViTConfig[
        batch=16, t=6, h=4,
        hidden=96, enc_heads=4, enc_layers=2,
        emb=96, proj_h=256,
        smoothed=32,
        pred_heads=4, pred_ff=256,
        depth=6,
        sig_num_proj=1024, sig_knots=17,
    ]](
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
        checkpoint_path=String("/tmp/lewm_pong.ckpt"),
        checkpoint_every=2000,
    )
