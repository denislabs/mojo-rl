"""LeWM eval-only driver — Pong, scaled config.

Loads the checkpoint written by `lewm_pong_pixel_train_gpu.mojo` and
runs only the eval phases. Finishes in a few minutes even at DEPTH=6.

Iterate on eval diagnostics (new H8/H9/..., different MPC horizons,
alternate CEM configs) without retraining.

CONFIG must match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import eval_lewm_offline_gpu
from mojo_rl.experimental.lewm.lewm_config import LeWMPongViTConfig


def main() raises:
    eval_lewm_offline_gpu[LeWMPongViTConfig[
        batch=16, t=6, h=4,
        hidden=96, enc_heads=4, enc_layers=2,
        emb=96, proj_h=256,
        smoothed=32,
        pred_heads=4, pred_ff=256,
        depth=6,
        sig_num_proj=1024, sig_knots=17,
    ]](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        checkpoint_path=String("/tmp/lewm_pong.ckpt"),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=3,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
    )
