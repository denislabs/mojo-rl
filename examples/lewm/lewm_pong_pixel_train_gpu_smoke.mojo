"""Smoke for the LeWM GPU trainer on Pong (tiny config).

Uses `LeWMPongViTConfig` defaults (BATCH=4, T=4, DEPTH=2, IN_CH=4,
IMG=84, PATCH=14, HIDDEN=32, EMB=32, …). Sub-minute compile + 200-step
training run exercising the full pipeline (encoder + cond_blocks +
projector + all eval phases). Writes a checkpoint to
/tmp/lewm_pong_smoke.ckpt for the eval-only smoke
(`lewm_pong_pixel_eval_gpu_smoke.mojo`) to consume.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import train_lewm_offline_gpu
from mojo_rl.experimental.lewm.lewm_config import LeWMPongViTConfig


def main() raises:
    train_lewm_offline_gpu[LeWMPongViTConfig[]](
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
