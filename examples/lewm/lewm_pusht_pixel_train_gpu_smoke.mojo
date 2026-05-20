"""LeWM trainer on PushT pixels — smoke config (paper-shaped).

Uses `LeWMPushTViTConfig` defaults (BATCH=4, T=4, DEPTH=2, IMG=224,
PATCH=14, N_PATCHES=256, EMB=96, HIDDEN=96, …). Tiny config to verify
wiring against the cached HF dataset.

The encoder workload is ~7x heavier than Pong's 36-patch config; expect
this smoke run to be slower than the Pong smoke at the same step count.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_smoke.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[]](
        num_steps=200,
        log_every=50,
        rng_seed=0xCAFE,
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pusht_smoke.ckpt"),
    )
