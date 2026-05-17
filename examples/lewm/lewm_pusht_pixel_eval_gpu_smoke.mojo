"""LeWM eval-only driver — PushT, smoke config.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_smoke.mojo`
and runs only the eval phases. CONFIG must match the training driver.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    eval_lewm_offline_gpu_pusht[LeWMPushTViTConfig[]](
        checkpoint_path=String("/tmp/lewm_pusht_smoke.ckpt"),
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
    )
