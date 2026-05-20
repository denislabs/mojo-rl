"""LeWM eval-only driver — PushT, scaled config.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu.mojo` and
runs only the eval phases. Main lever for iterating on PushT evals
(new diagnostics, MPC variants, longer rollouts via larger T) without
retraining.

CONFIG must match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    eval_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=4, h=3,
        depth=6,
    ]](
        checkpoint_path="lewm_pusht.ckpt",
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=2,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
    )
