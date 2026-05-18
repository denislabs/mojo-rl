"""LeWM eval-only driver — PushT, longer/wider checkpoint.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_long.mojo`
(`T=6, H=3, depth=6, 32k steps`) and runs the eval phases.

CONFIG must match the training driver byte-for-byte.

Horizon sweep: edit `mpc_horizon` between runs. Valid range with
`T=6, H=3` is `{1, 2, 3, 4}` (bound: `H + mpc_horizon - 1 ≤ T`).
Each eval run is ~12 min on NVIDIA. Compare `cem/expert`,
`cem/random_min`, and `cem_better_expert` across horizons to see
where the planning curve saturates.

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_long.mojo
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_long.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    eval_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=6, h=3,
        depth=6,
    ]](
        checkpoint_path=String("/tmp/lewm_pusht_long.ckpt"),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=4,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
    )
