"""LeWM trainer on PushT pixels — scaled config (paper-shaped).

Matches the LeWM paper config (`references/le-wm-main/config/train/lewm.yaml`)
on the shape axes that affect dynamics-prediction quality (DEPTH=6,
SIGReg lambda=0.09, knots=17, num_proj=1024).

Paper diverges on width (we use HIDDEN=96, EMB=96, PRED_HEADS=4,
PRED_FF=256 vs paper's embed_dim=192 / heads=16 / mlp_dim=2048 / dim_head=64).
That's the next scaling lever if convergence stalls.

H7 closed-loop drift on this config hits rollout_steps = T - H = 1
(sanity only). Bump T to 6 (and H to 3 or 4) to get a real drift curve.

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=4, h=3,
        depth=6,
    ]](
        num_steps=8000,
        log_every=200,
        rng_seed=0xCAFE,
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pusht.ckpt"),
        checkpoint_every=2000,
    )
