"""LeWM trainer on PushT pixels — longer + wider window.

Bumps `T=4 → 6` (history axis) and total budget `8k → 32k` steps vs
`lewm_pusht_pixel_train_gpu.mojo`. Same width as the original scaled
run (HIDDEN=EMB=96, DEPTH=6) so we isolate the data-and-horizon axis
before paying for paper-width (HIDDEN=192, PRED_FF=2048, …).

Why `T=6` matters: MPC horizon is bounded by `H + mpc_horizon - 1 ≤ T`,
so `T=4, H=3` capped CEM rollouts at 2 steps. `T=6, H=3` unlocks
`mpc_horizon ∈ {1, 2, 3, 4}` — the eval driver sweeps that range.

Wall time (NVIDIA, post sampler optimization @ ~20 it/s):
  - 32k steps × BATCH=16 ≈ 30-35 min training.
  - Eval-only afterwards: ~12 min/sweep.

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_long.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=6, h=3,
        depth=6,
    ]](
        num_steps=32000,
        log_every=500,
        rng_seed=0xCAFE,
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=4,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        checkpoint_path=String("/tmp/lewm_pusht_long.ckpt"),
        checkpoint_every=4000,
    )
