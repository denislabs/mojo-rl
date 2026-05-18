"""LeWM trainer on PushT pixels — paper width.

Matches the LeWM paper's ViT-Tiny + paper-predictor recipe on width:

| axis              | this driver | prior long run (96-wide) | paper ViT-Tiny |
|-------------------|-------------|--------------------------|----------------|
| `hidden`          | **192**     | 96                       | 192            |
| `enc_heads`       | **3**       | 4                        | 3 (head_dim=64)|
| `enc_layers`      | **12**      | 2                        | 12             |
| `emb`             | **192**     | 96                       | 192            |
| `proj_h`          | **2048**    | 256                      | 2048 (default) |
| `pred_heads`      | **16**      | 4                        | 16             |
| `pred_ff`         | **2048**    | 256                      | 2048           |

Everything else (BATCH=16, T=6, H=3, depth=6, 32k steps) matches
`lewm_pusht_pixel_train_gpu_long.mojo` so the cem/expert horizon curve
is the head-to-head comparison.

This is option (C) from `docs/LEWM_MBRL_RESEARCH.md` §6 H5 (2026-05-18
update) — testing whether width alone reduces H7 drift enough to let
CEM exploit the longer-horizon signal that the 96-wide model already
expresses (expert/random_mean=0.61 at h=3 on the long checkpoint).

Wall time on NVIDIA (estimate, paper width is ~10-15× more FLOPs/step
than the 96-wide config): **6-10 hours**. Drop `num_steps` to 16k for
a half-budget first-pass if compute is constrained — the long run
showed loss curves were nearly flat from step 12k onward at 96-wide.

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_train_gpu_paper.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=6, h=3,
        hidden=192, enc_heads=3, enc_layers=12,
        emb=192, proj_h=2048,
        pred_heads=16, pred_ff=2048,
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
        checkpoint_path=String("/tmp/lewm_pusht_paper.ckpt"),
        checkpoint_every=4000,
    )
