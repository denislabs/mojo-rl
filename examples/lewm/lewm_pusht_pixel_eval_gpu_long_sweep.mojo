"""LeWM eval-only driver — PushT, MPC horizon sweep on the long checkpoint.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_long.mojo`
(`T=6, H=3, depth=6, 32k steps`) and sweeps `mpc_horizon ∈ {1, 2, 3}`
back-to-back in one process — the missing horizons vs the training-time
h=4 eval. Each pass re-loads the 78MB checkpoint (~2-3s overhead per
horizon, negligible against the ~12min CEM cost).

H6 and H7 are identical between passes (no dependence on mpc_horizon),
so we disable them on horizons 2 and 3 — saves ~2-3 min total without
losing comparability. The h=1 pass keeps them on as a sanity check
against the training-time numbers.

Expected wall time on NVIDIA: ~3 × 12min ≈ 35-40min total.

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_long_sweep.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    alias CONFIG = LeWMPushTViTConfig[batch=16, t=6, h=3, depth=6]
    alias CKPT = "/tmp/lewm_pusht_long.ckpt"

    # ── horizon = 1 ────────────────────────────────────────────────
    print()
    print("################################################################")
    print("### MPC horizon sweep: pass 1 / 3 — mpc_horizon = 1")
    print("################################################################")
    eval_lewm_offline_gpu_pusht[CONFIG](
        checkpoint_path=String(CKPT),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=1,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        eval_shuffle_diag=True,
        eval_h7_closed_loop=True,
    )

    # ── horizon = 2 ────────────────────────────────────────────────
    print()
    print("################################################################")
    print("### MPC horizon sweep: pass 2 / 3 — mpc_horizon = 2")
    print("################################################################")
    eval_lewm_offline_gpu_pusht[CONFIG](
        checkpoint_path=String(CKPT),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=2,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        eval_shuffle_diag=False,
        eval_h7_closed_loop=False,
    )

    # ── horizon = 3 ────────────────────────────────────────────────
    print()
    print("################################################################")
    print("### MPC horizon sweep: pass 3 / 3 — mpc_horizon = 3")
    print("################################################################")
    eval_lewm_offline_gpu_pusht[CONFIG](
        checkpoint_path=String(CKPT),
        eval_steps=10,
        eval_samples=32,
        eval_seed=0xBEEF,
        mpc_horizon=3,
        cem_iters=5,
        cem_samples=64,
        cem_topk=8,
        cem_smoothing=0.5,
        eval_shuffle_diag=False,
        eval_h7_closed_loop=False,
    )
