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

Each pass also runs the Phase 4d receding-horizon MPC eval at
``rh_steps=3`` — testing whether short-horizon CEM + replanning beats
long-horizon open-loop CEM at this checkpoint width (the gating
hypothesis from ``project_lewm_horizon_sweep.md``). This 96-wide run is
the cheap dress rehearsal for the paper-width sweep.

Expected wall time on NVIDIA: ~3 × 35-40min ≈ 105-120min total
(rh_steps=3 triples per-pass CEM rollouts).

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_long_sweep.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    comptime CONFIG = LeWMPushTViTConfig[batch=16, t=6, h=3, depth=6]
    comptime CKPT = "/tmp/lewm_pusht_long.ckpt"

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
        rh_steps=3,
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
        rh_steps=3,
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
        rh_steps=3,
    )
