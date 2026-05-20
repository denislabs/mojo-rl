"""LeWM eval-only driver — PushT, MPC horizon sweep on the paper-width checkpoint.

Loads the checkpoint written by `lewm_pusht_pixel_train_gpu_paper.mojo`
(paper width: HIDDEN=192, ENC_LAYERS=12, PRED_FF=2048, PRED_HEADS=16,
proj_h=2048; T=6, H=3, depth=6, 32k steps) and sweeps
`mpc_horizon ∈ {1, 2, 3}` back-to-back. Combined with the in-training
h=4 eval, this gives the 4-point cem/expert vs horizon curve directly
comparable to the 96-wide long run's curve.

CONFIG must match the training driver byte-for-byte.

H6 and H7 are identical between passes (no dependence on mpc_horizon),
so we only run them on pass 1 — saves ~2-3 min total.

Each pass also runs the Phase 4d receding-horizon MPC eval at
``rh_steps=3``. The hypothesis (per ``project_lewm_horizon_sweep.md``):
``cem_rh`` at small ``mpc_horizon`` recovers the long-horizon model
informativeness via replanning, while keeping the CEM optimizer in its
high-competence regime. Compare ``cem_rh`` across passes (lower is
better) AND against open-loop ``cem`` at ``mpc_horizon=3`` (the
long-horizon baseline).

Expert RH zero-pads recorded actions past ``T``: at ``T=6, H=3``,
``rh_steps=3`` is fully clean only at ``mpc_horizon=1``
(``rh_steps ≤ T - H - mpc_horizon + 2 = 5 - mpc_horizon``); at
``mpc_horizon=2`` the last RH step is partially padded, at
``mpc_horizon=3`` the last two are. CEM/random RH are unaffected — they
generate fresh plans each step.

Expected wall time on NVIDIA: ~3 × 90-150min = **270-450min** (~4.5-7.5h)
— RH triples each pass (rh_steps=3 means 3× the CEM rollouts).

Run:
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_eval_gpu_paper_sweep.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    eval_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    comptime CONFIG = LeWMPushTViTConfig[
        batch=16, t=6, h=3,
        hidden=192, enc_heads=3, enc_layers=12,
        emb=192, proj_h=2048,
        pred_heads=16, pred_dim_head=64, pred_ff=2048,
        depth=6,
    ]
    comptime CKPT = "/tmp/lewm_pusht_paper.ckpt"

    # ── horizon = 1 ────────────────────────────────────────────────
    print()
    print("################################################################")
    print("### MPC horizon sweep [paper width]: pass 1 / 3 — mpc_horizon = 1")
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
    print("### MPC horizon sweep [paper width]: pass 2 / 3 — mpc_horizon = 2")
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
    print("### MPC horizon sweep [paper width]: pass 3 / 3 — mpc_horizon = 3")
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
