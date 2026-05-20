"""LeWM K=1 score-plans-batched parity test.

Regression test for the Phase-1 batched score-plan path
(`LeWMRolloutScoreCallback.score_plans_batched`). Single-plan
``score_plan`` and batched ``score_plans_batched`` with K=1 should run
the same kernels on the same inputs and produce identical scores —
otherwise the batched K-loop staging, the (1,) slot view at
``scores_dev_buf + k_idx``, or the new ``_run_mpc_rollout_no_readback``
helper has diverged from ``_run_mpc_shot``.

Setup:
  - Xavier-init two ``LeWMGPUState[LeWMPongViTConfig[]]`` instances with
    the same RNG seed so their params are bit-identical.
  - Fill ``emb_start_dev_buf`` / ``emb_goal_dev_buf`` on the callback
    with deterministic random data (same data for both callbacks).
  - Build a deterministic one-hot action plan ``(BATCH, needed, ACT)``.
  - Path A: ``callback_a.score_plan(view_3d)`` on state_a.
  - Path B: ``callback_b.score_plans_batched(view_4d_k1, scores_out)``
    on state_b with K=1.
  - Assert ``|single - batched|`` is below tolerance.

Two state instances (not one shared) so any BN running-stat updates
during the projector forward don't leak between calls — both callbacks
see the same "fresh" world model.

Run:
    pixi run -e apple mojo run -I . \\
      tests/experimental/lewm/test_lewm_score_plans_batched_parity.mojo
"""

from std.math import abs as math_abs
from std.memory import alloc
from std.random import seed as _set_seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.testing import assert_true

from layout import TileTensor, Idx, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.experimental.lewm.lewm_config import LeWMPongViTConfig
from mojo_rl.experimental.lewm.offline_trainer import LeWMGPUState
from mojo_rl.experimental.lewm.lewm_rollout_callback import (
    LeWMRolloutScoreCallback,
)


comptime CONFIG = LeWMPongViTConfig[]
comptime MPC_HORIZON: Int = 2
comptime NEEDED: Int = CONFIG.H + MPC_HORIZON - 1  # 3 + 2 - 1 = 4
comptime EMB_DIM: Int = CONFIG.EMB
comptime BATCH_DIM: Int = CONFIG.BATCH
comptime ACT_DIM: Int = CONFIG.ACT

comptime XAVIER_SEED: Int = 0xC0FFEE
comptime FILL_SEED: Int = 0xBADCAFE
# Bit-exact parity is the design intent; allow a small slack for any
# Apple-Metal nondeterminism. NVIDIA TF32 may need this loosened.
comptime PARITY_EPS: Float64 = 1e-6


def _fill_random_fp32(
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    """Fill ``n`` Float32 slots from the global RNG. Caller seeds."""
    for i in range(n):
        dst[i] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * 0.1)


def _build_action_plan(
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Deterministic one-hot plan: action ``(b + t) % ACT`` per (b, t)."""
    for b in range(BATCH_DIM):
        for t in range(NEEDED):
            var picked = (b + t) % ACT_DIM
            for a in range(ACT_DIM):
                dst[b * NEEDED * ACT_DIM + t * ACT_DIM + a] = (
                    Scalar[dtype](1.0)
                    if a == picked
                    else Scalar[dtype](0.0)
                )


def main() raises:
    print("=" * 70)
    print("LeWM score_plans_batched (K=1) parity test")
    print("=" * 70)
    print(
        "BATCH=", BATCH_DIM, " EMB=", EMB_DIM, " ACT=", ACT_DIM,
        " H=", CONFIG.H, " mpc_horizon=", MPC_HORIZON,
        " needed_actions=", NEEDED,
    )

    var ctx = DeviceContext()

    # ── Two independent state instances, same Xavier seed so params
    # are bit-identical. ────────────────────────────────────────────
    _set_seed(XAVIER_SEED)
    var state_a = LeWMGPUState[CONFIG](ctx, 0.0)
    _set_seed(XAVIER_SEED)
    var state_b = LeWMGPUState[CONFIG](ctx, 0.0)

    # ── Two callbacks, sharing the same horizon + needed_actions and
    # the same k_max=1 storage (the K=1 batched path uses only slot 0
    # of `scores_dev_buf`, the same slot the single-plan path uses). ─
    var callback_a = LeWMRolloutScoreCallback[CONFIG](
        state_a, ctx, MPC_HORIZON, NEEDED, k_max=1,
    )
    var callback_b = LeWMRolloutScoreCallback[CONFIG](
        state_b, ctx, MPC_HORIZON, NEEDED, k_max=1,
    )

    # ── Deterministic emb_start / emb_goal: fill a host scratch with
    # fixed RNG, copy to both callbacks' device buffers. ─────────────
    comptime EMB_SIZE = BATCH_DIM * EMB_DIM
    var emb_start_host = ctx.enqueue_create_host_buffer[dtype](EMB_SIZE)
    var emb_goal_host = ctx.enqueue_create_host_buffer[dtype](EMB_SIZE)
    _set_seed(FILL_SEED)
    _fill_random_fp32(emb_start_host.unsafe_ptr(), EMB_SIZE)
    _fill_random_fp32(emb_goal_host.unsafe_ptr(), EMB_SIZE)
    ctx.enqueue_copy(callback_a.emb_start_dev_buf, emb_start_host)
    ctx.enqueue_copy(callback_a.emb_goal_dev_buf, emb_goal_host)
    ctx.enqueue_copy(callback_b.emb_start_dev_buf, emb_start_host)
    ctx.enqueue_copy(callback_b.emb_goal_dev_buf, emb_goal_host)

    # ── Deterministic action plan on host. ───────────────────────────
    comptime PLAN_SIZE = BATCH_DIM * NEEDED * ACT_DIM
    var plan_host = alloc[Scalar[dtype]](PLAN_SIZE)
    _build_action_plan(plan_host)

    # ── Path A: single-plan ``score_plan`` over a 3-D
    # ``(BATCH, needed, ACT)`` tile-tensor view. ─────────────────────
    var single_view = TileTensor(
        plan_host,
        row_major(
            (Idx[BATCH_DIM](), Idx(NEEDED), Idx[ACT_DIM]())
        ),
    )
    var single_score = callback_a.score_plan(single_view)

    # ── Path B: batched ``score_plans_batched`` over a 4-D
    # ``(K=1, BATCH, needed, ACT)`` tile-tensor view. ────────────────
    var k1_view = TileTensor(
        plan_host,
        row_major(
            (Idx(1), Idx[BATCH_DIM](), Idx(NEEDED), Idx[ACT_DIM]())
        ),
    )
    var scores_out = List[Float64](length=1, fill=0.0)
    callback_b.score_plans_batched(k1_view, scores_out)
    var batched_score = scores_out[0]

    print()
    print("  single_score  =", single_score)
    print("  batched_score =", batched_score)
    var diff = single_score - batched_score
    var abs_diff = diff if diff >= 0.0 else -diff
    print("  |diff|        =", abs_diff)
    print("  tolerance     =", PARITY_EPS)

    assert_true(
        abs_diff <= PARITY_EPS,
        "K=1 parity failed: single - batched = " + String(diff)
        + " (tolerance " + String(PARITY_EPS) + ")",
    )

    plan_host.free()
    print()
    print("=== PASS: score_plans_batched(K=1) ≡ score_plan ===")
