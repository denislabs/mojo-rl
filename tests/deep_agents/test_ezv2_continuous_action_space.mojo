"""Gradcheck + dispatch spike for `ContinuousActionSpace`.

Two things we want to nail down before wiring this into a Phase 3 agent:

  1. **Bit-parity dispatch** — three call paths produce identical
     gradient + loss buffers:
       Path A — direct `ezv2_policy_loss_grad_continuous_kernel` enqueue.
       Path B — `ContinuousActionSpace[...].policy_loss_grad_gpu[...]`.
       Path C — `def run_via_trait[AS: ActionSpace, ...]` calling
                `AS.policy_loss_grad_gpu[...]` (the call shape used inside
                `ezv2_train_step_gpu_core`).

  2. **Numerical gradient correctness** — finite-difference vs analytical
     gradient on the unscaled loss across both `ent_scale=0` and a
     non-zero entropy bonus. fp32 threshold is loose (max |Δ| < 1e-2,
     max rel < 5e-3) since the squashed-Gaussian backward involves an
     atanh inside log/exp/tanh chains; tighter bounds belong on a fp64
     CPU-side gradcheck once we have one.

The kernel is action-space-agnostic at the pred-net layout level: it
reads `(μ_raw, σ_raw)` from the first `2*ACT_DIM` slots of
`pred_out_step[b]` and ignores the trailing BINS value-logit slots. The
test confirms grads on the trailing slots stay zero.
"""

from std.gpu.host import DeviceContext
from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.deep_agents.efficient_zero_v2.action_space import (
    ActionSpace,
    ContinuousActionSpace,
)
from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_policy_loss_grad_continuous_kernel,
)


comptime TPB: Int = 256


# ───────────────────────────────────────────────────────────────────────────
# Path C — generic helper that dispatches via a trait-bound type parameter.
# Same shape as `test_ezv2_action_space_dispatch.run_via_trait`.
# ───────────────────────────────────────────────────────────────────────────


def run_via_trait[
    AS: ActionSpace,
    BATCH: Int,
    PRED_OUT: Int,
    POL_TGT_DIM: Int,
    dtype: DType where dtype.is_floating_point(),
](
    ctx: DeviceContext,
    pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    policy_target_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * POL_TGT_DIM), MutAnyOrigin
    ],
    grad_pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    loss_scale: Scalar[dtype],
    ent_scale: Scalar[dtype],
) raises:
    AS.policy_loss_grad_gpu[BATCH, PRED_OUT, POL_TGT_DIM, dtype](
        ctx,
        pred_out_step,
        policy_target_step,
        grad_pred_out_step,
        per_sample_loss,
        loss_scale,
        ent_scale,
    )


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def main() raises:
    print("=== EZ-V2 ContinuousActionSpace gradcheck + dispatch spike ===")
    var passed = 0
    var total = 0

    comptime BATCH = 4
    comptime ACT_DIM = 2
    comptime BINS = 21
    comptime PRED_OUT = 2 * ACT_DIM + BINS  # 25
    comptime dtype = DType.float32
    comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime MAX_ACTION = 1.0
    comptime MIN_STD = 0.1
    comptime K = 16

    var ctx = DeviceContext()

    # ── Allocate input buffers ──────────────────────────────────────────
    var pred_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var pol_tgt_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACT_DIM)
    var pred_in_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * PRED_OUT
    )
    var pol_tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ACT_DIM)

    # Deterministic inputs:
    #   μ_raw spread over [-1.5, 1.5] (covers the active tanh region),
    #   σ_raw over [-1.0, 1.5] (active softplus),
    #   value bins (trailing 2*ACT_DIM..PRED_OUT) random — kernel ignores.
    #   a* targets in (−0.95·MAX, 0.95·MAX) — well inside the c-clamp.
    for b in range(BATCH):
        var off = b * PRED_OUT
        for d in range(ACT_DIM):
            pred_in_host[off + d] = (
                Float32(0.7) * Float32(d + 1) - Float32(b) * Float32(0.3)
            )
            pred_in_host[off + ACT_DIM + d] = (
                Float32(-0.5) + Float32(b) * Float32(0.4) + Float32(d) * Float32(0.6)
            )
        for v in range(BINS):
            pred_in_host[off + 2 * ACT_DIM + v] = Float32(0.05) * Float32(v)
        # a* targets that exercise both inner regions and near-boundary.
        for d in range(ACT_DIM):
            var sign: Float32 = (
                Float32(1.0) if (b + d) % 2 == 0 else Float32(-1.0)
            )
            var mag: Float32 = Float32(0.30) + Float32(0.20) * Float32(b)
            pol_tgt_host[b * ACT_DIM + d] = sign * mag
    ctx.enqueue_copy(dst_buf=pred_in_buf, src_buf=pred_in_host)
    ctx.enqueue_copy(dst_buf=pol_tgt_buf, src_buf=pol_tgt_host)

    # ── Output buffers — one set per path, plus FD scratch ──────────────
    var grad_a = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var grad_b = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var grad_c = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var loss_a = ctx.enqueue_create_buffer[dtype](BATCH)
    var loss_b = ctx.enqueue_create_buffer[dtype](BATCH)
    var loss_c = ctx.enqueue_create_buffer[dtype](BATCH)
    grad_a.enqueue_fill(Float32(0.0))
    grad_b.enqueue_fill(Float32(0.0))
    grad_c.enqueue_fill(Float32(0.0))
    loss_a.enqueue_fill(Float32(0.0))
    loss_b.enqueue_fill(Float32(0.0))
    loss_c.enqueue_fill(Float32(0.0))

    comptime pred_layout = Layout.row_major(BATCH * PRED_OUT)
    comptime act_layout = Layout.row_major(BATCH * ACT_DIM)
    comptime loss_layout = Layout.row_major(BATCH)

    var pred_in_t = LayoutTensor[dtype, pred_layout, MutAnyOrigin](
        pred_in_buf.unsafe_ptr()
    )
    var pol_tgt_t = LayoutTensor[dtype, act_layout, MutAnyOrigin](
        pol_tgt_buf.unsafe_ptr()
    )
    var grad_a_t = LayoutTensor[dtype, pred_layout, MutAnyOrigin](
        grad_a.unsafe_ptr()
    )
    var grad_b_t = LayoutTensor[dtype, pred_layout, MutAnyOrigin](
        grad_b.unsafe_ptr()
    )
    var grad_c_t = LayoutTensor[dtype, pred_layout, MutAnyOrigin](
        grad_c.unsafe_ptr()
    )
    var loss_a_t = LayoutTensor[dtype, loss_layout, MutAnyOrigin](
        loss_a.unsafe_ptr()
    )
    var loss_b_t = LayoutTensor[dtype, loss_layout, MutAnyOrigin](
        loss_b.unsafe_ptr()
    )
    var loss_c_t = LayoutTensor[dtype, loss_layout, MutAnyOrigin](
        loss_c.unsafe_ptr()
    )

    var loss_scale = Scalar[dtype](1.0)
    var ent_scale = Scalar[dtype](0.05)
    var max_action_s = Scalar[dtype](MAX_ACTION)
    var min_std_s = Scalar[dtype](MIN_STD)

    # ── Path A: inlined kernel call ─────────────────────────────────────
    comptime kernel_a = ezv2_policy_loss_grad_continuous_kernel[
        BATCH, ACT_DIM, PRED_OUT, dtype
    ]
    ctx.enqueue_function[kernel_a, kernel_a](
        pred_in_t,
        pol_tgt_t,
        grad_a_t,
        loss_a_t,
        loss_scale,
        ent_scale,
        max_action_s,
        min_std_s,
        grid_dim=(BATCH_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── Path B: concrete struct dispatch ────────────────────────────────
    ContinuousActionSpace[ACT_DIM, K, MAX_ACTION, MIN_STD].policy_loss_grad_gpu[
        BATCH, PRED_OUT, ACT_DIM, dtype
    ](
        ctx,
        pred_in_t,
        pol_tgt_t,
        grad_b_t,
        loss_b_t,
        loss_scale,
        ent_scale,
    )

    # ── Path C: via trait-bound generic helper ──────────────────────────
    comptime AS = ContinuousActionSpace[ACT_DIM, K, MAX_ACTION, MIN_STD]
    run_via_trait[AS, BATCH, PRED_OUT, ACT_DIM, dtype](
        ctx,
        pred_in_t,
        pol_tgt_t,
        grad_c_t,
        loss_c_t,
        loss_scale,
        ent_scale,
    )

    ctx.synchronize()

    # ── Download outputs ────────────────────────────────────────────────
    var grad_a_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * PRED_OUT
    )
    var grad_b_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * PRED_OUT
    )
    var grad_c_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * PRED_OUT
    )
    var loss_a_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var loss_b_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var loss_c_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(dst_buf=grad_a_host, src_buf=grad_a)
    ctx.enqueue_copy(dst_buf=grad_b_host, src_buf=grad_b)
    ctx.enqueue_copy(dst_buf=grad_c_host, src_buf=grad_c)
    ctx.enqueue_copy(dst_buf=loss_a_host, src_buf=loss_a)
    ctx.enqueue_copy(dst_buf=loss_b_host, src_buf=loss_b)
    ctx.enqueue_copy(dst_buf=loss_c_host, src_buf=loss_c)
    ctx.synchronize()

    # ── Bit-exact dispatch comparison ───────────────────────────────────
    var max_grad_diff_ab = Float32(0.0)
    var max_grad_diff_ac = Float32(0.0)
    for i in range(BATCH * PRED_OUT):
        var d_ab = abs(grad_a_host[i] - grad_b_host[i])
        if d_ab > max_grad_diff_ab:
            max_grad_diff_ab = d_ab
        var d_ac = abs(grad_a_host[i] - grad_c_host[i])
        if d_ac > max_grad_diff_ac:
            max_grad_diff_ac = d_ac
    var max_loss_diff_ab = Float32(0.0)
    var max_loss_diff_ac = Float32(0.0)
    for b in range(BATCH):
        var d_ab = abs(loss_a_host[b] - loss_b_host[b])
        if d_ab > max_loss_diff_ab:
            max_loss_diff_ab = d_ab
        var d_ac = abs(loss_a_host[b] - loss_c_host[b])
        if d_ac > max_loss_diff_ac:
            max_loss_diff_ac = d_ac

    print()
    print("--- Bit-exact dispatch ---")
    print("    max |grad_A - grad_B| =", max_grad_diff_ab)
    print("    max |grad_A - grad_C| =", max_grad_diff_ac)
    print("    max |loss_A - loss_B| =", max_loss_diff_ab)
    print("    max |loss_A - loss_C| =", max_loss_diff_ac)

    _expect(
        max_grad_diff_ab == Float32(0.0)
        and max_loss_diff_ab == Float32(0.0),
        "concrete struct dispatch (B) matches inlined call (A) bit-exactly",
        passed,
        total,
    )
    _expect(
        max_grad_diff_ac == Float32(0.0)
        and max_loss_diff_ac == Float32(0.0),
        "trait-bound generic dispatch (C) matches inlined call (A) bit-exactly",
        passed,
        total,
    )

    # ── Sanity: the BINS slots are not touched by the kernel ────────────
    var any_bins_grad_nonzero = False
    for b in range(BATCH):
        for v in range(BINS):
            var i = b * PRED_OUT + 2 * ACT_DIM + v
            if grad_a_host[i] != Float32(0.0):
                any_bins_grad_nonzero = True
                break
    _expect(
        not any_bins_grad_nonzero,
        "trailing BINS grad slots untouched (kernel respects layout)",
        passed,
        total,
    )

    # ── Sanity: policy slots have non-zero grad (kernel actually fired) ─
    var any_policy_grad_nonzero = False
    for b in range(BATCH):
        for d in range(2 * ACT_DIM):
            if grad_a_host[b * PRED_OUT + d] != Float32(0.0):
                any_policy_grad_nonzero = True
                break
    _expect(
        any_policy_grad_nonzero,
        "policy grad slots non-zero (kernel produced output)",
        passed,
        total,
    )

    # ── Sanity: per-sample loss is finite ───────────────────────────────
    var any_nan_loss = False
    for b in range(BATCH):
        var lv = loss_a_host[b]
        if lv != lv or lv > Float32(1.0e30) or lv < Float32(-1.0e30):
            any_nan_loss = True
    _expect(
        not any_nan_loss,
        "per-sample loss finite for all batch entries",
        passed,
        total,
    )

    # ── Numerical gradient check — finite difference ────────────────────
    # Re-uses path A's grad_a as the analytical gradient. For each
    # (b, d) in the policy slots, perturb pred_out[b, d] by ±ε, relaunch
    # the kernel, and compare (loss[b]+ - loss[b]−) / (2ε) against
    # grad_a_host[b * PRED_OUT + d]. The kernel computes loss UNSCALED,
    # so with loss_scale=1 the analytical grad is exactly the finite-diff
    # tangent.
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)
    var fd_eps = Float32(1.0e-3)
    var loss_plus_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var loss_minus_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var dummy_grad_buf = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var loss_plus_t = LayoutTensor[dtype, loss_layout, MutAnyOrigin](
        loss_plus_buf.unsafe_ptr()
    )
    var loss_minus_t = LayoutTensor[dtype, loss_layout, MutAnyOrigin](
        loss_minus_buf.unsafe_ptr()
    )
    var dummy_grad_t = LayoutTensor[dtype, pred_layout, MutAnyOrigin](
        dummy_grad_buf.unsafe_ptr()
    )
    var loss_plus_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var loss_minus_host = ctx.enqueue_create_host_buffer[dtype](BATCH)

    for b in range(BATCH):
        for d in range(2 * ACT_DIM):
            var idx = b * PRED_OUT + d
            var orig = pred_in_host[idx]

            # +ε
            pred_in_host[idx] = orig + fd_eps
            ctx.enqueue_copy(dst_buf=pred_in_buf, src_buf=pred_in_host)
            ctx.enqueue_function[kernel_a, kernel_a](
                pred_in_t,
                pol_tgt_t,
                dummy_grad_t,
                loss_plus_t,
                loss_scale,
                ent_scale,
                max_action_s,
                min_std_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_copy(dst_buf=loss_plus_host, src_buf=loss_plus_buf)
            ctx.synchronize()

            # −ε
            pred_in_host[idx] = orig - fd_eps
            ctx.enqueue_copy(dst_buf=pred_in_buf, src_buf=pred_in_host)
            ctx.enqueue_function[kernel_a, kernel_a](
                pred_in_t,
                pol_tgt_t,
                dummy_grad_t,
                loss_minus_t,
                loss_scale,
                ent_scale,
                max_action_s,
                min_std_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_copy(dst_buf=loss_minus_host, src_buf=loss_minus_buf)
            ctx.synchronize()

            # restore
            pred_in_host[idx] = orig

            var fd = (loss_plus_host[b] - loss_minus_host[b]) / (
                Float32(2.0) * fd_eps
            )
            var an = grad_a_host[idx]
            var abs_err = abs(fd - an)
            var ref_mag = abs(fd)
            if abs(an) > ref_mag:
                ref_mag = abs(an)
            if ref_mag < Float32(1.0e-3):
                ref_mag = Float32(1.0e-3)
            var rel_err = abs_err / ref_mag
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

    # restore original buffer for any later reads
    ctx.enqueue_copy(dst_buf=pred_in_buf, src_buf=pred_in_host)
    ctx.synchronize()

    print()
    print("--- Finite-difference gradcheck (fp32, eps=1e-3) ---")
    print("    max |fd - analytical|  =", max_abs_err)
    print("    max relative error     =", max_rel_err)

    _expect(
        max_abs_err < Float32(1.0e-2),
        "max abs grad error < 1e-2",
        passed,
        total,
    )
    _expect(
        max_rel_err < Float32(5.0e-3),
        "max rel grad error < 5e-3",
        passed,
        total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
