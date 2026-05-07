"""Spike: prove `ActionSpace` trait dispatch produces the same output as
a direct kernel enqueue.

Tests three call paths against the same input:

  Path A — inlined: direct `ezv2_policy_loss_grad_kernel` enqueue.
  Path B — concrete struct dispatch:
           `DiscreteActionSpace[ACT, K].policy_loss_grad_gpu[...]`.
  Path C — generic via trait constraint:
           `def run_via_trait[AS: ActionSpace, ...]` calling
           `AS.policy_loss_grad_gpu[...]`.

Path B verifies trait conformance (concrete struct conforming to
`ActionSpace` and exposing the static method dispatches correctly).
Path C verifies the actual deployment shape — the BPTT core will receive
`Config.ActSpace` (a trait-bound comptime type) and call
`Config.ActSpace.policy_loss_grad_gpu[...]`. If A == B == C bit-exactly,
the dispatch pattern is sound and the larger extraction can proceed.
"""

from std.gpu.host import DeviceContext
from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.deep_agents.efficient_zero_v2.action_space import (
    ActionSpace,
    DiscreteActionSpace,
)
from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_policy_loss_grad_kernel,
)


comptime TPB: Int = 256


# ───────────────────────────────────────────────────────────────────────────
# Path C — generic helper that dispatches via a trait-bound type parameter.
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
    """The shape of the call site inside `ezv2_train_step_gpu_core`.

    `AS` arrives as a trait-bound comptime type parameter — modeling
    `Config.ActSpace` from the agent's perspective. Inside the body we
    call `AS.policy_loss_grad_gpu[...]` exactly as the BPTT core will.
    """
    AS.policy_loss_grad_gpu[BATCH, PRED_OUT, POL_TGT_DIM, dtype](
        ctx,
        pred_out_step,
        policy_target_step,
        grad_pred_out_step,
        per_sample_loss,
        loss_scale,
        ent_scale,
    )


# ───────────────────────────────────────────────────────────────────────────
# Test scaffolding
# ───────────────────────────────────────────────────────────────────────────


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
    print("=== EZ-V2 ActionSpace dispatch spike ===")
    var passed = 0
    var total = 0

    comptime BATCH = 4
    comptime ACT = 5
    comptime BINS = 21
    comptime PRED_OUT = ACT + BINS  # 26
    comptime dtype = DType.float32
    comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

    var ctx = DeviceContext()

    # ── Allocate one set of input buffers (read-only across all paths) ──
    var pred_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * PRED_OUT)
    var pol_tgt_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACT)

    # Fill inputs with deterministic values so all three paths see the
    # same starting state.
    var pred_in_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * PRED_OUT
    )
    var pol_tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ACT)
    for i in range(BATCH * PRED_OUT):
        pred_in_host[i] = Float32(0.1) * Float32(i) - Float32(1.0)
    for b in range(BATCH):
        # Make policy_target a valid distribution per row.
        var s = Float32(0.0)
        for a in range(ACT):
            pol_tgt_host[b * ACT + a] = Float32(a + 1)  # 1..ACT
            s += Float32(a + 1)
        for a in range(ACT):
            pol_tgt_host[b * ACT + a] = pol_tgt_host[b * ACT + a] / s
    ctx.enqueue_copy(dst_buf=pred_in_buf, src_buf=pred_in_host)
    ctx.enqueue_copy(dst_buf=pol_tgt_buf, src_buf=pol_tgt_host)

    # ── Three sets of output buffers — one per path ─────────────────────
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

    # ── Build LayoutTensor views ────────────────────────────────────────
    comptime pred_layout = Layout.row_major(BATCH * PRED_OUT)
    comptime act_layout = Layout.row_major(BATCH * ACT)
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

    var loss_scale = Scalar[dtype](0.5)
    var ent_scale = Scalar[dtype](0.0)

    # ── Path A: inlined kernel call ─────────────────────────────────────
    comptime kernel_a = ezv2_policy_loss_grad_kernel[
        BATCH, ACT, PRED_OUT, dtype
    ]
    ctx.enqueue_function[kernel_a, kernel_a](
        pred_in_t,
        pol_tgt_t,
        grad_a_t,
        loss_a_t,
        loss_scale,
        grid_dim=(BATCH_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── Path B: concrete struct dispatch ────────────────────────────────
    DiscreteActionSpace[ACT, 4].policy_loss_grad_gpu[
        BATCH, PRED_OUT, ACT, dtype
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
    comptime AS = DiscreteActionSpace[ACT, 4]
    run_via_trait[AS, BATCH, PRED_OUT, ACT, dtype](
        ctx,
        pred_in_t,
        pol_tgt_t,
        grad_c_t,
        loss_c_t,
        loss_scale,
        ent_scale,
    )

    ctx.synchronize()

    # ── Download all outputs and compare ────────────────────────────────
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
    print("--- Bit-exact comparison ---")
    print("    max |grad_A - grad_B| =", max_grad_diff_ab)
    print("    max |grad_A - grad_C| =", max_grad_diff_ac)
    print("    max |loss_A - loss_B| =", max_loss_diff_ab)
    print("    max |loss_A - loss_C| =", max_loss_diff_ac)
    print()
    print("    grad_A[0..6] =", grad_a_host[0], grad_a_host[1],
          grad_a_host[2], grad_a_host[3], grad_a_host[4], grad_a_host[5])
    print("    loss_A      =", loss_a_host[0], loss_a_host[1],
          loss_a_host[2], loss_a_host[3])

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

    # Sanity: outputs are not all zero (kernel actually fired).
    var any_nonzero_grad = False
    for i in range(BATCH * PRED_OUT):
        if grad_a_host[i] != Float32(0.0):
            any_nonzero_grad = True
            break
    _expect(
        any_nonzero_grad,
        "path A produced non-zero gradient (kernel actually executed)",
        passed,
        total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
