"""Conditional transformer block (AdaLN-zero) for the LeWM AR predictor.

Inlines the orchestration validated by `test_adaln_with_attention.mojo`:

    raw_mod    = AdaLNMod[D](Swish(c))            # (BT, 3D) — Linear[D, 3D]
    shift, scale, gate = chunk(raw_mod, 3)
    ln_x       = LayerNormNoAffine(x_prev)
    mod_x      = ln_x * (1 + scale) + shift
    attn_out   = MultiHeadAttention[D, HEADS, T, causal=True](mod_x)
    x_next     = x_prev + gate * attn_out

With `AdaLNMod` params (W + b) zero-initialised, gate = 0 → block is the
identity. This is the "AdaLN-zero" initialisation scheme used by DiT and
the LeWM reference AR predictor.

This is a single-branch block (MSA branch only). The reference also has
an MLP branch — a copy-paste of the same pattern with TransformerFFN as
the inner module.

Both functions follow the test's hand-orchestrated pattern but use
`Linear[D, 3D]` as the modulator (combined MatMul + BiasAdd via AutoFused)
instead of separate ops, so the cache + grad buffers are smaller.
"""

from layout import Layout, LayoutTensor
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from ...nn.constants import dtype
from ...nn.model import Linear
from ...nn.composites import MultiHeadAttention
from ...nn.autodiff.primitives import (
    SwishOp,
    ModulateOp,
    GateOp,
    LayerNormNoAffineOp,
)


# AdaLN modulator: per-token Linear[D, 3D] (combined MatMul + BiasAdd).
comptime AdaLNMod[D: Int] = Linear[D, 3 * D]


# =============================================================================
# Forward
# =============================================================================
def cond_block_forward[
    BATCH: Int, T: Int, D: Int, HEADS: Int,
](
    x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    c_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    # Output
    mut x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Caches (filled by forward, consumed by backward)
    mut silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    mut ln_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mut mod_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut gate_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Intermediate (filled by forward, needed at backward to find scale/shift/gate slots)
    mut raw_mod_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    # Scratch
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut attn_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
) raises:
    var empty_p = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

    # c → Swish → AdaLNMod → raw_mod (BT, 3D).
    SwishOp[D].eval[BATCH * T](c_t, silu_buf_t, empty_p, silu_cache_t)
    AdaLNMod[D].forward[BATCH * T](
        silu_buf_t, raw_mod_t, adaln_params, adaln_state, adaln_cache_t
    )

    # LN(x_prev) → ln_out_buf.
    LayerNormNoAffineOp[D].eval[BATCH * T](
        x_prev_t, ln_out_buf_t, empty_p, ln_cache_t
    )

    # Pack [ln_x | scale | shift]. raw_mod layout: [shift | scale | gate].
    for b in range(BATCH * T):
        for i in range(D):
            mod_inp_buf_t[b, i] = ln_out_buf_t[b, i]
            mod_inp_buf_t[b, D + i] = raw_mod_t[b, D + i]
            mod_inp_buf_t[b, 2 * D + i] = raw_mod_t[b, i]
    ModulateOp[D].eval[BATCH * T](
        mod_inp_buf_t, mod_x_buf_t, empty_p, mod_cache_t
    )

    # Reshape (BT, D) ↔ (BATCH, T*D) for MSA.
    var mod_x_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](mod_x_buf_t.ptr)
    var attn_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](attn_out_buf_t.ptr)
    MultiHeadAttention[D, HEADS, T, True].forward[BATCH](
        mod_x_btd_t, attn_btd_t, msa_params, msa_state, msa_cache_t
    )

    # Pack [x_prev | gate | attn_out]. raw_mod[2D:3D] is gate.
    for b in range(BATCH * T):
        for i in range(D):
            gate_inp_buf_t[b, i] = x_prev_t[b, i]
            gate_inp_buf_t[b, D + i] = raw_mod_t[b, 2 * D + i]
            gate_inp_buf_t[b, 2 * D + i] = attn_out_buf_t[b, i]
    GateOp[D].eval[BATCH * T](
        gate_inp_buf_t, x_next_t, empty_p, gate_cache_t
    )


# =============================================================================
# Backward
# =============================================================================
def cond_block_backward[
    BATCH: Int, T: Int, D: Int, HEADS: Int,
](
    grad_x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    # Caches from forward
    silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    ln_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mod_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    gate_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Output grads
    mut grad_x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut grad_c_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut g_adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    mut g_msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    # Scratch grads (caller-owned)
    mut sgg_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sgao_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmx_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmi_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sglnout_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sglnin_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgrm_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sgsc_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
) raises:
    var empty_p = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
    var empty_gp = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

    # Gate.vjp: grad_x_next → grad_gate_inp = [grad_x_residual | grad_gate | grad_attn_out].
    GateOp[D].vjp[BATCH * T](
        grad_x_next_t, sgg_t, empty_p, gate_cache_t, empty_gp
    )
    # Initialise grad_x_prev with the residual contribution. Initialise the
    # gate slot of grad_raw_mod (sgrm_t) and the attn_out grad.
    for b in range(BATCH * T):
        for i in range(D):
            grad_x_prev_t[b, i] = sgg_t[b, i]
            sgrm_t[b, 2 * D + i] = sgg_t[b, D + i]
            sgao_t[b, i] = sgg_t[b, 2 * D + i]

    # MSA.backward (BATCH, T*D layout).
    var sgao_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgao_t.ptr)
    var sgmx_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgmx_t.ptr)
    MultiHeadAttention[D, HEADS, T, True].backward[BATCH](
        sgao_btd_t,
        sgmx_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        g_msa_params,
    )

    # Modulate.vjp: grad_mod_x → grad_mod_inp = [grad_ln_out | grad_scale | grad_shift].
    ModulateOp[D].vjp[BATCH * T](
        sgmx_t, sgmi_t, empty_p, mod_cache_t, empty_gp
    )
    # Fill grad_ln_out and grad_raw_mod's scale + shift slots.
    for b in range(BATCH * T):
        for i in range(D):
            sglnout_t[b, i] = sgmi_t[b, i]
            sgrm_t[b, D + i] = sgmi_t[b, D + i]  # scale slot of raw_mod
            sgrm_t[b, i] = sgmi_t[b, 2 * D + i]  # shift slot of raw_mod

    # LN(no affine).vjp: grad_ln_out → grad_ln_in, add to grad_x_prev.
    LayerNormNoAffineOp[D].vjp[BATCH * T](
        sglnout_t, sglnin_t, empty_p, ln_cache_t, empty_gp
    )
    for b in range(BATCH * T):
        for i in range(D):
            grad_x_prev_t[b, i] = grad_x_prev_t[b, i] + sglnin_t[b, i]

    # AdaLNMod.backward: grad_raw_mod (sgrm_t) → grad_silu (sgsc_t),
    # accumulating into g_adaln_params.
    AdaLNMod[D].backward[BATCH * T](
        sgrm_t,
        sgsc_t,
        adaln_params,
        adaln_state,
        adaln_cache_t,
        g_adaln_params,
    )
    # Swish.vjp: grad_silu → grad_c.
    SwishOp[D].vjp[BATCH * T](
        sgsc_t, grad_c_t, empty_p, silu_cache_t, empty_gp
    )


# =============================================================================
# GPU forward
# =============================================================================
def cond_block_forward_gpu[
    BATCH: Int, T: Int, D: Int, HEADS: Int,
](
    ctx: DeviceContext,
    x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    c_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Caches
    mut silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    mut ln_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mut mod_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut gate_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Intermediate
    mut raw_mod_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    # Scratch
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut attn_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    # Workspace shared by AdaLNMod and MSA (sized to max of both).
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
) raises:
    comptime BT: Int = BATCH * T
    var empty_p = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
    # DiffOp eval_gpu/vjp_gpu take a raw workspace pointer; element-wise
    # ops don't actually consume it, so a null is safe.
    var op_ws = UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)

    # c → Swish → silu_buf
    SwishOp[D].eval_gpu[BT, dtype](
        ctx, silu_buf_t, c_t, empty_p, silu_cache_t, op_ws,
    )
    # AdaLNMod: silu_buf → raw_mod (Linear[D, 3D]).
    AdaLNMod[D].forward_gpu[BT, dtype](
        ctx,
        raw_mod_t,
        silu_buf_t,
        adaln_params,
        adaln_state,
        adaln_cache_t,
        adaln_workspace,
    )
    # x_prev → LN → ln_out
    LayerNormNoAffineOp[D].eval_gpu[BT, dtype](
        ctx, ln_out_buf_t, x_prev_t, empty_p, ln_cache_t, op_ws,
    )

    # Pack mod_inp = [ln_out | scale=raw_mod[D:2D] | shift=raw_mod[0:D]]
    @parameter
    @always_inline
    def pack_mod_inp_kernel(
        ln_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
        raw_mod: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
        mod_inp: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
    ):
        var b = Int(global_idx.x)
        var d_idx = Int(global_idx.y)
        if b < BT and d_idx < D:
            mod_inp[b, d_idx] = ln_out[b, d_idx]
            mod_inp[b, D + d_idx] = raw_mod[b, D + d_idx]
            mod_inp[b, 2 * D + d_idx] = raw_mod[b, d_idx]

    comptime TPB_X = 16
    comptime TPB_Y = 16
    ctx.enqueue_function[pack_mod_inp_kernel](
        ln_out_buf_t,
        raw_mod_t,
        mod_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # Modulate
    ModulateOp[D].eval_gpu[BT, dtype](
        ctx, mod_x_buf_t, mod_inp_buf_t, empty_p, mod_cache_t, op_ws,
    )
    # MSA — view (BT, D) as (BATCH, T*D).
    var mod_x_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](mod_x_buf_t.ptr)
    var attn_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](attn_out_buf_t.ptr)
    MultiHeadAttention[D, HEADS, T, True].forward_gpu[BATCH, dtype](
        ctx,
        attn_btd_t,
        mod_x_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        msa_workspace,
    )

    # Pack gate_inp = [x_prev | gate=raw_mod[2D:3D] | attn_out]
    @parameter
    @always_inline
    def pack_gate_inp_kernel(
        x_prev: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
        raw_mod: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
        attn_out: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ],
        gate_inp: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
    ):
        var b = Int(global_idx.x)
        var d_idx = Int(global_idx.y)
        if b < BT and d_idx < D:
            gate_inp[b, d_idx] = x_prev[b, d_idx]
            gate_inp[b, D + d_idx] = raw_mod[b, 2 * D + d_idx]
            gate_inp[b, 2 * D + d_idx] = attn_out[b, d_idx]

    ctx.enqueue_function[pack_gate_inp_kernel](
        x_prev_t,
        raw_mod_t,
        attn_out_buf_t,
        gate_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # Gate
    GateOp[D].eval_gpu[BT, dtype](
        ctx, x_next_t, gate_inp_buf_t, empty_p, gate_cache_t, op_ws,
    )


# =============================================================================
# GPU backward
# =============================================================================
def cond_block_backward_gpu[
    BATCH: Int, T: Int, D: Int, HEADS: Int,
](
    ctx: DeviceContext,
    grad_x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    ln_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mod_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    gate_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut grad_x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut grad_c_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut g_adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    mut g_msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    mut sgg_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sgao_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmx_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmi_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sglnout_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sglnin_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgrm_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sgsc_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
) raises:
    comptime BT: Int = BATCH * T
    comptime TPB_X = 16
    comptime TPB_Y = 16

    var empty_p = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
    var empty_gp = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
    var op_ws = UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)

    # Gate.vjp: grad_x_next → sgg = [grad_x_residual | grad_gate | grad_attn_out].
    GateOp[D].vjp_gpu[BT, dtype](
        ctx, grad_x_next_t, sgg_t, empty_p, gate_cache_t, empty_gp, op_ws,
    )

    # Split sgg → grad_x_prev (residual slot), sgrm[2D:3D] (gate slot), sgao.
    @parameter
    @always_inline
    def split_gate_grad_kernel(
        sgg: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
        grad_x_prev: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ],
        sgrm: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
        sgao: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    ):
        var b = Int(global_idx.x)
        var d_idx = Int(global_idx.y)
        if b < BT and d_idx < D:
            grad_x_prev[b, d_idx] = sgg[b, d_idx]
            sgrm[b, 2 * D + d_idx] = sgg[b, D + d_idx]
            sgao[b, d_idx] = sgg[b, 2 * D + d_idx]

    ctx.enqueue_function[split_gate_grad_kernel](
        sgg_t,
        grad_x_prev_t,
        sgrm_t,
        sgao_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # MSA backward — operates on (BATCH, T*D) view.
    var sgao_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgao_t.ptr)
    var sgmx_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgmx_t.ptr)
    MultiHeadAttention[D, HEADS, T, True].backward_gpu[BATCH, dtype](
        ctx,
        sgmx_btd_t,
        sgao_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        g_msa_params,
        msa_workspace,
    )

    # Modulate.vjp: sgmx → sgmi = [grad_ln_out | grad_scale | grad_shift].
    ModulateOp[D].vjp_gpu[BT, dtype](
        ctx, sgmx_t, sgmi_t, empty_p, mod_cache_t, empty_gp, op_ws,
    )

    # Split sgmi → sglnout, sgrm[D:2D] (scale slot), sgrm[0:D] (shift slot).
    @parameter
    @always_inline
    def split_mod_grad_kernel(
        sgmi: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
        sglnout: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ],
        sgrm: LayoutTensor[
            dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin
        ],
    ):
        var b = Int(global_idx.x)
        var d_idx = Int(global_idx.y)
        if b < BT and d_idx < D:
            sglnout[b, d_idx] = sgmi[b, d_idx]
            sgrm[b, D + d_idx] = sgmi[b, D + d_idx]
            sgrm[b, d_idx] = sgmi[b, 2 * D + d_idx]

    ctx.enqueue_function[split_mod_grad_kernel](
        sgmi_t,
        sglnout_t,
        sgrm_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # LN(no affine).vjp: sglnout → sglnin.
    LayerNormNoAffineOp[D].vjp_gpu[BT, dtype](
        ctx, sglnout_t, sglnin_t, empty_p, ln_cache_t, empty_gp, op_ws,
    )

    # Accumulate sglnin into grad_x_prev (already holds the residual contribution).
    @parameter
    @always_inline
    def accum_lnin_kernel(
        grad_x_prev: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ],
        sglnin: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ],
    ):
        var b = Int(global_idx.x)
        var d_idx = Int(global_idx.y)
        if b < BT and d_idx < D:
            grad_x_prev[b, d_idx] = grad_x_prev[b, d_idx] + sglnin[b, d_idx]

    ctx.enqueue_function[accum_lnin_kernel](
        grad_x_prev_t,
        sglnin_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # AdaLNMod.backward: sgrm → sgsc, accumulating into g_adaln_params.
    AdaLNMod[D].backward_gpu[BT, dtype](
        ctx,
        sgsc_t,
        sgrm_t,
        adaln_params,
        adaln_state,
        adaln_cache_t,
        g_adaln_params,
        adaln_workspace,
    )
    # Swish.vjp: sgsc → grad_c.
    SwishOp[D].vjp_gpu[BT, dtype](
        ctx, sgsc_t, grad_c_t, empty_p, silu_cache_t, empty_gp, op_ws,
    )
