"""Conditional transformer block (AdaLN-zero) for the LeWM AR predictor.

Reference `references/le-wm-main/module.py:88-111` (ConditionalBlock).

Each block has TWO residual branches, each gated by AdaLN-zero:

    raw_mod = AdaLNMod[D](Swish(c))            # (BT, 6D) — Linear[D, 6D]
    shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp = chunk(raw_mod, 6)

    # MSA branch
    ln1_x   = LayerNormNoAffine(x_prev)
    mod1_x  = ln1_x * (1 + scale_msa) + shift_msa
    attn    = MSA(mod1_x)
    x_mid   = x_prev + gate_msa * attn

    # MLP branch
    ln2_x   = LayerNormNoAffine(x_mid)
    mod2_x  = ln2_x * (1 + scale_mlp) + shift_mlp
    mlp_out = MLP(mod2_x)                       # Linear→GELU→Linear
    x_next  = x_mid + gate_mlp * mlp_out

With AdaLNMod (W + b) zero-initialised, gate_msa = gate_mlp = 0 →
block is identity at step 0. This is the "AdaLN-zero" scheme used by
DiT and the LeWM reference.

The GPU forward/backward orchestrate the kernel chain by hand. The MSA
branch and the MLP branch share their pack/split helper kernels via an
OFFSET parameter — OFFSET=0 reads raw_mod's MSA slots, OFFSET=3*D reads
the MLP slots.

CPU `cond_block_forward` / `cond_block_backward` exist for backward
compatibility (existing CPU trainer + tests). They use ONLY the MSA
branch — the MLP slots of raw_mod are unused. Callers must zero
`sgrm_t[:, 3D..6D]` before invoking the CPU backward.
"""

from layout import Layout, LayoutTensor
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.memory import UnsafePointer


@always_inline
def _null_ptr[T: AnyType, O: Origin]() -> UnsafePointer[T, O]:
    """NULL UnsafePointer for zero-param / zero-workspace placeholders.

    Mojo nightly's comptime `unsafe_from_address=0` literal is rejected;
    the runtime-Int overload still accepts 0.
    """
    var addr: Int = 0
    return UnsafePointer[T, O](unsafe_from_address=addr)

from ...nn.constants import dtype
from ...nn.model import Linear, Sequential
from ...nn.model.autodiff_layers import GELU
from ...nn.composites import MultiHeadAttention, MultiHeadAttentionXL
from ...nn.autodiff.primitives import (
    SwishOp,
    ModulateOp,
    GateOp,
    LayerNormNoAffineOp,
)


# AdaLN modulator: per-token Linear[D, 6D] — produces 6 chunks
# (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
# raw_mod layout:
#   [0..D)        shift_msa
#   [D..2D)       scale_msa
#   [2D..3D)      gate_msa
#   [3D..4D)      shift_mlp
#   [4D..5D)      scale_mlp
#   [5D..6D)      gate_mlp
# Matches reference `references/le-wm-main/module.py:99` (Linear[D, 6*D]).
comptime AdaLNMod[D: Int] = Linear[D, 6 * D]


# Per-token feed-forward used by the MLP branch.
#   Linear[D, FF] → GELU → Linear[FF, D]
# Reference `references/le-wm-main/module.py:38-53` FeedForward.
# (We skip the internal `nn.LayerNorm(dim)` in their FFN: it sits right
# after `norm_no_affine + modulate`, which already standardises, so the
# inner LN is mostly redundant. Reduces kernel count + cache size.)
comptime CondMLP[D: Int, FF: Int] = Sequential[
    Linear[D, FF],
    GELU[FF],
    Linear[FF, D],
]


# =============================================================================
# Forward (CPU — backward-compat, MSA branch only)
# =============================================================================
def cond_block_forward[
    BATCH: Int,
    T: Int,
    D: Int,
    HEADS: Int,
    DIM_HEAD: Int,
](
    x_prev_t: LayoutTensor[dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin],
    c_t: LayoutTensor[dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin],
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
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
            BATCH, MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut gate_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut raw_mod_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 6 * D), MutAnyOrigin
    ],
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
    # Empty (zero-size) LayoutTensor slot; the pointer is never read.
    # Reuse `silu_buf_t`'s base ptr as a valid non-null placeholder.
    var empty_p = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](silu_buf_t.ptr)
    )

    SwishOp[D].eval[BATCH * T](c_t, silu_buf_t, empty_p, silu_cache_t)
    AdaLNMod[D].forward[BATCH * T](
        silu_buf_t, raw_mod_t, adaln_params, adaln_state, adaln_cache_t
    )
    LayerNormNoAffineOp[D].eval[BATCH * T](
        x_prev_t, ln_out_buf_t, empty_p, ln_cache_t
    )
    for b in range(BATCH * T):
        for i in range(D):
            mod_inp_buf_t[b, i] = ln_out_buf_t[b, i]
            mod_inp_buf_t[b, D + i] = raw_mod_t[b, D + i]
            mod_inp_buf_t[b, 2 * D + i] = raw_mod_t[b, i]
    ModulateOp[D].eval[BATCH * T](
        mod_inp_buf_t, mod_x_buf_t, empty_p, mod_cache_t
    )
    var mod_x_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](mod_x_buf_t.ptr)
    var attn_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](attn_out_buf_t.ptr)
    MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].forward[BATCH](
        mod_x_btd_t, attn_btd_t, msa_params, msa_state, msa_cache_t
    )
    for b in range(BATCH * T):
        for i in range(D):
            gate_inp_buf_t[b, i] = x_prev_t[b, i]
            gate_inp_buf_t[b, D + i] = raw_mod_t[b, 2 * D + i]
            gate_inp_buf_t[b, 2 * D + i] = attn_out_buf_t[b, i]
    GateOp[D].eval[BATCH * T](gate_inp_buf_t, x_next_t, empty_p, gate_cache_t)


# =============================================================================
# Backward (CPU — backward-compat, MSA branch only)
# =============================================================================
def cond_block_backward[
    BATCH: Int,
    T: Int,
    D: Int,
    HEADS: Int,
    DIM_HEAD: Int,
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
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].STATE_SIZE
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
            BATCH, MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].CACHE_SIZE
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
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
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
    # sgrm = gradient w.r.t. raw_mod (full AdaLN output) — 6D wide.
    # Caller must pre-zero sgrm_t[:, 3D..6D] (MLP slots) since CPU
    # backward fills only the MSA slots.
    mut sgrm_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 6 * D), MutAnyOrigin
    ],
    mut sgsc_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
) raises:
    # Zero-size LayoutTensor slots; pointers are never read.
    # Reuse sgg_t's base ptr as a valid non-null placeholder.
    var empty_p = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](sgg_t.ptr)
    )
    var empty_gp = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](sgg_t.ptr)
    )

    GateOp[D].vjp[BATCH * T](
        grad_x_next_t, sgg_t, empty_p, gate_cache_t, empty_gp
    )
    for b in range(BATCH * T):
        for i in range(D):
            grad_x_prev_t[b, i] = sgg_t[b, i]
            sgrm_t[b, 2 * D + i] = sgg_t[b, D + i]
            sgao_t[b, i] = sgg_t[b, 2 * D + i]

    var sgao_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgao_t.ptr)
    var sgmx_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgmx_t.ptr)
    MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].backward[BATCH](
        sgao_btd_t,
        sgmx_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        g_msa_params,
    )

    ModulateOp[D].vjp[BATCH * T](sgmx_t, sgmi_t, empty_p, mod_cache_t, empty_gp)
    for b in range(BATCH * T):
        for i in range(D):
            sglnout_t[b, i] = sgmi_t[b, i]
            sgrm_t[b, D + i] = sgmi_t[b, D + i]
            sgrm_t[b, i] = sgmi_t[b, 2 * D + i]

    LayerNormNoAffineOp[D].vjp[BATCH * T](
        sglnout_t, sglnin_t, empty_p, ln_cache_t, empty_gp
    )
    for b in range(BATCH * T):
        for i in range(D):
            grad_x_prev_t[b, i] = grad_x_prev_t[b, i] + sglnin_t[b, i]

    AdaLNMod[D].backward[BATCH * T](
        sgrm_t,
        sgsc_t,
        adaln_params,
        adaln_state,
        adaln_cache_t,
        g_adaln_params,
    )
    SwishOp[D].vjp[BATCH * T](sgsc_t, grad_c_t, empty_p, silu_cache_t, empty_gp)


# =============================================================================
# Module-level GPU helper kernels (shared between MSA and MLP branches via
# OFFSET parameter — OFFSET=0 reads raw_mod's MSA slots [0..3D), OFFSET=3*D
# reads MLP slots [3D..6D)).
# =============================================================================


def cb_pack_mod_inp_kernel[
    BT: Int,
    D: Int,
    OFFSET: Int,
](
    ln_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    raw_mod: LayoutTensor[dtype, Layout.row_major(BT, 6 * D), MutAnyOrigin],
    mod_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
):
    """Pack mod_inp = [ln_out | scale | shift].
    scale = raw_mod[:, OFFSET+D : OFFSET+2D]
    shift = raw_mod[:, OFFSET   : OFFSET+D].
    """
    var b = Int(global_idx.x)
    var d_idx = Int(global_idx.y)
    if b < BT and d_idx < D:
        mod_inp[b, d_idx] = ln_out[b, d_idx]
        mod_inp[b, D + d_idx] = raw_mod[b, OFFSET + D + d_idx]
        mod_inp[b, 2 * D + d_idx] = raw_mod[b, OFFSET + d_idx]


def cb_pack_gate_inp_kernel[
    BT: Int,
    D: Int,
    OFFSET: Int,
](
    x_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    raw_mod: LayoutTensor[dtype, Layout.row_major(BT, 6 * D), MutAnyOrigin],
    branch_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    gate_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
):
    """Pack gate_inp = [x | gate | branch_out].
    gate = raw_mod[:, OFFSET+2D : OFFSET+3D].
    """
    var b = Int(global_idx.x)
    var d_idx = Int(global_idx.y)
    if b < BT and d_idx < D:
        gate_inp[b, d_idx] = x_in[b, d_idx]
        gate_inp[b, D + d_idx] = raw_mod[b, OFFSET + 2 * D + d_idx]
        gate_inp[b, 2 * D + d_idx] = branch_out[b, d_idx]


def cb_split_gate_grad_kernel[
    BT: Int,
    D: Int,
    OFFSET: Int,
](
    sgg: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    grad_x_residual: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    sgrm: LayoutTensor[dtype, Layout.row_major(BT, 6 * D), MutAnyOrigin],
    grad_branch_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
):
    """Split sgg = grad of [x | gate | branch_out] → 3 outputs.
    sgrm[:, OFFSET+2D : OFFSET+3D] receives the gate gradient.
    """
    var b = Int(global_idx.x)
    var d_idx = Int(global_idx.y)
    if b < BT and d_idx < D:
        grad_x_residual[b, d_idx] = sgg[b, d_idx]
        sgrm[b, OFFSET + 2 * D + d_idx] = sgg[b, D + d_idx]
        grad_branch_out[b, d_idx] = sgg[b, 2 * D + d_idx]


def cb_split_mod_grad_kernel[
    BT: Int,
    D: Int,
    OFFSET: Int,
](
    sgmi: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    sglnout: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    sgrm: LayoutTensor[dtype, Layout.row_major(BT, 6 * D), MutAnyOrigin],
):
    """Split sgmi = grad of [ln_out | scale | shift] → 3 outputs.
    sgrm[:, OFFSET+D : OFFSET+2D] ← scale gradient
    sgrm[:, OFFSET   : OFFSET+D]  ← shift gradient.
    """
    var b = Int(global_idx.x)
    var d_idx = Int(global_idx.y)
    if b < BT and d_idx < D:
        sglnout[b, d_idx] = sgmi[b, d_idx]
        sgrm[b, OFFSET + D + d_idx] = sgmi[b, D + d_idx]
        sgrm[b, OFFSET + d_idx] = sgmi[b, 2 * D + d_idx]


def cb_accum_kernel[
    BT: Int,
    D: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
):
    """Formula: dst[b, d] += src[b, d]."""
    var b = Int(global_idx.x)
    var d_idx = Int(global_idx.y)
    if b < BT and d_idx < D:
        dst[b, d_idx] = dst[b, d_idx] + src[b, d_idx]


# =============================================================================
# GPU forward — dual branch (MSA + MLP)
# =============================================================================
def cond_block_forward_gpu[
    BATCH: Int,
    T: Int,
    D: Int,
    HEADS: Int,
    DIM_HEAD: Int,
    FF: Int,
](
    ctx: DeviceContext,
    x_prev_t: LayoutTensor[dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin],
    c_t: LayoutTensor[dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin],
    # Params + states
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mlp_params: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    mlp_state: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].STATE_SIZE), MutAnyOrigin
    ],
    # Output
    mut x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Caches — MSA branch
    mut silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    mut ln1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mut mod1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut gate1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Caches — MLP branch
    mut ln2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mut mod2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mut mlp_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, CondMLP[D, FF].CACHE_SIZE),
        MutAnyOrigin,
    ],
    mut gate2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Intermediates (needed at backward too)
    mut raw_mod_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 6 * D), MutAnyOrigin
    ],
    mut x_mid_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Scratch (reused between branches)
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
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    # Workspaces
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
    mlp_workspace: DeviceBuffer[dtype],
) raises:
    comptime BT: Int = BATCH * T
    comptime TPB_X = 16
    comptime TPB_Y = 16
    # Zero-length placeholders for ops with no params / no workspace.
    var empty_p = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        _null_ptr[Scalar[dtype], MutAnyOrigin]()
    )
    var op_ws = _null_ptr[Scalar[dtype], MutAnyOrigin]()

    # ---- Shared head: c → Swish → AdaLNMod → raw_mod (BT, 6D) ----
    SwishOp[D].eval_gpu[BT, dtype](
        ctx,
        silu_buf_t,
        c_t,
        empty_p,
        silu_cache_t,
        op_ws,
    )
    AdaLNMod[D].forward_gpu[BT, dtype](
        ctx,
        raw_mod_t,
        silu_buf_t,
        adaln_params,
        adaln_state,
        adaln_cache_t,
        adaln_workspace,
    )

    # ============================== MSA branch ==============================
    # LN1(x_prev) → ln_out_buf
    LayerNormNoAffineOp[D].eval_gpu[BT, dtype](
        ctx,
        ln_out_buf_t,
        x_prev_t,
        empty_p,
        ln1_cache_t,
        op_ws,
    )
    # pack mod_inp from MSA slots (OFFSET=0)
    ctx.enqueue_function[cb_pack_mod_inp_kernel[BT, D, 0]](
        ln_out_buf_t,
        raw_mod_t,
        mod_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # Modulate
    ModulateOp[D].eval_gpu[BT, dtype](
        ctx,
        mod_x_buf_t,
        mod_inp_buf_t,
        empty_p,
        mod1_cache_t,
        op_ws,
    )
    # MSA over (BATCH, T*D) view
    var mod_x_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](mod_x_buf_t.ptr)
    var attn_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](branch_out_buf_t.ptr)
    MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].forward_gpu[BATCH, dtype](
        ctx,
        attn_btd_t,
        mod_x_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        msa_workspace,
    )
    # pack gate_inp from MSA gate slot (OFFSET=0; gate at raw_mod[2D:3D])
    ctx.enqueue_function[cb_pack_gate_inp_kernel[BT, D, 0]](
        x_prev_t,
        raw_mod_t,
        branch_out_buf_t,
        gate_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # Gate1: x_mid = x_prev + gate_msa * attn_out
    GateOp[D].eval_gpu[BT, dtype](
        ctx,
        x_mid_buf_t,
        gate_inp_buf_t,
        empty_p,
        gate1_cache_t,
        op_ws,
    )

    # ============================== MLP branch ==============================
    # LN2(x_mid) → ln_out_buf  (REUSE scratch — ln_cache_t is separate)
    LayerNormNoAffineOp[D].eval_gpu[BT, dtype](
        ctx,
        ln_out_buf_t,
        x_mid_buf_t,
        empty_p,
        ln2_cache_t,
        op_ws,
    )
    # pack mod_inp from MLP slots (OFFSET=3*D)
    ctx.enqueue_function[cb_pack_mod_inp_kernel[BT, D, 3 * D]](
        ln_out_buf_t,
        raw_mod_t,
        mod_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # Modulate
    ModulateOp[D].eval_gpu[BT, dtype](
        ctx,
        mod_x_buf_t,
        mod_inp_buf_t,
        empty_p,
        mod2_cache_t,
        op_ws,
    )
    # MLP: per-token Linear→GELU→Linear (no T-mixing). Input/output as (BT, D).
    CondMLP[D, FF].forward_gpu[BT, dtype](
        ctx,
        branch_out_buf_t,
        mod_x_buf_t,
        mlp_params,
        mlp_state,
        mlp_cache_t,
        mlp_workspace,
    )
    # pack gate_inp from MLP gate slot (OFFSET=3*D; gate at raw_mod[5D:6D])
    ctx.enqueue_function[cb_pack_gate_inp_kernel[BT, D, 3 * D]](
        x_mid_buf_t,
        raw_mod_t,
        branch_out_buf_t,
        gate_inp_buf_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # Gate2: x_next = x_mid + gate_mlp * mlp_out
    GateOp[D].eval_gpu[BT, dtype](
        ctx,
        x_next_t,
        gate_inp_buf_t,
        empty_p,
        gate2_cache_t,
        op_ws,
    )


# =============================================================================
# GPU backward — dual branch (MLP first, then MSA)
# =============================================================================
def cond_block_backward_gpu[
    BATCH: Int,
    T: Int,
    D: Int,
    HEADS: Int,
    DIM_HEAD: Int,
    FF: Int,
](
    ctx: DeviceContext,
    grad_x_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Params + states
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mlp_params: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    mlp_state: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].STATE_SIZE), MutAnyOrigin
    ],
    # Caches from forward — MSA branch
    silu_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    adaln_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, AdaLNMod[D].CACHE_SIZE),
        MutAnyOrigin,
    ],
    ln1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mod1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    msa_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    gate1_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # Caches from forward — MLP branch
    ln2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D + 1), MutAnyOrigin
    ],
    mod2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    mlp_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * T, CondMLP[D, FF].CACHE_SIZE),
        MutAnyOrigin,
    ],
    gate2_cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 2 * D), MutAnyOrigin
    ],
    # x_mid_buf from forward (needed by gate2 residual side + ln2 input)
    x_mid_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
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
        Layout.row_major(
            MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    mut g_mlp_params: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    # Scratch grads (caller-owned, reused across branches where possible)
    mut sgg_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sgbo_t: LayoutTensor[  # grad of branch_out (attn or mlp output)
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmx_t: LayoutTensor[  # grad of mod_x
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sgmi_t: LayoutTensor[  # grad of mod_inp (3D)
        dtype, Layout.row_major(BATCH * T, 3 * D), MutAnyOrigin
    ],
    mut sglnout_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut sglnin_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # sgrm = gradient w.r.t. raw_mod — 6D, filled across both branches.
    mut sgrm_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, 6 * D), MutAnyOrigin
    ],
    mut sgsc_t: LayoutTensor[  # grad of silu output
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    mut grad_x_mid_t: LayoutTensor[  # grad flowing into x_mid (from MLP branch)
        dtype, Layout.row_major(BATCH * T, D), MutAnyOrigin
    ],
    # Workspaces
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
    mlp_workspace: DeviceBuffer[dtype],
) raises:
    comptime BT: Int = BATCH * T
    comptime TPB_X = 16
    comptime TPB_Y = 16
    # Zero-length placeholders for ops with no params / no workspace.
    var empty_p = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        _null_ptr[Scalar[dtype], MutAnyOrigin]()
    )
    var empty_gp = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        _null_ptr[Scalar[dtype], MutAnyOrigin]()
    )
    var op_ws = _null_ptr[Scalar[dtype], MutAnyOrigin]()

    # ============================== MLP branch backward =====================
    # Gate2.vjp: grad_x_next → sgg = [grad_x_mid_resid | grad_gate_mlp | grad_mlp_out]
    GateOp[D].vjp_gpu[BT, dtype](
        ctx,
        grad_x_next_t,
        sgg_t,
        empty_p,
        gate2_cache_t,
        empty_gp,
        op_ws,
    )
    # Split: residual → grad_x_mid; gate slot → sgrm[5D:6D]; branch_out grad → sgbo
    ctx.enqueue_function[cb_split_gate_grad_kernel[BT, D, 3 * D]](
        sgg_t,
        grad_x_mid_t,
        sgrm_t,
        sgbo_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # MLP.backward: grad_mlp_out → grad_mod_x_mlp (sgmx_t)
    CondMLP[D, FF].backward_gpu[BT, dtype](
        ctx,
        sgmx_t,
        sgbo_t,
        mlp_params,
        mlp_state,
        mlp_cache_t,
        g_mlp_params,
        mlp_workspace,
    )
    # Modulate2.vjp: sgmx → sgmi
    ModulateOp[D].vjp_gpu[BT, dtype](
        ctx,
        sgmx_t,
        sgmi_t,
        empty_p,
        mod2_cache_t,
        empty_gp,
        op_ws,
    )
    # Split sgmi → sglnout, sgrm[4D:5D] (scale_mlp), sgrm[3D:4D] (shift_mlp)
    ctx.enqueue_function[cb_split_mod_grad_kernel[BT, D, 3 * D]](
        sgmi_t,
        sglnout_t,
        sgrm_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # LN2.vjp: sglnout → sglnin
    LayerNormNoAffineOp[D].vjp_gpu[BT, dtype](
        ctx,
        sglnout_t,
        sglnin_t,
        empty_p,
        ln2_cache_t,
        empty_gp,
        op_ws,
    )
    # Accumulate sglnin into grad_x_mid (residual already there).
    ctx.enqueue_function[cb_accum_kernel[BT, D]](
        grad_x_mid_t,
        sglnin_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # ============================== MSA branch backward =====================
    # Gate1.vjp: grad_x_mid → sgg = [grad_x_prev_resid | grad_gate_msa | grad_attn_out]
    GateOp[D].vjp_gpu[BT, dtype](
        ctx,
        grad_x_mid_t,
        sgg_t,
        empty_p,
        gate1_cache_t,
        empty_gp,
        op_ws,
    )
    # Split: residual → grad_x_prev; gate slot → sgrm[2D:3D]; branch_out grad → sgbo
    ctx.enqueue_function[cb_split_gate_grad_kernel[BT, D, 0]](
        sgg_t,
        grad_x_prev_t,
        sgrm_t,
        sgbo_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # MSA.backward — operates on (BATCH, T*D) view.
    var sgbo_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgbo_t.ptr)
    var sgmx_btd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ](sgmx_t.ptr)
    MultiHeadAttentionXL[D, HEADS, DIM_HEAD, T, True].backward_gpu[BATCH, dtype](
        ctx,
        sgmx_btd_t,
        sgbo_btd_t,
        msa_params,
        msa_state,
        msa_cache_t,
        g_msa_params,
        msa_workspace,
    )
    # Modulate1.vjp: sgmx → sgmi
    ModulateOp[D].vjp_gpu[BT, dtype](
        ctx,
        sgmx_t,
        sgmi_t,
        empty_p,
        mod1_cache_t,
        empty_gp,
        op_ws,
    )
    # Split sgmi → sglnout, sgrm[D:2D] (scale_msa), sgrm[0:D] (shift_msa)
    ctx.enqueue_function[cb_split_mod_grad_kernel[BT, D, 0]](
        sgmi_t,
        sglnout_t,
        sgrm_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )
    # LN1.vjp: sglnout → sglnin
    LayerNormNoAffineOp[D].vjp_gpu[BT, dtype](
        ctx,
        sglnout_t,
        sglnin_t,
        empty_p,
        ln1_cache_t,
        empty_gp,
        op_ws,
    )
    # Accumulate sglnin into grad_x_prev (residual already there).
    ctx.enqueue_function[cb_accum_kernel[BT, D]](
        grad_x_prev_t,
        sglnin_t,
        grid_dim=(ceildiv(BT, TPB_X), ceildiv(D, TPB_Y)),
        block_dim=(TPB_X, TPB_Y),
    )

    # ============================== Shared tail =============================
    # AdaLNMod.backward: sgrm (6D, fully populated) → sgsc, accumulating g_adaln_params
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
    # Swish.vjp: sgsc → grad_c
    SwishOp[D].vjp_gpu[BT, dtype](
        ctx,
        sgsc_t,
        grad_c_t,
        empty_p,
        silu_cache_t,
        empty_gp,
        op_ws,
    )
