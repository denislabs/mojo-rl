"""MaskedAttention[DIM, N_HEADS, SEQ_LEN] — attention with a static additive
mask (Dreamer 4 modality-gated / block-causal attention).

PHASE 0 SPIKE (docs/DREAMER4_PORT_PLAN.md §4). CPU forward + 3-pass vjp only;
GPU paths are stubbed (raise) until the spike's go/no-go is taken.

Same qkv-major input layout and output-caching cache layout `[Q | K | V |
scores]` as `ScaledDotProductAttention`, so it is EXEMPT from the
param-grad-before-grad_input aliasing invariant (no params; backward reads
only `self.cache` + `grad_output`). The ONLY behavioural difference vs the
causal SDPA op is:

  1. A per-(i,j) additive bias `mask[i*SEQ+j] ∈ {0.0, NEG}` is added to the
     score before the softmax max-shift. A masked entry gets score ≈ NEG, so
     its softmax weight is identically 0.
  2. The key loop runs the full `j ∈ [0, SEQ)` range (no causal `j ≤ i`
     loop-bound shortcut) — the mask alone decides visibility.

This is the crux of the Dreamer 4 feasibility claim (§4.3): because a masked
attention weight is exactly 0, it is a fixed point of the softmax gradient
(`dV`, softmax-jvp, `dQ`, `dK` all vanish there), so the BACKWARD math is
identical to plain attention — only the loop range changes. The mask needs
zero backward special-casing.

The default mask built by `make` is all-zero (all-allow) → bit-identical to
`ScaledDotProductAttention[..., CAUSAL=False]`. Install a real mask with
`set_mask`. Causal SDPA is recoverable via `causal_mask` (so this op
subsumes the causal one; the causal op is kept for its loop-bound speed).
"""

from std.math import exp, sqrt
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.memory import AddressSpace
from linalg.bmm import batched_matmul

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for, ensure_cpu_buffer

# The BMM fast path reuses attention.mojo's pack/unpack/transpose/jvp kernels
# verbatim (they are mask-agnostic); only the softmax differs (adds the mask),
# so MaskedAttention defines its own `_masked_softmax_kernel` below.
from .attention import (
    _attn_pack_qkv_fwd_kernel,
    _attn_unpack_out_kernel,
    _attn_pack_in_bwd_kernel,
    _attn_softmax_jvp_kernel,
    _attn_transpose_from_cache_kernel,
    _attn_transpose_scores_kernel,
    _attn_unpack_grad_kernel,
)


comptime MASK_NEG: Float64 = -1.0e30


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — custom per-(b,h) path, mirroring attention.mojo's custom
# kernels with two differences: (1) the forward adds the additive mask to
# each score; (2) all j/i loops run the full [0, SEQ) range (no causal
# loop-bound). The backward kernels are otherwise byte-for-byte the
# CAUSAL=False attention kernels — masked weights are 0 in cache, so they
# contribute 0 to every gradient (plan §4.3). One block per (b,h); the
# shared mask buffer is [SEQ*SEQ] (batch- and head-independent).
# ──────────────────────────────────────────────────────────────────────


def _masked_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(SEQ * SEQ), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    # Step 1: cache this head's Q/K/V slice (for backward).
    var n_qkv = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_qkv:
        var i = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        cache.ptr[b * CACHE_SIZE + i * DIM + h_off + d] = rebind[Scalar[DT]](
            input.ptr[b * IN_DIM + i * DIM + h_off + d]
        )
        cache.ptr[b * CACHE_SIZE + K_OFF + i * DIM + h_off + d] = rebind[
            Scalar[DT]
        ](input.ptr[b * IN_DIM + K_OFF + i * DIM + h_off + d])
        cache.ptr[b * CACHE_SIZE + V_OFF + i * DIM + h_off + d] = rebind[
            Scalar[DT]
        ](input.ptr[b * IN_DIM + V_OFF + i * DIM + h_off + d])
        idx0 += bs

    # Step 2: per-row masked attention; threads stride over query rows i.
    var i = tid
    while i < SEQ:
        var max_score = Scalar[DT](-1e30)
        for j in range(SEQ):
            var s = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var q = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + i * DIM + h_off + d]
                )
                var k = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + K_OFF + j * DIM + h_off + d]
                )
                s += q * k
            s = s * scale + rebind[Scalar[DT]](mask.ptr[i * SEQ + j])
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[aidx] = s
            if s > max_score:
                max_score = s

        var sum_exp = Scalar[DT](0)
        for j in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var e = exp(rebind[Scalar[DT]](cache.ptr[aidx]) - max_score)
            cache.ptr[aidx] = e
            sum_exp += e

        var inv_sum = Scalar[DT](1) / sum_exp
        for j in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[aidx] = rebind[Scalar[DT]](cache.ptr[aidx]) * inv_sum

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(SEQ):
                var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                var v = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + V_OFF + j * DIM + h_off + d]
                )
                acc += rebind[Scalar[DT]](cache.ptr[aidx]) * v
            output.ptr[b * OUT_DIM + i * DIM + h_off + d] = acc
        i += bs


def _masked_zero_grad_kernel[
    BATCH: Int, IN_DIM: Int
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * IN_DIM:
        grad_input.ptr[idx] = Scalar[DT](0)


def _masked_dV_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int, V_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dV[j] = Σ_i attn[i,j] * grad_out[i]. Full i-range (mask gates via attn=0).
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var acc = Scalar[DT](0)
        for i in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var go = rebind[Scalar[DT]](
                grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
            )
            acc += rebind[Scalar[DT]](cache.ptr[aidx]) * go
        var dv_idx = b * IN_DIM + V_OFF + j * DIM + h_off + d
        grad_input.ptr[dv_idx] = grad_input.ptr[dv_idx] + acc
        idx0 += bs


def _masked_dscore_dQ_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # Per row i: dot_sum, then d_score (overwrites cache.attn), then dQ.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    var i = tid
    while i < SEQ:
        var dot_sum = Scalar[DT](0)
        for j in range(SEQ):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[DT]](
                    grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
                )
                var v = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            dot_sum += rebind[Scalar[DT]](cache.ptr[aidx]) * d_attn

        for j in range(SEQ):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[DT]](
                    grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
                )
                var v = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var attn_w = rebind[Scalar[DT]](cache.ptr[aidx])
            cache.ptr[aidx] = attn_w * (d_attn - dot_sum) * scale

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(SEQ):
                var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                var d_score = rebind[Scalar[DT]](cache.ptr[aidx])
                var k = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + K_OFF + j * DIM + h_off + d]
                )
                acc += d_score * k
            var dq_idx = b * IN_DIM + i * DIM + h_off + d
            grad_input.ptr[dq_idx] = grad_input.ptr[dq_idx] + acc
        i += bs


def _masked_dK_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, CACHE_SIZE: Int, K_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dK[j] = Σ_i d_score[i,j] * Q[i]. Reads d_score from cache.attn. Full range.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var acc = Scalar[DT](0)
        for i in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var d_score = rebind[Scalar[DT]](cache.ptr[aidx])
            var q = rebind[Scalar[DT]](cache.ptr[b * CACHE_SIZE + i * DIM + h_off + d])
            acc += d_score * q
        var dk_idx = b * IN_DIM + K_OFF + j * DIM + h_off + d
        grad_input.ptr[dk_idx] = grad_input.ptr[dk_idx] + acc
        idx0 += bs


# BMM fast path softmax — attention.mojo's `_attn_softmax_kernel` with the
# causal zero-fill replaced by an additive mask before the max-shift. Full
# range; masked entries get score ≈ NEG → softmax weight 0. 1 block per (b,h).
def _masked_softmax_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(SEQ * SEQ), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var bh_off = blk * SEQ * SEQ
    var cache_attn_base = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ
    var i = tid
    while i < SEQ:
        var row_off = bh_off + i * SEQ
        var cache_row = cache_attn_base + i * SEQ
        var mx = Scalar[DT](-1e30)
        for j in range(SEQ):
            var s = rebind[Scalar[DT]](scores.ptr[row_off + j]) * scale + rebind[
                Scalar[DT]
            ](mask.ptr[i * SEQ + j])
            scores.ptr[row_off + j] = s
            if s > mx:
                mx = s
        var se = Scalar[DT](0)
        for j in range(SEQ):
            var e = exp(rebind[Scalar[DT]](scores.ptr[row_off + j]) - mx)
            scores.ptr[row_off + j] = e
            se += e
        var inv = Scalar[DT](1) / se
        for j in range(SEQ):
            var w = rebind[Scalar[DT]](scores.ptr[row_off + j]) * inv
            scores.ptr[row_off + j] = w
            cache.ptr[cache_row + j] = w
        i += bs


# ──────────────────────────────────────────────────────────────────────
# Host mask builders (data-independent; depend only on token layout).
# ──────────────────────────────────────────────────────────────────────


def causal_mask(seq_len: Int) -> List[Scalar[DT]]:
    """Additive lower-triangular mask: position i may attend to j ≤ i."""
    var m = List[Scalar[DT]]()
    for i in range(seq_len):
        for j in range(seq_len):
            m.append(Scalar[DT](0.0) if j <= i else Scalar[DT](MASK_NEG))
    return m^


def all_allow_mask(seq_len: Int) -> List[Scalar[DT]]:
    """Additive all-zero mask → unrestricted (bidirectional) attention."""
    var m = List[Scalar[DT]]()
    for _ in range(seq_len * seq_len):
        m.append(Scalar[DT](0.0))
    return m^


def build_modality_mask[
    mode: StaticString
](
    modality_ids: List[Int], n_latents: Int, agent_mod_in: Int = -1
) -> List[Scalar[DT]]:
    """Port of `model.py:SpaceSelfAttentionModality._build_allow` (Dreamer 4).

    `modality_ids[k]` is the modality of token k; the first `n_latents` tokens
    are latent register tokens. Returns the additive `SEQ*SEQ` mask:
    allowed→0.0, disallowed→NEG.

    Modes (the `wm_*` modes treat the highest modality id as the AGENT modality
    and are only meaningful when agent tokens are present; they ignore
    `n_latents`):
      - "encoder": latents attend to all; non-latents only within own modality.
      - "decoder": latents attend only to latents; non-latents attend to
        latents + own modality.
      - "wm_agent": full mixing — every token attends to every token
        (model.py `wm_agent` returns all-ones).
      - "wm_agent_isolated": non-agent queries attend to all NON-agent keys;
        agent queries attend ONLY to agent keys (model.py `wm_agent_isolated`,
        which keeps agent tokens inert during world-model pretraining).
      - "wm_agent_bc": non-agent queries attend to all NON-agent keys; agent
        queries attend to ALL keys (every modality + themselves). This is the
        paper §3.3 imagination/BC mask: agent tokens read the full world state
        to predict actions/rewards, while nothing attends back to them so the
        world model cannot be contaminated by the task. (No reference code —
        the public reference implements pretraining only; ported from the
        paper.)
    """
    comptime assert (
        mode == "encoder"
        or mode == "decoder"
        or mode == "wm_agent"
        or mode == "wm_agent_isolated"
        or mode == "wm_agent_bc"
    ), "build_modality_mask: unknown mode"

    var S = len(modality_ids)
    # Agent modality. With `agent_mod_in >= 0` it is FIXED to that id (so a
    # layout with zero agent tokens — no token carries the id — yields full
    # mixing under the wm_* modes, since `k_is_agent`/`q_is_agent` are always
    # False). With the default -1 it is inferred as the max id present (the
    # agent tokens are the last modality inserted) — back-compat for callers
    # that always have agent tokens.
    var agent_mod = agent_mod_in
    if agent_mod < 0:
        agent_mod = 0
        for k in range(S):
            if modality_ids[k] > agent_mod:
                agent_mod = modality_ids[k]

    var m = List[Scalar[DT]]()
    for i in range(S):
        var is_q_lat = i < n_latents
        var q_is_agent = modality_ids[i] == agent_mod
        for j in range(S):
            var is_k_lat = j < n_latents
            var k_is_agent = modality_ids[j] == agent_mod
            var same_mod = modality_ids[i] == modality_ids[j]

            var allow: Bool
            comptime if mode == "encoder":
                # latents → all; non-latents → same modality.
                allow = True if is_q_lat else same_mod
            elif mode == "decoder":
                # latents → latents; non-latents → latents + same modality.
                if is_q_lat:
                    allow = is_k_lat
                else:
                    allow = is_k_lat or same_mod
            elif mode == "wm_agent":
                # Full mixing (model.py: returns all-ones).
                allow = True
            elif mode == "wm_agent_isolated":
                # Non-agent q → all non-agent keys; agent q → only agent keys
                # (inert agent tokens during world-model pretraining).
                if q_is_agent:
                    allow = k_is_agent
                else:
                    allow = not k_is_agent
            else:
                # "wm_agent_bc" — paper §3.3: agent q → all keys; non-agent q →
                # all non-agent keys (nothing attends back to agent tokens).
                if q_is_agent:
                    allow = True
                else:
                    allow = not k_is_agent

            m.append(Scalar[DT](0.0) if allow else Scalar[DT](MASK_NEG))
    return m^


# ──────────────────────────────────────────────────────────────────────
struct MaskedAttention[
    DIM: Int,
    N_HEADS: Int,
    SEQ_LEN: Int,
    USE_MAX_KERNELS: Bool = True,
](Module):
    comptime ARITY: Int = 1
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_LEN * Self.DIM * 3)
    comptime OUT_DIM = Self.SEQ_LEN * Self.DIM
    comptime K_OFF: Int = Self.SEQ_LEN * Self.DIM
    comptime V_OFF: Int = 2 * Self.SEQ_LEN * Self.DIM
    comptime ATTN_OFF: Int = 3 * Self.SEQ_LEN * Self.DIM
    comptime CACHE_SIZE: Int = (
        3 * Self.SEQ_LEN * Self.DIM
        + Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
    )
    # Per-sample bmm scratch: 4 packed slabs (SEQ*DIM) + 2 scores slabs
    # (N_HEADS*SEQ*SEQ). One reused device buffer, lazily sized to BATCH.
    comptime SCRATCH_UNIT: Int = (
        4 * Self.SEQ_LEN * Self.DIM
        + 2 * Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
    )

    var cache: List[Scalar[DT]]          # [BATCH, CACHE_SIZE]
    var mask: List[Scalar[DT]]           # [SEQ_LEN*SEQ_LEN] additive bias (CPU)
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var mask_dev: Optional[DeviceBuffer[DT]]   # [SEQ_LEN*SEQ_LEN] (GPU)
    var scratch_dev: Optional[DeviceBuffer[DT]]
    var scratch_n_batch: Int
    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.DIM % Self.N_HEADS == 0
        ), "MaskedAttention: DIM must be divisible by N_HEADS"
        self.cache = List[Scalar[DT]]()
        self.mask = List[Scalar[DT]]()
        self.cache_dev = None
        self.cache_n_batch = 0
        self.mask_dev = None
        self.scratch_dev = None
        self.scratch_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MaskedAttention: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        # Default = all-allow (bidirectional), equals non-causal SDPA.
        a.mask = all_allow_mask(Self.SEQ_LEN)
        comptime if target == "cpu":
            a.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["MaskedAttention.make[target='gpu']"](ctx)
            a.cache_dev = ctx_v.enqueue_create_buffer[DT](1)
            a.cache_n_batch = 0
            a.ts = TargetStorage.make_gpu(ctx_v)
            a._upload_mask_gpu()
        return a^

    def _upload_mask_gpu(mut self) raises:
        var ctx = self.ts.ctx.value()
        comptime N = Self.SEQ_LEN * Self.SEQ_LEN
        var dev = ctx.enqueue_create_buffer[DT](N)
        var host = ctx.enqueue_create_host_buffer[DT](N)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        for i in range(N):
            hp[i] = self.mask[i]
        ctx.enqueue_copy(dev, host)
        self.mask_dev = dev^

    def set_mask(mut self, var mask: List[Scalar[DT]]) raises:
        if len(mask) != Self.SEQ_LEN * Self.SEQ_LEN:
            raise Error("MaskedAttention.set_mask: expected SEQ_LEN*SEQ_LEN")
        self.mask = mask^
        if self.ts.ctx:
            self._upload_mask_gpu()

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.CACHE_SIZE
            )
            self.cache_n_batch = batch

    def _ensure_scratch_gpu(mut self, batch: Int) raises:
        if self.scratch_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.scratch_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.SCRATCH_UNIT
            )
            self.scratch_n_batch = batch

    @staticmethod
    def display_label() -> String:
        return String("MaskedAttention")

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["MaskedAttention", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime if target == "cpu":
            self._forward_cpu[BATCH](input, output_v)
        else:
            self._ensure_cache_gpu(BATCH)
            comptime if Self.USE_MAX_KERNELS:
                self._forward_gpu_bmm[BATCH](input, output_v)
            else:
                self._forward_gpu_custom[BATCH](input, output_v)

    def _forward_cpu[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        ensure_cpu_buffer(self.cache, BATCH * Self.CACHE_SIZE)
        var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        var cp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.cache.unsafe_ptr()
        )
        var mp = self.mask.unsafe_ptr()
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SD = Self.SEQ_LEN * Self.DIM
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for b in range(BATCH):
            for i in range(SD):
                cp[b * C + i] = ip[b * IN + i]
                cp[b * C + Self.K_OFF + i] = ip[b * IN + Self.K_OFF + i]
                cp[b * C + Self.V_OFF + i] = ip[b * IN + Self.V_OFF + i]

            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM
                for i in range(Self.SEQ_LEN):
                    # NOTE: full j-range — the mask, not a loop bound, gates.
                    var max_score: Float64 = -1e30
                    for j in range(Self.SEQ_LEN):
                        var score: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(ip[b * IN + i * Self.DIM + h_off + d])
                            var k = Float64(
                                ip[b * IN + Self.K_OFF + j * Self.DIM + h_off + d]
                            )
                            score += q * k
                        # scale, then additive mask bias.
                        score = score * scale + Float64(mp[i * Self.SEQ_LEN + j])
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](score)
                        if score > max_score:
                            max_score = score

                    var sum_exp: Float64 = 0.0
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var e = exp(Float64(cp[ai]) - max_score)
                        cp[ai] = Scalar[DT](e)
                        sum_exp += e

                    var inv = 1.0 / sum_exp
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](Float64(cp[ai]) * inv)

                    for d in range(Self.HEAD_DIM):
                        var acc: Float64 = 0.0
                        for j in range(Self.SEQ_LEN):
                            var ai = (
                                b * C + Self.ATTN_OFF
                                + h * Self.SEQ_LEN * Self.SEQ_LEN
                                + i * Self.SEQ_LEN + j
                            )
                            var v = Float64(
                                ip[b * IN + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            acc += Float64(cp[ai]) * v
                        op[b * OUT + i * Self.DIM + h_off + d] = Scalar[DT](acc)

    # ----- Backward --------------------------------------------------------
    # Identical math to ScaledDotProductAttention._vjp_cpu; only the j-loops
    # run the full [0, SEQ) range. Masked positions carry attn weight 0 in
    # `cache`, so they contribute 0 to every gradient — no special-casing.

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["MaskedAttention", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        comptime if target == "cpu":
            self._vjp_cpu[BATCH](grad_output_v, grad_input_v)
        else:
            comptime if Self.USE_MAX_KERNELS:
                self._vjp_gpu_bmm[BATCH](grad_output_v, grad_input_v)
            else:
                self._vjp_gpu_custom[BATCH](grad_output_v, grad_input_v)

    def _forward_gpu_custom[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime lay_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(BATCH, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(BATCH, Self.CACHE_SIZE)
        comptime lay_m = Layout.row_major(Self.SEQ_LEN * Self.SEQ_LEN)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        var in_lt = LayoutTensor[DT, lay_in, MutAnyOrigin](in_p)
        var out_lt = LayoutTensor[DT, lay_out, MutAnyOrigin](out_p)
        var c_lt = LayoutTensor[DT, lay_c, MutAnyOrigin](self.cache_dev.value())
        var m_lt = LayoutTensor[DT, lay_m, MutAnyOrigin](self.mask_dev.value())
        comptime kernel = _masked_fwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        self.ts.ctx.value().enqueue_function[kernel](
            out_lt, in_lt, c_lt, m_lt,
            grid_dim=BATCH * Self.N_HEADS, block_dim=TPB,
        )

    def _vjp_gpu_custom[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var ctx = self.ts.ctx.value()
        comptime lay_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(BATCH, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(BATCH, Self.CACHE_SIZE)
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
        var go_lt = LayoutTensor[DT, lay_out, MutAnyOrigin](go_p)
        var gi_lt = LayoutTensor[DT, lay_in, MutAnyOrigin](gi_p)
        var c_lt = LayoutTensor[DT, lay_c, MutAnyOrigin](self.cache_dev.value())
        comptime grid_bh = BATCH * Self.N_HEADS

        # 1) zero grad_input.
        comptime zk = _masked_zero_grad_kernel[BATCH, Self.IN_DIMS[0]]
        comptime zn = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
        ctx.enqueue_function[zk](gi_lt, grid_dim=zn, block_dim=TPB)
        # 2) dV (reads attn weights — must precede dscore_dQ overwrite).
        comptime dvk = _masked_dV_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.V_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dvk](gi_lt, go_lt, c_lt, grid_dim=grid_bh, block_dim=TPB)
        # 3) dscore + dQ (overwrites cache.attn with d_score).
        comptime dqk = _masked_dscore_dQ_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dqk](gi_lt, go_lt, c_lt, grid_dim=grid_bh, block_dim=TPB)
        # 4) dK (reads d_score from cache.attn).
        comptime dkk = _masked_dK_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.CACHE_SIZE, Self.K_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dkk](gi_lt, c_lt, grid_dim=grid_bh, block_dim=TPB)

    # ----- GPU BMM fast path (batched-GEMM; reuses attention.mojo glue) -----

    def _forward_gpu_bmm[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime BH = BATCH * Self.N_HEADS
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        var ctx = self.ts.ctx.value()
        self._ensure_scratch_gpu(BATCH)

        var sb = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.scratch_dev.value().unsafe_ptr()
        )
        var pq = sb + 0 * PACKED
        var pk = sb + 1 * PACKED
        var pv = sb + 2 * PACKED
        var pout = sb + 3 * PACKED
        var sc = sb + 4 * PACKED

        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        var in_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.IN_DIMS[0]), MutAnyOrigin
        ](in_p)
        var c_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ](self.cache_dev.value())
        var m_lt = LayoutTensor[
            DT, Layout.row_major(Self.SEQ_LEN * Self.SEQ_LEN), MutAnyOrigin
        ](self.mask_dev.value())

        # 1. pack QKV → (BH, SEQ, HEAD_DIM) + write cache.
        comptime pelems = BATCH * Self.SEQ_LEN * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        var pq_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pq)
        var pk_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pk)
        var pv_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pv)
        comptime pack_k = _attn_pack_qkv_fwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.CACHE_SIZE, PACKED,
        ]
        ctx.enqueue_function[pack_k](
            pq_lt, pk_lt, pv_lt, c_lt, in_lt, grid_dim=pblocks, block_dim=TPB
        )

        # 2. scores = Q @ Kᵀ.
        var scores_tt = TileTensor(
            sc, row_major[BH, Self.SEQ_LEN, Self.SEQ_LEN]()
        )
        var pq_tt = TileTensor(pq, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        var pk_tt = TileTensor(pk, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[transpose_b=True, target="gpu"](
            scores_tt, pq_tt, pk_tt, context=ctx
        )

        # 3. masked softmax in-place; mirror weights into cache.attn.
        var sc_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](sc)
        comptime sm_k = _masked_softmax_kernel[
            BATCH, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[sm_k](sc_lt, c_lt, m_lt, grid_dim=BH, block_dim=TPB)

        # 4. packed_out = attn @ V.
        var pout_tt = TileTensor(
            pout, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        var pv_tt = TileTensor(pv, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[target="gpu"](pout_tt, scores_tt, pv_tt, context=ctx)

        # 5. unpack → output.
        var out_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](out_p)
        var pout_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](
            pout
        )
        comptime up_k = _attn_unpack_out_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.OUT_DIM, PACKED,
        ]
        ctx.enqueue_function[up_k](
            out_lt, pout_lt, grid_dim=pblocks, block_dim=TPB
        )

    def _vjp_gpu_bmm[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime BH = BATCH * Self.N_HEADS
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        var ctx = self.ts.ctx.value()
        self._ensure_scratch_gpu(BATCH)

        # Same slot-recycling aliasing map as attention.mojo's _vjp_gpu_bmm.
        var sb = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.scratch_dev.value().unsafe_ptr()
        )
        var p0 = sb + 0 * PACKED
        var p1 = sb + 1 * PACKED
        var p2 = sb + 2 * PACKED
        var p3 = sb + 3 * PACKED
        var s0 = sb + 4 * PACKED
        var s1 = sb + 4 * PACKED + SCORES

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
        var go_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](go_p)
        var gi_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.IN_DIMS[0]), MutAnyOrigin
        ](gi_p)
        var c_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ](self.cache_dev.value())

        comptime pelems = BATCH * SL * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime sblocks = (SCORES + TPB - 1) // TPB

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        var pdout_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p0)
        var pq_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p1)
        var pk_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p2)
        var pv_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p3)
        comptime pin_k = _attn_pack_in_bwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, SL, HD,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE, PACKED,
        ]
        ctx.enqueue_function[pin_k](
            pdout_lt, pq_lt, pk_lt, pv_lt, go_lt, c_lt,
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. dattn(s0) = dout @ Vᵀ.
        var pdout_tt = TileTensor(p0, row_major[BH, SL, HD]())
        var pq_tt = TileTensor(p1, row_major[BH, SL, HD]())
        var pk_tt = TileTensor(p2, row_major[BH, SL, HD]())
        var pv_tt = TileTensor(p3, row_major[BH, SL, HD]())
        var dattn_tt = TileTensor(s0, row_major[BH, SL, SL]())
        batched_matmul[transpose_b=True, target="gpu"](
            dattn_tt, pdout_tt, pv_tt, context=ctx
        )

        # 3. softmax jvp → dscore(s1). Masked weights are 0 in cache.attn, so
        #    dscore is 0 there automatically (same kernel as attention).
        var dattn_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](s0)
        var dscore_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](s1)
        comptime jvp_k = _attn_softmax_jvp_kernel[
            BATCH, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[jvp_k](
            dscore_lt, dattn_lt, c_lt, grid_dim=BH, block_dim=TPB
        )

        # 4. attn_T(s0) = transpose(cache.attn).
        var attnT_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](s0)
        comptime tac_k = _attn_transpose_from_cache_kernel[
            BATCH, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[tac_k](
            attnT_lt, c_lt, grid_dim=sblocks, block_dim=TPB
        )

        # 5. dV(p3) = attn_T(s0) @ dout(p0).
        var attnT_tt = TileTensor(s0, row_major[BH, SL, SL]())
        var dV_tt = TileTensor(p3, row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dV_tt, attnT_tt, pdout_tt, context=ctx)

        # 6. dscore_T(s0) = transpose(dscore(s1)).
        comptime ts_k = _attn_transpose_scores_kernel[SL, SCORES, BH]
        ctx.enqueue_function[ts_k](
            attnT_lt, dscore_lt, grid_dim=sblocks, block_dim=TPB
        )

        # 7. dK(p0) = dscore_T(s0) @ Q(p1).
        var dscoreT_tt = TileTensor(s0, row_major[BH, SL, SL]())
        var dK_tt = TileTensor(p0, row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dK_tt, dscoreT_tt, pq_tt, context=ctx)

        # 8. dQ(p1) = dscore(s1) @ K(p2).
        var dscore_tt = TileTensor(s1, row_major[BH, SL, SL]())
        var dQ_tt = TileTensor(p1, row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dQ_tt, dscore_tt, pk_tt, context=ctx)

        # 9. unpack dQ(p1)/dK(p0)/dV(p3) → grad_input.
        var dQ_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p1)
        var dK_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p0)
        var dV_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](p3)
        comptime ug_k = _attn_unpack_grad_kernel[
            BATCH, Self.DIM, Self.N_HEADS, SL, HD, Self.IN_DIMS[0], PACKED,
        ]
        ctx.enqueue_function[ug_k](
            gi_lt, dQ_lt, dK_lt, dV_lt, grid_dim=pblocks, block_dim=TPB
        )

    def _vjp_cpu[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )
        var cp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.cache.unsafe_ptr()
        )
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for i in range(BATCH * IN):
            gip[i] = 0.0

        for b in range(BATCH):
            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM

                # Step 1: dV[j] += attn[i,j] * grad_out[i].
                for i in range(Self.SEQ_LEN):
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var dv_idx = (
                                b * IN + Self.V_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dv_idx] = gip[dv_idx] + Scalar[DT](attn_w * go)

                # Step 2: softmax backward → dQ, dK.
                for i in range(Self.SEQ_LEN):
                    var dot_sum: Float64 = 0.0
                    for j in range(Self.SEQ_LEN):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        dot_sum += d_attn * Float64(cp[ai])

                    for j in range(Self.SEQ_LEN):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        var d_score = attn_w * (d_attn - dot_sum) * scale
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(cp[b * C + i * Self.DIM + h_off + d])
                            var k = Float64(
                                cp[b * C + Self.K_OFF + j * Self.DIM + h_off + d]
                            )
                            var dq_idx = b * IN + i * Self.DIM + h_off + d
                            gip[dq_idx] = gip[dq_idx] + Scalar[DT](d_score * k)
                            var dk_idx = (
                                b * IN + Self.K_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dk_idx] = gip[dk_idx] + Scalar[DT](d_score * q)
