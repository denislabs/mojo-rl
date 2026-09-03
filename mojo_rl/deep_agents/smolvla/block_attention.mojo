# +--------------------------------------------------------------------------+ #
# | SmolVLA — attention with a static [Q_LEN, KV_LEN] block mask
# +--------------------------------------------------------------------------+ #
"""The one attention shape the denoising step needs and no existing leaf has.

    q [B, Q_LEN*DIM]   k,v [B, KV_LEN*DIM]   mask [Q_LEN, KV_LEN] additive
    out[b,i] = concat_h( softmax_j( q·kᵀ/sqrt(HEAD_DIM) + mask[i,j] ) · v )

`nn` already ships three attention leaves and none fits:

  * `ScaledDotProductAttention` — packed `[Q|K|V]` of ONE sequence.
  * `MaskedAttention` — has the static 2-D mask, but packed and `Q_LEN == KV_LEN`.
    Using it would mean recomputing the prefix's queries, which is what the KV
    cache exists to avoid.
  * `CrossAttention` — separate q/k/v and `Q_LEN != KV_LEN`, but its mask is a
    per-sample KEY PADDING mask `[B, KV_LEN]`. A key is masked for every query or
    for none, so it cannot express causality.

The denoising step needs `Q_LEN != KV_LEN` **and** a query-dependent mask: the 50
action queries attend to the whole cached prefix but only causally among
themselves.

⚠ **Only the SELF (even) layers need this.** A cross (odd) layer's suffix
attends to the entire prefix with nothing masked, so it can use `CrossAttention`
unmasked. Reaching for a mask there would be harmless and pointless; reaching
for `CrossAttention` in the even case would be silently wrong.

## Forward only

V1 is inference. This leaf has no `vjp` and is therefore not a `Module` — it is
driven by hand from `fused.mojo`, like `DecoderLayerWeights`. V2's post-training
needs the backward; the plan already books that, and the omission is loud (there
is no `vjp` to call) rather than a silently wrong one.

⚠ A fully-masked query row would divide by zero. Rows are renormalised by a
floored denominator and produce a zero context vector instead of NaN — the same
guard, and the same constant, as `cross_attention.mojo`.
"""

from std.math import exp, sqrt
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor


comptime BA_MASK_NEG: Scalar[DT] = Scalar[DT](-1.0e30)
comptime BA_DENOM_FLOOR: Scalar[DT] = Scalar[DT](1.0e-30)


def _ba_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
    k: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(QL * KL), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
):
    """One thread per (batch, head, query). Q_LEN is 50 and N_HEADS 15 here, so
    the row-per-thread map is a few hundred threads — small, but the work per
    row is KL*HD and the alternative (a tiled BMM) is not worth its complexity
    until this shows up in a profile."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * QL:
        return
    var i = idx % QL
    var r = idx // QL
    var h = r % N_HEADS
    var b = r // N_HEADS
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var qb = b * (QL * DIM) + i * DIM + h * HD

    # pass 1: max over masked scores
    var mx = BA_MASK_NEG
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        s = s * scale + m
        if s > mx:
            mx = s

    # pass 2: exponentiate, accumulate the context vector
    var denom = Scalar[DT](0)
    for d in range(HD):
        dst.ptr[unsafe_offset = qb + d] = Scalar[DT](0)
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        var w = exp(s * scale + m - mx)
        denom += w
        for d in range(HD):
            dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
                dst.ptr[unsafe_offset = qb + d]
            ) + w * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])

    var inv = Scalar[DT](1.0) / (denom if denom > BA_DENOM_FLOOR else Scalar[DT](1.0))
    if denom <= BA_DENOM_FLOOR:
        inv = Scalar[DT](0)   # fully masked row -> zero context, not NaN
    for d in range(HD):
        dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset = qb + d]
        ) * inv


struct BlockCrossAttention[
    DIM: Int, N_HEADS: Int, Q_LEN: Int, KV_LEN: Int
](Movable):
    comptime HD: Int = Self.DIM // Self.N_HEADS
    comptime QN: Int = Self.Q_LEN * Self.DIM
    comptime KN: Int = Self.KV_LEN * Self.DIM
    comptime MASK_N: Int = Self.Q_LEN * Self.KV_LEN

    var mask: Tensor
    var is_gpu: Bool

    def __init__(out self):
        comptime assert Self.DIM % Self.N_HEADS == 0, (
            "BlockCrossAttention: DIM must be divisible by N_HEADS"
        )
        self.mask = Tensor()
        self.is_gpu = False

    def __init__(out self, *, deinit move: Self):
        self.mask = move.mask^
        self.is_gpu = move.is_gpu

    @staticmethod
    def make[
        target: StaticString
    ](
        ref mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        if len(mask) != Self.MASK_N:
            raise Error(
                "BlockCrossAttention: mask must be Q_LEN*KV_LEN = "
                + String(Self.MASK_N) + ", got " + String(len(mask))
            )
        var a = Self()
        a.mask = Tensor.alloc(Self.MASK_N)
        for i in range(Self.MASK_N):
            a.mask.data[i] = mask[i]
        comptime if target != "cpu":
            a.mask.upload(ctx.value())
            a.is_gpu = True
        return a^

    def forward[
        target: StaticString, B: Int
    ](
        mut self, mut q: Tensor, mut k: Tensor, mut v: Tensor,
        mut out: Tensor, ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            out.ensure(B * Self.QN)
            var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](Self.HD))
            for b in range(B):
                for h in range(Self.N_HEADS):
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var mx = BA_MASK_NEG
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            s = s * scale + m
                            if s > mx:
                                mx = s
                        var denom = Scalar[DT](0)
                        for d in range(Self.HD):
                            out.data[qb + d] = Scalar[DT](0)
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            var w = exp(s * scale + m - mx)
                            denom += w
                            for d in range(Self.HD):
                                out.data[qb + d] += w * v.data[kb + d]
                        var inv = Scalar[DT](0)
                        if denom > BA_DENOM_FLOOR:
                            inv = Scalar[DT](1.0) / denom
                        for d in range(Self.HD):
                            out.data[qb + d] *= inv
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.QN)
            comptime rows = B * Self.N_HEADS * Self.Q_LEN
            comptime n_blocks = (rows + TPB - 1) // TPB
            c.enqueue_function[
                _ba_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.lt["gpu", Layout.row_major(B, Self.QN)](),
                k.lt["gpu", Layout.row_major(B, Self.KN)](),
                v.lt["gpu", Layout.row_major(B, Self.KN)](),
                self.mask.lt["gpu", Layout.row_major(Self.MASK_N)](),
                out.lt["gpu", Layout.row_major(B, Self.QN)](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
