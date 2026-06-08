"""TimeAttentionLatents[D, N_HEADS, T, S, N_LATENTS] — causal time attention
over the latent tokens of a (T, S, D) grid.

Dreamer 4's block-causal transformer factorizes attention into space layers
(over S tokens per frame) and a time layer every few blocks (causal over the
T frames). The space layers run at effective batch B·T (one frame per sample,
sequence S) via plain `Sequential` leaves. The time layer needs the *other*
grouping — causal attention over T per token — which no per-sample leaf can
express on the B·T layout. This leaf bridges that: it reads the full B·T
batch, regroups the **latent** tokens to (B·L, T) and runs a causal
`MultiHeadAttention` over T, then scatters back.

I/O (per the B·T layout): IN_DIM == OUT_DIM == S·D. Sample `bt = b*T + t`,
token `s`, channel `d` lives at `((b*T+t)*S + s)*D + d`. Only the first
`N_LATENTS` tokens are time-attended; non-latent outputs are **0** so the
enclosing `Residual` leaves them unchanged (the clean variant — the reference
`TimeSelfAttention` instead leaks norm(x) into non-latents via the residual,
which we treat as a bug and do not replicate).

Internally wraps `MultiHeadAttention[D, N_HEADS, T, causal=True]` driven at
batch B·L. Params (qkv/out projections) are owned by that inner module;
`for_each_param` / `zero_grad` delegate to it.

PHASE 1: CPU forward + vjp (gather → inner MHA → scatter). GPU follows.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for
from ..composites import MultiHeadAttention


# Gather full (B·T, S·D) latent tokens → packed (B·L, T·D); same index map is
# reused for the forward input and the backward grad_output.
def _tal_gather_kernel[
    B: Int, T: Int, S: Int, L: Int, D: Int
](
    src: LayoutTensor[DT, Layout.row_major(B * T * S * D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(B * L * T * D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= B * L * T * D:
        return
    var d = idx % D
    var rem = idx // D
    var t = rem % T
    var rem2 = rem // T
    var l = rem2 % L
    var b = rem2 // L
    dst.ptr[idx] = rebind[Scalar[DT]](src.ptr[((b * T + t) * S + l) * D + d])


# Scatter packed (B·L, T·D) → full (B·T, S·D), zeroing non-latent (s≥L)
# positions. Reused for the forward output and the backward grad_input.
def _tal_scatter_kernel[
    B: Int, T: Int, S: Int, L: Int, D: Int
](
    packed: LayoutTensor[DT, Layout.row_major(B * L * T * D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(B * T * S * D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= B * T * S * D:
        return
    var d = idx % D
    var rem = idx // D
    var s = rem % S
    var rem2 = rem // S
    var t = rem2 % T
    var b = rem2 // T
    if s < L:
        dst.ptr[idx] = rebind[Scalar[DT]](
            packed.ptr[(b * L + s) * T * D + t * D + d]
        )
    else:
        dst.ptr[idx] = Scalar[DT](0.0)


struct TimeAttentionLatents[
    D: Int, N_HEADS: Int, T: Int, S: Int, N_LATENTS: Int
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.S * Self.D)
    comptime OUT_DIM = Self.S * Self.D
    comptime MHA = MultiHeadAttention[Self.D, Self.N_HEADS, Self.T, True]

    var mha: Self.MHA
    # S5 Cache role — CPU pack scratch + 2 reused device packed buffers.
    var packed_in: Cache["tal_packed_in"]    # CPU scratch [B*L, T*D]
    var packed_out: Cache["tal_packed_out"]
    var grad_pout: Cache["tal_grad_pout"]
    var grad_pin: Cache["tal_grad_pin"]
    var pa: Cache["tal_pa"]   # device (gather→A, MHA A→B, scatter B)
    var pb: Cache["tal_pb"]
    var ts: TargetStorage

    def __init__(out self):
        self.mha = Self.MHA()
        self.packed_in = Cache["tal_packed_in"]()
        self.packed_out = Cache["tal_packed_out"]()
        self.grad_pout = Cache["tal_grad_pout"]()
        self.grad_pin = Cache["tal_grad_pin"]()
        self.pa = Cache["tal_pa"]()
        self.pb = Cache["tal_pb"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TimeAttentionLatents: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.mha = Self.MHA.make[target=target, INIT=INIT](ctx)
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("TimeAttentionLatents.make[gpu]: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    def _ensure_scratch_gpu(mut self, packed_n: Int) raises:
        var ctx = self.ts.ctx.value()
        self.pa.ensure_gpu(ctx, packed_n)
        self.pb.ensure_gpu(ctx, packed_n)

    @staticmethod
    def display_label() -> String:
        return String("TimeAttentionLatents")

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["TimeAttentionLatents", target](self.ts.target_tag)
        comptime assert BATCH % Self.T == 0, (
            "TimeAttentionLatents: BATCH (=B*T) must be divisible by T"
        )
        comptime B = BATCH // Self.T
        comptime BL = B * Self.N_LATENTS
        comptime TD = Self.T * Self.D
        var inp = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.packed_in.ensure_cpu(BL * TD)
            self.packed_out.ensure_cpu(BL * TD)
            var pin = TileTensor(
                mptr(self.packed_in.cpu_ptr()),
                row_major[BL, TD](),
            )
            # gather latents: packed_in[b,l,t,d] = input[b,t,s=l,d]
            for b in range(B):
                for l in range(Self.N_LATENTS):
                    for t in range(Self.T):
                        for d in range(Self.D):
                            pin[b * Self.N_LATENTS + l, t * Self.D + d] = inp[
                                b * Self.T + t, l * Self.D + d
                            ]
            var pout = TileTensor(
                mptr(self.packed_out.cpu_ptr()),
                row_major[BL, TD](),
            )
            self.mha.forward[target, BL, POLICY=POLICY](pin, output=pout)
            # scatter: out[b,t,s,d] = packed_out[b,s,t,d] if s<L else 0
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            var v = Scalar[DT](0.0)
                            if s < Self.N_LATENTS:
                                v = pout[b * Self.N_LATENTS + s, t * Self.D + d]
                            out[b * Self.T + t, s * Self.D + d] = v
        else:
            comptime PACKED = BL * TD
            comptime FULL = BATCH * Self.S * Self.D
            self._ensure_scratch_gpu(PACKED)
            var ctx = self.ts.ctx.value()
            var a = self.pa.dev.value()
            var b = self.pb.dev.value()
            var in_flat = LayoutTensor[DT, Layout.row_major(FULL), MutAnyOrigin](
                inp.ptr
            )
            var out_flat = LayoutTensor[DT, Layout.row_major(FULL), MutAnyOrigin](
                out.ptr
            )
            var a_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](a)
            var b_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](b)
            comptime gk = _tal_gather_kernel[
                B, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            ctx.enqueue_function[gk](
                in_flat, a_lt, grid_dim=(PACKED + TPB - 1) // TPB, block_dim=TPB
            )
            var a_tile = TileTensor(
                mptr(a.unsafe_ptr()),
                row_major[BL, TD](),
            )
            var b_tile = TileTensor(
                mptr(b.unsafe_ptr()),
                row_major[BL, TD](),
            )
            self.mha.forward[target, BL, POLICY=POLICY](a_tile, output=b_tile)
            comptime sk = _tal_scatter_kernel[
                B, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            ctx.enqueue_function[sk](
                b_lt, out_flat, grid_dim=(FULL + TPB - 1) // TPB, block_dim=TPB
            )

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        assert_tag_for["TimeAttentionLatents", target](self.ts.target_tag)
        comptime B = BATCH // Self.T
        comptime BL = B * Self.N_LATENTS
        comptime TD = Self.T * Self.D
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            self.grad_pout.ensure_cpu(BL * TD)
            self.grad_pin.ensure_cpu(BL * TD)
            var gpout = TileTensor(
                mptr(self.grad_pout.cpu_ptr()),
                row_major[BL, TD](),
            )
            # gather grad_output latents (non-latent outputs were 0 → no grad)
            for b in range(B):
                for l in range(Self.N_LATENTS):
                    for t in range(Self.T):
                        for d in range(Self.D):
                            gpout[b * Self.N_LATENTS + l, t * Self.D + d] = go[
                                b * Self.T + t, l * Self.D + d
                            ]
            var gpin = TileTensor(
                mptr(self.grad_pin.cpu_ptr()),
                row_major[BL, TD](),
            )
            self.mha.vjp[target, BL, POLICY=POLICY, mode=mode](gpout, gpin)
            # scatter grad to latent input positions; non-latents get 0
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            var v = Scalar[DT](0.0)
                            if s < Self.N_LATENTS:
                                v = gpin[b * Self.N_LATENTS + s, t * Self.D + d]
                            gi[b * Self.T + t, s * Self.D + d] = v
        else:
            comptime PACKED = BL * TD
            comptime FULL = BATCH * Self.S * Self.D
            self._ensure_scratch_gpu(PACKED)
            var ctx = self.ts.ctx.value()
            var a = self.pa.dev.value()
            var b = self.pb.dev.value()
            var go_flat = LayoutTensor[DT, Layout.row_major(FULL), MutAnyOrigin](
                go.ptr
            )
            var gi_flat = LayoutTensor[DT, Layout.row_major(FULL), MutAnyOrigin](
                gi.ptr
            )
            var a_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](a)
            var b_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](b)
            comptime gk = _tal_gather_kernel[
                B, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            ctx.enqueue_function[gk](
                go_flat, a_lt, grid_dim=(PACKED + TPB - 1) // TPB, block_dim=TPB
            )
            var a_tile = TileTensor(
                mptr(a.unsafe_ptr()),
                row_major[BL, TD](),
            )
            var b_tile = TileTensor(
                mptr(b.unsafe_ptr()),
                row_major[BL, TD](),
            )
            self.mha.vjp[target, BL, POLICY=POLICY, mode=mode](a_tile, b_tile)
            comptime sk = _tal_scatter_kernel[
                B, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            ctx.enqueue_function[sk](
                b_lt, gi_flat, grid_dim=(FULL + TPB - 1) // TPB, block_dim=TPB
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["TimeAttentionLatents", target](self.ts.target_tag)
        self.mha.for_each_param[target, V](prefix, visitor)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["TimeAttentionLatents", target](self.ts.target_tag)
        self.mha.for_each_state[target, V](prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["TimeAttentionLatents", target](self.ts.target_tag)
        self.mha.zero_grad[target]()
