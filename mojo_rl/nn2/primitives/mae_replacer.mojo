"""MAEReplacer[NP, D, P_MIN, P_MAX, SEED] — masked-autoencoding patch dropout.

Dreamer 4 trains the tokenizer with masked autoencoding: a random fraction of
the projected patch tokens is replaced by a learned `mask_token`, and the
reconstruction loss is applied only on the replaced patches
(`model.py:MAEReplacer` + `recon_loss_from_mae`). This leaf does the
replacement and remembers which patches were dropped (for the loss + backward).

Operates per frame at nn2-BATCH = B·T: IN_DIM == OUT_DIM == NP·D.

    p_bt   ~ U(p_min, p_max)                 (one drop-rate per frame)
    keep   = U(0,1) < (1 - p_bt)             (per patch)
    out    = where(keep, input, mask_token)

`mask_token` (D) is the only parameter; its grad accumulates the grad-output
of every dropped patch (batch-reduced). Kept patches pass the gradient
straight through; dropped patches get grad_input 0.

RNG: PhiloxRandom seeded by comptime SEED at offset `base + idx`, base =
`rng_step * STRIDE`. The step is bumped by `advance_rng()` (once per training
iter), NOT per forward — so a gradcheck that never advances sees a frozen mask.
The keep decision is computed in Float32 on BOTH CPU and GPU (Metal has no
Float64), so the masks are bit-identical across targets. The backward
RECOMPUTES the keep decision (no need to read the stored mask).

`mae_mask_ptr()` returns the per-patch `keep` flags (1.0 kept / 0.0 dropped).
CPU + GPU.
"""

from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer, AMPPolicy, NoAMP, Param, Cache, ParamVisitor,
    for_each_param_auto, zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


comptime MAE_RTPB = 64


@always_inline
def _kept(
    seed: UInt64, base: UInt64, batch: Int, np: Int, bt: Int, i: Int,
    pmin: Scalar[DT], span: Scalar[DT],
) -> Bool:
    """Per-patch keep decision (Float32, identical on CPU & GPU)."""
    var rp = PhiloxRandom(seed=seed, offset=base + UInt64(bt))
    var p_bt = pmin + span * Scalar[DT](rp.step_uniform()[0])
    var keep_prob = Scalar[DT](1.0) - p_bt
    var ri = PhiloxRandom(
        seed=seed, offset=base + UInt64(batch) + UInt64(bt * np + i)
    )
    var u = Scalar[DT](ri.step_uniform()[0])
    return u < keep_prob


def _mae_fwd_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, NP * D), MutAnyOrigin],
    mask_token: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, NP * D), MutAnyOrigin],
    keep: LayoutTensor[DT, Layout.row_major(BATCH * NP), MutAnyOrigin],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NP * D:
        return
    var d = idx % D
    var rem = idx // D
    var i = rem % NP
    var bt = rem // NP
    var kept = _kept(SEED, base, BATCH, NP, bt, i, pmin, span)
    if kept:
        output.ptr[idx] = rebind[Scalar[DT]](input.ptr[idx])
    else:
        output.ptr[idx] = rebind[Scalar[DT]](mask_token.ptr[d])
    if d == 0:
        keep.ptr[bt * NP + i] = Scalar[DT](1.0) if kept else Scalar[DT](0.0)


def _mae_grad_input_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, NP * D), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, NP * D), MutAnyOrigin],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NP * D:
        return
    var rem = idx // D
    var i = rem % NP
    var bt = rem // NP
    if _kept(SEED, base, BATCH, NP, bt, i, pmin, span):
        grad_input.ptr[idx] = rebind[Scalar[DT]](grad_output.ptr[idx])
    else:
        grad_input.ptr[idx] = Scalar[DT](0.0)


def _mae_grad_token_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, NP * D), MutAnyOrigin],
    grad_token: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    # One block per channel d; threads reduce over dropped (bt, i) patches.
    var d = Int(block_idx.x)
    if d >= D:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var p = t
    while p < BATCH * NP:
        var bt = p // NP
        var i = p % NP
        if not _kept(SEED, base, BATCH, NP, bt, i, pmin, span):
            acc += rebind[Scalar[DT]](grad_output.ptr[(bt * NP + i) * D + d])
        p += MAE_RTPB
    var total = block.sum[block_size=MAE_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_token.ptr[d] = rebind[Scalar[DT]](grad_token.ptr[d]) + total[0]


struct MAEReplacer[
    NP: Int, D: Int, P_MIN: Float64, P_MAX: Float64, SEED: UInt64
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NP * Self.D)
    comptime OUT_DIM = Self.NP * Self.D

    var mask_token: Param["mask_token", False, Self.D]
    var keep: Cache["keep"]
    var rng_step: UInt64
    var p_min_rt: Float64
    var p_max_rt: Float64
    var ts: TargetStorage

    def __init__(out self):
        self.mask_token = Param["mask_token", False, Self.D]()
        self.keep = Cache["keep"]()
        self.rng_step = 0
        self.p_min_rt = Self.P_MIN
        self.p_max_rt = Self.P_MAX
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MAEReplacer: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.mask_token = Param["mask_token", False, Self.D].make_cpu()
            INIT.init_bias(m.mask_token.value_unsafe_ptr_cpu(), Self.D)
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["MAEReplacer.make[gpu]"](ctx)
            m.mask_token = Param["mask_token", False, Self.D].make_gpu(ctx_v)
            var host = ctx_v.enqueue_create_host_buffer[DT](Self.D)
            ctx_v.synchronize()
            INIT.init_bias(host.unsafe_ptr(), Self.D)
            ctx_v.enqueue_copy(m.mask_token.val.dev.value(), host)
            ctx_v.synchronize()
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("MAEReplacer")

    def set_p(mut self, p_min: Float64, p_max: Float64):
        self.p_min_rt = p_min
        self.p_max_rt = p_max

    def advance_rng(mut self):
        self.rng_step += 1

    def mae_mask_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if self.keep.dev:
            return mptr(self.keep.dev.value().unsafe_ptr())
        return mptr(self.keep.cpu_ptr())

    def _ensure_keep_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.keep.ensure_gpu(ctx, batch * Self.NP)
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
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        var inp = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime if target == "cpu":
            self.keep.ensure_cpu(BATCH * Self.NP)
            var kp = self.keep.cpu_ptr()
            var mt = self.mask_token.value_unsafe_ptr_cpu()
            comptime STRIDE = UInt64(BATCH * (1 + Self.NP))
            var base = self.rng_step * STRIDE
            var pmin = Scalar[DT](self.p_min_rt)
            var span = Scalar[DT](self.p_max_rt - self.p_min_rt)
            for bt in range(BATCH):
                for i in range(Self.NP):
                    var kept = _kept(
                        Self.SEED, base, BATCH, Self.NP, bt, i, pmin, span
                    )
                    kp[bt * Self.NP + i] = (
                        Scalar[DT](1.0) if kept else Scalar[DT](0.0)
                    )
                    for d in range(Self.D):
                        if kept:
                            out[bt, i * Self.D + d] = inp[bt, i * Self.D + d]
                        else:
                            out[bt, i * Self.D + d] = mt[d]
        else:
            self._ensure_keep_gpu(BATCH)
            comptime STRIDE = UInt64(BATCH * (1 + Self.NP))
            var base = self.rng_step * STRIDE
            var pmin = Scalar[DT](self.p_min_rt)
            var span = Scalar[DT](self.p_max_rt - self.p_min_rt)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NP * Self.D), MutAnyOrigin
            ](inp.ptr)
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NP * Self.D), MutAnyOrigin
            ](out.ptr)
            var mt_lt = LayoutTensor[DT, Layout.row_major(Self.D), MutAnyOrigin](
                self.mask_token.val.dev.value()
            )
            var k_lt = LayoutTensor[
                DT, Layout.row_major(BATCH * Self.NP), MutAnyOrigin
            ](self.keep.dev.value())
            comptime nb = (BATCH * Self.NP * Self.D + 127) // 128
            comptime kern = _mae_fwd_kernel[BATCH, Self.NP, Self.D, Self.SEED]
            self.ts.ctx.value().enqueue_function[kern](
                in_lt, mt_lt, o_lt, k_lt, base, pmin, span,
                grid_dim=nb, block_dim=128,
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
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        comptime STRIDE = UInt64(BATCH * (1 + Self.NP))
        var base = self.rng_step * STRIDE
        var pmin = Scalar[DT](self.p_min_rt)
        var span = Scalar[DT](self.p_max_rt - self.p_min_rt)
        comptime if target == "cpu":
            var gmt = self.mask_token.grd.cpu.unsafe_ptr()
            for bt in range(BATCH):
                for i in range(Self.NP):
                    var kept = _kept(
                        Self.SEED, base, BATCH, Self.NP, bt, i, pmin, span
                    )
                    for d in range(Self.D):
                        var g = go[bt, i * Self.D + d]
                        if kept:
                            gi[bt, i * Self.D + d] = g
                        else:
                            gi[bt, i * Self.D + d] = Scalar[DT](0.0)
                            comptime if mode == "all":
                                gmt[d] += g
        else:
            var ctx = self.ts.ctx.value()
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NP * Self.D), MutAnyOrigin
            ](go.ptr)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NP * Self.D), MutAnyOrigin
            ](gi.ptr)
            comptime nb = (BATCH * Self.NP * Self.D + 127) // 128
            comptime gik = _mae_grad_input_kernel[
                BATCH, Self.NP, Self.D, Self.SEED
            ]
            ctx.enqueue_function[gik](
                go_lt, gi_lt, base, pmin, span, grid_dim=nb, block_dim=128
            )
            comptime if mode == "all":
                var gt_lt = LayoutTensor[
                    DT, Layout.row_major(Self.D), MutAnyOrigin
                ](self.mask_token.grd.dev.value())
                comptime gtk = _mae_grad_token_kernel[
                    BATCH, Self.NP, Self.D, Self.SEED
                ]
                ctx.enqueue_function[gtk](
                    go_lt, gt_lt, base, pmin, span,
                    grid_dim=Self.D, block_dim=MAE_RTPB,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
