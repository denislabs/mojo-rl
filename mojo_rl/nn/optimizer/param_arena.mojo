"""ParamArena — contiguous param packing shared by the grouped optimizers + clip.

`adopt[target](model)` (GPU; NO-OP on CPU) packs every Param into two contiguous
device buffers — `val` and `grd` — and REBINDS each Param's val/grd device buffer
to a `create_sub_buffer` slice. Forward reads / backward writes the slices
transparently, so all grads land contiguously in `grd`. A per-element `decay_mask`
(1 where the param wants weight decay) carries AdamW's selective decay with no
per-param offset scan.

This is the rule-of-three extraction: `Adam` (adds m/v arenas), `SGD` (stateless),
and arena grad-clip (`grad_clip.clip_arena_grads`) all build on it. The optimizer
owns its ParamArena; the model's param slices reference it (DeviceBuffer is
refcounted → destruction order is safe). GPU-only: on CPU the per-param path has
no launch overhead to collapse, so `adopt` does nothing and `adopted` stays False.

ParamArena IS the placement `ParamVisitor` (its `visit` reads its own `val`/`grd`
+ offset → no ownership transfer); the placement walk is comptime-gated to GPU so
the device ops never compile into the CPU path.
"""

from std.gpu import global_idx
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module
from ..core.named_params import named_params


struct ParamArena(Movable & ParamVisitor):
    var val: Tensor  # contiguous param-value arena
    var grd: Tensor  # contiguous gradient arena
    var decay_mask: Tensor  # per-element 0/1 weight-decay gate
    var total: Int
    var adopted: Bool
    var _off: Int  # running offset during the placement walk

    def __init__(out self):
        self.val = Tensor()
        self.grd = Tensor()
        self.decay_mask = Tensor()
        self.total = 0
        self.adopted = False
        self._off = 0

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """Placement (adopt walk, GPU only): copy the param's value into `val` at
        the running offset and rebind its val/grd buffers to arena slices.
        `param`/`grad` ARE the val/grd Tensors; the mut refs chain back to the
        model's Param, so the rebinds persist."""
        comptime if target == "gpu":
            var c = ctx.value()
            var vsub = self.val.dev.value().create_sub_buffer[DT](self._off, N)
            c.enqueue_copy(vsub, param.dev.value())  # preserve init values
            param.dev = Optional(vsub)
            param.n = N
            var gsub = self.grd.dev.value().create_sub_buffer[DT](self._off, N)
            grad.dev = Optional(gsub)
            grad.n = N
            self._off += N

    def adopt[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Pack `model` into the arena (GPU); NO-OP on CPU. Call ONCE after the
        model is made + initialized, before the first step."""
        comptime if target == "gpu":
            var c = ctx.value()
            var nps = named_params["gpu"](model)
            var total = 0
            for i in range(len(nps)):
                total += nps[i].size
            self.total = total

            var dm = Tensor.alloc(total)  # host decay mask
            var off = 0
            for i in range(len(nps)):
                var d = Scalar[DT](1.0) if nps[i].decay else Scalar[DT](0.0)
                for k in range(nps[i].size):
                    dm.data[off + k] = d
                off += nps[i].size
            dm.upload(c)
            self.decay_mask = dm^

            self.val = Tensor.alloc_gpu(c, total)  # zeroed
            self.grd = Tensor.alloc_gpu(c, total)
            self._off = 0
            model.for_each_param["gpu"](self, Optional(c))
            self.adopted = True

    def zero_grad(mut self) raises:
        """Zero the whole grad arena in ONE fill (vs N per-param fills)."""
        if self.adopted and self.total > 0:
            self.grd.dev.value().enqueue_fill(Scalar[DT](0))


def _polyak_kernel(
    target: Pointer[Scalar[DT], MutAnyOrigin],
    online: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    tau: Scalar[DT],
):
    """target[i] = (1-τ)·target[i] + τ·online[i] over the whole value arena."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i < total:
        target[unsafe_offset=i] = (Scalar[DT](1.0) - tau) * target[unsafe_offset=i] + tau * online[unsafe_offset=i]


def polyak_arenas(
    mut target: ParamArena,
    mut online: ParamArena,
    tau: Scalar[DT],
    ctx: DeviceContext,
) raises:
    """Grouped target-net soft-update: `target = (1-τ)·target + τ·online` in ONE
    kernel over the contiguous value arenas (vs N per-param launches via
    `Module.polyak_from`). Both models must be arena-backed (same param layout →
    same `total`). `online` is `mut` for the GPU-ABI (read-only in the kernel)."""
    if target.total == 0 or target.total != online.total:
        raise Error(
            "polyak_arenas: arena size mismatch (target "
            + String(target.total) + " vs online " + String(online.total) + ")"
        )
    var nblk = (target.total + TPB - 1) // TPB
    ctx.enqueue_function[_polyak_kernel](
        target.val.dev.value(),
        online.val.dev.value(),
        Int64(target.total),
        tau,
        grid_dim=nblk,
        block_dim=TPB,
    )
