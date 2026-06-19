"""GroupedAdam — single-kernel Adam/AdamW over a contiguous param arena (GPU).

The storage answer to the legacy NVIDIA-only grouped optimizer. Legacy laundered
each param's device ADDRESS through a `u64` table + `unsafe_from_address` in-kernel
(the nightly-warning source; Metal silently drops writes to host-captured device
addresses → NVIDIA-only). Here we instead make the params CONTIGUOUS:

  `adopt(model)` allocates four arenas (val/grd/m/v, each Σ param sizes), copies
  every param's value into `val_arena` at its offset, and REBINDS each Param's
  `val`/`grd` device buffer to a `create_sub_buffer` slice of the arena. Forward
  reads / backward writes the slices transparently, so all grads land contiguously
  in `grd_arena`. `m_arena`/`v_arena` are optimizer-owned (indexed flat, never on
  the Param). A per-element `decay_mask` (1 where the param wants decay) carries
  AdamW's selective decay with no per-param offset scan.

`step` is then ONE flat-grid kernel over `[0, total)` — the arenas are real
contiguous buffers passed as their OWN pointers (the accepted GPU-ABI MutAnyOrigin
boundary, NOT an address table), so it runs on Apple AND NVIDIA. Math is identical
to the per-param `Adam` → bit-parity with it.

GPU-only: on CPU the per-param `Adam` loop has no launch overhead to collapse.
Lifetime: GroupedAdam owns the arenas; the model's param slices reference them
(DeviceBuffer is refcounted, so destruction order is safe).
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module
from ..core.named_params import named_params


def _grouped_adam_kernel(
    val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grd: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v: UnsafePointer[Scalar[DT], MutAnyOrigin],
    decay: UnsafePointer[Scalar[DT], MutAnyOrigin],
    total: Int,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    wd: Scalar[DT],
):
    """One thread per arena element: the standard Adam/AdamW update. `decay[i]`
    (0/1) gates the decoupled `p -= lr·wd·p` per element (AdamW). The arenas are
    plain contiguous buffers — no per-param address resolution."""
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var p = val[i]
    if decay[i] != Scalar[DT](0.0):
        p -= lr * wd * p
    var g = grd[i]
    var m_new = beta1 * m[i] + (one - beta1) * g
    var v_new = beta2 * v[i] + (one - beta2) * g * g
    m[i] = m_new
    v[i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    val[i] = p - lr * m_hat / (sqrt(v_hat) + eps)


struct GroupedAdam(Movable & ParamVisitor):
    var val_arena: Tensor
    var grd_arena: Tensor
    var m_arena: Tensor
    var v_arena: Tensor
    var decay_mask: Tensor
    var total: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]
    var _adopt_off: Int  # running arena offset, used only during adopt's walk

    def __init__(
        out self,
        lr: Scalar[DT] = 1e-3,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ):
        self.val_arena = Tensor()
        self.grd_arena = Tensor()
        self.m_arena = Tensor()
        self.v_arena = Tensor()
        self.decay_mask = Tensor()
        self.total = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self._b1_pow = Scalar[DT](1.0)
        self._b2_pow = Scalar[DT](1.0)
        self.bc1 = Scalar[DT](1.0)
        self.bc2 = Scalar[DT](1.0)
        self._adopt_off = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """Placement visitor (used by `adopt` only): copy each param's value into
        `val_arena` at the running offset and rebind its val/grd device buffers to
        arena slices. `param`/`grad` ARE the val/grd Tensors (unpacked by
        Param.visit_with) and the mut refs chain back to the model's Param, so the
        rebinds persist. m/v stay optimizer-internal (indexed flat in step)."""
        var c = ctx.value()
        var vsub = self.val_arena.dev.value().create_sub_buffer[DT](
            self._adopt_off, N
        )
        c.enqueue_copy(vsub, param.dev.value())  # preserve init values
        param.dev = Optional(vsub)
        param.n = N
        var gsub = self.grd_arena.dev.value().create_sub_buffer[DT](
            self._adopt_off, N
        )  # grd_arena pre-zeroed; backward writes here
        grad.dev = Optional(gsub)
        grad.n = N
        self._adopt_off += N

    def adopt[
        M: Module
    ](mut self, mut model: M, ctx: DeviceContext) raises:
        """Arena-ify `model`: allocate the 4 arenas, copy param values in, rebind
        param val/grd buffers to arena slices, build the decay mask. Call ONCE
        after the model is made + initialized, before the first step."""
        var nps = named_params["gpu"](model)
        var total = 0
        for i in range(len(nps)):
            total += nps[i].size
        self.total = total

        # Decay mask (host-built from param metadata, uploaded once).
        var dm = Tensor.alloc(total)
        var off = 0
        for i in range(len(nps)):
            var d = Scalar[DT](1.0) if nps[i].decay else Scalar[DT](0.0)
            for k in range(nps[i].size):
                dm.data[off + k] = d
            off += nps[i].size
        dm.upload(ctx)
        self.decay_mask = dm^

        self.val_arena = Tensor.alloc_gpu(ctx, total)  # zeroed
        self.grd_arena = Tensor.alloc_gpu(ctx, total)
        self.m_arena = Tensor.alloc_gpu(ctx, total)
        self.v_arena = Tensor.alloc_gpu(ctx, total)
        # Walk with self as the placement visitor (reads self.val_arena/grd_arena
        # + self._adopt_off directly — no ownership transfer).
        self._adopt_off = 0
        model.for_each_param["gpu"](self, Optional(ctx))

    def begin_step(mut self):
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def zero_grad(mut self) raises:
        """Zero the whole grad arena in ONE fill (vs N per-param fills)."""
        if self.total > 0:
            self.grd_arena.dev.value().enqueue_fill(Scalar[DT](0))

    def step(mut self, ctx: DeviceContext) raises:
        """ONE kernel updates every param. Call after backward (grads in the
        arena) and a `begin_step` is done internally."""
        self.begin_step()
        if self.total == 0:
            return
        var nblk = (self.total + TPB - 1) // TPB
        ctx.enqueue_function[_grouped_adam_kernel](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.val_arena.dev.value().unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grd_arena.dev.value().unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.m_arena.dev.value().unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.v_arena.dev.value().unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.decay_mask.dev.value().unsafe_ptr()
            ),
            self.total,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.bc1,
            self.bc2,
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )
