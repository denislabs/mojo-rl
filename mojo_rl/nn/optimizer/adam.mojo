"""Adam — storage-native Adam / AdamW optimizer (CPU + GPU).

A `ParamVisitor`, but stateful: per-param 1st/2nd moments. Two GPU modes, ONE
class, IDENTICAL math (bit-parity, gated):

  - per-param (default, CPU + GPU): `step` walks `for_each_param`, one kernel /
    CPU loop per Param. The universal correctness path + CPU↔GPU parity check.
    Moments live on the `Param` (`m`/`v` Tensors), lazily zero-allocated.
  - arena (GPU, opt-in via `adopt`): a shared `ParamArena` packs params into
    contiguous val/grd buffers (+ this optimizer's own contiguous m/v arenas);
    `step` is ONE flat kernel over `[0,total)`, collapsing N launches → 1. Runs on
    Apple AND NVIDIA. `adopt` is a NO-OP on CPU → agent code is target-agnostic:

        opt.adopt[target](model, ctx); opt.step[target](model, ctx)

Decoupled weight decay (`wd > 0`, gated by `APPLY_DECAY` / the arena `decay_mask`)
makes this AdamW. Lifetime: the arena/moments are optimizer-owned; param slices
reference them (DeviceBuffer is refcounted → destruction order is safe).
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, ParamVersionBump
from ..core.module import Module
from .param_arena import ParamArena
from .grad_clip import (
    clip_grad_norm, clip_arena_grads, clip_arena_grads_captured,
)
from .optimizer import Optimizer


def _adam_update_kernel[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    wd: Scalar[DT],
    apply_decay: Int,
):
    """Per-param update (one Param, comptime size N).

    NOTE: `bc1/bc2` are host-baked here — fine for CPU and for the un-captured
    per-param GPU walk, but NOT CUDA-graph-safe (they'd freeze at capture-time).
    The capture path uses `adopt` → `_grouped_adam_kernel`, which reads β^t from
    a device buffer. Don't capture a non-adopted GPU optimizer."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var one = Scalar[DT](1.0)
    var p = rebind[Scalar[DT]](param[i])
    if apply_decay != 0:
        p -= lr * wd * p
    var g = rebind[Scalar[DT]](grad[i])
    var m_new = beta1 * rebind[Scalar[DT]](m[i]) + (one - beta1) * g
    var v_new = beta2 * rebind[Scalar[DT]](v[i]) + (one - beta2) * g * g
    m[i] = m_new
    v[i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    param[i] = p - lr * m_hat / (sqrt(v_hat) + eps)


def _adam_advance_pow_kernel(
    powbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Advance the device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (1
    thread). β^t lives on-device so it advances on every CUDA-graph REPLAY —
    a host-baked `bc` scalar would freeze at the capture-time step, scaling
    every replayed update by the early (t≈1) correction. Mirrors ScalarAdam's
    on-device `β^t` state."""
    if Int(global_idx.x) != 0:
        return
    powbuf[0] = powbuf[0] * beta1
    powbuf[1] = powbuf[1] * beta2


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
    powbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    wd: Scalar[DT],
):
    """Arena update (all params at once over runtime-length flat buffers).
    `bc1/bc2` are read from the device `powbuf` (`[β₁ᵗ, β₂ᵗ]`, advanced by
    `_adam_advance_pow_kernel` just before) so they advance under graph replay.
    """
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var bc1 = one - powbuf[0]
    var bc2 = one - powbuf[1]
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


def _adam_warmup_lr_kernel(
    lr_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    step_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_lr: Scalar[DT],
    warmup: Scalar[DT],
):
    """On-device LinearWarmup: `lr = target·min(step/warmup, 1)`, then `step+=1`
    (1 thread). Both `lr` and `step` live on-device, so the schedule advances on
    every CUDA-graph REPLAY — a host `opt.lr = sched.lr_at(t)` write would freeze
    at the capture-time (near-0 ramp) value. `warmup <= 0` → constant target."""
    if Int(global_idx.x) != 0:
        return
    var t = step_buf[0]
    var lr = target_lr
    if warmup > Scalar[DT](0.0) and t < warmup:
        lr = target_lr * t / warmup
    lr_buf[0] = lr
    step_buf[0] = t + Scalar[DT](1.0)


def _grouped_adam_kernel_devlr(
    val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grd: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v: UnsafePointer[Scalar[DT], MutAnyOrigin],
    decay: UnsafePointer[Scalar[DT], MutAnyOrigin],
    total: Int,
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    powbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lr_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    wd: Scalar[DT],
):
    """`_grouped_adam_kernel` but `lr` is read from the device `lr_buf` (written
    by `_adam_warmup_lr_kernel`) — the capture-safe scheduled-LR path. Identical
    math; only the LR source differs."""
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var lr = lr_buf[0]
    var bc1 = one - powbuf[0]
    var bc2 = one - powbuf[1]
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


struct Adam(Movable, ParamVisitor, Optimizer):
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
    # Arena mode (GPU, set by `adopt`): shared val/grd packing + own m/v arenas.
    var arena: ParamArena
    var m_arena: Tensor
    var v_arena: Tensor
    # Device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (GPU arena path only,
    # allocated by `adopt`). Advanced on-device each step so the correction
    # advances under CUDA-graph replay instead of freezing at the host-baked
    # capture-time value. CPU + per-param paths use the host `bc1/bc2`.
    var _pow_dev: Tensor
    # Persistent grad-clip scratch (GPU arena, allocated by `adopt`): block
    # partials + scale + pre-clip norm. Owned here so `clip_grads_device` does
    # NO per-call allocation (capture-safe). Empty until `adopt` on GPU.
    var _clip_partials: Tensor
    var _clip_scale: Tensor
    var _clip_norm: Tensor
    # Opt-in on-device LR schedule (GPU arena). When `_sched_active`, the grouped
    # step runs an on-device LinearWarmup into `_lr_dev` (step in `_step_dev`)
    # and the update reads `_lr_dev` — capture-safe. OFF by default → the update
    # uses the host `lr` (unchanged, zero overhead). Drive via
    # `attach_warmup_schedule`.
    var _lr_dev: Tensor
    var _step_dev: Tensor
    var _sched_active: Bool
    var _sched_target_lr: Scalar[DT]
    var _sched_warmup: Int
    # Host-driven device LR (GPU arena). When set, the grouped step reads `lr`
    # from `_lr_dev` (the device-LR kernel) but does NOT run the on-device
    # warmup ramp — the HOST writes the next LR into `_lr_dev` each step via
    # `push_lr_device` (a 1-elem `enqueue_fill`, NO realloc → the buffer pointer
    # is stable so a captured graph that baked it stays valid). This is the
    # capture-safe path for an ARBITRARY host-computed schedule (e.g. cosine),
    # vs `attach_warmup_schedule`'s on-device LinearWarmup. Mutually exclusive
    # with `_sched_active`.
    var _lr_host_driven: Bool

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](1e-3))

    def __init__(
        out self,
        lr: Scalar[DT],  # required → disambiguates from the no-arg ctor
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ):
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
        self._pow_dev = Tensor()
        self._clip_partials = Tensor()
        self._clip_scale = Tensor()
        self._clip_norm = Tensor()
        self._lr_dev = Tensor()
        self._step_dev = Tensor()
        self._sched_active = False
        self._sched_target_lr = Scalar[DT](0.0)
        self._sched_warmup = 0
        self._lr_host_driven = False
        self.arena = ParamArena()
        self.m_arena = Tensor()
        self.v_arena = Tensor()

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Once per step."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def adopt[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Engage arena mode (GPU); NO-OP on CPU. Packs val/grd via the shared
        ParamArena and allocates this optimizer's m/v arenas to match."""
        self.arena.adopt[target](model, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            self.m_arena = Tensor.alloc_gpu(c, self.arena.total)
            self.v_arena = Tensor.alloc_gpu(c, self.arena.total)
            # `[β₁ᵗ, β₂ᵗ]` seeded to β^0 = 1; advanced on-device each step.
            self._pow_dev = Tensor.alloc_gpu(c, 2)
            self._pow_dev.dev.value().enqueue_fill(Scalar[DT](1.0))
            # Persistent grad-clip scratch (one block-partials slot per TPB chunk
            # of the arena, + scale + norm) so `clip_grads_device` never allocs.
            var nblk = (self.arena.total + TPB - 1) // TPB
            self._clip_partials = Tensor.alloc_gpu(c, nblk if nblk > 0 else 1)
            self._clip_scale = Tensor.alloc_gpu(c, 1)
            self._clip_norm = Tensor.alloc_gpu(c, 1)
            # Device LR + step for the opt-in on-device schedule (seeded to the
            # current host lr / step 0; only used when `attach_warmup_schedule`).
            self._lr_dev = Tensor.alloc_gpu(c, 1)
            self._lr_dev.dev.value().enqueue_fill(self.lr)
            self._step_dev = Tensor.alloc_gpu(c, 1)
            self._step_dev.dev.value().enqueue_fill(Scalar[DT](0.0))

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param. GPU+adopted → one arena kernel;
        CPU or un-adopted GPU → per-param walk."""
        self.begin_step()
        comptime if target == "cpu":
            model.for_each_param["cpu"](self, ctx)
        else:
            if self.arena.adopted:
                self._grouped_step(ctx.value())
            else:
                model.for_each_param["gpu"](self, ctx)
        # AMP: invalidate cached bf16 weights — bump every param-value version so
        # leaves whose cached cast predates this step recast on next forward.
        # Host-only walk (no kernels); covers per-param AND arena paths. Not
        # CUDA-graph-capturable (host-side) — captured AMP is a Phase-5 concern.
        var _bump = ParamVersionBump()
        model.for_each_param[target](_bump, ctx)

    def _grouped_step(mut self, c: DeviceContext) raises:
        if self.arena.total == 0:
            return
        # Advance β^t on-device BEFORE the update reads it — captured into the
        # graph so it advances per replay (host `bc1/bc2` would freeze).
        c.enqueue_function[_adam_advance_pow_kernel](
            self._pow_dev.dev.value(),
            self.beta1,
            self.beta2,
            grid_dim=1,
            block_dim=1,
        )
        var nblk = (self.arena.total + TPB - 1) // TPB
        if self._sched_active:
            # On-device LinearWarmup → `_lr_dev` (advances `_step_dev`), then the
            # device-LR update kernel. Both captured → schedule advances per
            # replay. Default (no schedule) keeps the host-`lr` kernel below.
            c.enqueue_function[_adam_warmup_lr_kernel](
                self._lr_dev.dev.value(),
                self._step_dev.dev.value(),
                self._sched_target_lr,
                Scalar[DT](self._sched_warmup),
                grid_dim=1,
                block_dim=1,
            )
            c.enqueue_function[_grouped_adam_kernel_devlr](
                self.arena.val.dev.value(),
                self.arena.grd.dev.value(),
                self.m_arena.dev.value(),
                self.v_arena.dev.value(),
                self.arena.decay_mask.dev.value(),
                self.arena.total,
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.dev.value(),
                self._lr_dev.dev.value(),
                self.wd,
                grid_dim=nblk,
                block_dim=TPB,
            )
            return
        if self._lr_host_driven:
            # Host-driven device LR: read `lr` from `_lr_dev` (the host wrote it
            # via `push_lr_device` BEFORE this step / before the captured replay),
            # no on-device warmup ramp. Capture-safe — the kernel sequence is
            # fixed and `_lr_dev`'s pointer is stable.
            c.enqueue_function[_grouped_adam_kernel_devlr](
                self.arena.val.dev.value(),
                self.arena.grd.dev.value(),
                self.m_arena.dev.value(),
                self.v_arena.dev.value(),
                self.arena.decay_mask.dev.value(),
                self.arena.total,
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.dev.value(),
                self._lr_dev.dev.value(),
                self.wd,
                grid_dim=nblk,
                block_dim=TPB,
            )
            return
        c.enqueue_function[_grouped_adam_kernel](
            self.arena.val.dev.value(),
            self.arena.grd.dev.value(),
            self.m_arena.dev.value(),
            self.v_arena.dev.value(),
            self.arena.decay_mask.dev.value(),
            self.arena.total,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self._pow_dev.dev.value(),
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )

    def step_captured(mut self, c: DeviceContext) raises:
        """Capture-safe arena step for a CUDA-graph-captured train loop. Runs ONLY
        the grouped device step (`_grouped_step`): it advances `β^t` on the device
        `_pow_dev` and reads the device LR (`_lr_dev`, written eagerly by
        `push_lr_device`), so the correction + schedule advance on every replay.
        Unlike `step`, it does NO host work — no host `begin_step` counter bump and
        no AMP `ParamVersionBump` walk (both host-side ⇒ not capturable). Requires
        `adopt` (arena mode); a no-op if the arena is empty."""
        self._grouped_step(c)

    def zero_grad[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """GPU+adopted → zero the grad arena in ONE fill; else per-param via the
        model."""
        comptime if target == "gpu":
            if self.arena.adopted:
                self.arena.zero_grad()
                return
        model.zero_grad[target](ctx)

    def attach_warmup_schedule(mut self, target_lr: Scalar[DT], warmup_steps: Int):
        """Drive the LR with an ON-DEVICE LinearWarmup ramp (0 → target over
        `warmup_steps` update steps, then constant) so it's CUDA-graph-safe — the
        ramp advances on every replay. GPU-arena path only; a no-op stub for
        CPU/non-adopted (use host `set_lr` + a host schedule there). Replaces the
        host `opt.lr = sched.lr_at(t)` pattern, which freezes under capture."""
        self._sched_active = True
        self._sched_target_lr = target_lr
        self._sched_warmup = warmup_steps

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def push_lr_device[
        target: StaticString
    ](mut self, lr: Scalar[DT], ctx: Optional[DeviceContext] = None) raises:
        """Capture-safe LR update for an arbitrary host-computed schedule. On the
        GPU arena path, writes `lr` into the device `_lr_dev` buffer (1-elem
        `enqueue_fill`, NO realloc → stable pointer) and engages the device-LR
        kernel, so a CUDA-graph-captured `step` reads the FRESH LR on every
        replay instead of the host-baked capture-time value. Call this EAGERLY
        each step before the captured replay. Off the GPU-arena path it just
        sets the host `lr` (the un-captured kernel reads it directly). Keeps the
        host `lr` in sync either way (so `get_lr` stays meaningful)."""
        self.lr = lr
        comptime if target == "gpu":
            if self.arena.adopted:
                self._lr_host_driven = True
                self._lr_dev.dev.value().enqueue_fill(lr)

    def clip_grads[
        target: StaticString, M: Module
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """Global grad-norm clip, returns the pre-clip norm. GPU+adopted → arena
        reduction+scale (capture-safe, no per-param D2H); else → per-param
        `clip_grad_norm`. Symmetric across targets."""
        comptime if target == "gpu":
            if self.arena.adopted:
                return clip_arena_grads(self.arena, max_norm, ctx.value())
        return clip_grad_norm[target](model, max_norm, ctx)

    def clip_grads_device[
        target: StaticString, M: Module
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """CUDA-graph-safe grad-norm clip (GPU + adopted): on-device kernels over
        persistent scratch, NO allocation and NO D2H — drop this into the captured
        train-step sequence. The pre-clip norm lands in the device `_clip_norm`
        buffer; read it via `read_clip_norm` at flush cadence if you log it. Off
        the GPU-arena path it falls back to `clip_grads` (which D2Hs — NOT
        capture-safe; only correct when not capturing)."""
        comptime if target == "gpu":
            if self.arena.adopted:
                clip_arena_grads_captured(
                    self.arena, self._clip_partials, self._clip_scale,
                    self._clip_norm, max_norm, ctx.value(),
                )
                return
        _ = self.clip_grads[target, M](model, max_norm, ctx)

    def read_clip_norm(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        """D2H the last pre-clip grad norm from `clip_grads_device` (flush
        cadence only — NOT per step). 0 if never clipped on device."""
        if not self._clip_norm.dev:
            return Scalar[DT](0.0)
        self._clip_norm.download(ctx)
        return self._clip_norm.data[0]

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
        comptime if target == "cpu":
            m.ensure(N)  # lazy zero-alloc on first step
            v.ensure(N)
            var one = Scalar[DT](1.0)
            for i in range(N):
                var p = param.data[i]
                if apply_decay:
                    p -= self.lr * self.wd * p
                var g = grad.data[i]
                var m_new = self.beta1 * m.data[i] + (one - self.beta1) * g
                var v_new = self.beta2 * v.data[i] + (one - self.beta2) * g * g
                m.data[i] = m_new
                v.data[i] = v_new
                var m_hat = m_new / self.bc1
                var v_hat = v_new / self.bc2
                param.data[i] = p - self.lr * m_hat / (sqrt(v_hat) + self.eps)
        else:
            var c = ctx.value()
            if not m.dev:  # first step: allocate + zero the moments
                m.ensure_gpu(c, N)
                m.dev.value().enqueue_fill(Scalar[DT](0))
                v.ensure_gpu(c, N)
                v.dev.value().enqueue_fill(Scalar[DT](0))
            comptime layout = Layout.row_major(N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_adam_update_kernel[N]](
                param.lt["gpu", layout](),
                grad.lt["gpu", layout](),
                m.lt["gpu", layout](),
                v.lt["gpu", layout](),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.bc1,
                self.bc2,
                self.wd,
                Int(apply_decay),
                grid_dim=nblk,
                block_dim=TPB,
            )


# AdamW is Adam with decoupled weight decay (`wd > 0`, gated per param by
# APPLY_DECAY) — both the per-param and arena paths apply `p -= lr·wd·p` before
# the moment update. Construct as `AdamW(lr=..., wd=0.01)`.
comptime AdamW = Adam
