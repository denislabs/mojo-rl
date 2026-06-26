"""DreamerOpt — the DreamerV3 reference optimizer chain, storage-native.

Reproduces, bit-for-bit, the chain in `references/dreamerv3-main/embodied/jax/opt.py`:

    optax.chain(
        clip_by_agc(agc),                   # agc=0.3, pmin=1e-3
        scale_by_rms(beta2, eps),           # beta2=0.999, eps=1e-20
        scale_by_momentum(beta1, nesterov), # beta1=0.9, nesterov=False
        scale_by_learning_rate(sched),      # const + linear warmup
    )

NOT AdamW — the moment buffers are wired differently. The RMS transform runs
FIRST (normalising the AGC-clipped grad by its own running RMS); the momentum
buffer then accumulates the RMS-NORMALISED grad. Per element (after AGC has
scaled the whole-leaf gradient by `agc_scale`):

    g      = grad · agc_scale
    nu     = beta2·nu + (1-beta2)·g²          # rms 2nd moment
    nu_hat = nu / (1 - beta2^t)
    g_rms  = g / (sqrt(nu_hat) + eps)
    mu     = beta1·mu + (1-beta1)·g_rms        # momentum on the normalised g
    mu_hat = mu / (1 - beta1^t)
    p      = p - lr·mu_hat

AGC (adaptive gradient clipping) is per-Param (= per JAX pytree leaf):

    gnorm = ‖grad‖₂ ; pnorm = ‖param‖₂
    upper = agc_clip · max(agc_pmin, pnorm)
    agc_scale = min(1, upper/gnorm)   (== 1/max(1, gnorm/upper))

Storage port (vs `nn/optimizer/dreamer_opt.mojo`): DRAMATICALLY simpler. Storage's
`Param` already owns its two optimizer-moment Tensors (`m`, `v`), so the two
Dreamer moments map directly — **`mu` → `m` (momentum), `nu` → `v` (rms)** — and
the entire legacy `nu_flat`/`mu_flat`/`offsets`/`TargetStorage` flat-side-table
machinery is GONE. This is a per-param `ParamVisitor` (one CPU loop / GPU
kernel-pair per Param), mirroring storage `Adam`'s shape: `step` bumps the bias
correction (`begin_step`) then walks `for_each_param`; AGC's per-leaf reduction
fits inside the SAME `visit` (no separate pass), reusing one [1] device scale
slot across the sequential per-param walk.

For graph-owned params (DreamerV3's WM/AC loss graphs own params as ComputeGraph
nodes — a ComputeGraph is NOT a Module), drive it as
`opt.begin_step(); graph.for_each_param[target, DreamerOpt](prefix, opt, ctx)`.

Drive the warmup schedule from the trainer via `opt.lr = sched.lr_at(step)`
before each step (see `schedules.LinearWarmupSchedule`). No CUDA-graph / arena
mode in this port — the legacy per-param path was the default too.
"""

from std.math import sqrt
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module
from .optimizer import Optimizer


comptime AGC_TPB: Int = 128  # single-block reduction width


# ── GPU kernels ─────────────────────────────────────────────────────────
def _agc_scale_kernel[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    scale_buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    agc_clip: Scalar[DT],
    agc_pmin: Scalar[DT],
):
    """Single-block, AGC_TPB-thread tree reduction of ‖grad‖² and ‖param‖²,
    then thread 0 folds the clip formula and writes `scale_buf[0]`."""
    var t = Int(thread_idx.x)
    var g_sum: Scalar[DT] = 0.0
    var p_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var g = rebind[Scalar[DT]](grad[k])
        var p = rebind[Scalar[DT]](param[k])
        g_sum += g * g
        p_sum += p * p
        k += AGC_TPB
    var g_total = block.sum[block_size=AGC_TPB, broadcast=False](val=g_sum)
    var p_total = block.sum[block_size=AGC_TPB, broadcast=False](val=p_sum)
    if t == 0:
        var scale: Scalar[DT] = 1.0
        if agc_clip > Scalar[DT](0.0):
            var gnorm = sqrt(g_total[0])
            var pnorm = sqrt(p_total[0])
            var pclamp = pnorm if pnorm > agc_pmin else agc_pmin
            var upper = agc_clip * pclamp
            if upper > Scalar[DT](0.0):
                var ratio = gnorm / upper
                if ratio > Scalar[DT](1.0):
                    scale = Scalar[DT](1.0) / ratio
        scale_buf[0] = scale


def _dreamer_update_kernel[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],  # mu (momentum)
    v: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],  # nu (rms)
    scale_buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    powbuf: LayoutTensor[DT, Layout.row_major(2), MutAnyOrigin],
):
    """rms → momentum → lr, with the AGC scale read from `scale_buf[0]`. One
    thread per element. `bc1/bc2 = 1 − β^t` are read from the device `powbuf`
    (`[β₁ᵗ, β₂ᵗ]`, advanced by `_dreamer_advance_pow_kernel` once per step) so
    the bias correction advances under CUDA-graph REPLAY — a host-baked `bc`
    would freeze at the capture-time step. Mirrors storage Adam's `powbuf`."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var one: Scalar[DT] = 1.0
    var bc1 = one - rebind[Scalar[DT]](powbuf[0])
    var bc2 = one - rebind[Scalar[DT]](powbuf[1])
    var sc = rebind[Scalar[DT]](scale_buf[0])
    var g = rebind[Scalar[DT]](grad[i]) * sc
    var nu_new = beta2 * rebind[Scalar[DT]](v[i]) + (one - beta2) * g * g
    v[i] = nu_new
    var nu_hat = nu_new / bc2
    var g_rms = g / (sqrt(nu_hat) + eps)
    var mu_new = beta1 * rebind[Scalar[DT]](m[i]) + (one - beta1) * g_rms
    m[i] = mu_new
    var mu_hat = mu_new / bc1
    param[i] = rebind[Scalar[DT]](param[i]) - lr * mu_hat


def _dreamer_advance_pow_kernel(
    powbuf: LayoutTensor[DT, Layout.row_major(2), MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Advance the device bias-correction powers `[β₁ᵗ, β₂ᵗ]` (1 thread, once
    per step before the per-param walk). β^t lives on-device so it advances on
    every CUDA-graph replay. Mirrors `_adam_advance_pow_kernel`."""
    if Int(global_idx.x) != 0:
        return
    powbuf[0] = rebind[Scalar[DT]](powbuf[0]) * beta1
    powbuf[1] = rebind[Scalar[DT]](powbuf[1]) * beta2


struct DreamerOpt(Movable, ParamVisitor, Optimizer):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]  # momentum
    var beta2: Scalar[DT]  # rms
    var eps: Scalar[DT]
    var agc_clip: Scalar[DT]
    var agc_pmin: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]
    # One [1] device scale slot reused across the sequential per-param GPU walk
    # (AGC kernel writes it, the update kernel reads it; ordered on the stream).
    # Allocated lazily on the first GPU visit. Empty on CPU.
    var _scale: Tensor
    # Device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (GPU path only).
    # Advanced on-device once per step by `begin_step_gpu` so the correction
    # advances under CUDA-graph replay; the update kernel reads `bc = 1 − β^t`
    # from it. CPU path uses the host `bc1/bc2`. Empty until the first GPU step.
    var _pow_dev: Tensor

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](4e-5))

    def __init__(
        out self,
        lr: Scalar[DT],  # required → disambiguates from the no-arg ctor
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-20,
        agc_clip: Scalar[DT] = 0.3,
        agc_pmin: Scalar[DT] = 1e-3,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.agc_clip = agc_clip
        self.agc_pmin = agc_pmin
        self.t = 0
        self._b1_pow = Scalar[DT](1.0)
        self._b2_pow = Scalar[DT](1.0)
        self.bc1 = Scalar[DT](1.0)
        self.bc2 = Scalar[DT](1.0)
        self._scale = Tensor()
        self._pow_dev = Tensor()

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Once per step,
        BEFORE the param walk (the graph-driving trainer calls this explicitly)."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def begin_step_gpu(mut self, ctx: DeviceContext) raises:
        """GPU twin of `begin_step`: advance the device powers `[β₁ᵗ, β₂ᵗ]`
        on-device (capture-safe) once per step, BEFORE the per-param walk. Lazily
        allocates `_pow_dev=[1,1]` on the first call (β⁰). The host `t` is still
        bumped for diagnostics; host `bc1/bc2` are unused on GPU."""
        self.t += 1
        if not self._pow_dev.dev:
            self._pow_dev = Tensor.alloc_gpu(ctx, 2)
            self._pow_dev.dev.value().enqueue_fill(Scalar[DT](1.0))
        ctx.enqueue_function[_dreamer_advance_pow_kernel](
            self._pow_dev.lt["gpu", Layout.row_major(2)](),
            self.beta1, self.beta2, grid_dim=1, block_dim=1,
        )

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param (per-param walk, both targets).
        GPU advances the device `[β₁ᵗ, β₂ᵗ]` (capture-safe); CPU bumps host bc."""
        comptime if target == "gpu":
            self.begin_step_gpu(ctx.value())
        else:
            self.begin_step()
        model.for_each_param[target](self, ctx)

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,  # DreamerV3 config wd=0 → ignored (no decay term)
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            m.ensure(N)  # mu (momentum) — lazy zero-alloc on first step
            v.ensure(N)  # nu (rms)
            var one = Scalar[DT](1.0)

            # ── AGC: per-leaf ‖grad‖₂ and ‖param‖₂ → agc_scale ──
            var g_sumsq: Scalar[DT] = 0.0
            var p_sumsq: Scalar[DT] = 0.0
            for i in range(N):
                var gs = grad.data[i]
                var ps = param.data[i]
                g_sumsq += gs * gs
                p_sumsq += ps * ps
            var agc_scale: Scalar[DT] = 1.0
            if self.agc_clip > Scalar[DT](0.0):
                var gnorm = sqrt(g_sumsq)
                var pnorm = sqrt(p_sumsq)
                var pclamp = pnorm if pnorm > self.agc_pmin else self.agc_pmin
                var upper = self.agc_clip * pclamp
                if upper > Scalar[DT](0.0):
                    var ratio = gnorm / upper
                    if ratio > Scalar[DT](1.0):
                        agc_scale = one / ratio

            # ── rms → momentum → lr ──
            for i in range(N):
                var g = grad.data[i] * agc_scale
                var nu_new = self.beta2 * v.data[i] + (one - self.beta2) * g * g
                v.data[i] = nu_new
                var nu_hat = nu_new / self.bc2
                var g_rms = g / (sqrt(nu_hat) + self.eps)
                var mu_new = self.beta1 * m.data[i] + (one - self.beta1) * g_rms
                m.data[i] = mu_new
                var mu_hat = mu_new / self.bc1
                param.data[i] = param.data[i] - self.lr * mu_hat
        else:
            var c = ctx.value()
            if not m.dev:  # first step: allocate + zero the moments
                m.ensure_gpu(c, N)
                m.dev.value().enqueue_fill(Scalar[DT](0))
                v.ensure_gpu(c, N)
                v.dev.value().enqueue_fill(Scalar[DT](0))
            if not self._scale.dev:
                self._scale = Tensor.alloc_gpu(c, 1)
            comptime layout = Layout.row_major(N)
            # Pass A: per-leaf AGC scale → _scale[0] (single block).
            c.enqueue_function[_agc_scale_kernel[N]](
                param.lt["gpu", layout](),
                grad.lt["gpu", layout](),
                self._scale.lt["gpu", Layout.row_major(1)](),
                self.agc_clip,
                self.agc_pmin,
                grid_dim=1,
                block_dim=AGC_TPB,
            )
            # Pass B: rms → momentum → lr (grid over elements). Same stream →
            # ordered after Pass A, so _scale[0] is ready.
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_dreamer_update_kernel[N]](
                param.lt["gpu", layout](),
                grad.lt["gpu", layout](),
                m.lt["gpu", layout](),
                v.lt["gpu", layout](),
                self._scale.lt["gpu", Layout.row_major(1)](),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.lt["gpu", Layout.row_major(2)](),
                grid_dim=nblk,
                block_dim=TPB,
            )
