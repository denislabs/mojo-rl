"""DreamerOpt — the DreamerV3 reference optimizer chain.

Reproduces, bit-for-bit, the chain assembled in
`references/dreamerv3-main/dreamerv3/agent.py:_make_opt` from the custom
transforms in `references/dreamerv3-main/embodied/jax/opt.py`:

    optax.chain(
        clip_by_agc(agc),                  # agc=0.3, pmin=1e-3
        scale_by_rms(beta2, eps),          # beta2=0.999, eps=1e-20
        scale_by_momentum(beta1, nesterov),# beta1=0.9, nesterov=False
        # add_decayed_weights(wd, ...)     # SKIPPED — config default wd=0
        scale_by_learning_rate(sched),     # const + linear warmup
    )

NOT AdamW: the moment buffers are wired differently. In Adam the first
moment accumulates the RAW gradient. Here the RMS transform runs FIRST
(normalising the AGC-clipped grad by its own running RMS), and the
momentum buffer then accumulates the RMS-NORMALISED grad. Order matters —
see `step` for the exact sequence.

Per-element update (after AGC has scaled the whole-leaf gradient):

    nu     = beta2·nu + (1-beta2)·g²          # rms second moment
    nu_hat = nu / (1 - beta2^t)               # bias correction
    g_rms  = g / (sqrt(nu_hat) + eps)
    mu     = beta1·mu + (1-beta1)·g_rms        # momentum (on normalised g)
    mu_hat = mu / (1 - beta1^t)               # bias correction
    p      = p - lr·mu_hat

AGC (adaptive gradient clipping) is per-Param (= per JAX pytree leaf):

    gnorm = ||grad||_2   (over the whole flattened leaf)
    pnorm = ||param||_2
    upper = agc_clip · max(agc_pmin, pnorm)
    scale = 1 / max(1, gnorm / upper)          # == min(1, upper/gnorm)
    grad  = grad · scale

Unlike global grad-norm clipping (`core/grad_clip.mojo`, which needs all
params summed before scaling), AGC is leaf-local — the norms and the
scale are computed inside the SAME per-Param visit, no separate pass.

The `lr` field is a public mut hyperparam (matches Adam). Drive the
warmup schedule from the trainer via
`opt.lr = sched.lr_at(step); opt.step[...](model)`
(see `schedules.LinearWarmupSchedule`).

Storage / dispatch / Saveable layout all mirror `optimizer/adam.mojo`.
"""

from std.math import sqrt
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import ParamVisitor, GraphNode
from ..core.module import Module
from ..core.optimizer import Optimizer
from ..combinators.compute_graph import ComputeGraph
from ..core.saveable import Saveable
from ..core.save_scalar import _expect_kv_line
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


comptime AGC_TPB: Int = 128  # single-block reduction width; mirrors GC_TPB.


def _agc_scale_kernel(
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    slot: Int,
    n_elems: Int,
    agc_clip: Scalar[DT],
    agc_pmin: Scalar[DT],
):
    """Single-block, AGC_TPB-thread tree reduction of ‖grad‖² and ‖param‖²,
    then thread 0 computes the per-leaf AGC scale and writes
    `scale_buf[slot]`. Mirrors `grad_clip._sum_sq_partial_kernel` but
    reduces two quantities and folds in the clip formula."""
    var t = Int(thread_idx.x)
    var g_sum: Scalar[DT] = 0.0
    var p_sum: Scalar[DT] = 0.0
    var k = t
    while k < n_elems:
        var g = grad[k]
        var p = param[k]
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
        scale_buf[slot] = scale


def _dreamer_step_prep_kernel(
    step_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    # Single-thread on-device step bump + bias-correction update. Keeps the
    # step counter off the host so the GPU step is CUDA-graph capturable
    # (no host scalar baked into the update kernel's args). Mirrors AdamW's
    # `_adamw_step_prep_kernel`. `bc_ptr` layout: [beta1_pow_t, beta2_pow_t,
    # bias_correction1, bias_correction2].
    if Int(global_idx.x) == 0:
        step_ptr[0] = step_ptr[0] + UInt32(1)
        var b1_new = bc_ptr[0] * beta1
        var b2_new = bc_ptr[1] * beta2
        bc_ptr[0] = b1_new
        bc_ptr[1] = b2_new
        bc_ptr[2] = Scalar[DT](1.0) - b1_new
        bc_ptr[3] = Scalar[DT](1.0) - b2_new


def _dreamer_update_kernel(
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    nu_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mu_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    slot: Int,
    n_elems: Int,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
):
    """rms-then-momentum-then-lr, with the AGC scale read from
    `scale_buf[slot]` and bias-correction read from `bc_ptr`. One thread
    per element."""
    var i = Int(global_idx.x)
    if i < n_elems:
        var one: Scalar[DT] = 1.0
        var bc1 = bc_ptr[2]
        var bc2 = bc_ptr[3]
        var g = grad[i] * scale_buf[slot]
        var nu_new = beta2 * nu_off[i] + (one - beta2) * g * g
        nu_off[i] = nu_new
        var nu_hat = nu_new / bc2
        var g_rms = g / (sqrt(nu_hat) + eps)
        var mu_new = beta1 * mu_off[i] + (one - beta1) * g_rms
        mu_off[i] = mu_new
        var mu_hat = mu_new / bc1
        param[i] = param[i] - lr * mu_hat


def _zero_fill_kernel(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
):
    var i = Int(global_idx.x)
    if i < n_elems:
        ptr[i] = 0.0


# ──────────────────────────────────────────────────────────────────────
# CPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _DreamerCPUInitVisitor(ParamVisitor):
    var nu_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var mu_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var zero: Scalar[DT] = 0.0
        self.offsets_ptr[].append(len(self.nu_flat_ptr[]))
        for _ in range(n_elems):
            self.nu_flat_ptr[].append(zero)
            self.mu_flat_ptr[].append(zero)


@fieldwise_init
struct _DreamerCPUStepVisitor(ParamVisitor):
    var nu_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var mu_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var agc_clip: Scalar[DT]
    var agc_pmin: Scalar[DT]
    var bias_correction1: Scalar[DT]
    var bias_correction2: Scalar[DT]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var nu_ptr = self.nu_flat_ptr[].unsafe_ptr() + off
        var mu_ptr = self.mu_flat_ptr[].unsafe_ptr() + off

        # ── AGC: per-leaf ‖grad‖₂ and ‖param‖₂ (SIMD reduce + scalar tail) ──
        var one: Scalar[DT] = 1.0
        var g_acc = SIMD[DT, CPU_SIMD_W](0.0)
        var p_acc = SIMD[DT, CPU_SIMD_W](0.0)
        var j = 0
        while j + CPU_SIMD_W <= n_elems:
            var gv = g_ptr.load[width=CPU_SIMD_W](j)
            var pv = p_ptr.load[width=CPU_SIMD_W](j)
            g_acc += gv * gv
            p_acc += pv * pv
            j += CPU_SIMD_W
        var g_sumsq = g_acc.reduce_add()
        var p_sumsq = p_acc.reduce_add()
        while j < n_elems:
            var gs = g_ptr[j]
            var ps = p_ptr[j]
            g_sumsq += gs * gs
            p_sumsq += ps * ps
            j += 1
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

        # ── rms → momentum → lr (SIMD body + scalar tail) ──
        var b1_v = SIMD[DT, CPU_SIMD_W](self.beta1)
        var b2_v = SIMD[DT, CPU_SIMD_W](self.beta2)
        var omb1_v = SIMD[DT, CPU_SIMD_W](one) - b1_v
        var omb2_v = SIMD[DT, CPU_SIMD_W](one) - b2_v
        var bc1_v = SIMD[DT, CPU_SIMD_W](self.bias_correction1)
        var bc2_v = SIMD[DT, CPU_SIMD_W](self.bias_correction2)
        var lr_v = SIMD[DT, CPU_SIMD_W](self.lr)
        var eps_v = SIMD[DT, CPU_SIMD_W](self.eps)
        var scale_v = SIMD[DT, CPU_SIMD_W](agc_scale)
        var i = 0
        while i + CPU_SIMD_W <= n_elems:
            var g = g_ptr.load[width=CPU_SIMD_W](i) * scale_v
            var nu_old = nu_ptr.load[width=CPU_SIMD_W](i)
            var nu_new = b2_v * nu_old + omb2_v * g * g
            nu_ptr.store(i, nu_new)
            var nu_hat = nu_new / bc2_v
            var g_rms = g / (sqrt(nu_hat) + eps_v)
            var mu_old = mu_ptr.load[width=CPU_SIMD_W](i)
            var mu_new = b1_v * mu_old + omb1_v * g_rms
            mu_ptr.store(i, mu_new)
            var mu_hat = mu_new / bc1_v
            var p = p_ptr.load[width=CPU_SIMD_W](i)
            p_ptr.store(i, p - lr_v * mu_hat)
            i += CPU_SIMD_W
        while i < n_elems:
            var g = g_ptr[i] * agc_scale
            var nu_new = self.beta2 * nu_ptr[i] + (one - self.beta2) * g * g
            nu_ptr[i] = nu_new
            var nu_hat = nu_new / self.bias_correction2
            var g_rms = g / (sqrt(nu_hat) + self.eps)
            var mu_new = self.beta1 * mu_ptr[i] + (one - self.beta1) * g_rms
            mu_ptr[i] = mu_new
            var mu_hat = mu_new / self.bias_correction1
            p_ptr[i] = p_ptr[i] - self.lr * mu_hat
            i += 1
        self.idx += 1


struct _ZeroGradCPUVisitor(ParamVisitor):
    def __init__(out self):
        pass

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────
# GPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _DreamerGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var total_ptr: UnsafePointer[Int, MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.offsets_ptr[].append(self.total_ptr[])
        self.total_ptr[] = self.total_ptr[] + n_elems


@fieldwise_init
struct _DreamerGPUStepVisitor(ParamVisitor):
    var ctx: DeviceContext
    var nu_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mu_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var scale_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var bc_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var agc_clip: Scalar[DT]
    var agc_pmin: Scalar[DT]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var nu_off = self.nu_base + off
        var mu_off = self.mu_base + off
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)

        # Pass A: per-leaf AGC scale into scale_base[idx] (single block).
        self.ctx.enqueue_function[_agc_scale_kernel](
            p_ptr, g_ptr, self.scale_base, self.idx, n_elems,
            self.agc_clip, self.agc_pmin,
            grid_dim=1, block_dim=AGC_TPB,
        )
        # Pass B: rms → momentum → lr (grid over elements). Same stream →
        # ordered after Pass A, so scale_base[idx] is ready.
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_dreamer_update_kernel](
            p_ptr, g_ptr, nu_off, mu_off, self.scale_base, self.bc_base,
            self.idx, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.idx += 1


@fieldwise_init
struct _ZeroGradGPUVisitor(ParamVisitor):
    var ctx: DeviceContext

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var grad_w_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# DreamerOpt.
# ──────────────────────────────────────────────────────────────────────


struct DreamerOpt(Optimizer, Saveable):
    # CPU storage
    var nu_flat: List[Scalar[DT]]
    var mu_flat: List[Scalar[DT]]
    # GPU storage
    var nu_dev: Optional[DeviceBuffer[DT]]
    var mu_dev: Optional[DeviceBuffer[DT]]
    var scale_dev: Optional[DeviceBuffer[DT]]   # [N_PARAMS] AGC scale scratch
    # On-device step + bias-correction state (GPU path only). Keeps the step
    # counter off the host so the GPU `step` is CUDA-graph capturable. `bc_dev`
    # layout: [beta1_pow_t, beta2_pow_t, bias_correction1, bias_correction2].
    var step_dev: Optional[DeviceBuffer[DType.uint32]]
    var bc_dev: Optional[DeviceBuffer[DT]]
    # Shared bookkeeping
    var offsets: List[Int]
    var total_size: Int
    # Host-side step state (CPU path; GPU uses step_dev/bc_dev instead).
    var step_count: Int

    # Public mut hyperparams. Defaults match the DreamerV3 reference
    # (`agent.py:_make_opt` + `configs.yaml`).
    var lr: Scalar[DT]
    var beta1: Scalar[DT]   # momentum
    var beta2: Scalar[DT]   # rms
    var eps: Scalar[DT]
    var agc_clip: Scalar[DT]
    var agc_pmin: Scalar[DT]

    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    var ts: TargetStorage

    def __init__(out self):
        self.nu_flat = List[Scalar[DT]]()
        self.mu_flat = List[Scalar[DT]]()
        self.nu_dev = None
        self.mu_dev = None
        self.scale_dev = None
        self.step_dev = None
        self.bc_dev = None
        self.offsets = List[Int]()
        self.total_size = 0
        self.step_count = 0
        self.lr = Scalar[DT](4e-5)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-20)
        self.agc_clip = Scalar[DT](0.3)
        self.agc_pmin = Scalar[DT](1e-3)
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory — no hyperparams. User sets `opt.lr`
        etc. after (or drives `opt.lr` from a warmup schedule). `ctx=None`
        on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "DreamerOpt: target must be 'cpu' or 'gpu'"
        )
        var opt = Self()
        comptime if target == "cpu":
            var visitor = _DreamerCPUInitVisitor(
                nu_flat_ptr=UnsafePointer(to=opt.nu_flat),
                mu_flat_ptr=UnsafePointer(to=opt.mu_flat),
                offsets_ptr=UnsafePointer(to=opt.offsets),
            )
            model.for_each_param[target, _DreamerCPUInitVisitor](
                String(""), visitor,
            )
            opt.total_size = len(opt.nu_flat)
            opt.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DreamerOpt.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            var visitor = _DreamerGPUInitVisitor(
                offsets_ptr=UnsafePointer(to=opt.offsets),
                total_ptr=UnsafePointer(to=opt.total_size),
            )
            model.for_each_param[target, _DreamerGPUInitVisitor](
                String(""), visitor,
            )
            var nu_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            var mu_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            nu_real.enqueue_fill(0.0)
            mu_real.enqueue_fill(0.0)
            opt.nu_dev = nu_real^
            opt.mu_dev = mu_real^
            # One AGC-scale slot per Param (== len(offsets) after init walk).
            var scale_real = ctx_v.enqueue_create_buffer[DT](len(opt.offsets))
            scale_real.enqueue_fill(1.0)
            opt.scale_dev = scale_real^
            # On-device step counter (0) + bias-correction buffer
            # [beta1_pow_t=1, beta2_pow_t=1, bc1=0, bc2=0].
            var step_real = ctx_v.enqueue_create_buffer[DType.uint32](1)
            step_real.enqueue_fill(0)
            var bc_real = ctx_v.enqueue_create_buffer[DT](4)
            var bc_init_host = ctx_v.enqueue_create_host_buffer[DT](4)
            ctx_v.synchronize()
            bc_init_host.unsafe_ptr()[0] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[1] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[2] = Scalar[DT](0.0)
            bc_init_host.unsafe_ptr()[3] = Scalar[DT](0.0)
            ctx_v.enqueue_copy(bc_real, bc_init_host)
            opt.step_dev = step_real^
            opt.bc_dev = bc_real^
            opt.ts = TargetStorage.make_gpu(ctx_v)
        return opt^

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["DreamerOpt", target](self.ts.target_tag)
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target, _ZeroGradCPUVisitor](String(""), v)
        else:
            var v = _ZeroGradGPUVisitor(ctx=self.ts.ctx.value())
            model.for_each_param[target, _ZeroGradGPUVisitor](String(""), v)

    def step[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["DreamerOpt", target](self.ts.target_tag)

        comptime if target == "cpu":
            # Host step + bias-correction (CPU path only).
            self.step_count += 1
            self.beta1_pow_t = self.beta1_pow_t * self.beta1
            self.beta2_pow_t = self.beta2_pow_t * self.beta2
            var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
            var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t
            var visitor = _DreamerCPUStepVisitor(
                nu_flat_ptr=UnsafePointer(to=self.nu_flat),
                mu_flat_ptr=UnsafePointer(to=self.mu_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                agc_clip=self.agc_clip, agc_pmin=self.agc_pmin,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target, _DreamerCPUStepVisitor](
                String(""), visitor
            )
        else:
            var ctx = self.ts.ctx.value()
            var step_ptr: UnsafePointer[UInt32, MutAnyOrigin] = (
                self.step_dev.value().unsafe_ptr()
            )
            var bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
                self.bc_dev.value().unsafe_ptr()
            )
            # On-device step bump + bias-correction. No host scalar feeds the
            # update kernel → CUDA-graph capturable.
            ctx.enqueue_function[_dreamer_step_prep_kernel](
                step_ptr, bc_ptr, self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            var visitor = _DreamerGPUStepVisitor(
                ctx=ctx,
                nu_base=self.nu_dev.value().unsafe_ptr(),
                mu_base=self.mu_dev.value().unsafe_ptr(),
                scale_base=self.scale_dev.value().unsafe_ptr(),
                bc_base=bc_ptr,
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                agc_clip=self.agc_clip, agc_pmin=self.agc_pmin,
            )
            model.for_each_param[target, _DreamerGPUStepVisitor](
                String(""), visitor
            )

    # ──────────────────────────────────────────────────────────────────
    # ComputeGraph overloads — a `ComputeGraph` exposes the same
    # `for_each_param` walk as a `Module` but does NOT conform to the
    # `Module` trait (it uses `set_input` instead of the variadic forward
    # surface). DreamerV3's WM/head loss graphs own their params as graph
    # nodes, so these overloads let one DreamerOpt size/zero/step over a
    # whole graph's params. Bodies mirror make/zero_grad/step exactly.
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def make_graph[
        target: StaticString, OUT: Int, *NODES: GraphNode
    ](
        mut g: ComputeGraph[OUT, *NODES],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DreamerOpt.make_graph: target must be 'cpu' or 'gpu'"
        )
        var opt = Self()
        comptime if target == "cpu":
            var visitor = _DreamerCPUInitVisitor(
                nu_flat_ptr=UnsafePointer(to=opt.nu_flat),
                mu_flat_ptr=UnsafePointer(to=opt.mu_flat),
                offsets_ptr=UnsafePointer(to=opt.offsets),
            )
            g.for_each_param[target, _DreamerCPUInitVisitor](String(""), visitor)
            opt.total_size = len(opt.nu_flat)
            opt.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DreamerOpt.make_graph[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            var visitor = _DreamerGPUInitVisitor(
                offsets_ptr=UnsafePointer(to=opt.offsets),
                total_ptr=UnsafePointer(to=opt.total_size),
            )
            g.for_each_param[target, _DreamerGPUInitVisitor](String(""), visitor)
            var nu_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            var mu_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            nu_real.enqueue_fill(0.0)
            mu_real.enqueue_fill(0.0)
            opt.nu_dev = nu_real^
            opt.mu_dev = mu_real^
            var scale_real = ctx_v.enqueue_create_buffer[DT](len(opt.offsets))
            scale_real.enqueue_fill(1.0)
            opt.scale_dev = scale_real^
            # On-device step counter (0) + bias-correction buffer
            # [beta1_pow_t=1, beta2_pow_t=1, bc1=0, bc2=0].
            var step_real = ctx_v.enqueue_create_buffer[DType.uint32](1)
            step_real.enqueue_fill(0)
            var bc_real = ctx_v.enqueue_create_buffer[DT](4)
            var bc_init_host = ctx_v.enqueue_create_host_buffer[DT](4)
            ctx_v.synchronize()
            bc_init_host.unsafe_ptr()[0] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[1] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[2] = Scalar[DT](0.0)
            bc_init_host.unsafe_ptr()[3] = Scalar[DT](0.0)
            ctx_v.enqueue_copy(bc_real, bc_init_host)
            opt.step_dev = step_real^
            opt.bc_dev = bc_real^
            opt.ts = TargetStorage.make_gpu(ctx_v)
        return opt^

    def zero_grad_graph[
        target: StaticString, OUT: Int, *NODES: GraphNode
    ](mut self, mut g: ComputeGraph[OUT, *NODES]) raises:
        assert_tag_for["DreamerOpt", target](self.ts.target_tag)
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            g.for_each_param[target, _ZeroGradCPUVisitor](String(""), v)
        else:
            var v = _ZeroGradGPUVisitor(ctx=self.ts.ctx.value())
            g.for_each_param[target, _ZeroGradGPUVisitor](String(""), v)

    def step_graph[
        target: StaticString, OUT: Int, *NODES: GraphNode
    ](mut self, mut g: ComputeGraph[OUT, *NODES]) raises:
        assert_tag_for["DreamerOpt", target](self.ts.target_tag)
        comptime if target == "cpu":
            # Host step + bias-correction (CPU path only).
            self.step_count += 1
            self.beta1_pow_t = self.beta1_pow_t * self.beta1
            self.beta2_pow_t = self.beta2_pow_t * self.beta2
            var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
            var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t
            var visitor = _DreamerCPUStepVisitor(
                nu_flat_ptr=UnsafePointer(to=self.nu_flat),
                mu_flat_ptr=UnsafePointer(to=self.mu_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                agc_clip=self.agc_clip, agc_pmin=self.agc_pmin,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            g.for_each_param[target, _DreamerCPUStepVisitor](String(""), visitor)
        else:
            var ctx = self.ts.ctx.value()
            var step_ptr: UnsafePointer[UInt32, MutAnyOrigin] = (
                self.step_dev.value().unsafe_ptr()
            )
            var bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
                self.bc_dev.value().unsafe_ptr()
            )
            # On-device step bump + bias-correction. No host scalar feeds the
            # update kernel → CUDA-graph capturable.
            ctx.enqueue_function[_dreamer_step_prep_kernel](
                step_ptr, bc_ptr, self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            var visitor = _DreamerGPUStepVisitor(
                ctx=ctx,
                nu_base=self.nu_dev.value().unsafe_ptr(),
                mu_base=self.mu_dev.value().unsafe_ptr(),
                scale_base=self.scale_dev.value().unsafe_ptr(),
                bc_base=bc_ptr,
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                agc_clip=self.agc_clip, agc_pmin=self.agc_pmin,
            )
            g.for_each_param[target, _DreamerGPUStepVisitor](String(""), visitor)

    # ─────────────────────────── Saveable (CPU only) ───────────────────────────
    # Mirrors Adam's layout with the two extra AGC hyperparams. Topology-
    # derived state (offsets, total_size) is NOT saved; call
    # `DreamerOpt.make[target, M](model)` BEFORE `load` to size the in-memory
    # optimizer to the model. GPU mirrors are not saved; the trainer must
    # re-upload after load.

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + ".lr=" + String(self.lr) + "\n"
        out += prefix + ".beta1=" + String(self.beta1) + "\n"
        out += prefix + ".beta2=" + String(self.beta2) + "\n"
        out += prefix + ".eps=" + String(self.eps) + "\n"
        out += prefix + ".agc_clip=" + String(self.agc_clip) + "\n"
        out += prefix + ".agc_pmin=" + String(self.agc_pmin) + "\n"
        out += prefix + ".step_count=" + String(self.step_count) + "\n"
        out += prefix + ".beta1_pow_t=" + String(self.beta1_pow_t) + "\n"
        out += prefix + ".beta2_pow_t=" + String(self.beta2_pow_t) + "\n"
        out += prefix + ".nu_flat#size=" + String(self.total_size) + "\n"
        var nu_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.nu_flat.unsafe_ptr()
        )
        for k in range(self.total_size):
            out += String(nu_ptr[k]) + "\n"
        out += prefix + ".mu_flat#size=" + String(self.total_size) + "\n"
        var mu_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.mu_flat.unsafe_ptr()
        )
        for k in range(self.total_size):
            out += String(mu_ptr[k]) + "\n"

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        self.lr = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".lr")))
        idx += 1
        self.beta1 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta1")))
        idx += 1
        self.beta2 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta2")))
        idx += 1
        self.eps = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".eps")))
        idx += 1
        self.agc_clip = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".agc_clip")
        ))
        idx += 1
        self.agc_pmin = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".agc_pmin")
        ))
        idx += 1
        self.step_count = atol(_expect_kv_line(lines, idx, prefix + ".step_count"))
        idx += 1
        self.beta1_pow_t = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".beta1_pow_t")
        ))
        idx += 1
        self.beta2_pow_t = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".beta2_pow_t")
        ))
        idx += 1
        DreamerOpt._load_flat_section(
            lines, idx, prefix + ".nu_flat", self.nu_flat, self.total_size,
        )
        DreamerOpt._load_flat_section(
            lines, idx, prefix + ".mu_flat", self.mu_flat, self.total_size,
        )

    @staticmethod
    def _load_flat_section(
        lines: List[String],
        mut idx: Int,
        expected_prefix: String,
        mut target: List[Scalar[DT]],
        expected_size: Int,
    ) raises:
        if idx >= len(lines):
            raise Error(
                "DreamerOpt.load: out of input at `" + expected_prefix
                + "#size=...` (idx " + String(idx) + ")"
            )
        var header = lines[idx]
        var expected_header = (
            expected_prefix + "#size=" + String(expected_size)
        )
        if header != expected_header:
            raise Error(
                "DreamerOpt.load: section header mismatch. Expected `"
                + expected_header + "`, got `" + header + "`"
            )
        idx += 1
        var t_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            target.unsafe_ptr()
        )
        for k in range(expected_size):
            if idx >= len(lines):
                raise Error(
                    "DreamerOpt.load: short read for `" + expected_prefix
                    + "` at element " + String(k) + " of "
                    + String(expected_size)
                )
            t_ptr[k] = Scalar[DT](atof(lines[idx]))
            idx += 1
