"""SGD optimizer — momentum + L2-coupled weight decay (PyTorch semantics).

The optimizer EfficientZero-V2 uses for Atari (`torch.optim.SGD`, lr 0.2,
momentum 0.9, weight_decay 1e-4, no nesterov). nn2 previously had only
Adam/AdamW/Dreamer; this fills the gap for the EZv2-Atari parity port
(`docs/EZV2_ATARI_PARITY.md` §F).

Update rule (PyTorch `SGD`, nesterov=False), per element:

    d_p = grad
    if weight_decay: d_p = d_p + weight_decay * param      # L2-COUPLED (NOT
                                                           # AdamW's decoupled)
    vel = momentum * vel + d_p                              # vel starts at 0
    param = param - lr * vel

`lr` / `momentum` / `weight_decay` / `max_grad_norm` are public mut fields (an
external LR schedule pokes `lr`; the warmup→constant schedule lives in the
driver). One velocity buffer per parameter (`STATE_PER_PARAM=1`, vs Adam's two).
Global grad-norm clip mirrors AdamW (EZv2 uses `max_grad_norm=5`); `0.0`
(default) disables it. Per-tensor GPU path (works on Apple + NVIDIA; the
grouped multi-tensor launch is a future perf follow-up).

The per-param `apply_decay` flag is collected at init from each Param's `decay`
bit (Linear weight=True / bias=False; BN/LN γ,β=False), so weight decay only
touches the weights — matching the reference's parameter-group split.
"""

from std.sys import has_nvidia_gpu_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import ParamVisitor
from ..core.module import Module, mptr
from ..core.optimizer import Optimizer
from ..core.grad_clip import clip_grads_auto, clip_grads_auto_gpu, GradClipState
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _sgd_update_kernel(
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    vel_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
    lr: Scalar[DT],
    momentum: Scalar[DT],
    weight_decay: Scalar[DT],
    apply_decay: Int,
):
    var i = Int(global_idx.x)
    if i < n_elems:
        var p = param[i]
        var d_p = grad[i]
        if apply_decay != 0:
            d_p = d_p + weight_decay * p
        var v_new = momentum * vel_off[i] + d_p
        vel_off[i] = v_new
        param[i] = p - lr * v_new


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
struct _SGDCPUInitVisitor(ParamVisitor):
    var vel_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]

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
        self.offsets_ptr[].append(len(self.vel_flat_ptr[]))
        self.apply_decay_ptr[].append(apply_decay)
        for _ in range(n_elems):
            self.vel_flat_ptr[].append(zero)


@fieldwise_init
struct _SGDCPUStepVisitor(ParamVisitor):
    var vel_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var momentum: Scalar[DT]
    var weight_decay: Scalar[DT]

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
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var p_ptr = mptr(param.ptr)
        var g_ptr = mptr(grad.ptr)
        var vel_ptr = self.vel_flat_ptr[].unsafe_ptr() + off
        var mom_v = SIMD[DT, CPU_SIMD_W](self.momentum)
        var lr_v = SIMD[DT, CPU_SIMD_W](self.lr)
        var wd_scalar: Scalar[DT] = self.weight_decay if decay_flag else 0.0
        var wd_v = SIMD[DT, CPU_SIMD_W](wd_scalar)
        var i = 0
        while i + CPU_SIMD_W <= n_elems:
            var p = p_ptr.load[width=CPU_SIMD_W](i)
            var d_p = g_ptr.load[width=CPU_SIMD_W](i) + wd_v * p
            var v_new = mom_v * vel_ptr.load[width=CPU_SIMD_W](i) + d_p
            vel_ptr.store(i, v_new)
            p_ptr.store(i, p - lr_v * v_new)
            i += CPU_SIMD_W
        while i < n_elems:
            var p = p_ptr[i]
            var d_p = g_ptr[i] + wd_scalar * p
            var v_new = self.momentum * vel_ptr[i] + d_p
            vel_ptr[i] = v_new
            p_ptr[i] = p - self.lr * v_new
            i += 1
        self.idx += 1


@fieldwise_init
struct _ZeroGradCPUVisitor(ParamVisitor):
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
        var g_ptr = mptr(grad.ptr)
        for i in range(n_elems):
            g_ptr[i] = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# GPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _SGDGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
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
        self.apply_decay_ptr[].append(apply_decay)
        self.total_ptr[] += n_elems


@fieldwise_init
struct _SGDGPUStepVisitor(ParamVisitor):
    var ctx: DeviceContext
    var vel_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var momentum: Scalar[DT]
    var weight_decay: Scalar[DT]

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
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var vel_off = self.vel_base + off
        var param_w_ptr = mptr(param.ptr)
        var grad_w_ptr = mptr(grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_sgd_update_kernel](
            param_w_ptr, grad_w_ptr, vel_off, n_elems,
            self.lr, self.momentum, self.weight_decay,
            Int(1) if decay_flag else Int(0),
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
        var grad_w_ptr = mptr(grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# SGD.
# ──────────────────────────────────────────────────────────────────────


struct SGD(Optimizer):
    # CPU storage
    var vel_flat: List[Scalar[DT]]
    # GPU storage
    var vel_dev: Optional[DeviceBuffer[DT]]
    # Shared
    var offsets: List[Int]
    var apply_decay: List[Bool]
    var total_size: Int
    # Public mut hyperparameters.
    var lr: Scalar[DT]
    var momentum: Scalar[DT]
    var weight_decay: Scalar[DT]
    # Global L2 grad-norm clip threshold; 0.0 disables (bit-identical to no
    # clip). EZv2 sets 5.0. Lazy-allocated GPU clip state on first GPU step.
    var max_grad_norm: Scalar[DT]
    var _clip_state: Optional[GradClipState]

    var ts: TargetStorage

    def __init__(out self):
        self.vel_flat = List[Scalar[DT]]()
        self.vel_dev = None
        self.offsets = List[Int]()
        self.apply_decay = List[Bool]()
        self.total_size = 0
        self.lr = Scalar[DT](0.2)
        self.momentum = Scalar[DT](0.9)
        self.weight_decay = Scalar[DT](1e-4)
        self.max_grad_norm = Scalar[DT](0.0)
        self._clip_state = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "SGD: target must be 'cpu' or 'gpu'"
        )
        var opt = Self()
        comptime if target == "cpu":
            var visitor = _SGDCPUInitVisitor(
                vel_flat_ptr=UnsafePointer(to=opt.vel_flat),
                offsets_ptr=UnsafePointer(to=opt.offsets),
                apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
            )
            model.for_each_param[target, _SGDCPUInitVisitor](String(""), visitor)
            opt.total_size = len(opt.vel_flat)
            opt.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SGD.make[target='gpu']"](ctx)
            var visitor = _SGDGPUInitVisitor(
                offsets_ptr=UnsafePointer(to=opt.offsets),
                apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
                total_ptr=UnsafePointer(to=opt.total_size),
            )
            model.for_each_param[target, _SGDGPUInitVisitor](String(""), visitor)
            var vel_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            vel_real.enqueue_fill(0.0)
            opt.vel_dev = vel_real^
            opt.ts = TargetStorage.make_gpu(ctx_v)
        return opt^

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["SGD", target](self.ts.target_tag)
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
        assert_tag_for["SGD", target](self.ts.target_tag)

        # Global grad-norm clip (no-op when max_grad_norm == 0).
        comptime if target == "cpu":
            _ = clip_grads_auto[M, target](model, self.max_grad_norm)
        else:
            if self.max_grad_norm > Scalar[DT](0.0):
                if not self._clip_state:
                    self._clip_state = GradClipState.make(
                        self.ts.ctx.value(), len(self.offsets),
                    )
                clip_grads_auto_gpu[M](
                    model, self.ts.ctx.value(), self._clip_state.value(),
                    self.max_grad_norm,
                )

        comptime if target == "cpu":
            var visitor = _SGDCPUStepVisitor(
                vel_flat_ptr=UnsafePointer(to=self.vel_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                apply_decay_ptr=UnsafePointer(to=self.apply_decay),
                idx=0,
                lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            model.for_each_param[target, _SGDCPUStepVisitor](String(""), visitor)
        else:
            var visitor = _SGDGPUStepVisitor(
                ctx=self.ts.ctx.value(),
                vel_base=self.vel_dev.value().unsafe_ptr(),
                offsets_ptr=UnsafePointer(to=self.offsets),
                apply_decay_ptr=UnsafePointer(to=self.apply_decay),
                idx=0,
                lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            model.for_each_param[target, _SGDGPUStepVisitor](String(""), visitor)
