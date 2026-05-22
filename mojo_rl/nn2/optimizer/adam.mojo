"""Adam optimizer.

`make[target, M]` does not take algorithm-specific hyperparams;
they live as public mut fields on the optimizer struct:

    var opt = Adam.make[target="cpu", M=MyModel](model)
    opt.lr = Scalar[DT](3e-4)   # poke a public field after construction
    opt.step["cpu", M=MyModel](model)

The mut-field pattern lets external schedules (cosine LR, SAC alpha
annealing) update hyperparams per-step without rebuilding the optimizer.

Internals:
  * flat `m_flat`/`v_flat` Lists (CPU) or DeviceBuffers (GPU); offsets
    table maps each param's walk-order index to its start.
  * Visitors reach into Adam's storage via raw pointers (sidesteps
    Mojo's field-destroyed-mid-life rejection).
  * GPU kernel `_adam_update_kernel` — standard fp32 Adam math.
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import ParamVisitor
from ..core.module import Module
from ..core.optimizer import Optimizer
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (lifted from v1).
# ──────────────────────────────────────────────────────────────────────


def _adam_update_kernel(
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < n_elems:
        var one: Scalar[DT] = 1.0
        var g = grad[i]
        var m_new = beta1 * m_off[i] + (one - beta1) * g
        var v_new = beta2 * v_off[i] + (one - beta2) * g * g
        m_off[i] = m_new
        v_off[i] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + eps)


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
struct _AdamCPUInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
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
        self.offsets_ptr[].append(len(self.m_flat_ptr[]))
        for _ in range(n_elems):
            self.m_flat_ptr[].append(zero)
            self.v_flat_ptr[].append(zero)


@fieldwise_init
struct _AdamCPUStepVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
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
        var m_ptr = self.m_flat_ptr[].unsafe_ptr() + off
        var v_ptr = self.v_flat_ptr[].unsafe_ptr() + off
        var b1_v = SIMD[DT, CPU_SIMD_W](self.beta1)
        var b2_v = SIMD[DT, CPU_SIMD_W](self.beta2)
        var omb1_v = SIMD[DT, CPU_SIMD_W](1.0) - b1_v
        var omb2_v = SIMD[DT, CPU_SIMD_W](1.0) - b2_v
        var bc1_v = SIMD[DT, CPU_SIMD_W](self.bias_correction1)
        var bc2_v = SIMD[DT, CPU_SIMD_W](self.bias_correction2)
        var lr_v = SIMD[DT, CPU_SIMD_W](self.lr)
        var eps_v = SIMD[DT, CPU_SIMD_W](self.eps)
        var i = 0
        while i + CPU_SIMD_W <= n_elems:
            var g = g_ptr.load[width=CPU_SIMD_W](i)
            var m_old = m_ptr.load[width=CPU_SIMD_W](i)
            var v_old = v_ptr.load[width=CPU_SIMD_W](i)
            var m_new = b1_v * m_old + omb1_v * g
            var v_new = b2_v * v_old + omb2_v * g * g
            m_ptr.store(i, m_new)
            v_ptr.store(i, v_new)
            var m_hat = m_new / bc1_v
            var v_hat = v_new / bc2_v
            var p = p_ptr.load[width=CPU_SIMD_W](i)
            p_ptr.store(i, p - lr_v * m_hat / (sqrt(v_hat) + eps_v))
            i += CPU_SIMD_W
        var one: Scalar[DT] = 1.0
        while i < n_elems:
            var g = g_ptr[i]
            var m_old = m_ptr[i]
            var v_old = v_ptr[i]
            var m_new = self.beta1 * m_old + (one - self.beta1) * g
            var v_new = self.beta2 * v_old + (one - self.beta2) * g * g
            m_ptr[i] = m_new
            v_ptr[i] = v_new
            var m_hat = m_new / self.bias_correction1
            var v_hat = v_new / self.bias_correction2
            p_ptr[i] = p_ptr[i] - self.lr * m_hat / (sqrt(v_hat) + self.eps)
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
struct _AdamGPUInitVisitor(ParamVisitor):
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
struct _AdamGPUStepVisitor(ParamVisitor):
    var ctx: DeviceContext
    var m_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
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
        var m_off = self.m_base + off
        var v_off = self.v_base + off
        var param_w_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var grad_w_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adam_update_kernel](
            param_w_ptr, grad_w_ptr, m_off, v_off, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
            self.bias_correction1, self.bias_correction2,
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
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# Adam.
# ──────────────────────────────────────────────────────────────────────


struct Adam(Optimizer):
    # CPU storage
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    # GPU storage
    var m_dev: Optional[DeviceBuffer[DT]]
    var v_dev: Optional[DeviceBuffer[DT]]
    # Shared bookkeeping
    var offsets: List[Int]
    var total_size: Int
    var step_count: Int

    # Public mut hyperparams. Defaults match Kingma & Ba 2014.
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    var ts: TargetStorage

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.m_dev = None
        self.v_dev = None
        self.offsets = List[Int]()
        self.total_size = 0
        self.step_count = 0
        self.lr = Scalar[DT](0.001)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, M: Module](mut model: M) raises -> Self:
        """CPU factory — no hyperparams. User sets `opt.lr` etc. after."""
        comptime assert target == "cpu", (
            "Adam.make[target='gpu', M] requires a DeviceContext"
        )
        var opt = Self()
        var visitor = _AdamCPUInitVisitor(
            m_flat_ptr=UnsafePointer(to=opt.m_flat),
            v_flat_ptr=UnsafePointer(to=opt.v_flat),
            offsets_ptr=UnsafePointer(to=opt.offsets),
        )
        model.for_each_param[target, _AdamCPUInitVisitor](String(""), visitor)
        opt.total_size = len(opt.m_flat)
        opt.ts = TargetStorage.make_cpu()
        return opt^

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M, ctx: DeviceContext,
    ) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "Adam.make[target='cpu', M](model, ctx) — drop ctx for CPU"
        )
        var opt = Self()
        var visitor = _AdamGPUInitVisitor(
            offsets_ptr=UnsafePointer(to=opt.offsets),
            total_ptr=UnsafePointer(to=opt.total_size),
        )
        model.for_each_param[target, _AdamGPUInitVisitor](String(""), visitor)
        var m_real = ctx.enqueue_create_buffer[DT](opt.total_size)
        var v_real = ctx.enqueue_create_buffer[DT](opt.total_size)
        m_real.enqueue_fill(0.0)
        v_real.enqueue_fill(0.0)
        opt.m_dev = m_real^
        opt.v_dev = v_real^
        opt.ts = TargetStorage.make_gpu(ctx)
        return opt^

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["Adam", target](self.ts.target_tag)
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
        assert_tag_for["Adam", target](self.ts.target_tag)
        self.step_count += 1
        self.beta1_pow_t = self.beta1_pow_t * self.beta1
        self.beta2_pow_t = self.beta2_pow_t * self.beta2
        var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
        var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t

        comptime if target == "cpu":
            var visitor = _AdamCPUStepVisitor(
                m_flat_ptr=UnsafePointer(to=self.m_flat),
                v_flat_ptr=UnsafePointer(to=self.v_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target, _AdamCPUStepVisitor](String(""), visitor)
        else:
            var visitor = _AdamGPUStepVisitor(
                ctx=self.ts.ctx.value(),
                m_base=self.m_dev.value().unsafe_ptr(),
                v_base=self.v_dev.value().unsafe_ptr(),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target, _AdamGPUStepVisitor](String(""), visitor)
