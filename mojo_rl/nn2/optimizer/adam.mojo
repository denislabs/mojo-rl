"""Adam — Adam optimizer (Kingma & Ba 2014). Phase 2.4: method-level target.

Same algorithm + storage layout on both targets:
  - flat `m`, `v` buffers concatenating all params in walk-order
  - `offsets` table giving the start index of each param's slice

`make[target]` allocates the matching storage and stamps `_target_tag`.
`step[target]` and `zero_grad[target]` branch on the comptime target and
pass it through to `model.for_each_param[target]`.

Visitors borrow pointers to Adam's storage (sidesteps Mojo's "field
destroyed mid-life" rejection).
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Optimizer,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# CPU visitors
# ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _AdamCPUInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
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

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var p_ptr = param.ptr
        var g_ptr = grad.ptr
        var one: Scalar[DT] = 1.0
        for i in range(n_elems):
            var g = g_ptr[i]
            var m_old = self.m_flat_ptr[][off + i]
            var v_old = self.v_flat_ptr[][off + i]
            var m_new = self.beta1 * m_old + (one - self.beta1) * g
            var v_new = self.beta2 * v_old + (one - self.beta2) * g * g
            self.m_flat_ptr[][off + i] = m_new
            self.v_flat_ptr[][off + i] = v_new
            var m_hat = m_new / self.bias_correction1
            var v_hat = v_new / self.bias_correction2
            p_ptr[i] = p_ptr[i] - self.lr * m_hat / (sqrt(v_hat) + self.eps)
        self.idx += 1


struct _ZeroGradCPUVisitor(ParamVisitor):
    def __init__(out self):
        pass

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ) raises:
        var g_ptr = grad.ptr
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────────
# GPU visitors
# ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _AdamGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var total_ptr: UnsafePointer[Int, MutAnyOrigin]

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
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

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var m_off = self.m_base + off
        var v_off = self.v_base + off
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adam_update_kernel](
            param.ptr, grad.ptr, m_off, v_off, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
            self.bias_correction1, self.bias_correction2,
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.idx += 1


@fieldwise_init
struct _ZeroGradGPUVisitor(ParamVisitor):
    var ctx: DeviceContext

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ) raises:
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad.ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────────
# Adam — method-level target.
# ──────────────────────────────────────────────────────────────────────────

struct Adam(Optimizer):
    # CPU storage
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    # GPU storage
    var m_dev: Optional[DeviceBuffer[DT]]
    var v_dev: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]
    # Shared
    var offsets: List[Int]
    var total_size: Int
    var step_count: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]
    var _target_tag: Int8

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.m_dev = None
        self.v_dev = None
        self.ctx = None
        self.offsets = List[Int]()
        self.total_size = 0
        self.step_count = 0
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.beta1_pow_t = 1.0
        self.beta2_pow_t = 1.0
        self._target_tag = TARGET_UNINIT

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) raises -> Self:
        """CPU factory. Walks the model to size the flat m/v buffers."""
        comptime assert target == "cpu", (
            "Adam.make[target='gpu'] requires a DeviceContext"
        )
        var adam = Self()
        adam.lr = lr; adam.beta1 = beta1; adam.beta2 = beta2; adam.eps = eps
        var visitor = _AdamCPUInitVisitor(
            m_flat_ptr=UnsafePointer(to=adam.m_flat),
            v_flat_ptr=UnsafePointer(to=adam.v_flat),
            offsets_ptr=UnsafePointer(to=adam.offsets),
        )
        model.for_each_param[target](String(""), visitor)
        adam.total_size = len(adam.m_flat)
        adam._target_tag = TARGET_CPU
        return adam^

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: DeviceContext,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "Adam.make[target='cpu'](model, ctx) — drop ctx for CPU"
        )
        var adam = Self()
        adam.lr = lr; adam.beta1 = beta1; adam.beta2 = beta2; adam.eps = eps
        adam.ctx = ctx
        # First pass: compute total_size + offsets via init visitor.
        var visitor = _AdamGPUInitVisitor(
            offsets_ptr=UnsafePointer(to=adam.offsets),
            total_ptr=UnsafePointer(to=adam.total_size),
        )
        model.for_each_param[target](String(""), visitor)
        var m_real = ctx.enqueue_create_buffer[DT](adam.total_size)
        var v_real = ctx.enqueue_create_buffer[DT](adam.total_size)
        m_real.enqueue_fill(0.0)
        v_real.enqueue_fill(0.0)
        adam.m_dev = m_real^
        adam.v_dev = v_real^
        adam._target_tag = TARGET_GPU
        return adam^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Adam: method called with [target='" + String(target)
                + "'] but optimizer was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        self._assert_tag[target]()
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target](String(""), v)
        else:
            var v = _ZeroGradGPUVisitor(ctx=self.ctx.value())
            model.for_each_param[target](String(""), v)

    def step[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        self._assert_tag[target]()
        self.step_count += 1
        self.beta1_pow_t = self.beta1_pow_t * self.beta1
        self.beta2_pow_t = self.beta2_pow_t * self.beta2
        var bc1: Scalar[DT] = 1.0 - self.beta1_pow_t
        var bc2: Scalar[DT] = 1.0 - self.beta2_pow_t

        comptime if target == "cpu":
            var visitor = _AdamCPUStepVisitor(
                m_flat_ptr=UnsafePointer(to=self.m_flat),
                v_flat_ptr=UnsafePointer(to=self.v_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target](String(""), visitor)
        else:
            var visitor = _AdamGPUStepVisitor(
                ctx=self.ctx.value(),
                m_base=self.m_dev.value().unsafe_ptr(),
                v_base=self.v_dev.value().unsafe_ptr(),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target](String(""), visitor)
