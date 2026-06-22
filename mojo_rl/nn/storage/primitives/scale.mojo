"""Scale[DIM] — multiplies by a runtime scalar `multiplier` (storage surface).

Transformed from legacy `nn.primitives.Scale` (surface-only change). The CPU SIMD
loop and the GPU `_scale_kernel` are carried over verbatim.

Forward:  out = m·in
Backward: grad_in = m·grad_out

`multiplier` is a public mut field the caller updates per-step (SAC tracks the
moving α this way); `set_multiplier` mirrors the legacy `set_attr["multiplier"]`.
No params, no cache. Conforms to `Module`.

Deferred: the legacy device-resident `multiplier_ptr` / `_scale_dev_kernel`
(SAC on-device α, CUDA-graph capture) is NOT ported yet — not needed by the
current storage call sites.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernel (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _scale_kernel[
    N: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    multiplier: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * multiplier


def _scale_dev_kernel[
    N: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    mptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    # Device-resident multiplier variant — reads the scale factor from
    # `mptr[0]` instead of a baked scalar arg, so the value can be updated by
    # another GPU kernel (SAC's on-device alpha) without breaking CUDA-graph
    # capture. Every thread reads the same `mptr[0]`.
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * mptr[0]


struct Scale[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    @staticmethod
    def display_label() -> String:
        return String("Scale")

    var multiplier: Scalar[DT]
    # Optional device-resident multiplier source. When set (via `set_attr_ptr`),
    # the GPU forward/vjp read the scale factor from `multiplier_ptr[0]` instead
    # of baking `multiplier` into the kernel args — required for CUDA-graph
    # capture when another GPU kernel (SAC's on-device alpha) updates the value
    # each step. None -> baked-scalar path (bit-identical). CPU always uses the
    # host `multiplier`.
    var multiplier_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    def __init__(out self):
        self.multiplier = Scalar[DT](1.0)
        self.multiplier_ptr = None

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        var s = Self()
        return s^

    def set_multiplier(mut self, value: Scalar[DT]):
        self.multiplier = value

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "multiplier":
            self.multiplier = value

    def set_attr_ptr[
        ATTR: StaticString
    ](mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin]):
        comptime if ATTR == "multiplier":
            self.multiplier_ptr = p

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            var k = 0
            while k + CPU_SIMD_W <= N:
                out_p.store(k, in_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                out_p[k] = in_p[k] * self.multiplier
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            if self.multiplier_ptr:
                c.enqueue_function[_scale_dev_kernel[N]](
                    in0.lt["gpu", Layout.row_major(N)](),
                    out.lt["gpu", Layout.row_major(N)](),
                    self.multiplier_ptr.value(),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
            else:
                c.enqueue_function[_scale_kernel[N]](
                    in0.lt["gpu", Layout.row_major(N)](),
                    out.lt["gpu", Layout.row_major(N)](),
                    self.multiplier,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            var k = 0
            while k + CPU_SIMD_W <= N:
                gi_p.store(k, go_p.load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                gi_p[k] = go_p[k] * self.multiplier
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            if self.multiplier_ptr:
                c.enqueue_function[_scale_dev_kernel[N]](
                    grad_output.lt["gpu", Layout.row_major(N)](),
                    gin.lt["gpu", Layout.row_major(N)](),
                    self.multiplier_ptr.value(),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
            else:
                c.enqueue_function[_scale_kernel[N]](
                    grad_output.lt["gpu", Layout.row_major(N)](),
                    gin.lt["gpu", Layout.row_major(N)](),
                    self.multiplier,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
