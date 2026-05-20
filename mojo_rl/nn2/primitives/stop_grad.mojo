"""StopGrad[DIM] — identity forward, zero-fill backward. Phase 5.2.

Severs the gradient chain at this point in the network. Used to freeze
inputs to a loss graph (advantage normalization, PPO old log-probs,
target Q for DQN/SAC/TD3, MuZero/AlphaZero targets) without duplicating
the producer network.

Forward:  output[b, d] = input[b, d]      (memcpy)
Backward: grad_input[b, d] = 0             (severs upstream)

No cache. No parameters. Element-wise — POLICY ignored (any compute
dtype just copies bits; the zero-fill is dtype-uniform).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


def _stop_grad_forward_kernel[
    BATCH: Int,
    DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        output[b, d] = rebind[Scalar[DT]](input[b, d])


def _stop_grad_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        var zero: Scalar[DT] = 0.0
        grad_input[b, d] = zero


# ──────────────────────────────────────────────────────────────────────────
# StopGrad — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct StopGrad[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var ctx: Optional[DeviceContext]
    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT ignored (no params)."""
        comptime assert (
            target == "cpu"
        ), "StopGrad.make[target='gpu', INIT] requires a DeviceContext"
        var s = Self()
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "StopGrad.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var s = Self()
        s.ctx = ctx
        s._target_tag = TARGET_GPU
        return s^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "StopGrad: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag="
                + String(Int(self._target_tag))
                + ")"
            )

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert (
            input.flat_rank == 2
        ), "input must be rank-2 [BATCH, DIM]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = input[b, d]
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var input_w  = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var input_lt  = LayoutTensor[DT, layout, MutAnyOrigin](input_w.ptr)
            var output_lt = LayoutTensor[DT, layout, MutAnyOrigin](output_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _stop_grad_forward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                input_lt,
                output_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        # grad_output is discarded — that's the whole point of stop_grad.
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var zero: Scalar[DT] = 0.0
            for b in range(BATCH):
                for d in range(Self.DIM):
                    grad_input[b, d] = zero
        else:
            comptime layout = Layout.row_major(BATCH, Self.DIM)
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](grad_input_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _stop_grad_backward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                gi_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        # No params — backward_input is identical to backward (still zeros grad_input).
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        # StopGrad has no parameters — nothing to visit.
        pass

    def set_inference(mut self, value: Bool):
        # StopGrad behavior is identical in train and eval — flag stored
        # for trait conformance but has no behavioral effect.
        self._inference = value
