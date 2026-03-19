"""CompositeParams — auto-aligned param assembly/scatter for multi-model compositions.

Reduces ~60 lines of manual param assembly + gradient scatter boilerplate
to a single assemble() + scatter() call per algorithm.

Usage:
    from mojo_rl.nn.autodiff.composite_params import CompositeParams

    # SAC: actor + critic1 + critic2
    comptime Params = CompositeParams[ActorModel, CriticModel, CriticModel]

    # Assembly: separate buffers → combined
    var combined = InlineArray[Scalar[dtype], Params.TOTAL_SIZE](uninitialized=True)
    Params.assemble(combined.unsafe_ptr(), actor_params.ptr, critic1_params.ptr, critic2_params.ptr)

    # ... forward + backward through composed graph ...

    # Scatter: combined grads → separate buffers
    Params.scatter(combined_grads.unsafe_ptr(), actor_grads.ptr, critic1_grads.ptr, critic2_grads.ptr)

    # Or scatter with accumulation (+=):
    Params.scatter_add(combined_grads.unsafe_ptr(), actor_grads.ptr, critic1_grads.ptr, critic2_grads.ptr)

The alignment padding between models matches Sequential/DualPath conventions
(4-element = 16-byte boundary for GPU matmul safety).
"""

from ..constants import dtype
from ..model.model import Model
from layout import LayoutTensor, Layout
from std.builtin.variadics import Variadic


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
fn _cp_align4(x: Int) -> Int:
    """Round up to next multiple of 4 for GPU alignment."""
    return (x + 3) & ~3


struct CompositeParams[*MODELS: Model]:
    """Auto-aligned param layout for multi-model compositions.

    Compile-time constants:
        N: Number of models
        TOTAL_SIZE: Total param buffer size (with alignment padding)

    Methods:
        offset[idx]() → Int: Aligned byte offset for model idx
        assemble(...): Copy N separate param buffers into one combined buffer
        scatter(...): Copy combined grads back to N separate buffers (overwrite)
        scatter_add(...): Add combined grads to N separate buffers (accumulate)
    """

    comptime model_types = Variadic.types[T=Model, *Self.MODELS]
    comptime N = Variadic.size(Self.model_types)

    @staticmethod
    fn _total() -> Int:
        """Total param size with alignment padding between models."""
        var total = 0
        comptime for j in range(Self.N - 1):
            total += _cp_align4(Self.model_types[j].PARAM_SIZE)
        total += Self.model_types[Self.N - 1].PARAM_SIZE
        return total

    comptime TOTAL_SIZE: Int = Self._total()

    @staticmethod
    fn offset[idx: Int]() -> Int:
        """Aligned param offset for model idx."""
        var total = 0
        comptime for j in range(idx):
            total += _cp_align4(Self.model_types[j].PARAM_SIZE)
        return total

    @staticmethod
    fn assemble(
        dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        *sources: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Copy N separate param buffers into one combined buffer.

        Zeros the entire buffer first (covers alignment padding).
        """
        for i in range(Self.TOTAL_SIZE):
            dst[i] = Scalar[dtype](0.0)

        comptime for m in range(Self.N):
            var off = Self.offset[m]()
            var sz = Self.model_types[m].PARAM_SIZE
            var src = sources[m]
            for i in range(sz):
                dst[off + i] = src[i]

    @staticmethod
    fn scatter(
        src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        *dsts: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Copy combined grads back to N separate buffers (overwrite)."""
        comptime for m in range(Self.N):
            var off = Self.offset[m]()
            var sz = Self.model_types[m].PARAM_SIZE
            var d = dsts[m]
            for i in range(sz):
                d[i] = src[off + i]

    @staticmethod
    fn scatter_add(
        src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        *dsts: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Add combined grads to N separate buffers (accumulate)."""
        comptime for m in range(Self.N):
            var off = Self.offset[m]()
            var sz = Self.model_types[m].PARAM_SIZE
            var d = dsts[m]
            for i in range(sz):
                d[i] = d[i] + src[off + i]
