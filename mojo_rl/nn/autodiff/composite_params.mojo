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

from ..constants import dtype, gpu_align
from ..model.model import Model
from layout import LayoutTensor, Layout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
def _cp_align4(x: Int) -> Int:
    """GPU-aligned element count (16-byte aligned for any dtype)."""
    return gpu_align(x)


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

    comptime model_types = Self.MODELS
    comptime N = Self.model_types.size

    @staticmethod
    def _total() -> Int:
        """Total param size with alignment padding between models."""
        var total = 0
        comptime for j in range(Self.N - 1):
            total += _cp_align4(Self.model_types[j].PARAM_SIZE)
        total += Self.model_types[Self.N - 1].PARAM_SIZE
        return total

    comptime TOTAL_SIZE: Int = Self._total()

    @staticmethod
    def offset[idx: Int]() -> Int:
        """Aligned param offset for model idx."""
        var total = 0
        comptime for j in range(idx):
            total += _cp_align4(Self.model_types[j].PARAM_SIZE)
        return total

    @staticmethod
    def assemble[
        dtype: DType = DType.float32
    ](
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
    def scatter[
        dtype: DType = DType.float32
    ](
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
    def scatter_add[
        dtype: DType = DType.float32
    ](
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

    # =====================================================================
    # GPU assembly / scatter
    # =====================================================================

    @staticmethod
    def assemble_gpu[
        dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        *sources: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """GPU: copy N separate param DeviceBuffers into one combined buffer.

        Zeros alignment padding via memset, then launches copy kernels.
        All pointers must be device pointers.
        """
        # Zero the whole buffer (covers alignment padding)
        var dst_buf = DeviceBuffer[dtype](
            ctx, dst, Self.TOTAL_SIZE, owning=False
        )
        ctx.enqueue_memset(dst_buf, 0)

        comptime TPB = 256

        comptime for m in range(Self.N):
            comptime SZ = Self.model_types[m].PARAM_SIZE

            @parameter
            @always_inline
            def _cp_assemble_kernel(
                d: LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= SZ:
                    return
                d[i] = s[i]

            var off = Self.offset[m]()
            var d_t = LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin](
                dst + off
            )
            var s_t = LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin](
                sources[m]
            )

            comptime BLOCKS = (SZ + TPB - 1) // TPB
            ctx.enqueue_function[_cp_assemble_kernel, _cp_assemble_kernel](
                d_t,
                s_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )

    @staticmethod
    def scatter_add_gpu[
        dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        *dsts: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """GPU: add combined grads to N separate DeviceBuffers (accumulate).

        All pointers must be device pointers.
        """
        comptime TPB = 256

        comptime for m in range(Self.N):
            comptime SZ = Self.model_types[m].PARAM_SIZE

            @parameter
            @always_inline
            def _cp_scatter_add_kernel(
                d: LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin],
                s: LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= SZ:
                    return
                d[i] = rebind[Scalar[dtype]](d[i]) + rebind[Scalar[dtype]](s[i])

            var off = Self.offset[m]()
            var d_t = LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin](
                dsts[m]
            )
            var s_t = LayoutTensor[dtype, Layout.row_major(SZ), MutAnyOrigin](
                src + off
            )

            comptime BLOCKS = (SZ + TPB - 1) // TPB
            ctx.enqueue_function[
                _cp_scatter_add_kernel, _cp_scatter_add_kernel
            ](
                d_t,
                s_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
