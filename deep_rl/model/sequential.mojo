from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

# =============================================================================
# Sequential Containers
# =============================================================================
#
# NOTE: A truly variadic Sequential[*Types: Model] is not yet possible in Mojo
# because:
# 1. InlineArray[Model, N] can't work - Model is a trait, not a concrete type
#    with known size. Each layer (Linear, ReLU, etc.) has different memory layout.
# 2. Mojo doesn't support storing heterogeneous variadic types as struct fields.
#    There's no Tuple[*Types] or similar construct for variadic packs.
# 3. PARAM_SIZE would need compile-time sum across variadic types.
#
# SOLUTION: We define only Seq2 and compose larger sequences by nesting:
#   - Seq3 = Seq2[Seq2[L0, L1], L2]
#   - Seq4 = Seq2[Seq2[Seq2[L0, L1], L2], L3]
#   - etc.
#
# Since Seq2 implements Model, it can be nested indefinitely. The seq() helper
# functions handle the composition automatically:
#
#     var model = seq(Linear[2, 16, 4](), ReLU[16, 4](), Linear[16, 1, 4]())
#
# Benefits:
#   - Only one Sequential struct to maintain
#   - PARAM_SIZE composes correctly: Seq2[A, B].PARAM_SIZE = A + B
#   - Dimensions chain correctly: IN_DIM = first.IN_DIM, OUT_DIM = last.OUT_DIM
#   - Full compile-time type safety preserved
# =============================================================================


struct Seq2[L0: Model, L1: Model](Model):
    """Sequential container for 2 layers.

    Composes two modules where L0.OUT_DIM == L1.IN_DIM (enforced at instantiation).

    Cache layout: [L0's cache | L1's cache]
    CACHE_SIZE = L0.CACHE_SIZE + L1.CACHE_SIZE

    Workspace layout (GPU only): [inter_buf | L0's workspace | L1's workspace]
    - inter_buf: L0.OUT_DIM elements per sample (holds L0 output / L1 input)
    - Reused for both forward (intermediate activation) and backward (gradient intermediate)
    WORKSPACE_SIZE_PER_SAMPLE = L0.OUT_DIM + L0.WORKSPACE_SIZE_PER_SAMPLE + L1.WORKSPACE_SIZE_PER_SAMPLE

    Usage:
        var model = seq(Linear[2, 16](), ReLU[16]())
        # or with explicit types:
        var model = Seq2[Linear[2, 16], ReLU[16]](layer0, layer1)
    """

    comptime IN_DIM: Int = Self.L0.IN_DIM
    comptime OUT_DIM: Int = Self.L1.OUT_DIM
    comptime PARAM_SIZE: Int = Self.L0.PARAM_SIZE + Self.L1.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.L0.CACHE_SIZE + Self.L1.CACHE_SIZE
    # Workspace: intermediate buffer + nested workspaces
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.L0.OUT_DIM
        + Self.L0.WORKSPACE_SIZE_PER_SAMPLE
        + Self.L1.WORKSPACE_SIZE_PER_SAMPLE
    )

    var layer0: Self.L0
    var layer1: Self.L1

    fn __init__(out self, var l0: Self.L0, var l1: Self.L1):
        """Initialize with two layers."""
        self.layer0 = l0^
        self.layer1 = l1^

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.layer0 = other.layer0^
        self.layer1 = other.layer1^

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Copyable trait."""
        self.layer0 = other.layer0.copy()
        self.layer1 = other.layer1.copy()

    fn forward[
        BATCH: Int
    ](
        self,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass through all layers (all zero-copy views).

        Cache layout (blocked): [L0's cache | L1's cache]
        Params layout: [L0's params | L1's params]
        """
        # Intermediate buffer for L0 output / L1 input
        var buffer0_storage = InlineArray[
            Scalar[dtype], BATCH * Self.L0.OUT_DIM
        ](uninitialized=True)
        var buffer0 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.OUT_DIM), MutAnyOrigin
        ](buffer0_storage)

        # Create zero-copy views using pointer offsets
        var params_ptr = params.ptr
        var l1_params_ptr = params_ptr + Self.L0.PARAM_SIZE
        var l0_params = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](l1_params_ptr)

        var cache_ptr = cache.ptr
        var l0_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr)
        var l1_cache_ptr = cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.CACHE_SIZE), MutAnyOrigin
        ](l1_cache_ptr)

        # L0: input -> buffer0
        self.layer0.forward[BATCH](input, buffer0, l0_params, l0_cache)

        # L1: buffer0 -> output
        # Rebind buffer0 to L1's input type (dimensions match since L0.OUT_DIM == L1.IN_DIM)
        var l1_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.IN_DIM), MutAnyOrigin
        ](buffer0.ptr)
        self.layer1.forward[BATCH](l1_input, output, l1_params, l1_cache)

    fn forward[
        BATCH: Int
    ](
        self,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (for inference)."""
        # Intermediate buffer for L0 output / L1 input
        var buffer0_storage = InlineArray[
            Scalar[dtype], BATCH * Self.L0.OUT_DIM
        ](uninitialized=True)
        var buffer0 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.OUT_DIM), MutAnyOrigin
        ](buffer0_storage)

        # Create zero-copy views using pointer offsets
        var params_ptr = params.ptr
        var l0_params = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)
        var l1_params_ptr = params_ptr + Self.L0.PARAM_SIZE
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](l1_params_ptr)

        # L0: input -> buffer0
        self.layer0.forward[BATCH](input, buffer0, l0_params)

        # L1: buffer0 -> output
        var l1_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.IN_DIM), MutAnyOrigin
        ](buffer0.ptr)
        self.layer1.forward[BATCH](l1_input, output, l1_params)

    fn backward[
        BATCH: Int
    ](
        self,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass in reverse order (all zero-copy views).

        Cache layout (blocked): [L0's cache | L1's cache]
        Params layout: [L0's params | L1's params]
        Grads layout: [L0's grads | L1's grads]
        """
        # Create zero-copy views using pointer offsets
        var params_ptr = params.ptr
        var l0_params = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)
        var l1_params_ptr = params_ptr + Self.L0.PARAM_SIZE
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](l1_params_ptr)

        var grads_ptr = grads.ptr
        var l0_grads = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](grads_ptr)
        var l1_grads_ptr = grads_ptr + Self.L0.PARAM_SIZE
        var l1_grads = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](l1_grads_ptr)

        var cache_ptr = cache.ptr
        var l0_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr)
        var l1_cache_ptr = cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.CACHE_SIZE), MutAnyOrigin
        ](l1_cache_ptr)

        # Intermediate buffer for gradient at buffer0
        var grad_buffer0_storage = InlineArray[
            Scalar[dtype], BATCH * Self.L0.OUT_DIM
        ](uninitialized=True)
        var grad_buffer0 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.OUT_DIM), MutAnyOrigin
        ](grad_buffer0_storage)

        # L1 backward: grad_output -> grad_buffer0
        # Rebind grad_output to L1's output type
        var l1_grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.OUT_DIM), MutAnyOrigin
        ](grad_output.ptr)
        var l1_grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.IN_DIM), MutAnyOrigin
        ](grad_buffer0.ptr)
        self.layer1.backward[BATCH](
            l1_grad_output,
            l1_grad_input,
            l1_params,
            l1_cache,
            l1_grads,
        )

        # L0 backward: grad_buffer0 -> grad_input
        self.layer0.backward[BATCH](
            grad_buffer0,
            grad_input,
            l0_params,
            l0_cache,
            l0_grads,
        )
        # No copy-back needed - grads views modify the original in-place

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================
    #
    # Sequential orchestrates GPU calls for its child layers.
    # It allocates intermediate buffers on GPU and calls child _gpu methods.
    #
    # Buffer layouts:
    # - params_buf: [L0's params | L1's params]
    # - cache_buf: [L0's cache | L1's cache]
    # - grads_buf: [L0's grads | L1's grads]
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching.

        Orchestrates GPU calls for child layers:
        1. Create intermediate buffer for L0 output / L1 input
        2. L0: input -> intermediate
        3. L1: intermediate -> output

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] = [L0 cache | L1 cache].
        """
        # Allocate intermediate buffer on GPU for L0 output / L1 input
        var inter_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.L0.OUT_DIM
        )

        # Create views into params_buf for each layer (owning=False for non-owning views)
        # L0 params: offset 0, size L0.PARAM_SIZE
        # L1 params: offset L0.PARAM_SIZE, size L1.PARAM_SIZE
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # Create views into cache_buf for each layer
        # L0 cache: offset 0, size BATCH * L0.CACHE_SIZE
        # L1 cache: offset BATCH * L0.CACHE_SIZE, size BATCH * L1.CACHE_SIZE
        var l0_cache_ptr = cache_buf.unsafe_ptr()
        var l0_cache_buf = DeviceBuffer[dtype](
            ctx,
            l0_cache_ptr,
            BATCH * Self.L0.CACHE_SIZE,
            owning=False,
        )
        var l1_cache_ptr = l0_cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache_buf = DeviceBuffer[dtype](
            ctx,
            l1_cache_ptr,
            BATCH * Self.L1.CACHE_SIZE,
            owning=False,
        )

        # L0: input -> inter_buf
        Self.L0.forward_gpu[BATCH](
            ctx,
            inter_buf,
            input_buf,
            l0_params_buf,
            l0_cache_buf,
        )

        # L1: inter_buf -> output
        Self.L1.forward_gpu[BATCH](
            ctx,
            output_buf,
            inter_buf,
            l1_params_buf,
            l1_cache_buf,
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU without caching (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
        """
        # Allocate intermediate buffer on GPU
        var inter_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.L0.OUT_DIM
        )

        # Create views into params_buf (owning=False for non-owning views)
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # L0: input -> inter_buf
        Self.L0.forward_gpu_no_cache[BATCH](
            ctx,
            inter_buf,
            input_buf,
            l0_params_buf,
        )

        # L1: inter_buf -> output
        Self.L1.forward_gpu_no_cache[BATCH](
            ctx,
            output_buf,
            inter_buf,
            l1_params_buf,
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch backward pass on GPU.

        Orchestrates GPU calls for child layers in reverse order:
        1. L1: grad_output -> grad_inter
        2. L0: grad_inter -> grad_input

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient w.r.t. input [BATCH * IN_DIM] (written).
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] = [L0 cache | L1 cache].
            grads_buf: Parameter gradients [PARAM_SIZE] = [L0 grads | L1 grads] (written).
        """
        # Allocate intermediate gradient buffer on GPU
        var grad_inter_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.L0.OUT_DIM
        )

        # Create views into params_buf (owning=False for non-owning views)
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # Create views into cache_buf
        var l0_cache_ptr = cache_buf.unsafe_ptr()
        var l0_cache_buf = DeviceBuffer[dtype](
            ctx,
            cache_buf.unsafe_ptr(),
            BATCH * Self.L0.CACHE_SIZE,
            owning=False,
        )
        var l1_cache_ptr = l0_cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache_buf = DeviceBuffer[dtype](
            ctx,
            l1_cache_ptr,
            BATCH * Self.L1.CACHE_SIZE,
            owning=False,
        )

        # Create views into grads_buf
        var l0_grads_ptr = grads_buf.unsafe_ptr()
        var l0_grads_buf = DeviceBuffer[dtype](
            ctx,
            l0_grads_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_grads_ptr = l0_grads_ptr + Self.L0.PARAM_SIZE
        var l1_grads_buf = DeviceBuffer[dtype](
            ctx,
            l1_grads_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # L1 backward: grad_output -> grad_inter
        Self.L1.backward_gpu[BATCH](
            ctx,
            grad_inter_buf,
            grad_output_buf,
            l1_params_buf,
            l1_cache_buf,
            l1_grads_buf,
        )

        # L0 backward: grad_inter -> grad_input
        Self.L0.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_inter_buf,
            l0_params_buf,
            l0_cache_buf,
            l0_grads_buf,
        )

    # =========================================================================
    # GPU Methods with Pre-allocated Workspace (avoids internal allocation)
    # =========================================================================
    #
    # These methods use pre-allocated workspace buffers instead of allocating
    # intermediate buffers internally. This is much faster for repeated calls.
    #
    # Workspace layout: [inter_buf | L0's workspace | L1's workspace]
    # Total size: BATCH * WORKSPACE_SIZE_PER_SAMPLE
    # =========================================================================

    @staticmethod
    fn forward_gpu_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass using pre-allocated workspace (no internal allocation).

        Workspace layout: [inter_buf (BATCH * L0.OUT_DIM) | L0 workspace | L1 workspace]

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] = [L0 cache | L1 cache].
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        # Compute workspace offsets
        comptime INTER_SIZE = BATCH * Self.L0.OUT_DIM
        comptime L0_WS_SIZE = BATCH * Self.L0.WORKSPACE_SIZE_PER_SAMPLE
        comptime L1_WS_SIZE = BATCH * Self.L1.WORKSPACE_SIZE_PER_SAMPLE

        # Create views into workspace_buf
        var inter_buf = DeviceBuffer[dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            INTER_SIZE,
            owning=False,
        )

        # L0 workspace (may be empty for leaf layers)
        var l0_workspace_ptr = workspace_buf.unsafe_ptr() + INTER_SIZE
        var l0_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l0_workspace_ptr,
            L0_WS_SIZE if L0_WS_SIZE > 0 else 1,
            owning=False,
        )

        # L1 workspace (may be empty for leaf layers)
        var l1_workspace_ptr = l0_workspace_ptr + L0_WS_SIZE
        var l1_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l1_workspace_ptr,
            L1_WS_SIZE if L1_WS_SIZE > 0 else 1,
            owning=False,
        )

        # Create views into params_buf for each layer
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # Create views into cache_buf for each layer
        var l0_cache_ptr = cache_buf.unsafe_ptr()
        var l0_cache_buf = DeviceBuffer[dtype](
            ctx,
            l0_cache_ptr,
            BATCH * Self.L0.CACHE_SIZE,
            owning=False,
        )
        var l1_cache_ptr = l0_cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache_buf = DeviceBuffer[dtype](
            ctx,
            l1_cache_ptr,
            BATCH * Self.L1.CACHE_SIZE,
            owning=False,
        )

        # L0: input -> inter_buf (using workspace)
        Self.L0.forward_gpu_ws[BATCH](
            ctx,
            inter_buf,
            input_buf,
            l0_params_buf,
            l0_cache_buf,
            l0_workspace_buf,
        )

        # L1: inter_buf -> output (using workspace)
        Self.L1.forward_gpu_ws[BATCH](
            ctx,
            output_buf,
            inter_buf,
            l1_params_buf,
            l1_cache_buf,
            l1_workspace_buf,
        )

    @staticmethod
    fn forward_gpu_no_cache_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching, using pre-allocated workspace.

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        # Compute workspace offsets
        comptime INTER_SIZE = BATCH * Self.L0.OUT_DIM
        comptime L0_WS_SIZE = BATCH * Self.L0.WORKSPACE_SIZE_PER_SAMPLE
        comptime L1_WS_SIZE = BATCH * Self.L1.WORKSPACE_SIZE_PER_SAMPLE

        # Create views into workspace_buf
        var inter_buf = DeviceBuffer[dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            INTER_SIZE,
            owning=False,
        )

        var l0_workspace_ptr = workspace_buf.unsafe_ptr() + INTER_SIZE
        var l0_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l0_workspace_ptr,
            L0_WS_SIZE if L0_WS_SIZE > 0 else 1,
            owning=False,
        )

        var l1_workspace_ptr = l0_workspace_ptr + L0_WS_SIZE
        var l1_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l1_workspace_ptr,
            L1_WS_SIZE if L1_WS_SIZE > 0 else 1,
            owning=False,
        )

        # Create views into params_buf
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # L0: input -> inter_buf
        Self.L0.forward_gpu_no_cache_ws[BATCH](
            ctx,
            inter_buf,
            input_buf,
            l0_params_buf,
            l0_workspace_buf,
        )

        # L1: inter_buf -> output
        Self.L1.forward_gpu_no_cache_ws[BATCH](
            ctx,
            output_buf,
            inter_buf,
            l1_params_buf,
            l1_workspace_buf,
        )

    @staticmethod
    fn backward_gpu_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass using pre-allocated workspace (no internal allocation).

        Workspace layout: [grad_inter_buf (BATCH * L0.OUT_DIM) | L0 workspace | L1 workspace]
        Note: The same workspace layout as forward - the inter_buf region is reused.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient w.r.t. input [BATCH * IN_DIM] (written).
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [L0 params | L1 params].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] = [L0 cache | L1 cache].
            grads_buf: Parameter gradients [PARAM_SIZE] = [L0 grads | L1 grads] (written).
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        # Compute workspace offsets (same as forward)
        comptime INTER_SIZE = BATCH * Self.L0.OUT_DIM
        comptime L0_WS_SIZE = BATCH * Self.L0.WORKSPACE_SIZE_PER_SAMPLE
        comptime L1_WS_SIZE = BATCH * Self.L1.WORKSPACE_SIZE_PER_SAMPLE

        # Create views into workspace_buf (grad_inter_buf reuses same space as forward's inter_buf)
        var grad_inter_ptr = workspace_buf.unsafe_ptr()
        var grad_inter_buf = DeviceBuffer[dtype](
            ctx,
            grad_inter_ptr,
            INTER_SIZE,
            owning=False,
        )

        var l0_workspace_ptr = grad_inter_ptr + INTER_SIZE
        var l0_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l0_workspace_ptr,
            L0_WS_SIZE if L0_WS_SIZE > 0 else 1,
            owning=False,
        )

        var l1_workspace_ptr = l0_workspace_ptr + L0_WS_SIZE
        var l1_workspace_buf = DeviceBuffer[dtype](
            ctx,
            l1_workspace_ptr,
            L1_WS_SIZE if L1_WS_SIZE > 0 else 1,
            owning=False,
        )

        # Create views into params_buf
        var l0_params_ptr = params_buf.unsafe_ptr()
        var l0_params_buf = DeviceBuffer[dtype](
            ctx,
            l0_params_ptr,
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var l1_params_ptr = l0_params_ptr + Self.L0.PARAM_SIZE
        var l1_params_buf = DeviceBuffer[dtype](
            ctx,
            l1_params_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # Create views into cache_buf
        var l0_cache_ptr = cache_buf.unsafe_ptr()
        var l0_cache_buf = DeviceBuffer[dtype](
            ctx,
            l0_cache_ptr,
            BATCH * Self.L0.CACHE_SIZE,
            owning=False,
        )
        var l1_cache_ptr = l0_cache_ptr + BATCH * Self.L0.CACHE_SIZE
        var l1_cache_buf = DeviceBuffer[dtype](
            ctx,
            l1_cache_ptr,
            BATCH * Self.L1.CACHE_SIZE,
            owning=False,
        )

        # Create views into grads_buf
        var l0_grads_buf = DeviceBuffer[dtype](
            ctx,
            grads_buf.unsafe_ptr(),
            Self.L0.PARAM_SIZE,
            owning=False,
        )
        var grads_ptr = grads_buf.unsafe_ptr()
        var l1_grads_ptr = grads_ptr + Self.L0.PARAM_SIZE
        var l1_grads_buf = DeviceBuffer[dtype](
            ctx,
            l1_grads_ptr,
            Self.L1.PARAM_SIZE,
            owning=False,
        )

        # L1 backward: grad_output -> grad_inter
        Self.L1.backward_gpu_ws[BATCH](
            ctx,
            grad_inter_buf,
            grad_output_buf,
            l1_params_buf,
            l1_cache_buf,
            l1_grads_buf,
            l1_workspace_buf,
        )

        # L0 backward: grad_inter -> grad_input
        Self.L0.backward_gpu_ws[BATCH](
            ctx,
            grad_input_buf,
            grad_inter_buf,
            l0_params_buf,
            l0_cache_buf,
            l0_grads_buf,
            l0_workspace_buf,
        )


# =============================================================================
# Sequential Helper Functions (composing with Seq2)
# =============================================================================
#
# Instead of separate Seq3, Seq4, ... structs, we compose using Seq2:
#   Seq3 = Seq2[Seq2[L0, L1], L2]
#   Seq4 = Seq2[Seq2[Seq2[L0, L1], L2], L3]
#
# This works because Seq2 implements Model, so it can be nested.
# Benefits:
#   - Only one Sequential struct to maintain
#   - PARAM_SIZE composes correctly
#   - Dimensions chain correctly
# =============================================================================


fn seq[L0: Model, L1: Model](var l0: L0, var l1: L1) -> Seq2[L0, L1]:
    """Create a 2-layer sequential model.

    Usage:
        var model = seq(Linear[2, 16, 4](), ReLU[16, 4]())
    """
    return Seq2[L0, L1](l0^, l1^)


fn seq[
    L0: Model, L1: Model, L2: Model
](var l0: L0, var l1: L1, var l2: L2) -> Seq2[Seq2[L0, L1], L2]:
    """Create a 3-layer sequential model (composed from Seq2)."""
    return seq(seq(l0^, l1^), l2^)


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model
](var l0: L0, var l1: L1, var l2: L2, var l3: L3) -> Seq2[
    Seq2[Seq2[L0, L1], L2], L3
]:
    """Create a 4-layer sequential model (composed from Seq2)."""
    return seq(seq(l0^, l1^, l2^), l3^)


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model
](var l0: L0, var l1: L1, var l2: L2, var l3: L3, var l4: L4) -> Seq2[
    Seq2[Seq2[Seq2[L0, L1], L2], L3], L4
]:
    """Create a 5-layer sequential model (composed from Seq2)."""
    return seq(seq(seq(seq(l0^, l1^), l2^), l3^), l4^)


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model, L5: Model
](
    var l0: L0, var l1: L1, var l2: L2, var l3: L3, var l4: L4, var l5: L5
) -> Seq2[Seq2[Seq2[Seq2[Seq2[L0, L1], L2], L3], L4], L5]:
    """Create a 6-layer sequential model (composed from Seq2)."""
    return seq(seq(seq(seq(seq(l0^, l1^), l2^), l3^), l4^), l5^)


fn seq[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
](
    var l0: L0,
    var l1: L1,
    var l2: L2,
    var l3: L3,
    var l4: L4,
    var l5: L5,
    var l6: L6,
) -> Seq2[Seq2[Seq2[Seq2[Seq2[Seq2[L0, L1], L2], L3], L4], L5], L6]:
    """Create a 7-layer sequential model (composed from Seq2)."""
    return seq(seq(seq(seq(seq(seq(l0^, l1^), l2^), l3^), l4^), l5^), l6^)


fn seq[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
    L7: Model,
](
    var l0: L0,
    var l1: L1,
    var l2: L2,
    var l3: L3,
    var l4: L4,
    var l5: L5,
    var l6: L6,
    var l7: L7,
) -> Seq2[Seq2[Seq2[Seq2[Seq2[Seq2[Seq2[L0, L1], L2], L3], L4], L5], L6], L7]:
    """Create an 8-layer sequential model (composed from Seq2)."""
    return seq(
        seq(seq(seq(seq(seq(seq(l0^, l1^), l2^), l3^), l4^), l5^), l6^), l7^
    )
