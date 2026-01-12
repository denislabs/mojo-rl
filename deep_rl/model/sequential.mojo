from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim

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

    Usage:
        var model = seq(Linear[2, 16](), ReLU[16]())
        # or with explicit types:
        var model = Seq2[Linear[2, 16], ReLU[16]](layer0, layer1)
    """

    comptime IN_DIM: Int = Self.L0.IN_DIM
    comptime OUT_DIM: Int = Self.L1.OUT_DIM
    comptime PARAM_SIZE: Int = Self.L0.PARAM_SIZE + Self.L1.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.L0.CACHE_SIZE + Self.L1.CACHE_SIZE

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
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass through all layers (all zero-copy views).

        Cache layout (blocked): [L0's cache | L1's cache]
        Params layout: [L0's params | L1's params]
        """
        # Intermediate buffer for L0 output / L1 input
        var buffer0_storage = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )
        var buffer0 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.OUT_DIM), MutAnyOrigin
        ](buffer0_storage)

        # Create zero-copy views using pointer offsets
        var params_ptr = params.ptr
        var l0_params = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](params_ptr.offset(Self.L0.PARAM_SIZE))

        var cache_ptr = cache.ptr
        var l0_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr)
        var l1_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr.offset(BATCH * Self.L0.CACHE_SIZE))

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
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        """Forward pass without caching (for inference)."""
        # Intermediate buffer for L0 output / L1 input
        var buffer0_storage = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )
        var buffer0 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.OUT_DIM), MutAnyOrigin
        ](buffer0_storage)

        # Create zero-copy views using pointer offsets
        var params_ptr = params.ptr
        var l0_params = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](params_ptr.offset(Self.L0.PARAM_SIZE))

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
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
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
        var l1_params = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](params_ptr.offset(Self.L0.PARAM_SIZE))

        var grads_ptr = grads.ptr
        var l0_grads = LayoutTensor[
            dtype, Layout.row_major(Self.L0.PARAM_SIZE), MutAnyOrigin
        ](grads_ptr)
        var l1_grads = LayoutTensor[
            dtype, Layout.row_major(Self.L1.PARAM_SIZE), MutAnyOrigin
        ](grads_ptr.offset(Self.L0.PARAM_SIZE))

        var cache_ptr = cache.ptr
        var l0_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L0.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr)
        var l1_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.L1.CACHE_SIZE), MutAnyOrigin
        ](cache_ptr.offset(BATCH * Self.L0.CACHE_SIZE))

        # Intermediate buffer for gradient at buffer0
        var grad_buffer0_storage = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )
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
            l1_grad_output, l1_grad_input, l1_params, l1_cache, l1_grads
        )

        # L0 backward: grad_buffer0 -> grad_input
        self.layer0.backward[BATCH](
            grad_buffer0, grad_input, l0_params, l0_cache, l0_grads
        )
        # No copy-back needed - grads views modify the original in-place

    @always_inline
    @staticmethod
    fn forward_kernel[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        x: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.IN_DIM),
            ImmutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
        b: LayoutTensor[
            dtype,
            Layout.row_major(Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Forward pass on GPU - Seq2 requires separate kernel calls for each layer.

        Note: Seq2 composes layers with different weight structures.
        For GPU execution, call L0.forward_kernel and L1.forward_kernel separately
        from host code with appropriate intermediate buffers.
        This kernel signature exists only to satisfy the Model trait.
        """
        pass

    @always_inline
    @staticmethod
    fn backward_dx_kernel[
        BATCH: Int
    ](
        dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        dy: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Backward pass for input gradient on GPU - Seq2 requires separate kernel calls.

        Note: For GPU execution, call L1.backward_dx_kernel then L0.backward_dx_kernel
        separately from host code with appropriate intermediate buffers.
        """
        pass

    @always_inline
    @staticmethod
    fn backward_dW_db_kernel[
        BATCH: Int
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin],
        x: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.IN_DIM),
            ImmutAnyOrigin,
        ],
        dy: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Backward pass for weight/bias gradients on GPU - Seq2 requires separate kernel calls.

        Note: For GPU execution, call L0.backward_dW_db_kernel and L1.backward_dW_db_kernel
        separately from host code with appropriate buffers for each layer's parameters.
        """
        pass


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
