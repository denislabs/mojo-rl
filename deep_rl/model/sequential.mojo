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
        mut self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Forward pass through all layers.

        Cache layout: [L0's cache (BATCH * L0.CACHE_SIZE) | L1's cache (BATCH * L1.CACHE_SIZE)]
        """
        # Intermediate buffer for L0 output / L1 input
        var buffer0 = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )

        # Split cache for each layer
        var l0_cache = InlineArray[Scalar[dtype], BATCH * Self.L0.CACHE_SIZE](
            uninitialized=True
        )
        var l1_cache = InlineArray[Scalar[dtype], BATCH * Self.L1.CACHE_SIZE](
            uninitialized=True
        )

        # L0: input -> buffer0
        self.layer0.forward[BATCH](input, buffer0, l0_cache)

        # L1: buffer0 -> output (rebind buffer0 to L1's input type)
        self.layer1.forward[BATCH](
            rebind[InlineArray[Scalar[dtype], BATCH * Self.L1.IN_DIM]](buffer0),
            output,
            l1_cache,
        )

        # Copy caches back to combined cache
        for i in range(BATCH * Self.L0.CACHE_SIZE):
            cache[i] = l0_cache[i]
        for i in range(BATCH * Self.L1.CACHE_SIZE):
            cache[BATCH * Self.L0.CACHE_SIZE + i] = l1_cache[i]

    fn forward[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
    ):
        """Forward pass without caching (for inference)."""
        # Intermediate buffer for L0 output / L1 input
        var buffer0 = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )

        # L0: input -> buffer0
        self.layer0.forward[BATCH](input, buffer0)

        # L1: buffer0 -> output
        self.layer1.forward[BATCH](
            rebind[InlineArray[Scalar[dtype], BATCH * Self.L1.IN_DIM]](buffer0),
            output,
        )

    fn backward[
        BATCH: Int
    ](
        mut self,
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut grad_input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Backward pass in reverse order.

        Cache layout: [L0's cache (BATCH * L0.CACHE_SIZE) | L1's cache (BATCH * L1.CACHE_SIZE)]
        """
        # Extract caches for each layer
        var l0_cache = InlineArray[Scalar[dtype], BATCH * Self.L0.CACHE_SIZE](
            uninitialized=True
        )
        var l1_cache = InlineArray[Scalar[dtype], BATCH * Self.L1.CACHE_SIZE](
            uninitialized=True
        )
        for i in range(BATCH * Self.L0.CACHE_SIZE):
            l0_cache[i] = cache[i]
        for i in range(BATCH * Self.L1.CACHE_SIZE):
            l1_cache[i] = cache[BATCH * Self.L0.CACHE_SIZE + i]

        # Temp buffer for grad at buffer0
        var grad_buffer0 = InlineArray[Scalar[dtype], BATCH * Self.L0.OUT_DIM](
            uninitialized=True
        )

        # L1 backward: grad_output -> grad_buffer0
        self.layer1.backward[BATCH](
            rebind[InlineArray[Scalar[dtype], BATCH * Self.L1.OUT_DIM]](
                grad_output
            ),
            rebind[InlineArray[Scalar[dtype], BATCH * Self.L1.IN_DIM]](
                grad_buffer0
            ),
            l1_cache,
        )

        # L0 backward: grad_buffer0 -> grad_input
        self.layer0.backward[BATCH](grad_buffer0, grad_input, l0_cache)

    fn zero_grad(mut self):
        """Zero gradients of all layers."""
        self.layer0.zero_grad()
        self.layer1.zero_grad()

    fn get_params(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get concatenated parameters from all layers."""
        var params = InlineArray[Scalar[dtype], Self.PARAM_SIZE](
            uninitialized=True
        )
        var p0 = self.layer0.get_params()
        var p1 = self.layer1.get_params()

        for i in range(Self.L0.PARAM_SIZE):
            params[i] = p0[i]
        for i in range(Self.L1.PARAM_SIZE):
            params[Self.L0.PARAM_SIZE + i] = p1[i]

        return params^

    fn set_params(
        mut self, params: InlineArray[Scalar[dtype], Self.PARAM_SIZE]
    ):
        """Set parameters to all layers."""
        var p0 = InlineArray[Scalar[dtype], Self.L0.PARAM_SIZE](
            uninitialized=True
        )
        var p1 = InlineArray[Scalar[dtype], Self.L1.PARAM_SIZE](
            uninitialized=True
        )

        for i in range(Self.L0.PARAM_SIZE):
            p0[i] = params[i]
        for i in range(Self.L1.PARAM_SIZE):
            p1[i] = params[Self.L0.PARAM_SIZE + i]

        self.layer0.set_params(p0)
        self.layer1.set_params(p1)

    fn get_grads(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get concatenated gradients from all layers."""
        var grads = InlineArray[Scalar[dtype], Self.PARAM_SIZE](
            uninitialized=True
        )
        var g0 = self.layer0.get_grads()
        var g1 = self.layer1.get_grads()

        for i in range(Self.L0.PARAM_SIZE):
            grads[i] = g0[i]
        for i in range(Self.L1.PARAM_SIZE):
            grads[Self.L0.PARAM_SIZE + i] = g1[i]

        return grads^

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
