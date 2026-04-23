from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct Embedding[vocab_size: Int, embed_dim: Int](DiffOp):
    """Embedding: table lookup via one-hot input encoding.

    Input is a one-hot vector of size vocab_size per batch element.
    Forward: output = input @ W (equivalent to row lookup W[argmax(input)]).
    Backward: grad_W += input.T @ grad_output (scatters gradient to indexed row).

    This uses the one-hot encoding approach to fit the DiffOp interface
    (float input tensors). For large vocabularies (>10k), a specialized
    EmbeddingModel with integer indices would be more efficient.

    PARAM_SIZE = vocab_size * embed_dim (the embedding table)
    CACHE_SIZE = vocab_size (caches the one-hot input for backward)
    """

    comptime OP_ID: Int = OpID.EMBEDDING._value
    comptime IN_DIM: Int = Self.vocab_size
    comptime OUT_DIM: Int = Self.embed_dim
    comptime PARAM_SIZE: Int = Self.vocab_size * Self.embed_dim
    comptime CACHE_SIZE: Int = Self.vocab_size
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    def eval[
        BATCH: Int, dtype: DType = DType.float32
    ](
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
        # W is (vocab_size, embed_dim) stored row-major in params
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            MutAnyOrigin,
        ](params.ptr)

        for b in range(BATCH):
            # Cache input for backward
            for v in range(Self.vocab_size):
                cache[b, v] = input[b, v]

            # output = input @ W  (sparse: only the argmax row contributes)
            for j in range(Self.embed_dim):
                var acc: Scalar[dtype] = 0
                for v in range(Self.vocab_size):
                    acc += rebind[Scalar[dtype]](input[b, v]) * rebind[
                        Scalar[dtype]
                    ](W[v, j])
                output[b, j] = acc

    @staticmethod
    def vjp[
        BATCH: Int, dtype: DType = DType.float32
    ](
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            MutAnyOrigin,
        ](params.ptr)
        var dW = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            MutAnyOrigin,
        ](grad_params.ptr)

        for b in range(BATCH):
            # grad_input = grad_output @ W.T
            for v in range(Self.vocab_size):
                var acc: Scalar[dtype] = 0
                for j in range(Self.embed_dim):
                    acc += rebind[Scalar[dtype]](grad_output[b, j]) * rebind[
                        Scalar[dtype]
                    ](W[v, j])
                grad_input[b, v] = acc

            # dW += input.T @ grad_output (ACCUMULATE)
            for v in range(Self.vocab_size):
                var inp_val = rebind[Scalar[dtype]](cache[b, v])
                for j in range(Self.embed_dim):
                    dW[v, j] = rebind[Scalar[dtype]](
                        dW[v, j]
                    ) + inp_val * rebind[Scalar[dtype]](grad_output[b, j])

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.embed_dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            ImmutAnyOrigin,
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.embed_dim:
            return
        var b = idx // Self.embed_dim
        var j = idx % Self.embed_dim

        # Cache input (only thread with j==0 per batch to avoid races)
        if j == 0:
            for v in range(Self.vocab_size):
                cache[b, v] = rebind[Scalar[dtype]](input[b, v])

        # output[b, j] = sum_v(input[b, v] * W[v, j])
        var acc: Scalar[dtype] = 0
        for v in range(Self.vocab_size):
            acc += rebind[Scalar[dtype]](input[b, v]) * rebind[Scalar[dtype]](
                W[v, j]
            )
        output[b, j] = acc

    @always_inline
    @staticmethod
    def backward_dx_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.embed_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            ImmutAnyOrigin,
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.vocab_size:
            return
        var b = idx // Self.vocab_size
        var v = idx % Self.vocab_size

        # grad_input[b, v] = sum_j(grad_output[b, j] * W[v, j])
        var acc: Scalar[dtype] = 0
        for j in range(Self.embed_dim):
            acc += rebind[Scalar[dtype]](grad_output[b, j]) * rebind[
                Scalar[dtype]
            ](W[v, j])
        grad_input[b, v] = acc

    @always_inline
    @staticmethod
    def backward_dW_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        dW: LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            MutAnyOrigin,
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.embed_dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), ImmutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= Self.vocab_size * Self.embed_dim:
            return
        var v = idx // Self.embed_dim
        var j = idx % Self.embed_dim

        # dW[v, j] += sum_b(cache[b, v] * grad_output[b, j])
        var acc: Scalar[dtype] = 0
        for b in range(BATCH):
            acc += rebind[Scalar[dtype]](cache[b, v]) * rebind[Scalar[dtype]](
                grad_output[b, j]
            )
        dW[v, j] = dW[v, j] + acc

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def eval_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), ImmutAnyOrigin
        ](input.ptr)
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            ImmutAnyOrigin,
        ](params.ptr)
        var total_elements = BATCH * Self.embed_dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.embed_dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.vocab_size),
                ImmutAnyOrigin,
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.vocab_size, Self.embed_dim),
                ImmutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.vocab_size), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, W, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            W,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.embed_dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            ImmutAnyOrigin,
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.vocab_size), ImmutAnyOrigin
        ](cache.ptr)
        var dW = LayoutTensor[
            dtype,
            Layout.row_major(Self.vocab_size, Self.embed_dim),
            MutAnyOrigin,
        ](grad_params.ptr)

        # Kernel 1: grad_input = grad_output @ W.T
        var total_dx = BATCH * Self.vocab_size
        var grid_dx = (total_dx + TPB - 1) // TPB

        @parameter
        @always_inline
        def dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.vocab_size), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.embed_dim),
                ImmutAnyOrigin,
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.vocab_size, Self.embed_dim),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_dx_kernel_impl[BATCH, dtype](grad_input, grad_output, W)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            W,
            grid_dim=(grid_dx,),
            block_dim=(TPB,),
        )

        # Kernel 2: dW += input.T @ grad_output
        var total_dW = Self.vocab_size * Self.embed_dim
        var grid_dW = (total_dW + TPB - 1) // TPB

        @parameter
        @always_inline
        def dW_wrapper(
            dW: LayoutTensor[
                dtype,
                Layout.row_major(Self.vocab_size, Self.embed_dim),
                MutAnyOrigin,
            ],
            grad_output: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.embed_dim),
                ImmutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.vocab_size),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_dW_kernel_impl[BATCH, dtype](dW, grad_output, cache)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            dW,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_dW,),
            block_dim=(TPB,),
        )
