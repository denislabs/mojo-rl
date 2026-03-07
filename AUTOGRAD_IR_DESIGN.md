# Compile-Time Autograd + IR for Mojo NN

## The Core Insight

Your existing architecture already contains the seeds of an autograd system. The `Model` trait with `forward()` + `backward()`, and `Sequential` chaining them with `comptime for` loops — that IS a compile-time IR with manual differentiation. The goal is to **automate the backward generation** and **enable compile-time kernel fusion**, while keeping the zero-cost, stateless design that makes your framework fast.

The key realization: **Mojo's `comptime` + variadic type parameters let us do at compile time what JAX does at JIT time and PyTorch does at runtime.** No tape. No tracing. No overhead.

---

## Architecture: Three Layers

```
Layer 3: Fusion Passes          ← compile-time pattern matching on op chains
Layer 2: AutoDiffGraph          ← auto-generates backward from op composition
Layer 1: DiffOp primitives      ← atomic differentiable operations with VJPs
Layer 0: Existing Model trait   ← untouched, DiffOp composes INTO Model
```

Each layer builds on the previous. Layer 0 (your current code) stays exactly as-is. New layers compose down into `Model`-conforming types, so all existing training infrastructure (`NetworkState`, `Trainer`, `GPUNetworkState`, optimizers, losses) works unchanged.

---

## Layer 1: DiffOp — Differentiable Primitives

### The Trait

A `DiffOp` is a single atomic differentiable operation. It's smaller than a `Model` layer — a `Linear` layer would be composed from `MatMul + BiasAdd` primitives.

```mojo
trait DiffOp(Movable & ImplicitlyCopyable):
    """A single differentiable primitive operation.

    Each DiffOp knows:
    - Its OP_ID for compile-time pattern matching (fusion)
    - Its shape signature (IN_DIM → OUT_DIM)
    - How many parameters it owns
    - What it needs to cache for backward
    - Its forward computation
    - Its VJP (vector-Jacobian product) for backward
    """
    # Type identity for compile-time fusion pattern matching.
    # Using an Int enum avoids fragile string comparison via get_type_name[].
    # See OpID below.
    comptime OP_ID: Int

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int

    # --- CPU ---
    @staticmethod
    fn eval[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    )

    @staticmethod
    fn vjp[BATCH: Int](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grad_params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    )

    # --- GPU ---
    @staticmethod
    fn eval_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ) raises

    @staticmethod
    fn vjp_gpu[BATCH: Int](
        ctx: DeviceContext,
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grad_params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ) raises
```

### OpID — Type Identity for Fusion

Rather than relying on `get_type_name[]` (fragile string comparison), each primitive
declares a numeric ID. Fusion passes match on these IDs at compile time:

```mojo
# Compile-time enum for op identification.
# Using comptime members as enum values (same pattern as your Sentiment example).
struct OpID:
    var _value: Int

    comptime MATMUL         = OpID(1)
    comptime BIAS_ADD       = OpID(2)
    comptime ELEM_ADD       = OpID(3)
    comptime ELEM_MUL       = OpID(4)
    comptime SCALE          = OpID(5)
    comptime RELU           = OpID(10)
    comptime TANH           = OpID(11)
    comptime SIGMOID        = OpID(12)
    comptime MISH           = OpID(13)
    comptime SOFTMAX        = OpID(14)
    comptime LAYER_NORM     = OpID(20)
    comptime RMS_NORM       = OpID(21)
    comptime REDUCE_SUM     = OpID(30)
    comptime REDUCE_MEAN    = OpID(31)
    # Fused ops (100+)
    comptime FUSED_MATMUL_BIAS      = OpID(100)
    comptime FUSED_MATMUL_BIAS_RELU = OpID(101)
    comptime FUSED_MATMUL_BIAS_TANH = OpID(102)
    # Combinators (200+)
    comptime RESIDUAL       = OpID(200)
    comptime PARALLEL       = OpID(201)
    # User-defined (1000+)
    comptime USER_DEFINED   = OpID(1000)
```

Then fusion passes use clean integer comparison:

```mojo
# In the fusion pass:
comptime if (Self.ops[i].OP_ID == OpID.MATMUL._value
         and Self.ops[i+1].OP_ID == OpID.BIAS_ADD._value
         and Self.ops[i+2].OP_ID == OpID.RELU._value):
    # Replace with FusedMatMulBiasReLU
```

This is safer than `get_type_name[]` string matching and doesn't break if struct names
are refactored. It also works with `comptime if` since it's just integer comparison.

### Why Separate from Model?

`DiffOp` is intentionally NOT `Model`. The differences:

| | Model | DiffOp |
|---|---|---|
| Granularity | Coarse (Linear = matmul+bias+cache) | Fine (one math operation) |
| Backward | Hand-coded, layer-specific | Auto-composable VJP |
| Purpose | End-user API, training infrastructure | Building block for auto-differentiation |
| GPU kernels | Self-contained with launch config | @always_inline, designed for fusion |

A `DiffOp` composes INTO a `Model` via the `AutoDiffChain` (Layer 2). Users never interact with `DiffOp` directly unless they're defining new primitives.

### Concrete Primitives

#### MatMul — The Core Linear Transform

```mojo
struct MatMul[in_dim: Int, out_dim: Int](DiffOp):
    """y = x @ W  where x:(B, in_dim), W:(in_dim, out_dim), y:(B, out_dim)"""
    comptime OP_ID: Int = OpID.MATMUL._value
    comptime IN_DIM: Int = in_dim
    comptime OUT_DIM: Int = out_dim
    comptime PARAM_SIZE: Int = in_dim * out_dim  # W only, no bias
    comptime CACHE_SIZE: Int = in_dim            # cache input for dW

    @staticmethod
    fn eval[BATCH: Int](input, mut output, params, mut cache):
        # Create typed view of W from flat params
        var W = LayoutTensor[
            dtype, Layout.row_major(in_dim, out_dim), MutAnyOrigin
        ](params.ptr)

        # Cache input (needed for dW in backward)
        for b in range(BATCH):
            for i in range(in_dim):
                cache[b, i] = input[b, i]

        # output = input @ W
        for b in range(BATCH):
            for j in range(out_dim):
                var acc: Scalar[dtype] = 0
                for k in range(in_dim):
                    acc += input[b, k] * W[k, j]
                output[b, j] = acc

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        var W = LayoutTensor[
            dtype, Layout.row_major(in_dim, out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(in_dim, out_dim), MutAnyOrigin
        ](grad_params.ptr)

        # grad_input = grad_output @ W.T
        for b in range(BATCH):
            for i in range(in_dim):
                var acc: Scalar[dtype] = 0
                for j in range(out_dim):
                    acc += grad_output[b, j] * W[i, j]
                grad_input[b, i] = acc

        # dW += input.T @ grad_output  (ACCUMULATE)
        for b in range(BATCH):
            for i in range(in_dim):
                for j in range(out_dim):
                    dW[i, j] = dW[i, j] + cache[b, i] * grad_output[b, j]

    # GPU: @always_inline kernel using tiled matmul from existing gpu/matmul.mojo
    @always_inline
    @staticmethod
    fn eval_kernel[BATCH: Int](
        output: LayoutTensor[dtype, Layout.row_major(BATCH, out_dim), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, in_dim), ImmutAnyOrigin],
        W: LayoutTensor[dtype, Layout.row_major(in_dim, out_dim), ImmutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, in_dim), MutAnyOrigin],
    ):
        # Reuse existing tiled matmul pattern with shared memory
        # Key: @always_inline means this can be FUSED with adjacent ops
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var input_shared = LayoutTensor[
            dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var W_shared = LayoutTensor[
            dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (in_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < in_dim:
                input_shared[local_row, local_col] = input[global_row, in_col]
                # Cache input during the first tile pass
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = input[global_row, in_col]
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < in_dim and global_col < out_dim:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()
            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](input_shared[local_row, k]) *
                       rebind[Scalar[dtype]](W_shared[k, local_col])
            barrier()

        if global_row < BATCH and global_col < out_dim:
            output[global_row, global_col] = acc
```

#### BiasAdd — Broadcast Addition

```mojo
struct BiasAdd[dim: Int](DiffOp):
    """y = x + b  where x:(B, dim), b:(dim,), y:(B, dim)"""
    comptime OP_ID: Int = OpID.BIAS_ADD._value
    comptime IN_DIM: Int = dim
    comptime OUT_DIM: Int = dim
    comptime PARAM_SIZE: Int = dim    # the bias vector
    comptime CACHE_SIZE: Int = 0      # no cache needed

    @staticmethod
    fn eval[BATCH: Int](input, mut output, params, mut cache):
        for b in range(BATCH):
            for i in range(dim):
                output[b, i] = input[b, i] + params[i]

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        # grad_input = grad_output (identity for addition)
        for b in range(BATCH):
            for i in range(dim):
                grad_input[b, i] = grad_output[b, i]

        # grad_bias += sum(grad_output, axis=0)
        for b in range(BATCH):
            for i in range(dim):
                grad_params[i] = grad_params[i] + grad_output[b, i]

    # GPU: trivially fusible elementwise kernel
    @always_inline
    @staticmethod
    fn eval_kernel[BATCH: Int](
        output: LayoutTensor[dtype, Layout.row_major(BATCH, dim), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, dim), ImmutAnyOrigin],
        bias: LayoutTensor[dtype, Layout.row_major(dim), ImmutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * dim:
            return
        var row = idx // dim
        var col = idx % dim
        output[row, col] = input[row, col] + bias[col]
```

#### Elementwise Activations — Generic Pattern

```mojo
struct ReLUOp[dim: Int](DiffOp):
    """y = max(0, x)"""
    comptime OP_ID: Int = OpID.RELU._value
    comptime IN_DIM: Int = dim
    comptime OUT_DIM: Int = dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = dim  # cache pre-activation for backward

    @staticmethod
    fn eval[BATCH: Int](input, mut output, params, mut cache):
        for b in range(BATCH):
            for i in range(dim):
                var val = input[b, i]
                cache[b, i] = val
                output[b, i] = val if val > 0 else 0

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        for b in range(BATCH):
            for i in range(dim):
                grad_input[b, i] = grad_output[b, i] if cache[b, i] > 0 else 0

    @always_inline
    @staticmethod
    fn eval_kernel[BATCH: Int](
        output: LayoutTensor[dtype, Layout.row_major(BATCH, dim), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, dim), ImmutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, dim), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * dim:
            return
        var row = idx // dim
        var col = idx % dim
        var val = input[row, col]
        cache[row, col] = val
        output[row, col] = val if val > 0 else 0


struct TanhOp[dim: Int](DiffOp):
    """y = tanh(x)"""
    comptime OP_ID: Int = OpID.TANH._value
    comptime IN_DIM: Int = dim
    comptime OUT_DIM: Int = dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = dim  # cache tanh(x) output

    @staticmethod
    fn eval[BATCH: Int](input, mut output, params, mut cache):
        for b in range(BATCH):
            for i in range(dim):
                var t = math.tanh(input[b, i])
                cache[b, i] = t        # cache tanh output
                output[b, i] = t

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        # d/dx tanh(x) = 1 - tanh(x)^2
        for b in range(BATCH):
            for i in range(dim):
                var t = cache[b, i]
                grad_input[b, i] = grad_output[b, i] * (1 - t * t)


struct SigmoidOp[dim: Int](DiffOp):
    """y = sigmoid(x)"""
    comptime OP_ID: Int = OpID.SIGMOID._value
    comptime IN_DIM: Int = dim
    comptime OUT_DIM: Int = dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = dim

    @staticmethod
    fn eval[BATCH: Int](input, mut output, params, mut cache):
        for b in range(BATCH):
            for i in range(dim):
                var s = 1.0 / (1.0 + math.exp(-input[b, i]))
                cache[b, i] = s
                output[b, i] = s

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        for b in range(BATCH):
            for i in range(dim):
                var s = cache[b, i]
                grad_input[b, i] = grad_output[b, i] * s * (1 - s)


struct LayerNormOp[dim: Int, EPSILON: Scalar[dtype]](DiffOp):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta"""
    comptime IN_DIM: Int = dim
    comptime OUT_DIM: Int = dim
    comptime PARAM_SIZE: Int = dim * 2  # gamma + beta
    comptime CACHE_SIZE: Int = dim + 2  # normalized + mean + inv_std

    # forward and vjp follow standard layer norm derivation
    # ...
```

#### The Full Primitive Catalog

```
Arithmetic:     MatMul, BiasAdd, ElemAdd, ElemMul, Scale
Activations:    ReLUOp, TanhOp, SigmoidOp, MishOp, SoftmaxOp
Normalization:  LayerNormOp, RMSNormOp
Reduction:      ReduceSum, ReduceMean
Reshaping:      (future) Reshape, Transpose, Concat, Split
Attention:      (future) ScaledDotProduct, MultiHeadProjection
```

Each primitive: ~30-60 lines. Each hand-coded `Model` layer: ~200-400 lines. The savings compound with composition.

---

## Layer 2: AutoDiffChain — Automatic Backward Generation

### The Core Idea

`AutoDiffChain` takes a variadic list of `DiffOp` types and produces a `Model`-conforming struct with **automatically generated** `forward()` and `backward()`. This is the autograd engine.

```mojo
@fieldwise_init
struct AutoDiffChain[*OPS: DiffOp](Model):
    """Chains DiffOps sequentially and auto-generates backward pass.

    Given ops [A, B, C]:
      forward:  x → A.eval → B.eval → C.eval → y
      backward: dy → C.vjp → B.vjp → A.vjp → dx
                     ↓ dW_C    ↓ dW_B    ↓ dW_A  (accumulated to grads)

    All dimension checking happens at compile time.
    All intermediate buffers are stack-allocated.
    All offset computation is compile-time constant.
    """
    comptime ops = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N: Int = Variadic.size(Self.ops)

    # ── Shape validation at compile time ──
    # Verify dimension chain: ops[i].OUT_DIM == ops[i+1].IN_DIM
    # Uses comptime assert (replaces deprecated constrained[])
    comptime _VALIDATED: Bool = Self._validate_dims()

    @staticmethod
    fn _validate_dims() -> Bool:
        comptime for i in range(Self.N - 1):
            comptime assert (
                Self.ops[i].OUT_DIM == Self.ops[i + 1].IN_DIM
            ), (
                "dimension mismatch: op[" + str(i) + "].OUT_DIM ("
                + str(Self.ops[i].OUT_DIM) + ") != op["
                + str(i + 1) + "].IN_DIM (" + str(Self.ops[i + 1].IN_DIM) + ")"
            )
        return True

    # ── Compile-time constants (Model conformance) ──
    comptime IN_DIM: Int = Self.ops[0].IN_DIM
    comptime OUT_DIM: Int = Self.ops[Self.N - 1].OUT_DIM

    @staticmethod
    fn _sum_param_size() -> Int:
        var total = 0
        comptime for i in range(Self.N):
            total += Self.ops[i].PARAM_SIZE
        return total

    @staticmethod
    fn _sum_cache_size() -> Int:
        var total = 0
        comptime for i in range(Self.N):
            total += Self.ops[i].CACHE_SIZE
        return total

    @staticmethod
    fn _total_inter() -> Int:
        """Sum of intermediate buffer sizes (output dims for ops 0..N-2)."""
        var total = 0
        comptime for i in range(Self.N - 1):
            total += Self.ops[i].OUT_DIM
        return total

    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime CACHE_SIZE: Int = Self._sum_cache_size()
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self._total_inter()

    # ── Offset helpers (all comptime) ──

    @staticmethod
    fn _param_offset[idx: Int]() -> Int:
        var off = 0
        comptime for j in range(idx):
            off += Self.ops[j].PARAM_SIZE
        return off

    @staticmethod
    fn _cache_offset[idx: Int]() -> Int:
        var off = 0
        comptime for j in range(idx):
            off += Self.ops[j].CACHE_SIZE
        return off

    @staticmethod
    fn _inter_offset[idx: Int]() -> Int:
        var off = 0
        comptime for j in range(idx):
            off += Self.ops[j].OUT_DIM
        return off

    # ══════════════════════════════════════════════
    #  FORWARD — auto-generated from DiffOp chain
    # ══════════════════════════════════════════════

    @staticmethod
    fn forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ):
        comptime if Self.N == 1:
            # Single op: direct pass-through
            var p = LayoutTensor[
                dtype, Layout.row_major(Self.ops[0].PARAM_SIZE), MutAnyOrigin
            ](params.ptr)
            var c = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ops[0].CACHE_SIZE), MutAnyOrigin
            ](cache.ptr)
            Self.ops[0].eval[BATCH](input, output, p, c)
        else:
            # Multi-op: allocate intermediates, chain through
            var inter_buf = List[Scalar[dtype]](capacity=BATCH * Self._total_inter())
            for _ in range(BATCH * Self._total_inter()):
                inter_buf.append(0)
            var inter_ptr = inter_buf.unsafe_ptr()

            comptime for i in range(Self.N):
                # Slice params and cache for this op
                var li_p = LayoutTensor[
                    dtype, Layout.row_major(Self.ops[i].PARAM_SIZE), MutAnyOrigin
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ops[i].CACHE_SIZE), MutAnyOrigin
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == 0:
                    # First op: input → inter[0]
                    var li_out = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[0].OUT_DIM), MutAnyOrigin
                    ](inter_ptr)
                    Self.ops[0].eval[BATCH](input, li_out, li_p, li_c)

                elif i == Self.N - 1:
                    # Last op: inter[N-2] → output
                    var li_in = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].IN_DIM), MutAnyOrigin
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.ops[i].eval[BATCH](li_in, output, li_p, li_c)

                else:
                    # Middle: inter[i-1] → inter[i]
                    var li_in = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].IN_DIM), MutAnyOrigin
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].OUT_DIM), MutAnyOrigin
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.ops[i].eval[BATCH](li_in, li_out, li_p, li_c)

    # ══════════════════════════════════════════════════
    #  BACKWARD — AUTOMATICALLY GENERATED from VJPs
    # ══════════════════════════════════════════════════
    #
    #  This is the payoff. No hand-coding.
    #  The compiler walks the op chain in reverse and calls each VJP.

    @staticmethod
    fn backward[BATCH: Int](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        comptime if Self.N == 1:
            var p = LayoutTensor[
                dtype, Layout.row_major(Self.ops[0].PARAM_SIZE), MutAnyOrigin
            ](params.ptr)
            var c = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ops[0].CACHE_SIZE), MutAnyOrigin
            ](cache.ptr)
            var g = LayoutTensor[
                dtype, Layout.row_major(Self.ops[0].PARAM_SIZE), MutAnyOrigin
            ](grads.ptr)
            Self.ops[0].vjp[BATCH](grad_output, grad_input, p, c, g)
        else:
            # Gradient intermediates (same layout as forward intermediates)
            var grad_inter_buf = List[Scalar[dtype]](capacity=BATCH * Self._total_inter())
            for _ in range(BATCH * Self._total_inter()):
                grad_inter_buf.append(0)
            var gi_ptr = grad_inter_buf.unsafe_ptr()

            # Reverse iteration: N-1, N-2, ..., 0
            comptime for _ri in range(Self.N):
                comptime i = Self.N - 1 - _ri

                var li_p = LayoutTensor[
                    dtype, Layout.row_major(Self.ops[i].PARAM_SIZE), MutAnyOrigin
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ops[i].CACHE_SIZE), MutAnyOrigin
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_g = LayoutTensor[
                    dtype, Layout.row_major(Self.ops[i].PARAM_SIZE), MutAnyOrigin
                ](grads.ptr + Self._param_offset[i]())

                comptime if i == Self.N - 1 and i == 0:
                    # Single op (shouldn't reach here, but safety)
                    Self.ops[i].vjp[BATCH](grad_output, grad_input, li_p, li_c, li_g)

                elif i == Self.N - 1:
                    # Last op: grad_output → grad_inter[i-1]
                    var li_gi = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].IN_DIM), MutAnyOrigin
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.ops[i].vjp[BATCH](grad_output, li_gi, li_p, li_c, li_g)

                elif i == 0:
                    # First op: grad_inter[0] → grad_input
                    var li_go = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[0].OUT_DIM), MutAnyOrigin
                    ](gi_ptr)
                    Self.ops[0].vjp[BATCH](li_go, grad_input, li_p, li_c, li_g)

                else:
                    # Middle: grad_inter[i] → grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].OUT_DIM), MutAnyOrigin
                    ](gi_ptr + BATCH * Self._inter_offset[i]())
                    var li_gi = LayoutTensor[
                        dtype, Layout.row_major(BATCH, Self.ops[i].IN_DIM), MutAnyOrigin
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.ops[i].vjp[BATCH](li_go, li_gi, li_p, li_c, li_g)

    # GPU forward/backward follow the same pattern as Sequential's GPU methods,
    # using workspace buffers instead of List allocation.
    # Omitted for brevity — structurally identical to the CPU versions
    # but with ctx.enqueue_function[] calls.
```

### What This Gives You

Now defining new layers is trivial and backward is free:

```mojo
# These are COMPLETE layer definitions. No hand-coded backward needed.

# Linear layer = matmul + bias
comptime LinearAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d],
    BiasAdd[out_d],
]

# Linear + ReLU (currently hand-coded as LinearReLU)
comptime LinearReLUAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d],
    BiasAdd[out_d],
    ReLUOp[out_d],
]

# Linear + LayerNorm + Tanh (would require a new hand-coded layer today)
comptime NormedLinearTanhAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d],
    BiasAdd[out_d],
    LayerNormOp[out_d, 1e-5],
    TanhOp[out_d],
]

# Full MLP (composes with Sequential for multi-layer)
comptime MLP = Sequential[
    LinearReLUAD[784, 256],
    LinearReLUAD[256, 128],
    LinearAD[128, 10],
]
```

`MLP` conforms to `Model`. It works with `NetworkState`, `Trainer`, `GPUNetworkState`, optimizers, losses — everything. Zero changes to existing infrastructure.

---

## Layer 3: Compile-Time Fusion Passes

This is where Mojo's compile-time capabilities really shine. The idea: before generating code, analyze the `DiffOp` chain at compile time and replace patterns with fused implementations.

### The Fusion Registry

```mojo
# A FusedOp is a DiffOp that replaces a sequence of DiffOps
# with a single, optimized implementation (especially for GPU).

trait FusedOp(DiffOp):
    """A fused operation that replaces multiple sequential DiffOps.

    The fused op must produce identical results to the original sequence,
    but can use a single GPU kernel launch instead of multiple.
    """
    comptime FUSED_COUNT: Int  # how many original ops this replaces
```

#### Parameterized Fused Activation: `FusedMatMulBiasActivation`

Rather than duplicating ~500 lines per activation, fused matmul+bias+activation ops
are parameterized on an `Activation` trait:

```mojo
# nn/autodiff/fused/activation.mojo
trait Activation(Movable & ImplicitlyCopyable):
    comptime OP_ID: Int          # Matches standalone DiffOp OP_ID (RELU=10, TANH=11...)
    comptime FUSED_OP_ID: Int    # OP_ID for the fused variant (101, 102...)

    @staticmethod
    fn forward(pre_act: Scalar[dtype]) -> Scalar[dtype]

    @staticmethod
    fn cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]

    @staticmethod
    fn backward(cache_val: Scalar[dtype], grad_out: Scalar[dtype]) -> Scalar[dtype]

# Concrete activations:
# - ReLUActivation:    forward = max(0,x), cache = pre_act, backward = g if cache>0 else 0
# - TanhActivation:    forward = tanh(x),  cache = output,  backward = g*(1-cache^2)
# - SigmoidActivation: forward = σ(x),     cache = output,  backward = g*cache*(1-cache)
```

The single `FusedMatMulBiasActivation[in_dim, out_dim, ACT: Activation]` struct
(~500 lines) replaces what was ~1500 lines across 3 separate files. Only ~8 lines
differ between activations — the calls to `ACT.forward()`, `ACT.cache()`, and
`ACT.backward()` in the CPU eval/vjp and GPU kernels.

**Backward-compatible wrappers:** `FusedMatMulBiasReLU` and `FusedMatMulBiasTanh`
remain as concrete structs (thin delegation wrappers) rather than `comptime` aliases,
because Mojo nightly doesn't fold `comptime` member constants through parameterized
type aliases (see "Comptime alias limitation" in Open Questions below).

New activations can now be added with ~30 lines (the `Activation` impl) instead of
~500 lines (a full fused struct). `FusedMatMulBiasSigmoid` was the first new
activation added this way.

#### Example: Fused MatMul + BiasAdd + ReLU

```mojo
struct FusedMatMulBiasReLU[in_dim: Int, out_dim: Int](FusedOp):
    """Single-kernel implementation of y = relu(x @ W + b).

    Replaces: MatMul[in_dim, out_dim] → BiasAdd[out_dim] → ReLUOp[out_dim]
    Benefits:
    - 1 kernel launch instead of 3
    - Input loaded from global memory once (not 3 times)
    - Intermediate values stay in registers
    - Bias add and ReLU computed inline after matmul accumulation
    """
    comptime IN_DIM: Int = in_dim
    comptime OUT_DIM: Int = out_dim
    comptime PARAM_SIZE: Int = in_dim * out_dim + out_dim  # W + b
    comptime CACHE_SIZE: Int = in_dim + out_dim  # input + pre-activation
    comptime FUSED_COUNT: Int = 3

    @always_inline
    @staticmethod
    fn eval_kernel[BATCH: Int](
        output: LayoutTensor[dtype, Layout.row_major(BATCH, out_dim), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, in_dim), ImmutAnyOrigin],
        W: LayoutTensor[dtype, Layout.row_major(in_dim, out_dim), ImmutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(out_dim), ImmutAnyOrigin],
        cache_input: LayoutTensor[dtype, Layout.row_major(BATCH, in_dim), MutAnyOrigin],
        cache_pre_act: LayoutTensor[dtype, Layout.row_major(BATCH, out_dim), MutAnyOrigin],
    ):
        # Standard tiled matmul... but with fused bias+relu at the end
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        # ... (tiled matmul body, same as MatMul.eval_kernel) ...

        if global_row < BATCH and global_col < out_dim:
            # FUSED: bias add + relu in one shot, no intermediate write
            var pre_act = acc + b[global_col]
            cache_pre_act[global_row, global_col] = pre_act  # for backward
            output[global_row, global_col] = pre_act if pre_act > 0 else 0

    @staticmethod
    fn vjp[BATCH: Int](grad_output, mut grad_input, params, cache, mut grad_params):
        # Backward for fused op: relu_grad → bias_grad → matmul_grad
        # All in one pass over the data
        var W = LayoutTensor[
            dtype, Layout.row_major(in_dim, out_dim), MutAnyOrigin
        ](params.ptr)
        var cached_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, in_dim), MutAnyOrigin
        ](cache.ptr)
        var cached_pre_act = LayoutTensor[
            dtype, Layout.row_major(BATCH, out_dim), MutAnyOrigin
        ](cache.ptr + BATCH * in_dim)
        var dW = LayoutTensor[
            dtype, Layout.row_major(in_dim, out_dim), MutAnyOrigin
        ](grad_params.ptr)

        # Fused backward: compute relu mask, apply to gradients,
        # then matmul backward in one pass
        for b in range(BATCH):
            for j in range(out_dim):
                # ReLU backward: mask the gradient
                var masked_grad = grad_output[b, j] if cached_pre_act[b, j] > 0 else 0

                # Bias gradient (accumulate)
                grad_params[in_dim * out_dim + j] = (
                    grad_params[in_dim * out_dim + j] + masked_grad
                )

                # Weight gradient: dW += input.T @ masked_grad
                for i in range(in_dim):
                    dW[i, j] = dW[i, j] + cached_input[b, i] * masked_grad

            # Input gradient: grad_input = masked_grad_output @ W.T
            for i in range(in_dim):
                var acc: Scalar[dtype] = 0
                for j in range(out_dim):
                    var masked = grad_output[b, j] if cached_pre_act[b, j] > 0 else 0
                    acc += masked * W[i, j]
                grad_input[b, i] = acc
```

### The Fusion Pass

The fusion pass runs at compile time using `comptime if` to pattern-match on adjacent ops:

```mojo
struct FusedAutoDiffChain[*OPS: DiffOp](Model):
    """AutoDiffChain with compile-time fusion optimization.

    Before generating code, scans the op list for known fusible patterns
    and replaces them with FusedOp implementations.
    """
    comptime ops = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N: Int = Variadic.size(Self.ops)

    # ── Compile-time fusion analysis ──
    #
    # The idea: build a "fused ops" list at compile time.
    # Walk the original ops, and when a fusible pattern is found,
    # replace 2-3 ops with a single FusedOp.
    #
    # Pattern matching uses comptime if with type checks:

    @staticmethod
    fn _is_matmul_bias_relu_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul → BiasAdd → ReLU.

        Uses OP_ID for reliable type identification — no fragile string
        comparison via get_type_name[], no structural guessing.
        """
        comptime if idx + 2 >= Self.N:
            return False

        # Clean integer comparison on OP_ID
        return (
            Self.ops[idx].OP_ID == OpID.MATMUL._value
            and Self.ops[idx + 1].OP_ID == OpID.BIAS_ADD._value
            and Self.ops[idx + 2].OP_ID == OpID.RELU._value
        )

    @staticmethod
    fn _is_matmul_bias_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+2] is MatMul → BiasAdd."""
        comptime if idx + 1 >= Self.N:
            return False
        return (
            Self.ops[idx].OP_ID == OpID.MATMUL._value
            and Self.ops[idx + 1].OP_ID == OpID.BIAS_ADD._value
        )

    @staticmethod
    fn _is_matmul_bias_tanh_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul → BiasAdd → Tanh."""
        comptime if idx + 2 >= Self.N:
            return False
        return (
            Self.ops[idx].OP_ID == OpID.MATMUL._value
            and Self.ops[idx + 1].OP_ID == OpID.BIAS_ADD._value
            and Self.ops[idx + 2].OP_ID == OpID.TANH._value
        )

    # The fused chain delegates to AutoDiffChain with
    # substituted op types where fusion applies.
    # This creates a NEW type at compile time.
```

### How Fusion Works in Practice

The fusion mechanism works through **type-level rewriting**. Given:

```mojo
comptime Original = AutoDiffChain[
    MatMul[784, 256], BiasAdd[256], ReLUOp[256],    # ← fusible
    MatMul[256, 128], BiasAdd[128], ReLUOp[128],    # ← fusible
    MatMul[128, 10],  BiasAdd[10],                   # ← fusible (matmul+bias)
]
```

The fusion pass produces the equivalent of:

```mojo
comptime Fused = AutoDiffChain[
    FusedMatMulBiasReLU[784, 256],                   # 1 kernel instead of 3
    FusedMatMulBiasReLU[256, 128],                   # 1 kernel instead of 3
    FusedMatMulBias[128, 10],                         # 1 kernel instead of 2
]
```

**GPU kernel launches: 8 → 3.** And each fused kernel reads from global memory once instead of multiple times.

### Implementing the Pass with Comptime

The most practical approach with current Mojo is a **builder pattern** that checks for fusion at construction:

```mojo
# User-facing API: auto_fuse wraps AutoDiffChain with fusion
struct auto_fuse[*OPS: DiffOp](Model):
    """Convenience wrapper that applies fusion before building the chain."""

    # At compile time, scan for patterns and build fused list.
    # The comptime block runs ONCE during compilation.

    comptime ops = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N: Int = Variadic.size(Self.ops)

    # For patterns we CAN'T rewrite the type list (Mojo limitation),
    # we use a different strategy: the forward/backward methods
    # check for fusible sequences inline.

    @staticmethod
    fn forward_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut output: LayoutTensor[...],
        input: LayoutTensor[...],
        params: LayoutTensor[...],
        mut cache: LayoutTensor[...],
        workspace: DeviceBuffer[dtype],
    ) raises:
        var inter_ptr = workspace.unsafe_ptr()

        # Walk ops, fusing where possible
        var i = 0
        comptime for _i in range(Self.N):
            comptime if Self._is_matmul_bias_relu_at[_i]():
                # FUSED: launch single MatMul+Bias+ReLU kernel
                FusedMatMulBiasReLU[
                    Self.ops[_i].IN_DIM, Self.ops[_i].OUT_DIM
                ].eval_gpu[BATCH](ctx, ...)
                # Skip next 2 ops (they're consumed by the fused kernel)
                # (handled by compile-time index tracking)
            else:
                # UNFUSED: launch individual kernel
                Self.ops[_i].eval_gpu[BATCH](ctx, ...)
```

### Alternative: Explicit Fusion comptimees

A simpler approach that works TODAY with no language extensions:

```mojo
# Provide pre-fused "comptimees" that users can use directly.
# The fusion is explicit but zero-effort to use.

comptime DenseReLU[i: Int, o: Int] = AutoDiffChain[
    FusedMatMulBiasReLU[i, o]
]

comptime DenseTanh[i: Int, o: Int] = AutoDiffChain[
    FusedMatMulBiasTanh[i, o]
]

comptime Dense[i: Int, o: Int] = AutoDiffChain[
    FusedMatMulBias[i, o]
]

# User writes:
comptime MyModel = Sequential[
    DenseReLU[784, 256],
    DenseReLU[256, 128],
    Dense[128, 10],
]
# Gets: 3 kernel launches, auto-generated backward, full GPU support
```

---

## Layer 4 (Future): Graph IR for Non-Sequential Topologies

Sequential composition covers most cases, but some architectures need non-linear data flow: residual connections, multi-head attention, U-Net skip connections.

### The Problem

```
# ResNet block:  x → Linear → ReLU → Linear → (+) → ReLU
#                |___________________________|↑
#                        (skip connection)
```

This CAN'T be expressed as a flat sequence of DiffOps.

### The Nested Variadic Problem

The naive approach would be:
```mojo
struct Residual[*INNER_OPS: DiffOp](DiffOp):
    comptime inner = AutoDiffChain[*INNER_OPS]  # ← nested variadic
```

But Mojo **cannot resolve variadic type packs through multiple levels of nesting**.
Accessing `Residual.IN_DIM` would fail with "unbound parameter" because it tries to
resolve through `AutoDiffChain`'s variadic pack, which is itself inside `Residual`'s
variadic pack.

This is the exact problem documented in `physics3d/model/model_def.mojo`:
> "Mojo cannot resolve variadic type packs through multiple levels of nesting
> (accessing ModelDef.NQ would fail with 'unbound parameter' if ModelDef
> contained Bodies/Joints directly)."

### Solution: The Trait Gateway Pattern (from ModelDef)

The solution is the same pattern used in your physics engine. Instead of nesting
variadic packs, use **trait bounds** as the gateway between composition levels:

1. **`AutoDiffChain[*OPS]`** resolves its own variadics → exposes `IN_DIM`, `OUT_DIM`,
   `PARAM_SIZE`, `CACHE_SIZE` as comptime members → conforms to `Model`
2. **Combinators take `Model`-bounded types**, NOT raw variadic packs

```mojo
# ══════════════════════════════════════════════════════════
#  DiffChainLike — the "BodiesLike" equivalent for autograd
# ══════════════════════════════════════════════════════════
#
# This trait is what combinators bind to. It exposes the
# pre-resolved dimensions without needing to see the variadic
# pack inside. This IS the Model trait — no new trait needed!
#
# AutoDiffChain[*OPS: DiffOp] already conforms to Model.
# So combinators just take Model parameters.

struct Residual[Inner: Model](Model):
    """y = f(x) + x  where f conforms to Model.

    The Inner type has already resolved all its variadic internals.
    Residual only sees the trait interface: IN_DIM, OUT_DIM, PARAM_SIZE, etc.

    This is the same pattern as ModelDef[Bodies: BodiesLike, Joints: JointsLike]:
    - Bodies[*B: BodySpec] resolves its variadics → conforms to BodiesLike
    - ModelDef sees BodiesLike, never the raw variadic pack
    """
    comptime OP_ID: Int = OpID.RESIDUAL._value

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE + Self.IN_DIM  # + skip cache

    # Compile-time dimension check
    comptime assert Self.IN_DIM == Self.OUT_DIM, (
        "Residual requires IN_DIM == OUT_DIM for skip connection, got "
        + str(Self.IN_DIM) + " != " + str(Self.OUT_DIM)
    )

    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.Inner.WORKSPACE_SIZE_PER_SAMPLE

    @staticmethod
    fn forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ):
        # Inner chain cache view
        var inner_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)

        # Forward through inner chain
        Self.Inner.forward[BATCH](input, output, params, inner_cache)

        # Add skip connection: output += x
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                output[b, i] = output[b, i] + input[b, i]

        # Cache input for backward skip gradient
        var skip_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                skip_cache[b, i] = input[b, i]

    @staticmethod
    fn backward[BATCH: Int](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        var inner_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)

        # Backward through inner chain (auto-generated by AutoDiffChain)
        Self.Inner.backward[BATCH](grad_output, grad_input, params, inner_cache, grads)

        # Add skip gradient: d/dx (f(x) + x) = f'(x) + I
        # grad_input already has f'(x) from inner backward, add identity
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                grad_input[b, i] = grad_input[b, i] + grad_output[b, i]

    # GPU methods follow same pattern, delegating to Self.Inner.*_gpu


# ══════════════════════════════════════════════════════════
#  Parallel — two branches, concatenated output
# ══════════════════════════════════════════════════════════

struct Parallel[BranchA: Model, BranchB: Model](Model):
    """y = concat(f(x), g(x))  where f, g both conform to Model.

    Both branches receive the full input. Outputs are concatenated.
    BranchA and BranchB have already resolved their variadics.
    """
    comptime OP_ID: Int = OpID.PARALLEL._value

    comptime assert Self.BranchA.IN_DIM == Self.BranchB.IN_DIM, (
        "Parallel branches must have same IN_DIM"
    )

    comptime IN_DIM: Int = Self.BranchA.IN_DIM
    comptime OUT_DIM: Int = Self.BranchA.OUT_DIM + Self.BranchB.OUT_DIM
    comptime PARAM_SIZE: Int = Self.BranchA.PARAM_SIZE + Self.BranchB.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.BranchA.CACHE_SIZE + Self.BranchB.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.BranchA.WORKSPACE_SIZE_PER_SAMPLE
        + Self.BranchB.WORKSPACE_SIZE_PER_SAMPLE
    )

    # forward: run both branches, concatenate into output
    # backward: split grad_output, run both branch backwards
    # (each branch only gets its portion of the output gradient)


# ══════════════════════════════════════════════════════════
#  Repeat — weight-shared iteration
# ══════════════════════════════════════════════════════════

struct Repeat[N: Int, Inner: Model](Model):
    """y = f(f(...f(x)...))  applied N times with shared weights.

    Useful for transformer layers, recurrent unrolling, etc.
    """
    comptime assert Self.Inner.IN_DIM == Self.Inner.OUT_DIM, (
        "Repeat requires IN_DIM == OUT_DIM, got "
        + str(Self.Inner.IN_DIM) + " != " + str(Self.Inner.OUT_DIM)
    )

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE  # shared!
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE * N  # one cache per iteration
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.Inner.WORKSPACE_SIZE_PER_SAMPLE

    # forward: loop N times, caching each iteration
    # backward: loop N times in reverse, accumulating to shared grads
```

### Usage: The Two-Step Composition

This follows the exact same pattern as your half_cheetah_def.mojo:

```mojo
# Step 1: Build inner chains (variadic packs resolve HERE)
comptime ResBlockInner[dim: Int] = AutoDiffChain[
    MatMul[dim, dim], BiasAdd[dim], ReLUOp[dim],
    MatMul[dim, dim], BiasAdd[dim],
]
# ResBlockInner conforms to Model. Its variadics are fully resolved.
# ResBlockInner[256].IN_DIM == 256, .OUT_DIM == 256, etc.

# Step 2: Wrap in combinator (sees Model trait, NOT variadic pack)
comptime ResBlock[dim: Int] = Residual[ResBlockInner[dim]]

# Step 3: Compose into full model
comptime ResNet = Sequential[
    LinearAD[784, 256],
    ResBlock[256],
    ResBlock[256],
    ResBlock[256],
    LinearAD[256, 10],
]
```

Compare with the physics engine pattern:
```mojo
# Physics (same structure):
comptime CheetahBodies = Bodies[Torso, BThigh, BShin, ...]  # resolves variadics
comptime CheetahJoints = Joints[RootX, RootZ, ...]          # resolves variadics
comptime CheetahModel = ModelDef[CheetahBodies, CheetahJoints, ...]  # sees traits

# Autograd (same structure):
comptime FFN = AutoDiffChain[MatMul[256, 512], ReLUOp[512], MatMul[512, 256]]  # resolves variadics
comptime TransformerBlock = Residual[FFN]                                       # sees Model trait
comptime Transformer = Repeat[12, TransformerBlock]                             # sees Model trait
```

### What This Enables

With `Residual`, `Parallel`, `Repeat`, and `AutoDiffChain`:

- **ResNets**: `Residual[AutoDiffChain[dense layers]]`
- **U-Nets**: `Parallel` for encoder/decoder skip connections
- **Transformers**: `Repeat[N, Residual[AutoDiffChain[attn + ffn]]]`

All with auto-generated backward passes. All at compile time. No nested variadic issues.

---

## Migration Path

### Phase 1: Foundation (no breaking changes)

1. Define `DiffOp` trait in `nn/autodiff/op.mojo`
2. Implement core primitives: `MatMul`, `BiasAdd`, `ReLUOp`, `TanhOp`, `SigmoidOp`
3. Implement `AutoDiffChain` in `nn/autodiff/chain.mojo`
4. Verify: `AutoDiffChain[MatMul[2,64], BiasAdd[64], ReLUOp[64]]` produces identical outputs to `LinearReLU[2, 64]`

### Phase 2: Fusion

5. Implement `FusedMatMulBias`, `FusedMatMulBiasReLU`, `FusedMatMulBiasTanh`
6. Create convenience comptimees: `Dense`, `DenseReLU`, `DenseTanh`
7. Benchmark: fused AutoDiffChain vs hand-coded layers

### Phase 3: Combinators

8. Implement `Residual` combinator
9. Implement `Parallel` combinator
10. Build a small ResNet or transformer block as proof of concept

### Phase 4: Replace Hand-Coded Layers

11. Gradually replace `Linear`, `LinearReLU`, etc. with AutoDiffChain equivalents
12. Keep old implementations as reference/fallback
13. The `Sequential` container works unchanged — it accepts any `Model`

---

## Directory Structure

```
nn/
├── autodiff/
│   ├── __init__.mojo          # Public API: DiffOp, AutoDiffChain, comptimees
│   ├── op.mojo                # DiffOp trait definition
│   ├── chain.mojo             # AutoDiffChain (the autograd engine)
│   ├── primitives/
│   │   ├── matmul.mojo        # MatMul
│   │   ├── bias.mojo          # BiasAdd
│   │   ├── activations.mojo   # ReLUOp, TanhOp, SigmoidOp, MishOp
│   │   ├── norm.mojo          # LayerNormOp, RMSNormOp
│   │   └── reduce.mojo        # ReduceSum, ReduceMean
│   ├── fused/
│   │   ├── activation.mojo          # Activation trait + ReLU/Tanh/Sigmoid activations
│   │   ├── matmul_bias.mojo         # FusedMatMulBias (no activation, separate)
│   │   ├── matmul_bias_act.mojo     # FusedMatMulBiasActivation[i, o, ACT] (parameterized)
│   │   ├── matmul_bias_relu.mojo    # FusedMatMulBiasReLU (thin wrapper)
│   │   └── matmul_bias_tanh.mojo    # FusedMatMulBiasTanh (thin wrapper)
│   ├── combinators/
│   │   ├── residual.mojo      # Residual skip connection
│   │   ├── parallel.mojo      # Parallel branches + concat
│   │   └── repeat.mojo        # Weight-shared repetition
│   └── comptimees.mojo           # Dense, DenseReLU, DenseTanh, ResBlock
├── model/                     # Existing layers (untouched)
├── optimizer/                 # Existing optimizers (untouched)
├── loss/                      # Existing losses (untouched)
├── training/                  # Existing training infra (untouched)
└── gpu/                       # Existing GPU kernels (untouched)
```

---

## Comparison: Before and After

### Defining a New Layer

**Before (hand-coded):**
```mojo
struct LinearLayerNormReLU[in_d: Int, out_d: Int](Model):
    comptime IN_DIM = in_d
    comptime OUT_DIM = out_d
    comptime PARAM_SIZE = in_d * out_d + out_d + out_d * 2  # W + b + gamma + beta
    comptime CACHE_SIZE = in_d + out_d * 3  # input + pre_norm + normalized + pre_relu

    # ~50 lines: forward CPU
    # ~70 lines: backward CPU (chain rule through 3 operations)
    # ~80 lines: forward GPU kernel
    # ~100 lines: backward GPU kernel
    # Total: ~300 lines, error-prone manual gradient derivation
```

**After (composed):**
```mojo
comptime LinearLayerNormReLU[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d],
    BiasAdd[out_d],
    LayerNormOp[out_d, 1e-5],
    ReLUOp[out_d],
]
# Total: 5 lines. Backward auto-generated. Compile-time dimension checking.
```

### Training Loop

**Unchanged.** This is critical. The training loop doesn't know or care whether the model was hand-coded or auto-differentiated:

```mojo
comptime MyModel = Sequential[
    DenseReLU[784, 256],     # AutoDiffChain-based
    DenseReLU[256, 128],     # AutoDiffChain-based
    Linear[128, 10],         # Original hand-coded (still works!)
]

var state = NetworkState[MyModel, Adam[0.001]]()
state.initialize[Kaiming]()

# Training loop is identical
for epoch in range(100):
    state.zero_grads()
    Network[MyModel, Adam[0.001]].forward_with_cache[BATCH](input, output, ...)
    MSELoss.backward[BATCH, 10](output, target, grad_output)
    Network[MyModel, Adam[0.001]].backward[BATCH](grad_output, grad_input, ...)
    state.optimizer_step()
```

---

## What Makes This Different From PyTorch/JAX/tinygrad

| Feature | PyTorch | JAX | tinygrad | **This (Mojo)** |
|---------|---------|-----|----------|-----------------|
| When autodiff runs | Runtime (tape) | JIT time (tracing) | Runtime (lazy) | **Compile time** |
| Overhead | Dynamic dispatch, tape allocation | Tracing cost, XLA compile | Python interpreter | **Zero** |
| Fusion | torch.compile (optional) | XLA (opaque) | Scheduler + BEAM | **Explicit + composable** |
| Shape checking | Runtime errors | Shape tracing | Runtime errors | **Compile-time errors** |
| New layer backward | autograd (magic) | jaxpr (magic) | autograd (magic) | **Generated from VJPs** |
| Hackability | Complex C++ internals | Complex XLA/MLIR | Simple Python | **Simple Mojo** |

The key differentiator: **everything resolves at compile time into the same zero-cost code you'd write by hand.** No tape. No tracing. No interpreter. Just specialized, fused GPU kernels.

---

## Open Questions and Resolved Issues

### Resolved

1. **~~Type-level pattern matching~~** → **Solved: OP_ID enum.**
   Each DiffOp declares `comptime OP_ID: Int`. Fusion passes use clean integer
   comparison (`Self.ops[i].OP_ID == OpID.MATMUL._value`). No fragile
   `get_type_name[]` string comparison needed. `get_type_name[]` from Mojo's
   reflection module remains available as a fallback for debugging/logging.

2. **~~Recursive/nested variadic packs~~** → **Solved: Trait Gateway Pattern.**
   Combinators (`Residual`, `Parallel`, `Repeat`) take `Model`-bounded type
   parameters, NOT raw variadic packs. `AutoDiffChain[*OPS]` resolves its own
   variadics and conforms to `Model`. The combinator only sees the trait interface.
   This is the same pattern proven in `physics3d/model/model_def.mojo` where
   `ModelDef[Bodies: BodiesLike, Joints: JointsLike]` avoids nesting variadic
   packs by having `Bodies[*B]` and `Joints[*J]` resolve their own packs first.

3. **~~constrained[]~~** → **Replaced with `comptime assert`.**
   `constrained[]` is deprecated. All compile-time assertions in this design use
   `comptime assert condition, "error message"` instead.

4. **~~Comptime alias member folding~~** → **Documented limitation, workaround found.**
   Parameterized `comptime` type aliases like
   `comptime FusedMatMulBiasReLU[i, o] = FusedMatMulBiasActivation[i, o, ReLUActivation]`
   don't fold their member constants (`IN_DIM`, `OUT_DIM`, etc.) when used in
   compile-time contexts like `AutoDiffChain[FusedMatMulBiasReLU[2, 8], ...]`.
   The compiler keeps expressions like `FusedMatMulBiasActivation[2, 8, ReLUActivation].IN_DIM`
   symbolic instead of folding to `2`, causing "unfolded expression at parser time" errors.
   **Workaround**: Keep backward-compatible names as concrete structs (thin wrappers)
   that delegate all methods to `FusedMatMulBiasActivation`. The wrapper struct has its
   own `comptime IN_DIM: Int = Self.in_dim` which folds correctly. New activations
   (like `FusedMatMulBiasSigmoid`) can use `comptime` aliases if they're always used
   via `FusedMatMulBiasActivation[..., SigmoidActivation]` directly, or they need
   their own thin wrapper if used in `AutoDiffChain[...]`.

### Still Open

5. **Variadic type rewriting for automatic fusion**: Can we build a NEW variadic
   type list at compile time from an existing one? For example, scanning
   `[MatMul, BiasAdd, ReLU, MatMul, BiasAdd]` and producing
   `[FusedMatMulBiasReLU, FusedMatMulBias]` automatically. Current finding is that
   variadic type packs cannot be built incrementally, but this deserves further
   exploration. Possible angles:
   - **Recursive struct construction**: A struct that peels off 2-3 ops at a time
     and builds a nested `Seq2[FusedOp, Seq2[FusedOp, ...]]` chain via `comptime if`
     pattern matching on OP_IDs, similar to how `Sequential` nests `Seq2`.
   - **Mojo language evolution**: Future Mojo versions may support `comptime` type
     list construction or `comptime` variadic pack manipulation.
   - **Hybrid approach**: `FusionAnalyzer` detects patterns at compile time, then a
     macro-like `comptime fn` emits the fused chain for known sizes (1-8 layers).
   The pragmatic fallback remains explicit fused aliases (`Dense`, `DenseReLU`,
   `FusedChain.mlp_relu`, etc.) which already cover common RL architectures well.

6. **GPU kernel fusion across ops**: `@always_inline` kernels can theoretically
   be inlined into a single kernel by the compiler. Whether Mojo actually does
   this for `ctx.enqueue_function[]` calls needs testing. If not, the explicit
   `FusedOp` approach (single kernel per fused pattern) is the reliable path.

7. **Compile time for large models**: A 100-layer model means 100 iterations of
   `comptime for`. How does Mojo handle compile-time evaluation at this scale?
   Your Sequential already does this for moderate-sized models — worth
   benchmarking compilation time as model depth grows.

8. **DiffOp ↔ Model bridge**: `AutoDiffChain` conforms to `Model`, which means
   combinators can compose freely. But can a `Model` that ISN'T an `AutoDiffChain`
   (e.g., your existing hand-coded `Linear`) also be used inside `Residual`?
   Yes — since `Residual` takes `Inner: Model`, any `Model` works. This gives
   a clean migration path: wrap existing hand-coded layers in combinators today,
   migrate internals to `AutoDiffChain` later.
