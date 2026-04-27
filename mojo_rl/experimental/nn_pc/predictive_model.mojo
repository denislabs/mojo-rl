"""PCN base traits and activation types.

A "PCN level" bundles a weight matrix W and an activation f. The activation
acts on the *prediction side* (lower dim, IN_DIM), so it's bundled into the
layer rather than a standalone Sequential element — that keeps the dim
unambiguous and matches the PyTorch reference (PCNLayer = W + activation).

Convention (mirrors `nn.Linear[in, out]` exactly):
  - W shape: [IN_DIM, OUT_DIM]
  - IN_DIM = lower (predicted) dim — the "input" side in feedforward terms
  - OUT_DIM = upper (latent above) dim — the "output" side in feedforward terms
  - PCN predict goes top-down via W^T:  x_hat = ACT(x_above @ W^T)
    where x_above is [B, OUT_DIM] and x_hat is [B, IN_DIM].

Layers:
  - PCLinear[IN, OUT, ACT=PCReLU] — one PCN level
  - For the readout, use ACT=PCIdentity: PCLinear[NUM_CLASSES, TOP_HIDDEN, PCIdentity]

Activations:
  - PCReLU, PCIdentity (PCActivation trait — extensible).
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import TPB
from mojo_rl.nn.initializer import Initializer


# =============================================================================
# Activation trait
# =============================================================================


trait PCActivation(Movable & ImplicitlyCopyable):
    """Activation function used inside a PCLinear level.

    Provides CPU + GPU variants of:
      - apply: x_hat = f(a)
      - apply_derivative_mul: h = eps * f'(a) (gain-modulated error)
    """

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        ...

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        ...


struct PCReLU(PCActivation):
    """ReLU: f(a) = max(0, a); f'(a) = 1 if a > 0 else 0."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            for i in range(DIM):
                var v = a[b, i]
                x_hat[b, i] = v if v > 0 else Scalar[dtype](0)

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            for i in range(DIM):
                h[b, i] = eps[b, i] if a[b, i] > 0 else Scalar[dtype](0)

    # ── GPU ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_kernel[
        BATCH: Int, DIM: Int, dtype: DType,
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var v = rebind[Scalar[dtype]](a[b, i])
        x_hat[b, i] = v if v > 0 else Scalar[dtype](0)

    @staticmethod
    def _apply_deriv_mul_kernel[
        BATCH: Int, DIM: Int, dtype: DType,
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var av = rebind[Scalar[dtype]](a[b, i])
        h[b, i] = eps[b, i] if av > 0 else Scalar[dtype](0)

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        comptime k = Self._apply_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k, k](
            a, x_hat,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        comptime k = Self._apply_deriv_mul_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k, k](
            a, eps, h,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )


struct PCIdentity(PCActivation):
    """Identity: f(a) = a; f'(a) = 1.  Used for the readout layer."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            for i in range(DIM):
                x_hat[b, i] = a[b, i]

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            for i in range(DIM):
                h[b, i] = eps[b, i]

    # ── GPU ───────────────────────────────────────────────────────────────

    @staticmethod
    def _copy_kernel[
        BATCH: Int, DIM: Int, dtype: DType,
    ](
        src: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        dst: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        dst[b, i] = src[b, i]

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        comptime k = Self._copy_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k, k](
            a, x_hat,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ) raises:
        # Identity derivative -> h = eps (copy).
        comptime k = Self._copy_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k, k](
            eps, h,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )


# =============================================================================
# PCLayer trait — composable element of a PCSequential
# =============================================================================


trait PCLayer(Movable & ImplicitlyCopyable):
    """A PCN level — owns a W matrix + a bundled activation.

    Compile-time constants:
      - IN_DIM: predicted dim (the "lower" side in PCN; matches nn.Linear in_dim)
      - OUT_DIM: latent-above dim (matches nn.Linear out_dim)
      - PARAM_SIZE: total params (= IN_DIM * OUT_DIM for PCLinear; no biases)

    Required ops (declared on the trait so PCSequential / PCTrainer can dispatch
    through `layer_types[i].method[...](...)` at compile time).
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def gain_modulated_error[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def weight_grad_step[
        BATCH: Int, dtype: DType = DType.float32
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ):
        ...

    # ── GPU dispatches ─────────────────────────────────────────────────────

    @staticmethod
    def predict_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        ...

    @staticmethod
    def gain_modulated_error_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        ...

    @staticmethod
    def pull_back_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        ...

    @staticmethod
    def weight_grad_step_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ) raises:
        ...
