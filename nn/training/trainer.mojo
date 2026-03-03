"""Trainer: all-static training loops for neural networks.

All methods are @staticmethod — no stored state.  The caller owns and passes
NetworkState (CPU) or GPUNetworkState (GPU) directly, so GPU-only pipelines
never allocate a CPU state.

Usage:
    from nn import seq, Linear, ReLU, Adam, MSELoss, Kaiming
    from nn.training import Trainer, NetworkState, GPUNetworkState

    alias M = typeof(seq(Linear[4, 64](), ReLU[64](), Linear[64, 2]()))

    # CPU training — init_state creates and initializes NetworkState in one call
    var state = Trainer[M, Adam, MSELoss].init_state[Kaiming]()
    var result = Trainer[M, Adam, MSELoss].train[BATCH](
        mut state, input_t, target_t, epochs=100, print_every=10
    )

    # GPU-only training — init_state_gpu creates GPUNetworkState directly,
    # no persistent CPU NetworkState needed
    var gpu = Trainer[M, Adam, MSELoss].init_state_gpu[Kaiming](ctx)
    var result = Trainer[M, Adam, MSELoss].train_gpu[BATCH](
        mut gpu, ctx, input_t, target_t, epochs=100, print_every=10
    )

    # Evaluate — accepts params LayoutTensor, works for both CPU and GPU state
    var loss = Trainer[M, Adam, MSELoss].evaluate[BATCH](
        state.params_view(), input_t, target_t   # or gpu.params_view() if on CPU
    )
"""

from ..model import Model
from ..optimizer import Optimizer
from ..loss import LossFunction
from ..initializer import Initializer, Xavier
from ..constants import dtype
from .network_state import NetworkState
from .gpu_network_state import GPUNetworkState

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer


struct TrainResult(ImplicitlyCopyable, Movable):
    """Result of a training run."""

    var final_loss: Float64
    var epochs_trained: Int

    fn __init__(out self, final_loss: Float64, epochs_trained: Int):
        self.final_loss = final_loss
        self.epochs_trained = epochs_trained


struct Trainer[
    MODEL: Model,
    OPTIMIZER: Optimizer,
    LOSS_FUNCTION: LossFunction,
]:
    """All-static training loop namespace.

    No stored state — the caller manages NetworkState (CPU) or GPUNetworkState
    (GPU) and passes it to each method.  This means GPU-only training never
    allocates a CPU NetworkState.

    Parameters:
        MODEL: Stateless model architecture (implements Model trait).
        OPTIMIZER: Stateless optimizer (implements Optimizer trait).
        LOSS_FUNCTION: Stateless loss function (implements LossFunction trait).
    """

    # =========================================================================
    # State Initialization Helpers
    # =========================================================================

    @staticmethod
    fn init_state[
        INITIALIZER: Initializer = Xavier
    ]() -> NetworkState[Self.MODEL, Self.OPTIMIZER]:
        """Create and initialize a CPU NetworkState.

        Parameters:
            INITIALIZER: Weight initialization strategy (default: Xavier).

        Returns:
            Initialized NetworkState ready for CPU training or upload to GPU.
        """
        var state = NetworkState[Self.MODEL, Self.OPTIMIZER]()
        state.initialize[INITIALIZER]()
        return state^

    @staticmethod
    fn init_state_gpu[
        INITIALIZER: Initializer = Xavier
    ](ctx: DeviceContext) raises -> GPUNetworkState[Self.MODEL, Self.OPTIMIZER]:
        """Create a GPUNetworkState with initialized weights, no persistent CPU state.

        Allocates a transient CPU NetworkState, initializes weights, uploads to
        GPU, then discards the CPU copy.  The caller never needs to manage a
        CPU NetworkState for GPU-only training.

        Parameters:
            INITIALIZER: Weight initialization strategy (default: Xavier).

        Args:
            ctx: GPU device context.

        Returns:
            GPUNetworkState with initialized weights on device.
        """
        var state = NetworkState[Self.MODEL, Self.OPTIMIZER]()
        state.initialize[INITIALIZER]()
        var state = GPUNetworkState[Self.MODEL, Self.OPTIMIZER](ctx)
        state.upload_from(state, ctx)
        return state^

    # =========================================================================
    # CPU Training
    # =========================================================================

    @staticmethod
    fn train[
        BATCH: Int
    ](
        mut state: NetworkState[Self.MODEL, Self.OPTIMIZER],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        epochs: Int = 100,
        print_every: Int = 0,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ) -> TrainResult:
        """Train on CPU for the given number of epochs.

        Intermediate buffers are heap-allocated internally to avoid stack
        overflow.  The caller's NetworkState is updated in-place.

        Args:
            state: Network state (params, grads, optimizer state) — updated.
            input: Input tensor [BATCH, IN_DIM] — caller manages memory.
            target: Target tensor [BATCH, OUT_DIM] — caller manages memory.
            epochs: Number of training epochs.
            print_every: Print loss every N epochs (0 = never).
            checkpoint_every: Save checkpoint every N epochs (0 = never).
            checkpoint_path: Base path for checkpoint files.

        Returns:
            TrainResult with final_loss and epochs_trained.
        """
        comptime OUT_SIZE = BATCH * Self.MODEL.OUT_DIM
        comptime IN_SIZE = BATCH * Self.MODEL.IN_DIM
        comptime CACHE_SIZE_ = BATCH * Self.MODEL.CACHE_SIZE

        # Heap-allocated intermediate buffers
        var output_data = List[Scalar[dtype]](capacity=OUT_SIZE)
        var grad_out_data = List[Scalar[dtype]](capacity=OUT_SIZE)
        var grad_in_data = List[Scalar[dtype]](capacity=IN_SIZE)
        var cache_data = List[Scalar[dtype]](capacity=CACHE_SIZE_)
        for _ in range(OUT_SIZE):
            output_data.append(Scalar[dtype](0))
            grad_out_data.append(Scalar[dtype](0))
        for _ in range(IN_SIZE):
            grad_in_data.append(Scalar[dtype](0))
        for _ in range(CACHE_SIZE_):
            cache_data.append(Scalar[dtype](0))

        # 2D LayoutTensor views (zero-copy, created once from List.unsafe_ptr())
        var output_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output_data.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](grad_out_data.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_in_data.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        # State views — lvalue vars required (params/state are mut in step())
        var params = state.params_view()
        var grads = state.grads_view()
        var opt_state = state.state_view()

        var final_loss: Float64 = 0.0

        for epoch in range(epochs):
            Self.MODEL.forward[BATCH](input, output_t, params, cache_t)

            # Loss: CPU LossFunction takes 2D [BATCH, OUT_DIM] — no reshape needed
            var loss = Self.LOSS_FUNCTION.forward[BATCH, Self.MODEL.OUT_DIM](
                output_t, target
            )
            Self.LOSS_FUNCTION.backward[BATCH, Self.MODEL.OUT_DIM](
                output_t, target, grad_out_t
            )

            state.zero_grads()
            Self.MODEL.backward[BATCH](
                grad_out_t, grad_in_t, params, cache_t, grads
            )

            state.step_num += 1
            Self.OPTIMIZER.step[Self.MODEL.PARAM_SIZE](
                params, grads, opt_state, state.step_num
            )

            final_loss = loss

            if print_every > 0 and epoch % print_every == 0:
                print("Epoch " + String(epoch) + " - Loss: " + String(loss))

            if checkpoint_every > 0 and len(checkpoint_path) > 0:
                if (epoch + 1) % checkpoint_every == 0:
                    try:
                        state.save_checkpoint(checkpoint_path)
                        print("Checkpoint saved at epoch " + String(epoch + 1))
                    except:
                        print(
                            "Warning: failed to save checkpoint at epoch "
                            + String(epoch + 1)
                        )

        return TrainResult(final_loss, epochs)

    # =========================================================================
    # CPU Evaluation (no gradient computation)
    # =========================================================================

    @staticmethod
    fn evaluate[
        BATCH: Int
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
    ) -> Float64:
        """CPU forward pass + loss, no gradient computation.

        Accepts a params LayoutTensor so it works with either
        state.params_view() (CPU NetworkState) or — after downloading
        via gpu.download_to(state, ctx) — state.params_view() from GPU.

        Args:
            params: Model parameters [PARAM_SIZE] (e.g. state.params_view()).
            input: Input tensor [BATCH, IN_DIM].
            target: Target tensor [BATCH, OUT_DIM].

        Returns:
            Scalar loss value.
        """
        comptime OUT_SIZE = BATCH * Self.MODEL.OUT_DIM

        var output_data = List[Scalar[dtype]](capacity=OUT_SIZE)
        for _ in range(OUT_SIZE):
            output_data.append(Scalar[dtype](0))

        var output_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output_data.unsafe_ptr())

        # params is already an lvalue from the caller — pass directly
        Self.MODEL.forward[BATCH](input, output_t, params)

        return Self.LOSS_FUNCTION.forward[BATCH, Self.MODEL.OUT_DIM](
            output_t, target
        )

    # =========================================================================
    # GPU Training
    # =========================================================================

    @staticmethod
    fn train_gpu[
        BATCH: Int
    ](
        mut state: GPUNetworkState[Self.MODEL, Self.OPTIMIZER],
        ctx: DeviceContext,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        epochs: Int = 100,
        print_every: Int = 0,
    ) raises -> TrainResult:
        """Train on GPU for the given number of epochs.

        The caller owns GPUNetworkState and is responsible for uploading params
        before calling and downloading results afterwards (e.g. via
        gpu.download_to(state, ctx)).  Input/target are CPU-side LayoutTensors
        uploaded to device once before the loop.

        Args:
            state: GPU network state — updated in-place.
            ctx: GPU device context.
            input: CPU input tensor [BATCH, IN_DIM] — caller manages memory.
            target: CPU target tensor [BATCH, OUT_DIM] — caller manages memory.
            epochs: Number of training epochs.
            print_every: Print loss every N epochs (0 = never).

        Returns:
            TrainResult with final_loss and epochs_trained.
        """
        comptime IN_SIZE = BATCH * Self.MODEL.IN_DIM
        comptime OUT_SIZE = BATCH * Self.MODEL.OUT_DIM
        comptime CACHE_SIZE_ = BATCH * Self.MODEL.CACHE_SIZE
        comptime WS_SIZE = BATCH * Self.MODEL.WORKSPACE_SIZE_PER_SAMPLE

        # Upload input and target via pinned host buffers (once before loop).
        # LayoutTensor has no unsafe_ptr() — copy via 2D element indexing.
        var input_host = ctx.enqueue_create_host_buffer[dtype](IN_SIZE)
        var target_host = ctx.enqueue_create_host_buffer[dtype](OUT_SIZE)
        for row in range(BATCH):
            for col in range(Self.MODEL.IN_DIM):
                input_host[row * Self.MODEL.IN_DIM + col] = rebind[
                    Scalar[dtype]
                ](input[row, col])
        for row in range(BATCH):
            for col in range(Self.MODEL.OUT_DIM):
                target_host[row * Self.MODEL.OUT_DIM + col] = rebind[
                    Scalar[dtype]
                ](target[row, col])
        var input_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var target_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        ctx.enqueue_copy(input_buf, input_host)
        ctx.enqueue_copy(target_buf, target_host)

        # Per-epoch device buffers (allocated once, reused each epoch)
        var output_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE_)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var grad_in_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var ws_buf = ctx.enqueue_create_buffer[dtype](
            WS_SIZE if WS_SIZE > 0 else 1
        )

        # LayoutTensor views over device buffers (created once)
        var input_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input_buf.unsafe_ptr())
        var target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](target_buf.unsafe_ptr())
        var output_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](grad_out_buf.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_in_buf.unsafe_ptr())
        var loss_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            loss_buf.unsafe_ptr()
        )
        var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

        var final_loss: Float64 = 0.0

        for epoch in range(epochs):
            state.zero_grads(ctx)

            # Fresh lvalue views each epoch
            var params = state.params_view()
            var grads = state.grads_view()

            Self.MODEL.forward_gpu[BATCH](
                ctx, output_t, input_t, params, cache_t, ws_buf
            )
            Self.LOSS_FUNCTION.backward_gpu[BATCH, Self.MODEL.OUT_DIM](
                ctx, grad_out_t, output_t, target_t
            )
            Self.MODEL.backward_gpu[BATCH](
                ctx, grad_in_t, grad_out_t, params, cache_t, grads, ws_buf
            )
            state.optimizer_step(ctx)

            if print_every > 0 and epoch % print_every == 0:
                Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                    ctx, loss_t, output_t, target_t
                )
                ctx.enqueue_copy(loss_host, loss_buf)
                ctx.synchronize()
                final_loss = Float64(loss_host[0])
                print(
                    "Epoch " + String(epoch) + " - Loss: " + String(final_loss)
                )

        # Compute final loss if the last epoch was not a print epoch
        if print_every == 0 or (epochs - 1) % print_every != 0:
            Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                ctx, loss_t, output_t, target_t
            )
            ctx.enqueue_copy(loss_host, loss_buf)
            ctx.synchronize()
            final_loss = Float64(loss_host[0])

        return TrainResult(final_loss, epochs)
