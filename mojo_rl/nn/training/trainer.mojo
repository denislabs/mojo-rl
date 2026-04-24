"""Trainer: all-static training loops for neural networks.

All methods are @staticmethod — no stored state.  The caller owns and passes
NetworkState (CPU) or GPUNetworkState (GPU) directly, so GPU-only pipelines
never allocate a CPU state.

Usage:
    from mojo_rl.nn import seq, Linear, ReLU, Adam, MSELoss, Kaiming
    from mojo_rl.nn.training import Trainer, NetworkState, GPUNetworkState

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
from ..constants import dtype as default_dtype, TPB
from .network_state import NetworkState
from .gpu_network_state import GPUNetworkState

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from std.sys import has_nvidia_gpu_accelerator


struct TrainResult(ImplicitlyCopyable, Movable):
    """Result of a training run."""

    var final_loss: Float64
    var epochs_trained: Int

    def __init__(out self, final_loss: Float64, epochs_trained: Int):
        self.final_loss = final_loss
        self.epochs_trained = epochs_trained


# =============================================================================
# GPU helper kernels for mini-batch shuffling
# =============================================================================
# All state (indices permutation + RNG seed) lives in LayoutTensor over device
# memory so the full train-shuffle-gather-step sequence is CUDA-graph capturable.


@always_inline
def _init_identity_indices_kernel[
    N: Int,
](indices: LayoutTensor[DType.int32, Layout.row_major(N), MutAnyOrigin]):
    """Fill indices[i] = i. Parallel over N threads."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < N:
        indices[i] = Int32(i)


@always_inline
def _fisher_yates_shuffle_kernel[
    N: Int,
](
    indices: LayoutTensor[DType.int32, Layout.row_major(N), MutAnyOrigin],
    seed_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Serial Fisher-Yates shuffle on a single GPU thread.

    60k serial iterations on one thread is fast enough at ~once-per-epoch
    cadence (a few ms). Parallel shuffles (sort-by-random-key) are faster
    but require a device sort, which this codebase does not have.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var s = seed_buf.ptr[0]
    var philox = PhiloxRandom(seed=s, offset=0)
    for i in range(N - 1, 0, -1):
        var r = philox.step_uniform()
        # Metal does not support Float64 — use Float32 throughout
        var j = Int(Float32(r[0]) * Float32(i + 1))
        if j > i:
            j = i
        var tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp


@always_inline
def _increment_seed_kernel(
    seed_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Bump the device-side RNG seed so each epoch has a different permutation.
    """
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        seed_buf.ptr[0] = seed_buf.ptr[0] + UInt64(1)


@always_inline
def _gather_rows_kernel[
    N_TOTAL: Int,
    BATCH: Int,
    DIM: Int,
    dtype: DType,
](
    batch_out: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    full: LayoutTensor[dtype, Layout.row_major(N_TOTAL, DIM), MutAnyOrigin],
    indices: LayoutTensor[DType.int32, Layout.row_major(N_TOTAL), MutAnyOrigin],
    offset: Int,
):
    """batch_out[b, d] = full[indices[offset + b], d]. Parallel over BATCH*DIM.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * DIM:
        return
    var b = i // DIM
    var d = i % DIM
    var src = Int(indices[offset + b])
    batch_out[b, d] = full[src, d]


struct Trainer[
    MODEL: Model,
    OPTIMIZER: Optimizer,
    LOSS_FUNCTION: LossFunction,
    dtype: DType = default_dtype,
]:
    """All-static training loop namespace.

    No stored state — the caller manages NetworkState (CPU) or GPUNetworkState
    (GPU) and passes it to each method.  This means GPU-only training never
    allocates a CPU NetworkState.

    Parameters:
        MODEL: Stateless model architecture (implements Model trait).
        OPTIMIZER: Stateless optimizer (implements Optimizer trait).
        LOSS_FUNCTION: Stateless loss function (implements LossFunction trait).
        dtype: Data type for all tensors and buffers (default: DType.float32).
    """

    # =========================================================================
    # State Initialization Helpers
    # =========================================================================

    @staticmethod
    def init_state[
        INITIALIZER: Initializer = Xavier[]
    ]() -> NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]:
        """Create and initialize a CPU NetworkState.

        Parameters:
            INITIALIZER: Weight initialization strategy (default: Xavier).

        Returns:
            Initialized NetworkState ready for CPU training or upload to GPU.
        """
        var state = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]()
        state.initialize[INITIALIZER]()
        return state^

    @staticmethod
    def init_state_gpu[
        INITIALIZER: Initializer = Xavier[]
    ](ctx: DeviceContext) raises -> GPUNetworkState[
        Self.MODEL, Self.OPTIMIZER, Self.dtype
    ]:
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
        var cpu = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]()
        cpu.initialize[INITIALIZER]()
        var gpu = GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](ctx)
        gpu.upload_from(cpu, ctx)
        return gpu^

    # =========================================================================
    # CPU Training
    # =========================================================================

    @staticmethod
    def train[
        BATCH: Int
    ](
        mut state: NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype],
        input: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
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
        var output_data = List[Scalar[Self.dtype]](capacity=OUT_SIZE)
        var grad_out_data = List[Scalar[Self.dtype]](capacity=OUT_SIZE)
        var grad_in_data = List[Scalar[Self.dtype]](capacity=IN_SIZE)
        var cache_data = List[Scalar[Self.dtype]](capacity=CACHE_SIZE_)
        for _ in range(OUT_SIZE):
            output_data.append(Scalar[Self.dtype](0))
            grad_out_data.append(Scalar[Self.dtype](0))
        for _ in range(IN_SIZE):
            grad_in_data.append(Scalar[Self.dtype](0))
        for _ in range(CACHE_SIZE_):
            cache_data.append(Scalar[Self.dtype](0))

        # 2D LayoutTensor views (zero-copy, created once from List.unsafe_ptr())
        var output_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](output_data.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](grad_out_data.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_in_data.unsafe_ptr())
        var cache_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_data.unsafe_ptr())

        # State views — lvalue vars required (params/state are mut in step())
        var params = state.params_view()
        var grads = state.grads_view()
        var opt_state = state.opt_state_view()
        var model_state = state.model_state_view()
        var opt_global = state.opt_global_state_view()

        var final_loss: Float64 = 0.0

        for epoch in range(epochs):
            Self.MODEL.forward[BATCH](input, output_t, params, model_state, cache_t)

            # Loss: CPU LossFunction takes 2D [BATCH, OUT_DIM] — no reshape needed
            var loss = Self.LOSS_FUNCTION.forward[BATCH, Self.MODEL.OUT_DIM](
                output_t, target
            )
            Self.LOSS_FUNCTION.backward[BATCH, Self.MODEL.OUT_DIM](
                output_t, target, grad_out_t
            )

            state.zero_grads()
            Self.MODEL.backward[BATCH](
                grad_out_t, grad_in_t, params, model_state, cache_t, grads
            )

            state.step_num += 1
            Self.OPTIMIZER.step[Self.MODEL.PARAM_SIZE](
                params, grads, opt_state, opt_global, state.step_num
            )

            final_loss = loss

            if print_every > 0 and epoch % print_every == 0:
                print("Epoch " + String(epoch) + " - Loss: " + String(loss))

            if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
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
    def evaluate[
        BATCH: Int
    ](
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        mut model_state: LayoutTensor[
            Self.dtype, Layout.row_major(Self.MODEL.STATE_SIZE), MutAnyOrigin
        ],
        input: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
    ) -> Float64:
        """CPU forward pass + loss, no gradient computation.

        Accepts params + model_state LayoutTensors so it works with either
        CPU NetworkState views or — after downloading via
        gpu.download_to(state, ctx) — the same CPU views.

        Args:
            params: Model parameters [PARAM_SIZE] (e.g. state.params_view()).
            model_state: Persistent non-trainable state [STATE_SIZE]
                (e.g. state.model_state_view()). Zero-length for stateless models.
            input: Input tensor [BATCH, IN_DIM].
            target: Target tensor [BATCH, OUT_DIM].

        Returns:
            Scalar loss value.
        """
        comptime OUT_SIZE = BATCH * Self.MODEL.OUT_DIM

        var output_data = List[Scalar[Self.dtype]](capacity=OUT_SIZE)
        for _ in range(OUT_SIZE):
            output_data.append(Scalar[Self.dtype](0))

        var output_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](output_data.unsafe_ptr())

        # params is already an lvalue from the caller — pass directly
        Self.MODEL.forward[BATCH](input, output_t, params, model_state)

        return Self.LOSS_FUNCTION.forward[BATCH, Self.MODEL.OUT_DIM](
            output_t, target
        )

    # =========================================================================
    # GPU Training
    # =========================================================================

    @staticmethod
    def train_gpu[
        BATCH: Int,
        USE_CUDA_GRAPH: Bool = False,
    ](
        mut state: GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype],
        ctx: DeviceContext,
        input: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        epochs: Int = 100,
        print_every: Int = 0,
    ) raises -> TrainResult:
        """Train on GPU for the given number of epochs.

        The caller owns GPUNetworkState and is responsible for uploading params
        before calling and downloading results afterwards (e.g. via
        gpu.download_to(state, ctx)).  Input/target are CPU-side LayoutTensors
        uploaded to device once before the loop.

        Parameters:
            BATCH: Number of samples per batch.
            USE_CUDA_GRAPH: When True, captures one epoch's kernel sequence
                into a CUDA graph and replays it for all subsequent epochs.
                Eliminates per-kernel launch overhead. Requires LD_PRELOAD
                with libcuda_intercept.so (set by pixi nvidia env).
                No-op on non-NVIDIA platforms.

        Args:
            state: GPU network state (params, grads, optimizer state) — updated.
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
        var input_host = ctx.enqueue_create_host_buffer[Self.dtype](IN_SIZE)
        var target_host = ctx.enqueue_create_host_buffer[Self.dtype](OUT_SIZE)
        for row in range(BATCH):
            for col in range(Self.MODEL.IN_DIM):
                input_host[row * Self.MODEL.IN_DIM + col] = rebind[
                    Scalar[Self.dtype]
                ](input[row, col])
        for row in range(BATCH):
            for col in range(Self.MODEL.OUT_DIM):
                target_host[row * Self.MODEL.OUT_DIM + col] = rebind[
                    Scalar[Self.dtype]
                ](target[row, col])
        var input_buf = ctx.enqueue_create_buffer[Self.dtype](IN_SIZE)
        var target_buf = ctx.enqueue_create_buffer[Self.dtype](OUT_SIZE)
        ctx.enqueue_copy(input_buf, input_host)
        ctx.enqueue_copy(target_buf, target_host)

        # Per-epoch device buffers (allocated once, reused each epoch)
        var output_buf = ctx.enqueue_create_buffer[Self.dtype](OUT_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[Self.dtype](CACHE_SIZE_)
        var grad_out_buf = ctx.enqueue_create_buffer[Self.dtype](OUT_SIZE)
        var grad_in_buf = ctx.enqueue_create_buffer[Self.dtype](IN_SIZE)
        var loss_buf = ctx.enqueue_create_buffer[Self.dtype](1)
        var ws_buf = ctx.enqueue_create_buffer[Self.dtype](
            WS_SIZE if WS_SIZE > 0 else 1
        )

        # LayoutTensor views over device buffers (created once)
        var input_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input_buf.unsafe_ptr())
        var target_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](target_buf.unsafe_ptr())
        var output_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](output_buf.unsafe_ptr())
        var cache_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_buf.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](grad_out_buf.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_in_buf.unsafe_ptr())
        var loss_t = LayoutTensor[
            Self.dtype, Layout.row_major(1), MutAnyOrigin
        ](loss_buf.unsafe_ptr())
        var loss_host = ctx.enqueue_create_host_buffer[Self.dtype](1)

        var final_loss: Float64 = 0.0

        # --- Helper: run one training epoch (pure GPU, no host ops) ---
        @always_inline
        def _run_one_epoch() raises unified {
            read ctx,
            mut state,
            read input_t,
            read target_t,
            mut output_t,
            mut cache_t,
            mut grad_out_t,
            mut grad_in_t,
            read ws_buf,
        }:
            state.zero_grads(ctx)
            var params = state.params_view()
            var grads = state.grads_view()
            var model_state = state.model_state_view()
            Self.MODEL.forward_gpu[BATCH](
                ctx, output_t, input_t, params, model_state, cache_t, ws_buf
            )
            Self.LOSS_FUNCTION.backward_gpu[BATCH, Self.MODEL.OUT_DIM](
                ctx, grad_out_t, output_t, target_t
            )
            Self.MODEL.backward_gpu[BATCH](
                ctx, grad_in_t, grad_out_t, params, model_state, cache_t, grads, ws_buf
            )
            state.optimizer_step(ctx)

        comptime if USE_CUDA_GRAPH and has_nvidia_gpu_accelerator():
            from mojo_rl.cuda import CUDAGraph

            # Warmup: run one epoch to ensure stream is discoverable
            _run_one_epoch()
            ctx.synchronize()

            # Capture one epoch into a CUDA graph
            var graph = CUDAGraph(ctx)
            graph.begin_capture()
            _run_one_epoch()
            graph.end_capture()

            # Replay for remaining epochs (first epoch already ran)
            for _ in range(epochs - 1):
                graph.replay()

        else:
            for epoch in range(epochs):
                _run_one_epoch()

                if print_every > 0 and epoch % print_every == 0:
                    Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                        ctx, loss_t, output_t, target_t
                    )
                    ctx.enqueue_copy(loss_host, loss_buf)
                    ctx.synchronize()
                    final_loss = Float64(loss_host[0])
                    print(
                        "Epoch "
                        + String(epoch)
                        + " - Loss: "
                        + String(final_loss)
                    )

        # Compute final loss (always runs outside capture)
        Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
            ctx, loss_t, output_t, target_t
        )
        ctx.enqueue_copy(loss_host, loss_buf)
        ctx.synchronize()
        final_loss = Float64(loss_host[0])

        return TrainResult(final_loss, epochs)

    # =========================================================================
    # GPU Mini-batch Training
    # =========================================================================

    @staticmethod
    def train_gpu_minibatch[
        BATCH: Int,
        N_TOTAL: Int,
        USE_CUDA_GRAPH: Bool = True,
    ](
        mut state: GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype],
        ctx: DeviceContext,
        input: LayoutTensor[
            Self.dtype,
            Layout.row_major(N_TOTAL, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        target: LayoutTensor[
            Self.dtype,
            Layout.row_major(N_TOTAL, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        epochs: Int = 1,
        print_every_batches: Int = 0,
        shuffle: Bool = False,
        rng_seed: UInt64 = 42,
    ) raises -> TrainResult:
        """Train on GPU with mini-batch SGD over a full dataset.

        Unlike train_gpu (which repeats one fixed batch for N epochs), this
        iterates the caller-provided dataset in BATCH-sized slices. Input and
        target must already live on the device — caller uploads once before
        the loop. Last partial batch (N_TOTAL % BATCH samples) is dropped.

        When shuffle=True a device-resident permutation of [0, N_TOTAL) is
        Fisher-Yates-shuffled per epoch using PhiloxRandom, and each batch is
        gathered from input/target through that permutation. Both the
        permutation and the RNG seed live in LayoutTensors so the whole
        per-epoch kernel chain is CUDA-graph capturable.

        Parameters:
            BATCH: Samples per gradient step.
            N_TOTAL: Total samples in the dataset (comptime).
            USE_CUDA_GRAPH: When True on NVIDIA, captures one epoch's kernel
                sequence into a CUDA graph and replays for subsequent epochs.
                Requires LD_PRELOAD with libcuda_intercept.so (pixi nvidia
                env). No-op on non-NVIDIA. Implies print_every_batches=0
                (enqueue_copy + sync can't happen during capture).

        Args:
            state: GPU network state — updated in place.
            ctx: GPU device context.
            input: Device tensor [N_TOTAL, IN_DIM].
            target: Device tensor [N_TOTAL, OUT_DIM].
            epochs: Number of passes through the dataset.
            print_every_batches: Print batch loss every N batches (0 = never).
                Ignored when USE_CUDA_GRAPH=True.
            shuffle: If True, re-shuffle sample order each epoch on device.
            rng_seed: Initial seed for the shuffle PRNG. Ignored if shuffle
                is False. Incremented by 1 each epoch via a device-side kernel.

        Returns:
            TrainResult with final-batch loss and total epochs completed.
        """
        comptime NUM_BATCHES = N_TOTAL // BATCH
        comptime CACHE_SIZE_ = BATCH * Self.MODEL.CACHE_SIZE
        comptime WS_SIZE = BATCH * Self.MODEL.WORKSPACE_SIZE_PER_SAMPLE

        # Per-batch device buffers (allocated once, reused across all batches)
        var output_buf = ctx.enqueue_create_buffer[Self.dtype](
            BATCH * Self.MODEL.OUT_DIM
        )
        var cache_buf = ctx.enqueue_create_buffer[Self.dtype](CACHE_SIZE_)
        var grad_out_buf = ctx.enqueue_create_buffer[Self.dtype](
            BATCH * Self.MODEL.OUT_DIM
        )
        var grad_in_buf = ctx.enqueue_create_buffer[Self.dtype](
            BATCH * Self.MODEL.IN_DIM
        )
        var loss_buf = ctx.enqueue_create_buffer[Self.dtype](1)
        var ws_buf = ctx.enqueue_create_buffer[Self.dtype](
            WS_SIZE if WS_SIZE > 0 else 1
        )
        var loss_host = ctx.enqueue_create_host_buffer[Self.dtype](1)

        var output_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](output_buf.unsafe_ptr())
        var cache_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_buf.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](grad_out_buf.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ](grad_in_buf.unsafe_ptr())
        var loss_t = LayoutTensor[
            Self.dtype, Layout.row_major(1), MutAnyOrigin
        ](loss_buf.unsafe_ptr())

        var final_loss: Float64 = 0.0
        var total_batches_trained: Int = 0

        # ── Shuffle-only device buffers (allocated only when shuffle=True) ──
        var indices_buf = ctx.enqueue_create_buffer[DType.int32](
            N_TOTAL if shuffle else 1
        )
        var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
        var batch_input_buf = ctx.enqueue_create_buffer[Self.dtype](
            (BATCH * Self.MODEL.IN_DIM) if shuffle else 1
        )
        var batch_target_buf = ctx.enqueue_create_buffer[Self.dtype](
            (BATCH * Self.MODEL.OUT_DIM) if shuffle else 1
        )

        var indices_t = LayoutTensor[
            DType.int32, Layout.row_major(N_TOTAL), MutAnyOrigin
        ](indices_buf.unsafe_ptr())
        var seed_t = LayoutTensor[
            DType.uint64, Layout.row_major(1), MutAnyOrigin
        ](seed_buf.unsafe_ptr())
        var shuf_input_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ](batch_input_buf.unsafe_ptr())
        var shuf_target_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ](batch_target_buf.unsafe_ptr())

        if shuffle:
            # Seed the device RNG buffer from the host (one-time, pre-loop)
            var seed_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
            seed_host.unsafe_ptr()[0] = rng_seed
            ctx.enqueue_copy(seed_buf, seed_host)

            # Fill indices with [0, 1, ..., N_TOTAL)
            comptime init_blocks = (N_TOTAL + TPB - 1) // TPB
            ctx.enqueue_function[
                _init_identity_indices_kernel[N_TOTAL],
                _init_identity_indices_kernel[N_TOTAL],
            ](indices_t, grid_dim=(init_blocks,), block_dim=(TPB,))

        comptime gather_in_blocks = (BATCH * Self.MODEL.IN_DIM + TPB - 1) // TPB
        comptime gather_tg_blocks = (
            BATCH * Self.MODEL.OUT_DIM + TPB - 1
        ) // TPB

        # --- Helper: run one training epoch (pure GPU, no host syncs) ---
        # Captures everything it reads/writes so it can be a graph body.
        @always_inline
        def _run_one_epoch() raises unified {
            read ctx,
            mut state,
            read input,
            read target,
            mut output_t,
            mut cache_t,
            mut grad_out_t,
            mut grad_in_t,
            mut shuf_input_t,
            mut shuf_target_t,
            mut indices_t,
            mut seed_t,
            read ws_buf,
            read shuffle,
        }:
            if shuffle:
                ctx.enqueue_function[
                    _fisher_yates_shuffle_kernel[N_TOTAL],
                    _fisher_yates_shuffle_kernel[N_TOTAL],
                ](indices_t, seed_t, grid_dim=(1,), block_dim=(1,))
                ctx.enqueue_function[
                    _increment_seed_kernel, _increment_seed_kernel
                ](seed_t, grid_dim=(1,), block_dim=(1,))

            for batch_idx in range(NUM_BATCHES):
                var batch_input: LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.MODEL.IN_DIM),
                    MutAnyOrigin,
                ]
                var batch_target: LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
                    MutAnyOrigin,
                ]

                if shuffle:
                    ctx.enqueue_function[
                        _gather_rows_kernel[
                            N_TOTAL, BATCH, Self.MODEL.IN_DIM, Self.dtype
                        ],
                        _gather_rows_kernel[
                            N_TOTAL, BATCH, Self.MODEL.IN_DIM, Self.dtype
                        ],
                    ](
                        shuf_input_t,
                        input,
                        indices_t,
                        batch_idx * BATCH,
                        grid_dim=(gather_in_blocks,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        _gather_rows_kernel[
                            N_TOTAL, BATCH, Self.MODEL.OUT_DIM, Self.dtype
                        ],
                        _gather_rows_kernel[
                            N_TOTAL, BATCH, Self.MODEL.OUT_DIM, Self.dtype
                        ],
                    ](
                        shuf_target_t,
                        target,
                        indices_t,
                        batch_idx * BATCH,
                        grid_dim=(gather_tg_blocks,),
                        block_dim=(TPB,),
                    )
                    batch_input = shuf_input_t
                    batch_target = shuf_target_t
                else:
                    batch_input = LayoutTensor[
                        Self.dtype,
                        Layout.row_major(BATCH, Self.MODEL.IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr + batch_idx * BATCH * Self.MODEL.IN_DIM)
                    batch_target = LayoutTensor[
                        Self.dtype,
                        Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
                        MutAnyOrigin,
                    ](target.ptr + batch_idx * BATCH * Self.MODEL.OUT_DIM)

                state.zero_grads(ctx)
                var params = state.params_view()
                var grads = state.grads_view()
                var model_state = state.model_state_view()
                Self.MODEL.forward_gpu[BATCH](
                    ctx, output_t, batch_input, params, model_state, cache_t, ws_buf
                )
                Self.LOSS_FUNCTION.backward_gpu[BATCH, Self.MODEL.OUT_DIM](
                    ctx, grad_out_t, output_t, batch_target
                )
                Self.MODEL.backward_gpu[BATCH](
                    ctx,
                    grad_in_t,
                    grad_out_t,
                    params,
                    model_state,
                    cache_t,
                    grads,
                    ws_buf,
                )
                state.optimizer_step(ctx)

        # --- Main loop: either graph capture + replay, or plain re-run ---
        comptime if USE_CUDA_GRAPH and has_nvidia_gpu_accelerator():
            from mojo_rl.cuda import CUDAGraph

            # Warmup: run one epoch to ensure stream is discoverable
            _run_one_epoch()
            ctx.synchronize()

            # Capture one epoch into a CUDA graph
            var graph = CUDAGraph(ctx)
            graph.begin_capture()
            _run_one_epoch()
            graph.end_capture()

            # Replay for remaining epochs (first epoch already ran)
            for _ in range(epochs - 1):
                graph.replay()

        else:
            for epoch in range(epochs):
                _run_one_epoch()

                # Post-epoch diagnostic (optional, never inside a graph)
                if print_every_batches > 0:
                    # Compute loss on the last batch from this epoch.
                    # shuf_* hold the last gathered batch in shuffle mode;
                    # otherwise reconstruct the last contiguous slice.
                    var last_in: LayoutTensor[
                        Self.dtype,
                        Layout.row_major(BATCH, Self.MODEL.IN_DIM),
                        MutAnyOrigin,
                    ]
                    var last_tg: LayoutTensor[
                        Self.dtype,
                        Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
                        MutAnyOrigin,
                    ]
                    if shuffle:
                        last_in = shuf_input_t
                        last_tg = shuf_target_t
                    else:
                        comptime last_off_in = (
                            (NUM_BATCHES - 1) * BATCH * Self.MODEL.IN_DIM
                        )
                        comptime last_off_tg = (
                            (NUM_BATCHES - 1) * BATCH * Self.MODEL.OUT_DIM
                        )
                        last_in = LayoutTensor[
                            Self.dtype,
                            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
                            MutAnyOrigin,
                        ](input.ptr + last_off_in)
                        last_tg = LayoutTensor[
                            Self.dtype,
                            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
                            MutAnyOrigin,
                        ](target.ptr + last_off_tg)
                    var params_peek = state.params_view()
                    var model_state_peek = state.model_state_view()
                    Self.MODEL.forward_gpu[BATCH](
                        ctx, output_t, last_in, params_peek, model_state_peek, cache_t, ws_buf
                    )
                    Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                        ctx, loss_t, output_t, last_tg
                    )
                    ctx.enqueue_copy(loss_host, loss_buf)
                    ctx.synchronize()
                    print(
                        "  epoch "
                        + String(epoch + 1)
                        + "/"
                        + String(epochs)
                        + "  last-batch loss="
                        + String(Float64(loss_host[0]))
                    )

        # Final loss on the last batch actually trained on.
        # Shuffle mode: shuf_input_t/shuf_target_t still hold the last
        # gathered batch. Contiguous mode: reconstruct the last-batch slice.
        var last_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ]
        var last_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ]
        if shuffle:
            last_input = shuf_input_t
            last_target = shuf_target_t
        else:
            var last_batch_idx = NUM_BATCHES - 1
            last_input = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.MODEL.IN_DIM),
                MutAnyOrigin,
            ](input.ptr + last_batch_idx * BATCH * Self.MODEL.IN_DIM)
            last_target = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
                MutAnyOrigin,
            ](target.ptr + last_batch_idx * BATCH * Self.MODEL.OUT_DIM)
        var params_final = state.params_view()
        var model_state_final = state.model_state_view()
        Self.MODEL.forward_gpu[BATCH](
            ctx, output_t, last_input, params_final, model_state_final, cache_t, ws_buf
        )
        Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
            ctx, loss_t, output_t, last_target
        )
        ctx.enqueue_copy(loss_host, loss_buf)
        ctx.synchronize()
        final_loss = Float64(loss_host[0])

        return TrainResult(final_loss, epochs)
