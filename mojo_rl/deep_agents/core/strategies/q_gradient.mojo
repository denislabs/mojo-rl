"""Q-value gradient computation strategies for DQN family agents.

Controls how the MSE loss gradient is computed and scattered into Q-space.

Implementations:
  - ManualQGradient: Hand-written sparse MSE gradient (original inline code)
  - AutodiffQGradient: Uses GatherOp backward to scatter dMSE/dQ(s,a) into Q-space

Both produce identical [BATCH, ACTIONS] gradients in Q-space.  The QOutput
strategy (DirectQ / DuelingQ) transforms them into raw-output space afterward.

The autodiff path is more composable: swapping MSE for Huber or adding
auxiliary losses only requires changing the scalar gradient fed into GatherOp.vjp.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.autodiff.primitives.gather import GatherOp
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Gather,
    Slice,
    MSELoss,
    HuberLoss,
)
from mojo_rl.nn.autodiff.combinators import SplitApply


# =============================================================================
# QGradient trait
# =============================================================================


trait QGradient:
    """Trait for Q-value gradient computation strategies.

    Given:
      - q_values  [BATCH, ACTIONS]  predicted Q-values for all actions
      - targets   [BATCH]           TD target scalars
      - actions   [BATCH]           action indices (stored as float)

    Produce:
      - grad_q    [BATCH, ACTIONS]  dLoss/dQ  (sparse: only taken action is nonzero)

    Implementations may need workspace for caching (AutodiffQGradient uses
    GatherOp cache).  The workspace size is declared via gpu_ws_size.
    The caller pre-allocates a DeviceBuffer of that size and passes it in.
    """

    @staticmethod
    def gpu_ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        """GPU workspace floats needed per call (0 for ManualQGradient)."""
        ...

    @staticmethod
    def compute_grad_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        targets: InlineArray[Scalar[dtype], BATCH],
        actions: InlineArray[Scalar[dtype], BATCH],
        mut grad_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> Float64:
        """Compute dMSE/dQ on CPU. Returns mean squared loss."""
        ...

    @staticmethod
    def compute_grad_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        q_values: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        grad_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute dMSE/dQ on GPU using pre-allocated workspace."""
        ...


# =============================================================================
# ManualQGradient -- hand-written sparse MSE (original inline code)
# =============================================================================


struct ManualQGradient(QGradient):
    """Sparse MSE gradient: grad_q[b, a] = 2*(Q[b,a] - target[b])/BATCH if a==action else 0.

    This is the original hand-written gradient that was inline in dqn_agent.mojo.
    Zero workspace required.
    """

    @staticmethod
    def gpu_ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        return 0

    @staticmethod
    def compute_grad_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        targets: InlineArray[Scalar[dtype], BATCH],
        actions: InlineArray[Scalar[dtype], BATCH],
        mut grad_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> Float64:
        """Hand-written sparse MSE gradient on CPU."""
        var total_loss: Float64 = 0.0
        for b in range(BATCH):
            var action = Int(actions[b])
            var q_pred = q_values[b * ACTIONS + action]
            var td_err = q_pred - targets[b]
            total_loss += Float64(td_err * td_err)
            for a in range(ACTIONS):
                if a == action:
                    grad_q[b * ACTIONS + a] = (
                        Scalar[dtype](2.0) * td_err / Scalar[dtype](BATCH)
                    )
                else:
                    grad_q[b * ACTIONS + a] = Scalar[dtype](0.0)
        return total_loss / Float64(BATCH)

    @staticmethod
    def compute_grad_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        q_values: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        grad_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises -> None:
        """Hand-written sparse MSE gradient on GPU. One thread per sample."""

        @always_inline
        def kernel(
            grd: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var action = Int(act[b])
            var td_error = qv[b, action] - tgt[b]
            for a in range(ACTIONS):
                if a == action:
                    grd[b, a] = (
                        Scalar[dtype](2.0) * td_error / Scalar[dtype](BATCH)
                    )
                else:
                    grd[b, a] = Scalar[dtype](0.0)

        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kernel, kernel](
            grad_q,
            q_values,
            targets,
            actions,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# AutodiffQGradient -- true composed loss graph (Gather → MSE)
# =============================================================================


struct AutodiffQGradient[LossOp: Model = MSELoss](QGradient):
    """Autodiff Q-gradient using composed loss graph.

    The DQN loss is expressed as a composed Model:
        Input: [Q_values(A) || action_idx(1) || target(1)] = A + 2

        LossGraph = Sequential[
            SplitApply[Gather[A], Slice[1,0,1], A+1],  → [Q(s,a), target]
            LossOp,                                      → loss(Q(s,a), target)
        ]

    LossOp defaults to MSELoss. Use HuberLoss[delta] for robust DQN:
        AutodiffQGradient[HuberLoss[1.0]]

    Forward packs Q-values, action index, and target into one tensor.
    Backward produces sparse gradient in Q-space via automatic VJP.
    """

    @staticmethod
    def gpu_ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        """Total GPU workspace floats for compute_grad_gpu (pre-allocated once).

        Layout within the workspace buffer:
          [0]                          loss_in    BATCH * LOSS_IN
          [BATCH*LOSS_IN]              loss_out   BATCH
          [...]                        cache      max(1, BATCH * LOSS_CS)
          [...]                        params     max(1, PARAM_SIZE)
          [...]                        grads      max(1, PARAM_SIZE)
          [...]                        grad_seed  BATCH
          [...]                        grad_in    BATCH * LOSS_IN
          [...]                        fwd_ws     max(1, BATCH * WS_PER_SAMPLE)
        """
        comptime LOSS_IN = ACTIONS + 2
        comptime LossGraph = Sequential[
            SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
            Self.LossOp,
        ]
        comptime LOSS_CS = LossGraph.CACHE_SIZE
        comptime PS = max(1, LossGraph.PARAM_SIZE)
        comptime FWD_WS = max(1, BATCH * LossGraph.WORKSPACE_SIZE_PER_SAMPLE)
        return (
            BATCH * LOSS_IN  # loss_in
            + BATCH  # loss_out
            + max(1, BATCH * LOSS_CS)  # cache
            + PS  # params
            + PS  # grads
            + BATCH  # grad_seed
            + BATCH * LOSS_IN  # grad_in
            + FWD_WS  # fwd/bwd workspace
        )

    @staticmethod
    def compute_grad_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        targets: InlineArray[Scalar[dtype], BATCH],
        actions: InlineArray[Scalar[dtype], BATCH],
        mut grad_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> Float64:
        """Autodiff Q-gradient on CPU via composed loss graph."""
        comptime LOSS_IN = ACTIONS + 2
        comptime LossGraph = Sequential[
            SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
            Self.LossOp,
        ]
        comptime LOSS_CS = LossGraph.CACHE_SIZE

        # Pack: [Q_values(A) || action_idx(1) || target(1)]
        var loss_in = InlineArray[Scalar[dtype], BATCH * LOSS_IN](
            uninitialized=True
        )
        for b in range(BATCH):
            for a in range(ACTIONS):
                loss_in[b * LOSS_IN + a] = q_values[b * ACTIONS + a]
            loss_in[b * LOSS_IN + ACTIONS] = actions[b]
            loss_in[b * LOSS_IN + ACTIONS + 1] = targets[b]

        var loss_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
        ](loss_in.unsafe_ptr())

        # Forward
        var loss_out = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        var loss_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LossGraph.OUT_DIM), MutAnyOrigin
        ](loss_out.unsafe_ptr())
        var cache = InlineArray[Scalar[dtype], BATCH * LOSS_CS](
            uninitialized=True
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_CS), MutAnyOrigin
        ](cache.unsafe_ptr())
        var params = InlineArray[Scalar[dtype], max(1, LossGraph.PARAM_SIZE)](
            fill=Scalar[dtype](0.0)
        )
        var params_t = LayoutTensor[
            dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())

        LossGraph.forward[BATCH](loss_in_t, loss_out_t, params_t, cache_t)

        # Mean loss
        var total_loss: Float64 = 0.0
        for b in range(BATCH):
            total_loss += Float64(loss_out[b])
        var loss = total_loss / Float64(BATCH)

        # Backward with seed 1/BATCH
        var grad_seed = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        var inv_batch = Scalar[dtype](1.0 / Float64(BATCH))
        for b in range(BATCH):
            grad_seed[b] = inv_batch
        var grad_seed_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LossGraph.OUT_DIM), MutAnyOrigin
        ](grad_seed.unsafe_ptr())

        var grad_in = InlineArray[Scalar[dtype], BATCH * LOSS_IN](
            uninitialized=True
        )
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
        ](grad_in.unsafe_ptr())
        var grads = InlineArray[Scalar[dtype], max(1, LossGraph.PARAM_SIZE)](
            fill=Scalar[dtype](0.0)
        )
        var grads_t = LayoutTensor[
            dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())

        LossGraph.backward[BATCH](
            grad_seed_t, grad_in_t, params_t, cache_t, grads_t
        )

        # Extract first ACTIONS columns as grad_q
        for b in range(BATCH):
            for a in range(ACTIONS):
                grad_q[b * ACTIONS + a] = grad_in[b * LOSS_IN + a]

        return loss

    @staticmethod
    def compute_grad_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        q_values: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        grad_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises -> None:
        """Autodiff Q-gradient on GPU via composed loss graph.

        Uses pre-allocated workspace buffer (see gpu_ws_size for layout).
        """
        comptime LOSS_IN = ACTIONS + 2
        comptime LossGraph = Sequential[
            SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
            Self.LossOp,
        ]
        comptime LOSS_CS = LossGraph.CACHE_SIZE
        comptime PS = max(1, LossGraph.PARAM_SIZE)
        comptime FWD_WS = max(1, BATCH * LossGraph.WORKSPACE_SIZE_PER_SAMPLE)
        comptime CACHE_SZ = max(1, BATCH * LOSS_CS)
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # Slice pre-allocated workspace into sub-buffers
        var ptr = workspace.unsafe_ptr()
        var off = 0

        # loss_in: [BATCH * LOSS_IN]
        var loss_in_ptr = ptr + off
        off += BATCH * LOSS_IN

        # loss_out: [BATCH]
        var loss_out_ptr = ptr + off
        off += BATCH

        # cache: [max(1, BATCH * LOSS_CS)]
        var cache_ptr = ptr + off
        off += CACHE_SZ

        # params: [PS]
        var params_ptr = ptr + off
        off += PS

        # grads: [PS]
        var grads_ptr = ptr + off
        off += PS

        # grad_seed: [BATCH]
        var grad_seed_ptr = ptr + off
        off += BATCH

        # grad_in: [BATCH * LOSS_IN]
        var grad_in_ptr = ptr + off
        off += BATCH * LOSS_IN

        # fwd_ws: [FWD_WS] — reuse workspace tail as forward/backward ws
        var ws_buf = DeviceBuffer[dtype](ctx, ptr + off, FWD_WS, owning=False)

        # Create LayoutTensor views
        var loss_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
        ](loss_in_ptr)

        # Pack: [Q_values(A) || action_idx(1) || target(1)]
        @always_inline
        def pack_k(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            for a in range(ACTIONS):
                dst[b, a] = qv[b, a]
            dst[b, ACTIONS] = act[b]
            dst[b, ACTIONS + 1] = tgt[b]

        ctx.enqueue_function[pack_k, pack_k](
            loss_in_t,
            q_values,
            actions,
            targets,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # Forward
        var loss_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LossGraph.OUT_DIM), MutAnyOrigin
        ](loss_out_ptr)
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_CS), MutAnyOrigin
        ](cache_ptr)
        var params_t = LayoutTensor[
            dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin
        ](params_ptr)

        LossGraph.forward_gpu[BATCH](
            ctx, loss_out_t, loss_in_t, params_t, cache_t, ws_buf
        )

        # Fill grad_seed with 1/BATCH on GPU (no host alloc needed)
        var grad_seed_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LossGraph.OUT_DIM), MutAnyOrigin
        ](grad_seed_ptr)

        @always_inline
        def fill_seed_k(
            seed: LayoutTensor[
                dtype, Layout.row_major(BATCH, LossGraph.OUT_DIM), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            seed[b, 0] = Scalar[dtype](1.0 / Float64(BATCH))

        ctx.enqueue_function[fill_seed_k, fill_seed_k](
            grad_seed_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
        ](grad_in_ptr)
        var grads_t = LayoutTensor[
            dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin
        ](grads_ptr)

        # Backward
        LossGraph.backward_gpu[BATCH](
            ctx, grad_in_t, grad_seed_t, params_t, cache_t, grads_t, ws_buf
        )

        # Extract first ACTIONS columns to grad_q
        @always_inline
        def extract_k(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, LOSS_IN), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            for a in range(ACTIONS):
                dst[b, a] = src[b, a]

        ctx.enqueue_function[extract_k, extract_k](
            grad_q,
            grad_in_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
