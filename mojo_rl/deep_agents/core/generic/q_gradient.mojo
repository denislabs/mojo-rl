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
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.autodiff.primitives.gather import GatherOp


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
    GatherOp cache).  The workspace size is declared via ws_size.
    """

    @staticmethod
    fn ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        """Workspace floats needed per call (0 for ManualQGradient)."""
        ...

    @staticmethod
    fn compute_grad_cpu[
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
    fn compute_grad_gpu[
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
    ) raises -> None:
        """Compute dMSE/dQ on GPU."""
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
    fn ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        return 0

    @staticmethod
    fn compute_grad_cpu[
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
                        Scalar[dtype](2.0)
                        * td_err
                        / Scalar[dtype](BATCH)
                    )
                else:
                    grad_q[b * ACTIONS + a] = Scalar[dtype](0.0)
        return total_loss / Float64(BATCH)

    @staticmethod
    fn compute_grad_gpu[
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
    ) raises -> None:
        """Hand-written sparse MSE gradient on GPU. One thread per sample."""

        @always_inline
        fn kernel(
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
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](BATCH)
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
# AutodiffQGradient -- uses GatherOp backward for sparse scatter
# =============================================================================


struct AutodiffQGradient(QGradient):
    """Autodiff-based Q-gradient using GatherOp.

    Conceptually performs:
        1. GatherOp.eval([Q_values || action_idx]) -> Q(s,a)  [forward, cached]
        2. dMSE/dQ(s,a) = 2 * (Q(s,a) - target) / BATCH     [scalar grad]
        3. GatherOp.vjp(dMSE/dQ(s,a)) -> sparse grad in Q-space [backward]

    Steps 1-3 produce the same result as ManualQGradient, but the backward
    pass is delegated to GatherOp.vjp which can be reused / composed.

    Workspace: BATCH * GatherOp[ACTIONS].CACHE_SIZE (= BATCH * 1) for the
    GatherOp cache, plus BATCH * GatherOp[ACTIONS].IN_DIM for the packed input,
    plus BATCH * 1 for the scalar grad_output.
    """

    @staticmethod
    fn ws_size[BATCH: Int, ACTIONS: Int]() -> Int:
        """Workspace = gather_input + gather_cache + grad_output."""
        # gather_input: [BATCH, ACTIONS+1], cache: [BATCH, 1], grad_out: [BATCH, 1]
        return BATCH * (ACTIONS + 1) + BATCH * 1 + BATCH * 1

    @staticmethod
    fn compute_grad_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        targets: InlineArray[Scalar[dtype], BATCH],
        actions: InlineArray[Scalar[dtype], BATCH],
        mut grad_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> Float64:
        """Autodiff Q-gradient on CPU using GatherOp.eval + GatherOp.vjp."""
        comptime GATHER_IN = ACTIONS + 1

        # -- Step 1: Pack [Q_values || action_idx] and run GatherOp.eval --
        var gather_input = InlineArray[Scalar[dtype], BATCH * GATHER_IN](
            uninitialized=True
        )
        for b in range(BATCH):
            for a in range(ACTIONS):
                gather_input[b * GATHER_IN + a] = q_values[b * ACTIONS + a]
            gather_input[b * GATHER_IN + ACTIONS] = actions[b]

        # Wrap as LayoutTensors for GatherOp
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
        ](gather_input.unsafe_ptr())

        var gathered = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gathered.unsafe_ptr())

        var cache = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](cache.unsafe_ptr())

        # Dummy params (GatherOp has PARAM_SIZE=0, but trait needs the tensor)
        var dummy_p = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0.0))
        var dp_t = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
            dummy_p.unsafe_ptr()
        )

        GatherOp[ACTIONS].eval[BATCH](gi_t, go_t, dp_t, cache_t)

        # -- Step 2: Compute scalar MSE gradient --
        var total_loss: Float64 = 0.0
        var grad_out = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        for b in range(BATCH):
            var q_a = gathered[b]
            var td_err = q_a - targets[b]
            total_loss += Float64(td_err * td_err)
            grad_out[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](BATCH)
            )
        var loss = total_loss / Float64(BATCH)

        # -- Step 3: GatherOp.vjp to scatter into Q-space --
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_out.unsafe_ptr())

        # grad_input is [BATCH, GATHER_IN] -- we reuse gather_input storage
        var grad_input = InlineArray[Scalar[dtype], BATCH * GATHER_IN](
            uninitialized=True
        )
        var grad_input_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
        ](grad_input.unsafe_ptr())

        var dummy_gp = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0.0))
        var dgp_t = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
            dummy_gp.unsafe_ptr()
        )

        GatherOp[ACTIONS].vjp[BATCH](
            grad_out_t, grad_input_t, dp_t, cache_t, dgp_t
        )

        # Copy first ACTIONS columns (skip index column) into grad_q
        for b in range(BATCH):
            for a in range(ACTIONS):
                grad_q[b * ACTIONS + a] = grad_input[b * GATHER_IN + a]

        return loss

    @staticmethod
    fn compute_grad_gpu[
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
    ) raises -> None:
        """Autodiff Q-gradient on GPU using GatherOp kernels.

        Three-phase approach:
          Phase A: Pack [Q || action] -> gather_input, run GatherOp.eval_kernel
          Phase B: Compute scalar MSE grad = 2*(Q(s,a) - target)/BATCH
          Phase C: GatherOp.backward_kernel -> sparse grad in Q-space
        """
        comptime GATHER_IN = ACTIONS + 1

        # Allocate workspace buffers
        var gather_input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * GATHER_IN
        )
        var gathered_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * GATHER_IN
        )

        # LayoutTensor views
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
        ](gather_input_buf.unsafe_ptr())
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
            gathered_buf.unsafe_ptr()
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_out_buf.unsafe_ptr())
        var grad_input_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())

        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ---- Phase A: Pack gather input and eval ----
        @always_inline
        fn pack_and_eval_kernel(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            gathered_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            ca: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            # Pack Q-values and action index
            for a in range(ACTIONS):
                gi[b, a] = qv[b, a]
            gi[b, ACTIONS] = act[b]
            # GatherOp eval inline
            var idx = Int(Float64(rebind[Scalar[dtype]](act[b])))
            gathered_out[b, 0] = qv[b, idx]
            ca[b, 0] = act[b]

        ctx.enqueue_function[pack_and_eval_kernel, pack_and_eval_kernel](
            gi_t,
            q_values,
            actions,
            go_t,
            cache_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase B: Compute scalar MSE gradient ----
        @always_inline
        fn mse_grad_kernel(
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            gathered: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var td_err = gathered[b, 0] - tgt[b]
            go[b, 0] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](BATCH)
            )

        ctx.enqueue_function[mse_grad_kernel, mse_grad_kernel](
            grad_out_t,
            go_t,
            targets,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase C: GatherOp backward -> sparse grad in Q-space ----
        @always_inline
        fn scatter_kernel(
            gi_grad: LayoutTensor[
                dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
            ],
            go_grad: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            ca: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
        ):
            # GatherOp backward logic: scatter grad to selected index
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var idx = Int(Float64(rebind[Scalar[dtype]](ca[b, 0])))
            var g = rebind[Scalar[dtype]](go_grad[b, 0])
            var zero = Scalar[dtype](0.0)
            for i in range(GATHER_IN):
                gi_grad[b, i] = zero
            gi_grad[b, idx] = g

        ctx.enqueue_function[scatter_kernel, scatter_kernel](
            grad_input_t,
            grad_out_t,
            cache_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Copy first ACTIONS columns to grad_q ----
        @always_inline
        fn copy_grad_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, GATHER_IN), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            for a in range(ACTIONS):
                dst[b, a] = src[b, a]

        ctx.enqueue_function[copy_grad_kernel, copy_grad_kernel](
            grad_q,
            grad_input_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
