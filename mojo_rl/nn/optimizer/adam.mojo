# =============================================================================
# Adam Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.math import sqrt, exp, log
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct Adam[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
](Optimizer):
    """Adam optimizer with adaptive learning rates.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    GLOBAL_STATE_SIZE = 1: one Float32 slot bit-patterning a UInt32 step counter.
    On GPU, a preamble kernel bumps the device counter inside `step_gpu` so
    Adam's bias correction stays correct under CUDA-graph replay (Phase 4 of
    docs/STATE_SIZE_DESIGN.md). The CPU `step()` path keeps using the host
    `step_num` parameter directly.

    Hyperparameters are compile-time struct parameters.
    """

    comptime STATE_PER_PARAM: Int = 2
    comptime GLOBAL_STATE_SIZE: Int = 1

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def step[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        mut opt_global_state: LayoutTensor[
            dtype, Layout.row_major(Self.GLOBAL_STATE_SIZE), MutAnyOrigin
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """Adam update step.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state layout `(PARAM_SIZE, 2)`: m at col 0, v at col 1.
            opt_global_state: Unused on the CPU path (the device counter is
                only consulted by `step_gpu`). Required by the trait signature.
            step_num: Current step (1-based), used for bias correction.
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)

        for i in range(PARAM_SIZE):
            var g = grads[i]
            var m = state[i, 0]
            var v = state[i, 1]

            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            state[i, 0] = m_new
            state[i, 1] = v_new

            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)

    # =========================================================================
    # GPU kernel implementation
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel_impl[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        state: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
        ],
        counter: LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ],
        lr: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        log_beta1: Scalar[dtype],
        log_beta2: Scalar[dtype],
    ):
        """Adam optimizer kernel.

        state layout: (PARAM_SIZE, 2) where state[i, 0] = m, state[i, 1] = v.

        Bias corrections are computed inside the kernel from the post-bump
        device counter so they stay correct under CUDA-graph replay.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var step_f = rebind[Scalar[DType.uint32]](counter[0]).cast[dtype]()
        var one = Scalar[dtype](1.0)
        var bc1 = one - exp(log_beta1 * step_f)
        var bc2 = one - exp(log_beta2 * step_f)

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        state[idx, 0] = m_new
        state[idx, 1] = v_new

        var m_hat = m_new / bc1
        var v_hat = v_new / bc2

        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * m_hat / (
            sqrt(v_hat) + eps
        )

    # =========================================================================
    # GPU launcher
    # =========================================================================

    @staticmethod
    def step_gpu[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        mut opt_global_state: LayoutTensor[
            dtype, Layout.row_major(Self.GLOBAL_STATE_SIZE), MutAnyOrigin
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ) raises:
        """Launch Adam optimization step on GPU.

        Bumps the device-side step counter (preamble kernel) and then runs the
        Adam update kernel, which reads the post-bump counter to compute the
        bias correction. This keeps bias correction correct under CUDA-graph
        replay (the host `step_num` argument is preserved as a bridge but is
        unused on this path).

        Args:
            ctx: GPU device context.
            params: Parameters [PARAM_SIZE] (modified in place).
            grads: Gradients [PARAM_SIZE].
            state: State [PARAM_SIZE, 2] (m and v moments).
            opt_global_state: One Float32 slot bit-patterning a UInt32 step
                counter. Bumped by a 1-thread preamble kernel before the main
                update.
            step_num: Unused on the GPU path (kept for trait-signature compat
                with optimizers that still rely on a host counter).
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)
        var log_beta1 = Scalar[dtype](log(Self.BETA1))
        var log_beta2 = Scalar[dtype](log(Self.BETA2))

        # Bit-cast the Float32 opt_global_state slot into a UInt32 step counter
        # view. The counter persists across calls (and CUDA-graph replays) and
        # is bumped by the preamble kernel below so each launch sees a fresh
        # step value.
        var counter_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](opt_global_state.ptr.bitcast[Scalar[DType.uint32]]())

        @parameter
        @always_inline
        def bump_kernel(
            c: LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ],
        ):
            if Int(thread_idx.x) == 0:
                c[0] = c[0] + UInt32(1)

        ctx.enqueue_function[bump_kernel, bump_kernel](
            counter_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        @parameter
        @always_inline
        def kernel_wrapper(
            params: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
            ],
            counter: LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ],
            lr: Scalar[dtype],
            beta1: Scalar[dtype],
            beta2: Scalar[dtype],
            eps: Scalar[dtype],
            log_beta1: Scalar[dtype],
            log_beta2: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params,
                grads,
                state,
                counter,
                lr,
                beta1,
                beta2,
                eps,
                log_beta1,
                log_beta2,
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            counter_t,
            lr,
            beta1,
            beta2,
            eps,
            log_beta1,
            log_beta2,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
