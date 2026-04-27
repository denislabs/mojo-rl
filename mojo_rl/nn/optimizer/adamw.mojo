# =============================================================================
# AdamW Optimizer (Adam with Decoupled Weight Decay)
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.math import sqrt, exp, log
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct AdamW[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
    WEIGHT_DECAY: Float64 = 0.01,
](Optimizer):
    """AdamW optimizer - Adam with decoupled weight decay.

    The key difference from Adam: weight decay is applied directly to parameters,
    not through the gradient. This leads to better generalization.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    GLOBAL_STATE_SIZE = 2:
        - slot 0: UInt32 step counter (bit-patterned), bumped by a preamble
          kernel inside `step_gpu` (Phase 4 of docs/STATE_SIZE_DESIGN.md).
        - slot 1: Scalar[dtype] lr_scale, written by GPUNetworkState.set_lr_scale
          and read by the kernel each step so LR schedules survive
          CUDA-graph replay.
    The CPU `step()` path keeps using the host `step_num` parameter directly.

    All hyperparameters are compile-time struct parameters.
    """

    comptime STATE_PER_PARAM: Int = 2
    comptime GLOBAL_STATE_SIZE: Int = 2

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
        """AdamW update step."""
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var v = rebind[Scalar[dtype]](state[i, 1])

            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            state[i, 0] = m_new
            state[i, 1] = v_new

            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            var p = rebind[Scalar[dtype]](params[i])
            params[i] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

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
        lr_scale_view: LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ],
        base_lr: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        log_beta1: Scalar[dtype],
        log_beta2: Scalar[dtype],
        wd_factor: Scalar[dtype],
    ):
        """AdamW kernel. Bias corrections are computed inside the kernel from
        the post-bump device counter and `lr_scale` is read from a 1-element
        device view so both survive CUDA-graph replay.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var step_f = rebind[Scalar[DType.uint32]](counter[0]).cast[dtype]()
        var one = Scalar[dtype](1.0)
        var bc1 = one - exp(log_beta1 * step_f)
        var bc2 = one - exp(log_beta2 * step_f)
        var lr = base_lr * rebind[Scalar[dtype]](lr_scale_view[0])

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        state[idx, 0] = m_new
        state[idx, 1] = v_new

        var m_hat = m_new / bc1
        var v_hat = v_new / bc2

        var p = rebind[Scalar[dtype]](params[idx])
        params[idx] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

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
    ) raises:
        """Launch AdamW optimization step on GPU. See Adam.step_gpu — same
        preamble-bump-then-step pattern with `lr_scale` read from
        opt_global_state[1]; `step_num` is unused on this path.
        """
        var base_lr = Scalar[dtype](Self.LR)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)
        var log_beta1 = Scalar[dtype](log(Self.BETA1))
        var log_beta2 = Scalar[dtype](log(Self.BETA2))
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

        # Slot 0 → UInt32 step counter view; slot 1 → lr_scale view.
        var counter_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](opt_global_state.ptr.bitcast[Scalar[DType.uint32]]())
        var lr_scale_view = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](opt_global_state.ptr + 1)

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
            lr_scale_view: LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ],
            base_lr: Scalar[dtype],
            beta1: Scalar[dtype],
            beta2: Scalar[dtype],
            eps: Scalar[dtype],
            log_beta1: Scalar[dtype],
            log_beta2: Scalar[dtype],
            wd_factor: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params,
                grads,
                state,
                counter,
                lr_scale_view,
                base_lr,
                beta1,
                beta2,
                eps,
                log_beta1,
                log_beta2,
                wd_factor,
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            counter_t,
            lr_scale_view,
            base_lr,
            beta1,
            beta2,
            eps,
            log_beta1,
            log_beta2,
            wd_factor,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
