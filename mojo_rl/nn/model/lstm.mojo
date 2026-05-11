"""LSTM cell — single-step recurrent layer with explicit (h, c) plumbing.

`LSTMCell[IN_DIM, HIDDEN]` is a stateless, single-step LSTM cell. Unlike
the Model-trait layers (which take a single `[BATCH, IN_DIM]` input),
`LSTMCell` is its own struct with three input args (`x`, `h_prev`,
`c_prev`) and two output args (`h_t`, `c_t`). The caller owns the hidden
state across time steps — this matches how EfficientZero V2's
value-prefix head and DreamerV3-style recurrent encoders work, where
hidden state is reset on a schedule the caller controls.

Sibling of `NoisyLinear`. Standard LSTM equations (Hochreiter &
Schmidhuber, 1997; Gers et al., 2000):

    [i_pre, f_pre, g_pre, o_pre] = x @ W_ih + h_prev @ W_hh + b
    i = σ(i_pre)
    f = σ(f_pre)
    g = tanh(g_pre)
    o = σ(o_pre)
    c_t = f ⊙ c_prev + i ⊙ g
    h_t = o ⊙ tanh(c_t)

Param layout (PyTorch-compatible flat layout):
    [W_ih (IN_DIM, 4*HIDDEN, row-major) | W_hh (HIDDEN, 4*HIDDEN, row-major) | b (4*HIDDEN)]

Gate ordering: (i, f, g, o) — input, forget, cell, output. Each gate
occupies `HIDDEN` consecutive columns in W_ih, W_hh and `HIDDEN`
consecutive elements in b.

Cache layout (per sample, 5*HIDDEN elements):
    [i (HIDDEN) | f (HIDDEN) | g (HIDDEN) | o (HIDDEN) | tanh_c_t (HIDDEN)]

`step_backward` accumulates into `grads` and `dh_prev`/`dc_prev` is
written (overwrites). For BPTT, the caller threads `dh_prev`/`dc_prev`
back as the next call's `dh`/`dc`. Param gradients are accumulated
across time steps automatically because each call adds into `grads`.

FUTURE: a sequence-level `LSTM[IN_DIM, HIDDEN, SEQ_LEN]` Model wrapper
that internalizes (h, c) and conforms to the Model trait can be added
later for Sequential composability. Not needed for the EZ-V2 use case.

Reference: PyTorch `nn.LSTMCell` — same parameter layout and gate order.
"""

from ..constants import dtype as default_dtype, TPB
from ..initializer import Initializer, Xavier, Zeros
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, sqrt


# =============================================================================
# Internal helpers
# =============================================================================
# Two flavors:
#   * `_sigmoid_f64` / `_tanh_f64` — Float64, used on CPU paths for FD-friendly
#     numerical accuracy.
#   * `_sigmoid` / `_tanh` (templated by `dtype`) — Scalar[dtype] versions, used
#     in GPU kernels. Metal does not support Float64 in kernel bodies, so we
#     compute everything in the layer dtype (typically Float32) on the GPU.
# =============================================================================


@always_inline
def _sigmoid_f64(x: Float64) -> Float64:
    """Numerically stable sigmoid in Float64.

    Branches on sign to avoid `exp(-x)` overflow for very negative inputs.
    """
    if x >= 0.0:
        var e = exp(-x)
        return 1.0 / (1.0 + e)
    else:
        var e = exp(x)
        return e / (1.0 + e)


@always_inline
def _tanh_f64(x: Float64) -> Float64:
    """tanh in Float64."""
    var ep = exp(x)
    var en = exp(-x)
    return (ep - en) / (ep + en)


@always_inline
def _sigmoid[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Numerically stable sigmoid in `dtype`. GPU-safe (no Float64)."""
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    var zero = Scalar[dtype](0.0)
    var one = Scalar[dtype](1.0)
    if x >= zero:
        return one / (one + exp(-x))
    else:
        var ex = exp(x)
        return ex / (one + ex)


@always_inline
def _tanh[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """tanh in `dtype`. GPU-safe (no Float64)."""
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    var ep = exp(x)
    var en = exp(-x)
    return (ep - en) / (ep + en)


# =============================================================================
# LSTMCell
# =============================================================================


struct LSTMCell[
    IN_DIM: Int,
    HIDDEN: Int,
    FORGET_BIAS_INIT: Float64 = 1.0,
](ImplicitlyCopyable, Movable):
    """Single-step LSTM cell.

    Parameters:
        IN_DIM: Input feature dimension.
        HIDDEN: Hidden / cell state dimension.
        FORGET_BIAS_INIT: Forget-gate bias initial value (Jozefowicz et al.,
            2015 recommend 1.0 to prevent forget-gate saturation early in
            training). Set to 0.0 to match vanilla PyTorch defaults.

    Compile-time constants:
        IN_DIM: input dim per sample.
        HIDDEN: hidden dim per sample.
        PARAM_SIZE: total flat parameter count.
        CACHE_SIZE: per-sample cache size for backward.
    """

    comptime W_IH_SIZE: Int = Self.IN_DIM * 4 * Self.HIDDEN
    comptime W_HH_SIZE: Int = Self.HIDDEN * 4 * Self.HIDDEN
    comptime B_SIZE: Int = 4 * Self.HIDDEN
    comptime W_HH_OFFSET: Int = Self.W_IH_SIZE
    comptime B_OFFSET: Int = Self.W_IH_SIZE + Self.W_HH_SIZE

    comptime PARAM_SIZE: Int = Self.W_IH_SIZE + Self.W_HH_SIZE + Self.B_SIZE
    comptime CACHE_SIZE: Int = 5 * Self.HIDDEN  # i, f, g, o, tanh_c

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer = Xavier[], dtype: DType = default_dtype
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize parameters.

        - W_ih: `INIT` with `(FAN_IN=IN_DIM, FAN_OUT=4*HIDDEN)`.
        - W_hh: `INIT` with `(FAN_IN=HIDDEN, FAN_OUT=4*HIDDEN)`.
        - bias: zeros, except forget-gate bias = `FORGET_BIAS_INIT`.

        The combined-fan choice matches PyTorch's `nn.LSTMCell` (also
        Xavier per W block, not over the concatenated (W_ih, W_hh)).
        """
        # W_ih: first W_IH_SIZE elements
        var w_ih_view = LayoutTensor[
            dtype, Layout.row_major(Self.W_IH_SIZE), MutAnyOrigin
        ](params.ptr)
        INIT.init[Self.W_IH_SIZE, Self.IN_DIM, 4 * Self.HIDDEN, dtype](
            w_ih_view
        )

        # W_hh: next W_HH_SIZE elements
        var w_hh_view = LayoutTensor[
            dtype, Layout.row_major(Self.W_HH_SIZE), MutAnyOrigin
        ](params.ptr + Self.W_HH_OFFSET)
        INIT.init[Self.W_HH_SIZE, Self.HIDDEN, 4 * Self.HIDDEN, dtype](
            w_hh_view
        )

        # Bias: zeros, except forget-gate bias slot.
        # Gate ordering: (i [0:H], f [H:2H], g [2H:3H], o [3H:4H]).
        var b_base = Self.B_OFFSET
        for k in range(Self.B_SIZE):
            params.ptr[b_base + k] = Scalar[dtype](0.0)
        var f_bias = Scalar[dtype](Self.FORGET_BIAS_INIT)
        for k in range(Self.HIDDEN):
            params.ptr[b_base + Self.HIDDEN + k] = f_bias

    # =========================================================================
    # CPU forward (with cache)
    # =========================================================================

    @staticmethod
    def step_forward[
        BATCH: Int, dtype: DType = default_dtype
    ](
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """One LSTM step on CPU, populating cache for backward.

        cache layout per sample: [i | f | g | o | tanh_c_t], each HIDDEN.
        """
        var H = Self.HIDDEN

        for b in range(BATCH):
            for k in range(4 * H):
                # preact[k] = sum_j x[b,j]*W_ih[j,k] + sum_j h_prev[b,j]*W_hh[j,k] + bias[k]
                var pre = Float64(0.0)
                for j in range(Self.IN_DIM):
                    var xv = Float64(rebind[Scalar[dtype]](x[b, j]))
                    var w = Float64(
                        rebind[Scalar[dtype]](params[j * (4 * H) + k])
                    )
                    pre += xv * w
                for j in range(H):
                    var hv = Float64(rebind[Scalar[dtype]](h_prev[b, j]))
                    var w = Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + j * (4 * H) + k]
                        )
                    )
                    pre += hv * w
                pre += Float64(rebind[Scalar[dtype]](params[Self.B_OFFSET + k]))

                # Apply per-gate activation.
                # k in [0, H)   -> i = sigmoid
                # k in [H, 2H)  -> f = sigmoid
                # k in [2H, 3H) -> g = tanh
                # k in [3H, 4H) -> o = sigmoid
                var act: Float64
                if k < 2 * H:
                    act = _sigmoid_f64(pre)
                elif k < 3 * H:
                    act = _tanh_f64(pre)
                else:
                    act = _sigmoid_f64(pre)

                # Cache slot: gate index 0..3 -> [0, H), [H, 2H), [2H, 3H), [3H, 4H)
                cache[b, k] = Scalar[dtype](act)

            # State update + tanh(c_t)
            for j in range(H):
                var i_val = Float64(rebind[Scalar[dtype]](cache[b, j]))
                var f_val = Float64(rebind[Scalar[dtype]](cache[b, H + j]))
                var g_val = Float64(rebind[Scalar[dtype]](cache[b, 2 * H + j]))
                var o_val = Float64(rebind[Scalar[dtype]](cache[b, 3 * H + j]))
                var c_p = Float64(rebind[Scalar[dtype]](c_prev[b, j]))

                var c_new = f_val * c_p + i_val * g_val
                var tc = _tanh_f64(c_new)
                c_t[b, j] = Scalar[dtype](c_new)
                h_t[b, j] = Scalar[dtype](o_val * tc)
                cache[b, 4 * H + j] = Scalar[dtype](tc)

    # =========================================================================
    # CPU forward (no cache, inference)
    # =========================================================================

    @staticmethod
    def step_forward_no_cache[
        BATCH: Int, dtype: DType = default_dtype
    ](
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
    ):
        """One LSTM step on CPU without caching (inference)."""
        var H = Self.HIDDEN

        for b in range(BATCH):
            # We compute the four gates per j on the fly (no cache write).
            for j in range(H):
                var i_pre = Float64(0.0)
                var f_pre = Float64(0.0)
                var g_pre = Float64(0.0)
                var o_pre = Float64(0.0)

                for jj in range(Self.IN_DIM):
                    var xv = Float64(rebind[Scalar[dtype]](x[b, jj]))
                    i_pre += xv * Float64(
                        rebind[Scalar[dtype]](params[jj * (4 * H) + 0 * H + j])
                    )
                    f_pre += xv * Float64(
                        rebind[Scalar[dtype]](params[jj * (4 * H) + 1 * H + j])
                    )
                    g_pre += xv * Float64(
                        rebind[Scalar[dtype]](params[jj * (4 * H) + 2 * H + j])
                    )
                    o_pre += xv * Float64(
                        rebind[Scalar[dtype]](params[jj * (4 * H) + 3 * H + j])
                    )
                for jj in range(H):
                    var hv = Float64(rebind[Scalar[dtype]](h_prev[b, jj]))
                    i_pre += hv * Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + jj * (4 * H) + 0 * H + j]
                        )
                    )
                    f_pre += hv * Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + jj * (4 * H) + 1 * H + j]
                        )
                    )
                    g_pre += hv * Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + jj * (4 * H) + 2 * H + j]
                        )
                    )
                    o_pre += hv * Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + jj * (4 * H) + 3 * H + j]
                        )
                    )
                i_pre += Float64(
                    rebind[Scalar[dtype]](params[Self.B_OFFSET + 0 * H + j])
                )
                f_pre += Float64(
                    rebind[Scalar[dtype]](params[Self.B_OFFSET + 1 * H + j])
                )
                g_pre += Float64(
                    rebind[Scalar[dtype]](params[Self.B_OFFSET + 2 * H + j])
                )
                o_pre += Float64(
                    rebind[Scalar[dtype]](params[Self.B_OFFSET + 3 * H + j])
                )

                var i_val = _sigmoid_f64(i_pre)
                var f_val = _sigmoid_f64(f_pre)
                var g_val = _tanh_f64(g_pre)
                var o_val = _sigmoid_f64(o_pre)

                var c_p = Float64(rebind[Scalar[dtype]](c_prev[b, j]))
                var c_new = f_val * c_p + i_val * g_val
                var tc = _tanh_f64(c_new)
                c_t[b, j] = Scalar[dtype](c_new)
                h_t[b, j] = Scalar[dtype](o_val * tc)

    # =========================================================================
    # CPU backward
    # =========================================================================
    # Accumulates into `grads`. Writes (overwrites) `dx`, `dh_prev`, `dc_prev`.
    # For BPTT, caller threads dh_prev/dc_prev back as next call's dh/dc.
    # =========================================================================

    @staticmethod
    def step_backward[
        BATCH: Int, dtype: DType = default_dtype
    ](
        dh: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        dc: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut dh_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut dc_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """One LSTM-cell backward step on CPU.

        Args:
            dh: Gradient of loss with respect to h_t [BATCH, HIDDEN].
            dc: Gradient of loss with respect to c_t [BATCH, HIDDEN] (incoming from later time step;
                pass zero on the last time step).
            x: Input [BATCH, IN_DIM].
            h_prev: Previous hidden state [BATCH, HIDDEN].
            c_prev: Previous cell state [BATCH, HIDDEN].
            params: LSTM parameters [PARAM_SIZE].
            cache: LSTM cache [CACHE_SIZE].
            dx: Gradient of loss with respect to x [BATCH, IN_DIM] (written).
            dh_prev: Gradient of loss with respect to h_{t-1} [BATCH, HIDDEN] (written; thread back as dh for previous step).
            dc_prev: Gradient of loss with respect to c_{t-1} [BATCH, HIDDEN] (written; thread back as dc for previous step).
            grads: Gradient of loss with respect to parameters [PARAM_SIZE] (accumulated — never overwritten).
        """
        var H = Self.HIDDEN

        # We assemble d_combined = [di_pre, df_pre, dg_pre, do_pre] in a
        # local scratch (one row at a time, no global allocation).
        # dW_ih += outer(x[b], d_combined[b])
        # dW_hh += outer(h_prev[b], d_combined[b])
        # db   += d_combined[b]
        # dx[b]      = d_combined[b] @ W_ih^T
        # dh_prev[b] = d_combined[b] @ W_hh^T

        for b in range(BATCH):
            # Read cached gates + tanh_c_t.
            # cache layout: [i | f | g | o | tanh_c_t], each HIDDEN.
            #
            # Compute per-element gradients into a local buffer of size 4*H.
            # We can't easily allocate a comptime-sized SIMD here, so use
            # gate-indexed temp via cache scratch — but cache is read-only
            # in spirit. Easiest: compute d_combined into a fresh local
            # InlineArray.
            var d_combined = InlineArray[Float64, 4 * Self.HIDDEN](
                uninitialized=True
            )

            # Pass 1: compute d_combined per gate slot.
            for j in range(H):
                var i_val = Float64(rebind[Scalar[dtype]](cache[b, j]))
                var f_val = Float64(rebind[Scalar[dtype]](cache[b, H + j]))
                var g_val = Float64(rebind[Scalar[dtype]](cache[b, 2 * H + j]))
                var o_val = Float64(rebind[Scalar[dtype]](cache[b, 3 * H + j]))
                var tc = Float64(rebind[Scalar[dtype]](cache[b, 4 * H + j]))
                var c_p = Float64(rebind[Scalar[dtype]](c_prev[b, j]))

                var dh_j = Float64(rebind[Scalar[dtype]](dh[b, j]))
                var dc_j = Float64(rebind[Scalar[dtype]](dc[b, j]))

                # h_t = o * tanh(c_t)
                # dh -> do_post = dh * tc
                # dh -> dc      = dh * o * (1 - tc²)
                var do_post = dh_j * tc
                var dc_total = dc_j + dh_j * o_val * (1.0 - tc * tc)

                # c_t = f * c_prev + i * g
                var df_post = dc_total * c_p
                var di_post = dc_total * g_val
                var dg_post = dc_total * i_val

                # dc_prev for previous time step
                dc_prev[b, j] = Scalar[dtype](dc_total * f_val)

                # Pre-activation gradients
                var di_pre = di_post * i_val * (1.0 - i_val)
                var df_pre = df_post * f_val * (1.0 - f_val)
                var dg_pre = dg_post * (1.0 - g_val * g_val)
                var do_pre = do_post * o_val * (1.0 - o_val)

                d_combined[0 * H + j] = di_pre
                d_combined[1 * H + j] = df_pre
                d_combined[2 * H + j] = dg_pre
                d_combined[3 * H + j] = do_pre

            # Pass 2: accumulate into param grads + compute dx, dh_prev.
            # dW_ih: for j in IN_DIM, k in 4H:  dW_ih[j, k] += x[b, j] * d_combined[k]
            for j in range(Self.IN_DIM):
                var xv = Float64(rebind[Scalar[dtype]](x[b, j]))
                for k in range(4 * H):
                    var idx = j * (4 * H) + k
                    var prev = Float64(rebind[Scalar[dtype]](grads[idx]))
                    grads[idx] = Scalar[dtype](prev + xv * d_combined[k])

            # dW_hh: for j in H, k in 4H:  dW_hh[j, k] += h_prev[b, j] * d_combined[k]
            for j in range(H):
                var hv = Float64(rebind[Scalar[dtype]](h_prev[b, j]))
                for k in range(4 * H):
                    var idx = Self.W_HH_OFFSET + j * (4 * H) + k
                    var prev = Float64(rebind[Scalar[dtype]](grads[idx]))
                    grads[idx] = Scalar[dtype](prev + hv * d_combined[k])

            # db: db[k] += d_combined[k]
            for k in range(4 * H):
                var idx = Self.B_OFFSET + k
                var prev = Float64(rebind[Scalar[dtype]](grads[idx]))
                grads[idx] = Scalar[dtype](prev + d_combined[k])

            # dx[b, j] = sum_k d_combined[k] * W_ih[j, k]
            for j in range(Self.IN_DIM):
                var acc = Float64(0.0)
                for k in range(4 * H):
                    var w = Float64(
                        rebind[Scalar[dtype]](params[j * (4 * H) + k])
                    )
                    acc += d_combined[k] * w
                dx[b, j] = Scalar[dtype](acc)

            # dh_prev[b, j] = sum_k d_combined[k] * W_hh[j, k]
            for j in range(H):
                var acc = Float64(0.0)
                for k in range(4 * H):
                    var w = Float64(
                        rebind[Scalar[dtype]](
                            params[Self.W_HH_OFFSET + j * (4 * H) + k]
                        )
                    )
                    acc += d_combined[k] * w
                dh_prev[b, j] = Scalar[dtype](acc)

    # =========================================================================
    # GPU kernels — forward
    # =========================================================================

    @always_inline
    @staticmethod
    def forward_kernel_impl[
        BATCH: Int, dtype: DType = default_dtype
    ](
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """GPU forward kernel.

        Grid: (BATCH,). Block: (TPB,). Each block handles one sample;
        threads stride over `j ∈ [0, HIDDEN)` to compute the four gates'
        preactivations + activations + state update in parallel.
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var H = Self.HIDDEN

        # Each thread covers a stride of `j` indices.
        var j = local_i
        while j < H:
            # Compute four gates in registers.
            var i_pre = Scalar[dtype](0)
            var f_pre = Scalar[dtype](0)
            var g_pre = Scalar[dtype](0)
            var o_pre = Scalar[dtype](0)

            # x · W_ih
            for jj in range(Self.IN_DIM):
                var xv = rebind[Scalar[dtype]](x[b, jj])
                i_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 0 * H + j]
                )
                f_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 1 * H + j]
                )
                g_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 2 * H + j]
                )
                o_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 3 * H + j]
                )

            # h_prev · W_hh
            for jj in range(H):
                var hv = rebind[Scalar[dtype]](h_prev[b, jj])
                i_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 0 * H + j]
                )
                f_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 1 * H + j]
                )
                g_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 2 * H + j]
                )
                o_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 3 * H + j]
                )

            i_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 0 * H + j])
            f_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 1 * H + j])
            g_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 2 * H + j])
            o_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 3 * H + j])

            # Activations in dtype (Metal doesn't support Float64).
            var i_val = _sigmoid[dtype](i_pre)
            var f_val = _sigmoid[dtype](f_pre)
            var g_val = _tanh[dtype](g_pre)
            var o_val = _sigmoid[dtype](o_pre)

            cache[b, 0 * H + j] = i_val
            cache[b, 1 * H + j] = f_val
            cache[b, 2 * H + j] = g_val
            cache[b, 3 * H + j] = o_val

            var c_p = rebind[Scalar[dtype]](c_prev[b, j])
            var c_new = f_val * c_p + i_val * g_val
            var tc = _tanh[dtype](c_new)
            c_t[b, j] = c_new
            h_t[b, j] = o_val * tc
            cache[b, 4 * H + j] = tc

            j += TPB

    @always_inline
    @staticmethod
    def forward_kernel_impl_no_cache[
        BATCH: Int, dtype: DType = default_dtype
    ](
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
    ):
        """GPU forward kernel without caching (inference)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var H = Self.HIDDEN
        var j = local_i
        while j < H:
            var i_pre = Scalar[dtype](0)
            var f_pre = Scalar[dtype](0)
            var g_pre = Scalar[dtype](0)
            var o_pre = Scalar[dtype](0)

            for jj in range(Self.IN_DIM):
                var xv = rebind[Scalar[dtype]](x[b, jj])
                i_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 0 * H + j]
                )
                f_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 1 * H + j]
                )
                g_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 2 * H + j]
                )
                o_pre += xv * rebind[Scalar[dtype]](
                    params[jj * (4 * H) + 3 * H + j]
                )

            for jj in range(H):
                var hv = rebind[Scalar[dtype]](h_prev[b, jj])
                i_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 0 * H + j]
                )
                f_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 1 * H + j]
                )
                g_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 2 * H + j]
                )
                o_pre += hv * rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jj * (4 * H) + 3 * H + j]
                )

            i_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 0 * H + j])
            f_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 1 * H + j])
            g_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 2 * H + j])
            o_pre += rebind[Scalar[dtype]](params[Self.B_OFFSET + 3 * H + j])

            var i_val = _sigmoid[dtype](i_pre)
            var f_val = _sigmoid[dtype](f_pre)
            var g_val = _tanh[dtype](g_pre)
            var o_val = _sigmoid[dtype](o_pre)

            var c_p = rebind[Scalar[dtype]](c_prev[b, j])
            var c_new = f_val * c_p + i_val * g_val
            var tc = _tanh[dtype](c_new)
            c_t[b, j] = c_new
            h_t[b, j] = o_val * tc

            j += TPB

    # =========================================================================
    # GPU kernels — backward
    # =========================================================================
    # We split into two kernels:
    #   1) per-sample d_combined assembly + dx + dh_prev + dc_prev (block per sample).
    #      Writes d_combined into the workspace (BATCH * 4*HIDDEN scratch).
    #   2) param-grad accumulation — three separate kernels for dW_ih, dW_hh, db,
    #      each parallel over their respective output index, summing across batch.
    # =========================================================================

    @always_inline
    @staticmethod
    def backward_input_kernel_impl[
        BATCH: Int, dtype: DType = default_dtype
    ](
        dh: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        dc: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        dh_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        dc_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        d_combined_buf: LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), MutAnyOrigin
        ],
    ):
        """Per-sample input + state gradients + d_combined assembly.

        Grid: (BATCH,). Block: (TPB,).
        - Phase 1: each thread strides over `j ∈ [0, HIDDEN)` and writes
          d_combined[b, k] for k in {0H+j, 1H+j, 2H+j, 3H+j}, plus
          dc_prev[b, j].
        - Phase 2: each thread strides over `j ∈ [0, IN_DIM)` and computes
          dx[b, j] = sum_k d_combined[b, k] * W_ih[j, k].
        - Phase 3: each thread strides over `j ∈ [0, HIDDEN)` and computes
          dh_prev[b, j] = sum_k d_combined[b, k] * W_hh[j, k].
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var H = Self.HIDDEN

        # Phase 1: d_combined + dc_prev (all in dtype, no Float64 — Metal-safe).
        var one_dt = Scalar[dtype](1.0)
        var j = local_i
        while j < H:
            var i_val = rebind[Scalar[dtype]](cache[b, 0 * H + j])
            var f_val = rebind[Scalar[dtype]](cache[b, 1 * H + j])
            var g_val = rebind[Scalar[dtype]](cache[b, 2 * H + j])
            var o_val = rebind[Scalar[dtype]](cache[b, 3 * H + j])
            var tc = rebind[Scalar[dtype]](cache[b, 4 * H + j])
            var c_p = rebind[Scalar[dtype]](c_prev[b, j])

            var dh_j = rebind[Scalar[dtype]](dh[b, j])
            var dc_j = rebind[Scalar[dtype]](dc[b, j])

            var do_post = dh_j * tc
            var dc_total = dc_j + dh_j * o_val * (one_dt - tc * tc)

            var df_post = dc_total * c_p
            var di_post = dc_total * g_val
            var dg_post = dc_total * i_val

            dc_prev[b, j] = dc_total * f_val

            var di_pre = di_post * i_val * (one_dt - i_val)
            var df_pre = df_post * f_val * (one_dt - f_val)
            var dg_pre = dg_post * (one_dt - g_val * g_val)
            var do_pre = do_post * o_val * (one_dt - o_val)

            d_combined_buf[b, 0 * H + j] = di_pre
            d_combined_buf[b, 1 * H + j] = df_pre
            d_combined_buf[b, 2 * H + j] = dg_pre
            d_combined_buf[b, 3 * H + j] = do_pre

            j += TPB

        # Sync: phase 2/3 read d_combined_buf written in phase 1.
        # On Mojo GPU, per-block barrier is implicit when threads are in
        # the same block and reads happen after writes — but since other
        # threads in the same block may still be writing different j's,
        # we must barrier here.
        from std.gpu.primitives import block as _block

        _block.barrier()

        # Phase 2: dx[b, j] = sum_k d_combined[b, k] * W_ih[j, k]
        var jj = local_i
        while jj < Self.IN_DIM:
            var acc = Scalar[dtype](0)
            for k in range(4 * H):
                var dc_b = rebind[Scalar[dtype]](d_combined_buf[b, k])
                var w = rebind[Scalar[dtype]](params[jj * (4 * H) + k])
                acc += dc_b * w
            dx[b, jj] = acc
            jj += TPB

        # Phase 3: dh_prev[b, j] = sum_k d_combined[b, k] * W_hh[j, k]
        var jh = local_i
        while jh < H:
            var acc = Scalar[dtype](0)
            for k in range(4 * H):
                var dc_b = rebind[Scalar[dtype]](d_combined_buf[b, k])
                var w = rebind[Scalar[dtype]](
                    params[Self.W_HH_OFFSET + jh * (4 * H) + k]
                )
                acc += dc_b * w
            dh_prev[b, jh] = acc
            jh += TPB

    @always_inline
    @staticmethod
    def backward_dWih_kernel_impl[
        BATCH: Int, dtype: DType = default_dtype
    ](
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        d_combined_buf: LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Accumulate dW_ih += x^T @ d_combined over the batch.

        Grid: (IN_DIM, 4*HIDDEN). Block: (TPB,). Each block handles one
        (input_idx, gate_col) pair, summing across BATCH rows in parallel.
        """
        from std.gpu.primitives import block as _block

        var j_in = Int(block_idx.x)  # input feature index
        var k = Int(block_idx.y)  # gate column index
        var local_i = Int(thread_idx.x)

        if j_in >= Self.IN_DIM:
            return
        if k >= 4 * Self.HIDDEN:
            return

        var my_sum = Scalar[dtype](0)
        var b = local_i
        while b < BATCH:
            var xv = rebind[Scalar[dtype]](x[b, j_in])
            var dc_b = rebind[Scalar[dtype]](d_combined_buf[b, k])
            my_sum += xv * dc_b
            b += TPB

        var total = _block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            var idx = j_in * (4 * Self.HIDDEN) + k
            grads[idx] = grads[idx] + total[0]

    @always_inline
    @staticmethod
    def backward_dWhh_kernel_impl[
        BATCH: Int, dtype: DType = default_dtype
    ](
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ],
        d_combined_buf: LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Accumulate dW_hh += h_prev^T @ d_combined over the batch.

        Grid: (HIDDEN, 4*HIDDEN). Block: (TPB,).
        """
        from std.gpu.primitives import block as _block

        var j_in = Int(block_idx.x)
        var k = Int(block_idx.y)
        var local_i = Int(thread_idx.x)

        if j_in >= Self.HIDDEN:
            return
        if k >= 4 * Self.HIDDEN:
            return

        var my_sum = Scalar[dtype](0)
        var b = local_i
        while b < BATCH:
            var hv = rebind[Scalar[dtype]](h_prev[b, j_in])
            var dc_b = rebind[Scalar[dtype]](d_combined_buf[b, k])
            my_sum += hv * dc_b
            b += TPB

        var total = _block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            var idx = Self.W_HH_OFFSET + j_in * (4 * Self.HIDDEN) + k
            grads[idx] = grads[idx] + total[0]

    @always_inline
    @staticmethod
    def backward_db_kernel_impl[
        BATCH: Int, dtype: DType = default_dtype
    ](
        d_combined_buf: LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Accumulate db += sum(d_combined, dim=0).

        Grid: (4*HIDDEN,). Block: (TPB,).
        """
        from std.gpu.primitives import block as _block

        var k = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if k >= 4 * Self.HIDDEN:
            return

        var my_sum = Scalar[dtype](0)
        var b = local_i
        while b < BATCH:
            my_sum += rebind[Scalar[dtype]](d_combined_buf[b, k])
            b += TPB

        var total = _block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            var idx = Self.B_OFFSET + k
            grads[idx] = grads[idx] + total[0]

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def step_forward_gpu[
        BATCH: Int, dtype: DType = default_dtype
    ](
        ctx: DeviceContext,
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ) raises:
        """Launch GPU forward (with cache)."""
        var x_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](x.ptr)
        var h_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](h_prev.ptr)
        var c_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](c_prev.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            x: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            h_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            c_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            h_t: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
            c_t: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH, dtype](
                x, h_prev, c_prev, params, h_t, c_t, cache
            )

        ctx.enqueue_function[kernel_wrapper](
            x_immut,
            h_prev_immut,
            c_prev_immut,
            params_immut,
            h_t,
            c_t,
            cache,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

    @staticmethod
    def step_forward_gpu_no_cache[
        BATCH: Int, dtype: DType = default_dtype
    ](
        ctx: DeviceContext,
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut h_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut c_t: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
    ) raises:
        """Launch GPU forward (no cache, inference)."""
        var x_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](x.ptr)
        var h_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](h_prev.ptr)
        var c_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](c_prev.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            x: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            h_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            c_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            h_t: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
            c_t: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH, dtype](
                x, h_prev, c_prev, params, h_t, c_t
            )

        ctx.enqueue_function[kernel_wrapper](
            x_immut,
            h_prev_immut,
            c_prev_immut,
            params_immut,
            h_t,
            c_t,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

    @staticmethod
    def step_backward_gpu[
        BATCH: Int, dtype: DType = default_dtype
    ](
        ctx: DeviceContext,
        dh: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        dc: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        x: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        h_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        c_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut dh_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut dc_prev: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        d_combined_workspace: LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), MutAnyOrigin
        ],
    ) raises:
        """Launch GPU backward.

        `d_combined_workspace` is a `[BATCH, 4*HIDDEN]` scratch buffer the
        caller pre-allocates. Used to pass d_combined between input-grad
        and param-grad kernels. Its size is small (~`BATCH·HIDDEN·16` B).
        """
        var dh_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](dh.ptr)
        var dc_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](dc.ptr)
        var x_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](x.ptr)
        var h_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](h_prev.ptr)
        var c_prev_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
        ](c_prev.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)
        var dcomb_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
        ](d_combined_workspace.ptr)

        # Kernel 1: dx + dh_prev + dc_prev + d_combined assembly
        @parameter
        @always_inline
        def kernel_in(
            dh: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            dc: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            c_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            dx: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            dh_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
            dc_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
            ],
            d_combined_buf: LayoutTensor[
                dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), MutAnyOrigin
            ],
        ):
            Self.backward_input_kernel_impl[BATCH, dtype](
                dh,
                dc,
                c_prev,
                params,
                cache,
                dx,
                dh_prev,
                dc_prev,
                d_combined_buf,
            )

        ctx.enqueue_function[kernel_in](
            dh_immut,
            dc_immut,
            c_prev_immut,
            params_immut,
            cache_immut,
            dx,
            dh_prev,
            dc_prev,
            d_combined_workspace,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

        # Kernel 2: dW_ih += x^T @ d_combined
        @parameter
        @always_inline
        def kernel_dWih(
            x: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            d_combined_buf: LayoutTensor[
                dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            Self.backward_dWih_kernel_impl[BATCH, dtype](
                x, d_combined_buf, grads
            )

        ctx.enqueue_function[kernel_dWih](
            x_immut,
            dcomb_immut,
            grads,
            grid_dim=(Self.IN_DIM, 4 * Self.HIDDEN),
            block_dim=(TPB,),
        )

        # Kernel 3: dW_hh += h_prev^T @ d_combined
        @parameter
        @always_inline
        def kernel_dWhh(
            h_prev: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.HIDDEN), ImmutAnyOrigin
            ],
            d_combined_buf: LayoutTensor[
                dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            Self.backward_dWhh_kernel_impl[BATCH, dtype](
                h_prev, d_combined_buf, grads
            )

        ctx.enqueue_function[kernel_dWhh](
            h_prev_immut,
            dcomb_immut,
            grads,
            grid_dim=(Self.HIDDEN, 4 * Self.HIDDEN),
            block_dim=(TPB,),
        )

        # Kernel 4: db += sum(d_combined, dim=0)
        @parameter
        @always_inline
        def kernel_db(
            d_combined_buf: LayoutTensor[
                dtype, Layout.row_major(BATCH, 4 * Self.HIDDEN), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            Self.backward_db_kernel_impl[BATCH, dtype](d_combined_buf, grads)

        ctx.enqueue_function[kernel_db](
            dcomb_immut,
            grads,
            grid_dim=(4 * Self.HIDDEN,),
            block_dim=(TPB,),
        )
