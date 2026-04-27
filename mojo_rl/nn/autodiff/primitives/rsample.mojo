"""RSampleOp: Reparameterized sampling with tanh squashing (DiffOp).

Implements the reparameterization trick used in SAC:
    z = mean + exp(log_std) * noise
    action = tanh(z) * action_scale
    log_prob = sum_j(gaussian_log_prob_j - squash_correction_j)

This replaces ~100 lines of manual backward code in actor_loss.mojo by
encoding the full forward/backward as a standard DiffOp that composes
with Sequential/AutoDiffChain.

Input layout:  [BATCH, 2 * action_dim] = [mean || tanh(raw_log_std)]
  (from Parallel[mean_head, LinearTanh[log_std_head]] actor architecture)

Output layout: [BATCH, action_dim + 1] = [action || log_prob]
  (log_prob is the summed log probability per sample, scalar per row)

Cache layout:  [BATCH, 3 * action_dim] = [tanh_z || noise || log_std_rescaled]
  (tanh_z is the UNSCALED tanh(z), not action_scale * tanh(z), so that the
   backward can reuse it directly for the entropy term 2*tanh_z and the
   squash Jacobian 1 - tanh_z²)

The affine rescaling of log_std is handled internally:
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (tanh_val + 1)

On backward, the chain rule through this rescaling is automatic — no manual
AFFINE_SCALE needed by the caller.

action_scale: Output scale for the action (action = action_scale * tanh(z)).
  Default 1.0 matches the original behavior. When != 1.0, the backward
  chain-rules an extra action_scale factor through the tanh so the critic
  sees a consistent action distribution.
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import exp, log, sqrt, tanh, cos, pi
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


# Box-Muller transform for Gaussian noise (CPU only)
@always_inline
def _gaussian_noise() -> Float64:
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


comptime EPS: Float64 = 1e-6
comptime LOG_2PI: Float64 = 1.8378770664093453
comptime LOG_2: Float64 = 0.6931471805599453


struct RSampleOp[
    action_dim: Int,
    log_std_min: Float64 = -5.0,
    log_std_max: Float64 = 2.0,
    action_scale: Float64 = 1.0,
](DiffOp):
    """Reparameterized sampling with tanh squashing.

    Encapsulates the full rsample forward/backward as a DiffOp.
    Composes naturally with Sequential after a Parallel[mean, LinearTanh[log_std]]
    actor architecture.

    IN_DIM = 2 * action_dim  (mean || tanh(raw_log_std))
    OUT_DIM = action_dim + 1 (action || log_prob)
    PARAM_SIZE = 0           (no learnable parameters)
    CACHE_SIZE = 3 * action_dim (unscaled tanh_z, noise, rescaled log_std)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 1
    comptime IN_DIM: Int = 2 * Self.action_dim
    comptime OUT_DIM: Int = Self.action_dim + 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 3 * Self.action_dim
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 1  # RNG seed slot (read from GPU buffer for CUDA graph safety)

    # Derivative of the affine rescaling: d(log_std)/d(tanh_val)
    comptime AFFINE_DERIV: Float64 = 0.5 * (Self.log_std_max - Self.log_std_min)

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval
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
        comptime A = Self.action_dim
        # Cache layout offsets (per sample):
        #   [0 .. A)        = action (tanh(z))
        #   [A .. 2A)       = noise
        #   [2A .. 3A)      = rescaled log_std

        for b in range(BATCH):
            var total_log_prob: Float64 = 0.0

            for j in range(A):
                var mean = Float64(rebind[Scalar[dtype]](input[b, j]))
                var tanh_raw = Float64(rebind[Scalar[dtype]](input[b, A + j]))

                # Affine rescaling: log_std in [log_std_min, log_std_max]
                var ls = Self.log_std_min + Self.AFFINE_DERIV * (tanh_raw + 1.0)

                # Sample noise
                var noise = _gaussian_noise()

                # Reparameterized sample: z = mean + exp(log_std) * noise
                var std = exp(ls)
                var z = mean + std * noise

                # Tanh squashing (unscaled tanh(z), used for backward + log_prob)
                var exp_z = exp(z)
                var exp_neg_z = exp(-z)
                var tanh_z = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

                # Gaussian log probability
                var log_gaussian = -0.5 * (LOG_2PI + 2.0 * ls + noise * noise)

                # Squashing correction (numerically stable form)
                # log(1 - tanh²(z)) = 2·(log(2) - |z| - log(1 + exp(-2|z|)))
                var abs_z = z if z >= 0.0 else -z
                var squash_correction = 2.0 * (
                    LOG_2 - abs_z - log(1.0 + exp(-2.0 * abs_z))
                )
                total_log_prob += log_gaussian - squash_correction

                # Write output (SCALED action)
                output[b, j] = Scalar[dtype](tanh_z * Self.action_scale)

                # Cache UNSCALED tanh_z for backward (entropy + squash Jacobian)
                cache[b, j] = Scalar[dtype](tanh_z)
                cache[b, A + j] = Scalar[dtype](noise)  # noise
                cache[b, 2 * A + j] = Scalar[dtype](ls)  # rescaled log_std

            output[b, A] = Scalar[dtype](total_log_prob)

    # =========================================================================
    # CPU vjp
    # =========================================================================

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
        """Backward through reparameterization trick.

        grad_output[:, :A] = grad_action (e.g., -dQ/da from critic)
        grad_output[:, A]  = grad_log_prob (e.g., alpha/batch for entropy)

        Produces:
        grad_input[:, :A]  = grad_mean
        grad_input[:, A:]  = grad_tanh_raw_log_std (includes affine chain rule)
        """
        comptime A = Self.action_dim

        for b in range(BATCH):
            var glp = Float64(rebind[Scalar[dtype]](grad_output[b, A]))

            for j in range(A):
                var ga = Float64(rebind[Scalar[dtype]](grad_output[b, j]))
                # Cache stores UNSCALED tanh_z; downstream ops receive
                # action = action_scale * tanh_z, so chain-rule ga through
                # the scale (ga is dL/d(scaled_action)).
                var a = Float64(rebind[Scalar[dtype]](cache[b, j]))
                var noise = Float64(rebind[Scalar[dtype]](cache[b, A + j]))
                var ls = Float64(rebind[Scalar[dtype]](cache[b, 2 * A + j]))

                var std = exp(ls)

                # d(scaled_action)/d(z) = action_scale * (1 - tanh²(z))
                var dtanh_dz = (1.0 - a * a) * Self.action_scale

                # d(log_prob)/d(z) from stable squash correction
                # d/dz[-log(1 - tanh²(z))] = 2·tanh(z) = 2·a  (scale-independent)
                var dlogprob_dz = 2.0 * a

                # Total gradient w.r.t. z
                var grad_z = ga * dtanh_dz + glp * dlogprob_dz

                # grad_mean = grad_z (d(z)/d(mean) = 1)
                grad_input[b, j] = Scalar[dtype](grad_z)

                # grad_log_std = grad_z * std * noise + glp * (-1)
                var grad_ls = grad_z * std * noise + glp * (-1.0)

                # Chain rule through affine rescaling:
                # d(log_std)/d(tanh_raw) = AFFINE_DERIV
                # But tanh_raw is the OUTPUT of LinearTanh, so the tanh
                # derivative is already handled by LinearTanh's backward.
                # We only need the affine scale: d(ls)/d(tanh_val) = AFFINE_DERIV
                grad_input[b, A + j] = Scalar[dtype](
                    grad_ls * Self.AFFINE_DERIV
                )

    # =========================================================================
    # GPU eval
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        ws: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        """One thread per batch element. Loops over action_dim internally.

        RNG seed is read from ws[0] (GPU buffer, CUDA graph safe — value
        changes between graph replays unlike a baked scalar arg).
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        comptime A = Self.action_dim
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        # Read RNG seed from workspace buffer (not a scalar — CUDA graph safe)
        var rng_seed = UInt64(Int(ws.ptr[0]))

        var total_log_prob = Scalar[dtype](0.0)

        for j in range(A):
            var mean = rebind[Scalar[dtype]](input[b, j])
            var tanh_raw = rebind[Scalar[dtype]](input[b, A + j])

            # Affine rescaling
            var ls = Scalar[dtype](Self.log_std_min) + Scalar[dtype](
                Self.AFFINE_DERIV
            ) * (tanh_raw + Scalar[dtype](1.0))

            # PhiloxRandom Box-Muller for Gaussian noise (GPU-safe, no Float64)
            var philox = PhiloxRandom(
                seed=rng_seed + UInt64(b) * UInt64(A) + UInt64(j),
                offset=0,
            )
            var rand_vals = philox.step_uniform()
            var u1 = Float32(rand_vals[0]) + Float32(1e-8)
            var u2 = Float32(rand_vals[1])
            var mag = sqrt(Float32(-2.0) * log(u1))
            var noise = Scalar[dtype](
                mag * cos(u2 * Float32(6.283185307179586))
            )

            var std = exp(ls)
            var z = mean + std * noise
            var tanh_z = tanh(z)

            # Log probability
            var log_gaussian = Scalar[dtype](-0.5) * (
                Scalar[dtype](LOG_2PI) + Scalar[dtype](2.0) * ls + noise * noise
            )
            # Squashing correction (numerically stable form)
            # log(1 - tanh²(z)) = 2·(log(2) - |z| - log(1 + exp(-2|z|)))
            var abs_z = z if z >= Scalar[dtype](0.0) else -z
            var squash_correction = Scalar[dtype](2.0) * (
                Scalar[dtype](LOG_2) - abs_z
                - log(Scalar[dtype](1.0) + exp(Scalar[dtype](-2.0) * abs_z))
            )
            total_log_prob += log_gaussian - squash_correction

            # Output SCALED action; cache UNSCALED tanh_z for backward
            output[b, j] = tanh_z * Scalar[dtype](Self.action_scale)
            cache[b, j] = tanh_z
            cache[b, A + j] = noise
            cache[b, 2 * A + j] = ls

        output[b, A] = total_log_prob

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
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB
        # Workspace[0] contains the RNG seed (written by caller before forward).
        # Using a buffer instead of a scalar makes this CUDA graph safe.
        var ws_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](workspace)

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            ws: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache, ws)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            ws_t,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU vjp
    # =========================================================================

    @always_inline
    @staticmethod
    def vjp_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """One thread per (batch, action_dim) element."""
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        comptime A = Self.action_dim
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * A:
            return

        var b = idx // A
        var j = idx % A

        var ga = rebind[Scalar[dtype]](grad_output[b, j])
        var glp = rebind[Scalar[dtype]](grad_output[b, A])
        # Cache stores UNSCALED tanh_z; ga is dL/d(scaled_action) where
        # scaled_action = action_scale * tanh_z, so we chain-rule action_scale
        # through the tanh derivative below.
        var a = rebind[Scalar[dtype]](cache[b, j])
        var noise = rebind[Scalar[dtype]](cache[b, A + j])
        var ls = rebind[Scalar[dtype]](cache[b, 2 * A + j])

        var std = exp(ls)
        var one = Scalar[dtype](1.0)

        # d(scaled_action)/d(z) = action_scale * (1 - tanh_z²)
        var dtanh_dz = (one - a * a) * Scalar[dtype](Self.action_scale)

        # d(log_prob)/d(z) from stable squash correction
        # d/dz[-log(1 - tanh²(z))] = 2·tanh(z) = 2·a (scale-independent)
        var dlogprob_dz = Scalar[dtype](2.0) * a

        var grad_z = ga * dtanh_dz + glp * dlogprob_dz

        # grad_mean
        grad_input[b, j] = grad_z

        # grad_log_std * affine_deriv
        var grad_ls = grad_z * std * noise + glp * Scalar[dtype](-1.0)
        grad_input[b, A + j] = grad_ls * Scalar[dtype](Self.AFFINE_DERIV)

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
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)
        comptime total = BATCH * Self.action_dim
        var grid_x = (total + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            c: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.vjp_kernel_impl[BATCH, dtype](gi, go, c)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
