"""NoisyLinear layer for Noisy DQN (Fortunato et al., 2018).

Replaces standard Linear with learned parametric noise on weights:
  w = mu_w + sigma_w * f(eps_i) * f(eps_j)    (factorized noise)
  b = mu_b + sigma_b * f(eps_j)

where f(x) = sign(x) * sqrt(|x|) and eps ~ N(0,1).

Parameter layout [PARAM_SIZE = 2*in*out + 2*out]:
  [0 : in*out)                  mu_w
  [in*out : 2*in*out)           sigma_w
  [2*in*out : 2*in*out+out)     mu_b
  [2*in*out+out : end)          sigma_b

Cache layout per sample [CACHE_SIZE = 2*in + out]:
  [0 : in)                      input x
  [in : 2*in)                   noise_p  (f(eps_i), shared across batch)
  [2*in : 2*in+out)             noise_q  (f(eps_j), shared across batch)

Forward with cache (training): uses noisy weights for exploration.
Forward without cache (inference): uses mu-only weights (deterministic).
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.math import sqrt, abs, log, cos
from std.random import random_float64
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# Noise helper: f(x) = sign(x) * sqrt(|x|)
# =============================================================================


@always_inline
def _noise_transform(x: Scalar[dtype]) -> Scalar[dtype]:
    """Factorized noise transform: f(x) = sign(x) * sqrt(|x|)."""
    var ax = abs(x)
    var s = sqrt(ax)
    if x < 0:
        return -s
    return s


# =============================================================================
# CPU noise generation using Box-Muller
# =============================================================================


@always_inline
def _gaussian_sample() -> Scalar[dtype]:
    """Generate a single N(0,1) sample using Box-Muller."""
    var u1 = Scalar[dtype](random_float64(0.0, 1.0))
    var u2 = Scalar[dtype](random_float64(0.0, 1.0))
    # Clamp u1 away from 0 to avoid log(0)
    if u1 < 1e-7:
        u1 = Scalar[dtype](1e-7)
    from std.math import log, cos

    var pi2 = Scalar[dtype](6.283185307179586)
    return sqrt(Scalar[dtype](-2.0) * log(u1)) * cos(pi2 * u2)


# =============================================================================
# NoisyLinear Model
# =============================================================================


struct NoisyLinear[in_dim: Int, out_dim: Int](Model):
    """Noisy linear layer with factorized Gaussian noise.

    Replaces epsilon-greedy exploration with learned parametric noise.
    Training forward uses noisy weights; inference forward uses mu-only.

    Parameters:
        in_dim: Input dimension.
        out_dim: Output dimension.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = 2 * Self.in_dim * Self.out_dim + 2 * Self.out_dim
    comptime CACHE_SIZE: Int = 2 * Self.in_dim + Self.out_dim
    # Workspace stores noise_p [in_dim] + noise_q [out_dim] for GPU kernels.
    # Shared across batch but sized per-sample to ensure adequate allocation.
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.in_dim + Self.out_dim

    # Parameter offsets
    comptime MU_W_OFFSET: Int = 0
    comptime SIGMA_W_OFFSET: Int = Self.in_dim * Self.out_dim
    comptime MU_B_OFFSET: Int = 2 * Self.in_dim * Self.out_dim
    comptime SIGMA_B_OFFSET: Int = 2 * Self.in_dim * Self.out_dim + Self.out_dim

    # Cache offsets
    comptime CACHE_INPUT_OFFSET: Int = 0
    comptime CACHE_NOISE_P_OFFSET: Int = Self.in_dim
    comptime CACHE_NOISE_Q_OFFSET: Int = 2 * Self.in_dim

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
        INIT: Initializer
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize mu_w with standard init, sigma with 0.5/sqrt(in_dim)."""
        # Initialize mu_w using the provided initializer
        comptime MU_W_SIZE = Self.in_dim * Self.out_dim
        var mu_w_t = LayoutTensor[
            dtype, Layout.row_major(MU_W_SIZE), MutAnyOrigin
        ](params.ptr + Self.MU_W_OFFSET)
        INIT.init[MU_W_SIZE, Self.in_dim, Self.out_dim](mu_w_t)

        # Initialize sigma_w to constant 0.5 / sqrt(in_dim)
        var sigma_init = Scalar[dtype](0.5 / sqrt(Float64(Self.in_dim)))
        for i in range(MU_W_SIZE):
            params[Self.SIGMA_W_OFFSET + i] = sigma_init

        # Initialize mu_b to 0
        for j in range(Self.out_dim):
            params[Self.MU_B_OFFSET + j] = Scalar[dtype](0.0)

        # Initialize sigma_b to 0.5 / sqrt(in_dim)
        for j in range(Self.out_dim):
            params[Self.SIGMA_B_OFFSET + j] = sigma_init

    # =========================================================================
    # Forward with cache (training — noisy)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int
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
        """Forward with noise (training). Caches input + noise for backward."""
        # Generate factorized noise once for all samples
        var noise_p = InlineArray[Scalar[dtype], Self.in_dim](
            uninitialized=True
        )
        var noise_q = InlineArray[Scalar[dtype], Self.out_dim](
            uninitialized=True
        )
        for i in range(Self.in_dim):
            noise_p[i] = _noise_transform(_gaussian_sample())
        for j in range(Self.out_dim):
            noise_q[j] = _noise_transform(_gaussian_sample())

        for b in range(BATCH):
            # Cache input
            for i in range(Self.in_dim):
                cache[b, Self.CACHE_INPUT_OFFSET + i] = input[b, i]
            # Cache noise (same for all samples)
            for i in range(Self.in_dim):
                cache[b, Self.CACHE_NOISE_P_OFFSET + i] = noise_p[i]
            for j in range(Self.out_dim):
                cache[b, Self.CACHE_NOISE_Q_OFFSET + j] = noise_q[j]

            # Compute y = x @ W_noisy + b_noisy
            for j in range(Self.out_dim):
                var mu_b = params[Self.MU_B_OFFSET + j]
                var sigma_b = params[Self.SIGMA_B_OFFSET + j]
                var acc = mu_b + sigma_b * noise_q[j]
                for i in range(Self.in_dim):
                    var mu_w = params[Self.MU_W_OFFSET + i * Self.out_dim + j]
                    var sigma_w = params[
                        Self.SIGMA_W_OFFSET + i * Self.out_dim + j
                    ]
                    var w = mu_w + sigma_w * noise_p[i] * noise_q[j]
                    acc += rebind[Scalar[dtype]](input[b, i]) * w
                output[b, j] = acc

    # =========================================================================
    # Forward without cache (inference — mu-only, no noise)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int
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
    ):
        """Forward with noise but no caching (for action selection).

        Noise provides exploration. No cache needed since this isn't
        used for gradient computation.
        """
        # Generate factorized noise
        var noise_p = InlineArray[Scalar[dtype], Self.in_dim](
            uninitialized=True
        )
        var noise_q = InlineArray[Scalar[dtype], Self.out_dim](
            uninitialized=True
        )
        for i in range(Self.in_dim):
            noise_p[i] = _noise_transform(_gaussian_sample())
        for j in range(Self.out_dim):
            noise_q[j] = _noise_transform(_gaussian_sample())

        for b in range(BATCH):
            for j in range(Self.out_dim):
                var mu_b = rebind[Scalar[dtype]](params[Self.MU_B_OFFSET + j])
                var sigma_b = rebind[Scalar[dtype]](
                    params[Self.SIGMA_B_OFFSET + j]
                )
                var acc = mu_b + sigma_b * noise_q[j]
                for i in range(Self.in_dim):
                    var mu_w = rebind[Scalar[dtype]](
                        params[Self.MU_W_OFFSET + i * Self.out_dim + j]
                    )
                    var sigma_w = rebind[Scalar[dtype]](
                        params[Self.SIGMA_W_OFFSET + i * Self.out_dim + j]
                    )
                    var w = mu_w + sigma_w * noise_p[i] * noise_q[j]
                    acc += rebind[Scalar[dtype]](input[b, i]) * w
                output[b, j] = acc

    # =========================================================================
    # Backward
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: compute grad_input and accumulate grads for mu and sigma.
        """
        # Read noise from cache (same for all samples, use sample 0)
        var noise_p = InlineArray[Scalar[dtype], Self.in_dim](
            uninitialized=True
        )
        var noise_q = InlineArray[Scalar[dtype], Self.out_dim](
            uninitialized=True
        )
        for i in range(Self.in_dim):
            noise_p[i] = rebind[Scalar[dtype]](
                cache[0, Self.CACHE_NOISE_P_OFFSET + i]
            )
        for j in range(Self.out_dim):
            noise_q[j] = rebind[Scalar[dtype]](
                cache[0, Self.CACHE_NOISE_Q_OFFSET + j]
            )

        for b in range(BATCH):
            # Read cached input
            # Compute grad_input: dx = dy @ W.T
            for i in range(Self.in_dim):
                var acc = Scalar[dtype](0.0)
                for j in range(Self.out_dim):
                    var mu_w = rebind[Scalar[dtype]](
                        params[Self.MU_W_OFFSET + i * Self.out_dim + j]
                    )
                    var sigma_w = rebind[Scalar[dtype]](
                        params[Self.SIGMA_W_OFFSET + i * Self.out_dim + j]
                    )
                    var w = mu_w + sigma_w * noise_p[i] * noise_q[j]
                    acc += rebind[Scalar[dtype]](grad_output[b, j]) * w
                grad_input[b, i] = acc

            # Accumulate parameter gradients
            var x_cached = InlineArray[Scalar[dtype], Self.in_dim](
                uninitialized=True
            )
            for i in range(Self.in_dim):
                x_cached[i] = rebind[Scalar[dtype]](
                    cache[b, Self.CACHE_INPUT_OFFSET + i]
                )

            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    var dy = rebind[Scalar[dtype]](grad_output[b, j])
                    var x_i = x_cached[i]
                    var xdy = x_i * dy
                    # dmu_w += x * dy
                    grads[Self.MU_W_OFFSET + i * Self.out_dim + j] += xdy
                    # dsigma_w += x * dy * noise_p[i] * noise_q[j]
                    grads[Self.SIGMA_W_OFFSET + i * Self.out_dim + j] += (
                        xdy * noise_p[i] * noise_q[j]
                    )

            for j in range(Self.out_dim):
                var dy = rebind[Scalar[dtype]](grad_output[b, j])
                # dmu_b += dy
                grads[Self.MU_B_OFFSET + j] += dy
                # dsigma_b += dy * noise_q[j]
                grads[Self.SIGMA_B_OFFSET + j] += dy * noise_q[j]

    # =========================================================================
    # GPU Forward with cache (training — noisy)
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward with noise. One thread per (batch, out_dim)."""
        # Use workspace to store noise_p [in_dim] + noise_q [out_dim]
        # noise is generated by thread 0 of first block, then all threads use it
        var noise_ptr = workspace.unsafe_ptr()
        var noise_p_t = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
        ](noise_ptr)
        var noise_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](noise_ptr + Self.in_dim)

        # Generate a varying seed on CPU (different each forward call)
        var seed_base = UInt64(
            random_float64(0.0, Float64(UInt32.MAX))
        ) * UInt64(2654435761)

        # Step 1: Generate noise in workspace
        @always_inline
        def gen_noise_kernel(
            np: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
            ],
            nq: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
            base_seed: Scalar[DType.uint64],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            var bs = UInt64(base_seed)
            # Generate noise_p
            if tid < Self.in_dim:
                var rng = PhiloxRandom(
                    seed=bs + UInt64(tid),
                    offset=0,
                )
                var vals = rng.step_uniform()
                var u1 = vals[0]
                if u1 < 1e-7:
                    u1 = 1e-7
                var gauss = Scalar[dtype](
                    sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * vals[1])
                )
                var ax = abs(gauss)
                var sg = sqrt(ax)
                np[tid] = -sg if gauss < 0 else sg
            # Generate noise_q
            if tid < Self.out_dim:
                var rng = PhiloxRandom(
                    seed=bs + UInt64(tid + Self.in_dim + 10000),
                    offset=0,
                )
                var vals = rng.step_uniform()
                var u1 = vals[0]
                if u1 < 1e-7:
                    u1 = 1e-7
                var gauss = Scalar[dtype](
                    sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * vals[1])
                )
                var ax = abs(gauss)
                var sg = sqrt(ax)
                nq[tid] = -sg if gauss < 0 else sg

        comptime NOISE_DIM = max(Self.in_dim, Self.out_dim)
        var seed_s = Scalar[DType.uint64](seed_base)
        ctx.enqueue_function[gen_noise_kernel, gen_noise_kernel](
            noise_p_t,
            noise_q_t,
            seed_s,
            grid_dim=((NOISE_DIM + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        # Step 2: Forward pass + cache
        @always_inline
        def fwd_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            inp: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            par: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            cch: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            np: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
            ],
            nq: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.out_dim:
                return
            var b = idx // Self.out_dim
            var j = idx % Self.out_dim

            # Cache input and noise (all threads for this batch element)
            if j == 0:
                for i in range(Self.in_dim):
                    cch[b, Self.CACHE_INPUT_OFFSET + i] = inp[b, i]
                    cch[b, Self.CACHE_NOISE_P_OFFSET + i] = np[i]
                for jj in range(Self.out_dim):
                    cch[b, Self.CACHE_NOISE_Q_OFFSET + jj] = nq[jj]

            # Compute output[b, j]
            var nq_j = rebind[Scalar[dtype]](nq[j])
            var mu_b = rebind[Scalar[dtype]](par[Self.MU_B_OFFSET + j])
            var sigma_b = rebind[Scalar[dtype]](par[Self.SIGMA_B_OFFSET + j])
            var acc = mu_b + sigma_b * nq_j

            for i in range(Self.in_dim):
                var x_i = rebind[Scalar[dtype]](inp[b, i])
                var mu_w = rebind[Scalar[dtype]](
                    par[Self.MU_W_OFFSET + i * Self.out_dim + j]
                )
                var sigma_w = rebind[Scalar[dtype]](
                    par[Self.SIGMA_W_OFFSET + i * Self.out_dim + j]
                )
                var np_i = rebind[Scalar[dtype]](np[i])
                var w = mu_w + sigma_w * np_i * nq_j
                acc += x_i * w

            dst[b, j] = acc

        comptime TOTAL = BATCH * Self.out_dim
        ctx.enqueue_function[fwd_kernel, fwd_kernel](
            output,
            input,
            params,
            cache,
            noise_p_t,
            noise_q_t,
            grid_dim=((TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Forward without cache (inference — mu-only)
    # =========================================================================

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward with noise, no caching (for action selection)."""
        # Generate noise into workspace, then compute noisy forward
        var noise_ptr = workspace.unsafe_ptr()
        var noise_p_t = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
        ](noise_ptr)
        var noise_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](noise_ptr + Self.in_dim)

        # Generate a varying seed on CPU
        var seed_base_nc = UInt64(
            random_float64(0.0, Float64(UInt32.MAX))
        ) * UInt64(2654435761) + UInt64(99991)

        # Generate noise
        @always_inline
        def gen_noise_nc(
            np: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
            ],
            nq: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
            base_seed: Scalar[DType.uint64],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            var bs = UInt64(base_seed)
            if tid < Self.in_dim:
                var rng = PhiloxRandom(
                    seed=bs + UInt64(tid),
                    offset=0,
                )
                var vals = rng.step_uniform()
                var u1 = vals[0]
                if u1 < 1e-7:
                    u1 = 1e-7
                var gauss = Scalar[dtype](
                    sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * vals[1])
                )
                var ax = abs(gauss)
                var sg = sqrt(ax)
                np[tid] = -sg if gauss < 0 else sg
            if tid < Self.out_dim:
                var rng = PhiloxRandom(
                    seed=bs + UInt64(tid + Self.in_dim + 10000),
                    offset=0,
                )
                var vals = rng.step_uniform()
                var u1 = vals[0]
                if u1 < 1e-7:
                    u1 = 1e-7
                var gauss = Scalar[dtype](
                    sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * vals[1])
                )
                var ax = abs(gauss)
                var sg = sqrt(ax)
                nq[tid] = -sg if gauss < 0 else sg

        comptime NOISE_DIM = max(Self.in_dim, Self.out_dim)
        var seed_nc_s = Scalar[DType.uint64](seed_base_nc)
        ctx.enqueue_function[gen_noise_nc, gen_noise_nc](
            noise_p_t,
            noise_q_t,
            seed_nc_s,
            grid_dim=((NOISE_DIM + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        # Noisy forward
        @always_inline
        def fwd_noisy_nc_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            inp: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            par: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            np: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim), MutAnyOrigin
            ],
            nq: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.out_dim:
                return
            var b = idx // Self.out_dim
            var j = idx % Self.out_dim

            var nq_j = rebind[Scalar[dtype]](nq[j])
            var mu_b = rebind[Scalar[dtype]](par[Self.MU_B_OFFSET + j])
            var sigma_b = rebind[Scalar[dtype]](par[Self.SIGMA_B_OFFSET + j])
            var acc = mu_b + sigma_b * nq_j
            for i in range(Self.in_dim):
                var x_i = rebind[Scalar[dtype]](inp[b, i])
                var mu_w = rebind[Scalar[dtype]](
                    par[Self.MU_W_OFFSET + i * Self.out_dim + j]
                )
                var sigma_w = rebind[Scalar[dtype]](
                    par[Self.SIGMA_W_OFFSET + i * Self.out_dim + j]
                )
                var np_i = rebind[Scalar[dtype]](np[i])
                acc += x_i * (mu_w + sigma_w * np_i * nq_j)
            dst[b, j] = acc

        comptime TOTAL = BATCH * Self.out_dim
        ctx.enqueue_function[fwd_noisy_nc_kernel, fwd_noisy_nc_kernel](
            output,
            input,
            params,
            noise_p_t,
            noise_q_t,
            grid_dim=((TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    def backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU backward. Kernel 1: grad_input. Kernel 2: param grads."""

        # Kernel 1: Compute grad_input — one thread per (batch, in_dim)
        @always_inline
        def grad_input_kernel(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            par: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            cch: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.in_dim:
                return
            var b = idx // Self.in_dim
            var i = idx % Self.in_dim

            # Read noise from cache (sample 0)
            var np_i = rebind[Scalar[dtype]](
                cch[0, Self.CACHE_NOISE_P_OFFSET + i]
            )

            var acc = Scalar[dtype](0.0)
            for j in range(Self.out_dim):
                var nq_j = rebind[Scalar[dtype]](
                    cch[0, Self.CACHE_NOISE_Q_OFFSET + j]
                )
                var mu_w = rebind[Scalar[dtype]](
                    par[Self.MU_W_OFFSET + i * Self.out_dim + j]
                )
                var sigma_w = rebind[Scalar[dtype]](
                    par[Self.SIGMA_W_OFFSET + i * Self.out_dim + j]
                )
                var w = mu_w + sigma_w * np_i * nq_j
                acc += rebind[Scalar[dtype]](go[b, j]) * w
            gi[b, i] = acc

        comptime GI_TOTAL = BATCH * Self.in_dim
        ctx.enqueue_function[grad_input_kernel, grad_input_kernel](
            grad_input,
            grad_output,
            params,
            cache,
            grid_dim=((GI_TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        # Kernel 2: Accumulate param grads — one thread per (i, j) pair
        # Each thread loops over batch dimension to accumulate
        @always_inline
        def param_grad_kernel(
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            par: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            cch: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            grd: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            comptime WEIGHT_SIZE = Self.in_dim * Self.out_dim

            if idx < WEIGHT_SIZE:
                # Weight gradients: dmu_w and dsigma_w
                var i = idx // Self.out_dim
                var j = idx % Self.out_dim
                var np_i = rebind[Scalar[dtype]](
                    cch[0, Self.CACHE_NOISE_P_OFFSET + i]
                )
                var nq_j = rebind[Scalar[dtype]](
                    cch[0, Self.CACHE_NOISE_Q_OFFSET + j]
                )
                var noise_ij = np_i * nq_j

                var dmu_acc = Scalar[dtype](0.0)
                var dsigma_acc = Scalar[dtype](0.0)
                for b in range(BATCH):
                    var x_i = rebind[Scalar[dtype]](
                        cch[b, Self.CACHE_INPUT_OFFSET + i]
                    )
                    var dy_j = rebind[Scalar[dtype]](go[b, j])
                    var xdy = x_i * dy_j
                    dmu_acc += xdy
                    dsigma_acc += xdy * noise_ij
                grd[Self.MU_W_OFFSET + idx] = (
                    rebind[Scalar[dtype]](grd[Self.MU_W_OFFSET + idx]) + dmu_acc
                )
                grd[Self.SIGMA_W_OFFSET + idx] = (
                    rebind[Scalar[dtype]](grd[Self.SIGMA_W_OFFSET + idx])
                    + dsigma_acc
                )

            elif idx < WEIGHT_SIZE + Self.out_dim:
                # Bias gradients: dmu_b and dsigma_b
                var j = idx - WEIGHT_SIZE
                var nq_j = rebind[Scalar[dtype]](
                    cch[0, Self.CACHE_NOISE_Q_OFFSET + j]
                )
                var dmu_b_acc = Scalar[dtype](0.0)
                var dsigma_b_acc = Scalar[dtype](0.0)
                for b in range(BATCH):
                    var dy_j = rebind[Scalar[dtype]](go[b, j])
                    dmu_b_acc += dy_j
                    dsigma_b_acc += dy_j * nq_j
                grd[Self.MU_B_OFFSET + j] = (
                    rebind[Scalar[dtype]](grd[Self.MU_B_OFFSET + j]) + dmu_b_acc
                )
                grd[Self.SIGMA_B_OFFSET + j] = (
                    rebind[Scalar[dtype]](grd[Self.SIGMA_B_OFFSET + j])
                    + dsigma_b_acc
                )

        comptime PG_TOTAL = Self.in_dim * Self.out_dim + Self.out_dim
        ctx.enqueue_function[param_grad_kernel, param_grad_kernel](
            grad_output,
            params,
            cache,
            grads,
            grid_dim=((PG_TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
