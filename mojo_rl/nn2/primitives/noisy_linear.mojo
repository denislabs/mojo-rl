"""NoisyLinear[IN, OUT] — Factorized Gaussian Noisy Linear (Fortunato et al. 2018).

Replaces `y = x @ W + b` with `y = x @ W_eff + b_eff` where:

    W_eff[i, j] = μ_W[i, j] + σ_W[i, j] · f(ε_in[i]) · f(ε_out[j])
    b_eff[j]    = μ_b[j]    + σ_b[j]    · f(ε_out[j])

with `f(x) = sign(x) · sqrt(|x|)` and `ε_in[i], ε_out[j] ~ N(0, 1)`
sampled fresh per forward call. The factorization means only IN + OUT
noise samples are needed per forward (instead of IN × OUT), at the
cost of correlations between W entries — but empirically matches the
canonical Gaussian-noise NoisyDQN performance (Fortunato §3.2).

**4 trainable params** (PARAM_SIZE = 2·IN·OUT + 2·OUT):
  - μ_W  [IN × OUT]  — mean weights
  - σ_W  [IN × OUT]  — std-multiplier weights
  - μ_b  [OUT]       — mean biases
  - σ_b  [OUT]       — std-multiplier biases

**Init (Fortunato §3.2)**:
  - μ ~ Uniform(−1/√IN, +1/√IN)
  - σ = σ_0 / √IN   (σ_0 = 0.5)

Backward:
  Let n_W[i,j] = f(ε_in[i]) · f(ε_out[j]),  n_b[j] = f(ε_out[j])
    ∂L/∂μ_W[i, j] = Σ_b x[b, i] · grad_out[b, j]
    ∂L/∂σ_W[i, j] = ∂L/∂μ_W[i, j] · n_W[i, j]
    ∂L/∂μ_b[j]    = Σ_b grad_out[b, j]
    ∂L/∂σ_b[j]    = ∂L/∂μ_b[j] · n_b[j]
    ∂L/∂x[b, i]   = Σ_j grad_out[b, j] · W_eff[i, j]

CPU + GPU. GPU forward/vjp lift the CPU mathematical operations into
kernels and reuse `box_muller_normal_gpu` (Philox) for on-device noise
sampling — same sequence-determinism story as SAC's GPU action sampling.

Use in Noisy DQN: replace the last `Linear[H, NA]` in the Q-net with
`NoisyLinear[H, NA]`. Drop ε-greedy at the trainer level (set
`epsilon=0`, `epsilon_min=0`).
"""

from std.math import sqrt as fsqrt, log as flog, cos as fcos, pi
from std.memory import alloc
from std.random import random_float64
from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_gpu_buffer,
)
from .linear import _transpose_kernel, _accum_kernel
from ..random.box_muller import (
    box_muller_normal_gpu,
    box_muller_normal_gpu_dev,
    advance_rng_offset_kernel,
)


# ──────────────────────────────────────────────────────────────────────
# CPU helpers — Gaussian via Box-Muller, factorized f(x) = sign(x)·√|x|.
# Inlined into forward to avoid Mojo-nightly tuple-return ergonomics.
# ──────────────────────────────────────────────────────────────────────


def _fnoise(x: Scalar[DT]) -> Scalar[DT]:
    """`f(x) = sign(x) · sqrt(|x|)`."""
    if x >= Scalar[DT](0.0):
        return fsqrt(x)
    return -fsqrt(-x)


def _sample_factorized_noise(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int,
):
    """Fill `ptr[0..n]` with `f(z)` where `z ~ N(0, 1)` via Box-Muller.
    Each Box-Muller draw yields a pair (cos, sin) of independent normals;
    we consume both before drawing fresh uniforms."""
    var k = 0
    while k + 1 < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        var z0 = Scalar[DT](r * fcos(theta))
        var z1 = Scalar[DT](r * fcos(theta + 0.5 * pi))
        ptr[k]     = _fnoise(z0)
        ptr[k + 1] = _fnoise(z1)
        k += 2
    if k < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        var z0 = Scalar[DT](r * fcos(theta))
        ptr[k] = _fnoise(z0)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ──────────────────────────────────────────────────────────────────────


def _apply_f_noise_kernel[N: Int](
    noise: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """In-place `noise[k] = sign(noise[k]) · sqrt(|noise[k]|)`."""
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](noise[idx])
        if x >= Scalar[DT](0.0):
            noise[idx] = fsqrt(x)
        else:
            noise[idx] = -fsqrt(-x)


def _scale_inplace_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    s: Scalar[DT],
):
    """In-place `buf[k] *= s`. Used to scale the output noise vector by
    `_noise_scale` (0.0 → deterministic mean weights for eval)."""
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = rebind[Scalar[DT]](buf[idx]) * s


def _materialize_w_eff_kernel[IN: Int, OUT: Int](
    mu_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    sigma_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    n_in: LayoutTensor[DT, Layout.row_major(IN), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    w_eff: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
):
    """`w_eff[i, j] = mu_w[i, j] + sigma_w[i, j] · n_in[i] · n_out[j]`."""
    var idx = Int(global_idx.x)
    var total = IN * OUT
    if idx < total:
        var i = idx // OUT
        var j = idx % OUT
        w_eff[i, j] = (
            rebind[Scalar[DT]](mu_w[i, j])
            + rebind[Scalar[DT]](sigma_w[i, j])
            * rebind[Scalar[DT]](n_in[i])
            * rebind[Scalar[DT]](n_out[j])
        )


def _materialize_b_eff_kernel[OUT: Int](
    mu_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    sigma_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    b_eff: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """`b_eff[j] = mu_b[j] + sigma_b[j] · n_out[j]`."""
    var j = Int(global_idx.x)
    if j < OUT:
        b_eff[j] = (
            rebind[Scalar[DT]](mu_b[j])
            + rebind[Scalar[DT]](sigma_b[j]) * rebind[Scalar[DT]](n_out[j])
        )


def _noisy_bias_add_kernel[BATCH: Int, OUT: Int](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    b_eff: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """`output[b, j] += b_eff[j]`."""
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[DT]](output[b, j]) + rebind[
            Scalar[DT]
        ](b_eff[j])


def _grad_b_pair_reduce_kernel[BATCH: Int, OUT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    grad_mu_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    grad_sigma_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """`s = Σ_b grad_out[b, col]`; `grad_mu_b[col] += s`,
    `grad_sigma_b[col] += s · n_out[col]`. ONE BLOCK per output column +
    `block.sum` reduction → full occupancy (vs the old one-thread-per-column
    serial-BATCH-loop). Both accumulate (Adam zero_grad clears at step start).
    Launch: grid_dim=OUT, block_dim=TPB."""
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= OUT:
        return
    var my_s: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_s += rebind[Scalar[DT]](grad_output[bi, col])
        bi += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my_s)
    if t == 0:
        var s = total[0]
        grad_mu_b[col] = rebind[Scalar[DT]](grad_mu_b[col]) + s
        grad_sigma_b[col] = (
            rebind[Scalar[DT]](grad_sigma_b[col])
            + s * rebind[Scalar[DT]](n_out[col])
        )


def _scaled_accum_factorized_kernel[IN: Int, OUT: Int](
    dst: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    n_in: LayoutTensor[DT, Layout.row_major(IN), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """`dst[i,j] += src[i,j] · n_in[i] · n_out[j]` (factorized-noise scale).
    Accumulates grad_sigma_w from the shared dW = cacheᵀ @ grad_output (which
    is grad_mu_w). One thread per (i, j) — full occupancy."""
    var idx = Int(global_idx.x)
    if idx < IN * OUT:
        var i = idx // OUT
        var j = idx % OUT
        dst[i, j] = (
            rebind[Scalar[DT]](dst[i, j])
            + rebind[Scalar[DT]](src[i, j])
            * rebind[Scalar[DT]](n_in[i])
            * rebind[Scalar[DT]](n_out[j])
        )


struct NoisyLinear[IN: Int, OUT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT

    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT
    # Fortunato §3.2 factorized: σ_W = σ_b = σ_0 / √IN, σ_0 = 0.5.
    comptime SIGMA0 = Scalar[DT](0.5)

    var mu_w:     Param["mu_w",     True,  Self.W_SIZE]
    var sigma_w:  Param["sigma_w",  True,  Self.W_SIZE]
    var mu_b:     Param["mu_b",     False, Self.B_SIZE]
    var sigma_b:  Param["sigma_b",  False, Self.B_SIZE]

    var _noise_in:  Scratch["noise_in",  Self.IN]
    var _noise_out: Scratch["noise_out", Self.OUT]
    var _w_eff:     Scratch["w_eff",     Self.W_SIZE]
    var _b_eff:     Scratch["b_eff",     Self.B_SIZE]

    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    # GPU Philox bookkeeping (unused on CPU path). `_noise_seed` is set
    # at make() to a unique value per leaf. Slice 5: the per-forward
    # Philox offset is device-resident (`_noise_offset_dev`, 1-elem
    # uint64) and advanced by `advance_rng_offset_kernel` so the forward
    # is CUDA-graph capturable — was a host `_noise_offset += ...`.
    # GPU-only runtime state; not serialized (None on CPU).
    var _noise_seed: UInt64
    var _noise_offset_dev: Optional[DeviceBuffer[DType.uint64]]

    # Noise magnitude multiplier (host scalar). 1.0 = normal factorized
    # noise (training / noisy exploration); 0.0 = mean weights only
    # (deterministic greedy — used for eval, since for ε=0 Noisy nets the
    # acting policy is *already* the noisy argmax, so a meaningful eval must
    # turn the noise off). Toggled via `set_attr["noise_scale"]`, which
    # `Sequential` broadcasts to every child. Scaling the OUTPUT noise vector
    # alone scales BOTH W-noise (σ·f(εᵢₙ)·f(εₒᵤₜ)) and b-noise (σ·f(εₒᵤₜ))
    # linearly. `×1.0` is exact in IEEE float, so the default path is
    # bit-identical to before.
    var _noise_scale: Scalar[DT]

    # Backward grad_w temporaries (GPU) — see Linear. cacheᵀ[IN, BATCH] (lazy,
    # BATCH-sized) + dW_tmp[IN, OUT] (W_SIZE, fixed). dW = cacheᵀ @ grad_output
    # (one tensor-core max_matmul) is grad_mu_w; grad_sigma_w is the same dW
    # scaled by the factorized noise. Replaces the naive serial pair kernel.
    var cacheT_dev: Optional[DeviceBuffer[DT]]
    var cacheT_n: Int
    var dW_tmp_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.mu_w    = Param["mu_w",    True,  Self.W_SIZE]()
        self.sigma_w = Param["sigma_w", True,  Self.W_SIZE]()
        self.mu_b    = Param["mu_b",    False, Self.B_SIZE]()
        self.sigma_b = Param["sigma_b", False, Self.B_SIZE]()
        self._noise_in  = Scratch["noise_in",  Self.IN]()
        self._noise_out = Scratch["noise_out", Self.OUT]()
        self._w_eff     = Scratch["w_eff",     Self.W_SIZE]()
        self._b_eff     = Scratch["b_eff",     Self.B_SIZE]()
        self._cached_input_ptr = None
        self._noise_seed = UInt64(0)
        self._noise_offset_dev = None
        self._noise_scale = Scalar[DT](1.0)
        self.cacheT_dev = None
        self.cacheT_n = 0
        self.dW_tmp_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified factory.

        Init scheme (matches the legacy `nn/model/noisy_linear.mojo`):
          µ_W ← INIT.init_weight(...)         — Xavier/Kaiming/Lecun
          µ_b ← 0
          σ_W ← σ_0 / √IN                     — Fortunato §3.2 factorized
          σ_b ← σ_0 / √IN

        Empirically: using the framework-standard INIT for µ_W (vs
        Fortunato's strict U(-1/√p, 1/√p)) lets the signal magnitude
        dominate σ-noise at initialization, which is necessary for
        small networks (e.g. CartPole 64-unit MLPs) to bootstrap. On
        deep networks like Atari CNNs both inits work.

        On GPU the host-side INIT is run into a host buffer then uploaded —
        same pattern as `Linear.make[target='gpu']`. The Philox seed is
        derived from `random_float64()` at make-time so two NoisyLinears
        in the same Q-net get distinct streams; the trainer / test can
        override via `set_noise_seed` for deterministic runs.
        """
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "NoisyLinear: target must be 'cpu' or 'gpu'"
        var nl = Self()

        # σ_W = σ_b = σ_0 / √IN (Fortunato §3.2 factorized).
        var sigma_init = Self.SIGMA0 / Scalar[DT](
            fsqrt(Float64(Self.IN))
        )

        comptime if target == "cpu":
            nl.mu_w    = Param["mu_w",    True,  Self.W_SIZE].make_cpu()
            nl.sigma_w = Param["sigma_w", True,  Self.W_SIZE].make_cpu()
            nl.mu_b    = Param["mu_b",    False, Self.B_SIZE].make_cpu()
            nl.sigma_b = Param["sigma_b", False, Self.B_SIZE].make_cpu()
            INIT.init_weight(
                nl.mu_w.value_unsafe_ptr_cpu(),
                Self.W_SIZE, Self.IN, Self.OUT,
            )
            INIT.init_bias(nl.mu_b.value_unsafe_ptr_cpu(), Self.B_SIZE)
            var sg_w_p = nl.sigma_w.value_unsafe_ptr_cpu()
            for k in range(Self.W_SIZE):
                sg_w_p[k] = sigma_init
            var sg_b_p = nl.sigma_b.value_unsafe_ptr_cpu()
            for k in range(Self.B_SIZE):
                sg_b_p[k] = sigma_init
            nl.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("NoisyLinear.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            nl.mu_w    = Param["mu_w",    True,  Self.W_SIZE].make_gpu(ctx_v)
            nl.sigma_w = Param["sigma_w", True,  Self.W_SIZE].make_gpu(ctx_v)
            nl.mu_b    = Param["mu_b",    False, Self.B_SIZE].make_gpu(ctx_v)
            nl.sigma_b = Param["sigma_b", False, Self.B_SIZE].make_gpu(ctx_v)
            # Init on host, then upload — mirrors Linear.make[gpu].
            var muw_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var sgw_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var mub_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            var sgb_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(
                muw_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT,
            )
            INIT.init_bias(mub_host.unsafe_ptr(), Self.B_SIZE)
            for k in range(Self.W_SIZE):
                sgw_host.unsafe_ptr()[k] = sigma_init
            for k in range(Self.B_SIZE):
                sgb_host.unsafe_ptr()[k] = sigma_init
            ctx_v.enqueue_copy(nl.mu_w.value_dev.value(),    muw_host)
            ctx_v.enqueue_copy(nl.sigma_w.value_dev.value(), sgw_host)
            ctx_v.enqueue_copy(nl.mu_b.value_dev.value(),    mub_host)
            ctx_v.enqueue_copy(nl.sigma_b.value_dev.value(), sgb_host)
            ctx_v.synchronize()
            nl.ts = TargetStorage.make_gpu(ctx_v)

        # Seed Philox stream — keeps GPU forwards reproducible per seed()
        # while distinct between layers (random_float64 advances the
        # global RNG state used by other CPU draws too).
        nl._noise_seed = UInt64(random_float64() * Float64(1 << 31))
        comptime if target == "gpu":
            var noff = ctx.value().enqueue_create_buffer[DType.uint64](1)
            noff.enqueue_fill(UInt64(0))
            nl._noise_offset_dev = noff^
            # Fixed [IN, OUT] dW scratch for the max_matmul grad_w path; cacheT
            # stays None (lazily sized to BATCH on first backward).
            nl.dW_tmp_dev = ctx.value().enqueue_create_buffer[DT](Self.W_SIZE)

        init_scratch_auto[Self, target=target](nl, ctx)
        return nl^

    def set_noise_seed(mut self, seed: UInt64) raises:
        """Deterministic-test hook. Resets the device offset to 0 too."""
        self._noise_seed = seed
        if self._noise_offset_dev:
            self._noise_offset_dev.value().enqueue_fill(UInt64(0))

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Catch the `noise_scale` broadcast (from `Sequential.set_attr`):
        1.0 = normal noisy exploration, 0.0 = deterministic mean weights
        (greedy eval). All other attrs are ignored (no-op), matching the
        `Module` default for param-bearing leaves."""
        comptime if ATTR == "noise_scale":
            self._noise_scale = value

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "NoisyLinear: target must be 'cpu' or 'gpu'"
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT](output)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            # 1. Sample fresh factorized noise.
            var ni_p = self._noise_in.cpu_ptr()
            var no_p = self._noise_out.cpu_ptr()
            _sample_factorized_noise(ni_p, Self.IN)
            _sample_factorized_noise(no_p, Self.OUT)

            # Scale the OUTPUT noise by `_noise_scale` (1.0 normal; 0.0 →
            # deterministic mean weights for eval). Scaling n_out alone
            # scales both W-noise and b-noise linearly. `×1.0` is exact.
            for j in range(Self.OUT):
                no_p[j] = no_p[j] * self._noise_scale

            # 2. Materialize W_eff and b_eff.
            var mu_w_p = self.mu_w.value_unsafe_ptr_cpu()
            var sg_w_p = self.sigma_w.value_unsafe_ptr_cpu()
            var mu_b_p = self.mu_b.value_unsafe_ptr_cpu()
            var sg_b_p = self.sigma_b.value_unsafe_ptr_cpu()
            var w_eff_p = self._w_eff.cpu_ptr()
            var b_eff_p = self._b_eff.cpu_ptr()
            for i in range(Self.IN):
                var ni = ni_p[i]
                for j in range(Self.OUT):
                    var idx = i * Self.OUT + j
                    w_eff_p[idx] = mu_w_p[idx] + sg_w_p[idx] * ni * no_p[j]
            for j in range(Self.OUT):
                b_eff_p[j] = mu_b_p[j] + sg_b_p[j] * no_p[j]

            # 3. output = x @ W_eff + b_eff. The matmul runs through BLAS
            #    (Apple Accelerate) like Linear's CPU path — previously a
            #    naive BATCH·IN·OUT scalar loop, which was THE Rainbow CPU
            #    bottleneck (SAC is fast on CPU because Linear was already
            #    ported; NoisyLinear had been missed).
            var w_eff_tt = TileTensor(w_eff_p, row_major[Self.IN, Self.OUT]())
            max_matmul[target="cpu"](output_v, input_v, w_eff_tt, None)
            for b in range(BATCH):
                for j in range(Self.OUT):
                    out_p[b * Self.OUT + j] = (
                        out_p[b * Self.OUT + j] + b_eff_p[j]
                    )
        else:
            var ctx = self.ts.ctx.value()
            var ni_p = self._noise_in.dev_ptr()
            var no_p = self._noise_out.dev_ptr()
            var w_eff_p = self._w_eff.dev_ptr()
            var b_eff_p = self._b_eff.dev_ptr()

            # 1. Sample fresh N(0,1) noise via Philox/Box-Muller, reading
            #    the Philox offset from the device buffer and advancing it
            #    on-device (CUDA-graph capturable) — same offset sequence
            #    the host `_noise_offset += ...` produced. Then apply
            #    f(x) = sign(x)·sqrt(|x|) in place.
            var off_lt = LayoutTensor[
                DType.uint64, Layout.row_major(1), MutAnyOrigin,
            ](self._noise_offset_dev.value().unsafe_ptr())
            box_muller_normal_gpu_dev[Self.IN](
                ctx, ni_p, self._noise_seed, off_lt,
            )
            comptime adv_in = advance_rng_offset_kernel[
                ((Self.IN + 1) // 2) * 2
            ]
            ctx.enqueue_function[adv_in](off_lt, grid_dim=1, block_dim=1)
            box_muller_normal_gpu_dev[Self.OUT](
                ctx, no_p, self._noise_seed, off_lt,
            )
            comptime adv_out = advance_rng_offset_kernel[
                ((Self.OUT + 1) // 2) * 2
            ]
            ctx.enqueue_function[adv_out](off_lt, grid_dim=1, block_dim=1)
            var ni_lt = LayoutTensor[
                DT, Layout.row_major(Self.IN), MutAnyOrigin,
            ](ni_p)
            var no_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT), MutAnyOrigin,
            ](no_p)
            comptime n_blocks_in = (Self.IN + TPB - 1) // TPB
            comptime apply_in = _apply_f_noise_kernel[Self.IN]
            ctx.enqueue_function[apply_in](
                ni_lt, grid_dim=n_blocks_in, block_dim=TPB,
            )
            comptime n_blocks_out = (Self.OUT + TPB - 1) // TPB
            comptime apply_out = _apply_f_noise_kernel[Self.OUT]
            ctx.enqueue_function[apply_out](
                no_lt, grid_dim=n_blocks_out, block_dim=TPB,
            )

            # Scale the OUTPUT noise by `_noise_scale` (1.0 normal; 0.0 →
            # deterministic mean weights for eval). Scaling n_out alone
            # scales both W-noise and b-noise linearly. `×1.0` is exact.
            comptime scale_out = _scale_inplace_kernel[Self.OUT]
            ctx.enqueue_function[scale_out](
                no_lt, self._noise_scale,
                grid_dim=n_blocks_out, block_dim=TPB,
            )

            # 2. Materialize W_eff and b_eff on device.
            var muw_lt = LayoutTensor[
                DT, Layout.row_major(Self.IN, Self.OUT), MutAnyOrigin,
            ](self.mu_w.value_dev.value().unsafe_ptr())
            var sgw_lt = LayoutTensor[
                DT, Layout.row_major(Self.IN, Self.OUT), MutAnyOrigin,
            ](self.sigma_w.value_dev.value().unsafe_ptr())
            var mub_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT), MutAnyOrigin,
            ](self.mu_b.value_dev.value().unsafe_ptr())
            var sgb_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT), MutAnyOrigin,
            ](self.sigma_b.value_dev.value().unsafe_ptr())
            var we_lt = LayoutTensor[
                DT, Layout.row_major(Self.IN, Self.OUT), MutAnyOrigin,
            ](w_eff_p)
            var be_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT), MutAnyOrigin,
            ](b_eff_p)
            comptime n_blocks_w = (Self.W_SIZE + TPB - 1) // TPB
            comptime w_kernel = _materialize_w_eff_kernel[Self.IN, Self.OUT]
            ctx.enqueue_function[w_kernel](
                muw_lt, sgw_lt, ni_lt, no_lt, we_lt,
                grid_dim=n_blocks_w, block_dim=TPB,
            )
            comptime n_blocks_be = (Self.OUT + TPB - 1) // TPB
            comptime be_kernel = _materialize_b_eff_kernel[Self.OUT]
            ctx.enqueue_function[be_kernel](
                mub_lt, sgb_lt, no_lt, be_lt,
                grid_dim=n_blocks_be, block_dim=TPB,
            )

            # 3. output = input @ W_eff.
            var w_eff_tt = TileTensor(
                self._w_eff.dev.value(),
                row_major[Self.IN, Self.OUT](),
            )
            max_matmul[target="gpu"](output_v, input_v, w_eff_tt, ctx)

            # 4. Bias add b_eff.
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT), MutAnyOrigin,
            ](out_p)
            comptime n_blocks_ba = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime ba_kernel = _noisy_bias_add_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[ba_kernel](
                out_lt, be_lt, grid_dim=n_blocks_ba, block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "NoisyLinear: target must be 'cpu' or 'gpu'"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        var grad_out_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_in_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_out_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in_v.ptr)

        # BACKWARD-ORDER INVARIANT: `_cached_input_ptr` aliases the
        # orchestrator's input slab — and Sequential reuses the SAME
        # `mid_cpu[N-2]` slab as the grad_input destination for this
        # leaf. So we MUST read `x_p` (param-grad accumulation) before
        # writing `gi_p` (grad_x). Mirrors Linear's invariant at
        # `primitives/linear.mojo:390-397`. Holds for both CPU and GPU.

        comptime if target == "cpu":
            var ni_p = self._noise_in.cpu_ptr()
            var no_p = self._noise_out.cpu_ptr()
            var w_eff_p = self._w_eff.cpu_ptr()
            var x_p = self._cached_input_ptr.value()

            # 1. Param grads (mode="all" only) — reads x_p before grad_x writes.
            #    grad_mu_b[j]    = Σ_b grad_out[b, j]
            #    grad_sigma_b[j] = grad_mu_b[j] * f(ε_out[j])
            #    grad_mu_w[i,j]  = Σ_b x[b, i] * grad_out[b, j]
            #    grad_sigma_w[i,j] = grad_mu_w[i,j] * f(ε_in[i]) * f(ε_out[j])
            comptime if mode == "all":
                var g_mu_w = self.mu_w.grad_unsafe_ptr_cpu()
                var g_sg_w = self.sigma_w.grad_unsafe_ptr_cpu()
                var g_mu_b = self.mu_b.grad_unsafe_ptr_cpu()
                var g_sg_b = self.sigma_b.grad_unsafe_ptr_cpu()
                # grad_b — cheap O(BATCH·OUT) reduction; keep scalar.
                for j in range(Self.OUT):
                    var sb: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        sb = sb + go_p[b * Self.OUT + j]
                    g_mu_b[j] = g_mu_b[j] + sb
                    g_sg_b[j] = g_sg_b[j] + sb * no_p[j]
                # grad_w: dW = xᵀ @ grad_output via BLAS (Apple Accelerate),
                # mirroring Linear's CPU backward — was a naive BATCH·IN·OUT
                # scalar loop. Transpose x into cT FIRST: that consumes the
                # read of `x_p` before grad_x clobbers the aliased input slab
                # (the leaf backward-order invariant). grad_mu_w += dW,
                # grad_sigma_w += dW · f(ε_in[i]) · f(ε_out[j]).
                var cT_buf = alloc[Scalar[DT]](BATCH * Self.IN)
                var dW_buf = alloc[Scalar[DT]](Self.IN * Self.OUT)
                for b in range(BATCH):
                    for i in range(Self.IN):
                        cT_buf[i * BATCH + b] = x_p[b * Self.IN + i]
                var cT_tt = TileTensor(cT_buf, row_major[Self.IN, BATCH]())
                var dW_tt = TileTensor(
                    dW_buf, row_major[Self.IN, Self.OUT]()
                )
                max_matmul[target="cpu"](dW_tt, cT_tt, grad_out_v, None)
                for i in range(Self.IN):
                    var ni = ni_p[i]
                    for j in range(Self.OUT):
                        var idx = i * Self.OUT + j
                        var dw = dW_buf[idx]
                        g_mu_w[idx] = g_mu_w[idx] + dw
                        g_sg_w[idx] = g_sg_w[idx] + dw * ni * no_p[j]
                dW_buf.free()
                cT_buf.free()

            # 2. grad_x = grad_output @ W_effᵀ via BLAS (was a naive
            #    BATCH·IN·OUT scalar loop). After step 1's read of `x_p`;
            #    this write clobbers the input slab `x_p` aliases.
            var w_eff_tt = TileTensor(w_eff_p, row_major[Self.IN, Self.OUT]())
            max_matmul[transpose_b=True, target="cpu"](
                grad_in_v, grad_out_v, w_eff_tt, None
            )
        else:
            var ctx = self.ts.ctx.value()
            var ni_p = self._noise_in.dev_ptr()
            var no_p = self._noise_out.dev_ptr()
            var ni_lt = LayoutTensor[
                DT, Layout.row_major(Self.IN), MutAnyOrigin,
            ](ni_p)
            var no_lt = LayoutTensor[
                DT, Layout.row_major(Self.OUT), MutAnyOrigin,
            ](no_p)

            # 1. Param grads (mode="all" only) — fused pairs (mu/sigma).
            comptime if mode == "all":
                var go_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.OUT), MutAnyOrigin,
                ](go_p)
                var g_mu_b_lt = LayoutTensor[
                    DT, Layout.row_major(Self.OUT), MutAnyOrigin,
                ](self.mu_b.grad_dev.value().unsafe_ptr())
                var g_sg_b_lt = LayoutTensor[
                    DT, Layout.row_major(Self.OUT), MutAnyOrigin,
                ](self.sigma_b.grad_dev.value().unsafe_ptr())
                # grad_b pair — block-per-column reduction (full occupancy).
                comptime gb_kernel = _grad_b_pair_reduce_kernel[
                    BATCH, Self.OUT
                ]
                ctx.enqueue_function[gb_kernel](
                    go_lt, no_lt, g_mu_b_lt, g_sg_b_lt,
                    grid_dim=Self.OUT, block_dim=TPB,
                )

                # grad_w pair via transpose + max_matmul (tensor cores):
                #   dW = cacheᵀ @ grad_output     → grad_mu_w increment
                #   grad_mu_w    += dW
                #   grad_sigma_w += dW · n_in[i] · n_out[j]
                # Replaces the naive serial per-(i,j) pair kernel.
                ensure_gpu_buffer(
                    self.cacheT_dev, self.cacheT_n, BATCH * Self.IN, ctx,
                )
                var cache_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.IN), MutAnyOrigin,
                ](self._cached_input_ptr.value())
                var cacheT_lt = LayoutTensor[
                    DT, Layout.row_major(Self.IN, BATCH), MutAnyOrigin,
                ](self.cacheT_dev.value())
                comptime n_blocks_t = (BATCH * Self.IN + TPB - 1) // TPB
                comptime t_kernel = _transpose_kernel[BATCH, Self.IN]
                ctx.enqueue_function[t_kernel](
                    cache_lt, cacheT_lt,
                    grid_dim=n_blocks_t, block_dim=TPB,
                )
                var cacheT_tt = TileTensor(
                    self.cacheT_dev.value(), row_major[Self.IN, BATCH](),
                )
                var dW_tmp_tt = TileTensor(
                    self.dW_tmp_dev.value(), row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="gpu"](
                    dW_tmp_tt, cacheT_tt, grad_out_v, ctx,
                )
                # grad_mu_w += dW_tmp  (flat accumulate)
                comptime gw_flat = Layout.row_major(Self.W_SIZE)
                var g_mu_w_flat = LayoutTensor[DT, gw_flat, MutAnyOrigin](
                    self.mu_w.grad_dev.value().unsafe_ptr()
                )
                var dW_tmp_flat = LayoutTensor[DT, gw_flat, MutAnyOrigin](
                    self.dW_tmp_dev.value()
                )
                comptime n_blocks_acc = (Self.W_SIZE + TPB - 1) // TPB
                comptime acc_kernel = _accum_kernel[Self.W_SIZE]
                ctx.enqueue_function[acc_kernel](
                    g_mu_w_flat, dW_tmp_flat,
                    grid_dim=n_blocks_acc, block_dim=TPB,
                )
                # grad_sigma_w += dW_tmp · n_in[i] · n_out[j]
                var g_sg_w_lt = LayoutTensor[
                    DT, Layout.row_major(Self.IN, Self.OUT), MutAnyOrigin,
                ](self.sigma_w.grad_dev.value().unsafe_ptr())
                var dW_tmp_2d = LayoutTensor[
                    DT, Layout.row_major(Self.IN, Self.OUT), MutAnyOrigin,
                ](self.dW_tmp_dev.value())
                comptime sf_kernel = _scaled_accum_factorized_kernel[
                    Self.IN, Self.OUT
                ]
                ctx.enqueue_function[sf_kernel](
                    g_sg_w_lt, dW_tmp_2d, ni_lt, no_lt,
                    grid_dim=n_blocks_acc, block_dim=TPB,
                )

            # 2. grad_x = grad_output @ W_eff^T (may alias cache).
            var w_eff_tt = TileTensor(
                self._w_eff.dev.value(),
                row_major[Self.IN, Self.OUT](),
            )
            max_matmul[transpose_b=True, target="gpu"](
                grad_in_v, grad_out_v, w_eff_tt, ctx,
            )

    # ----- Param / grad walkers (reflection-derived) ----------------------
    # CRITICAL: without these overrides, NoisyLinear inherits the default
    # `zero_grad` no-op from the Module trait — and the trainer's opt.step
    # ends up applying *accumulated* gradients (μ_W grad sums across train
    # steps), poisoning Adam updates. Mirrors `Linear.for_each_param` /
    # `Linear.zero_grad` at primitives/linear.mojo:595-607.

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
