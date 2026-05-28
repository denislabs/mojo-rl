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

CPU-only initial port. GPU path raises a comptime assert. Suitable
for NoisyDQN smoke + 30k CartPole convergence; GPU smoke deferred.

Use in Noisy DQN: replace the last `Linear[H, NA]` in the Q-net with
`NoisyLinear[H, NA]`. Drop ε-greedy at the trainer level (set
`epsilon=0`, `epsilon_min=0`).
"""

from std.math import sqrt as fsqrt, log as flog, cos as fcos, pi
from std.random import random_float64
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import LayoutTensor, TileTensor, row_major

from ..constants import DT
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
from ..core.target_storage import TargetStorage, assert_tag_for


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

    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
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
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ](unsafe_from_address=0)
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

        CPU-only initial port; GPU path will land in a follow-up commit.
        """
        comptime assert (
            target == "cpu"
        ), "NoisyLinear: GPU target not yet supported (CPU-only port)."
        var nl = Self()
        nl.mu_w    = Param["mu_w",    True,  Self.W_SIZE].make_cpu()
        nl.sigma_w = Param["sigma_w", True,  Self.W_SIZE].make_cpu()
        nl.mu_b    = Param["mu_b",    False, Self.B_SIZE].make_cpu()
        nl.sigma_b = Param["sigma_b", False, Self.B_SIZE].make_cpu()

        # µ_W ← INIT (typically Xavier in DQN); µ_b ← 0.
        INIT.init_weight(
            nl.mu_w.value_unsafe_ptr_cpu(),
            Self.W_SIZE, Self.IN, Self.OUT,
        )
        INIT.init_bias(nl.mu_b.value_unsafe_ptr_cpu(), Self.B_SIZE)

        # σ_W = σ_b = σ_0 / √IN (Fortunato §3.2 factorized).
        var sigma_init = Self.SIGMA0 / Scalar[DT](
            fsqrt(Float64(Self.IN))
        )
        var sg_w_p = nl.sigma_w.value_unsafe_ptr_cpu()
        for k in range(Self.W_SIZE):
            sg_w_p[k] = sigma_init
        var sg_b_p = nl.sigma_b.value_unsafe_ptr_cpu()
        for k in range(Self.B_SIZE):
            sg_b_p[k] = sigma_init

        nl.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target=target](nl, ctx)
        return nl^

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
            target == "cpu"
        ), "NoisyLinear: GPU forward not yet supported."
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT](output)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        self._cached_input_ptr = in_p

        # 1. Sample fresh factorized noise.
        var ni_p = self._noise_in.cpu_ptr()
        var no_p = self._noise_out.cpu_ptr()
        _sample_factorized_noise(ni_p, Self.IN)
        _sample_factorized_noise(no_p, Self.OUT)

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

        # 3. Standard linear: output = x @ W_eff + b_eff.
        for b in range(BATCH):
            for j in range(Self.OUT):
                var s: Scalar[DT] = 0.0
                for i in range(Self.IN):
                    s = s + in_p[b * Self.IN + i] * w_eff_p[i * Self.OUT + j]
                out_p[b * Self.OUT + j] = s + b_eff_p[j]

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
            target == "cpu"
        ), "NoisyLinear: GPU vjp not yet supported."
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["NoisyLinear", target](self.ts.target_tag)
        var grad_out_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_in_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_out_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_in_v.ptr)

        var ni_p = self._noise_in.cpu_ptr()
        var no_p = self._noise_out.cpu_ptr()
        var w_eff_p = self._w_eff.cpu_ptr()
        var x_p = self._cached_input_ptr

        # BACKWARD-ORDER INVARIANT: `_cached_input_ptr` aliases the
        # orchestrator's input slab — and Sequential reuses the SAME
        # `mid_cpu[N-2]` slab as the grad_input destination for this
        # leaf. So we MUST read `x_p` (param-grad accumulation) before
        # writing `gi_p` (grad_x). Mirrors Linear's invariant at
        # `primitives/linear.mojo:390-397`.

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
            for j in range(Self.OUT):
                var sb: Scalar[DT] = 0.0
                for b in range(BATCH):
                    sb = sb + go_p[b * Self.OUT + j]
                g_mu_b[j] = g_mu_b[j] + sb
                g_sg_b[j] = g_sg_b[j] + sb * no_p[j]
            for i in range(Self.IN):
                var ni = ni_p[i]
                for j in range(Self.OUT):
                    var s: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        s = (
                            s
                            + x_p[b * Self.IN + i]
                            * go_p[b * Self.OUT + j]
                        )
                    var idx = i * Self.OUT + j
                    g_mu_w[idx] = g_mu_w[idx] + s
                    g_sg_w[idx] = g_sg_w[idx] + s * ni * no_p[j]

        # 2. grad_x = grad_output @ W_eff^T  (after step 1; clobbers the
        #    input slab `x_p` aliases).
        for b in range(BATCH):
            for i in range(Self.IN):
                var s: Scalar[DT] = 0.0
                for j in range(Self.OUT):
                    s = (
                        s
                        + go_p[b * Self.OUT + j]
                        * w_eff_p[i * Self.OUT + j]
                    )
                gi_p[b * Self.IN + i] = s

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
