"""RSample[ACT] — reparameterized squashed-Gaussian sample as a Module.

Phase 8.4. Wraps the squashed-Gaussian Reparameterization trick (currently
in `nn2.loss.sac_actor_loss.squashed_gaussian_sample` + `sac_actor_backward`)
behind the `Module` trait so loss graphs can be composed.

Topology:
    input  [BATCH, 2*ACT]   packed [mu | log_std]
    output [BATCH, ACT+1]   packed [action | log_prob]

    action[b, j]   = action_scale · tanh(mu_j + exp(clamp(log_std_j, -5, 2)) · z_j)
    log_prob[b]    = Σ_j [ -0.5·z_j² - log_std_j - 0.5·log(2π)
                           - log(action_scale·(1 - tanh²) + ε) ]

`z` (BATCH × ACT) is drawn fresh on every forward via the nn2 Box-Muller
helper (caller-side `std.random.seed`). `z`, the raw input, and the
log_std clamp mask are cached for backward.

Backward consumes the full `grad_output [BATCH, ACT+1]` packed
`[grad_action | grad_log_prob]`:
    grad_input[b, j]      = grad_action[b, j] · ∂a_j/∂mu_j
                          + grad_log_prob[b]  · ∂log_prob/∂mu_j
    grad_input[b, ACT+j]  = grad_action[b, j] · ∂a_j/∂log_std_j
                          + grad_log_prob[b]  · ∂log_prob/∂log_std_j
                          (masked to zero on clamp boundaries)

`action_scale` is a public runtime field on the struct (default 1.0).
Set it after `make` (e.g. `rs.action_scale = 2.0` for Pendulum).

Phase 8.4 ships CPU only. GPU `make`/`forward`/`backward` paths raise.
The first GPU SAC env will pull the kernels through.

Free-function form parity. With identical inputs / weights / z, the
composed-form pipeline using RSample produces actor gradients equivalent
to the (free-function) `squashed_gaussian_sample` + `sac_actor_backward`
pair — see `tests/nn2/test_rsample.mojo` for the bit-level check.
"""

from std.math import exp, log, tanh
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout

from ..constants import DT
from ..core import (
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)
from ..random.box_muller import box_muller_normal


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime EPS_TANH_CORR: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN:
        return LOG_STD_MIN
    elif ls > LOG_STD_MAX:
        return LOG_STD_MAX
    return ls


struct RSample[ACT: Int](Module):
    comptime IN_DIM = 2 * Self.ACT
    comptime OUT_DIM = Self.ACT + 1

    # Public knobs.
    var action_scale: Scalar[DT]

    # Backward caches (CPU). z_cache mirrors the noise drawn on forward.
    # in_cache mirrors the raw input — we re-read mu/log_std on backward
    # since the caller may overwrite the input tensor before backward.
    var z_cache: List[Scalar[DT]]
    var in_cache: List[Scalar[DT]]
    var cache_n_batch: Int

    # GPU placeholders (Phase 8.4 ships CPU only).
    var z_cache_dev: Optional[DeviceBuffer[DT]]
    var in_cache_dev: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.action_scale = Scalar[DT](1.0)
        self.z_cache = List[Scalar[DT]]()
        self.in_cache = List[Scalar[DT]]()
        self.cache_n_batch = 0
        self.z_cache_dev = None
        self.in_cache_dev = None
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. INIT ignored (RSample is parameterless)."""
        comptime assert (
            target == "cpu"
        ), "RSample.make[target='gpu', INIT] requires a DeviceContext"
        var r = Self()
        r._target_tag = TARGET_CPU
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "RSample.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        # Phase 8.4: GPU not yet wired. Stamp the tag so trait dispatch
        # works, but every method raises until kernels land.
        var r = Self()
        r.ctx = ctx
        r._target_tag = TARGET_GPU
        return r^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "RSample: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        if self.cache_n_batch < batch:
            self.z_cache.resize(batch * Self.ACT, Scalar[DT](0.0))
            self.in_cache.resize(batch * (2 * Self.ACT), Scalar[DT](0.0))
            self.cache_n_batch = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        """Forward = sample fresh z, then squashed-Gaussian forward.

        Input  [BATCH, 2*ACT]   packed [mu | log_std]
        Output [BATCH, ACT+1]   packed [action | log_prob]

        Output[b, j]     for j in [0, ACT)      = action[b, j]
        Output[b, ACT]                          = log_prob[b]
        """
        comptime assert input.flat_rank == 2, "input must be rank-2 [BATCH, 2*ACT]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, ACT+1]"
        comptime assert Self.ACT >= 1, "RSample[ACT]: ACT >= 1"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            # Cache raw input + draw fresh z.
            var in_p = self.in_cache.unsafe_ptr()
            var z_p = self.z_cache.unsafe_ptr()
            for b in range(BATCH):
                for j in range(2 * Self.ACT):
                    in_p[b * (2 * Self.ACT) + j] = input[b, j]
            box_muller_normal(z_p, BATCH * Self.ACT)

            for b in range(BATCH):
                var lp_total: Scalar[DT] = 0.0
                for j in range(Self.ACT):
                    var mu = input[b, j]
                    var ls = _clamp_log_std(input[b, Self.ACT + j])
                    var std = exp(ls)
                    var zj = z_p[b * Self.ACT + j]
                    var pre = mu + std * zj
                    var y = tanh(pre)
                    output[b, j] = self.action_scale * y
                    var one_minus_y2 = Scalar[DT](1.0) - y * y
                    var corr = self.action_scale * one_minus_y2 + EPS_TANH_CORR
                    lp_total += (
                        Scalar[DT](-0.5) * zj * zj
                        - ls
                        - Scalar[DT](0.5) * LOG_2PI
                        - log(corr)
                    )
                output[b, Self.ACT] = lp_total
        else:
            raise Error("RSample[ACT]: GPU path not yet implemented (Phase 8.4 CPU only)")

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        """Backward — split grad_output into [grad_action | grad_log_prob]
        and route through the squashed-Gaussian analytical Jacobian.

        grad_output [BATCH, ACT+1]   packed [grad_action | grad_log_prob]
        grad_input  [BATCH, 2*ACT]   packed [grad_mu | grad_log_std]
        """
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var in_p = self.in_cache.unsafe_ptr()
            var z_p = self.z_cache.unsafe_ptr()
            for b in range(BATCH):
                var grad_lp = grad_output[b, Self.ACT]
                for j in range(Self.ACT):
                    var mu = in_p[b * (2 * Self.ACT) + j]
                    var ls_raw = in_p[b * (2 * Self.ACT) + Self.ACT + j]
                    var ls = _clamp_log_std(ls_raw)
                    var ls_clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)
                    var std = exp(ls)
                    var zj = z_p[b * Self.ACT + j]
                    var pre = mu + std * zj
                    var y = tanh(pre)
                    var one_minus_y2 = Scalar[DT](1.0) - y * y
                    var c_corr = self.action_scale * one_minus_y2
                    var corr = c_corr + EPS_TANH_CORR

                    var da_dmu = self.action_scale * one_minus_y2
                    var da_dls = self.action_scale * one_minus_y2 * zj * std
                    var dlp_dmu = (Scalar[DT](2.0) * y * c_corr) / corr
                    var dlp_dls = (
                        Scalar[DT](-1.0)
                        + (Scalar[DT](2.0) * y * c_corr * zj * std) / corr
                    )

                    var ga = grad_output[b, j]
                    var gmu = ga * da_dmu + grad_lp * dlp_dmu
                    var gls = ga * da_dls + grad_lp * dlp_dls

                    grad_input[b, j] = gmu
                    if ls_clamped:
                        grad_input[b, Self.ACT + j] = Scalar[DT](0.0)
                    else:
                        grad_input[b, Self.ACT + j] = gls
        else:
            raise Error("RSample[ACT]: GPU backward not yet implemented (Phase 8.4 CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        # No parameters — backward_input ≡ backward.
        self.backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        # No parameters.
        pass

    def set_inference(mut self, value: Bool):
        # RSample always samples fresh z — `_inference` is stored for
        # trait conformance but has no behavioral effect. A future
        # variant could route z to a deterministic value (mu) when
        # inference=True, but Phase 8.4 keeps it as-is.
        self._inference = value
