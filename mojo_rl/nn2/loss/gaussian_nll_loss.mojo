"""GaussianNLLLoss[DIM] — diagonal Gaussian negative-log-likelihood.

Phase I.1.a. The probabilistic head used by MBPO's dynamics ensemble:
each member predicts `(µ, logvar)` per output dim, and the training
target is to maximise the log-likelihood of the observed `(reward, Δobs)`
under that diagonal-Gaussian distribution.

Conforms to nn2's `Loss` trait, so `Trainer[..., LOSS=GaussianNLLLoss[DIM]]`
slots in next to MSELoss / SoftCrossEntropyLoss.

Tensor conventions:

  - **logits**  shape `BATCH × (2*DIM)` — first `DIM` columns are
    means, next `DIM` columns are raw logvars (pre-clamp).
  - **targets** shape `BATCH × DIM` — observed values whose likelihood
    we maximise.
  - **grad_logits** shape `BATCH × (2*DIM)` — first `DIM` rows of
    output are `d_loss/d_µ`, next `DIM` are `d_loss/d_raw_logvar`.

Math (per-row, summed over DIM, then averaged over BATCH):

  σ²       = exp(clamp(raw_logvar, [LOGVAR_MIN, LOGVAR_MAX]))
  loss_row = Σᵢ ½·(yᵢ - µᵢ)² · σ⁻²ᵢ + ½·clamped_logvarᵢ
  loss     = (1/BATCH) · Σ_row loss_row

  d_loss/d_µ_i             = (µᵢ - yᵢ) · σ⁻²ᵢ / BATCH
  d_loss/d_raw_logvar_i    = [ ½ - ½·(yᵢ - µᵢ)²·σ⁻²ᵢ ] / BATCH   (in-clamp)
  d_loss/d_raw_logvar_i    = 0                                       (clamped)

Default logvar bounds `[-10, -2]` match MBPO reference
(`deep_agents/core/agents/mbpo_agent.mojo:182-183` — CPU fixed-bounds
path; the GPU production agent treats them as learnable parameters,
which is deferred to a future I.1.* phase).

Scope (I.1.a):
  - CPU forward + vjp validated against analytic + FD.
  - GPU paths stubbed to raise — no consumer yet. Trait dispatch is
    comptime so the stubs incur zero binary cost from CPU callers.
"""

from std.math import exp
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT, CPU_SIMD_W
from ..core import Loss, AMPPolicy, NoAMP
from ..core.target_storage import TargetStorage, assert_tag_for


struct GaussianNLLLoss[
    DIM: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Loss):
    """Diagonal Gaussian NLL with clamped logvar bounds.

    `OUT_DIM = 2*DIM` so consumers that pre-allocate grad_logits via
    the `Loss.OUT_DIM` trait member size the buffer correctly."""

    comptime OUT_DIM: Int = 2 * Self.DIM

    # Cached `(µ - y)` and clamped σ⁻² per element, both BATCH × DIM,
    # written on forward and consumed on vjp.  Avoids re-clamping +
    # re-multiplying on backward.
    var cache_diff: List[Scalar[DT]]      # (µ - y) per element
    var cache_inv_var: List[Scalar[DT]]   # exp(-clamped_logvar) per element
    var cache_in_clamp: List[Scalar[DT]]  # 1.0 if in clamp range else 0.0

    var ts: TargetStorage

    def __init__(out self):
        self.cache_diff = List[Scalar[DT]]()
        self.cache_inv_var = List[Scalar[DT]]()
        self.cache_in_clamp = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "GaussianNLLLoss.make[target='gpu'] requires a DeviceContext"
        )
        var loss = Self()
        loss.ts = TargetStorage.make_cpu()
        return loss^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "GaussianNLLLoss.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        # GPU path deferred — see module docstring.
        raise Error(
            "GaussianNLLLoss GPU not yet implemented (I.1.a is CPU-only)"
        )

    def _ensure_cpu(mut self, batch: Int):
        var need = batch * Self.DIM
        if len(self.cache_diff) < need:
            self.cache_diff.resize(need, Scalar[DT](0.0))
            self.cache_inv_var.resize(need, Scalar[DT](0.0))
            self.cache_in_clamp.resize(need, Scalar[DT](0.0))

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        logits: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        comptime assert logits.flat_rank == 2, "logits rank-2"
        comptime assert targets.flat_rank == 2, "targets rank-2"
        assert_tag_for["GaussianNLLLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_cpu(BATCH)
            var lp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](logits.ptr)
            var tp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](targets.ptr)
            var diff_p = self.cache_diff.unsafe_ptr()
            var ivar_p = self.cache_inv_var.unsafe_ptr()
            var clamp_p = self.cache_in_clamp.unsafe_ptr()
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            var total = Scalar[DT](0.0)
            for b in range(BATCH):
                # logits layout: [µ₀..µ_{DIM-1}, lv₀..lv_{DIM-1}]  per row.
                var lo = b * (2 * Self.DIM)
                var to = b * Self.DIM
                var co = b * Self.DIM
                for i in range(Self.DIM):
                    var mu = lp[lo + i]
                    var raw_lv = lp[lo + Self.DIM + i]
                    var y = tp[to + i]
                    var in_clamp = Scalar[DT](1.0)
                    var lv = raw_lv
                    if lv > lv_max:
                        lv = lv_max
                        in_clamp = Scalar[DT](0.0)
                    elif lv < lv_min:
                        lv = lv_min
                        in_clamp = Scalar[DT](0.0)
                    var inv_var = exp(-lv)
                    var d = mu - y
                    diff_p[co + i] = d
                    ivar_p[co + i] = inv_var
                    clamp_p[co + i] = in_clamp
                    total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](0.5) * lv
            return total / Scalar[DT](BATCH)
        else:
            raise Error(
                "GaussianNLLLoss.forward['gpu'] not implemented"
            )

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_logits: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert targets.flat_rank == 2, "targets rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits rank-2"
        assert_tag_for["GaussianNLLLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            var gp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_logits.ptr)
            var diff_p = self.cache_diff.unsafe_ptr()
            var ivar_p = self.cache_inv_var.unsafe_ptr()
            var clamp_p = self.cache_in_clamp.unsafe_ptr()
            var inv_b = Scalar[DT](1.0) / Scalar[DT](BATCH)
            for b in range(BATCH):
                var go = b * (2 * Self.DIM)
                var co = b * Self.DIM
                for i in range(Self.DIM):
                    var d = diff_p[co + i]            # (µ - y)
                    var inv_v = ivar_p[co + i]        # exp(-lv_clamped)
                    var ic = clamp_p[co + i]          # 1 in-clamp, 0 clamped
                    # d_loss/d_µ = (µ-y) · σ⁻² / BATCH.
                    gp[go + i] = d * inv_v * inv_b
                    # d_loss/d_raw_lv = (½ - ½·d²·σ⁻²) / BATCH · in_clamp.
                    var d_lv = (
                        Scalar[DT](0.5)
                        - Scalar[DT](0.5) * d * d * inv_v
                    ) * inv_b * ic
                    gp[go + Self.DIM + i] = d_lv
        else:
            raise Error(
                "GaussianNLLLoss.vjp['gpu'] not implemented"
            )
