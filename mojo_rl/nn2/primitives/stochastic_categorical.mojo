"""StochasticCategorical[N] — Gumbel-Max categorical sampler with
straight-through gradient.

Architectural slot mirrors `GaussianHead` / `RSample`: takes per-row
logits, draws a hard one-hot sample, also computes the log-prob of that
sample under the categorical(softmax(logits)) distribution. Packed
output convention matches RSample.

  input  [BATCH, N]      raw logits
  output [BATCH, N + 1]  packed [one_hot_sample(N) | log_prob(1)]

Forward:
  1.  Draw Gumbel noise: g_i = -log(-log(U(0,1) clamp))
  2.  sample_idx[b] = argmax_i(logits[b, i] + g_i)
  3.  one_hot[b, i] = 1 if i == sample_idx[b] else 0
  4.  log_prob[b]  = log_softmax(logits)[b, sample_idx[b]]
                  = logits[b, sample_idx[b]] − log_sum_exp(logits[b])

Backward (straight-through softmax estimator, à la DreamerV3):
  Treat the discrete sample as if it were `softmax(logits)` for the
  purpose of gradient flow ("Categorical with straight-through trick" in
  Hafner et al. 2023). The one-hot output's gradient is therefore
    d sample[b, j] / d logits[b, k]  =  sm[b, j] · (δ_jk − sm[b, k])
  where `sm = softmax(logits)`.

  Combined with the log-prob output's gradient
    d log_prob[b] / d logits[b, k]  =  δ(k, sample_idx[b]) − sm[b, k]
  the per-row backward becomes
    grad_logits[b, k] = sum_j grad_sample[b, j] · sm[b, j] · (δ_jk − sm[b, k])
                       + grad_log_prob[b] · (δ(k, sample_idx[b]) − sm[b, k])

  Caching: forward stores `softmax(logits)` and `sample_idx` per row so
  backward can run without re-computing the softmax.

This is CPU-only in the first iteration. The kernel pattern is identical
to RSample/GaussianHead — a GPU port lands when DreamerV3 actually needs
it.
"""

from std.math import exp, log
from std.random import random_float64
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for, ensure_cpu_buffer


# ──────────────────────────────────────────────────────────────────────
# StochasticCategorical — categorical sampler with packed output.
# ──────────────────────────────────────────────────────────────────────


struct StochasticCategorical[N: Int](Module):
    comptime IN_DIM = Self.N
    comptime OUT_DIM = Self.N + 1

    var ts: TargetStorage

    # Forward caches (CPU):
    #   _sm[b, i] = softmax(logits)[b, i]            shape BATCH × N
    #   _sample_idx[b] = argmax(logits + gumbel)     shape BATCH
    var _sm: List[Scalar[DT]]
    var _sample_idx: List[Int]
    var _cache_n_batch: Int

    # Public knob: per-call you can drive the RNG yourself by setting
    # `_use_global_random=False` and providing `_gumbel_override`. Default
    # behavior pulls from `std.random.random_float64`.
    # (Kept minimal for now — RSample's RNG advance pattern fits CPU's
    # `std.random` only; a GPU port would add philox seed/offset like
    # box_muller.)

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self._sm = List[Scalar[DT]]()
        self._sample_idx = List[Int]()
        self._cache_n_batch = 0

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "StochasticCategorical.make[target='gpu', INIT] not implemented yet"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "cpu"
        ), "StochasticCategorical GPU path not implemented yet"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    def _ensure_cache_cpu(mut self, batch: Int):
        var sm_needed = batch * Self.N
        if len(self._sm) < sm_needed:
            self._sm.resize(sm_needed, 0.0)
        if len(self._sample_idx) < batch:
            self._sample_idx.resize(batch, 0)
        self._cache_n_batch = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2 [BATCH, N]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, N+1]"
        assert_tag_for["StochasticCategorical", target](self.ts.target_tag)
        comptime assert (
            target == "cpu"
        ), "StochasticCategorical only supports CPU in this revision"

        self._ensure_cache_cpu(BATCH)

        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
        var sm_p = self._sm.unsafe_ptr()
        var idx_p = self._sample_idx.unsafe_ptr()
        comptime OUT = Self.N + 1

        for b in range(BATCH):
            var in_base = b * Self.N
            var out_base = b * OUT

            # 1. log_sum_exp for softmax + log_prob.
            var max_l = in_p[in_base]
            for c in range(1, Self.N):
                var v = in_p[in_base + c]
                if v > max_l:
                    max_l = v
            var sum_exp: Scalar[DT] = 0.0
            for c in range(Self.N):
                sum_exp += exp(in_p[in_base + c] - max_l)
            var lse = max_l + log(sum_exp)

            # 2. cache softmax + sample via Gumbel-max.
            var best_score = Scalar[DT](-1e30)
            var best_idx = 0
            for c in range(Self.N):
                var l = in_p[in_base + c]
                sm_p[in_base + c] = exp(l - lse)
                # Sample Gumbel noise inline:  g = -log(-log(u))
                var u = random_float64()
                if u < 1e-10:
                    u = 1e-10
                var g = -log(-log(Scalar[DT](u)))
                var score = l + g
                if score > best_score:
                    best_score = score
                    best_idx = c
            idx_p[b] = best_idx

            # 3. write packed output.
            for c in range(Self.N):
                out_p[out_base + c] = Scalar[DT](1.0) if c == best_idx else Scalar[DT](0.0)
            out_p[out_base + Self.N] = in_p[in_base + best_idx] - lse

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2 [BATCH, N+1]"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2 [BATCH, N]"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["StochasticCategorical", target](self.ts.target_tag)
        comptime assert target == "cpu", "GPU path not implemented yet"

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
        var sm_p = self._sm.unsafe_ptr()
        var idx_p = self._sample_idx.unsafe_ptr()
        comptime OUT = Self.N + 1

        for b in range(BATCH):
            var in_base = b * Self.N
            var out_base = b * OUT

            # Pull out the two gradient signals.
            # grad_sample[j] = go_p[out_base + j]  (one-hot output)
            # grad_log_prob  = go_p[out_base + N]
            var grad_lp = go_p[out_base + Self.N]
            var sample_idx = idx_p[b]

            # Pre-compute the "expected grad_sample under sm":
            #   E[grad_sample] = sum_i sm[i] · grad_sample[i]
            # because grad_logits[b, k] from the sample branch is
            #   sm[k] · (grad_sample[k] − E[grad_sample]).
            var exp_g: Scalar[DT] = 0.0
            for c in range(Self.N):
                exp_g += sm_p[in_base + c] * go_p[out_base + c]

            # Write grad_logits per element.
            for k in range(Self.N):
                var sm_k = sm_p[in_base + k]
                var dgs = sm_k * (go_p[out_base + k] - exp_g)
                # log_prob branch:
                #   d log_prob / d logits[k] = δ(k, sample_idx) − sm[k]
                var dgl_factor = Scalar[DT](1.0) if k == sample_idx else Scalar[DT](0.0)
                var dgl = grad_lp * (dgl_factor - sm_k)
                gi_p[in_base + k] = dgs + dgl
