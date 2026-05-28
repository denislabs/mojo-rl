"""GRUCell[IN_DIM, HIDDEN_DIM] — PyTorch-equivalent GRU cell as
BinaryModule (input, hidden) → new_hidden.

Math (PyTorch convention):
  r = σ(x · W_ir + b_ir + h · W_hr + b_hr)         reset gate
  z = σ(x · W_iz + b_iz + h · W_hz + b_hz)         update gate
  n = tanh(x · W_in + b_in + r ⊙ (h · W_hn + b_hn)) new candidate
  h' = (1 − z) ⊙ n + z ⊙ h

Storage convention (row-major):
  W_ih [IN, 3·H]  — columns 0..H = r, H..2H = z, 2H..3H = n
  W_hh [H,  3·H]  — same column split
  b_ih [3·H]
  b_hh [3·H]

Caches (BATCH-sized, allocated lazily):
  r [B, H], z [B, H], n [B, H]                    activations
  hn_pre [B, H]                                   W_hn·h + b_hn (pre-r-gate)

Backward (BinaryModule, mode-aware):
  Given dh' = grad_output (shape [B, H]):
    dz       = dh' ⊙ (h − n)
    dn       = dh' ⊙ (1 − z)
    d_pre_n  = dn ⊙ (1 − n²)
    d_in_n   = d_pre_n
    dr_x_hn  = d_pre_n              # the `r·hn` summand
    dr       = dr_x_hn ⊙ hn_pre
    d_hn     = dr_x_hn ⊙ r          # gradient on hn_pre (pre-r-gate)
    d_pre_r  = dr ⊙ r ⊙ (1 − r)
    d_pre_z  = dz ⊙ z ⊙ (1 − z)
    d_ir = d_pre_r,  d_hr = d_pre_r
    d_iz = d_pre_z,  d_hz = d_pre_z
    d_in = d_in_n              # only the input-projection part of n's pre-act
    # hn already accounts for the r-gate

  Param grads (mode == "all"):
    d_W_ih [IN, 3H] += x^T · [d_ir | d_iz | d_in]
    d_b_ih [3H]     += sum_B [d_ir | d_iz | d_in]
    d_W_hh [H,  3H] += h^T · [d_hr | d_hz | d_hn]
    d_b_hh [3H]     += sum_B [d_hr | d_hz | d_hn]

  Input grads:
    d_x = [d_ir | d_iz | d_in] · W_ih^T
    d_h = [d_hr | d_hz | d_hn] · W_hh^T + dh' ⊙ z   # last term: direct h
                                                       path through `z·h`

CPU only in this revision — the GPU port mirrors the pattern of `Linear`
+ `Tanh` once needed by DreamerV3.
"""

from std.math import exp, tanh
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    for_each_param_auto,
    zero_grad_auto,
    ParamVisitor,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for, ensure_cpu_buffer


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


# ──────────────────────────────────────────────────────────────────────
# GRUCell.
# ──────────────────────────────────────────────────────────────────────


struct GRUCell[IN_: Int, HIDDEN: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM = Self.IN_
    comptime OUT_DIM = Self.HIDDEN

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.IN_
        d[1] = Self.HIDDEN
        return d
    comptime W_IH_SIZE = Self.IN_ * (3 * Self.HIDDEN)
    comptime W_HH_SIZE = Self.HIDDEN * (3 * Self.HIDDEN)
    comptime B_IH_SIZE = 3 * Self.HIDDEN

    var ts: TargetStorage

    # Parameters — Param fields are walked by reflection (for_each_param_auto).
    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b_ih: Param["b_ih", False, Self.B_IH_SIZE]
    var b_hh: Param["b_hh", False, Self.B_IH_SIZE]

    # Forward caches.
    var _r_cache: List[Scalar[DT]]   # [BATCH, H]
    var _z_cache: List[Scalar[DT]]   # [BATCH, H]
    var _n_cache: List[Scalar[DT]]   # [BATCH, H]
    var _hn_pre:  List[Scalar[DT]]   # [BATCH, H]  W_hn·h + b_hn
    var _x_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _h_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _cache_batch: Int

    # ------------------------------------------------------------------
    # Defaultable + factories.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b_ih = Param["b_ih", False, Self.B_IH_SIZE]()
        self.b_hh = Param["b_hh", False, Self.B_IH_SIZE]()
        self._r_cache = List[Scalar[DT]]()
        self._z_cache = List[Scalar[DT]]()
        self._n_cache = List[Scalar[DT]]()
        self._hn_pre  = List[Scalar[DT]]()
        self._x_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._h_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._cache_batch = 0

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified factory. CPU-only for now; GPU path not implemented."""
        comptime assert target == "cpu" or target == "gpu", (
            "GRUCell: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            raise Error("GRUCell.make[target='gpu'] not implemented yet")
        var g = Self()
        g.ts = TargetStorage.make_cpu()
        g.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_cpu()
        g.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_cpu()
        g.b_ih = Param["b_ih", False, Self.B_IH_SIZE].make_cpu()
        g.b_hh = Param["b_hh", False, Self.B_IH_SIZE].make_cpu()
        INIT.init_weight(
            g.W_ih.value_unsafe_ptr_cpu(),
            Self.W_IH_SIZE, Self.IN_, 3 * Self.HIDDEN,
        )
        INIT.init_weight(
            g.W_hh.value_unsafe_ptr_cpu(),
            Self.W_HH_SIZE, Self.HIDDEN, 3 * Self.HIDDEN,
        )
        INIT.init_bias(g.b_ih.value_unsafe_ptr_cpu(), Self.B_IH_SIZE)
        INIT.init_bias(g.b_hh.value_unsafe_ptr_cpu(), Self.B_IH_SIZE)
        return g^

    # ------------------------------------------------------------------
    # Param-walker overrides (param-bearing leaf).
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        zero_grad_auto[Self, target](self)

    # ------------------------------------------------------------------
    # Cache management.
    # ------------------------------------------------------------------

    def _ensure_cache(mut self, batch: Int):
        var needed = batch * Self.HIDDEN
        if len(self._r_cache) < needed:
            self._r_cache.resize(needed, 0.0)
        if len(self._z_cache) < needed:
            self._z_cache.resize(needed, 0.0)
        if len(self._n_cache) < needed:
            self._n_cache.resize(needed, 0.0)
        if len(self._hn_pre) < needed:
            self._hn_pre.resize(needed, 0.0)
        self._cache_batch = batch

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

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
        comptime assert target == "cpu", "GRUCell only supports CPU"
        assert_tag_for["GRUCell", target](self.ts.target_tag)

        self._ensure_cache(BATCH)

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN
        comptime IN_ = Self.IN_

        var x_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inputs[0].ptr)
        var h_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inputs[1].ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
        self._x_ptr = x_p
        self._h_ptr = h_p

        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var b_ih_p = self.b_ih.value_unsafe_ptr_cpu()
        var b_hh_p = self.b_hh.value_unsafe_ptr_cpu()
        var r_c = self._r_cache.unsafe_ptr()
        var z_c = self._z_cache.unsafe_ptr()
        var n_c = self._n_cache.unsafe_ptr()
        var hn_c = self._hn_pre.unsafe_ptr()

        # For each row b:
        #   1. compute ix_j = sum_k x[b,k]*W_ih[k,j] + b_ih[j]  for j in [0, 3H)
        #   2. compute hx_j = sum_k h[b,k]*W_hh[k,j] + b_hh[j]  for j in [0, 3H)
        #   3. r = σ(ix[0:H]  + hx[0:H])
        #      z = σ(ix[H:2H] + hx[H:2H])
        #      hn_pre = hx[2H:3H]
        #      n = tanh(ix[2H:3H] + r ⊙ hn_pre)
        #      h' = (1-z)*n + z*h
        for b in range(BATCH):
            var x_off = b * IN_
            var h_off = b * H
            var out_off = b * H
            var c_off = b * H

            # Slot scratch on stack via inline loops (no large temporaries).
            for col in range(H):
                # ir + hr
                var ir: Scalar[DT] = b_ih_p[col]
                var hr: Scalar[DT] = b_hh_p[col]
                for k in range(IN_):
                    ir += x_p[x_off + k] * W_ih_p[k * THREE_H + col]
                for k in range(H):
                    hr += h_p[h_off + k] * W_hh_p[k * THREE_H + col]
                var rg = _sigmoid(ir + hr)
                r_c[c_off + col] = rg

                # iz + hz
                var iz: Scalar[DT] = b_ih_p[H + col]
                var hz: Scalar[DT] = b_hh_p[H + col]
                for k in range(IN_):
                    iz += x_p[x_off + k] * W_ih_p[k * THREE_H + H + col]
                for k in range(H):
                    hz += h_p[h_off + k] * W_hh_p[k * THREE_H + H + col]
                var zg = _sigmoid(iz + hz)
                z_c[c_off + col] = zg

                # in_pre = sum + b_in (only x-side part)
                var in_pre: Scalar[DT] = b_ih_p[2 * H + col]
                for k in range(IN_):
                    in_pre += x_p[x_off + k] * W_ih_p[k * THREE_H + 2 * H + col]
                # hn_pre = sum + b_hn (h-side part — full pre-activation
                # for the n branch, before the r gate).
                var hn_p: Scalar[DT] = b_hh_p[2 * H + col]
                for k in range(H):
                    hn_p += h_p[h_off + k] * W_hh_p[k * THREE_H + 2 * H + col]
                hn_c[c_off + col] = hn_p

                var ng = tanh(in_pre + rg * hn_p)
                n_c[c_off + col] = ng

                # h' = (1 − z) * n + z * h
                out_p[out_off + col] = (
                    (Scalar[DT](1.0) - zg) * ng + zg * h_p[h_off + col]
                )

    # ------------------------------------------------------------------
    # Backward.
    # ------------------------------------------------------------------

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
        comptime assert target == "cpu", "GRUCell only supports CPU"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GRUCell", target](self.ts.target_tag)

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN
        comptime IN_ = Self.IN_

        var x_p = self._x_ptr
        var h_p = self._h_ptr
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var dW_ih_p = self.W_ih.grad_unsafe_ptr_cpu()
        var dW_hh_p = self.W_hh.grad_unsafe_ptr_cpu()
        var db_ih_p = self.b_ih.grad_unsafe_ptr_cpu()
        var db_hh_p = self.b_hh.grad_unsafe_ptr_cpu()
        var r_c = self._r_cache.unsafe_ptr()
        var z_c = self._z_cache.unsafe_ptr()
        var n_c = self._n_cache.unsafe_ptr()
        var hn_c = self._hn_pre.unsafe_ptr()

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var dx_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[0].ptr)
        var dh_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[1].ptr)

        # PARAM-GRAD-FIRST INVARIANT: like Linear, x_p and h_p MAY alias
        # the orchestrator's input slabs that dx_p / dh_p write to. We
        # compute all parameter grads (which read x, h) and the input
        # grads (which read W) into stack scratch first, then write
        # dx, dh as the final step. Since dx and dh are produced from
        # the per-row d_pre_* signals (not from x, h directly), it's safe
        # to interleave per-row.

        # Initialize grad inputs to zero (we accumulate into them per element).
        for b in range(BATCH):
            for k in range(IN_):
                dx_p[b * IN_ + k] = 0.0
            for k in range(H):
                dh_p[b * H + k] = 0.0

        for b in range(BATCH):
            var x_off = b * IN_
            var h_off = b * H
            var c_off = b * H

            for col in range(H):
                var dh_now = go_p[c_off + col]
                var rg = r_c[c_off + col]
                var zg = z_c[c_off + col]
                var ng = n_c[c_off + col]
                var hn_v = hn_c[c_off + col]
                var h_val = h_p[h_off + col]

                # Gate / candidate gradients.
                var dz = dh_now * (h_val - ng)
                var dn = dh_now * (Scalar[DT](1.0) - zg)
                var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)  # tanh'

                # Split d_pre_n across input-projection (d_in_n) and
                # `r * hn` summand.
                var d_in_n = d_pre_n
                var dr_x_hn = d_pre_n
                var dr = dr_x_hn * hn_v          # gradient on r
                var d_hn = dr_x_hn * rg          # gradient on hn_pre

                var d_pre_r = dr * rg * (Scalar[DT](1.0) - rg)  # sigmoid'
                var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)  # sigmoid'

                # Combined d-vectors per gate index.
                var d_ir = d_pre_r
                var d_iz = d_pre_z
                var d_in_g = d_in_n
                var d_hr_g = d_pre_r
                var d_hz_g = d_pre_z
                var d_hn_g = d_hn

                # ----- Param grads (mode == "all" only) -----
                comptime if mode == "all":
                    # b_ih: sum across batch
                    db_ih_p[col]         += d_ir
                    db_ih_p[H + col]     += d_iz
                    db_ih_p[2 * H + col] += d_in_g
                    db_hh_p[col]         += d_hr_g
                    db_hh_p[H + col]     += d_hz_g
                    db_hh_p[2 * H + col] += d_hn_g

                    # W_ih [IN, 3H] += x^T · d_ix
                    for k in range(IN_):
                        var xv = x_p[x_off + k]
                        dW_ih_p[k * THREE_H + col]         += xv * d_ir
                        dW_ih_p[k * THREE_H + H + col]     += xv * d_iz
                        dW_ih_p[k * THREE_H + 2 * H + col] += xv * d_in_g
                    # W_hh [H, 3H] += h^T · d_hx
                    for k in range(H):
                        var hv = h_p[h_off + k]
                        dW_hh_p[k * THREE_H + col]         += hv * d_hr_g
                        dW_hh_p[k * THREE_H + H + col]     += hv * d_hz_g
                        dW_hh_p[k * THREE_H + 2 * H + col] += hv * d_hn_g

                # ----- Input grads -----
                # d_x[k] += d_ir·W_ih[k, col] + d_iz·W_ih[k, H+col] + d_in_g·W_ih[k, 2H+col]
                for k in range(IN_):
                    dx_p[x_off + k] += (
                        d_ir   * W_ih_p[k * THREE_H + col]
                        + d_iz * W_ih_p[k * THREE_H + H + col]
                        + d_in_g * W_ih_p[k * THREE_H + 2 * H + col]
                    )
                # d_h[k] += d_hr_g·W_hh[k, col] + d_hz_g·W_hh[k, H+col] + d_hn_g·W_hh[k, 2H+col]
                # Plus the direct path through `z · h` — only for k = col.
                for k in range(H):
                    dh_p[h_off + k] += (
                        d_hr_g   * W_hh_p[k * THREE_H + col]
                        + d_hz_g * W_hh_p[k * THREE_H + H + col]
                        + d_hn_g * W_hh_p[k * THREE_H + 2 * H + col]
                    )
                # Direct path: ∂h'/∂h_col = z_col
                dh_p[h_off + col] += dh_now * zg
