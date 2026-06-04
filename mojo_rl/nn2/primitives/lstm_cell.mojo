"""LSTMCell[IN_, HIDDEN] — PyTorch-equivalent LSTM cell for nn2.

Unlike a feed-forward `Module`, an LSTM threads TWO states (h, c) across
time and is trained with BPTT, so it exposes an explicit recurrent API
(`step_forward` / `step_backward` / `step_forward_no_cache`) rather than
the single-input/single-output `Module.forward`. The caller owns the
(h, c) state and a per-timestep cache buffer, and runs the BPTT loop
(see `examples/nn2/lstm/`). This matches the legacy `mojo_rl.nn` LSTMCell
and the nn2 `GRUCell` math conventions.

Parameters are nn2 `Param` fields, so `for_each_param` / `zero_grad`
work and the cell composes with nn2 `Adam` / checkpointing. The cell
still conforms to `Module` (for the optimizer's `M: Module` bound), but
its `Module.forward` / `vjp` raise — use the step API.

Math (PyTorch convention, gates packed [i | f | g | o], each HIDDEN):
    preact = x · W_ih + h_prev · W_hh + b           [BATCH, 4·H]
    i = σ(preact[0:H]),  f = σ(preact[H:2H])
    g = tanh(preact[2H:3H]),  o = σ(preact[3H:4H])
    c_t = f ⊙ c_prev + i ⊙ g
    h_t = o ⊙ tanh(c_t)

Storage (row-major):
    W_ih [IN, 4·H]   W_hh [H, 4·H]   b [4·H]
Cache (per timestep, BATCH-major): [i | f | g | o | tanh(c_t)], 5·H wide.

CPU only in this revision (mirrors GRUCell); GPU port deferred.
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
from ..core.target_storage import TargetStorage, assert_tag_for


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


struct LSTMCell[IN_: Int, HIDDEN: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM = 2 * Self.HIDDEN  # packed [h ; c]

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.IN_
        d[1] = 2 * Self.HIDDEN
        return d

    comptime W_IH_SIZE = Self.IN_ * (4 * Self.HIDDEN)
    comptime W_HH_SIZE = Self.HIDDEN * (4 * Self.HIDDEN)
    comptime B_SIZE = 4 * Self.HIDDEN
    comptime CACHE_SIZE = 5 * Self.HIDDEN  # [i | f | g | o | tanh_c]

    var ts: TargetStorage
    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b:    Param["b",    False, Self.B_SIZE]

    # ------------------------------------------------------------------
    # Defaultable + factories.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b    = Param["b",    False, Self.B_SIZE]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified factory. CPU-only for now; GPU path not implemented."""
        comptime assert target == "cpu" or target == "gpu", (
            "LSTMCell: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            raise Error("LSTMCell.make[target='gpu'] not implemented yet")
        var m = Self()
        m.ts = TargetStorage.make_cpu()
        m.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_cpu()
        m.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_cpu()
        m.b    = Param["b",    False, Self.B_SIZE].make_cpu()
        INIT.init_weight(
            m.W_ih.value_unsafe_ptr_cpu(),
            Self.W_IH_SIZE, Self.IN_, 4 * Self.HIDDEN,
        )
        INIT.init_weight(
            m.W_hh.value_unsafe_ptr_cpu(),
            Self.W_HH_SIZE, Self.HIDDEN, 4 * Self.HIDDEN,
        )
        INIT.init_bias(m.b.value_unsafe_ptr_cpu(), Self.B_SIZE)
        return m^

    # ------------------------------------------------------------------
    # Param-walker overrides (so nn2 Adam / zero_grad / checkpoint work).
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        zero_grad_auto[Self, target](self)

    # ------------------------------------------------------------------
    # Module conformance — recurrent cell uses the step API instead.
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
        raise Error(
            "LSTMCell is recurrent — use step_forward/step_backward "
            "(see examples/nn2/lstm), not Module.forward"
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
        raise Error(
            "LSTMCell is recurrent — use step_backward "
            "(see examples/nn2/lstm), not Module.vjp"
        )

    # ------------------------------------------------------------------
    # Recurrent step API.
    # ------------------------------------------------------------------

    def step_forward[BATCH: Int](
        mut self,
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut h_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut c_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut cache: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """One LSTM step on CPU; writes h_t, c_t, and the backward cache
        ([i | f | g | o | tanh_c], 5·H wide)."""
        assert_tag_for["LSTMCell", "cpu"](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN
        var xv = typed_view[BATCH, Self.IN_](x)
        var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
        var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
        var ht = typed_view_mut[BATCH, Self.HIDDEN](h_t)
        var ct = typed_view_mut[BATCH, Self.HIDDEN](c_t)
        var cc = typed_view_mut[BATCH, Self.CACHE_SIZE](cache)
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var b_p = self.b.value_unsafe_ptr_cpu()

        for bi in range(BATCH):
            for k in range(FOURH):
                var pre: Scalar[DT] = b_p[k]
                for j in range(Self.IN_):
                    pre += xv[bi, j] * W_ih_p[j * FOURH + k]
                for j in range(H):
                    pre += hp[bi, j] * W_hh_p[j * FOURH + k]
                var act: Scalar[DT]
                if k < 3 * H:
                    act = _sigmoid(pre) if k < 2 * H else tanh(pre)
                else:
                    act = _sigmoid(pre)
                cc[bi, k] = act
            for j in range(H):
                var i_v = cc[bi, j]
                var f_v = cc[bi, H + j]
                var g_v = cc[bi, 2 * H + j]
                var o_v = cc[bi, 3 * H + j]
                var c_new = f_v * cp[bi, j] + i_v * g_v
                var tc = tanh(c_new)
                ct[bi, j] = c_new
                ht[bi, j] = o_v * tc
                cc[bi, 4 * H + j] = tc

    def step_forward_no_cache[BATCH: Int](
        mut self,
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut h_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut c_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Inference step (no cache) — for eval / sampling."""
        assert_tag_for["LSTMCell", "cpu"](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN
        var xv = typed_view[BATCH, Self.IN_](x)
        var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
        var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
        var ht = typed_view_mut[BATCH, Self.HIDDEN](h_t)
        var ct = typed_view_mut[BATCH, Self.HIDDEN](c_t)
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var b_p = self.b.value_unsafe_ptr_cpu()

        for bi in range(BATCH):
            var gates = InlineArray[Scalar[DT], 4 * Self.HIDDEN](fill=0.0)
            for k in range(FOURH):
                var pre: Scalar[DT] = b_p[k]
                for j in range(Self.IN_):
                    pre += xv[bi, j] * W_ih_p[j * FOURH + k]
                for j in range(H):
                    pre += hp[bi, j] * W_hh_p[j * FOURH + k]
                if k < 3 * H:
                    gates[k] = _sigmoid(pre) if k < 2 * H else tanh(pre)
                else:
                    gates[k] = _sigmoid(pre)
            for j in range(H):
                var c_new = gates[H + j] * cp[bi, j] + gates[j] * gates[2 * H + j]
                ct[bi, j] = c_new
                ht[bi, j] = gates[3 * H + j] * tanh(c_new)

    def step_backward[BATCH: Int](
        mut self,
        dh: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        dc: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        cache: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dx: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dh_prev: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dc_prev: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """One BPTT step. `dh`/`dc` are incoming grads w.r.t. h_t / c_t
        (pass dc=0 at the last timestep). Writes dx, dh_prev, dc_prev
        (thread the latter two back as dh/dc for the previous step) and
        ACCUMULATES into the cell's parameter grads."""
        assert_tag_for["LSTMCell", "cpu"](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN
        var dh_v = typed_view[BATCH, Self.HIDDEN](dh)
        var dc_v = typed_view[BATCH, Self.HIDDEN](dc)
        var xv = typed_view[BATCH, Self.IN_](x)
        var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
        var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
        var cc = typed_view[BATCH, Self.CACHE_SIZE](cache)
        var dxv = typed_view_mut[BATCH, Self.IN_](dx)
        var dhp = typed_view_mut[BATCH, Self.HIDDEN](dh_prev)
        var dcp = typed_view_mut[BATCH, Self.HIDDEN](dc_prev)
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var dW_ih_p = self.W_ih.grad_unsafe_ptr_cpu()
        var dW_hh_p = self.W_hh.grad_unsafe_ptr_cpu()
        var db_p = self.b.grad_unsafe_ptr_cpu()

        for bi in range(BATCH):
            # Pass 1: per-element pre-activation grads → d_combined[4H].
            var d_comb = InlineArray[Scalar[DT], 4 * Self.HIDDEN](fill=0.0)
            for j in range(H):
                var i_v = cc[bi, j]
                var f_v = cc[bi, H + j]
                var g_v = cc[bi, 2 * H + j]
                var o_v = cc[bi, 3 * H + j]
                var tc = cc[bi, 4 * H + j]
                var dh_j = dh_v[bi, j]
                var dc_j = dc_v[bi, j]

                var do_post = dh_j * tc
                var dc_total = dc_j + dh_j * o_v * (Scalar[DT](1.0) - tc * tc)
                var df_post = dc_total * cp[bi, j]
                var di_post = dc_total * g_v
                var dg_post = dc_total * i_v
                dcp[bi, j] = dc_total * f_v

                d_comb[j]         = di_post * i_v * (Scalar[DT](1.0) - i_v)
                d_comb[H + j]     = df_post * f_v * (Scalar[DT](1.0) - f_v)
                d_comb[2 * H + j] = dg_post * (Scalar[DT](1.0) - g_v * g_v)
                d_comb[3 * H + j] = do_post * o_v * (Scalar[DT](1.0) - o_v)

            # Pass 2: accumulate param grads, compute dx / dh_prev.
            for j in range(Self.IN_):
                var xvj = xv[bi, j]
                for k in range(FOURH):
                    dW_ih_p[j * FOURH + k] += xvj * d_comb[k]
            for j in range(H):
                var hvj = hp[bi, j]
                for k in range(FOURH):
                    dW_hh_p[j * FOURH + k] += hvj * d_comb[k]
            for k in range(FOURH):
                db_p[k] += d_comb[k]

            for j in range(Self.IN_):
                var acc: Scalar[DT] = 0.0
                for k in range(FOURH):
                    acc += d_comb[k] * W_ih_p[j * FOURH + k]
                dxv[bi, j] = acc
            for j in range(H):
                var acc: Scalar[DT] = 0.0
                for k in range(FOURH):
                    acc += d_comb[k] * W_hh_p[j * FOURH + k]
                dhp[bi, j] = acc
