"""SPIKE — de-risks LeWM nn port path A (docs/LEWM_NN_PORT_PLAN.md §5.1).

Path A makes the LeWM AR-predictor `ConditionalTransformerBlock` an
ARITY=2 `Module` (inputs `x`, `c`) bound into the JEPA loss graph as an
`ExternalNode` (params owned by the trainer, like SAC's actor/critics).

Existing tests cover *parts* of this but not the union:
  - `test_compute_graph_external.py::test_external_binary_node`: ARITY=2
    ExternalNode forward only, and a PARAMLESS op (BinarySub).
  - `test_hetero_binary_graph`: ARITY=2 backward to both inputs, but via
    an owned `Node` and a paramless op (Concat).

The genuinely unproven combination for path A:
  *a PARAM-BEARING custom ARITY=2 Module bound as an ExternalNode, run
   forward + vjp, routing gradients to BOTH input slots AND accumulating
   its own parameter gradient.*

`ToyBlock[D]` (`y[b,i] = x[b,i]*w[i] + c[b,i]`, learnable w) is the
minimal stand-in for the real cond block. If this passes, path A's graph
mechanics are sound and the port can proceed to the primitives.

Closed-form on the chosen inputs:
  y      = x*w + c
  grad_x = grad_out * w
  grad_c = grad_out
  grad_w[i] += sum_b grad_out[b,i] * x[b,i]
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators import ComputeGraph, InputSlot, ExternalNode
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from mojo_rl.nn.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# ToyBlock[D] — minimal param-bearing ARITY=2 module.  y = x*w + c.
# Caches x for the w-gradient (vjp has no access to forward inputs).
# CPU-only spike: the GPU branch raises (never instantiated here).
# ──────────────────────────────────────────────────────────────────────


struct ToyBlock[D: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D)
    comptime OUT_DIM = Self.D

    var w: Param["w", True, Self.D]
    var _xc: List[Scalar[DT]]  # cached x for grad_w
    var ts: TargetStorage

    def __init__(out self):
        self.w = Param["w", True, Self.D]()
        self._xc = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var tb = Self()
        comptime if target == "cpu":
            tb.w = Param["w", True, Self.D].make_cpu()
            tb.ts = TargetStorage.make_cpu()
        else:
            raise Error("ToyBlock: CPU-only spike")
        return tb^

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
        assert_tag_for["ToyBlock", target](self.ts.target_tag)
        var x = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var c = typed_view[BATCH, Self.IN_DIMS[1]](inputs[1])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            if len(self._xc) < BATCH * Self.D:
                self._xc.resize(BATCH * Self.D, Scalar[DT](0.0))
            var w_v = TileTensor(self.w.value, row_major[Self.D]())
            for b in range(BATCH):
                for i in range(Self.D):
                    self._xc[b * Self.D + i] = rebind[Scalar[DT]](x[b, i])
                    out[b, i] = x[b, i] * w_v[i] + c[b, i]
        else:
            raise Error("ToyBlock: CPU-only spike")

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
        assert_tag_for["ToyBlock", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gx = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        var gc = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])

        comptime if target == "cpu":
            var w_v = TileTensor(self.w.value, row_major[Self.D]())
            for b in range(BATCH):
                for i in range(Self.D):
                    gx[b, i] = go[b, i] * w_v[i]
                    gc[b, i] = go[b, i]
            comptime if mode == "all":
                var gw = TileTensor(self.w.grad, row_major[Self.D]())
                for b in range(BATCH):
                    for i in range(Self.D):
                        gw[i] += go[b, i] * self._xc[b * Self.D + i]
        else:
            raise Error("ToyBlock: CPU-only spike")

    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["ToyBlock", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["ToyBlock", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)


def test_external_binary_param_block() raises:
    print("test_external_binary_param_block ...")
    comptime BATCH = 3
    comptime D = 2

    comptime G = ComputeGraph[
        D,
        InputSlot["x", D],
        InputSlot["c", D],
        ExternalNode["blk", ToyBlock[D], "x", "c"],
    ]
    var g = G.make[target="cpu", INIT=Kaiming]()

    # Trainer-owned param-bearing ARITY=2 module. w = [2, -3].
    var blk = ToyBlock[D].make[target="cpu", INIT=Kaiming]()
    blk.w.value[0] = Scalar[DT](2.0)
    blk.w.value[1] = Scalar[DT](-3.0)
    blk.zero_grad["cpu"]()
    g.set_external["blk", ToyBlock[D]](blk)

    var x_buf = alloc[Scalar[DT]](BATCH * D)
    var c_buf = alloc[Scalar[DT]](BATCH * D)
    var out_buf = alloc[Scalar[DT]](BATCH * D)
    var go_buf = alloc[Scalar[DT]](BATCH * D)
    for b in range(BATCH):
        for i in range(D):
            x_buf[b * D + i] = Scalar[DT](Float64(b) + 1.0 + 0.5 * Float64(i))
            c_buf[b * D + i] = Scalar[DT](10.0 * Float64(b) + Float64(i))
            go_buf[b * D + i] = Scalar[DT](0.1 * Float64(b * D + i + 1))

    var x_t = TileTensor(x_buf, row_major[BATCH, D]())
    var c_t = TileTensor(c_buf, row_major[BATCH, D]())
    var out_t = TileTensor(out_buf, row_major[BATCH, D]())
    g.set_input["x", BATCH](x_t)
    g.set_input["c", BATCH](c_t)
    g.forward["cpu", BATCH](out_t)

    # (1) forward: out = x*w + c
    for b in range(BATCH):
        for i in range(D):
            var w_i = Scalar[DT](2.0) if i == 0 else Scalar[DT](-3.0)
            var want = x_buf[b * D + i] * w_i + c_buf[b * D + i]
            assert_true(
                (out_buf[b * D + i] - want).__abs__() < Scalar[DT](1e-5),
                "forward: out must equal x*w + c",
            )

    var go_t = TileTensor(go_buf, row_major[BATCH, D]())
    g.vjp["cpu", BATCH](go_t)

    # (2) grad to BOTH input slots
    var gx_p = g.grad_input_ptr["x"]()
    var gc_p = g.grad_input_ptr["c"]()
    for b in range(BATCH):
        for i in range(D):
            var w_i = Scalar[DT](2.0) if i == 0 else Scalar[DT](-3.0)
            var want_gx = go_buf[b * D + i] * w_i
            var want_gc = go_buf[b * D + i]
            assert_true(
                (gx_p[b * D + i] - want_gx).__abs__() < Scalar[DT](1e-5),
                "backward: grad_x must equal grad_out * w",
            )
            assert_true(
                (gc_p[b * D + i] - want_gc).__abs__() < Scalar[DT](1e-5),
                "backward: grad_c must equal grad_out",
            )

    # (3) parameter gradient accumulated in the trainer-owned module:
    #     grad_w[i] = sum_b grad_out[b,i] * x[b,i]
    for i in range(D):
        var want_gw: Scalar[DT] = 0.0
        for b in range(BATCH):
            want_gw += go_buf[b * D + i] * x_buf[b * D + i]
        var got_gw = blk.w.grad[i]
        assert_true(
            (got_gw - want_gw).__abs__() < Scalar[DT](1e-5),
            "param grad: w.grad must equal sum_b grad_out*x",
        )

    x_buf.free()
    c_buf.free()
    out_buf.free()
    go_buf.free()
    # Keep `blk` alive past the last graph call (set_external stored its ptr).
    _ = blk^
    print("  ok — forward, grad-to-both-inputs, and param-grad all correct")


def main() raises:
    print("=" * 70)
    print("SPIKE: param-bearing ARITY=2 ExternalNode (LeWM path A de-risk)")
    print("=" * 70)
    test_external_binary_param_block()
    print("=" * 70)
    print("SPIKE PASSED — path A graph mechanics are sound")
    print("=" * 70)
