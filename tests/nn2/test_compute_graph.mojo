"""CPU tests for ComputeGraph v2 (Phase 10D).

Validates:
  - Variadic `*NODES: GraphNode` storage + comptime name resolution
  - Fan-out from `"input"` (both Linear branches read the graph input)
  - Mixed unary + binary nodes in one graph
  - Backward scatter-add: BinaryElemMin's winner-mask routes grad to the
    correct producer; producer's `_grad_out_buf` accumulates contributions
  - External `grad_input` returned to the caller (FD-checkable)
  - `for_each_param` walks every wrapped Module's params with namespaced
    prefixes

Graph topology:

    input [B, 2]
        │
        ├─→ a: Linear[2, 3]  →  out_a [B, 3]
        │                            │
        └─→ b: Linear[2, 3]  →  out_b [B, 3]
                                     │
                          m: BinaryElemMin[3] (in0="a", in1="b")
                                     │
                                  out [B, 3]

`min` routes gradient: each col c in output gets its grad routed to
whichever of {Linear_a, Linear_b} produced the smaller activation at
that (b, c). Both Linears `backward` then scatter-add into the same
external `_grad_input_buf` — that's the second fan-out exercise.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true, assert_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.combinators import (
    ComputeGraph,
    UnaryNode,
    BinaryNode,
)
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.initializer import Zero


# ──────────────────────────────────────────────────────────────────────────
# Graph type alias used by all tests.
# ──────────────────────────────────────────────────────────────────────────


comptime IN_DIM = 2
comptime OUT_DIM = 3

comptime FanOutMinGraph = ComputeGraph[
    IN_DIM, OUT_DIM,
    UnaryNode["a", Linear[IN_DIM, OUT_DIM], "input"],
    UnaryNode["b", Linear[IN_DIM, OUT_DIM], "input"],
    BinaryNode["m", BinaryElemMin[OUT_DIM], "a", "b"],
]


def _seed_weights(mut g: FanOutMinGraph) raises:
    """Overwrite each Linear's weight/bias with deterministic non-trivial
    values so the two branches diverge and ElemMin's winner-mask is
    non-degenerate."""
    for r in range(IN_DIM):
        for c in range(OUT_DIM):
            g.nodes[0].op.weight[r * OUT_DIM + c] = Scalar[DT](
                Float32(r) * 0.3 + Float32(c) * 0.1 - 0.2
            )
    for c in range(OUT_DIM):
        g.nodes[0].op.bias[c] = Scalar[DT](Float32(c) * 0.05 + 0.1)

    for r in range(IN_DIM):
        for c in range(OUT_DIM):
            g.nodes[1].op.weight[r * OUT_DIM + c] = Scalar[DT](
                -Float32(r) * 0.25 + Float32(c) * 0.4 + 0.15
            )
    for c in range(OUT_DIM):
        g.nodes[1].op.bias[c] = Scalar[DT](-Float32(c) * 0.07 + 0.2)


def _manual_forward(
    g: FanOutMinGraph,
    input_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_min_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    batch: Int,
):
    """min(Linear_a(input), Linear_b(input)) by hand, for parity check."""
    for b in range(batch):
        for c in range(OUT_DIM):
            var s_a: Scalar[DT] = g.nodes[0].op.bias[c]
            var s_b: Scalar[DT] = g.nodes[1].op.bias[c]
            for r in range(IN_DIM):
                s_a += input_buf[b * IN_DIM + r] * g.nodes[0].op.weight[
                    r * OUT_DIM + c
                ]
                s_b += input_buf[b * IN_DIM + r] * g.nodes[1].op.weight[
                    r * OUT_DIM + c
                ]
            out_min_buf[b * OUT_DIM + c] = s_a if s_a < s_b else s_b


# ──────────────────────────────────────────────────────────────────────────
# Test 1 — Forward parity (graph vs hand-computed min(Linear_a, Linear_b)).
# ──────────────────────────────────────────────────────────────────────────


def test_forward_parity() raises:
    comptime BATCH = 4
    var g = FanOutMinGraph.make[target="cpu", INIT=Zero]()
    _seed_weights(g)

    var input_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        input_buf[i] = Scalar[DT](Float32(i) * 0.31 - 0.5)
    var input_t = TileTensor(input_buf, row_major[BATCH, IN_DIM]())

    var out_buf = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_DIM]())
    g.forward["cpu", BATCH](input_t, out_t)

    var out_min = alloc[Scalar[DT]](BATCH * OUT_DIM)
    _manual_forward(g, input_buf, out_min, BATCH)

    for i in range(BATCH * OUT_DIM):
        assert_almost_equal(out_buf[i], out_min[i], atol=1e-6)

    input_buf.free(); out_buf.free(); out_min.free()
    print("  test_forward_parity PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Test 2 — FD gradcheck on grad_input.
# ──────────────────────────────────────────────────────────────────────────


def test_backward_grad_input_fd() raises:
    comptime BATCH = 3
    var g = FanOutMinGraph.make[target="cpu", INIT=Zero]()
    _seed_weights(g)

    var input_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        input_buf[i] = Scalar[DT](Float32(i) * 0.27 - 0.4)
    var input_t = TileTensor(input_buf, row_major[BATCH, IN_DIM]())

    var out_buf = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_DIM]())
    g.forward["cpu", BATCH](input_t, out_t)

    var go_buf = alloc[Scalar[DT]](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        go_buf[i] = Scalar[DT](Float32(i) * 0.19 + 0.2)
    var go_t = TileTensor(go_buf, row_major[BATCH, OUT_DIM]())

    var gi_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    var gi_t = TileTensor(gi_buf, row_major[BATCH, IN_DIM]())
    g.backward["cpu", BATCH](go_t, gi_t)

    # FD on each input element. Virtual loss L = Σ go · output.
    var eps: Scalar[DT] = 1e-3
    var max_rel: Scalar[DT] = 0.0
    for idx in range(BATCH * IN_DIM):
        var orig = input_buf[idx]

        input_buf[idx] = orig + eps
        g.forward["cpu", BATCH](input_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT_DIM):
            L_plus += go_buf[k] * out_buf[k]

        input_buf[idx] = orig - eps
        g.forward["cpu", BATCH](input_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT_DIM):
            L_minus += go_buf[k] * out_buf[k]

        input_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel:
            max_rel = rel
    print("  ComputeGraph FD max_rel(grad_input)=", max_rel)
    assert_true(max_rel < Scalar[DT](5e-3), "ComputeGraph FD grad_input too loose")

    input_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()
    print("  test_backward_grad_input_fd PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Test 3 — for_each_param walks every wrapped Linear with namespaced prefix.
# ──────────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _NameCollector(ParamVisitor):
    """Pushes visited param names into a caller-owned list."""

    var names_ptr: UnsafePointer[List[String], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
        ) raises:
        self.names_ptr[].append(name)


def test_for_each_param_walks_wrapped() raises:
    """`for_each_param` should walk both wrapped Linears with name prefix
    derived from each node's NAME (`"a."`, `"b."`). BinaryElemMin has no
    params, so it doesn't show up."""
    var g = FanOutMinGraph.make[target="cpu", INIT=Zero]()

    var names = List[String]()
    var collector = _NameCollector(names_ptr=UnsafePointer(to=names))
    g.for_each_param["cpu"](String(""), collector)

    # 2 Linears × {weight, bias} = 4 params; BinaryElemMin contributes 0.
    assert_equal(len(names), 4)

    var seen_a_w = False
    var seen_a_b = False
    var seen_b_w = False
    var seen_b_b = False
    for nm in names:
        if nm.startswith("a.") and nm.endswith(".weight"):
            seen_a_w = True
        if nm.startswith("a.") and nm.endswith(".bias"):
            seen_a_b = True
        if nm.startswith("b.") and nm.endswith(".weight"):
            seen_b_w = True
        if nm.startswith("b.") and nm.endswith(".bias"):
            seen_b_b = True
    assert_true(seen_a_w, "missing a.weight")
    assert_true(seen_a_b, "missing a.bias")
    assert_true(seen_b_w, "missing b.weight")
    assert_true(seen_b_b, "missing b.bias")

    print("  test_for_each_param_walks_wrapped PASSED")


def main() raises:
    print("=" * 70)
    print("nn2 Phase 10D — ComputeGraph v2 CPU tests")
    print("=" * 70)
    test_forward_parity()
    test_backward_grad_input_fd()
    test_for_each_param_walks_wrapped()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
