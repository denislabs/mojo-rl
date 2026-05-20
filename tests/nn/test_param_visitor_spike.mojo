"""Spike: validate the `ParamVisitor` pattern for `nn2/` (docs/NN2_DESIGN.md).

Question this spike answers:

  Can a stateful struct that OWNS storage expose its `TileTensor` view to
  a trait-bounded visitor, and have mutations through the visitor be
  visible to the owner afterward?

Strategy:
  1. `Module` trait declares `for_each_param[V: ParamVisitor]`.
  2. `Linear` (Module) owns two `UnsafePointer`s — weight + bias.
  3. `SequentialOf2[A: Module, B: Module]` (Module) owns two children.
  4. `ParamVisitor` declares `visit[L: TensorLayout]` taking a TileTensor.
     TileTensor is passed BY VALUE — it's a thin pointer + layout view, so
     mutations through `param.ptr[i] = ...` go to caller-owned storage.
  5. `FillVisitor(val)` writes `val` into every param.
  6. `SumVisitor` reads every param, accumulates a scalar sum.
  7. Run FillVisitor(v), then SumVisitor; assert sum == n * v.
  8. Re-read via the owner's TileTensor view to assert mutations stuck.

CPU only. No GPU. No autodiff.

Run:
    pixi run mojo run -I . tests/nn/test_param_visitor_spike.mojo
"""

from std.memory import alloc
from std.testing import assert_equal
from layout import TileTensor, TensorLayout, row_major

comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────────
# ParamVisitor trait — generic over the param's layout (shape).
# ──────────────────────────────────────────────────────────────────────────

trait ParamVisitor(ImplicitlyDestructible):
    """Visitor invoked once per parameter in a module tree.

    `visit` is parametric over the TileTensor's layout so that 2D weights
    and 1D biases dispatch through one method. `n_elems` is passed
    explicitly for the spike (production version will recover it from the
    TileTensor's runtime_layout API).
    """

    def visit[L: TensorLayout](
        mut self,
        name: StaticString,
        param: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        ...


# ──────────────────────────────────────────────────────────────────────────
# Module trait — uniform tree-walk API for leaves and combinators.
# Extends Movable so combinators can transfer-construct children.
# ──────────────────────────────────────────────────────────────────────────

trait Module(Movable & ImplicitlyDestructible):
    """A neural network module that owns its parameter storage and can
    yield each parameter to a visitor."""

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: StaticString,
        mut visitor: V,
    ):
        ...


# ──────────────────────────────────────────────────────────────────────────
# Linear — leaf module. Owns weight + bias storage. Constructs TileTensor
# views inline in `for_each_param` (no separate accessor methods, since
# the typed return signature for views is awkward in nightly).
# ──────────────────────────────────────────────────────────────────────────

struct Linear[IN: Int, OUT: Int](Module):
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    var weight_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var bias_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.weight_ptr = alloc[Scalar[DT]](Self.W_SIZE)
        self.bias_ptr = alloc[Scalar[DT]](Self.B_SIZE)
        for i in range(Self.W_SIZE):
            self.weight_ptr[i] = 0.0
        for i in range(Self.B_SIZE):
            self.bias_ptr[i] = 0.0

    def __del__(deinit self):
        self.weight_ptr.free()
        self.bias_ptr.free()

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: StaticString,
        mut visitor: V,
    ):
        # For the spike, names are fixed ("weight" / "bias"). Production
        # version would concatenate `prefix + ".weight"`.
        visitor.visit(
            StaticString("weight"),
            TileTensor(self.weight_ptr, row_major[Self.IN, Self.OUT]()),
            Self.W_SIZE,
        )
        visitor.visit(
            StaticString("bias"),
            TileTensor(self.bias_ptr, row_major[Self.OUT]()),
            Self.B_SIZE,
        )


# ──────────────────────────────────────────────────────────────────────────
# SequentialOf2 — combinator. Owns two children, recurses into each.
# Hardcoded arity 2 for the spike; production uses variadic *L.
# ──────────────────────────────────────────────────────────────────────────

struct SequentialOf2[A: Module, B: Module](Module):
    var first: Self.A
    var second: Self.B

    def __init__(out self, var first: Self.A, var second: Self.B):
        self.first = first^
        self.second = second^

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: StaticString,
        mut visitor: V,
    ):
        self.first.for_each_param(prefix, visitor)
        self.second.for_each_param(prefix, visitor)


# ──────────────────────────────────────────────────────────────────────────
# FillVisitor — writes a constant into every param.
# ──────────────────────────────────────────────────────────────────────────

struct FillVisitor(ParamVisitor):
    var value: Scalar[DT]
    var visits: Int

    def __init__(out self, value: Scalar[DT]):
        self.value = value
        self.visits = 0

    def visit[L: TensorLayout](
        mut self,
        name: StaticString,
        param: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        var ptr = param.ptr
        for i in range(n_elems):
            ptr[i] = self.value
        self.visits += 1


# ──────────────────────────────────────────────────────────────────────────
# SumVisitor — reads every param and accumulates.
# ──────────────────────────────────────────────────────────────────────────

struct SumVisitor(ParamVisitor):
    var total: Scalar[DT]
    var count: Int
    var visits: Int

    def __init__(out self):
        self.total = 0.0
        self.count = 0
        self.visits = 0

    def visit[L: TensorLayout](
        mut self,
        name: StaticString,
        param: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        var ptr = param.ptr
        for i in range(n_elems):
            self.total += ptr[i]
        self.count += n_elems
        self.visits += 1


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

def test_single_linear_visitor() raises:
    """Visitor mutates a single Linear; mutations visible to a second visitor."""
    var lin = Linear[4, 3]()  # 12 + 3 = 15 params

    var fill = FillVisitor(1.0)
    lin.for_each_param(StaticString("layer0"), fill)
    assert_equal(fill.visits, 2)  # weight + bias

    var s = SumVisitor()
    lin.for_each_param(StaticString("layer0"), s)
    assert_equal(s.visits, 2)
    assert_equal(s.count, 15)
    assert_equal(s.total, 15.0)
    print("  test_single_linear_visitor PASSED")


def test_sequential_visitor() raises:
    """Combinator recurses; visitor sees params from both children."""
    var net = SequentialOf2(Linear[4, 3](), Linear[3, 2]())
    # 12+3 + 6+2 = 23 params

    var fill = FillVisitor(2.0)
    net.for_each_param(StaticString("net"), fill)
    assert_equal(fill.visits, 4)  # 2 layers × (weight + bias)

    var s = SumVisitor()
    net.for_each_param(StaticString("net"), s)
    assert_equal(s.visits, 4)
    assert_equal(s.count, 23)
    assert_equal(s.total, 46.0)  # 23 * 2.0
    print("  test_sequential_visitor PASSED")


def test_mutation_persists_in_owner() raises:
    """The load-bearing assertion: writes via the visitor land in the
    owner's storage and are readable through a fresh view."""
    var lin = Linear[2, 2]()

    var fill = FillVisitor(7.5)
    lin.for_each_param(StaticString("lin"), fill)

    # Read back through a fresh TileTensor view over the owner's pointers.
    var w = TileTensor(lin.weight_ptr, row_major[2, 2]())
    var b = TileTensor(lin.bias_ptr, row_major[2]())
    for i in range(2):
        for j in range(2):
            assert_equal(w[i, j], 7.5)
    for i in range(2):
        assert_equal(b[i], 7.5)
    print("  test_mutation_persists_in_owner PASSED")


def main() raises:
    print("=" * 60)
    print("ParamVisitor spike (docs/NN2_DESIGN.md, open question #1)")
    print("=" * 60)
    test_single_linear_visitor()
    test_sequential_visitor()
    test_mutation_persists_in_owner()
    print("=" * 60)
    print("ALL PASSED — visitor pattern validated.")
    print("=" * 60)
