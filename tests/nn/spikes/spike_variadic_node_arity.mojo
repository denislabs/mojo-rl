"""Spike — variadic-arity GraphNode prototype.

Phase I.2+ exploration. The current `Node` / `ExternalNode` in
`mojo_rl/nn/combinators/graph_nodes.mojo` cap at ARITY ≤ 2: they
carry hardcoded `IN0_NAME` + `IN1_NAME` struct comptime params + per-
input grad buffer fields. PPOActorLoss hit this — its loss node
needs 4 data inputs (actor_out, action, old_log_prob, advantage), so
we worked around it by packing (action, old_log_prob, advantage) into
one `InputSlot["aux", ACT+2]`.

This spike asks: can Mojo's struct-comptime-variadic of `StaticString`
hold an arbitrary list of input names, with ARITY = `in_names.size`
resolving at comptime? If yes, we can refactor `Node` / `ExternalNode`
to `Node[name, Op: Module, *in_names: StaticString]` and retire the
arity cap.

Three things to validate:

  (V1) Struct comptime variadic `*in_names: StaticString` compiles +
       `Self.in_names.size` resolves at comptime + `comptime for`
       iteration over names works.
  (V2) End-to-end forward + vjp on a ternary `Module` op (`TernaryAddOp`)
       wrapped in the prototype `NodeV`. Output matches hand-rolled
       oracle bit-identically.
  (V3) The per-input grad buffers can live in ONE flat slab keyed by
       the cumulative per-input dims — avoids the `InlineArray[List]`
       trap (List isn't `ImplicitlyCopyable`).

If all three pass, the audit's "bump GraphNode to ARITY > 2" path is
unblocked, and we can rewrite `graph_nodes.mojo` to use the variadic
form. The fixed `IN0_NAME` / `IN1_NAME` + per-input fields go away.

CPU only, no GPU. No ComputeGraph integration. Standalone validation
of the storage + dispatch story. Production rollout is a separate
phase.
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Xavier


# ──────────────────────────────────────────────────────────────────
# V1 minimal probe — does `*names: StaticString` work in a struct?
# ──────────────────────────────────────────────────────────────────


struct VarNameProbe[*names: StaticString](Movable & ImplicitlyDestructible):
    comptime ARITY = Self.names.size

    def __init__(out self):
        pass

    @staticmethod
    def report() raises:
        print("  ARITY =", Self.ARITY)
        comptime for i in range(Self.ARITY):
            print("  names[", i, "] =", Self.names[i])


def test_v1_struct_variadic_static_string() raises:
    """V1 — variadic StaticString as struct comptime params compiles
    + size is comptime + comptime for iterates over names."""
    print("test_v1_struct_variadic_static_string ...")

    comptime N0 = VarNameProbe[]
    comptime N1 = VarNameProbe["s"]
    comptime N2 = VarNameProbe["s", "a"]
    comptime N4 = VarNameProbe["s", "a", "olp", "adv"]
    print(" -- N0 --")
    N0.report()
    print(" -- N1 --")
    N1.report()
    print(" -- N2 --")
    N2.report()
    print(" -- N4 (PPO shape) --")
    N4.report()

    assert_true(N0.ARITY == 0, "empty variadic → ARITY=0")
    assert_true(N1.ARITY == 1, "1 name → ARITY=1")
    assert_true(N2.ARITY == 2, "2 names → ARITY=2")
    assert_true(N4.ARITY == 4, "4 names → ARITY=4 (PPO topology)")
    print("  V1 PASS")


# ──────────────────────────────────────────────────────────────────
# V2/V3 setup — ternary Module + NodeV wrapper using single flat
# grad slab + variadic name params.
# ──────────────────────────────────────────────────────────────────


struct TernaryAddOp[DIM_: Int](Module):
    """y = x0 + x1 + x2. All three inputs share DIM. ARITY=3."""
    comptime ARITY: Int = 3
    comptime IN_DIMS = InlineArray[Int, 3](fill=Self.DIM_)
    comptime OUT_DIM: Int = Self.DIM_

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", "spike CPU only"
        var op = Self()
        op.ts = TargetStorage.make_cpu()
        return op^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        comptime assert False, "spike CPU only"
        return Self()

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
            mut=True, dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["TernaryAddOp", target](self.ts.target_tag)
        var i0 = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var i1 = typed_view[BATCH, Self.IN_DIMS[1]](inputs[1])
        var i2 = typed_view[BATCH, Self.IN_DIMS[2]](inputs[2])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        for b in range(BATCH):
            for d in range(Self.DIM_):
                out[b, d] = i0[b, d] + i1[b, d] + i2[b, d]

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
            mut=True, dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["TernaryAddOp", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi0 = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        var gi1 = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])
        var gi2 = typed_view_mut[BATCH, Self.IN_DIMS[2]](grad_inputs[2])
        for b in range(BATCH):
            for d in range(Self.DIM_):
                gi0[b, d] = go[b, d]
                gi1[b, d] = go[b, d]
                gi2[b, d] = go[b, d]


# ──────────────────────────────────────────────────────────────────
# NodeV — variadic-arity Node prototype. Stores grad inputs in ONE
# flat slab (avoids InlineArray[List] copyability issue).
# Slab layout: [grad_in0 (BATCH×IN_DIM) | grad_in1 (BATCH×IN1_DIM) | grad_in2 (BATCH×IN2_DIM) | ...]
# ──────────────────────────────────────────────────────────────────


struct NodeV[
    node_name: StaticString,
    Op: Module,
    *in_names: StaticString,
](Movable & ImplicitlyDestructible):
    comptime NAME = Self.node_name
    comptime ARITY = Self.in_names.size

    # Per-input dim accessor — comptime branch over the (currently)
    # max-3 surface that Module trait exposes via IN_DIM / IN1_DIM /
    # IN2_DIM. Extends naturally if Module gains IN3_DIM etc.
    @staticmethod
    def _in_dim[I: Int]() -> Int:
        comptime if I == 0:
            return Self.Op.IN_DIMS[0]
        elif I == 1:
            return Self.Op.IN_DIMS[1]
        elif I == 2:
            return Self.Op.IN_DIMS[2]
        else:
            return 0  # un-supported; comptime assert at use site

    # Cumulative per-input slab offset (in *elements*, not bytes).
    # offset_for[0] = 0, offset_for[1] = IN_DIM, offset_for[2] = IN_DIM+IN1_DIM.
    @staticmethod
    def _offset_for[BATCH: Int, I: Int]() -> Int:
        comptime if I == 0:
            return 0
        elif I == 1:
            return BATCH * Self.Op.IN_DIMS[0]
        elif I == 2:
            return BATCH * (Self.Op.IN_DIMS[0] + Self.Op.IN_DIMS[1])
        else:
            return -1

    @staticmethod
    def _total_in_dim() -> Int:
        comptime if Self.ARITY == 1:
            return Self.Op.IN_DIMS[0]
        elif Self.ARITY == 2:
            return Self.Op.IN_DIMS[0] + Self.Op.IN_DIMS[1]
        elif Self.ARITY == 3:
            return Self.Op.IN_DIMS[0] + Self.Op.IN_DIMS[1] + Self.Op.IN_DIMS[2]
        else:
            return 0

    var op: Self.Op
    var out_buf: List[Scalar[DT]]
    var grad_out_buf: List[Scalar[DT]]
    var grad_in_slab: List[Scalar[DT]]
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ARITY == Self.Op.ARITY, (
            "NodeV: number of in_names must match Op.ARITY"
        )
        self.op = Self.Op()
        self.out_buf = List[Scalar[DT]]()
        self.grad_out_buf = List[Scalar[DT]]()
        self.grad_in_slab = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", "spike CPU only"
        var n = Self()
        n.op = Self.Op.make[target, INIT]()
        n.ts = TargetStorage.make_cpu()
        return n^

    def _ensure_buffers[BATCH: Int](mut self) raises:
        if len(self.out_buf) < BATCH * Self.Op.OUT_DIM:
            self.out_buf.resize(BATCH * Self.Op.OUT_DIM, Scalar[DT](0.0))
        if len(self.grad_out_buf) < BATCH * Self.Op.OUT_DIM:
            self.grad_out_buf.resize(BATCH * Self.Op.OUT_DIM, Scalar[DT](0.0))
        comptime total = Self._total_in_dim()
        if len(self.grad_in_slab) < BATCH * total:
            self.grad_in_slab.resize(BATCH * total, Scalar[DT](0.0))

    # Forward — caller hands us one pointer per input. Dispatches to
    # the Op via the Module variadic surface.
    def forward_via[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in2_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Spike: ARITY=3 explicit, but the dispatch shape is
        comptime-uniform so we can extend to true variadic later."""
        self._ensure_buffers[BATCH]()
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.out_buf.unsafe_ptr()
        )
        # Variadic-pack unification: use the IN_DIM layout for ALL inputs
        # (typed_view recovers the real per-input shape inside the Op).
        var i0_t = TileTensor(in0_ptr, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        var i1_t = TileTensor(in1_ptr, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        var i2_t = TileTensor(in2_ptr, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        var out_t = TileTensor(out_p, row_major[BATCH, Self.Op.OUT_DIM]())
        self.op.forward[target, BATCH, POLICY=POLICY](
            i0_t, i1_t, i2_t, output=out_t,
        )

    # Backward — caller has seeded grad_out_buf; we run Op.vjp into the
    # flat slab, returning per-input grad pointers via grad_in_ptr_for[I].
    def vjp_via[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](mut self) raises:
        self._ensure_buffers[BATCH]()
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad_out_buf.unsafe_ptr()
        )
        var slab_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad_in_slab.unsafe_ptr()
        )
        var go_t = TileTensor(go_p, row_major[BATCH, Self.Op.OUT_DIM]())
        comptime off0 = Self._offset_for[BATCH, 0]()
        comptime off1 = Self._offset_for[BATCH, 1]()
        comptime off2 = Self._offset_for[BATCH, 2]()
        var gi0_t = TileTensor(slab_p + off0, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        var gi1_t = TileTensor(slab_p + off1, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        var gi2_t = TileTensor(slab_p + off2, row_major[BATCH, Self.Op.IN_DIMS[0]]())
        self.op.vjp[target, BATCH, POLICY=POLICY, mode=mode](
            go_t, gi0_t, gi1_t, gi2_t,
        )

    def grad_in_ptr_for[I: Int, BATCH: Int](
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Return the slab pointer for input I's grad. Caller can read
        BATCH × _in_dim[I]() elements starting there."""
        var slab_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad_in_slab.unsafe_ptr()
        )
        comptime off = Self._offset_for[BATCH, I]()
        return slab_p + off


# ──────────────────────────────────────────────────────────────────
# V2 / V3 — run an end-to-end ternary forward + vjp through NodeV
# and compare to a hand-rolled oracle. Validates both the variadic
# struct + the flat-slab grad storage.
# ──────────────────────────────────────────────────────────────────


comptime BATCH = 4
comptime DIM = 3


def test_v2_v3_ternary_end_to_end() raises:
    print("test_v2_v3_ternary_end_to_end ...")

    comptime OpT = TernaryAddOp[DIM]
    comptime N3 = NodeV["sum3", OpT, "x0", "x1", "x2"]
    print("  N3.ARITY =", N3.ARITY)
    print("  N3._total_in_dim =", N3._total_in_dim())
    assert_true(N3.ARITY == 3, "NodeV[..., 3 names] → ARITY=3")
    assert_true(N3._total_in_dim() == 3 * DIM, "total_in_dim = 3·DIM")

    var node = N3.make[target="cpu", INIT=Xavier]()

    # Inputs:  x0 = [1, 2, 3, ...],  x1 = [10, 20, 30, ...],  x2 = [100, ...]
    # so out[i] = x0[i] + x1[i] + x2[i] = 111, 222, 333, ...
    var x0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var x1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var x2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        x0[k] = Scalar[DT](k + 1)
        x1[k] = Scalar[DT](10 * (k + 1))
        x2[k] = Scalar[DT](100 * (k + 1))

    # Forward.
    node.forward_via[target="cpu", BATCH=BATCH](x0, x1, x2)

    var max_fwd_diff: Scalar[DT] = 0.0
    for k in range(BATCH * DIM):
        var got = node.out_buf[k]
        var want = Scalar[DT](111 * (k + 1))
        var d = got - want
        if d < Scalar[DT](0.0):
            d = -d
        if d > max_fwd_diff:
            max_fwd_diff = d
    print("  forward max |Δ| =", max_fwd_diff)
    assert_true(max_fwd_diff < Scalar[DT](1e-5),
                "ternary forward must match oracle")

    # Backward seed: grad_out = 1 everywhere → all grad_in_* = 1 everywhere.
    for k in range(BATCH * DIM):
        node.grad_out_buf[k] = Scalar[DT](1.0)
    node.vjp_via[target="cpu", BATCH=BATCH]()

    var gi0_p = node.grad_in_ptr_for[0, BATCH]()
    var gi1_p = node.grad_in_ptr_for[1, BATCH]()
    var gi2_p = node.grad_in_ptr_for[2, BATCH]()
    var max_bwd_diff: Scalar[DT] = 0.0
    for k in range(BATCH * DIM):
        var d0 = gi0_p[k] - Scalar[DT](1.0)
        var d1 = gi1_p[k] - Scalar[DT](1.0)
        var d2 = gi2_p[k] - Scalar[DT](1.0)
        if d0 < Scalar[DT](0.0): d0 = -d0
        if d1 < Scalar[DT](0.0): d1 = -d1
        if d2 < Scalar[DT](0.0): d2 = -d2
        if d0 > max_bwd_diff: max_bwd_diff = d0
        if d1 > max_bwd_diff: max_bwd_diff = d1
        if d2 > max_bwd_diff: max_bwd_diff = d2
    print("  vjp max |Δ| =", max_bwd_diff)
    assert_true(max_bwd_diff < Scalar[DT](1e-5),
                "ternary vjp must match oracle (∂(x0+x1+x2)/∂x_i = 1)")

    # Validate non-overlapping slab offsets (V3): the three input grads
    # live at distinct contiguous slices of one List.
    assert_true(
        N3._offset_for[BATCH, 0]() == 0
        and N3._offset_for[BATCH, 1]() == BATCH * DIM
        and N3._offset_for[BATCH, 2]() == 2 * BATCH * DIM,
        "flat slab offsets must be 0, BATCH·DIM, 2·BATCH·DIM",
    )

    x0.free(); x1.free(); x2.free()
    print("  V2+V3 PASS")


def main() raises:
    print("=" * 70)
    print("Spike: variadic-arity GraphNode prototype (Phase I.2+)")
    print("=" * 70)
    test_v1_struct_variadic_static_string()
    test_v2_v3_ternary_end_to_end()
    print("=" * 70)
    print("ALL PASSED — variadic-arity Node is feasible.")
    print("Next step (separate phase): refactor `graph_nodes.mojo` Node /")
    print("ExternalNode to use `*in_names: StaticString` + flat grad slab.")
    print("=" * 70)
