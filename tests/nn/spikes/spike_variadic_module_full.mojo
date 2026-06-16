"""Three follow-up probes for the variadic-Module refactor (post-I.2.5).

Q3. Heterogeneous-dim leaf body: can a leaf's `forward` do
    `comptime for k in range(ARITY): typed_view[BATCH, Self.IN_DIMS[k]](inputs[k])`
    when inputs[k] have different real dims? Validates the leaf-side
    comptime iteration shape.

Q4. ARITY ≥ 5 sanity. Does Mojo handle `InlineArray[Int, N]` at N=8 +
    `comptime for k in range(8)` over a variadic Module? Confirms there's
    no hidden ARITY ceiling we'd hit when DreamerV3 imagination losses
    land with 5-7 inputs.

Q5. `List[List[Scalar[DT]]]` as a struct field replacing the per-input
    grad buf fields (`_grad_in0_buf`, ..., `_grad_in3_buf`). InlineArray
    won't take List (not ImplicitlyCopyable); does List-of-List work?
    Tests: construction, lazy-grow each inner List, read .unsafe_ptr()
    for each.
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import typed_view, typed_view_mut


# ──────────────────────────────────────────────────────────────────
# Q3 — heterogeneous-dim leaf body via comptime for + IN_DIMS[k]
# ──────────────────────────────────────────────────────────────────


struct HeteroLeaf[*DIMS: Int](Movable & ImplicitlyDestructible):
    """3-input op: takes inputs of dims DIMS[0..3] and produces a
    [BATCH, 1] output = sum of all inputs[k][b, 0] across k."""
    comptime ARITY = Self.DIMS.size
    comptime IN_DIMS: InlineArray[Int, Self.ARITY] = Self._build()
    comptime OUT_DIM = 1

    @staticmethod
    def _build() -> InlineArray[Int, Self.ARITY]:
        var d = InlineArray[Int, Self.ARITY](fill=0)
        comptime for k in range(Self.ARITY):
            d[k] = Self.DIMS[k]
        return d

    def __init__(out self):
        pass

    def forward[BATCH: Int](
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
        var out_v = typed_view_mut[BATCH, 1](output)
        # Zero the output.
        for b in range(BATCH):
            out_v[b, 0] = Scalar[DT](0.0)
        # Heterogeneous per-input dims via comptime for.
        comptime for k in range(Self.ARITY):
            comptime dim_k = Self.IN_DIMS[k]
            var in_k = typed_view[BATCH, dim_k](inputs[k])
            for b in range(BATCH):
                # Sum first element of each input row to validate
                # the typed_view recovered the right shape.
                out_v[b, 0] = out_v[b, 0] + in_k[b, 0]


def test_q3_hetero_leaf() raises:
    print("Q3: heterogeneous-dim leaf body ...")
    comptime BATCH = 2
    var i0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2)
    var i1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 5)
    var i2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 1)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 1)

    # i0[b, 0] = 10, i1[b, 0] = 100, i2[b, 0] = 1000.
    for b in range(BATCH):
        i0[b * 2] = Scalar[DT](10.0)
        i1[b * 5] = Scalar[DT](100.0)
        i2[b * 1] = Scalar[DT](1000.0)
    # Hetero-variadic: pack all with IN_DIMS[0]-shaped Layout (=2) so
    # the variadic unifies; typed_view recovers real shape inside.
    var i0_t = TileTensor(i0, row_major[BATCH, 2]())
    var i1_t = TileTensor(i1, row_major[BATCH, 2]())
    var i2_t = TileTensor(i2, row_major[BATCH, 2]())
    var out_t = TileTensor(out, row_major[BATCH, 1]())

    var leaf = HeteroLeaf[2, 5, 1]()
    leaf.forward[BATCH](i0_t, i1_t, i2_t, output=out_t)
    print("  out[0] =", out[0], " (want 1110.0 = 10 + 100 + 1000)")
    print("  out[1] =", out[1], " (want 1110.0)")
    var ok = out[0] == Scalar[DT](1110.0) and out[1] == Scalar[DT](1110.0)
    print("  Q3:", "PASS" if ok else "FAIL")
    i0.free(); i1.free(); i2.free(); out.free()


# ──────────────────────────────────────────────────────────────────
# Q4 — ARITY = 8 sanity (InlineArray + comptime for at higher arity)
# ──────────────────────────────────────────────────────────────────


struct Arity8Probe[*DIMS: Int](Movable & ImplicitlyDestructible):
    comptime ARITY = Self.DIMS.size
    comptime IN_DIMS: InlineArray[Int, Self.ARITY] = Self._build()

    @staticmethod
    def _build() -> InlineArray[Int, Self.ARITY]:
        var d = InlineArray[Int, Self.ARITY](fill=0)
        comptime for k in range(Self.ARITY):
            d[k] = Self.DIMS[k]
        return d

    def __init__(out self):
        pass

    @staticmethod
    def report():
        print("  ARITY =", Self.ARITY)
        comptime for k in range(Self.ARITY):
            print("    IN_DIMS[", k, "] =", Self.IN_DIMS[k])


def test_q4_arity_8() raises:
    print("Q4: ARITY=8 ...")
    comptime A8 = Arity8Probe[3, 5, 7, 11, 13, 17, 19, 23]
    A8.report()
    print("  Q4: PASS (compiled + iterated 8 entries)")


# ──────────────────────────────────────────────────────────────────
# Q5 — List[List[Scalar[DT]]] as the consolidated grad-buf field
# ──────────────────────────────────────────────────────────────────


struct ListOfListProbe(Movable & ImplicitlyDestructible):
    """Replaces 4 per-input grad-buf fields with one List[List[...]].
    Tests: ctor → lazy-grow each inner List → read .unsafe_ptr() for each."""
    var _grad_ins_buf: List[List[Scalar[DT]]]

    def __init__(out self):
        # Pre-populate with 4 empty inner Lists (one per input slot).
        self._grad_ins_buf = List[List[Scalar[DT]]]()
        for _ in range(4):
            self._grad_ins_buf.append(List[Scalar[DT]]())

    def ensure[ARITY: Int, BATCH: Int](
        mut self, in_dims: InlineArray[Int, ARITY],
    ) raises:
        comptime for k in range(ARITY):
            var want = BATCH * in_dims[k]
            if len(self._grad_ins_buf[k]) < want:
                self._grad_ins_buf[k].resize(want, Scalar[DT](0.0))

    def write_marker[ARITY: Int, BATCH: Int](
        mut self, in_dims: InlineArray[Int, ARITY],
    ):
        """Write a per-input marker into each inner buffer's [0] slot
        so we can verify the buffers are distinct (not aliased)."""
        comptime for k in range(ARITY):
            self._grad_ins_buf[k][0] = Scalar[DT](100 * (k + 1))

    def read_marker(self, k: Int) -> Scalar[DT]:
        return self._grad_ins_buf[k][0]

    def get_ptr(mut self, k: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._grad_ins_buf[k].unsafe_ptr()
        )


def _build_ppo_dims() -> InlineArray[Int, 4]:
    """Heterogeneous [6, 3, 1, 1] dims (PPOObjective shape)."""
    var d = InlineArray[Int, 4](fill=0)
    d[0] = 6
    d[1] = 3
    d[2] = 1
    d[3] = 1
    return d


def test_q5_list_of_list() raises:
    print("Q5: List[List[Scalar[DT]]] as grad-bufs field ...")
    var probe = ListOfListProbe()
    var in_dims = _build_ppo_dims()
    probe.ensure[4, 8](in_dims)
    probe.write_marker[4, 8](in_dims)
    print("  inner[0][0] =", probe.read_marker(0), " (want 100)")
    print("  inner[1][0] =", probe.read_marker(1), " (want 200)")
    print("  inner[2][0] =", probe.read_marker(2), " (want 300)")
    print("  inner[3][0] =", probe.read_marker(3), " (want 400)")
    # Verify pointers are distinct.
    var p0 = probe.get_ptr(0)
    var p1 = probe.get_ptr(1)
    var p2 = probe.get_ptr(2)
    var p3 = probe.get_ptr(3)
    var distinct = (p0 != p1) and (p1 != p2) and (p2 != p3) and (p0 != p3)
    print("  pointers distinct:", distinct)
    var ok = (
        probe.read_marker(0) == Scalar[DT](100)
        and probe.read_marker(1) == Scalar[DT](200)
        and probe.read_marker(2) == Scalar[DT](300)
        and probe.read_marker(3) == Scalar[DT](400)
        and distinct
    )
    print("  Q5:", "PASS" if ok else "FAIL")


def main() raises:
    print("=" * 70)
    print("Variadic-Module follow-up probes (Q3 + Q4 + Q5)")
    print("=" * 70)
    test_q3_hetero_leaf()
    print()
    test_q4_arity_8()
    print()
    test_q5_list_of_list()
    print("=" * 70)
