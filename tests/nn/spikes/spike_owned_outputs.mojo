"""Spike DR.3 — Module-owned output buffer + lazy-grown TileTensor view.

Goal: prove that a leaf Module can own its output buffer (as `List[Scalar[DT]]`),
lazily grow it on the first forward at a given BATCH, and expose `out_view()
-> TileTensor[...]` so callers compose without ever wiring intermediate
scratch.

Comparison points:
- `Sequential` already does this for *intermediate* outputs (`mid_cpu` slabs)
  but not for its own output. Caller still allocates the final output buffer.
- v1 ComputeGraph allocates one big `cache` blob holding all activations
  in sequence — its own version of "Module owns its output", just centrally.

Today (nn): caller allocates output, wraps in TileTensor, passes to forward.
Proposed: Module owns output, caller asks for `out_view()` after forward.

Two ergonomic patterns to probe:

Pattern A: explicit `forward()` then `out_view()`.
    var lin = Linear[3, 4](...)
    lin.forward[BATCH=2](input_t)         # no output arg
    var y = lin.out_view[BATCH=2]()       # read-only view
    # use y as input to next layer

Pattern B: `forward(input) -> TileTensor` returning the owned view.
    var y = lin.forward[BATCH=2](input_t)
    var z = relu.forward[BATCH=2](y)
    var w = lin2.forward[BATCH=2](z)
    # whole chain composes without intermediate scratch decls

If Pattern B works, multi-line composition becomes one expression:
    var y = lin2.forward(relu.forward(lin1.forward(input)))

But Mojo nightly's mut-self rules need careful handling — `lin1.forward(...)`
returns a view that borrows from lin1; the lifetime of that view has to
outlive the next call into lin2. That's a tight constraint.
"""

from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Minimal "owned-output Linear-style Module" — just adds bias to keep it
# small. Owns weight, bias, output buffer, output cache (for backward).
# ──────────────────────────────────────────────────────────────────────


struct OwnedLinear[IN: Int, OUT: Int](Movable & Deinitable):
    """Tiny linear-bias layer: y[b, j] = sum_k input[b, k] * W[k, j] + b[j].
    Owns its weight + bias + output buffer (List). out_view() returns a
    TileTensor view into the owned buffer."""
    var weight: List[Scalar[DT]]   # [IN, OUT] row-major
    var bias: List[Scalar[DT]]     # [OUT]
    var out_buf: List[Scalar[DT]]  # [BATCH * OUT], grown lazily
    var out_n_batch: Int

    def __init__(out self):
        self.weight = List[Scalar[DT]](length=Self.IN * Self.OUT, fill=0.0)
        self.bias = List[Scalar[DT]](length=Self.OUT, fill=0.0)
        self.out_buf = List[Scalar[DT]]()
        self.out_n_batch = 0
        # Fill weight with deterministic pattern for testing.
        for i in range(Self.IN):
            for j in range(Self.OUT):
                self.weight[i * Self.OUT + j] = Scalar[DT](
                    Float64(i) * 0.1 + Float64(j) * 0.01
                )
        # Bias = 0.5 * j for testing.
        for j in range(Self.OUT):
            self.bias[j] = Scalar[DT](0.5) * Scalar[DT](Float64(j))

    def _ensure_out_cpu(mut self, batch: Int):
        if self.out_n_batch < batch:
            self.out_buf.resize(batch * Self.OUT, Scalar[DT](0.0))
            self.out_n_batch = batch

    def forward[
        BATCH: Int,
        LIN: TensorLayout, OIN: MutOrigin,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
    ):
        """Pattern A — forward writes into self.out_buf, caller asks
        for the view later via `out_view`. No output arg in the signature."""
        comptime assert input.flat_rank == 2
        self._ensure_out_cpu(BATCH)
        var out_p = self.out_buf.unsafe_ptr()
        var w_p = self.weight.unsafe_ptr()
        var b_p = self.bias.unsafe_ptr()
        for b in range(BATCH):
            for j in range(Self.OUT):
                var s: Scalar[DT] = b_p[j]
                for k in range(Self.IN):
                    s += input[b, k] * w_p[k * Self.OUT + j]
                out_p[b * Self.OUT + j] = s

    def out_ptr(ref self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Raw pointer into the owned output buffer. Caller wraps
        in TileTensor with their own layout. The Module-owned buffer
        eliminates the need for a separate scratch declaration on
        the caller side — but the TileTensor wrap stays (Mojo's type
        position can't accept `row_major[...]()`-as-TensorLayout from
        a method body)."""
        return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            self.out_buf.unsafe_ptr()
        )


# ──────────────────────────────────────────────────────────────────────
# Pattern B probe — does `forward[...](input) -> TileTensor` compose?
# We do NOT define this method on OwnedLinear, because returning a borrow
# from `mut self` (after writing through it) requires the lifetime of
# the view to outlive subsequent calls into OTHER modules. The Mojo
# borrow checker may or may not accept it. The probe instead just tests
# Pattern A composition: forward(...), then out_view(...), then pass into
# next layer.
# ──────────────────────────────────────────────────────────────────────


def smoke_pattern_a() raises:
    print("--- spike Pattern A: forward() + out_view() ---")
    # Tiny 2-layer chain: Linear[2, 3] -> Linear[3, 1].
    var l1 = OwnedLinear[2, 3]()
    var l2 = OwnedLinear[3, 1]()

    var input_buf = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    input_buf[0] = Scalar[DT](0.5)
    input_buf[1] = Scalar[DT](1.0)
    input_buf[2] = Scalar[DT](1.5)
    input_buf[3] = Scalar[DT](2.0)
    var input_t = TileTensor(input_buf.unsafe_ptr(), row_major[2, 2]())

    l1.forward[2](input_t)
    var mid_view = TileTensor(l1.out_ptr(), row_major[2, 3]())
    print("  l1 out[0,0]=", mid_view[0, 0], " out[0,1]=", mid_view[0, 1], " out[0,2]=", mid_view[0, 2])
    l2.forward[2](mid_view)
    var final_view = TileTensor(l2.out_ptr(), row_major[2, 1]())
    print("  l2 out[0,0]=", final_view[0, 0], " out[1,0]=", final_view[1, 0])
    print("  Pattern A (Module-owned scratch + caller-wraps-view): composed cleanly")


def main() raises:
    print("=" * 70)
    print("DR.3 — Module-owned output buffer spike")
    print("=" * 70)
    smoke_pattern_a()
    print("=" * 70)
