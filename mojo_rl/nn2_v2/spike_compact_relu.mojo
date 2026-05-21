"""Compact ReLU using the boilerplate-reduction patterns from
spike_reflection_probe.mojo. Measures the actual LOC win vs.
nn2/primitives/relu.mojo (290 LOC).

Patterns applied:

  1. `@fieldwise_init` auto-synthesizes `__init__` from field decls.
  2. Module-level free `assert_tag_for[name, target]` instead of a
     per-struct `_assert_tag` method.
  3. `TargetStorage` bundles `(_target_tag, _inference, ctx)` into one
     field — three fields collapsed to one, default init centralized.
  4. Module-level free `ensure_cpu_buffer` / `ensure_gpu_buffer`
     helpers instead of per-struct `_ensure_cache_*`.
  5. Unified-buffer design (Follow-up #1) removes the cache fields
     entirely on input-caching layers: ReLU stores only a pointer alias.

NOT applied (would require Mojo nightly features that don't exist):

  - Field-iteration codegen to auto-derive `for_each_param`. None of
    the probes succeeded for this; `for_each_param` stays manual
    (mechanical, ~4 lines per Linear).
  - Auto-generated `make[target, INIT]` factory. The factory body is
    layer-specific (Linear initializes weight + bias; LayerNorm
    initializes γ=1, β=0; ReLU is parameterless) so a single template
    won't fit. Doable as a code-gen macro but Mojo macros are not
    nightly-stable.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2_v2.spike_reflection_probe import TargetStorage, assert_tag_for


comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────
# Shared free helpers (Pattern 4).
# Today's nn2 has 22 copies of essentially this code.
# ──────────────────────────────────────────────────────────────────────


def ensure_cpu_buffer(mut buf: List[Scalar[DT]], needed: Int):
    if len(buf) < needed:
        buf.resize(needed, Scalar[DT](0.0))


def ensure_gpu_buffer(
    mut buf: Optional[DeviceBuffer[DT]],
    mut cap: Int,
    needed: Int,
    ctx: DeviceContext,
) raises:
    if cap < needed:
        buf = ctx.enqueue_create_buffer[DT](needed)
        cap = needed


# ──────────────────────────────────────────────────────────────────────
# CompactReLU — all the techniques above + unified-buffer design.
# ──────────────────────────────────────────────────────────────────────


struct CompactReLU[DIM: Int](Defaultable & Movable & ImplicitlyDestructible):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    # Pattern 3: one `ts` field collapses the 3-field cluster.
    var ts: TargetStorage
    # Pattern 5 (unified-buffer): pointer alias, no owned cache.
    var cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.ts = TargetStorage.make_default()
        self.cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", "Compact spike supports CPU only"
        var r = Self()
        r.ts.target_tag = Int8(1)
        return r^

    def forward[target: StaticString, BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        # Pattern 2: one-line tag check, no per-struct method.
        assert_tag_for["CompactReLU", target](self.ts.target_tag)
        self.cached_input_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](input.ptr)
        for b in range(BATCH):
            for d in range(Self.DIM):
                var x = input[b, d]
                output[b, d] = x if x > Scalar[DT](0.0) else Scalar[DT](0.0)

    def backward[target: StaticString, BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input rank-2"
        assert_tag_for["CompactReLU", target](self.ts.target_tag)
        var cache = TileTensor(
            self.cached_input_ptr, row_major[BATCH, Self.DIM]()
        )
        for b in range(BATCH):
            for d in range(Self.DIM):
                grad_input[b, d] = (
                    grad_output[b, d]
                    if cache[b, d] > Scalar[DT](0.0)
                    else Scalar[DT](0.0)
                )

    def set_inference(mut self, value: Bool):
        self.ts.inference = value


# ──────────────────────────────────────────────────────────────────────
# Quick smoke test.
# ──────────────────────────────────────────────────────────────────────


from std.memory import alloc


def main() raises:
    var r = CompactReLU[3].make[target="cpu"]()

    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](6)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](6)
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](6)
    var gi_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](6)

    in_p[0] = -1.0;  in_p[1] = 0.5;  in_p[2] = 2.0
    in_p[3] = 1.0;   in_p[4] = -0.5; in_p[5] = -3.0

    var in_t  = TileTensor(in_p,  row_major[2, 3]())
    var out_t = TileTensor(out_p, row_major[2, 3]())
    r.forward["cpu", 2](in_t, out_t)
    print("out =", out_p[0], out_p[1], out_p[2], out_p[3], out_p[4], out_p[5])
    # Expect: 0.0, 0.5, 2.0, 1.0, 0.0, 0.0

    for k in range(6):
        go_p[k] = 1.0
    var go_t = TileTensor(go_p, row_major[2, 3]())
    var gi_t = TileTensor(gi_p, row_major[2, 3]())
    r.backward["cpu", 2](go_t, gi_t)
    print("grad =", gi_p[0], gi_p[1], gi_p[2], gi_p[3], gi_p[4], gi_p[5])
    # Expect: 0.0, 1.0, 1.0, 1.0, 0.0, 0.0

    var expected = List[Scalar[DT]]()
    expected.append(0.0)
    expected.append(0.5)
    expected.append(2.0)
    expected.append(1.0)
    expected.append(0.0)
    expected.append(0.0)
    var ok_fwd = True
    for k in range(6):
        if out_p[k] != expected[k]:
            ok_fwd = False
    var grad_expected = List[Scalar[DT]]()
    grad_expected.append(0.0)
    grad_expected.append(1.0)
    grad_expected.append(1.0)
    grad_expected.append(1.0)
    grad_expected.append(0.0)
    grad_expected.append(0.0)
    var ok_bwd = True
    for k in range(6):
        if gi_p[k] != grad_expected[k]:
            ok_bwd = False

    if ok_fwd and ok_bwd:
        print("PASS — compact ReLU forward + backward match expected")
    else:
        print("FAIL — compact ReLU output mismatch")
        raise Error("compact ReLU smoke test failed")

    in_p.free()
    out_p.free()
    go_p.free()
    gi_p.free()
