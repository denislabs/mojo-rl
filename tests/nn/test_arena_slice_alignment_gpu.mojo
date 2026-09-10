"""Every arena param slice starts on a 16-byte boundary — the gate for the
misaligned-GEMM launch failure.

MAX's `multistage_gemm` loads its A/B operands with 16-byte (float4) vectors.
`ParamArena` used to pack params at ELEMENT granularity, so one param of odd
size shifted every later param off the boundary and the first Linear whose
weight landed there faulted on its very first tile load:

    Invalid __global__ read of size 16 bytes ... Access at 0x...078 is misaligned
    CUDA call failed: CUDA_ERROR_LAUNCH_FAILED (unspecified launch failure)

That is how the BFM-Zero G1 actor died on the 5090 (docs §12.5): obs 527 gave
the tower two odd-width norm params, which put the branch-A `Linear[1024,512]`
weight at element 543774 — 2 mod 4, i.e. 8 bytes off.

Checked here, on a net built to reproduce the shape:
1. every param slice's offset into the arena is `PARAM_ALIGN`-aligned, and its
   absolute device pointer is 16-byte aligned (what the GEMM actually sees);
2. the `m`/`v` slices land on the SAME offsets as `val`/`grd` — the two offset
   walks (`ParamArena.visit`, `adam._MomentPlacer.visit`) must not drift, or a
   param's moments silently alias a DIFFERENT param's values;
3. NON-VACUITY: the unaligned packing this replaced is recomputed here and
   asserted to have misaligned at least one param, so the gate would FAIL on
   the old code rather than pass on an empty walk.

Run: pixi run -e apple mojo run -I . tests/nn/test_arena_slice_alignment_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.param_arena import PARAM_ALIGN


# Odd widths on purpose: W = 5*7 = 35 and b = 7 are both ≢ 0 mod 4, which is
# exactly what obs 527 did to the G1 actor.
comptime NET = Sequential[Linear[5, 7], Linear[7, 6], Linear[6, 4]]
comptime VEC = 16  # bytes: the GEMM's float4 operand load


struct _Offsets(ParamVisitor):
    var val_base: Int
    var m_base: Int
    var names: List[String]
    var sizes: List[Int]
    var val_off: List[Int]
    var grd_off: List[Int]
    var m_off: List[Int]
    var v_off: List[Int]
    var val_addr: List[Int]

    def __init__(out self, val_base: Int, m_base: Int):
        self.val_base = val_base
        self.m_base = m_base
        self.names = List[String]()
        self.sizes = List[Int]()
        self.val_off = List[Int]()
        self.grd_off = List[Int]()
        self.m_off = List[Int]()
        self.v_off = List[Int]()
        self.val_addr = List[Int]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var pa = Int(param.dev.value().unsafe_ptr())
        self.names.append(name)
        self.sizes.append(N)
        self.val_addr.append(pa)
        self.val_off.append(pa - self.val_base)
        self.grd_off.append(Int(grad.dev.value().unsafe_ptr()))
        self.m_off.append(Int(m.dev.value().unsafe_ptr()) - self.m_base)
        self.v_off.append(Int(v.dev.value().unsafe_ptr()))


def main() raises:
    var c = DeviceContext()
    print("Arena slice alignment (PARAM_ALIGN =", PARAM_ALIGN, "elements =", PARAM_ALIGN * 4, "bytes)")

    var net = NET.make["gpu", Deterministic](Optional(c))
    var opt = Adam(lr=1e-3)
    opt.adopt["gpu"](net, Optional(c))

    var o = _Offsets(
        Int(opt.arena.val.dev.value().unsafe_ptr()),
        Int(opt.m_arena.dev.value().unsafe_ptr()),
    )
    net.for_each_param["gpu"](o, Optional(c))

    var n = len(o.sizes)
    print("  params walked:", n)
    assert_true(n > 0, "vacuous: the param walk visited nothing")

    # ---- 3. non-vacuity: the OLD packing must have been misaligned -------
    var packed = 0
    var would_fault = 0
    for i in range(n):
        if (packed * 4) % VEC != 0:
            would_fault += 1
        packed += o.sizes[i]
    print("  params the element-packed layout misaligned:", would_fault, "of", n)
    assert_true(
        would_fault > 0,
        "vacuous gate: this net's param sizes are all 16-byte multiples, so"
        " element packing would have aligned them anyway — pick odd widths",
    )

    # ---- 1 + 2. the alignment and the two walks agreeing -----------------
    var bad_align = 0
    var bad_addr = 0
    var bad_moment = 0
    for i in range(n):
        if o.val_off[i] % (PARAM_ALIGN * 4) != 0:
            bad_align += 1
            print("    MISALIGNED slice:", o.names[i], "offset", o.val_off[i])
        if o.val_addr[i] % VEC != 0:
            bad_addr += 1
            print("    MISALIGNED device ptr:", o.names[i], "addr mod", VEC, "=", o.val_addr[i] % VEC)
        if o.m_off[i] != o.val_off[i]:
            bad_moment += 1
            print("    MOMENT DRIFT:", o.names[i], "val", o.val_off[i], "vs m", o.m_off[i])

    print("  misaligned slice offsets:", bad_align)
    print("  misaligned device pointers:", bad_addr)
    print("  m/v offsets disagreeing with val/grd:", bad_moment)

    assert_true(bad_align == 0, "some param slices are not PARAM_ALIGN-aligned")
    assert_true(bad_addr == 0, "some param device pointers are not 16-byte aligned")
    assert_true(bad_moment == 0, "the moment walk drifted from the value walk")
    print("PASS")
