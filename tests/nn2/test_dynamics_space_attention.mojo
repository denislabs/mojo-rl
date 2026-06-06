"""DynamicsSpaceAttention — dynamics-layout space-attention leaf (Phase 3.2).

    pixi run            mojo run -I . tests/nn2/test_dynamics_space_attention.mojo  # CPU
    pixi run -e apple   mojo run -I . tests/nn2/test_dynamics_space_attention.mojo  # +GPU

Two decisive checks for the dynamics agent-token space attention:

  1. PARITY (NAGENT>0): the wrapper must be BIT-IDENTICAL (forward + vjp, CPU
     and GPU) to a bare MaskedAttention with the dynamics `wm_agent_bc` mask
     installed by hand — proving the mask-build-in-make + delegation is right.
  2. COLLAPSE (NAGENT=0): with no agent tokens the `wm_agent_bc` mask must
     reduce to full mixing — bit-identical to a bare MaskedAttention with the
     DEFAULT (all-allow) mask. This is what makes the agent-capable dynamics
     byte-identical to the unconditional one when NAGENT=0.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.masked_attention import (
    MaskedAttention,
    build_modality_mask,
)
from mojo_rl.nn2.primitives.dynamics_space_attention import (
    DynamicsSpaceAttention,
    DYN_MOD_AGENT,
)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def _dyn_ids[NSP: Int, NREG: Int, NAGENT: Int]() -> List[Int]:
    var ids = List[Int]()
    ids.append(0)
    ids.append(1)
    ids.append(2)
    for _ in range(NSP):
        ids.append(3)
    for _ in range(NREG):
        ids.append(4)
    for _ in range(NAGENT):
        ids.append(5)
    return ids^


def test_cpu_parity[
    D: Int, NH: Int, NSP: Int, NREG: Int, NAGENT: Int, USE_AGENT_MASK: Bool
](name: String) raises:
    print(name, "(cpu) ...")
    comptime S = 3 + NSP + NREG + NAGENT
    comptime BATCH = 2
    comptime IN_N = BATCH * S * D * 3
    comptime OUT_N = BATCH * S * D

    var wrap = DynamicsSpaceAttention[
        D, NH, NSP, NREG, NAGENT, "wm_agent_bc"
    ].make[target="cpu", INIT=Zero]()
    var bare = MaskedAttention[D, NH, S].make[target="cpu", INIT=Zero]()
    comptime if USE_AGENT_MASK:
        bare.set_mask(
            build_modality_mask["wm_agent_bc"](
                _dyn_ids[NSP, NREG, NAGENT](), 0, agent_mod_in=DYN_MOD_AGENT
            )
        )
    # else: leave the bare op's DEFAULT all-allow mask (full mixing) installed.

    var x = _alloc(IN_N)
    var go = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.7)
    for i in range(OUT_N):
        go[i] = _spread(i, 0.9)
    var xt = TileTensor(x, row_major[BATCH, S * D * 3]())
    var got = TileTensor(go, row_major[BATCH, S * D]())

    var yw = _alloc(OUT_N)
    var yb = _alloc(OUT_N)
    var giw = _alloc(IN_N)
    var gib = _alloc(IN_N)
    var ywt = TileTensor(yw, row_major[BATCH, S * D]())
    var ybt = TileTensor(yb, row_major[BATCH, S * D]())
    var giwt = TileTensor(giw, row_major[BATCH, S * D * 3]())
    var gibt = TileTensor(gib, row_major[BATCH, S * D * 3]())
    wrap.forward["cpu", BATCH](xt, output=ywt)
    bare.forward["cpu", BATCH](xt, output=ybt)
    wrap.vjp["cpu", BATCH](got, giwt)
    bare.vjp["cpu", BATCH](got, gibt)

    var mf = _maxdiff(yw, yb, OUT_N)
    var mb = _maxdiff(giw, gib, IN_N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf == 0.0 and mb == 0.0, name + ": cpu parity (must be exact)")
    print("  ok")


def test_gpu_parity[
    D: Int, NH: Int, NSP: Int, NREG: Int, NAGENT: Int
](ctx: DeviceContext, name: String) raises:
    print(name, "(gpu) ...")
    comptime S = 3 + NSP + NREG + NAGENT
    comptime BATCH = 2
    comptime IN_N = BATCH * S * D * 3
    comptime OUT_N = BATCH * S * D

    var wrap = DynamicsSpaceAttention[
        D, NH, NSP, NREG, NAGENT, "wm_agent_bc"
    ].make[target="gpu", INIT=Zero](ctx)
    var bare = MaskedAttention[D, NH, S].make[target="gpu", INIT=Zero](ctx)
    bare.set_mask(
        build_modality_mask["wm_agent_bc"](
            _dyn_ids[NSP, NREG, NAGENT](), 0, agent_mod_in=DYN_MOD_AGENT
        )
    )

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var ywh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var ybh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var giwh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gibh = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.7)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 0.9)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var ywd = ctx.enqueue_create_buffer[DT](OUT_N)
    var ybd = ctx.enqueue_create_buffer[DT](OUT_N)
    var giwd = ctx.enqueue_create_buffer[DT](IN_N)
    var gibd = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, S * D * 3]())
    var got = TileTensor(_mao(god), row_major[BATCH, S * D]())
    var ywt = TileTensor(_mao(ywd), row_major[BATCH, S * D]())
    var ybt = TileTensor(_mao(ybd), row_major[BATCH, S * D]())
    var giwt = TileTensor(_mao(giwd), row_major[BATCH, S * D * 3]())
    var gibt = TileTensor(_mao(gibd), row_major[BATCH, S * D * 3]())
    wrap.forward["gpu", BATCH](xt, output=ywt)
    bare.forward["gpu", BATCH](xt, output=ybt)
    wrap.vjp["gpu", BATCH](got, giwt)
    bare.vjp["gpu", BATCH](got, gibt)
    ctx.enqueue_copy(ywh, ywd)
    ctx.enqueue_copy(ybh, ybd)
    ctx.enqueue_copy(giwh, giwd)
    ctx.enqueue_copy(gibh, gibd)
    ctx.synchronize()

    var mf = _maxdiff(ywh.unsafe_ptr(), ybh.unsafe_ptr(), OUT_N)
    var mb = _maxdiff(giwh.unsafe_ptr(), gibh.unsafe_ptr(), IN_N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf == 0.0 and mb == 0.0, name + ": gpu parity (must be exact)")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("DynamicsSpaceAttention — leaf parity (Phase 3.2)")
    print("=" * 70)
    # NAGENT>0: parity vs hand-installed dynamics BC mask.
    test_cpu_parity[4, 2, 4, 2, 1, True]("bc_agent1")
    test_cpu_parity[4, 2, 3, 1, 2, True]("bc_agent2")
    # NAGENT=0: must collapse to full mixing (= default all-allow mask).
    test_cpu_parity[4, 2, 4, 2, 0, False]("bc_noagent_collapse")
    var ctx = DeviceContext()
    test_gpu_parity[4, 2, 4, 2, 1](ctx, "bc_agent1")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
