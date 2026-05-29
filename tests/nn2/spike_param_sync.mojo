"""SPIKE (PR5c Step 4): name-keyed param sync WMCoreGraph → WMImagineGraph.

Two graphs init'd with DIFFERENT random params. After `sync_params`, the
imagine graph's core (`nd`) must match the core graph's `nd` for the same
(deter, stoch, action) input — confirming the shared core/prior params
were copied by name. Run:
`pixi run mojo run -I . tests/nn2/spike_param_sync.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming, Xavier
from mojo_rl.deep_agents2.dreamerv3.wm import WMCoreGraph, WMImagineGraph
from mojo_rl.deep_agents2.dreamerv3.param_sync import (
    collect_params, apply_params,
)

comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 1
comptime TOKEN = 8
comptime SC = STOCH * CLASSES
comptime CARRY = 2 + DETER + SC
comptime FEAT = DETER + SC


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, seed: Int):
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        p[i] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)


def main() raises:
    print("=" * 70)
    print("SPIKE (PR5c Step 4): param sync WMCoreGraph -> WMImagineGraph")
    print("=" * 70)
    var core = WMCoreGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make[
        "cpu", INIT=Kaiming
    ]()
    var imag = WMImagineGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT].make[
        "cpu", INIT=Xavier
    ]()
    var names = List[String]()
    var ptrs = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
    var lens = List[Int]()
    collect_params["cpu"](core, names, ptrs, lens)
    apply_params["cpu"](imag, names, ptrs, lens)

    var deter = _a(B * DETER)
    var stoch = _a(B * SC)
    var action = _a(B * ACT)
    var tokens = _a(B * TOKEN)
    _pseudo(deter, B * DETER, 1)
    _pseudo(stoch, B * SC, 2)
    _pseudo(action, B * ACT, 3)
    _pseudo(tokens, B * TOKEN, 4)

    core.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    core.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    core.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    core.set_input["tokens", B](TileTensor(tokens, row_major[B, TOKEN]()))
    var cout = _a(B * CARRY)
    var cout_t = TileTensor(cout, row_major[B, CARRY]())
    core.forward["cpu", B](cout_t)
    var core_nd = core.node_out_ptr["nd"]()

    imag.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    imag.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    imag.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    var iout = _a(B * FEAT)
    var iout_t = TileTensor(iout, row_major[B, FEAT]())
    imag.forward["cpu", B](iout_t)
    var imag_nd = imag.node_out_ptr["nd"]()

    var m: Scalar[DT] = 0.0
    for i in range(B * DETER):
        var d = core_nd[i] - imag_nd[i]
        var ad = d if d >= 0 else -d
        if ad > m:
            m = ad
    print("  max|core.nd - imag.nd| after sync =", m)
    assert_true(m < Scalar[DT](1e-6), "param sync: nd must match")
    print("=" * 70)
    print("SPIKE PASSED — core/prior params synced by name")
    print("=" * 70)
    _ = deter; _ = stoch; _ = action; _ = tokens; _ = cout; _ = iout