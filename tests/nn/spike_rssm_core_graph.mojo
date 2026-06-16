"""SPIKE: express RSSM `_core` as a declarative ComputeGraph (no hand-written
forward/backward), validate vs the existing jax-validated pr5b2 fixture.

Goal: decide whether the nn composition infra is clean + correct enough to
redesign the whole WM around it. The graph below is the deliverable to judge
for READABILITY; the asserts judge CORRECTNESS (≤1e-4 vs jax-via-pr5b2).

Run: `pixi run mojo run -I . tests/nn/spike_rssm_core_graph.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import NoAMP
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import InputSlot, Node
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.block_linear import BlockLinear
from mojo_rl.nn.primitives.rms_norm import RMSNorm
from mojo_rl.nn.primitives.gelu import GELU
from mojo_rl.deep_agents.dreamerv3.rssm_ops import (
    ActionSquash, BlockGroupAssemble, GRUGate,
)

comptime FIX4 = "tests/nn/dreamerv3/fixtures/pr4_fixture.txt"
comptime FIX5 = "tests/nn/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime SC = STOCH * CLASSES                  # 15
comptime DHIN = DETER + 3 * H * BLOCKS         # 160
comptime GRU_OUT = 3 * DETER                   # 48


# ─── THE WHOLE `_core`, as a graph. Compare this to the 150-line core_vjp. ──
comptime CoreGraph = ComputeGraph[
    DETER,
    InputSlot["deter", DETER],
    InputSlot["stoch", SC],
    InputSlot["action", ACT],
    Node["a",    ActionSquash[ACT],                                "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], GELU[H]], "deter"],
    Node["x1",   Sequential[Linear[SC, H],    RMSNorm[H], GELU[H]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H],   RMSNorm[H], GELU[H]], "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DHIN, DETER, BLOCKS], RMSNorm[DETER], GELU[DETER]], "dhin"],
    Node["gru",  BlockLinear[DETER, GRU_OUT, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                           "gru", "deter"],
]


def _split_lines(content: String) raises -> List[String]:
    var lines = List[String]()
    var cur = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(cur); cur = String("")
        else:
            cur += chr(Int(c))
    if cur.byte_length() > 0:
        lines.append(cur)
    return lines^


def _read_flat(lines: List[String], name: String) raises -> List[Scalar[DT]]:
    var pfx = name + "#size="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            var n = atol(String(lines[i][byte=pfx.byte_length():]))
            var out = List[Scalar[DT]]()
            for k in range(n):
                out.append(Scalar[DT](atof(lines[i + 1 + k])))
            return out^
    raise Error("fixture: not found: " + name)


def _buf(src: List[Scalar[DT]]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(src))
    for i in range(len(src)):
        p[i] = src[i]
    return p


def _load(ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
          lines: List[String], name: String) raises:
    var v = _read_flat(lines, name)
    for i in range(len(v)):
        ptr[i] = v[i]


def _diff(got: UnsafePointer[Scalar[DT], MutAnyOrigin],
          exp: List[Scalar[DT]]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(exp)):
        var d = got[i] - exp[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def main() raises:
    print("=" * 70)
    print("SPIKE: RSSM _core as ComputeGraph (vs pr5b2 jax fixture)")
    print("=" * 70)
    var c4: String
    with open(FIX4, "r") as f:
        c4 = String(f.read())
    var c5: String
    with open(FIX5, "r") as f:
        c5 = String(f.read())
    var pl = _split_lines(c4)
    var gl = _split_lines(c5)

    var graph = CoreGraph.make["cpu", INIT=Zero]()

    # ── load core params into the graph's nodes (pr4 = core_fn init) ──
    # node indices: 0..2 InputSlots; 3=a; 4=x0; 5=x1; 6=x2; 7=dhin; 8=h;
    # 9=gru; 10=nd.  x* are Sequential[Linear, RMSNorm, GELU].
    _load(graph.nodes[4].op.children[0].weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0/kernel")
    _load(graph.nodes[4].op.children[0].bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0/bias")
    _load(graph.nodes[4].op.children[1].gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin0norm/scale")
    _load(graph.nodes[5].op.children[0].weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1/kernel")
    _load(graph.nodes[5].op.children[0].bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1/bias")
    _load(graph.nodes[5].op.children[1].gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin1norm/scale")
    _load(graph.nodes[6].op.children[0].weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2/kernel")
    _load(graph.nodes[6].op.children[0].bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2/bias")
    _load(graph.nodes[6].op.children[1].gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynin2norm/scale")
    _load(graph.nodes[8].op.children[0].weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0/kernel")
    _load(graph.nodes[8].op.children[0].bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0/bias")
    _load(graph.nodes[8].op.children[1].gamma.value_unsafe_ptr_cpu(), pl, "p.rssm/dynhid0norm/scale")
    _load(graph.nodes[9].op.weight.value_unsafe_ptr_cpu(), pl, "p.rssm/dyngru/kernel")
    _load(graph.nodes[9].op.bias.value_unsafe_ptr_cpu(), pl, "p.rssm/dyngru/bias")

    # ── inputs ──
    var deter = _buf(_read_flat(pl, "in.deter0"))
    var stoch = _buf(_read_flat(pl, "in.stoch0"))
    var action = _buf(_read_flat(pl, "in.action"))
    graph.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    graph.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    graph.set_input["action", B](TileTensor(action, row_major[B, ACT]()))

    # ── forward ──
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DETER)
    var out_t = TileTensor(out, row_major[B, DETER]())
    graph.forward["cpu", B](out_t)
    var dcore = _diff(out, _read_flat(pl, "out.core"))
    print("  forward (new_deter) diff =", dcore)
    assert_true(dcore < Scalar[DT](1e-4), "graph _core forward parity")

    # ── backward: seed grad_output = core.g_out, read input grads ──
    var go = _buf(_read_flat(gl, "core.g_out"))
    graph.vjp["cpu", B](TileTensor(go, row_major[B, DETER]()))

    var gd_ptr = graph.grad_input_ptr["deter"]()
    var gs_ptr = graph.grad_input_ptr["stoch"]()
    var ga_ptr = graph.grad_input_ptr["action"]()
    var dd = _diff(gd_ptr, _read_flat(gl, "core.g_deter"))
    var ds = _diff(gs_ptr, _read_flat(gl, "core.g_stoch"))
    var da = _diff(ga_ptr, _read_flat(gl, "core.g_action"))
    print("  grad_deter =", dd, " grad_stoch =", ds, " grad_action =", da)
    assert_true(dd < Scalar[DT](1e-4), "graph grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "graph grad_stoch")
    assert_true(da < Scalar[DT](1e-4), "graph grad_action")

    # ── param grads (sampled across the 3 paths) ──
    var dk_in0 = _diff(graph.nodes[4].op.children[0].weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynin0/kernel"))
    var dk_hid = _diff(graph.nodes[8].op.children[0].weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dynhid0/kernel"))
    var dk_gru = _diff(graph.nodes[9].op.weight.grad_unsafe_ptr_cpu(),
                       _read_flat(gl, "gcore.dyngru/kernel"))
    print("  param grads: dynin0.k =", dk_in0, " dynhid0.k =", dk_hid,
          " dyngru.k =", dk_gru)
    assert_true(dk_in0 < Scalar[DT](1e-4), "graph grad dynin0.kernel")
    assert_true(dk_hid < Scalar[DT](1e-4), "graph grad dynhid0.kernel")
    assert_true(dk_gru < Scalar[DT](1e-4), "graph grad dyngru.kernel")

    print("=" * 70)
    print("SPIKE PASSED — _core composes cleanly + matches jax")
    print("=" * 70)
    _ = deter; _ = stoch; _ = action; _ = go
