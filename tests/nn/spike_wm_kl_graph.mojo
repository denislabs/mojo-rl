"""SPIKE chunk 3: the full single-step WM dyn/rep loss as ONE ComputeGraph.

The crux test of the redesign: `nd` (new_deter, GRUGate output) fans out to
BOTH the obs head (→ rep KL) AND the prior (→ dyn KL). The graph must
auto-accumulate the two gradient paths at `nd` — exactly the hand-wired
"3-path grad_deter" / loss_vjp assembly, now done by the framework.

Validated against the pr5b2 `wm.*` jax fixture (cotangent ones on [dyn,rep]):
input grads (deter/stoch/action/tokens) + param grads across all 3 paths.

Run: `pixi run mojo run -I . tests/nn/spike_wm_kl_graph.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import InputSlot, Node
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.block_linear import BlockLinear
from mojo_rl.nn.primitives.rms_norm import RMSNorm
from mojo_rl.nn.primitives.gelu import GELU
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.deep_agents.dreamerv3.rssm_ops import (
    ActionSquash, BlockGroupAssemble, GRUGate,
)
from mojo_rl.deep_agents.dreamerv3.onehot_kl import OneHotKLLoss

comptime F4 = "tests/nn/dreamerv3/fixtures/pr4_fixture.txt"
comptime F5 = "tests/nn/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime TOKEN = 8
comptime SC = STOCH * CLASSES
comptime DHIN = DETER + 3 * H * BLOCKS
comptime GRU_OUT = 3 * DETER
comptime OBSIN = DETER + TOKEN


# ── The full single-step WM dyn/rep loss. Output [B,2] = [dyn, rep]. ──
comptime WMKLGraph = ComputeGraph[
    2,
    InputSlot["deter", DETER],
    InputSlot["stoch", SC],
    InputSlot["action", ACT],
    InputSlot["tokens", TOKEN],
    Node["a",    ActionSquash[ACT],                                "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], GELU[H]], "deter"],
    Node["x1",   Sequential[Linear[SC, H],    RMSNorm[H], GELU[H]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H],   RMSNorm[H], GELU[H]], "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DHIN, DETER, BLOCKS], RMSNorm[DETER], GELU[DETER]], "dhin"],
    Node["gru",  BlockLinear[DETER, GRU_OUT, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                           "gru", "deter"],
    # obs head → post (fans out from nd)
    Node["obsin",  Concat[DETER, TOKEN],                           "nd", "tokens"],
    Node["obshid", Sequential[Linear[OBSIN, H], RMSNorm[H], GELU[H]], "obsin"],
    Node["post",   Linear[H, SC],                                  "obshid"],
    # prior → prior_logit (also fans out from nd)
    Node["pr0",   Sequential[Linear[DETER, H], RMSNorm[H], GELU[H]], "nd"],
    Node["pr1",   Sequential[Linear[H, H],     RMSNorm[H], GELU[H]], "pr0"],
    Node["prior", Linear[H, SC],                                    "pr1"],
    # KL → [dyn, rep]
    Node["kl",    OneHotKLLoss[STOCH, CLASSES],                     "post", "prior"],
]


def _lines(path: String) raises -> List[String]:
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var out = List[String]()
    var cur = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            out.append(cur); cur = String("")
        else:
            cur += chr(Int(c))
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _read(lines: List[String], name: String) raises -> List[Scalar[DT]]:
    var pfx = name + "#size="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            var n = atol(String(lines[i][byte=pfx.byte_length():]))
            var o = List[Scalar[DT]]()
            for k in range(n):
                o.append(Scalar[DT](atof(lines[i + 1 + k])))
            return o^
    raise Error("not found: " + name)


def _buf(s: List[Scalar[DT]]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(s))
    for i in range(len(s)):
        p[i] = s[i]
    return p


def _set(ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], lines: List[String],
         name: String) raises:
    var v = _read(lines, name)
    for i in range(len(v)):
        ptr[i] = v[i]


def _diff(got: UnsafePointer[Scalar[DT], MutAnyOrigin],
          exp_: List[Scalar[DT]]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(exp_)):
        var d = got[i] - exp_[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def main() raises:
    print("=" * 70)
    print("SPIKE chunk 3: full WM dyn/rep loss as ComputeGraph (fan-out @ nd)")
    print("=" * 70)
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var graph = WMKLGraph.make["cpu", INIT=Zero]()

    # node indices: 0-3 InputSlots; 4 a; 5 x0; 6 x1; 7 x2; 8 dhin; 9 h;
    # 10 gru; 11 nd; 12 obsin; 13 obshid; 14 post; 15 pr0; 16 pr1; 17 prior;
    # 18 kl.  Load pwm.* (the wm-state params).
    _set(graph.nodes[5].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/kernel")
    _set(graph.nodes[5].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/bias")
    _set(graph.nodes[5].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin0norm/scale")
    _set(graph.nodes[6].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/kernel")
    _set(graph.nodes[6].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/bias")
    _set(graph.nodes[6].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin1norm/scale")
    _set(graph.nodes[7].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin2/kernel")
    _set(graph.nodes[7].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin2/bias")
    _set(graph.nodes[7].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin2norm/scale")
    _set(graph.nodes[9].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0/kernel")
    _set(graph.nodes[9].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0/bias")
    _set(graph.nodes[9].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0norm/scale")
    _set(graph.nodes[10].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/kernel")
    _set(graph.nodes[10].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/bias")
    _set(graph.nodes[13].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.obs0/kernel")
    _set(graph.nodes[13].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.obs0/bias")
    _set(graph.nodes[13].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.obs0norm/scale")
    _set(graph.nodes[14].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/kernel")
    _set(graph.nodes[14].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/bias")
    _set(graph.nodes[15].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.prior0/kernel")
    _set(graph.nodes[15].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.prior0/bias")
    _set(graph.nodes[15].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.prior0norm/scale")
    _set(graph.nodes[16].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.prior1/kernel")
    _set(graph.nodes[16].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.prior1/bias")
    _set(graph.nodes[16].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.prior1norm/scale")
    _set(graph.nodes[17].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.priorlogit/kernel")
    _set(graph.nodes[17].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.priorlogit/bias")

    # inputs (deter0/stoch0/action/tokens shared with pr4/pr5b2)
    var deter = _buf(_read(p4, "in.deter0"))
    var stoch = _buf(_read(p4, "in.stoch0"))
    var action = _buf(_read(p4, "in.action"))
    var tokens = _buf(_read(p4, "in.tokens"))
    graph.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    graph.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    graph.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    graph.set_input["tokens", B](TileTensor(tokens, row_major[B, TOKEN]()))

    # forward → [B,2]; backward seeded with ones (d_dyn=d_rep=1 per row)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 2)
    var out_t = TileTensor(out, row_major[B, 2]())
    graph.forward["cpu", B](out_t)
    var seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 2)
    for i in range(B * 2):
        seed[i] = 1.0
    var seed_t = TileTensor(seed, row_major[B, 2]())
    graph.vjp["cpu", B](seed_t)

    var dd = _diff(graph.grad_input_ptr["deter"](), _read(g5, "wm.g_deter"))
    var ds = _diff(graph.grad_input_ptr["stoch"](), _read(g5, "wm.g_stoch"))
    var da = _diff(graph.grad_input_ptr["action"](), _read(g5, "wm.g_action"))
    var dt = _diff(graph.grad_input_ptr["tokens"](), _read(g5, "wm.g_tokens"))
    print("  grad inputs: deter", dd, " stoch", ds, " action", da, " tokens", dt)
    assert_true(dd < Scalar[DT](1e-4), "wm graph grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "wm graph grad_stoch")
    assert_true(da < Scalar[DT](1e-4), "wm graph grad_action")
    assert_true(dt < Scalar[DT](1e-4), "wm graph grad_tokens")

    # param grads across all 3 paths (core / obs / prior)
    var din0 = _diff(graph.nodes[5].op.children[0].weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.dynin0/kernel"))
    var dgru = _diff(graph.nodes[10].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.dyngru/kernel"))
    var dobs = _diff(graph.nodes[13].op.children[0].weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.obs0/kernel"))
    var dobl = _diff(graph.nodes[14].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.obslogit/kernel"))
    var dpr0 = _diff(graph.nodes[15].op.children[0].weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.prior0/kernel"))
    var dprl = _diff(graph.nodes[17].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.priorlogit/kernel"))
    print("  params: dynin0", din0, " dyngru", dgru, " obs0", dobs,
          " obslogit", dobl, " prior0", dpr0, " priorlogit", dprl)
    assert_true(din0 < Scalar[DT](1e-4), "wm graph dynin0.kernel")
    assert_true(dgru < Scalar[DT](1e-4), "wm graph dyngru.kernel")
    assert_true(dobs < Scalar[DT](1e-4), "wm graph obs0.kernel")
    assert_true(dobl < Scalar[DT](1e-4), "wm graph obslogit.kernel")
    assert_true(dpr0 < Scalar[DT](1e-4), "wm graph prior0.kernel")
    assert_true(dprl < Scalar[DT](1e-4), "wm graph priorlogit.kernel")

    print("=" * 70)
    print("SPIKE PASSED — full WM dyn/rep loss graph; fan-out @ nd auto-routed")
    print("=" * 70)
    _ = deter; _ = stoch; _ = action; _ = tokens; _ = out; _ = seed
