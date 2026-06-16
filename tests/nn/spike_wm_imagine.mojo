"""SPIKE (PR5c Step 3): WMImagineGraph forward parity.

The imagination step graph (core → prior → ST sample → feat) must
reproduce the core forward of the actual reference: node `nd` ==
`out.core` (new deter). The in-context prior is `prior(nd)`; pr4's
`out.prior` is the *standalone* `prior(deter0)` unit vector, so the prior
MLP forward is validated separately by feeding `deter0` to a `DreamerPrior`.
All params from pr4 `p.rssm/*` (pr5b2 `pwm.*` is an INDEPENDENT param set —
matches pr4 on core/obs but NOT on prior).
Run: `pixi run mojo run -I . tests/nn/spike_wm_imagine.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.deep_agents.dreamerv3.wm import WMImagineGraph
from mojo_rl.deep_agents.dreamerv3.nets import DreamerPrior

comptime F4 = "tests/nn/dreamerv3/fixtures/pr4_fixture.txt"
comptime F5 = "tests/nn/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime SC = STOCH * CLASSES


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
    print("SPIKE (PR5c Step 3): WMImagineGraph forward parity")
    print("=" * 70)
    var p4 = _lines(F4)
    var g = WMImagineGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT].make[
        "cpu", INIT=Zero
    ]()
    # idx: 0-2 slots, 3 a, 4 x0, 5 x1, 6 x2, 7 dhin, 8 h, 9 gru, 10 nd,
    # 11 pr0, 12 pr1, 13 prior, 14 stoch_new, 15 feat
    _set(g.nodes[4].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin0/kernel")
    _set(g.nodes[4].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin0/bias")
    _set(g.nodes[4].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin0norm/scale")
    _set(g.nodes[5].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin1/kernel")
    _set(g.nodes[5].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin1/bias")
    _set(g.nodes[5].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin1norm/scale")
    _set(g.nodes[6].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin2/kernel")
    _set(g.nodes[6].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin2/bias")
    _set(g.nodes[6].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/dynin2norm/scale")
    _set(g.nodes[8].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/dynhid0/kernel")
    _set(g.nodes[8].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/dynhid0/bias")
    _set(g.nodes[8].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/dynhid0norm/scale")
    _set(g.nodes[9].op.weight.value_unsafe_ptr_cpu(), p4, "p.rssm/dyngru/kernel")
    _set(g.nodes[9].op.bias.value_unsafe_ptr_cpu(), p4, "p.rssm/dyngru/bias")
    _set(g.nodes[11].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/kernel")
    _set(g.nodes[11].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/bias")
    _set(g.nodes[11].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0norm/scale")
    _set(g.nodes[12].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/kernel")
    _set(g.nodes[12].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/bias")
    _set(g.nodes[12].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1norm/scale")
    _set(g.nodes[13].op.weight.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/kernel")
    _set(g.nodes[13].op.bias.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/bias")

    var deter = _buf(_read(p4, "in.deter0"))
    var stoch = _buf(_read(p4, "in.stoch0"))
    var action = _buf(_read(p4, "in.action"))
    g.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    g.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    g.set_input["action", B](TileTensor(action, row_major[B, ACT]()))

    comptime FEAT = DETER + SC
    var feat: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FEAT)
    var feat_t = TileTensor(feat, row_major[B, FEAT]())
    g.forward["cpu", B](feat_t)

    # nd (new deter) must match the reference core output.
    var dnd = _diff(g.node_out_ptr["nd"](), _read(p4, "out.core"))
    print("  nd(core)", dnd)
    assert_true(dnd < Scalar[DT](1e-4), "imagine nd==out.core")

    # feat[:, :DETER] == nd (the Concat passthrough); finite everywhere.
    var feat_nd_diff: Scalar[DT] = 0.0
    var nd = g.node_out_ptr["nd"]()
    for b in range(B):
        for k in range(DETER):
            var d = feat[b * FEAT + k] - nd[b * DETER + k]
            var ad = d if d >= 0 else -d
            if ad > feat_nd_diff: feat_nd_diff = ad
    assert_true(feat_nd_diff < Scalar[DT](1e-6), "feat[:DETER]==nd")
    for i in range(B * FEAT):
        assert_true(feat[i] == feat[i], "feat finite")
    print("  feat[:DETER]==nd", feat_nd_diff, " feat finite — ok")

    # The in-context prior is prior(nd); pr4's out.prior is the standalone
    # prior(deter0) unit vector — validate the prior MLP forward that way.
    var pr = DreamerPrior[DETER, H, SC].make["cpu", INIT=Zero]()
    _set(pr.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/kernel")
    _set(pr.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/bias")
    _set(pr.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0norm/scale")
    _set(pr.children[3].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/kernel")
    _set(pr.children[3].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/bias")
    _set(pr.children[4].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1norm/scale")
    _set(pr.children[6].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/kernel")
    _set(pr.children[6].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/bias")
    var deter0 = _buf(_read(p4, "in.deter0"))
    var prout: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var prout_t = TileTensor(prout, row_major[B, SC]())
    pr.forward["cpu", B](TileTensor(deter0, row_major[B, DETER]()), output=prout_t)
    var dpr = _diff(prout, _read(p4, "out.prior"))
    print("  prior MLP fwd(deter0) vs out.prior", dpr)
    assert_true(dpr < Scalar[DT](1e-4), "DreamerPrior fwd==out.prior")
    print("  ok")
    print("=" * 70)
    print("SPIKE PASSED — imagine step forward matches core reference")
    print("=" * 70)
    _ = deter; _ = stoch; _ = action; _ = feat; _ = deter0; _ = prout
