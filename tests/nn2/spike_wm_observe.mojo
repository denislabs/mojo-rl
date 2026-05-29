"""SPIKE (PR5c Step 2): WMObserveGraph forward parity + WMLossGraph carry.

(1) `WMObserveGraph.forward` → `post` matches `out.obs_logit`, and the
    carry `node_out_ptr["nd"]` matches `out.obs_deter` (pr4 fixture).
(2) `WMLossGraph` forward: cols 0..4 finite; the carry passthrough cols
    (5..5+DETER = nd, then stoch_new) equal `node_out_ptr["nd"]` /
    `node_out_ptr["stoch_new"]` (so the trainer can read the carry from
    the output slab and inject its grad back via the same columns).

Run: `pixi run mojo run -I . tests/nn2/spike_wm_observe.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.dreamerv3.wm import WMObserveGraph, WMLossGraph

comptime F4 = "tests/nn2/dreamerv3/fixtures/pr4_fixture.txt"
comptime F5 = "tests/nn2/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime TOKEN = 8
comptime SC = STOCH * CLASSES
comptime OBS = 4
comptime DEC_U = 8
comptime HU = 8
comptime SBINS = 7


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


def test_observe_graph() raises:
    print("(1) WMObserveGraph forward parity vs out.obs_logit/out.obs_deter ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var g = WMObserveGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make[
        "cpu", INIT=Zero
    ]()
    # node idx: 0-3 slots, 4 a, 5 x0, 6 x1, 7 x2, 8 dhin, 9 h, 10 gru, 11 nd,
    # 12 obsin, 13 obshid, 14 post
    _set(g.nodes[5].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/kernel")
    _set(g.nodes[5].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/bias")
    _set(g.nodes[5].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin0norm/scale")
    _set(g.nodes[6].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/kernel")
    _set(g.nodes[6].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/bias")
    _set(g.nodes[6].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin1norm/scale")
    _set(g.nodes[7].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin2/kernel")
    _set(g.nodes[7].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin2/bias")
    _set(g.nodes[7].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin2norm/scale")
    _set(g.nodes[9].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0/kernel")
    _set(g.nodes[9].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0/bias")
    _set(g.nodes[9].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynhid0norm/scale")
    _set(g.nodes[10].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/kernel")
    _set(g.nodes[10].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/bias")
    _set(g.nodes[13].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.obs0/kernel")
    _set(g.nodes[13].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.obs0/bias")
    _set(g.nodes[13].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.obs0norm/scale")
    _set(g.nodes[14].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/kernel")
    _set(g.nodes[14].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/bias")

    var deter = _buf(_read(p4, "in.deter0"))
    var stoch = _buf(_read(p4, "in.stoch0"))
    var action = _buf(_read(p4, "in.action"))
    var tokens = _buf(_read(p4, "in.tokens"))
    g.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    g.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    g.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    g.set_input["tokens", B](TileTensor(tokens, row_major[B, TOKEN]()))

    var post: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var post_t = TileTensor(post, row_major[B, SC]())
    g.forward["cpu", B](post_t)

    var dpost = _diff(post, _read(p4, "out.obs_logit"))
    var dnd = _diff(g.node_out_ptr["nd"](), _read(p4, "out.obs_deter"))
    print("  post(obs_logit)", dpost, "  nd(obs_deter)", dnd)
    assert_true(dpost < Scalar[DT](1e-4), "observe post parity")
    assert_true(dnd < Scalar[DT](1e-4), "observe nd parity")
    print("  ok")
    _ = deter; _ = stoch; _ = action; _ = tokens; _ = post


def test_loss_graph_carry() raises:
    print("(2) WMLossGraph forward: carry passthrough cols == nd/stoch_new ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var g = WMLossGraph[
        DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN, OBS, DEC_U, HU, SBINS
    ].make["cpu", INIT=Zero]()
    # node idx shift: slots 0-6, then a=7,x0=8,x1=9,x2=10,dhin=11,h=12,
    # gru=13,nd=14,obsin=15,obshid=16,post=17, ...
    _set(g.nodes[8].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/kernel")
    _set(g.nodes[8].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin0/bias")
    _set(g.nodes[8].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin0norm/scale")
    _set(g.nodes[9].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/kernel")
    _set(g.nodes[9].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.dynin1/bias")
    _set(g.nodes[9].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.dynin1norm/scale")
    _set(g.nodes[13].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/kernel")
    _set(g.nodes[13].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.dyngru/bias")
    _set(g.nodes[17].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/kernel")
    _set(g.nodes[17].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.obslogit/bias")

    var deter = _buf(_read(p4, "in.deter0"))
    var stoch = _buf(_read(p4, "in.stoch0"))
    var action = _buf(_read(p4, "in.action"))
    var tokens = _buf(_read(p4, "in.tokens"))
    var rtgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OBS)
    for i in range(B * OBS):
        rtgt[i] = Scalar[DT](0.1) * Scalar[DT](i)
    var rew_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var con_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    for i in range(B):
        rew_t[i] = 0.5; con_t[i] = 1.0
    g.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    g.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    g.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    g.set_input["tokens", B](TileTensor(tokens, row_major[B, TOKEN]()))
    g.set_input["recon_target", B](TileTensor(rtgt, row_major[B, OBS]()))
    g.set_input["rew_target", B](TileTensor(rew_t, row_major[B, 1]()))
    g.set_input["con_target", B](TileTensor(con_t, row_major[B, 1]()))

    comptime OUT = 5 + DETER + SC
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT)
    var out_t = TileTensor(out, row_major[B, OUT]())
    g.forward["cpu", B](out_t)

    # carry passthrough cols 5..5+DETER == node nd; next SC == stoch_new
    var nd = g.node_out_ptr["nd"]()
    var sn = g.node_out_ptr["stoch_new"]()
    var max_nd: Scalar[DT] = 0.0
    var max_sn: Scalar[DT] = 0.0
    for b in range(B):
        for k in range(DETER):
            var d = out[b * OUT + 5 + k] - nd[b * DETER + k]
            var ad = d if d >= 0 else -d
            if ad > max_nd: max_nd = ad
        for k in range(SC):
            var d2 = out[b * OUT + 5 + DETER + k] - sn[b * SC + k]
            var ad2 = d2 if d2 >= 0 else -d2
            if ad2 > max_sn: max_sn = ad2
    print("  carry-passthrough diff: nd", max_nd, " stoch_new", max_sn)
    assert_true(max_nd < Scalar[DT](1e-6), "carry nd passthrough")
    assert_true(max_sn < Scalar[DT](1e-6), "carry stoch_new passthrough")
    # losses finite
    for b in range(B):
        for c in range(5):
            var v = out[b * OUT + c]
            assert_true(v == v, "loss col finite")
    print("  ok")
    _ = deter; _ = stoch; _ = action; _ = tokens; _ = rtgt; _ = rew_t
    _ = con_t; _ = out


def main() raises:
    print("=" * 70)
    print("SPIKE (PR5c Step 2): WMObserveGraph + WMLossGraph carry")
    print("=" * 70)
    test_observe_graph()
    test_loss_graph_carry()
    print("=" * 70)
    print("SPIKE PASSED — observe forward parity + carry passthrough")
    print("=" * 70)
