"""SPIKE (PR5c Step 1): the full single-step WM loss as ONE ComputeGraph.

Extends `spike_wm_kl_graph.mojo` (dyn/rep) into the complete WM loss:
dyn, rep, recon, rew, con — output `[B,5]`. The new loss-Module ops
(`SymlogMSELoss`, `TwoHotLoss`, `BinaryLoss`) attach to the decoder /
reward / cont `Sequential` nets; `StraightThroughSample` feeds the
sampled stoch into both the decoder and the head feat.

Validation:
  (a) dyn/rep columns of the full graph still match the `pr5b2` `wm.*`
      jax fixture (seed grad_output on cols 0,1 only) — the decoder/head
      branches contribute zero, so the core/obs/prior grads are unchanged.
  (b) the new loss ops + nets, in standalone mini-graphs fed the
      per-component fixtures, match `dec.*`/`gdec.*`, `hd.*`/`grew.*`,
      `hd.*`/`gcon.*` (the cotangent the ops produce == the manual
      seeding validated in `spike_dreamer_nets.mojo`).
  (c) the full graph forward + full backward (ones on all 5 cols) is
      finite.

Run: `pixi run mojo run -I . tests/nn/spike_wm_full_graph.mojo`
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
    ActionSquash, BlockGroupAssemble, GRUGate, StraightThroughSample,
)
from mojo_rl.deep_agents.dreamerv3.onehot_kl import OneHotKLLoss
from mojo_rl.deep_agents.dreamerv3.wm_loss_ops import (
    SymlogMSELoss, TwoHotLoss, BinaryLoss,
)
from mojo_rl.deep_agents.dreamerv3.nets import (
    DreamerDecoder, DreamerRewardMLP, DreamerContMLP,
)

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
comptime SC = STOCH * CLASSES         # 15
comptime DHIN = DETER + 3 * H * BLOCKS
comptime GRU_OUT = 3 * DETER
comptime OBSIN = DETER + TOKEN
comptime OBS = 4
comptime DEC_U = 8
comptime HU = 8
comptime HBINS = 255                  # standalone reward fixture bins
comptime FEATIN = SC + DETER          # decoder input  (stoch, deter)
comptime HFEAT = DETER + SC           # head feat      (deter, stoch)
# small odd bins for the full-graph smoke (decoder/heads Zero-init).
comptime SBINS = 7


# ── helpers ───────────────────────────────────────────────────────────
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


def _finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        var v = p[i]
        # NaN != itself; Inf - Inf logic: just check |v| < 1e30 and v==v
        if not (v == v):
            return False
        var av = v if v >= Scalar[DT](0) else -v
        if av > Scalar[DT](1e30):
            return False
    return True


# ── (b1) decoder + symlog-mse loss as a standalone graph ──────────────
comptime DecGraph = ComputeGraph[
    1,
    InputSlot["stoch", SC],
    InputSlot["deter", DETER],
    InputSlot["rtgt", OBS],
    Node["decin", Concat[SC, DETER],                  "stoch", "deter"],
    Node["dec",   DreamerDecoder[FEATIN, OBS, DEC_U], "decin"],
    Node["recon", SymlogMSELoss[OBS],                 "dec", "rtgt"],
]


def test_decoder_loss_graph() raises:
    print("(b1) decoder + SymlogMSELoss graph vs dec.*/gdec.* ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var g = DecGraph.make["cpu", INIT=Zero]()
    # nodes: 0 stoch,1 deter,2 rtgt,3 decin,4 dec(Sequential),5 recon
    # dec children: 0 L,1 N,2 G,3 L,4 N,5 G,6 L(pred)
    _set(g.nodes[4].op.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear0/kernel")
    _set(g.nodes[4].op.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear0/bias")
    _set(g.nodes[4].op.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/norm0/scale")
    _set(g.nodes[4].op.children[3].weight.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear1/kernel")
    _set(g.nodes[4].op.children[3].bias.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear1/bias")
    _set(g.nodes[4].op.children[4].gamma.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/norm1/scale")
    _set(g.nodes[4].op.children[6].weight.value_unsafe_ptr_cpu(), p4, "p.dec/vec/vec/pred/kernel")
    _set(g.nodes[4].op.children[6].bias.value_unsafe_ptr_cpu(), p4, "p.dec/vec/vec/pred/bias")

    var stoch = _buf(_read(p4, "in.dec_stoch"))
    var deter = _buf(_read(p4, "in.dec_deter"))
    var rtgt = _buf(_read(p4, "in.recon_target"))
    g.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    g.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    g.set_input["rtgt", B](TileTensor(rtgt, row_major[B, OBS]()))

    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    var out_t = TileTensor(out, row_major[B, 1]())
    g.forward["cpu", B](out_t)
    var seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    for i in range(B):
        seed[i] = 1.0
    g.vjp["cpu", B](TileTensor(seed, row_major[B, 1]()))

    var ds = _diff(g.grad_input_ptr["stoch"](), _read(g5, "dec.g_stoch"))
    var dd = _diff(g.grad_input_ptr["deter"](), _read(g5, "dec.g_deter"))
    var dkp = _diff(g.nodes[4].op.children[6].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "gdec.vec/vec/pred/kernel"))
    print("  g_stoch", ds, " g_deter", dd, " pred.k", dkp)
    assert_true(ds < Scalar[DT](1e-4), "dec graph g_stoch")
    assert_true(dd < Scalar[DT](1e-4), "dec graph g_deter")
    assert_true(dkp < Scalar[DT](1e-4), "dec graph pred.kernel")
    print("  ok")
    _ = stoch; _ = deter; _ = rtgt; _ = out; _ = seed


# ── (b2) reward MLP + twohot loss as a standalone graph ───────────────
comptime RewGraph = ComputeGraph[
    1,
    InputSlot["feat", HFEAT],
    InputSlot["rtgt", 1],
    Node["rew",  DreamerRewardMLP[HFEAT, HU, HBINS], "feat"],
    Node["rewl", TwoHotLoss[HBINS],                  "rew", "rtgt"],
]


def test_reward_loss_graph() raises:
    print("(b2) reward MLP + TwoHotLoss graph vs hd.rew_g_feat/grew.* ...")
    var g5 = _lines(F5)
    var g = RewGraph.make["cpu", INIT=Zero]()
    # nodes: 0 feat,1 rtgt,2 rew(Sequential: L,N,G,L),3 rewl(TwoHotLoss)
    _set(g.nodes[2].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "prew.mlp/linear0/kernel")
    _set(g.nodes[2].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "prew.mlp/linear0/bias")
    _set(g.nodes[2].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "prew.mlp/norm0/scale")
    _set(g.nodes[2].op.children[3].weight.value_unsafe_ptr_cpu(), g5, "prew.head/logits/kernel")
    _set(g.nodes[2].op.children[3].bias.value_unsafe_ptr_cpu(), g5, "prew.head/logits/bias")
    # overwrite the op's symexp bins with the fixture's bins.
    _set(g.nodes[3].op.bins_unsafe_ptr(), g5, "hd.bins")

    var feat = _buf(_read(g5, "hd.feat"))
    var rtgt_list = _read(g5, "hd.rew_target")
    var rtgt = _buf(rtgt_list)   # [B] → [B,1]
    g.set_input["feat", B](TileTensor(feat, row_major[B, HFEAT]()))
    g.set_input["rtgt", B](TileTensor(rtgt, row_major[B, 1]()))

    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    var out_t = TileTensor(out, row_major[B, 1]())
    g.forward["cpu", B](out_t)
    var seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    for i in range(B):
        seed[i] = 1.0
    var seed_t = TileTensor(seed, row_major[B, 1]())
    g.vjp["cpu", B](seed_t)

    var df = _diff(g.grad_input_ptr["feat"](), _read(g5, "hd.rew_g_feat"))
    var dk = _diff(g.nodes[2].op.children[0].weight.grad_unsafe_ptr_cpu(),
                   _read(g5, "grew.mlp/linear0/kernel"))
    print("  g_feat", df, " lin0.k", dk)
    assert_true(df < Scalar[DT](1e-4), "rew graph g_feat")
    assert_true(dk < Scalar[DT](1e-4), "rew graph lin0.kernel")
    print("  ok")
    _ = feat; _ = rtgt; _ = out; _ = seed


# ── (b3) cont MLP + binary loss as a standalone graph ─────────────────
comptime ConGraph = ComputeGraph[
    1,
    InputSlot["feat", HFEAT],
    InputSlot["ctgt", 1],
    Node["con",  DreamerContMLP[HFEAT, HU], "feat"],
    Node["conl", BinaryLoss,                "con", "ctgt"],
]


def test_cont_loss_graph() raises:
    print("(b3) cont MLP + BinaryLoss graph vs hd.con_g_feat/gcon.* ...")
    var g5 = _lines(F5)
    var g = ConGraph.make["cpu", INIT=Zero]()
    # nodes: 0 feat,1 ctgt,2 con(Sequential: L,N,G,L),3 conl(BinaryLoss)
    _set(g.nodes[2].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pcon.mlp/linear0/kernel")
    _set(g.nodes[2].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pcon.mlp/linear0/bias")
    _set(g.nodes[2].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pcon.mlp/norm0/scale")
    _set(g.nodes[2].op.children[3].weight.value_unsafe_ptr_cpu(), g5, "pcon.head/logit/kernel")
    _set(g.nodes[2].op.children[3].bias.value_unsafe_ptr_cpu(), g5, "pcon.head/logit/bias")

    var feat = _buf(_read(g5, "hd.feat"))
    var ctgt = _buf(_read(g5, "hd.con_target"))
    g.set_input["feat", B](TileTensor(feat, row_major[B, HFEAT]()))
    g.set_input["ctgt", B](TileTensor(ctgt, row_major[B, 1]()))

    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    var out_t = TileTensor(out, row_major[B, 1]())
    g.forward["cpu", B](out_t)
    var seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 1)
    for i in range(B):
        seed[i] = 1.0
    var seed_t = TileTensor(seed, row_major[B, 1]())
    g.vjp["cpu", B](seed_t)

    var df = _diff(g.grad_input_ptr["feat"](), _read(g5, "hd.con_g_feat"))
    var dk = _diff(g.nodes[2].op.children[0].weight.grad_unsafe_ptr_cpu(),
                   _read(g5, "gcon.mlp/linear0/kernel"))
    print("  g_feat", df, " lin0.k", dk)
    assert_true(df < Scalar[DT](1e-4), "con graph g_feat")
    assert_true(dk < Scalar[DT](1e-4), "con graph lin0.kernel")
    print("  ok")
    _ = feat; _ = ctgt; _ = out; _ = seed


# ── (a)+(c) the full WM loss graph. Output [B,5] = [dyn,rep,recon,rew,con].
comptime WMFullGraph = ComputeGraph[
    5,
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
    Node["obsin",  Concat[DETER, TOKEN],                           "nd", "tokens"],
    Node["obshid", Sequential[Linear[OBSIN, H], RMSNorm[H], GELU[H]], "obsin"],
    Node["post",   Linear[H, SC],                                  "obshid"],
    Node["pr0",   Sequential[Linear[DETER, H], RMSNorm[H], GELU[H]], "nd"],
    Node["pr1",   Sequential[Linear[H, H],     RMSNorm[H], GELU[H]], "pr0"],
    Node["prior", Linear[H, SC],                                    "pr1"],
    Node["kl",    OneHotKLLoss[STOCH, CLASSES],                     "post", "prior"],
    # extra input slots (appended → core/obs/prior node indices unchanged)
    InputSlot["recon_target", OBS],
    InputSlot["rew_target", 1],
    InputSlot["con_target", 1],
    # straight-through sampled stoch, fed to both decoder + head feat
    Node["stoch_new", StraightThroughSample[STOCH, CLASSES],        "post"],
    # decoder branch (recon)
    Node["decin", Concat[SC, DETER],                  "stoch_new", "nd"],
    Node["dec",   DreamerDecoder[FEATIN, OBS, DEC_U], "decin"],
    Node["recon", SymlogMSELoss[OBS],                 "dec", "recon_target"],
    # reward + cont branches share feat = concat([nd, stoch_new])
    Node["feat",  Concat[DETER, SC],                  "nd", "stoch_new"],
    Node["rew",   DreamerRewardMLP[HFEAT, HU, SBINS], "feat"],
    Node["rewl",  TwoHotLoss[SBINS],                  "rew", "rew_target"],
    Node["con",   DreamerContMLP[HFEAT, HU],          "feat"],
    Node["conl",  BinaryLoss,                         "con", "con_target"],
    # assemble [dyn,rep, recon, rew, con] → [B,5]
    Node["out",   Concat[2, 1, 1, 1],   "kl", "recon", "rewl", "conl"],
]


def _load_wm(mut g: WMFullGraph, g5: List[String]) raises:
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
    _set(g.nodes[15].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.prior0/kernel")
    _set(g.nodes[15].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.prior0/bias")
    _set(g.nodes[15].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.prior0norm/scale")
    _set(g.nodes[16].op.children[0].weight.value_unsafe_ptr_cpu(), g5, "pwm.prior1/kernel")
    _set(g.nodes[16].op.children[0].bias.value_unsafe_ptr_cpu(), g5, "pwm.prior1/bias")
    _set(g.nodes[16].op.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pwm.prior1norm/scale")
    _set(g.nodes[17].op.weight.value_unsafe_ptr_cpu(), g5, "pwm.priorlogit/kernel")
    _set(g.nodes[17].op.bias.value_unsafe_ptr_cpu(), g5, "pwm.priorlogit/bias")


def test_full_graph() raises:
    print("(a)+(c) full WM loss graph [B,5] ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var g = WMFullGraph.make["cpu", INIT=Zero]()
    _load_wm(g, g5)

    var deter = _buf(_read(p4, "in.deter0"))
    var stoch = _buf(_read(p4, "in.stoch0"))
    var action = _buf(_read(p4, "in.action"))
    var tokens = _buf(_read(p4, "in.tokens"))
    # decoder/head targets: arbitrary finite values (params Zero → smoke).
    var rtgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OBS)
    for i in range(B * OBS):
        rtgt[i] = Scalar[DT](0.1) * Scalar[DT](i)
    var rew_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var con_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    for i in range(B):
        rew_t[i] = Scalar[DT](0.5)
        con_t[i] = Scalar[DT](1.0)
    g.set_input["deter", B](TileTensor(deter, row_major[B, DETER]()))
    g.set_input["stoch", B](TileTensor(stoch, row_major[B, SC]()))
    g.set_input["action", B](TileTensor(action, row_major[B, ACT]()))
    g.set_input["tokens", B](TileTensor(tokens, row_major[B, TOKEN]()))
    g.set_input["recon_target", B](TileTensor(rtgt, row_major[B, OBS]()))
    g.set_input["rew_target", B](TileTensor(rew_t, row_major[B, 1]()))
    g.set_input["con_target", B](TileTensor(con_t, row_major[B, 1]()))

    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 5)
    var out_t = TileTensor(out, row_major[B, 5]())
    g.forward["cpu", B](out_t)
    assert_true(_finite(out, B * 5), "full graph forward finite")

    # (a) seed only cols 0,1 (dyn,rep) → reproduce wm.* grads exactly.
    var seedA: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 5)
    for b in range(B):
        for c in range(5):
            seedA[b * 5 + c] = 1.0 if c < 2 else 0.0
    var seedA_t = TileTensor(seedA, row_major[B, 5]())
    g.vjp["cpu", B](seedA_t)
    var dd = _diff(g.grad_input_ptr["deter"](), _read(g5, "wm.g_deter"))
    var ds = _diff(g.grad_input_ptr["stoch"](), _read(g5, "wm.g_stoch"))
    var da = _diff(g.grad_input_ptr["action"](), _read(g5, "wm.g_action"))
    var dt = _diff(g.grad_input_ptr["tokens"](), _read(g5, "wm.g_tokens"))
    print("  (a) grad: deter", dd, " stoch", ds, " action", da, " tokens", dt)
    assert_true(dd < Scalar[DT](1e-4), "full graph wm grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "full graph wm grad_stoch")
    assert_true(da < Scalar[DT](1e-4), "full graph wm grad_action")
    assert_true(dt < Scalar[DT](1e-4), "full graph wm grad_tokens")
    var din0 = _diff(g.nodes[5].op.children[0].weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.dynin0/kernel"))
    var dgru = _diff(g.nodes[10].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.dyngru/kernel"))
    var dobl = _diff(g.nodes[14].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.obslogit/kernel"))
    var dprl = _diff(g.nodes[17].op.weight.grad_unsafe_ptr_cpu(),
                     _read(g5, "gwm.priorlogit/kernel"))
    print("  (a) params: dynin0", din0, " dyngru", dgru, " obslogit", dobl,
          " priorlogit", dprl)
    assert_true(din0 < Scalar[DT](1e-4), "full graph dynin0.kernel")
    assert_true(dgru < Scalar[DT](1e-4), "full graph dyngru.kernel")
    assert_true(dobl < Scalar[DT](1e-4), "full graph obslogit.kernel")
    assert_true(dprl < Scalar[DT](1e-4), "full graph priorlogit.kernel")

    # (c) full backward (ones on all 5 cols) → finite grads everywhere.
    var seedC: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * 5)
    for i in range(B * 5):
        seedC[i] = 1.0
    var seedC_t = TileTensor(seedC, row_major[B, 5]())
    g.vjp["cpu", B](seedC_t)
    assert_true(_finite(g.grad_input_ptr["deter"](), B * DETER),
                "full graph (c) grad_deter finite")
    assert_true(_finite(g.grad_input_ptr["stoch"](), B * SC),
                "full graph (c) grad_stoch finite")
    assert_true(_finite(g.nodes[10].op.weight.grad_unsafe_ptr_cpu(),
                        DETER * GRU_OUT), "full graph (c) dyngru finite")
    print("  (c) full backward finite — ok")
    _ = deter; _ = stoch; _ = action; _ = tokens
    _ = rtgt; _ = rew_t; _ = con_t; _ = out; _ = seedA; _ = seedC


def main() raises:
    print("=" * 70)
    print("SPIKE (PR5c Step 1): full WM loss as ComputeGraph [B,5]")
    print("=" * 70)
    test_decoder_loss_graph()
    test_reward_loss_graph()
    test_cont_loss_graph()
    test_full_graph()
    print("=" * 70)
    print("SPIKE PASSED — full WM loss graph (dyn,rep,recon,rew,con)")
    print("=" * 70)
