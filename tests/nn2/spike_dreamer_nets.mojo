"""SPIKE chunk 1: DreamerV3 nets as pure `Sequential` aliases, validated vs
the EXISTING jax fixtures (pr4 forward, pr5b2 grads). No hand-written
forward/backward — `Sequential` does it all.

Covers: Encoder, Decoder, RSSM prior, reward-MLP, cont-MLP.
Run: `pixi run mojo run -I . tests/nn2/spike_dreamer_nets.mojo`
"""

from std.memory import alloc
from std.math import log1p, exp
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerEncoder, DreamerDecoder, DreamerPrior, DreamerRewardMLP,
    DreamerContMLP,
)
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_loss_backward

comptime F4 = "tests/nn2/dreamerv3/fixtures/pr4_fixture.txt"
comptime F5 = "tests/nn2/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime H = 12
comptime SC = 15
comptime OBS = 4
comptime ENC_U = 8
comptime DEC_U = 8
comptime FEATIN = SC + DETER       # 31
comptime HFEAT = DETER + SC        # 31
comptime HU = 8
comptime HBINS = 255


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


def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


def test_encoder() raises:
    print("encoder (Sequential[Symlog, (Linear,RMSNorm,GELU)x2]) ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var enc = DreamerEncoder[OBS, ENC_U].make["cpu", INIT=Zero]()
    # children: 0 Symlog, 1 Linear, 2 RMSNorm, 3 GELU, 4 Linear, 5 RMSNorm, 6 GELU
    _set(enc.children[1].weight.value_unsafe_ptr_cpu(), p4, "p.enc/mlp0/kernel")
    _set(enc.children[1].bias.value_unsafe_ptr_cpu(), p4, "p.enc/mlp0/bias")
    _set(enc.children[2].gamma.value_unsafe_ptr_cpu(), p4, "p.enc/mlp0norm/scale")
    _set(enc.children[4].weight.value_unsafe_ptr_cpu(), p4, "p.enc/mlp1/kernel")
    _set(enc.children[4].bias.value_unsafe_ptr_cpu(), p4, "p.enc/mlp1/bias")
    _set(enc.children[5].gamma.value_unsafe_ptr_cpu(), p4, "p.enc/mlp1norm/scale")

    var obs = _buf(_read(p4, "in.obs_vec"))
    var tok: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * ENC_U)
    var tok_t = TileTensor(tok, row_major[B, ENC_U]())
    enc.forward["cpu", B](TileTensor(obs, row_major[B, OBS]()), output=tok_t)
    var df = _diff(tok, _read(p4, "out.enc_tok"))
    print("  fwd diff =", df)
    assert_true(df < Scalar[DT](1e-4), "encoder fwd")

    var gt = _buf(_read(g5, "enc.g_tok"))
    var gobs: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OBS)
    var gobs_t = TileTensor(gobs, row_major[B, OBS]())
    enc.vjp["cpu", B](TileTensor(gt, row_major[B, ENC_U]()), gobs_t)
    var dobs = _diff(gobs, _read(g5, "enc.g_obs"))
    var dk0 = _diff(enc.children[1].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "genc.mlp0/kernel"))
    var dk1 = _diff(enc.children[4].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "genc.mlp1/kernel"))
    print("  g_obs =", dobs, " mlp0.k =", dk0, " mlp1.k =", dk1)
    assert_true(dobs < Scalar[DT](1e-4), "encoder g_obs")
    assert_true(dk0 < Scalar[DT](1e-4), "encoder mlp0.kernel")
    assert_true(dk1 < Scalar[DT](1e-4), "encoder mlp1.kernel")
    print("  ok")


def test_decoder() raises:
    print("decoder (Sequential[(Linear,RMSNorm,GELU)x2, Linear]) ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var dec = DreamerDecoder[FEATIN, OBS, DEC_U].make["cpu", INIT=Zero]()
    # children: 0 L,1 N,2 G,3 L,4 N,5 G,6 L(pred)
    _set(dec.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear0/kernel")
    _set(dec.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear0/bias")
    _set(dec.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/norm0/scale")
    _set(dec.children[3].weight.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear1/kernel")
    _set(dec.children[3].bias.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/linear1/bias")
    _set(dec.children[4].gamma.value_unsafe_ptr_cpu(), p4, "p.dec/mlp/norm1/scale")
    _set(dec.children[6].weight.value_unsafe_ptr_cpu(), p4, "p.dec/vec/vec/pred/kernel")
    _set(dec.children[6].bias.value_unsafe_ptr_cpu(), p4, "p.dec/vec/vec/pred/bias")

    # input = concat([stoch_flat, deter])
    var deter = _read(p4, "in.dec_deter")
    var stoch = _read(p4, "in.dec_stoch")
    var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FEATIN)
    for b in range(B):
        for k in range(SC):
            inp[b * FEATIN + k] = stoch[b * SC + k]
        for k in range(DETER):
            inp[b * FEATIN + SC + k] = deter[b * DETER + k]
    var pred: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OBS)
    var pred_t = TileTensor(pred, row_major[B, OBS]())
    dec.forward["cpu", B](TileTensor(inp, row_major[B, FEATIN]()), output=pred_t)
    var dp = _diff(pred, _read(p4, "out.dec_pred"))
    print("  fwd(pred) diff =", dp)
    assert_true(dp < Scalar[DT](1e-4), "decoder fwd")

    # grad_pred = 2*(pred - symlog(target)); vjp → grad_input split + params
    var target = _read(p4, "in.recon_target")
    var gpred: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OBS)
    for i in range(B * OBS):
        gpred[i] = Scalar[DT](2.0) * (pred[i] - _symlog(target[i]))
    var ginp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FEATIN)
    var ginp_t = TileTensor(ginp, row_major[B, FEATIN]())
    dec.vjp["cpu", B](TileTensor(gpred, row_major[B, OBS]()), ginp_t)
    # split ginp → stoch (first SC) / deter (rest)
    var g_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var g_deter: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DETER)
    for b in range(B):
        for k in range(SC):
            g_stoch[b * SC + k] = ginp[b * FEATIN + k]
        for k in range(DETER):
            g_deter[b * DETER + k] = ginp[b * FEATIN + SC + k]
    var dd = _diff(g_deter, _read(g5, "dec.g_deter"))
    var ds = _diff(g_stoch, _read(g5, "dec.g_stoch"))
    var dkp = _diff(dec.children[6].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "gdec.vec/vec/pred/kernel"))
    print("  g_deter =", dd, " g_stoch =", ds, " pred.k =", dkp)
    assert_true(dd < Scalar[DT](1e-4), "decoder g_deter")
    assert_true(ds < Scalar[DT](1e-4), "decoder g_stoch")
    assert_true(dkp < Scalar[DT](1e-4), "decoder pred.kernel")
    print("  ok")


def test_prior() raises:
    print("prior (Sequential[(Linear,RMSNorm,GELU)x2, Linear]) ...")
    var p4 = _lines(F4)
    var g5 = _lines(F5)
    var pr = DreamerPrior[DETER, H, SC].make["cpu", INIT=Zero]()
    _set(pr.children[0].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/kernel")
    _set(pr.children[0].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0/bias")
    _set(pr.children[1].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior0norm/scale")
    _set(pr.children[3].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/kernel")
    _set(pr.children[3].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1/bias")
    _set(pr.children[4].gamma.value_unsafe_ptr_cpu(), p4, "p.rssm/prior1norm/scale")
    _set(pr.children[6].weight.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/kernel")
    _set(pr.children[6].bias.value_unsafe_ptr_cpu(), p4, "p.rssm/priorlogit/bias")

    var deter = _buf(_read(p4, "in.deter0"))
    var logit: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var logit_t = TileTensor(logit, row_major[B, SC]())
    pr.forward["cpu", B](TileTensor(deter, row_major[B, DETER]()), output=logit_t)
    var glog = _buf(_read(g5, "prior.g_out"))
    var gdet: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DETER)
    var gdet_t = TileTensor(gdet, row_major[B, DETER]())
    pr.vjp["cpu", B](TileTensor(glog, row_major[B, SC]()), gdet_t)
    var dd = _diff(gdet, _read(g5, "prior.g_deter"))
    var dk0 = _diff(pr.children[0].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "gprior.prior0/kernel"))
    var dkl = _diff(pr.children[6].weight.grad_unsafe_ptr_cpu(),
                    _read(g5, "gprior.priorlogit/kernel"))
    print("  g_deter =", dd, " prior0.k =", dk0, " priorlogit.k =", dkl)
    assert_true(dd < Scalar[DT](1e-4), "prior g_deter")
    assert_true(dk0 < Scalar[DT](1e-4), "prior prior0.kernel")
    assert_true(dkl < Scalar[DT](1e-4), "prior priorlogit.kernel")
    print("  ok")


def test_reward() raises:
    print("reward MLP (Sequential[Linear,RMSNorm,GELU,Linear]) ...")
    var g5 = _lines(F5)
    var h = DreamerRewardMLP[HFEAT, HU, HBINS].make["cpu", INIT=Zero]()
    _set(h.children[0].weight.value_unsafe_ptr_cpu(), g5, "prew.mlp/linear0/kernel")
    _set(h.children[0].bias.value_unsafe_ptr_cpu(), g5, "prew.mlp/linear0/bias")
    _set(h.children[1].gamma.value_unsafe_ptr_cpu(), g5, "prew.mlp/norm0/scale")
    _set(h.children[3].weight.value_unsafe_ptr_cpu(), g5, "prew.head/logits/kernel")
    _set(h.children[3].bias.value_unsafe_ptr_cpu(), g5, "prew.head/logits/bias")

    var feat = _buf(_read(g5, "hd.feat"))
    var bins = _buf(_read(g5, "hd.bins"))
    var target = _read(g5, "hd.rew_target")
    var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * HBINS)
    var logits_t = TileTensor(logits, row_major[B, HBINS]())
    h.forward["cpu", B](TileTensor(feat, row_major[B, HFEAT]()), output=logits_t)
    # grad on logits = twohot CE backward (cotangent 1)
    var glog: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * HBINS)
    for i in range(B * HBINS):
        glog[i] = 0.0
    for b in range(B):
        twohot_loss_backward[HBINS](logits, b * HBINS, bins, target[b], 1.0, glog)
    var gfeat: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * HFEAT)
    var gfeat_t = TileTensor(gfeat, row_major[B, HFEAT]())
    h.vjp["cpu", B](TileTensor(glog, row_major[B, HBINS]()), gfeat_t)
    var df = _diff(gfeat, _read(g5, "hd.rew_g_feat"))
    var dk = _diff(h.children[0].weight.grad_unsafe_ptr_cpu(),
                   _read(g5, "grew.mlp/linear0/kernel"))
    print("  g_feat =", df, " lin0.k =", dk)
    assert_true(df < Scalar[DT](1e-4), "reward g_feat")
    assert_true(dk < Scalar[DT](1e-4), "reward lin0.kernel")
    print("  ok")


def test_cont() raises:
    print("cont MLP (Sequential[Linear,RMSNorm,GELU,Linear]) ...")
    var g5 = _lines(F5)
    var h = DreamerContMLP[HFEAT, HU].make["cpu", INIT=Zero]()
    _set(h.children[0].weight.value_unsafe_ptr_cpu(), g5, "pcon.mlp/linear0/kernel")
    _set(h.children[0].bias.value_unsafe_ptr_cpu(), g5, "pcon.mlp/linear0/bias")
    _set(h.children[1].gamma.value_unsafe_ptr_cpu(), g5, "pcon.mlp/norm0/scale")
    _set(h.children[3].weight.value_unsafe_ptr_cpu(), g5, "pcon.head/logit/kernel")
    _set(h.children[3].bias.value_unsafe_ptr_cpu(), g5, "pcon.head/logit/bias")

    var feat = _buf(_read(g5, "hd.feat"))
    var target = _read(g5, "hd.con_target")
    var logit: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var logit_t = TileTensor(logit, row_major[B, 1]())
    h.forward["cpu", B](TileTensor(feat, row_major[B, HFEAT]()), output=logit_t)
    # grad on logit = sigmoid(logit) - target
    var glog: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    for b in range(B):
        var s = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-logit[b]))
        glog[b] = s - target[b]
    var gfeat: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * HFEAT)
    var gfeat_t = TileTensor(gfeat, row_major[B, HFEAT]())
    h.vjp["cpu", B](TileTensor(glog, row_major[B, 1]()), gfeat_t)
    var df = _diff(gfeat, _read(g5, "hd.con_g_feat"))
    var dk = _diff(h.children[0].weight.grad_unsafe_ptr_cpu(),
                   _read(g5, "gcon.mlp/linear0/kernel"))
    print("  g_feat =", df, " lin0.k =", dk)
    assert_true(df < Scalar[DT](1e-4), "cont g_feat")
    assert_true(dk < Scalar[DT](1e-4), "cont lin0.kernel")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SPIKE chunk 1: DreamerV3 nets as Sequential (vs existing fixtures)")
    print("=" * 70)
    test_encoder()
    test_decoder()
    test_prior()
    test_reward()
    test_cont()
    print("=" * 70)
    print("ALL PASSED — nets compose as pure Sequential, match jax")
    print("=" * 70)
