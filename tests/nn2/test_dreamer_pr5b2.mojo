"""PR-5b part 2 parity: WM-module backward (encoder/decoder vjp) vs jax.vjp.

Params + inputs loaded from `pr4_fixture.txt` (identical construction);
gradient ground truth from `pr5b2_fixture.txt` (jax.vjp of the actual
reference). All ≤1e-4.

(RSSM `_core`/`_prior` vjp validated in test_dreamer_pr5b3.mojo once landed.)

Run: `pixi run mojo run -I . tests/nn2/test_dreamer_pr5b2.mojo`
"""

from std.memory import alloc
from std.math import log1p
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.encoder import Encoder
from mojo_rl.deep_agents2.dreamerv3.decoder import Decoder
from mojo_rl.deep_agents2.dreamerv3.heads import RewardHead, ContHead

comptime PR4 = "tests/nn2/dreamerv3/fixtures/pr4_fixture.txt"
comptime PR5B2 = "tests/nn2/dreamerv3/fixtures/pr5b2_fixture.txt"
comptime B = 2
comptime DETER = 16
comptime STOCH = 3
comptime CLASSES = 5
comptime SC = STOCH * CLASSES
comptime FEATIN = SC + DETER
comptime OBS = 4
comptime ENC_UNITS = 8
comptime DEC_UNITS = 8
comptime HFEAT = SC + DETER     # feat2tensor = concat([deter, stoch_flat])
comptime HU = 8
comptime HBINS = 255


def _split_lines(content: String) raises -> List[String]:
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    if current.byte_length() > 0:
        lines.append(current)
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
    raise Error("fixture: section not found: " + name)


def _buf(src: List[Scalar[DT]]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(src))
    for i in range(len(src)):
        p[i] = src[i]
    return p


def _load(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lines: List[String],
    name: String,
) raises:
    var v = _read_flat(lines, name)
    for i in range(len(v)):
        ptr[i] = v[i]


def _diff(
    got: UnsafePointer[Scalar[DT], MutAnyOrigin], expected: List[Scalar[DT]]
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(expected)):
        var d = got[i] - expected[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


def test_encoder_vjp() raises:
    print("test_encoder_vjp ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)

    var enc = Encoder[OBS, ENC_UNITS].make["cpu"]()
    _load(enc.lin0.weight.value_unsafe_ptr_cpu(), pl, "p.enc/mlp0/kernel")
    _load(enc.lin0.bias.value_unsafe_ptr_cpu(), pl, "p.enc/mlp0/bias")
    _load(enc.n0.gamma.value_unsafe_ptr_cpu(), pl, "p.enc/mlp0norm/scale")
    _load(enc.lin1.weight.value_unsafe_ptr_cpu(), pl, "p.enc/mlp1/kernel")
    _load(enc.lin1.bias.value_unsafe_ptr_cpu(), pl, "p.enc/mlp1/bias")
    _load(enc.n1.gamma.value_unsafe_ptr_cpu(), pl, "p.enc/mlp1norm/scale")

    var obs = _buf(_read_flat(pl, "in.obs_vec"))
    var g_tok = _buf(_read_flat(gl, "enc.g_tok"))
    var g_obs = alloc[Scalar[DT]](B * OBS)
    enc.vjp[B](obs, g_tok, g_obs)

    var dobs = _diff(g_obs, _read_flat(gl, "enc.g_obs"))
    var dk0 = _diff(enc.lin0.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "genc.mlp0/kernel"))
    var db0 = _diff(enc.lin0.bias.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "genc.mlp0/bias"))
    var ds0 = _diff(enc.n0.gamma.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "genc.mlp0norm/scale"))
    var dk1 = _diff(enc.lin1.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "genc.mlp1/kernel"))
    var ds1 = _diff(enc.n1.gamma.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "genc.mlp1norm/scale"))
    print("  g_obs", dobs, " k0", dk0, " b0", db0, " s0", ds0, " k1", dk1,
          " s1", ds1)
    assert_true(dobs < Scalar[DT](1e-4), "encoder grad_obs")
    assert_true(dk0 < Scalar[DT](1e-4), "encoder grad lin0.kernel")
    assert_true(db0 < Scalar[DT](1e-4), "encoder grad lin0.bias")
    assert_true(ds0 < Scalar[DT](1e-4), "encoder grad n0.scale")
    assert_true(dk1 < Scalar[DT](1e-4), "encoder grad lin1.kernel")
    assert_true(ds1 < Scalar[DT](1e-4), "encoder grad n1.scale")
    print("  ok")
    _ = obs


def test_decoder_vjp() raises:
    print("test_decoder_vjp ...")
    var p4: String
    with open(PR4, "r") as f:
        p4 = String(f.read())
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var pl = _split_lines(p4)
    var gl = _split_lines(g5)

    var dec = Decoder[FEATIN, OBS, DEC_UNITS].make["cpu"]()
    _load(dec.lin0.weight.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/linear0/kernel")
    _load(dec.lin0.bias.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/linear0/bias")
    _load(dec.n0.gamma.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/norm0/scale")
    _load(dec.lin1.weight.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/linear1/kernel")
    _load(dec.lin1.bias.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/linear1/bias")
    _load(dec.n1.gamma.value_unsafe_ptr_cpu(), pl, "p.dec/mlp/norm1/scale")
    _load(dec.pred.weight.value_unsafe_ptr_cpu(), pl, "p.dec/vec/vec/pred/kernel")
    _load(dec.pred.bias.value_unsafe_ptr_cpu(), pl, "p.dec/vec/vec/pred/bias")

    var deter = _buf(_read_flat(pl, "in.dec_deter"))
    var stoch = _buf(_read_flat(pl, "in.dec_stoch"))
    var target = _buf(_read_flat(pl, "in.recon_target"))

    var pred = alloc[Scalar[DT]](B * OBS)
    dec.forward[B](stoch, deter, SC, DETER, pred)
    # grad_pred = d(recon_loss)/d(pred) = 2·(pred − symlog(target))
    var grad_pred: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        B * OBS
    )
    for i in range(B * OBS):
        grad_pred[i] = Scalar[DT](2.0) * (pred[i] - _symlog(target[i]))

    var g_stoch = alloc[Scalar[DT]](B * SC)
    var g_deter = alloc[Scalar[DT]](B * DETER)
    dec.vjp[B](stoch, deter, SC, DETER, grad_pred, g_stoch, g_deter)

    var dd = _diff(g_deter, _read_flat(gl, "dec.g_deter"))
    var ds = _diff(g_stoch, _read_flat(gl, "dec.g_stoch"))
    var dk0 = _diff(dec.lin0.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gdec.mlp/linear0/kernel"))
    var dkp = _diff(dec.pred.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gdec.vec/vec/pred/kernel"))
    var dbp = _diff(dec.pred.bias.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gdec.vec/vec/pred/bias"))
    print("  g_deter", dd, " g_stoch", ds, " k0", dk0, " predK", dkp,
          " predB", dbp)
    assert_true(dd < Scalar[DT](1e-4), "decoder grad_deter")
    assert_true(ds < Scalar[DT](1e-4), "decoder grad_stoch")
    assert_true(dk0 < Scalar[DT](1e-4), "decoder grad lin0.kernel")
    assert_true(dkp < Scalar[DT](1e-4), "decoder grad pred.kernel")
    assert_true(dbp < Scalar[DT](1e-4), "decoder grad pred.bias")
    print("  ok")
    _ = deter; _ = stoch; _ = target


def _ones(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n)
    for i in range(n):
        p[i] = 1.0
    return p


def test_reward_head_vjp() raises:
    print("test_reward_head_vjp (twohot MLPHead) ...")
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var gl = _split_lines(g5)
    var h = RewardHead[HFEAT, HU, HBINS].make["cpu"]()
    _load(h.lin0.weight.value_unsafe_ptr_cpu(), gl, "prew.mlp/linear0/kernel")
    _load(h.lin0.bias.value_unsafe_ptr_cpu(), gl, "prew.mlp/linear0/bias")
    _load(h.n0.gamma.value_unsafe_ptr_cpu(), gl, "prew.mlp/norm0/scale")
    _load(h.logits.weight.value_unsafe_ptr_cpu(), gl, "prew.head/logits/kernel")
    _load(h.logits.bias.value_unsafe_ptr_cpu(), gl, "prew.head/logits/bias")
    var bins = _buf(_read_flat(gl, "hd.bins"))
    var feat = _buf(_read_flat(gl, "hd.feat"))
    var target = _buf(_read_flat(gl, "hd.rew_target"))
    var cot = _ones(B)
    var g_feat = alloc[Scalar[DT]](B * HFEAT)
    h.loss_vjp[B](feat, bins, target, cot, g_feat)
    var df = _diff(g_feat, _read_flat(gl, "hd.rew_g_feat"))
    var dk0 = _diff(h.lin0.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "grew.mlp/linear0/kernel"))
    var dkl = _diff(h.logits.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "grew.head/logits/kernel"))
    var dbl = _diff(h.logits.bias.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "grew.head/logits/bias"))
    print("  g_feat", df, " lin0.k", dk0, " logits.k", dkl, " logits.b", dbl)
    assert_true(df < Scalar[DT](1e-4), "reward grad_feat")
    assert_true(dk0 < Scalar[DT](1e-4), "reward grad lin0.kernel")
    assert_true(dkl < Scalar[DT](1e-4), "reward grad logits.kernel")
    assert_true(dbl < Scalar[DT](1e-4), "reward grad logits.bias")
    print("  ok")


def test_cont_head_vjp() raises:
    print("test_cont_head_vjp (binary MLPHead) ...")
    var g5: String
    with open(PR5B2, "r") as f:
        g5 = String(f.read())
    var gl = _split_lines(g5)
    var h = ContHead[HFEAT, HU].make["cpu"]()
    _load(h.lin0.weight.value_unsafe_ptr_cpu(), gl, "pcon.mlp/linear0/kernel")
    _load(h.lin0.bias.value_unsafe_ptr_cpu(), gl, "pcon.mlp/linear0/bias")
    _load(h.n0.gamma.value_unsafe_ptr_cpu(), gl, "pcon.mlp/norm0/scale")
    _load(h.logit.weight.value_unsafe_ptr_cpu(), gl, "pcon.head/logit/kernel")
    _load(h.logit.bias.value_unsafe_ptr_cpu(), gl, "pcon.head/logit/bias")
    var feat = _buf(_read_flat(gl, "hd.feat"))
    var target = _buf(_read_flat(gl, "hd.con_target"))
    var cot = _ones(B)
    var g_feat = alloc[Scalar[DT]](B * HFEAT)
    h.loss_vjp[B](feat, target, cot, g_feat)
    var df = _diff(g_feat, _read_flat(gl, "hd.con_g_feat"))
    var dk0 = _diff(h.lin0.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gcon.mlp/linear0/kernel"))
    var dkl = _diff(h.logit.weight.grad_unsafe_ptr_cpu(),
                    _read_flat(gl, "gcon.head/logit/kernel"))
    print("  g_feat", df, " lin0.k", dk0, " logit.k", dkl)
    assert_true(df < Scalar[DT](1e-4), "cont grad_feat")
    assert_true(dk0 < Scalar[DT](1e-4), "cont grad lin0.kernel")
    assert_true(dkl < Scalar[DT](1e-4), "cont grad logit.kernel")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-5b2 WM-module backward parity (encoder/decoder vjp vs jax.vjp)")
    print("=" * 70)
    test_encoder_vjp()
    test_decoder_vjp()
    test_reward_head_vjp()
    test_cont_head_vjp()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
