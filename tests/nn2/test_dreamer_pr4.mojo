"""PR-4 parity: RSSM + Encoder + Decoder forward/loss vs the ACTUAL reference.

Ground truth `tests/nn2/dreamerv3/fixtures/pr4_fixture.txt` is produced by
`extract_pr4.py`, which runs Hafner's reference `rssm.py` through ninjax
with `COMPUTE_DTYPE` forced to float32. We load the dumped params + inputs,
run our forward, and assert ≤1e-4.

THIS IS THE PR-4 HARD-STOP GATE — the legacy DreamerV3 port was never
validated, and this is the check that catches "we ported the wrong math".

Consolidates the plan's three test files (rssm_forward / rssm_loss /
encoder_decoder) — same fixture, same parser.

Run: `pixi run mojo run -I . tests/nn2/test_dreamer_pr4.mojo`
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.rssm import RSSM
from mojo_rl.deep_agents2.dreamerv3.encoder import Encoder
from mojo_rl.deep_agents2.dreamerv3.decoder import Decoder


comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/pr4_fixture.txt"

# Fixture dims (must match extract_pr4.py).
comptime B = 2
comptime DETER = 16
comptime HIDDEN = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime TOKEN = 8
comptime OBS = 4
comptime ENC_UNITS = 8
comptime DEC_UNITS = 8
comptime SC = STOCH * CLASSES          # 15
comptime FEATIN = SC + DETER           # 31


# ── fixture parsing ─────────────────────────────────────────────────────


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


def _max_diff_vs(
    got: UnsafePointer[Scalar[DT], MutAnyOrigin],
    expected: List[Scalar[DT]],
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(expected)):
        var d = got[i] - expected[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


# ── RSSM (load params once, reuse) ──────────────────────────────────────


def _load_rssm(mut m: RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN],
               lines: List[String]) raises:
    _load(m.dynin0.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin0/kernel")
    _load(m.dynin0.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin0/bias")
    _load(m.dynin0n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin0norm/scale")
    _load(m.dynin1.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin1/kernel")
    _load(m.dynin1.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin1/bias")
    _load(m.dynin1n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin1norm/scale")
    _load(m.dynin2.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin2/kernel")
    _load(m.dynin2.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin2/bias")
    _load(m.dynin2n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/dynin2norm/scale")
    _load(m.dynhid0.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/dynhid0/kernel")
    _load(m.dynhid0.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/dynhid0/bias")
    _load(m.dynhid0n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/dynhid0norm/scale")
    _load(m.dyngru.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/dyngru/kernel")
    _load(m.dyngru.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/dyngru/bias")
    _load(m.prior0.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/prior0/kernel")
    _load(m.prior0.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/prior0/bias")
    _load(m.prior0n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/prior0norm/scale")
    _load(m.prior1.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/prior1/kernel")
    _load(m.prior1.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/prior1/bias")
    _load(m.prior1n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/prior1norm/scale")
    _load(m.priorlogit.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/priorlogit/kernel")
    _load(m.priorlogit.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/priorlogit/bias")
    _load(m.obs0.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/obs0/kernel")
    _load(m.obs0.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/obs0/bias")
    _load(m.obs0n.gamma.value_unsafe_ptr_cpu(), lines, "p.rssm/obs0norm/scale")
    _load(m.obslogit.weight.value_unsafe_ptr_cpu(), lines, "p.rssm/obslogit/kernel")
    _load(m.obslogit.bias.value_unsafe_ptr_cpu(), lines, "p.rssm/obslogit/bias")


def test_rssm() raises:
    print("test_rssm (core / prior / observe / loss) ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    var m = RSSM[DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make[
        "cpu"
    ](unimix=0.01, free_nats=1.0)
    _load_rssm(m, lines)

    var deter0 = _buf(_read_flat(lines, "in.deter0"))
    var stoch0 = _buf(_read_flat(lines, "in.stoch0"))
    var action = _buf(_read_flat(lines, "in.action"))
    var tokens = _buf(_read_flat(lines, "in.tokens"))

    # _core
    var core = alloc[Scalar[DT]](B * DETER)
    m.core[B](deter0, stoch0, action, core)
    var dcore = _max_diff_vs(core, _read_flat(lines, "out.core"))
    print("  core diff =", dcore)
    assert_true(dcore < Scalar[DT](1e-4), "RSSM._core parity vs reference")

    # _prior(deter0)
    var pr = alloc[Scalar[DT]](B * SC)
    m.prior[B](deter0, pr)
    var dpr = _max_diff_vs(pr, _read_flat(lines, "out.prior"))
    print("  prior diff =", dpr)
    assert_true(dpr < Scalar[DT](1e-4), "RSSM._prior parity vs reference")

    # observe → obs_deter (== core) + obs_logit
    var od = alloc[Scalar[DT]](B * DETER)
    var ol = alloc[Scalar[DT]](B * SC)
    m.observe[B](deter0, stoch0, action, tokens, od, ol)
    var dod = _max_diff_vs(od, _read_flat(lines, "out.obs_deter"))
    var dol = _max_diff_vs(ol, _read_flat(lines, "out.obs_logit"))
    print("  obs_deter diff =", dod, " obs_logit diff =", dol)
    assert_true(dod < Scalar[DT](1e-4), "RSSM.observe deter parity")
    assert_true(dol < Scalar[DT](1e-4), "RSSM.observe obslogit parity")

    # loss dyn/rep (MutAnyOrigin l-values — OneHotKL.forward takes `mut`)
    var dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    m.loss[B](deter0, stoch0, action, tokens, dyn, rep)
    var ddyn = _max_diff_vs(dyn, _read_flat(lines, "out.dyn"))
    var drep = _max_diff_vs(rep, _read_flat(lines, "out.rep"))
    print("  dyn diff =", ddyn, " rep diff =", drep)
    assert_true(ddyn < Scalar[DT](1e-4), "RSSM dyn loss parity")
    assert_true(drep < Scalar[DT](1e-4), "RSSM rep loss parity")
    print("  ok")
    _ = deter0; _ = stoch0; _ = action; _ = tokens


def test_encoder() raises:
    print("test_encoder ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    var enc = Encoder[OBS, ENC_UNITS].make["cpu"]()
    _load(enc.lin0.weight.value_unsafe_ptr_cpu(), lines, "p.enc/mlp0/kernel")
    _load(enc.lin0.bias.value_unsafe_ptr_cpu(), lines, "p.enc/mlp0/bias")
    _load(enc.n0.gamma.value_unsafe_ptr_cpu(), lines, "p.enc/mlp0norm/scale")
    _load(enc.lin1.weight.value_unsafe_ptr_cpu(), lines, "p.enc/mlp1/kernel")
    _load(enc.lin1.bias.value_unsafe_ptr_cpu(), lines, "p.enc/mlp1/bias")
    _load(enc.n1.gamma.value_unsafe_ptr_cpu(), lines, "p.enc/mlp1norm/scale")

    var obs = _buf(_read_flat(lines, "in.obs_vec"))
    var tok = alloc[Scalar[DT]](B * ENC_UNITS)
    enc.forward[B](obs, tok)
    var d = _max_diff_vs(tok, _read_flat(lines, "out.enc_tok"))
    print("  enc_tok diff =", d)
    assert_true(d < Scalar[DT](1e-4), "Encoder parity vs reference")
    print("  ok")
    _ = obs


def test_decoder() raises:
    print("test_decoder ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    var dec = Decoder[FEATIN, OBS, DEC_UNITS].make["cpu"]()
    _load(dec.lin0.weight.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/linear0/kernel")
    _load(dec.lin0.bias.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/linear0/bias")
    _load(dec.n0.gamma.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/norm0/scale")
    _load(dec.lin1.weight.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/linear1/kernel")
    _load(dec.lin1.bias.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/linear1/bias")
    _load(dec.n1.gamma.value_unsafe_ptr_cpu(), lines, "p.dec/mlp/norm1/scale")
    _load(dec.pred.weight.value_unsafe_ptr_cpu(), lines, "p.dec/vec/vec/pred/kernel")
    _load(dec.pred.bias.value_unsafe_ptr_cpu(), lines, "p.dec/vec/vec/pred/bias")

    var deter = _buf(_read_flat(lines, "in.dec_deter"))
    var stoch = _buf(_read_flat(lines, "in.dec_stoch"))
    var target = _buf(_read_flat(lines, "in.recon_target"))
    var pred = alloc[Scalar[DT]](B * OBS)
    dec.forward[B](stoch, deter, SC, DETER, pred)
    var dp = _max_diff_vs(pred, _read_flat(lines, "out.dec_pred"))
    print("  dec_pred diff =", dp)
    assert_true(dp < Scalar[DT](1e-4), "Decoder pred parity vs reference")

    var rl = alloc[Scalar[DT]](B)
    Decoder[FEATIN, OBS, DEC_UNITS].recon_loss[B](pred, target, rl)
    var dl = _max_diff_vs(rl, _read_flat(lines, "out.recon_loss"))
    print("  recon_loss diff =", dl)
    assert_true(dl < Scalar[DT](1e-4), "Decoder recon_loss parity vs reference")
    print("  ok")
    _ = deter; _ = stoch; _ = target


def main() raises:
    print("=" * 70)
    print("PR-4 DreamerV3 RSSM/Encoder/Decoder parity (vs ACTUAL reference)")
    print("=" * 70)
    test_rssm()
    test_encoder()
    test_decoder()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
