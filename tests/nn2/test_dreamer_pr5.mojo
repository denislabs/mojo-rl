"""PR-5a parity: distributional heads + AC losses vs the ACTUAL reference.

Ground truth `tests/nn2/dreamerv3/fixtures/pr5_fixture.txt` from
`extract_pr5.py` (reference `outs.TwoHot/Normal`, `agent.imag_loss/repl_loss`,
`utils.Normalize`/`SlowModel`). All ≤1e-4.

Covers: TwoHot pred/loss, symexp bin generation, bounded_normal logp/entropy,
imag_loss (policy/value/ret), repl_loss (repval/ret). (The standalone
`SlowModelHead` Polyak prototype was retired 2026-05-29 — the live trainer
uses `polyak_module` from `polyak.mojo`; its `sm.*` fixture is left in place.)

Run: `pixi run mojo run -I . tests/nn2/test_dreamer_pr5.mojo`
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.twohot import (
    twohot_pred, twohot_loss, symexp_twohot_bins,
)
from mojo_rl.deep_agents2.dreamerv3.dists import (
    bounded_mean, bounded_std, normal_logp, normal_entropy,
)
from mojo_rl.deep_agents2.dreamerv3.imag_loss import imag_loss_cpu
from mojo_rl.deep_agents2.dreamerv3.repl_loss import repl_loss_cpu
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize


comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/pr5_fixture.txt"
comptime BK = 2
comptime H = 3
comptime T = 4          # H+1
comptime ACT = 1
comptime BINS = 41
comptime TM1 = T - 1


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


def _get_scalar(lines: List[String], key: String) raises -> Scalar[DT]:
    var pfx = key + "="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            return Scalar[DT](atof(String(lines[i][byte=pfx.byte_length():])))
    raise Error("fixture: key not found: " + key)


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


def test_twohot() raises:
    print("test_twohot (pred / loss / bins) ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    var bins = _buf(_read_flat(lines, "bins"))
    var logits = _buf(_read_flat(lines, "th.logits"))
    var target = _buf(_read_flat(lines, "th.target"))
    var n = BK * T

    var pred = alloc[Scalar[DT]](n)
    var loss = alloc[Scalar[DT]](n)
    for i in range(n):
        pred[i] = twohot_pred[BINS](logits, i * BINS, bins)
        loss[i] = twohot_loss[BINS](logits, i * BINS, bins, target[i])
    var dp = _diff(pred, _read_flat(lines, "th.pred"))
    var dl = _diff(loss, _read_flat(lines, "th.loss"))
    print("  pred diff =", dp, " loss diff =", dl)
    assert_true(dp < Scalar[DT](1e-4), "TwoHot pred parity")
    assert_true(dl < Scalar[DT](1e-4), "TwoHot loss parity")

    # symexp bin generation (255). Bins span ±4.85e8 → use RELATIVE diff
    # (absolute float32 error at that magnitude is ~hundreds, but relative
    # ~5e-7).
    var sbins: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](255)
    symexp_twohot_bins[255](sbins)
    var sref = _read_flat(lines, "sbins255")
    var dsr: Scalar[DT] = 0.0
    for i in range(255):
        var d = sbins[i] - sref[i]
        var ad = d if d >= Scalar[DT](0) else -d
        var ar = sref[i] if sref[i] >= Scalar[DT](0) else -sref[i]
        var rel = ad / (ar + Scalar[DT](1.0))
        if rel > dsr:
            dsr = rel
    print("  symexp bins(255) rel diff =", dsr)
    assert_true(dsr < Scalar[DT](1e-4), "symexp_twohot_bins parity (relative)")
    print("  ok")


def test_normal() raises:
    print("test_normal (bounded_normal logp / entropy) ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var minstd = _get_scalar(lines, "cfg.minstd")
    var maxstd = _get_scalar(lines, "cfg.maxstd")
    var mean_raw = _buf(_read_flat(lines, "nm.mean_raw"))
    var std_raw = _buf(_read_flat(lines, "nm.std_raw"))
    var act = _buf(_read_flat(lines, "nm.act"))
    var n = BK * T * ACT

    var logp = alloc[Scalar[DT]](n)
    var ent = alloc[Scalar[DT]](n)
    for i in range(n):
        var mean = bounded_mean(mean_raw[i])
        var std = bounded_std(std_raw[i], minstd, maxstd)
        logp[i] = normal_logp(act[i], mean, std)
        ent[i] = normal_entropy(std)
    var dlp = _diff(logp, _read_flat(lines, "nm.logp"))
    var de = _diff(ent, _read_flat(lines, "nm.entropy"))
    print("  logp diff =", dlp, " entropy diff =", de)
    assert_true(dlp < Scalar[DT](1e-4), "Normal logp parity")
    assert_true(de < Scalar[DT](1e-4), "Normal entropy parity")
    print("  ok")


def test_imag_loss() raises:
    print("test_imag_loss (policy / value / ret) ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var bins = _buf(_read_flat(lines, "bins"))
    var act = _buf(_read_flat(lines, "il.act"))
    var rew = _buf(_read_flat(lines, "il.rew"))
    var con = _buf(_read_flat(lines, "il.con"))
    var vlogits = _buf(_read_flat(lines, "il.vlogits"))
    var svlogits = _buf(_read_flat(lines, "il.svlogits"))
    var pmean = _buf(_read_flat(lines, "il.pmean"))
    var pstd_raw = _buf(_read_flat(lines, "il.pstd_raw"))
    var minstd = _get_scalar(lines, "cfg.minstd")
    var maxstd = _get_scalar(lines, "cfg.maxstd")
    var lam = _get_scalar(lines, "cfg.lam")
    var actent = _get_scalar(lines, "cfg.actent")
    var slowreg = _get_scalar(lines, "cfg.slowreg")

    var retnorm = PercentileNormalize.make(
        String("perc"), rate=0.01, perclo=5.0, perchi=95.0, limit=1.0,
        debias=False,
    )
    var pol = alloc[Scalar[DT]](BK * TM1)
    var vall = alloc[Scalar[DT]](BK * TM1)
    var ret = alloc[Scalar[DT]](BK * TM1)
    imag_loss_cpu[BK, T, ACT, BINS](
        act, rew, con, vlogits, svlogits, pmean, pstd_raw, bins,
        minstd, maxstd, lam, actent, slowreg, retnorm, pol, vall, ret,
    )
    var dpol = _diff(pol, _read_flat(lines, "il.policy_loss"))
    var dval = _diff(vall, _read_flat(lines, "il.value_loss"))
    var dret = _diff(ret, _read_flat(lines, "il.ret"))
    print("  policy diff =", dpol, " value diff =", dval, " ret diff =", dret)
    assert_true(dpol < Scalar[DT](1e-4), "imag_loss policy parity")
    assert_true(dval < Scalar[DT](1e-4), "imag_loss value parity")
    assert_true(dret < Scalar[DT](1e-4), "imag_loss ret parity")
    print("  ok")


def test_repl_loss() raises:
    print("test_repl_loss (repval / ret) ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var bins = _buf(_read_flat(lines, "bins"))
    var last = _buf(_read_flat(lines, "rl.last"))
    var term = _buf(_read_flat(lines, "rl.term"))
    var rew = _buf(_read_flat(lines, "rl.rew"))
    var boot = _buf(_read_flat(lines, "rl.boot"))
    var vlogits = _buf(_read_flat(lines, "rl.vlogits"))
    var svlogits = _buf(_read_flat(lines, "rl.svlogits"))
    var horizon = _get_scalar(lines, "cfg.horizon")
    var lam = _get_scalar(lines, "cfg.lam")
    var slowreg = _get_scalar(lines, "cfg.slowreg")

    var repval = alloc[Scalar[DT]](BK * TM1)
    var ret = alloc[Scalar[DT]](BK * TM1)
    repl_loss_cpu[BK, T, BINS](
        last, term, rew, boot, vlogits, svlogits, bins,
        horizon, lam, slowreg, repval, ret,
    )
    var drv = _diff(repval, _read_flat(lines, "rl.repval"))
    var dret = _diff(ret, _read_flat(lines, "rl.ret"))
    print("  repval diff =", drv, " ret diff =", dret)
    assert_true(drv < Scalar[DT](1e-4), "repl_loss repval parity")
    assert_true(dret < Scalar[DT](1e-4), "repl_loss ret parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-5a DreamerV3 heads + AC losses parity (vs ACTUAL reference)")
    print("=" * 70)
    test_twohot()
    test_normal()
    test_imag_loss()
    test_repl_loss()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
