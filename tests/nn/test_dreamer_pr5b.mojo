"""PR-5b parity: AC + value-head loss BACKWARD vs jax.vjp of the reference.

Ground truth `pr5b_fixture.txt` from `extract_pr5b.py` (jax.vjp of the real
`imag_loss`/`repl_loss` + twohot CE, cotangent=ones). Validates the
actor-critic gradient (the most important backward) to ≤1e-4:

  * twohot CE backward (grad w.r.t. logits)  → value/reward head training
  * imag_loss backward → grad w.r.t. value logits, policy mean/std raw
  * repl_loss backward → grad w.r.t. value logits

The WM-module vjps (RSSM/enc/dec with forward-cache retention) are PR5b
part 2. Run: `pixi run mojo run -I . tests/nn/test_dreamer_pr5b.mojo`
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.twohot import twohot_loss_backward
from mojo_rl.deep_agents.dreamerv3.imag_loss import imag_loss_backward
from mojo_rl.deep_agents.dreamerv3.repl_loss import repl_loss_backward


comptime FIXTURE = "tests/nn/dreamerv3/fixtures/pr5b_fixture.txt"
comptime BK = 2
comptime T = 4
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


def _ones(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n)
    for i in range(n):
        p[i] = 1.0
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


def test_twohot_backward() raises:
    print("test_twohot_backward ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var bins = _buf(_read_flat(lines, "bins"))
    var logits = _buf(_read_flat(lines, "th.logits"))
    var target = _buf(_read_flat(lines, "th.target"))
    var n = BK * T
    var grad = alloc[Scalar[DT]](n * BINS)
    for i in range(n * BINS):
        grad[i] = 0.0
    for i in range(n):
        twohot_loss_backward[BINS](logits, i * BINS, bins, target[i], 1.0, grad)
    var d = _diff(grad, _read_flat(lines, "th.g_logits"))
    print("  g_logits diff =", d)
    assert_true(d < Scalar[DT](1e-4), "twohot CE backward parity")
    print("  ok")


def test_imag_loss_backward() raises:
    print("test_imag_loss_backward ...")
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
    var rscale = _get_scalar(lines, "il.rscale")

    var g_vlogits = alloc[Scalar[DT]](BK * T * BINS)
    var g_pmean = alloc[Scalar[DT]](BK * T * ACT)
    var g_pstd = alloc[Scalar[DT]](BK * T * ACT)
    var d_policy = _ones(BK * TM1)
    var d_value = _ones(BK * TM1)
    imag_loss_backward[BK, T, ACT, BINS](
        act, rew, con, vlogits, svlogits, pmean, pstd_raw, bins,
        minstd, maxstd, lam, actent, slowreg, rscale, d_policy, d_value,
        g_vlogits, g_pmean, g_pstd,
    )
    var dv = _diff(g_vlogits, _read_flat(lines, "il.g_vlogits"))
    var dm = _diff(g_pmean, _read_flat(lines, "il.g_pmean"))
    var ds = _diff(g_pstd, _read_flat(lines, "il.g_pstd_raw"))
    print("  g_vlogits diff =", dv, " g_pmean diff =", dm, " g_pstd diff =", ds)
    assert_true(dv < Scalar[DT](1e-4), "imag_loss grad_vlogits parity")
    assert_true(dm < Scalar[DT](1e-4), "imag_loss grad_pmean parity")
    assert_true(ds < Scalar[DT](1e-4), "imag_loss grad_pstd_raw parity")
    print("  ok")


def test_repl_loss_backward() raises:
    print("test_repl_loss_backward ...")
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

    var g_vlogits = alloc[Scalar[DT]](BK * T * BINS)
    var d_repval = _ones(BK * TM1)
    repl_loss_backward[BK, T, BINS](
        last, term, rew, boot, vlogits, svlogits, bins,
        horizon, lam, slowreg, d_repval, g_vlogits,
    )
    var dv = _diff(g_vlogits, _read_flat(lines, "rl.g_vlogits"))
    print("  g_vlogits diff =", dv)
    assert_true(dv < Scalar[DT](1e-4), "repl_loss grad_vlogits parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-5b AC + value-head loss backward parity (vs jax.vjp)")
    print("=" * 70)
    test_twohot_backward()
    test_imag_loss_backward()
    test_repl_loss_backward()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
