"""PR-2 DreamerV3 math-primitive parity vs real jax.

Ground truth: `tests/nn/dreamerv3/fixtures/pr2_fixture.txt` (extract_pr2.py).
Covers OneHotKL forward+backward + PercentileNormalize. (The standalone
`lambda_return` op was retired 2026-05-29 — the live AC path inlines the
λ-return recurrence in `imag_loss`; its fixture section is left in place.)
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.onehot_kl import OneHotKL
from mojo_rl.deep_agents.dreamerv3.normalize import PercentileNormalize


comptime FIXTURE = "tests/nn/dreamerv3/fixtures/pr2_fixture.txt"


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


def _get_scalar(lines: List[String], key: String) raises -> Float64:
    var pfx = key + "="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            return atof(String(lines[i][byte=pfx.byte_length():]))
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


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def _buf(src: List[Scalar[DT]]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(src))
    for i in range(len(src)):
        p[i] = src[i]
    return p


# ── OneHotKL ──────────────────────────────────────────────────────────────


def test_onehot_kl() raises:
    print("test_onehot_kl ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    comptime KB = 2
    comptime STOCH = 3
    comptime CLASSES = 4
    comptime G = STOCH * CLASSES
    var unimix = Scalar[DT](_get_scalar(lines, "kl.unimix"))
    var free = Scalar[DT](_get_scalar(lines, "kl.free_nats"))
    var post = _buf(_read_flat(lines, "kl.post"))
    var prior = _buf(_read_flat(lines, "kl.prior"))
    var dyn_ref = _read_flat(lines, "kl.dyn")
    var rep_ref = _read_flat(lines, "kl.rep")
    var gpost_ref = _read_flat(lines, "kl.gpost")
    var gprior_ref = _read_flat(lines, "kl.gprior")

    var kl = OneHotKL[STOCH, CLASSES].make(unimix, free)
    var dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB)
    var rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB)
    kl.forward[KB](post, prior, dyn, rep)

    var got_dyn = List[Scalar[DT]]()
    var got_rep = List[Scalar[DT]]()
    for b in range(KB):
        got_dyn.append(dyn[b])
        got_rep.append(rep[b])
    var dd = _max_abs_diff(got_dyn, dyn_ref)
    var dr = _max_abs_diff(got_rep, rep_ref)
    print("  dyn diff =", dd, " rep diff =", dr)
    assert_true(dd < Scalar[DT](1e-4), "OneHotKL dyn forward parity")
    assert_true(dr < Scalar[DT](1e-4), "OneHotKL rep forward parity")

    # Upstream d_dyn = d_rep = 1 (matches fixture loss = Σ dyn + Σ rep).
    var d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB)
    var d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB)
    for b in range(KB):
        d_dyn[b] = 1.0
        d_rep[b] = 1.0
    var gpost: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB * G)
    var gprior: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](KB * G)
    kl.backward[KB](d_dyn, d_rep, gpost, gprior)

    var got_gpost = List[Scalar[DT]]()
    var got_gprior = List[Scalar[DT]]()
    for i in range(KB * G):
        got_gpost.append(gpost[i])
        got_gprior.append(gprior[i])
    var dgp = _max_abs_diff(got_gpost, gpost_ref)
    var dgpr = _max_abs_diff(got_gprior, gprior_ref)
    print("  gpost diff =", dgp, " gprior diff =", dgpr)
    assert_true(dgp < Scalar[DT](1e-4), "OneHotKL grad_post parity")
    assert_true(dgpr < Scalar[DT](1e-4), "OneHotKL grad_prior parity")
    print("  ok")


# ── PercentileNormalize ──────────────────────────────────────────────────


def test_percentile_normalize() raises:
    print("test_percentile_normalize ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var rate = Scalar[DT](_get_scalar(lines, "pn.rate"))
    var perclo = Scalar[DT](_get_scalar(lines, "pn.perclo"))
    var perchi = Scalar[DT](_get_scalar(lines, "pn.perchi"))
    var limit = Scalar[DT](_get_scalar(lines, "pn.limit"))
    var n_updates = Int(_get_scalar(lines, "pn.n_updates"))
    var sample = Int(_get_scalar(lines, "pn.sample_size"))
    var offset_ref = Scalar[DT](_get_scalar(lines, "pn.offset"))
    var scale_ref = Scalar[DT](_get_scalar(lines, "pn.scale"))
    var inputs = _buf(_read_flat(lines, "pn.inputs"))

    # retnorm config: debias=False.
    var pn = PercentileNormalize.make(
        String("perc"), rate, perclo, perchi, limit, debias=False
    )
    for u in range(n_updates):
        var chunk = inputs + (u * sample)
        pn.update(chunk, sample)
    var st = pn.stats()
    var off = st[0]
    var sc = st[1]
    var doff = off - offset_ref
    var adoff = doff if doff >= Scalar[DT](0) else -doff
    var dsc = sc - scale_ref
    var adsc = dsc if dsc >= Scalar[DT](0) else -dsc
    print("  offset", off, "vs", offset_ref, " scale", sc, "vs", scale_ref)
    assert_true(adoff < Scalar[DT](1e-4), "PercentileNormalize offset parity")
    assert_true(adsc < Scalar[DT](1e-4), "PercentileNormalize scale parity")

    # 'none' impl → identity (0, 1).
    var none = PercentileNormalize.make(String("none"))
    var st0 = none.stats()
    assert_true(st0[0] == Scalar[DT](0.0), "none offset == 0")
    assert_true(st0[1] == Scalar[DT](1.0), "none scale == 1")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-2 DreamerV3 math primitives (vs jax)")
    print("=" * 70)
    test_onehot_kl()
    test_percentile_normalize()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
