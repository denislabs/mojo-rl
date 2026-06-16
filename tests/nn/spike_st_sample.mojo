"""SPIKE: StraightThroughSample backward vs jax (st_fixture).

Forward = argmax one-hot (placeholder; trainer wires Philox sample). The
backward — the trainable straight-through path — is exact: grad flows through
the unimix softmax, independent of the sampled index. Validated ≤1e-4.

Run: `pixi run mojo run -I . tests/nn/spike_st_sample.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.deep_agents.dreamerv3.rssm_ops import StraightThroughSample

comptime FIX = "tests/nn/dreamerv3/fixtures/st_fixture.txt"
comptime B = 2
comptime STOCH = 3
comptime CLASSES = 5
comptime SC = STOCH * CLASSES


def _lines() raises -> List[String]:
    var content: String
    with open(FIX, "r") as f:
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


def main() raises:
    print("SPIKE: StraightThroughSample backward vs jax ...")
    var lines = _lines()
    var st = StraightThroughSample[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var z = _buf(_read(lines, "st.z"))
    var cot = _buf(_read(lines, "st.cot"))
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var out_t = TileTensor(out, row_major[B, SC]())
    st.forward["cpu", B](TileTensor(z, row_major[B, SC]()), output=out_t)
    var gz: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * SC)
    var gz_t = TileTensor(gz, row_major[B, SC]())
    st.vjp["cpu", B](TileTensor(cot, row_major[B, SC]()), gz_t)
    var expg = _read(lines, "st.g_z")
    var m: Scalar[DT] = 0.0
    for i in range(len(expg)):
        var d = gz[i] - expg[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    print("  grad_z diff =", m)
    assert_true(m < Scalar[DT](1e-4), "StraightThroughSample backward parity")
    print("  ok — ST backward matches jax")
