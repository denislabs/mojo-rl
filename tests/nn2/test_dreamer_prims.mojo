"""PR-1 DreamerV3 primitive parity tests vs real jax.

Ground truth: `tests/nn2/dreamerv3/fixtures/prims_fixture.txt` (generated
by `extract_prims.py` under real jax). Each primitive's forward + vjp are
compared to jax's forward + `jax.vjp` gradients at ≤1e-4.

Covers: GELU (tanh approx), SiLU, RMSNorm, BlockLinear.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.element_op import ElementOp
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.primitives.block_linear import BlockLinear
from mojo_rl.nn2.initializer import Zero


comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/prims_fixture.txt"


# ── Fixture parsing helpers ─────────────────────────────────────────────


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


def _list_to_buf(
    src: List[Scalar[DT]],
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](len(src))
    for i in range(len(src)):
        p[i] = src[i]
    return p


# ── Elementwise op parity (GELU, SiLU) ─────────────────────────────────


def _check_elementwise[
    OP: ElementOp, DIM: Int
](lines: List[String], name: String) raises:
    comptime BATCH = 1
    var xs = _read_flat(lines, name + ".x")
    var gos = _read_flat(lines, name + ".go")
    var y_ref = _read_flat(lines, name + ".y")
    var gx_ref = _read_flat(lines, name + ".gx")
    var n = len(xs)

    var op = Elementwise[DIM, OP].make[target="cpu", INIT=Zero]()
    var x = _list_to_buf(xs)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n)
    var go = _list_to_buf(gos)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    op.forward["cpu", BATCH](x_t, output=y_t)
    op.vjp["cpu", BATCH](go_t, gi_t)

    var got_y = List[Scalar[DT]]()
    var got_gx = List[Scalar[DT]]()
    for i in range(n):
        got_y.append(y[i])
        got_gx.append(gi[i])
    var df = _max_abs_diff(got_y, y_ref)
    var db = _max_abs_diff(got_gx, gx_ref)
    print("  " + name + " fwd diff =", df, " bwd diff =", db)
    assert_true(df < Scalar[DT](1e-4), name + " forward parity vs jax")
    assert_true(db < Scalar[DT](1e-4), name + " backward parity vs jax")
    _ = x  # keep input slab alive through vjp (input-cache alias)


def test_gelu_parity() raises:
    print("test_gelu_parity ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    _check_elementwise[GELUOp, 32](lines, "gelu")
    print("  ok")


def test_silu_parity() raises:
    print("test_silu_parity ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    _check_elementwise[SwishOp, 32](lines, "silu")
    print("  ok")


# ── RMSNorm parity ──────────────────────────────────────────────────────


def test_rmsnorm_parity() raises:
    print("test_rmsnorm_parity ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    comptime BATCH = 3
    comptime DIM = 8
    comptime N = BATCH * DIM
    var xs = _read_flat(lines, "rms.x")
    var gamma_fix = _read_flat(lines, "rms.gamma")
    var gos = _read_flat(lines, "rms.go")
    var y_ref = _read_flat(lines, "rms.y")
    var gx_ref = _read_flat(lines, "rms.gx")
    var ggamma_ref = _read_flat(lines, "rms.ggamma")

    var rn = RMSNorm[DIM].make[target="cpu", INIT=Zero]()
    # Overwrite γ (make() sets it to 1) with the fixture's random γ.
    var g_ptr = rn.gamma.value_unsafe_ptr_cpu()
    for k in range(DIM):
        g_ptr[k] = gamma_fix[k]

    var x = _list_to_buf(xs)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go = _list_to_buf(gos)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    rn.forward["cpu", BATCH](x_t, output=y_t)
    rn.zero_grad["cpu"]()
    rn.vjp["cpu", BATCH](go_t, gi_t)

    var got_y = List[Scalar[DT]]()
    var got_gx = List[Scalar[DT]]()
    var got_gg = List[Scalar[DT]]()
    for i in range(N):
        got_y.append(y[i])
        got_gx.append(gi[i])
    for k in range(DIM):
        got_gg.append(rn.gamma.grad[k])

    var df = _max_abs_diff(got_y, y_ref)
    var db = _max_abs_diff(got_gx, gx_ref)
    var dg = _max_abs_diff(got_gg, ggamma_ref)
    print("  rms fwd diff =", df, " gx diff =", db, " ggamma diff =", dg)
    assert_true(df < Scalar[DT](1e-4), "RMSNorm forward parity vs jax")
    assert_true(db < Scalar[DT](1e-4), "RMSNorm grad_x parity vs jax")
    assert_true(dg < Scalar[DT](1e-4), "RMSNorm grad_gamma parity vs jax")
    _ = x
    print("  ok")


# ── BlockLinear parity ──────────────────────────────────────────────────


def test_blocklinear_parity() raises:
    print("test_blocklinear_parity ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    comptime BATCH = 4
    comptime IN = 12
    comptime OUT = 16
    comptime BLK = 4
    var xs = _read_flat(lines, "bl.x")
    var kernel = _read_flat(lines, "bl.kernel")      # [BLK, IN/BLK, OUT/BLK]
    var bias_fix = _read_flat(lines, "bl.bias")
    var gos = _read_flat(lines, "bl.go")
    var y_ref = _read_flat(lines, "bl.y")
    var gx_ref = _read_flat(lines, "bl.gx")
    var gk_ref = _read_flat(lines, "bl.gkernel")
    var gb_ref = _read_flat(lines, "bl.gbias")

    var bl = BlockLinear[IN, OUT, BLK].make[target="cpu", INIT=Zero]()
    var w_ptr = bl.weight.value_unsafe_ptr_cpu()
    for k in range(len(kernel)):
        w_ptr[k] = kernel[k]
    var b_ptr = bl.bias.value_unsafe_ptr_cpu()
    for k in range(OUT):
        b_ptr[k] = bias_fix[k]

    var x = _list_to_buf(xs)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go = _list_to_buf(gos)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, OUT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN]())

    bl.forward["cpu", BATCH](x_t, output=y_t)
    bl.zero_grad["cpu"]()
    bl.vjp["cpu", BATCH](go_t, gi_t)

    var got_y = List[Scalar[DT]]()
    var got_gx = List[Scalar[DT]]()
    var got_gk = List[Scalar[DT]]()
    var got_gb = List[Scalar[DT]]()
    for i in range(BATCH * OUT):
        got_y.append(y[i])
    for i in range(BATCH * IN):
        got_gx.append(gi[i])
    for k in range(len(kernel)):
        got_gk.append(bl.weight.grad[k])
    for k in range(OUT):
        got_gb.append(bl.bias.grad[k])

    var df = _max_abs_diff(got_y, y_ref)
    var dgx = _max_abs_diff(got_gx, gx_ref)
    var dgk = _max_abs_diff(got_gk, gk_ref)
    var dgb = _max_abs_diff(got_gb, gb_ref)
    print(
        "  bl fwd diff =", df, " gx =", dgx, " gkernel =", dgk, " gbias =", dgb
    )
    assert_true(df < Scalar[DT](1e-4), "BlockLinear forward parity vs jax")
    assert_true(dgx < Scalar[DT](1e-4), "BlockLinear grad_x parity vs jax")
    assert_true(dgk < Scalar[DT](1e-4), "BlockLinear grad_kernel parity vs jax")
    assert_true(dgb < Scalar[DT](1e-4), "BlockLinear grad_bias parity vs jax")
    _ = x
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-1 DreamerV3 primitive parity (vs jax)")
    print("=" * 70)
    test_gelu_parity()
    test_silu_parity()
    test_rmsnorm_parity()
    test_blocklinear_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
