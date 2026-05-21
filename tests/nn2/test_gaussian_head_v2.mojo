"""Parity test: GaussianHeadV2 matches v1 GaussianHead bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.primitives.gaussian_head_v2 import GaussianHeadV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4
comptime IN  = 6
comptime ACT = 3


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    var h1 = GaussianHead[IN, ACT].make[target="cpu", INIT=Kaiming]()
    var h2 = GaussianHeadV2[IN, ACT].make[target="cpu", INIT=Zero]()

    # Mirror h1's params into h2.
    var w_size = IN * ACT
    var w1 = h1.weight.unsafe_ptr()
    var b1 = h1.bias.unsafe_ptr()
    var l1 = h1.log_std.unsafe_ptr()
    var w2 = h2.weight.value_unsafe_ptr_cpu()
    var b2 = h2.bias.value_unsafe_ptr_cpu()
    var l2 = h2.log_std.value_unsafe_ptr_cpu()
    for k in range(w_size):
        w2[k] = w1[k]
    for k in range(ACT):
        b2[k] = b1[k]
        # Pick a log_std outside [LOG_STD_MIN, LOG_STD_MAX] to exercise clamp
        l1[k] = Scalar[DT](k) * Scalar[DT](2.5) - Scalar[DT](3.0)
        l2[k] = l1[k]

    # ── Forward parity ───────────────────────────────────────────────
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var o1_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var o2_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    _seed(in_p, BATCH * IN, UInt64(0x77777777))

    var in_t  = TileTensor(in_p, row_major[BATCH, IN]())
    var o1_t  = TileTensor(o1_p, row_major[BATCH, 2 * ACT]())
    var o2_t  = TileTensor(o2_p, row_major[BATCH, 2 * ACT]())
    h1.forward["cpu", BATCH](in_t, o1_t)
    h2.forward["cpu", BATCH](in_t, o2_t)

    var ok_fwd = True
    for k in range(BATCH * 2 * ACT):
        if o1_p[k] != o2_p[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward(mode='all') ─────────────────────────────────────────
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    _seed(go_p, BATCH * 2 * ACT, UInt64(0x99999999))
    for k in range(BATCH * IN):
        gi1_p[k] = Scalar[DT](0.0)
        gi2_p[k] = Scalar[DT](0.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, 2 * ACT]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, IN]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, IN]())
    h1.backward["cpu", BATCH](go_t, gi1_t)
    h2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

    var ok_gi = True
    for k in range(BATCH * IN):
        if gi1_p[k] != gi2_p[k]:
            ok_gi = False
    var ok_gw = True
    var gw1 = h1.grad_w.unsafe_ptr()
    var gw2 = h2.weight.grad_unsafe_ptr_cpu()
    for k in range(w_size):
        if gw1[k] != gw2[k]:
            ok_gw = False
    var ok_gb = True
    var ok_gls = True
    var gb1 = h1.grad_b.unsafe_ptr()
    var gb2 = h2.bias.grad_unsafe_ptr_cpu()
    var gl1 = h1.grad_ls.unsafe_ptr()
    var gl2 = h2.log_std.grad_unsafe_ptr_cpu()
    for k in range(ACT):
        if gb1[k] != gb2[k]:
            ok_gb = False
        if gl1[k] != gl2[k]:
            ok_gls = False
    print(
        "backward(all): grad_in=", "PASS" if ok_gi else "FAIL",
        " grad_w=",   "PASS" if ok_gw else "FAIL",
        " grad_b=",   "PASS" if ok_gb else "FAIL",
        " grad_log_std=", "PASS" if ok_gls else "FAIL",
    )

    # ── Backward(mode='input_only') ──────────────────────────────────
    var gw_pre = List[Scalar[DT]]()
    var gb_pre = List[Scalar[DT]]()
    var gl_pre = List[Scalar[DT]]()
    for k in range(w_size):
        gw_pre.append(gw2[k])
    for k in range(ACT):
        gb_pre.append(gb2[k])
        gl_pre.append(gl2[k])

    var gi1b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi2b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * IN):
        gi1b[k] = Scalar[DT](0.0)
        gi2b[k] = Scalar[DT](0.0)
    var gi1bt = TileTensor(gi1b, row_major[BATCH, IN]())
    var gi2bt = TileTensor(gi2b, row_major[BATCH, IN]())
    h1.backward_input["cpu", BATCH](go_t, gi1bt)
    h2.backward["cpu", BATCH, mode="input_only"](go_t, gi2bt)

    var ok_bwi = True
    for k in range(BATCH * IN):
        if gi1b[k] != gi2b[k]:
            ok_bwi = False
    var ok_clean = True
    for k in range(w_size):
        if gw2[k] != gw_pre[k]:
            ok_clean = False
    for k in range(ACT):
        if gb2[k] != gb_pre[k] or gl2[k] != gl_pre[k]:
            ok_clean = False
    print(
        "backward(input_only): grad_in=", "PASS" if ok_bwi else "FAIL",
        " params_unchanged=", "PASS" if ok_clean else "FAIL",
    )

    # ── zero_grad ────────────────────────────────────────────────────
    h2.zero_grad[target="cpu"]()
    var ok_zg = True
    for k in range(w_size):
        if gw2[k] != Scalar[DT](0.0):
            ok_zg = False
    for k in range(ACT):
        if gb2[k] != Scalar[DT](0.0) or gl2[k] != Scalar[DT](0.0):
            ok_zg = False
    print("zero_grad: PASS" if ok_zg else "zero_grad: FAIL")

    var all_ok = (
        ok_fwd and ok_gi and ok_gw and ok_gb and ok_gls
        and ok_bwi and ok_clean and ok_zg
    )
    if all_ok:
        print()
        print("PASS — GaussianHeadV2 is bit-identical to v1 on CPU.")
    else:
        raise Error("gaussian_head_v2 parity test failed")

    in_p.free()
    o1_p.free()
    o2_p.free()
    go_p.free()
    gi1_p.free()
    gi2_p.free()
    gi1b.free()
    gi2b.free()
