"""Tests for Transpose2DOp, TokenMean, PatchEmbed, and ViT composites.

Verifies:
  1. Transpose2DOp:
     - dimension invariants
     - bitwise round-trip: Transpose2DOp[B, A] o Transpose2DOp[A, B] == identity
     - finite-difference gradcheck
  2. TokenMean:
     - shape invariants
     - forward matches the reference mean
     - finite-difference gradcheck (CPU)
  3. PatchEmbed: shape propagation Conv2D → Transpose2D matches the attention
     IN_DIM convention.
  4. ViT: shape propagation end-to-end + forward smoke + end-to-end gradcheck.

Run:
    pixi run mojo run -I . tests/nn/test_vit.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import Transpose2DOp, TokenMean as _TokenMeanOp
from mojo_rl.nn.composites import PatchEmbed, ViT
from layout import Layout, LayoutTensor


def make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


def make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-0.5, 0.5)))
    return lst^


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


# =============================================================================
# Test 1: Transpose2DOp shape + round-trip + gradcheck
# =============================================================================
def test_transpose_2d() -> Int:
    print_header("Transpose2DOp: shape + round-trip + gradcheck")
    var fails = 0
    seed(11)

    comptime A = 3
    comptime B = 5
    comptime BATCH = 2
    comptime Op = Transpose2DOp[A, B]
    comptime InvOp = Transpose2DOp[B, A]

    check(Op.IN_DIM == A * B, "IN_DIM = A*B = " + String(Op.IN_DIM), fails)
    check(Op.OUT_DIM == A * B, "OUT_DIM = A*B = " + String(Op.OUT_DIM), fails)
    check(Op.PARAM_SIZE == 0, "PARAM_SIZE = 0", fails)
    check(Op.CACHE_SIZE == 0, "CACHE_SIZE = 0", fails)

    # Round-trip: input → Transpose[A,B] → Transpose[B,A] → original
    var inp = make_rand_list(BATCH * Op.IN_DIM)
    var mid = make_list(BATCH * Op.IN_DIM)
    var rt = make_list(BATCH * Op.IN_DIM)
    var dummy_p = make_list(1)
    var dummy_c = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var mid_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
    ](mid.unsafe_ptr())
    var rt_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, InvOp.OUT_DIM), MutAnyOrigin
    ](rt.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](dummy_p.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](dummy_c.unsafe_ptr())

    Op.eval[BATCH, dtype](inp_t, mid_t, p_t, c_t)
    InvOp.eval[BATCH, dtype](mid_t, rt_t, p_t, c_t)

    var max_diff: Float64 = 0
    for i in range(BATCH * Op.IN_DIM):
        var d = math_abs(Float64(inp[i]) - Float64(rt[i]))
        if d > max_diff:
            max_diff = d
    check(
        max_diff < 1e-7,
        "round-trip Transpose[B,A] o Transpose[A,B] == identity (max diff = "
        + String(max_diff) + ")",
        fails,
    )

    # Spot-check: out[b, j*A + i] == in[b, i*B + j].
    var spot_err: Float64 = 0
    for b in range(BATCH):
        for i in range(A):
            for j in range(B):
                var got = Float64(mid[b * Op.IN_DIM + j * A + i])
                var expected = Float64(inp[b * Op.IN_DIM + i * B + j])
                var d = math_abs(got - expected)
                if d > spot_err:
                    spot_err = d
    check(
        spot_err < 1e-7,
        "out[b, j*A+i] == in[b, i*B+j] for all (b,i,j) (max diff = "
        + String(spot_err) + ")",
        fails,
    )

    # Finite-diff gradcheck.
    var go = make_rand_list(BATCH * Op.OUT_DIM)
    var out_data = make_list(BATCH * Op.OUT_DIM)
    var gi_data = make_list(BATCH * Op.IN_DIM)
    var gp = make_list(1)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp.unsafe_ptr())
    Op.eval[BATCH, dtype](inp_t, out_t, p_t, c_t)
    Op.vjp[BATCH, dtype](go_t, gi_t, p_t, c_t, gp_t)

    var eps: Float64 = 1e-3
    var max_err: Float64 = 0
    for idx in range(BATCH * Op.IN_DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var op_data = make_list(BATCH * Op.OUT_DIM)
        var op_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
        ](op_data.unsafe_ptr())
        Op.eval[BATCH, dtype](inp_t, op_t, p_t, c_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var om_data = make_list(BATCH * Op.OUT_DIM)
        var om_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
        ](om_data.unsafe_ptr())
        Op.eval[BATCH, dtype](inp_t, om_t, p_t, c_t)
        inp[idx] = orig

        var fd: Float64 = 0
        for j in range(BATCH * Op.OUT_DIM):
            fd += Float64(go[j]) * (Float64(op_data[j]) - Float64(om_data[j])) / (2.0 * eps)
        var an = Float64(gi_data[idx])
        var err = math_abs(fd - an)
        if math_abs(fd) < 1e-4 and math_abs(an) < 1e-4:
            continue
        var rel = err / (math_abs(fd) + math_abs(an) + 1e-8)
        if rel > max_err:
            max_err = rel
    check(max_err < 5e-2, "Transpose2D gradcheck max rel err = " + String(max_err), fails)

    return fails


# =============================================================================
# Test 2: TokenMean shape + forward + gradcheck
# =============================================================================
def test_token_mean() -> Int:
    print_header("TokenMean: shape + forward + gradcheck")
    var fails = 0
    seed(13)

    comptime SEQ = 4
    comptime DIM = 3
    comptime BATCH = 2
    comptime Op = _TokenMeanOp[SEQ, DIM]

    check(Op.IN_DIM == SEQ * DIM, "IN_DIM = SEQ*DIM = " + String(Op.IN_DIM), fails)
    check(Op.OUT_DIM == DIM, "OUT_DIM = DIM = " + String(Op.OUT_DIM), fails)

    var inp = make_rand_list(BATCH * Op.IN_DIM)
    var go = make_rand_list(BATCH * Op.OUT_DIM)
    var dummy_p = make_list(1)
    var dummy_c = make_list(1)
    var out_data = make_list(BATCH * Op.OUT_DIM)
    var gi_data = make_list(BATCH * Op.IN_DIM)
    var gp = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](dummy_p.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](dummy_c.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp.unsafe_ptr())

    Op.eval[BATCH, dtype](inp_t, out_t, p_t, c_t)

    # Verify: out[b, d] == mean over t of in[b, t * dim + d].
    var max_fwd: Float64 = 0
    for b in range(BATCH):
        for d in range(DIM):
            var s: Float64 = 0
            for t in range(SEQ):
                s += Float64(inp[b * Op.IN_DIM + t * DIM + d])
            var expected = s / Float64(SEQ)
            var got = Float64(out_data[b * DIM + d])
            var diff = math_abs(expected - got)
            if diff > max_fwd:
                max_fwd = diff
    check(max_fwd < 1e-6, "forward matches reference mean (max diff = " + String(max_fwd) + ")", fails)

    # Backward: grad_in[b, t*dim+d] should equal grad_out[b, d] / SEQ.
    Op.vjp[BATCH, dtype](go_t, gi_t, p_t, c_t, gp_t)
    var max_bwd: Float64 = 0
    for b in range(BATCH):
        for t in range(SEQ):
            for d in range(DIM):
                var got = Float64(gi_data[b * Op.IN_DIM + t * DIM + d])
                var expected = Float64(go[b * DIM + d]) / Float64(SEQ)
                var diff = math_abs(got - expected)
                if diff > max_bwd:
                    max_bwd = diff
    check(max_bwd < 1e-7, "backward matches reference grad (max diff = " + String(max_bwd) + ")", fails)

    return fails


# =============================================================================
# Test 3: PatchEmbed shape propagation + forward
# =============================================================================
def test_patch_embed() -> Int:
    print_header("PatchEmbed: shape propagation Conv2D → Transpose2D")
    var fails = 0
    seed(17)

    # Tiny CIFAR-like config: 8x8 image, 4x4 patches → 4 patches.
    comptime IC = 3
    comptime IMG = 8
    comptime PATCH = 4
    comptime D = 16
    comptime NP = (IMG // PATCH) * (IMG // PATCH)  # 4
    comptime PE = PatchEmbed[IC, IMG, IMG, PATCH, D, NP]
    comptime BATCH = 2

    check(PE.IN_DIM == IC * IMG * IMG, "IN_DIM = C*H*W = " + String(PE.IN_DIM), fails)
    check(PE.OUT_DIM == NP * D, "OUT_DIM = n_patches * embed_dim = " + String(PE.OUT_DIM), fails)
    check(PE.PARAM_SIZE > 0, "PARAM_SIZE > 0 (Conv2D weights + bias) = " + String(PE.PARAM_SIZE), fails)

    # Forward smoke: random params + input, confirm output is non-NaN.
    var inp = make_rand_list(BATCH * PE.IN_DIM)
    var params = make_rand_list(PE.PARAM_SIZE)
    for i in range(PE.PARAM_SIZE):
        params[i] = Scalar[dtype](Float64(params[i]) * 0.1)
    var out_data = make_list(BATCH * PE.OUT_DIM)
    var cache_data = make_list(BATCH * PE.CACHE_SIZE)
    var state = make_list(PE.STATE_SIZE if PE.STATE_SIZE > 0 else 1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PE.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PE.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(PE.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(PE.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PE.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    PE.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)

    var has_nan = False
    var any_nonzero = False
    for i in range(BATCH * PE.OUT_DIM):
        var v = Float64(out_data[i])
        if v != v:
            has_nan = True
        if math_abs(v) > 1e-9:
            any_nonzero = True
    check(not has_nan, "PatchEmbed forward produced no NaN", fails)
    check(any_nonzero, "PatchEmbed forward output is non-trivial", fails)

    return fails


# =============================================================================
# Test 4: ViT shape propagation + forward + gradcheck
# =============================================================================
def test_vit() -> Int:
    print_header("ViT: shape propagation + forward + gradcheck")
    var fails = 0
    seed(19)

    # Tiny ViT — keeps gradcheck cost manageable.
    comptime IC = 3
    comptime IMG = 8
    comptime PATCH = 4
    comptime D = 16
    comptime H = 4    # head_dim = 4
    comptime N = 1
    comptime NP = (IMG // PATCH) * (IMG // PATCH)  # 4
    comptime NCLS = 5
    comptime Model = ViT[IC, IMG, IMG, PATCH, D, H, N, NP, NCLS]
    comptime BATCH = 1

    check(Model.IN_DIM == IC * IMG * IMG, "IN_DIM = C*H*W = " + String(Model.IN_DIM), fails)
    check(Model.OUT_DIM == NCLS, "OUT_DIM = n_classes = " + String(Model.OUT_DIM), fails)
    check(Model.PARAM_SIZE > 0, "PARAM_SIZE > 0 = " + String(Model.PARAM_SIZE), fails)

    # Forward smoke.
    var inp = make_rand_list(BATCH * Model.IN_DIM)
    var params = make_rand_list(Model.PARAM_SIZE)
    for i in range(Model.PARAM_SIZE):
        params[i] = Scalar[dtype](Float64(params[i]) * 0.1)
    var go = make_rand_list(BATCH * Model.OUT_DIM)
    var state = make_list(Model.STATE_SIZE if Model.STATE_SIZE > 0 else 1)
    var out_data = make_list(BATCH * Model.OUT_DIM)
    var cache_data = make_list(BATCH * Model.CACHE_SIZE)
    var gi_data = make_list(BATCH * Model.IN_DIM)
    var gp_data = make_list(Model.PARAM_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Model.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Model.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)
    Model.backward[BATCH, dtype](go_t, gi_t, p_t, s_t, c_t, gp_t)

    var has_nan = False
    for i in range(BATCH * Model.OUT_DIM):
        var v = Float64(out_data[i])
        if v != v:
            has_nan = True
    check(not has_nan, "ViT forward produced no NaN", fails)

    # Finite-diff gradcheck on inputs.
    var eps: Float64 = 1e-3
    var max_err: Float64 = 0
    for idx in range(BATCH * Model.IN_DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var op_data = make_list(BATCH * Model.OUT_DIM)
        var ocache = make_list(BATCH * Model.CACHE_SIZE)
        var op_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](op_data.unsafe_ptr())
        var opc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](ocache.unsafe_ptr())
        Model.forward[BATCH, dtype](inp_t, op_t, p_t, s_t, opc_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var om_data = make_list(BATCH * Model.OUT_DIM)
        var omc = make_list(BATCH * Model.CACHE_SIZE)
        var om_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](om_data.unsafe_ptr())
        var omc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](omc.unsafe_ptr())
        Model.forward[BATCH, dtype](inp_t, om_t, p_t, s_t, omc_t)
        inp[idx] = orig

        var fd: Float64 = 0
        for j in range(BATCH * Model.OUT_DIM):
            fd += Float64(go[j]) * (Float64(op_data[j]) - Float64(om_data[j])) / (2.0 * eps)
        var an = Float64(gi_data[idx])
        var err = math_abs(fd - an)
        if math_abs(fd) < 2e-4 and math_abs(an) < 2e-4:
            continue
        var rel = err / (math_abs(fd) + math_abs(an) + 1e-8)
        if rel > max_err:
            max_err = rel
    check(max_err < 5e-2, "ViT grad_input gradcheck max rel err = " + String(max_err), fails)

    return fails


def main() raises:
    var total_fails = 0
    total_fails += test_transpose_2d()
    total_fails += test_token_mean()
    total_fails += test_patch_embed()
    total_fails += test_vit()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL VIT TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)
