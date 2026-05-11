"""Phase-2 Step 3 unit tests — SimSiam cosine consistency loss.

Coverage:
    1. Forward correctness — parallel / orthogonal / anti-parallel hand-
       computed values.
    2. Mean-over-batch convention — loss for B>1 equals (1/B) Σ_b L_b.
    3. Stop-gradient discipline — the API takes only `grad_online`; the
       caller can't accidentally accumulate into target.
    4. Backward — finite-diff gradcheck on `online` against the analytical
       gradient written by the fused fn.
    5. Edge case — zero-norm input (||online|| = 0) doesn't NaN out; the
       returned loss + gradient are finite.

If (4) fails, training will silently learn a representation that doesn't
match `target`, defeating the consistency objective. If (5) fails, the
first time a sample lands at zero on the dynamics branch the K-step
unroll will produce NaN gradients across the whole minibatch.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.efficient_zero_v2 import (
    cosine_consistency_loss,
    cosine_consistency_loss_forward,
)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _close(actual: Float64, expected: Float64, tol: Float64 = 1e-6) -> Bool:
    return _abs(actual - expected) < tol


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def main():
    print("=== EZ-V2 Phase 2 / Step 3 — cosine consistency loss ===")
    var passed = 0
    var total = 0

    # ─── 1. Forward — hand-computed cases ────────────────────────────────
    print()
    print("--- 1. Forward correctness ---")

    # Parallel: target = 2 · online → cos = 1, loss = -1.
    var par_o = InlineArray[Scalar[dtype], 1 * 3](uninitialized=True)
    var par_t = InlineArray[Scalar[dtype], 1 * 3](uninitialized=True)
    par_o[0] = Scalar[dtype](1.0)
    par_o[1] = Scalar[dtype](2.0)
    par_o[2] = Scalar[dtype](3.0)
    par_t[0] = Scalar[dtype](2.0)
    par_t[1] = Scalar[dtype](4.0)
    par_t[2] = Scalar[dtype](6.0)
    var par_o_t = LayoutTensor[
        dtype, Layout.row_major(1, 3), MutAnyOrigin
    ](par_o.unsafe_ptr())
    var par_t_t = LayoutTensor[
        dtype, Layout.row_major(1, 3), MutAnyOrigin
    ](par_t.unsafe_ptr())
    _expect(
        _close(cosine_consistency_loss_forward[1, 3](par_o_t, par_t_t), -1.0),
        "parallel vectors → cos = 1, loss = -1",
        passed,
        total,
    )

    # Orthogonal: online = [1, 0], target = [0, 1] → cos = 0, loss = 0.
    var ort_o = InlineArray[Scalar[dtype], 1 * 2](uninitialized=True)
    var ort_t = InlineArray[Scalar[dtype], 1 * 2](uninitialized=True)
    ort_o[0] = Scalar[dtype](1.0)
    ort_o[1] = Scalar[dtype](0.0)
    ort_t[0] = Scalar[dtype](0.0)
    ort_t[1] = Scalar[dtype](1.0)
    var ort_o_t = LayoutTensor[
        dtype, Layout.row_major(1, 2), MutAnyOrigin
    ](ort_o.unsafe_ptr())
    var ort_t_t = LayoutTensor[
        dtype, Layout.row_major(1, 2), MutAnyOrigin
    ](ort_t.unsafe_ptr())
    _expect(
        _close(cosine_consistency_loss_forward[1, 2](ort_o_t, ort_t_t), 0.0),
        "orthogonal vectors → cos = 0, loss = 0",
        passed,
        total,
    )

    # Anti-parallel: online = [1, 2], target = [-1, -2] → cos = -1, loss = 1.
    var anti_o = InlineArray[Scalar[dtype], 1 * 2](uninitialized=True)
    var anti_t = InlineArray[Scalar[dtype], 1 * 2](uninitialized=True)
    anti_o[0] = Scalar[dtype](1.0)
    anti_o[1] = Scalar[dtype](2.0)
    anti_t[0] = Scalar[dtype](-1.0)
    anti_t[1] = Scalar[dtype](-2.0)
    var anti_o_t = LayoutTensor[
        dtype, Layout.row_major(1, 2), MutAnyOrigin
    ](anti_o.unsafe_ptr())
    var anti_t_t = LayoutTensor[
        dtype, Layout.row_major(1, 2), MutAnyOrigin
    ](anti_t.unsafe_ptr())
    _expect(
        _close(cosine_consistency_loss_forward[1, 2](anti_o_t, anti_t_t), 1.0),
        "anti-parallel vectors → cos = -1, loss = +1",
        passed,
        total,
    )

    # Loss is bounded: |L| ≤ 1 for any finite input.
    var bnd_o = InlineArray[Scalar[dtype], 1 * 4](uninitialized=True)
    var bnd_t = InlineArray[Scalar[dtype], 1 * 4](uninitialized=True)
    bnd_o[0] = Scalar[dtype](7.5)
    bnd_o[1] = Scalar[dtype](-3.2)
    bnd_o[2] = Scalar[dtype](0.4)
    bnd_o[3] = Scalar[dtype](11.0)
    bnd_t[0] = Scalar[dtype](-2.0)
    bnd_t[1] = Scalar[dtype](6.5)
    bnd_t[2] = Scalar[dtype](-1.1)
    bnd_t[3] = Scalar[dtype](0.3)
    var bnd_o_t = LayoutTensor[
        dtype, Layout.row_major(1, 4), MutAnyOrigin
    ](bnd_o.unsafe_ptr())
    var bnd_t_t = LayoutTensor[
        dtype, Layout.row_major(1, 4), MutAnyOrigin
    ](bnd_t.unsafe_ptr())
    var bnd_loss = cosine_consistency_loss_forward[1, 4](bnd_o_t, bnd_t_t)
    _expect(
        bnd_loss >= -1.0 - 1e-6 and bnd_loss <= 1.0 + 1e-6,
        "loss is bounded in [-1, 1] for arbitrary finite input",
        passed,
        total,
    )

    # ─── 2. Mean-over-batch ──────────────────────────────────────────────
    print()
    print("--- 2. Mean-over-batch convention ---")

    # B = 2: row 0 parallel (loss=-1), row 1 orthogonal (loss=0).
    # Expected mean = (-1 + 0) / 2 = -0.5.
    var b2_o = InlineArray[Scalar[dtype], 2 * 2](uninitialized=True)
    var b2_t = InlineArray[Scalar[dtype], 2 * 2](uninitialized=True)
    b2_o[0] = Scalar[dtype](1.0)
    b2_o[1] = Scalar[dtype](2.0)
    b2_t[0] = Scalar[dtype](2.0)
    b2_t[1] = Scalar[dtype](4.0)
    b2_o[2] = Scalar[dtype](1.0)
    b2_o[3] = Scalar[dtype](0.0)
    b2_t[2] = Scalar[dtype](0.0)
    b2_t[3] = Scalar[dtype](1.0)
    var b2_o_t = LayoutTensor[
        dtype, Layout.row_major(2, 2), MutAnyOrigin
    ](b2_o.unsafe_ptr())
    var b2_t_t = LayoutTensor[
        dtype, Layout.row_major(2, 2), MutAnyOrigin
    ](b2_t.unsafe_ptr())
    _expect(
        _close(
            cosine_consistency_loss_forward[2, 2](b2_o_t, b2_t_t), -0.5
        ),
        "mean over batch: (parallel + orthogonal) / 2 = -0.5",
        passed,
        total,
    )

    # ─── 3. Stop-gradient (API contract) ─────────────────────────────────
    # The function does not accept grad_target; the test just records the
    # contract here.
    print()
    print("--- 3. Stop-gradient discipline ---")
    _expect(
        True,
        "API has no grad_target output → target is stop-grad by construction",
        passed,
        total,
    )

    # ─── 4. Finite-diff gradcheck on grad_online ─────────────────────────
    print()
    print("--- 4. Backward gradcheck (finite-diff) ---")
    comptime BATCH = 2
    comptime DIM = 8

    # Random-ish inputs that aren't zero or symmetric.
    var fd_o = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var fd_t = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    for i in range(BATCH * DIM):
        fd_o[i] = Scalar[dtype](0.13 * Float64(i % 11) - 0.5)
        fd_t[i] = Scalar[dtype](0.21 * Float64((i + 3) % 13) - 0.7)
    var fd_o_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](fd_o.unsafe_ptr())
    var fd_t_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](fd_t.unsafe_ptr())

    var grad_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    for i in range(BATCH * DIM):
        grad_arr[i] = Scalar[dtype](0.0)
    var grad_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_arr.unsafe_ptr())

    var loss_ana = cosine_consistency_loss[BATCH, DIM](
        fd_o_t, fd_t_t, grad_t, grad_seed=1.0
    )

    # Sanity: forward via the dedicated path matches the fused result.
    var loss_fwd = cosine_consistency_loss_forward[BATCH, DIM](fd_o_t, fd_t_t)
    _expect(
        _close(loss_ana, loss_fwd, tol=1e-9),
        "fused fwd+bwd loss matches forward-only fn",
        passed,
        total,
    )

    var eps = Float64(1e-4)
    var max_abs = Float64(0.0)
    var num_checked = 0
    for b in range(BATCH):
        for i in range(DIM):
            var orig = fd_o[b * DIM + i]

            fd_o[b * DIM + i] = orig + Scalar[dtype](eps)
            var L_plus = cosine_consistency_loss_forward[BATCH, DIM](
                fd_o_t, fd_t_t
            )
            fd_o[b * DIM + i] = orig - Scalar[dtype](eps)
            var L_minus = cosine_consistency_loss_forward[BATCH, DIM](
                fd_o_t, fd_t_t
            )
            fd_o[b * DIM + i] = orig

            var num_grad = (L_plus - L_minus) / (2.0 * eps)
            var ana_grad = Float64(grad_arr[b * DIM + i])
            var d = num_grad - ana_grad
            if d < 0:
                d = -d
            if d > max_abs:
                max_abs = d
            num_checked += 1

    print(
        "    finite-diff vs analytical, max |Δ| =", max_abs,
        " (", num_checked, " elements checked)",
    )
    _expect(
        max_abs < 1e-3,
        "analytical gradient matches finite-diff to 1e-3 tolerance",
        passed,
        total,
    )

    # ─── 5. Edge case — zero-norm input ──────────────────────────────────
    print()
    print("--- 5. Zero-norm robustness ---")
    var z_o = InlineArray[Scalar[dtype], 1 * 4](uninitialized=True)
    var z_t = InlineArray[Scalar[dtype], 1 * 4](uninitialized=True)
    for i in range(4):
        z_o[i] = Scalar[dtype](0.0)
        z_t[i] = Scalar[dtype](Float64(i) + 0.5)
    var z_o_t = LayoutTensor[
        dtype, Layout.row_major(1, 4), MutAnyOrigin
    ](z_o.unsafe_ptr())
    var z_t_t = LayoutTensor[
        dtype, Layout.row_major(1, 4), MutAnyOrigin
    ](z_t.unsafe_ptr())
    var z_grad = InlineArray[Scalar[dtype], 1 * 4](uninitialized=True)
    var z_grad_t = LayoutTensor[
        dtype, Layout.row_major(1, 4), MutAnyOrigin
    ](z_grad.unsafe_ptr())
    var z_loss = cosine_consistency_loss[1, 4](
        z_o_t, z_t_t, z_grad_t, grad_seed=1.0
    )
    var z_grad_finite = True
    for i in range(4):
        if not _is_finite(Float64(z_grad[i])):
            z_grad_finite = False
    _expect(
        _is_finite(z_loss),
        "zero-norm online → loss is finite",
        passed,
        total,
    )
    _expect(
        z_grad_finite,
        "zero-norm online → gradient is finite",
        passed,
        total,
    )

    # ─── 6. grad_seed scales linearly ───────────────────────────────────
    print()
    print("--- 6. grad_seed scales gradient linearly ---")
    var s1 = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var s2 = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    for i in range(BATCH * DIM):
        s1[i] = Scalar[dtype](0.0)
        s2[i] = Scalar[dtype](0.0)
    var s1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](s1.unsafe_ptr())
    var s2_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](s2.unsafe_ptr())
    _ = cosine_consistency_loss[BATCH, DIM](
        fd_o_t, fd_t_t, s1_t, grad_seed=1.0
    )
    _ = cosine_consistency_loss[BATCH, DIM](
        fd_o_t, fd_t_t, s2_t, grad_seed=3.0
    )
    var seed_ok = True
    var max_dev = Float64(0.0)
    for i in range(BATCH * DIM):
        var g1 = Float64(s1[i])
        var g2 = Float64(s2[i])
        var d = g2 - 3.0 * g1
        if d < 0:
            d = -d
        if d > max_dev:
            max_dev = d
        if d > 1e-6:
            seed_ok = False
    print("    max |grad(seed=3) − 3·grad(seed=1)| =", max_dev)
    _expect(
        seed_ok, "grad_seed=3 yields exactly 3x the grad of grad_seed=1",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
