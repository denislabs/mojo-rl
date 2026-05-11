"""TD-MPC2 — extract_z_from_za_grad kernel overwrite contract (Test 6).

Regression test for Bug 1 (project_tdmpc2_extract_z_overwrite_bug.md):
`tdmpc2_extract_z_from_za_grad_kernel` was using `grad_z[i,k] += grad_za[i,k]`
while production `_wm_bptt_gpu` calls it 6+ times per BPTT step into the
same persistent grad_z_dyn_buf without zeroing between calls. The bug
caused the encoder gradient to weigh **7×dyn + 6×rew + 5×Q1 + 4×Q2 +
3×Q3 + 2×Q4 + 1×Q5** per iter — over-weighting consistency-via-dynamics
by 7× and pulling the encoder into trivial collapse.

The fix changed the kernel to overwrite (`grad_z[i,k] = grad_za[i,k]`).
This test pins that contract.

Sub-tests:
  6a — single call: pre-fill grad_z with nonzero junk; after extract,
       grad_z must equal grad_za[:, :LATENT] (no remnant of junk).
  6b — three sequential calls into the SAME buffer with three different
       grad_za inputs: each result must equal its own grad_za[:, :LATENT]
       (no accumulation across calls).
  6c — production-style multi-loss flow: extract → copy(carry=dyn) →
       extract(rew) → add_into(carry, dyn) → extract(q) → add_into(carry, dyn).
       Carry must equal dyn + rew + q (each contributing exactly once),
       NOT 3×dyn + 2×rew + q (the bug pattern).
"""

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.deep_agents.tdmpc2.kernels import (
    tdmpc2_extract_z_from_za_grad_kernel,
    tdmpc2_add_into_kernel,
)
from mojo_rl.deep_agents.core.kernels import copy_buffer_kernel


comptime BATCH = 4
comptime LATENT = 16
comptime ACT = 2
comptime ZA = LATENT + ACT
comptime BLOCKS = (BATCH + TPB - 1) // TPB


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def main() raises:
    print("=" * 70)
    print("TD-MPC2 Test 6 — extract_z_from_za_grad overwrite contract")
    print("=" * 70)

    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        # Reusable launch alias.
        comptime extract_z = tdmpc2_extract_z_from_za_grad_kernel[
            dtype, BATCH, LATENT, ACT
        ]
        comptime add_into = tdmpc2_add_into_kernel[dtype, BATCH * LATENT]
        comptime copy_kernel = copy_buffer_kernel[dtype, BATCH * LATENT]

        # ─── 6a — single call overwrites pre-existing junk ──────────────
        print()
        print("--- 6a. Single extract overwrites pre-existing values ---")

        var grad_za_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var grad_z_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        for i in range(BATCH * ZA):
            grad_za_host[i] = Scalar[dtype](Float64(i) * 0.1 - 1.0)
        # Pre-fill grad_z host buffer with sentinel "junk" we want
        # overwritten.
        for i in range(BATCH * LATENT):
            grad_z_host[i] = Scalar[dtype](99.0)
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_z_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        ctx.enqueue_copy(grad_za_dev, grad_za_host)
        ctx.enqueue_copy(grad_z_dev, grad_z_host)

        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_dev.unsafe_ptr())
        var grad_z_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_dev.unsafe_ptr())

        ctx.enqueue_function[extract_z, extract_z](
            grad_za_t, grad_z_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

        ctx.enqueue_copy(grad_z_host, grad_z_dev)
        ctx.synchronize()

        var max_err: Float64 = 0.0
        for b in range(BATCH):
            for k in range(LATENT):
                var got = Float64(grad_z_host[b * LATENT + k])
                var expected = Float64(grad_za_host[b * ZA + k])
                var d = _abs(got - expected)
                if d > max_err:
                    max_err = d
        print("    max |Δ| (got vs grad_za[:LATENT]) =", max_err)
        _expect(
            max_err < 1e-6,
            "6a — extract overwrites: result == grad_za[:, :LATENT]",
            passed, total,
        )

        # ─── 6b — three sequential calls don't accumulate ───────────────
        print()
        print("--- 6b. Sequential extracts on shared buffer don't accumulate ---")

        # Three different grad_za inputs.
        var grad_za_a_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var grad_za_b_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var grad_za_c_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        for i in range(BATCH * ZA):
            grad_za_a_host[i] = Scalar[dtype](Float64(i) * 0.13 + 0.5)
            grad_za_b_host[i] = Scalar[dtype](Float64(i) * 0.07 - 0.2)
            grad_za_c_host[i] = Scalar[dtype](Float64(i) * 0.21 + 1.1)
        var grad_za_a_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_b_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_c_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        ctx.enqueue_copy(grad_za_a_dev, grad_za_a_host)
        ctx.enqueue_copy(grad_za_b_dev, grad_za_b_host)
        ctx.enqueue_copy(grad_za_c_dev, grad_za_c_host)

        var grad_za_a_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_a_dev.unsafe_ptr())
        var grad_za_b_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_b_dev.unsafe_ptr())
        var grad_za_c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_c_dev.unsafe_ptr())

        # Pre-fill grad_z with junk one more time.
        for i in range(BATCH * LATENT):
            grad_z_host[i] = Scalar[dtype](42.0)
        ctx.enqueue_copy(grad_z_dev, grad_z_host)

        # Three sequential calls — final result must match C only,
        # NOT (junk + A + B + C) under buggy += semantics.
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_a_t, grad_z_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_b_t, grad_z_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_c_t, grad_z_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        ctx.enqueue_copy(grad_z_host, grad_z_dev)
        ctx.synchronize()

        max_err = 0.0
        var max_err_buggy: Float64 = 0.0
        for b in range(BATCH):
            for k in range(LATENT):
                var got = Float64(grad_z_host[b * LATENT + k])
                # Correct: result == C[:LATENT].
                var expected_correct = Float64(grad_za_c_host[b * ZA + k])
                var d = _abs(got - expected_correct)
                if d > max_err:
                    max_err = d
                # Buggy: result == 42 + A[:LATENT] + B[:LATENT] + C[:LATENT].
                var expected_buggy = (
                    42.0
                    + Float64(grad_za_a_host[b * ZA + k])
                    + Float64(grad_za_b_host[b * ZA + k])
                    + Float64(grad_za_c_host[b * ZA + k])
                )
                var d_buggy = _abs(got - expected_buggy)
                if d_buggy > max_err_buggy:
                    max_err_buggy = d_buggy

        print("    max |Δ| vs correct (C[:LATENT]) =", max_err)
        print(
            "    max |Δ| vs buggy (junk+A+B+C) =", max_err_buggy,
            " — should be LARGE if not buggy",
        )
        _expect(
            max_err < 1e-6,
            "6b — final result matches last call's grad_za only",
            passed, total,
        )
        _expect(
            max_err_buggy > 1.0,
            "6b — result does NOT match buggy accumulation pattern",
            passed, total,
        )

        # ─── 6c — production multi-loss flow with shared grad_z_dyn ─────
        # Mirrors _wm_bptt_gpu pattern at one BPTT iter:
        #   extract_z(grad_za_dyn, grad_z_dyn)         [grad_z_dyn = dyn]
        #   copy(grad_z_dyn → grad_z_carry)            [carry = dyn]
        #   extract_z(grad_za_rew, grad_z_dyn)         [grad_z_dyn = rew (overwrite)]
        #   add_into(grad_z_carry, grad_z_dyn)         [carry = dyn + rew]
        #   extract_z(grad_za_q,   grad_z_dyn)         [grad_z_dyn = q]
        #   add_into(grad_z_carry, grad_z_dyn)         [carry = dyn + rew + q]
        # ────────────────────────────────────────────────────────────────
        print()
        print("--- 6c. Production multi-loss flow: carry = dyn + rew + q (no doubling) ---")

        var grad_z_dyn_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var grad_z_carry_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )

        # Init both with junk; the kernels should overwrite/zero them out.
        for i in range(BATCH * LATENT):
            grad_z_host[i] = Scalar[dtype](7.0)
        ctx.enqueue_copy(grad_z_dyn_dev, grad_z_host)
        for i in range(BATCH * LATENT):
            grad_z_host[i] = Scalar[dtype](0.0)  # carry must start at 0
        ctx.enqueue_copy(grad_z_carry_dev, grad_z_host)

        var grad_z_dyn_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_dyn_dev.unsafe_ptr())
        var grad_z_dyn_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](grad_z_dyn_dev.unsafe_ptr())
        var grad_z_carry_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](grad_z_carry_dev.unsafe_ptr())

        # 1. dyn extract → grad_z_dyn = grad_za_a[:LATENT]
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_a_t, grad_z_dyn_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        # 2. copy → carry = grad_z_dyn (= dyn part)
        ctx.enqueue_function[copy_kernel, copy_kernel](
            grad_z_carry_flat,
            grad_z_dyn_flat,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
        # 3. rew extract → grad_z_dyn = grad_za_b[:LATENT] (overwrite)
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_b_t, grad_z_dyn_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        # 4. add → carry += grad_z_dyn (now = dyn + rew)
        ctx.enqueue_function[add_into, add_into](
            grad_z_carry_flat,
            grad_z_dyn_flat,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
        # 5. q extract → grad_z_dyn = grad_za_c[:LATENT] (overwrite)
        ctx.enqueue_function[extract_z, extract_z](
            grad_za_c_t, grad_z_dyn_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        # 6. add → carry += grad_z_dyn (now = dyn + rew + q)
        ctx.enqueue_function[add_into, add_into](
            grad_z_carry_flat,
            grad_z_dyn_flat,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        var carry_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        ctx.enqueue_copy(carry_host, grad_z_carry_dev)
        ctx.synchronize()

        # Reference: carry = grad_za_a[:LATENT] + grad_za_b[:LATENT]
        #                  + grad_za_c[:LATENT]  (each ×1).
        # Buggy alternative under += extract_z:
        #   carry = 3*dyn + 2*rew + q
        # (where dyn=A[:L], rew=B[:L], q=C[:L]; see audit Bug 1).
        max_err = 0.0
        max_err_buggy = 0.0
        for b in range(BATCH):
            for k in range(LATENT):
                var got = Float64(carry_host[b * LATENT + k])
                var dyn_v = Float64(grad_za_a_host[b * ZA + k])
                var rew_v = Float64(grad_za_b_host[b * ZA + k])
                var q_v = Float64(grad_za_c_host[b * ZA + k])
                var expected_correct = dyn_v + rew_v + q_v
                var d = _abs(got - expected_correct)
                if d > max_err:
                    max_err = d
                # Buggy pattern: 3×dyn + 2×rew + q
                var expected_buggy = 3.0 * dyn_v + 2.0 * rew_v + q_v
                var d_buggy = _abs(got - expected_buggy)
                if d_buggy > max_err_buggy:
                    max_err_buggy = d_buggy
        print("    max |Δ| vs correct (dyn + rew + q) =", max_err)
        print(
            "    max |Δ| vs buggy (3·dyn + 2·rew + q) =",
            max_err_buggy,
            " — should be LARGE if not buggy",
        )
        _expect(
            max_err < 1e-5,
            "6c — production flow yields carry = dyn + rew + q (no doubling)",
            passed, total,
        )
        _expect(
            max_err_buggy > 0.5,
            "6c — production flow does NOT match buggy 3×dyn pattern",
            passed, total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
