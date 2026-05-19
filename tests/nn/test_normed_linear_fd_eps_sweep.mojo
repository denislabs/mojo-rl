"""Diagnostic: is `Seq[NL, NL, Linear]` gradcheck failure a real bug or FD noise?

Runs the failing chain (matches `test_normed_linear_parity.mojo` Test 8) at a
range of FD step sizes. The two possibilities:

- **FD truncation / FP roundoff (not a bug)**: At eps too large, central-diff
  truncation error `~ eps² · ∂³L/∂p³` dominates. At eps too small, roundoff
  in `(out_plus - out_minus)` dominates. The sweet spot is around
  `sqrt(machine_eps_fp32) ≈ 3e-4`. If max_rel improves toward the sweet spot
  then degrades both ways, the analytical gradient is correct.
- **Real gradient bug**: max_rel stays roughly constant across eps, and the
  worst-case fd vs ana values disagree by a consistent margin even as the
  numerator gets smaller.

Run:
    pixi run mojo run -I . tests/nn/test_normed_linear_fd_eps_sweep.mojo
"""

from std.math import abs as math_abs
from std.memory import alloc
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import NormedLinear, Linear, Sequential
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming


def fill_pseudo[BS: Int, D: Int](
    t: LayoutTensor[dtype, Layout.row_major(BS, D), MutAnyOrigin], salt: Int
):
    """Deterministic pseudo-random fill matching the parity test's helper."""
    for b in range(BS):
        for d in range(D):
            var v = Scalar[dtype](
                Float64((b * D + d + salt) * 17 % 137) / 137.0 - 0.5
            )
            t[b, d] = v


def sweep_eps(eps_values: List[Float64]) raises:
    comptime D = 8
    comptime BS = 2
    comptime M = Sequential[
        NormedLinear[D, D], NormedLinear[D, D], Linear[D, D]
    ]
    comptime PS = M.PARAM_SIZE

    var state = NetworkState[M, Adam[]]()
    state.initialize[Kaiming[]]()
    var p = state.params_view()
    var s = state.model_state_view()

    var input_ptr = alloc[Scalar[dtype]](BS * D)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, D), MutAnyOrigin](
        input_ptr
    )
    fill_pseudo[BS, D](input_t, salt=10)

    var grad_out_ptr = alloc[Scalar[dtype]](BS * D)
    var grad_out = LayoutTensor[dtype, Layout.row_major(BS, D), MutAnyOrigin](
        grad_out_ptr
    )
    fill_pseudo[BS, D](grad_out, salt=11)

    var out_ptr = alloc[Scalar[dtype]](BS * D)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, D), MutAnyOrigin](
        out_ptr
    )
    var cache_ptr = alloc[Scalar[dtype]](BS * M.CACHE_SIZE)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, M.CACHE_SIZE), MutAnyOrigin
    ](cache_ptr)
    M.forward[BS](input_t, out_t, p, s, cache_t)

    # Analytical gradient via backward
    var grad_in_ptr = alloc[Scalar[dtype]](BS * D)
    var grad_in = LayoutTensor[dtype, Layout.row_major(BS, D), MutAnyOrigin](
        grad_in_ptr
    )
    var g_view = state.grads_view()
    for i in range(PS):
        g_view[i] = 0
    var grad_out_copy_ptr = alloc[Scalar[dtype]](BS * D)
    var grad_out_copy = LayoutTensor[
        dtype, Layout.row_major(BS, D), MutAnyOrigin
    ](grad_out_copy_ptr)
    for b in range(BS):
        for j in range(D):
            grad_out_copy[b, j] = grad_out[b, j]
    M.backward[BS](grad_out_copy, grad_in, p, s, cache_t, g_view)

    var analytical = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        analytical[i] = g_view[i][0]

    # Sweep FD eps values; track (max_rel, max_abs, worst_idx, fd, ana) per eps.
    print(
        "  eps       | max_abs    | max_rel   | worst_idx | fd_at_worst |"
        " ana_at_worst"
    )
    print(
        "  ----------+------------+-----------+-----------+-------------+"
        "-------------"
    )
    for k in range(len(eps_values)):
        var eps = eps_values[k]
        var max_rel: Float64 = 0.0
        var max_abs: Float64 = 0.0
        var worst_idx = 0
        var fd_at_worst: Float64 = 0.0
        var ana_at_worst: Float64 = 0.0
        for i in range(PS):
            var orig = Float64(p[i][0])
            p[i] = Scalar[dtype](orig + eps)
            M.forward[BS](input_t, out_t, p, s, cache_t)
            var L_plus: Float64 = 0.0
            for b in range(BS):
                for j in range(D):
                    L_plus += Float64(out_t[b, j][0]) * Float64(
                        grad_out[b, j][0]
                    )
            p[i] = Scalar[dtype](orig - eps)
            M.forward[BS](input_t, out_t, p, s, cache_t)
            var L_minus: Float64 = 0.0
            for b in range(BS):
                for j in range(D):
                    L_minus += Float64(out_t[b, j][0]) * Float64(
                        grad_out[b, j][0]
                    )
            p[i] = Scalar[dtype](orig)
            var fd_grad = (L_plus - L_minus) / (2.0 * eps)
            var ana_grad = Float64(analytical[i][0])
            var abs_err = math_abs(fd_grad - ana_grad)
            var denom = math_abs(fd_grad) + math_abs(ana_grad) + 1e-8
            var rel_err = abs_err / denom
            if abs_err > max_abs:
                max_abs = abs_err
            if rel_err > max_rel:
                max_rel = rel_err
                worst_idx = i
                fd_at_worst = fd_grad
                ana_at_worst = ana_grad
        print(
            "  ",
            eps,
            "|",
            max_abs,
            "|",
            max_rel,
            "|",
            worst_idx,
            "|",
            fd_at_worst,
            "|",
            ana_at_worst,
        )


def main() raises:
    print("=" * 78)
    print("FD eps sweep on Seq[NormedLinear, NormedLinear, Linear]")
    print("=" * 78)
    print(
        "If max_rel improves toward eps ≈ 3e-4 (≈ sqrt(fp32 machine eps)) and"
        " degrades on both sides, the analytical gradient is correct and the"
        " test simply hits FD precision limits.\nIf max_rel stays flat or"
        " degrades monotonically, suspect a real gradient bug.\n"
    )

    var eps_values = List[Float64]()
    eps_values.append(3e-2)
    eps_values.append(1e-2)
    eps_values.append(3e-3)
    eps_values.append(1e-3)  # default in parity test
    eps_values.append(3e-4)  # theoretical sweet spot for FP32
    eps_values.append(1e-4)
    eps_values.append(3e-5)
    eps_values.append(1e-5)

    sweep_eps(eps_values)

    print()
    print("=" * 78)
    print("Done.")
    print("=" * 78)
