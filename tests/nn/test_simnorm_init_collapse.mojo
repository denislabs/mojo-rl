"""SimNorm init-time output diversity probe.

Reproduces the 'trivial-collapse attractor' documented in
`mojo_rl/deep_agents/tdmpc2/world_model.mojo:223-232`, which is why the
encoder uses Normal(0, 0.05) where reference TD-MPC2 uses
trunc_normal_(std=0.02). The doc claims std(z) across the batch starts
at ~0.022 with σ=0.02 and decays to ~0.0005 during training.

Two probes:
  1. SimNorm forward correctness on a small known input (one-hot,
     uniform, saturating, linear ramp groups).
  2. Sweep Normal(0, σ) for σ ∈ {0.01, 0.02, 0.03, 0.05, 0.10} on the
     PRODUCTION encoder shape (OBS=17, ENC=256, LATENT=512, SIMPLEX=8).
     Report per-dim std(z) across a 256-sample standard-Gaussian batch
     and per-group softmax peakiness.

Run:
    pixi run mojo run -I . tests/nn/test_simnorm_init_collapse.mojo
"""

from std.math import sqrt, exp, log, cos
from std.random import seed, random_float64
from std.memory import alloc

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Sequential,
    NormedLinear,
    Linear,
    LayerNorm,
    SimNorm,
)
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Initializer, Normal


comptime PI: Float64 = 3.141592653589793


# Optimizer is irrelevant — we never step. NetworkState requires one.
comptime DummyOpt = Adam[LR=1.0]


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("  PASS:", label)
        passed += 1
    else:
        print("  FAIL:", label)


# =============================================================================
# Generic forward helper — must be parametric on M so the compiler can
# resolve M.STATE_SIZE / M.CACHE_SIZE at call site (avoids the recursive-
# reference issue you hit when invoking `SimNorm[...].forward` directly).
# =============================================================================


def _forward_once[
    M: Model, BS: Int, INIT: Initializer
](
    input_t: LayoutTensor[
        dtype, Layout.row_major(BS, M.IN_DIM), MutAnyOrigin
    ],
    mut output_t: LayoutTensor[
        dtype, Layout.row_major(BS, M.OUT_DIM), MutAnyOrigin
    ],
) raises:
    """Build NetworkState[M], init with INIT, run M.forward, free cache."""
    var net = NetworkState[M, DummyOpt]()
    net.initialize[INIT]()

    var cs = M.CACHE_SIZE if M.CACHE_SIZE > 0 else 1
    var cache_ptr = alloc[Scalar[dtype]](BS * cs)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, M.CACHE_SIZE), MutAnyOrigin
    ](cache_ptr)

    M.forward[BS](
        input_t,
        output_t,
        net.params_view(),
        net.model_state_view(),
        cache_t,
    )

    cache_ptr.free()


# =============================================================================
# Test 1: SimNorm forward correctness on a known input
# =============================================================================


def test_simnorm_forward_correctness(mut passed: Int, mut total: Int) raises:
    print("\n=== Test 1: SimNorm forward on known input ===")

    comptime DIM = 16
    comptime SD = 8
    comptime B = 2
    comptime SN = SimNorm[DIM, SD]

    var input_ptr = alloc[Scalar[dtype]](B * DIM)
    var output_ptr = alloc[Scalar[dtype]](B * DIM)

    # Layout per sample (DIM=16 = 2 groups of 8):
    #   S0G0 [0..7]:    one-hot at position 0 — softmax peaked
    #   S0G1 [8..15]:   all zeros — uniform softmax (0.125 each)
    #   S1G0 [16..23]:  saturating peak at position 3 (value 10.0)
    #   S1G1 [24..31]:  linear ramp 0..7 — graded softmax
    for i in range(B * DIM):
        (input_ptr + i)[] = Scalar[dtype](0.0)
    (input_ptr + 0)[] = Scalar[dtype](1.0)
    (input_ptr + 16 + 3)[] = Scalar[dtype](10.0)
    for k in range(SD):
        (input_ptr + 24 + k)[] = Scalar[dtype](Float64(k))

    var input_t = LayoutTensor[
        dtype, Layout.row_major(B, SN.IN_DIM), MutAnyOrigin
    ](input_ptr)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(B, SN.OUT_DIM), MutAnyOrigin
    ](output_ptr)

    _forward_once[SN, B, Normal[0.0, 1.0]](input_t, output_t)

    # Sample 0 group 0: softmax([1, 0, ..., 0]) → e / (e + 7) ≈ 0.3796
    var exp1 = exp(1.0)
    var expected_s0g0_p0 = exp1 / (exp1 + 7.0)
    var expected_s0g0_p1 = 1.0 / (exp1 + 7.0)
    _expect(
        abs(Float64((output_ptr + 0)[]) - expected_s0g0_p0) < 1e-5,
        "S0G0[0] = e/(e+7) → "
        + String(Float64((output_ptr + 0)[]))[byte=:8]
        + " (expected "
        + String(expected_s0g0_p0)[byte=:8]
        + ")",
        passed,
        total,
    )
    _expect(
        abs(Float64((output_ptr + 1)[]) - expected_s0g0_p1) < 1e-5,
        "S0G0[1] = 1/(e+7) → "
        + String(Float64((output_ptr + 1)[]))[byte=:8]
        + " (expected "
        + String(expected_s0g0_p1)[byte=:8]
        + ")",
        passed,
        total,
    )

    # Sample 0 group 1: uniform softmax of zeros → 0.125 everywhere
    var s0g1_max_dev: Float64 = 0.0
    for k in range(SD):
        var dev = abs(Float64((output_ptr + 8 + k)[]) - 0.125)
        if dev > s0g1_max_dev:
            s0g1_max_dev = dev
    _expect(
        s0g1_max_dev < 1e-5,
        "S0G1 uniform softmax (max |dev - 0.125| = "
        + String(s0g1_max_dev)[byte=:9]
        + ")",
        passed,
        total,
    )

    # Sample 1 group 0: saturating at pos 3 → ~1.0 there
    _expect(
        Float64((output_ptr + 16 + 3)[]) > 0.999,
        "S1G0[3] saturated to "
        + String(Float64((output_ptr + 16 + 3)[]))[byte=:8],
        passed,
        total,
    )

    # Each group sums to 1.0 (probability simplex invariant)
    for s in range(B):
        for g in range(2):
            var grp_sum: Float64 = 0.0
            for k in range(SD):
                grp_sum += Float64((output_ptr + s * DIM + g * SD + k)[])
            _expect(
                abs(grp_sum - 1.0) < 1e-5,
                "S"
                + String(s)
                + "G"
                + String(g)
                + " sum = "
                + String(grp_sum)[byte=:9],
                passed,
                total,
            )

    input_ptr.free()
    output_ptr.free()


# =============================================================================
# Test 2: Encoder output diversity at production dims, sweeping init std
# =============================================================================

# Production TDMPC2 HalfCheetah encoder dimensions.
comptime OBS = 17
comptime ENC = 256
comptime LATENT = 512
comptime SIMPLEX = 8
comptime BATCH = 256

comptime EncModel = Sequential[
    NormedLinear[OBS, ENC],
    Linear[ENC, LATENT],
    LayerNorm[LATENT],
    SimNorm[LATENT, SIMPLEX],
]


def _measure_diversity[
    M: Model, BS: Int
](
    z_t: LayoutTensor[
        dtype, Layout.row_major(BS, M.OUT_DIM), MutAnyOrigin
    ],
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Return (mean_std_per_dim, mean_z, mean_grp_max, max_grp_max).

    Assumes M.OUT_DIM is a SimNorm output (groups of SIMPLEX=8 summing to 1).
    """
    comptime D = M.OUT_DIM
    comptime N_GROUPS = D // SIMPLEX

    var sum_std: Float64 = 0.0
    var sum_z: Float64 = 0.0
    for k in range(D):
        var s: Float64 = 0.0
        var sq: Float64 = 0.0
        for b in range(BS):
            var v = Float64(z_t[b, k][0])
            s += v
            sq += v * v
            sum_z += v
        var mean_k = s / Float64(BS)
        var var_k = sq / Float64(BS) - mean_k * mean_k
        if var_k < 0.0:
            var_k = 0.0
        sum_std += sqrt(var_k)
    var mean_std_per_dim = sum_std / Float64(D)
    var mean_z = sum_z / Float64(BS * D)

    var sum_grp_max: Float64 = 0.0
    var max_grp_max: Float64 = 0.0
    for b in range(BS):
        for g in range(N_GROUPS):
            var base = g * SIMPLEX
            var grp_max: Float64 = Float64(z_t[b, base][0])
            for kk in range(1, SIMPLEX):
                var v = Float64(z_t[b, base + kk][0])
                if v > grp_max:
                    grp_max = v
            sum_grp_max += grp_max
            if grp_max > max_grp_max:
                max_grp_max = grp_max
    var mean_grp_max = sum_grp_max / Float64(BS * N_GROUPS)

    return (mean_std_per_dim, mean_z, mean_grp_max, max_grp_max)


def _print_row(sigma: String, r: Tuple[Float64, Float64, Float64, Float64]):
    print(
        "  σ="
        + sigma
        + "    mean_std(z)="
        + String(r[0])[byte=:9]
        + "    mean(z)="
        + String(r[1])[byte=:7]
        + "    mean_grp_max="
        + String(r[2])[byte=:7]
        + "    max_grp_max="
        + String(r[3])[byte=:7]
    )


def _sweep_one[
    M: Model, BS: Int, INIT_STD: Float64
](
    obs_t: LayoutTensor[
        dtype, Layout.row_major(BS, M.IN_DIM), MutAnyOrigin
    ],
    mut z_t: LayoutTensor[
        dtype, Layout.row_major(BS, M.OUT_DIM), MutAnyOrigin
    ],
    sigma_label: String,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    _forward_once[M, BS, Normal[0.0, INIT_STD]](obs_t, z_t)
    var r = _measure_diversity[M, BS](z_t)
    _print_row(sigma_label, r)
    return r


def test_init_sweep(mut passed: Int, mut total: Int) raises:
    print(
        "\n=== Test 2: Encoder output diversity at production dims"
        " (OBS=17, ENC=256, LATENT=512, SIMPLEX=8, BATCH=256) ==="
    )
    print(
        "  Sweep Normal(0, σ) on Linear/NormedLinear W;"
        " LayerNorm γ=1; SimNorm has no params."
    )

    var obs_ptr = alloc[Scalar[dtype]](BATCH * OBS)
    var z_ptr = alloc[Scalar[dtype]](BATCH * LATENT)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EncModel.IN_DIM), MutAnyOrigin
    ](obs_ptr)
    var z_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EncModel.OUT_DIM), MutAnyOrigin
    ](z_ptr)

    # Standard-Gaussian observations via Box–Muller.
    seed(42)
    var i = 0
    while i < BATCH * OBS:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-10:
            u1 = 1e-10
        var rr = sqrt(-2.0 * log(u1))
        var z0 = rr * cos(2.0 * PI * u2)
        (obs_ptr + i)[] = Scalar[dtype](z0)
        i += 1

    print("  --- obs ~ N(0, 1) ---")
    var _r01 = _sweep_one[EncModel, BATCH, 0.01](obs_t, z_t, "0.01")
    var r02 = _sweep_one[EncModel, BATCH, 0.02](obs_t, z_t, "0.02")
    var _r03 = _sweep_one[EncModel, BATCH, 0.03](obs_t, z_t, "0.03")
    var r05 = _sweep_one[EncModel, BATCH, 0.05](obs_t, z_t, "0.05")
    var _r10 = _sweep_one[EncModel, BATCH, 0.10](obs_t, z_t, "0.10")

    # Also try near-zero obs (HC at reset: joints near 0).
    print()
    print("  --- obs ~ N(0, 0.01) — small-scale (close to fresh-reset HC) ---")
    for ii in range(BATCH * OBS):
        (obs_ptr + ii)[] = Scalar[dtype](
            Float64((obs_ptr + ii)[]) * 0.01
        )
    var _rs01 = _sweep_one[EncModel, BATCH, 0.01](obs_t, z_t, "0.01")
    var _rs02 = _sweep_one[EncModel, BATCH, 0.02](obs_t, z_t, "0.02")
    var _rs05 = _sweep_one[EncModel, BATCH, 0.05](obs_t, z_t, "0.05")

    print()
    print("  Reference (analytical):")
    print(
        "    Each group sums to 1 over SIMPLEX=8 → mean_z = 1/8 = 0.125"
        " exactly (simplex invariant)."
    )
    print(
        "    Full collapse: per-dim std → 0, mean_grp_max → 0.125."
    )
    print(
        "    Healthy: per-dim std ≳ 0.05–0.15, mean_grp_max ≳ 0.25."
    )

    # Simplex invariant: regardless of init, mean_z must be 1/SIMPLEX.
    _expect(
        abs(r02[1] - 0.125) < 1e-4,
        "σ=0.02 mean(z) = "
        + String(r02[1])[byte=:8]
        + " (expected 0.125 — simplex invariant)",
        passed,
        total,
    )

    # σ=0.05/σ=0.02 ratio — informational. Linear-in-σ would give ~2.5;
    # super-linear would confirm σ=0.02 sits in low-diversity regime.
    if r02[0] > 0.0 and r05[0] > 0.0:
        var ratio = r05[0] / r02[0]
        print(
            "\n  σ=0.05 / σ=0.02 std-ratio: "
            + String(ratio)[byte=:6]
            + " (expect ~2.5 if linear in σ; >>2.5 = σ=0.02 in"
            " low-diversity regime)"
        )

    # σ=0.05 must give measurable diversity (production default — else
    # collapse is structural and not init-tunable).
    _expect(
        r05[0] > 0.01,
        "σ=0.05 std(z) = "
        + String(r05[0])[byte=:8]
        + " > 0.01 (production default — must be diverse)",
        passed,
        total,
    )

    # σ=0.02 informational: doc claimed std~0.022 at init. If this is
    # >>0.022 the collapse is training-driven, not init-driven.
    _expect(
        r02[0] >= 0.0,
        "σ=0.02 std(z) = "
        + String(r02[0])[byte=:8]
        + " (doc claimed ~0.022 at step 0)",
        passed,
        total,
    )

    obs_ptr.free()
    z_ptr.free()


# =============================================================================
# Main
# =============================================================================


def main() raises:
    print("=" * 70)
    print("SimNorm init-time output diversity probe")
    print("=" * 70)

    var passed: Int = 0
    var total: Int = 0

    test_simnorm_forward_correctness(passed, total)
    test_init_sweep(passed, total)

    print()
    print("=" * 70)
    print(
        "Summary: " + String(passed) + " / " + String(total) + " checks passed"
    )
    print("=" * 70)
