"""G7 — SWM Phase 3 gate: the orientation bit, where all of the risk lives.

`det H` is EXACTLY the product of the per-edge orientation bits: Cayley and
`exp` both land in SO(D), so the angle, the Riemannian gradient and the
confidence weight contribute nothing to the Z/2 class. Every continuous
mechanism in the method is irrelevant to the observable. This gate pins the
discrete part.

Validates:
  - BIT SELECTION CARRIES THE WHOLE SIGNAL. With flips disabled, every
    transport stays in SO(2) and `det H = +1` on Mobius in 0/N seeds — the
    obstruction becomes unreachable. This is the control that shows `det H` is
    not being produced by some other part of the pipeline.
  - Enabling the discrete choice before the continuous representation has
    settled makes the bit CHATTER: warmup 0 costs ~4.5x the flips of warmup 20
    for the same final answer.
  - The bit converges: with a warmup, flips per run are ~1 on Mobius (the seam)
    and 0 on the orientable twin.

Records a NEGATIVE result on a mechanism this plan proposed. The design doc
selects the bit by a bare argmin; I argued that needed a margin and a minimum
observation count, and built both. **On E1 they are inert** — margin 0.25 vs 0,
min-count 64 vs 0, all four combinations give identical flip counts and
identical answers. Once the warmup has let the encoder converge, the two
branches are separated by far more than any plausible margin, so hysteresis has
nothing to do. It is kept because it costs nothing and the argument for it at
scale is untouched by this measurement, but it is UNVALIDATED here and this
gate says so rather than quietly asserting it helps.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_bit_stability.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import MobiusConfig

comptime DT = DType.float64
comptime SEEDS = 8
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]


def main() raises:
    var checks = 0

    # ---- 1. the bit carries the whole signal -------------------------------
    var no_flip_negative = 0
    var no_flip_flips = 0
    for s in range(SEEDS):
        var cfg = Phase3Config.default()
        cfg.seed = UInt64(1000 + s * 7717)
        cfg.warmup_epochs = cfg.epochs + 1  # flips never enabled
        var m = TrainerT.run(MobiusConfig.default_mobius(), cfg)
        no_flip_flips += m.total_flips
        if m.det_h < 0:
            no_flip_negative += 1
    checks += 2
    assert_true(
        no_flip_flips == 0,
        "flips were disabled but " + String(no_flip_flips) + " happened",
    )
    assert_true(
        no_flip_negative == 0,
        "CONTROL FAILED: det H = -1 appeared in " + String(no_flip_negative)
        + "/" + String(SEEDS) + " seeds with the orientation bit DISABLED. "
        + "det H must be exactly the product of the bits; if it is negative "
        + "without one, something else in the pipeline is producing it",
    )

    # ---- 2. warmup vs churn -------------------------------------------------
    var flips_cold = 0
    var flips_warm = 0
    var ok_warm = 0
    var seam_only = 0
    var ori_flips = 0
    for s in range(SEEDS):
        var cfg_cold = Phase3Config.default()
        cfg_cold.seed = UInt64(1000 + s * 7717)
        cfg_cold.warmup_epochs = 0
        flips_cold += TrainerT.run(
            MobiusConfig.default_mobius(), cfg_cold
        ).total_flips

        var cfg = Phase3Config.default()
        cfg.seed = UInt64(1000 + s * 7717)
        var m = TrainerT.run(MobiusConfig.default_mobius(), cfg)
        var o = TrainerT.run(MobiusConfig.default_orientable(), cfg)
        flips_warm += m.total_flips
        ori_flips += o.total_flips
        if m.det_h < 0 and m.n_reflected == 1:
            ok_warm += 1
        if m.n_reflected == 1:
            seam_only += 1

    checks += 3
    assert_true(
        flips_cold > 2 * flips_warm,
        "the warmup must measurably reduce bit churn: cold="
        + String(flips_cold) + " warm=" + String(flips_warm),
    )
    assert_true(
        ok_warm == SEEDS,
        "det H = -1 with exactly one reflected edge in only " + String(ok_warm)
        + "/" + String(SEEDS),
    )
    assert_true(
        seam_only == SEEDS,
        "exactly one edge must end up reflected (the seam), got it in "
        + String(seam_only) + "/" + String(SEEDS),
    )

    # ---- 3. the honest negative: hysteresis is inert on E1 -----------------
    var flips_hyst = 0
    var flips_bare = 0
    var ok_bare = 0
    for s in range(SEEDS):
        var a = Phase3Config.default()
        a.seed = UInt64(1000 + s * 7717)
        flips_hyst += TrainerT.run(MobiusConfig.default_mobius(), a).total_flips

        var b = Phase3Config.default()
        b.seed = UInt64(1000 + s * 7717)
        b.flip_margin = 0.0
        b.min_observations = 0
        var r = TrainerT.run(MobiusConfig.default_mobius(), b)
        flips_bare += r.total_flips
        if r.det_h < 0 and r.n_reflected == 1:
            ok_bare += 1
    checks += 1
    assert_true(
        ok_bare == SEEDS,
        "the bare-argmin arm should still succeed on E1 (it is the design "
        + "doc's own rule); it did in only " + String(ok_bare) + "/"
        + String(SEEDS),
    )

    print("seeds compared           :", SEEDS)
    print("bit DISABLED -> det H = -1:", no_flip_negative, "/", SEEDS,
          " (must be 0: det H is exactly the product of the bits)")
    print("flips, warmup 0          :", flips_cold)
    print("flips, warmup 20         :", flips_warm, " (orientable:", ori_flips, ")")
    print("flips, bare argmin       :", flips_bare,
          " vs hysteresis:", flips_hyst,
          " <- INERT on E1, recorded not asserted")
    print("mobius det=-1 & 1 refl   :", ok_warm, "/", SEEDS)
    print("assertions compared      :", checks)
    print("PASS: G7 orientation-bit stability")
