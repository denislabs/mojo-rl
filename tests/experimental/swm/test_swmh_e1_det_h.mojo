"""G6 — SWM Phase 3 gate, and the DECISION POINT of the whole plan (P1).

Hypothesis 4.0 says the topologically relevant part of the state is SEPARABLE:
carried by an orthogonally transported channel, with the rest in ordinary
content. Everything after Phase 3 is built on it. This gate asks whether it
holds with LEARNED encoders on observations that mix a transported landmark
with a non-transported per-cell texture.

Both worlds run ON THE SAME BINARY, and that is not a formality. A learned
encoder whose gauge is unstable across places manufactures obstructions, so
`det H = -1` on Mobius alone proves nothing: it has to be accompanied by
`det H = +1` on the orientable twin. Measured during development, before the
per-place anti-collapse term existed: `det H = -1` came out in 2/6 Mobius seeds
AND 2/6 orientable seeds — a fair coin that would have read as a successful P1.

Validates:
  - Mobius   -> det H = -1 with EXACTLY ONE reflected edge (the seam)
  - orientable -> det H = +1 with ZERO reflected edges  [NEGATIVE CONTROL]
  - the frame channel found the landmark (R^2 high) and rejected the texture
    (R^2 low) — the direct measurement of hypothesis 4.0, not a proxy
  - the frame channel did NOT collapse: anisotropy well above 0, and non-zero
    within-place spread. A collapsed frame makes a reflection act like the
    identity, so `det H = +1` would be INVALID rather than negative, and the two
    are indistinguishable from det H alone.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_e1_det_h.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import MobiusConfig

comptime DT = DType.float64
comptime SEEDS = 24
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]


def main() raises:
    var checks = 0
    var mob_ok = 0
    var ori_ok = 0
    var false_obstructions = 0
    var lm_min = 1.0
    var nu_max = 0.0
    var aniso_min = 1.0
    var wps_min = 1e9

    for s in range(SEEDS):
        var cfg = Phase3Config.default()
        cfg.seed = UInt64(1000 + s * 7717)

        var m = TrainerT.run(MobiusConfig.default_mobius(), cfg)
        var o = TrainerT.run(MobiusConfig.default_orientable(), cfg)

        # ---- the frame channel must be VALID before det H means anything ---
        for r in [m, o]:
            checks += 2
            assert_true(
                r.u_anisotropy > 0.05,
                "seed " + String(s) + ": frame channel COLLAPSED (anisotropy "
                + String(r.u_anisotropy) + ") — det H is meaningless here",
            )
            assert_true(
                r.within_place_std > 0.05,
                "seed " + String(s) + ": frame channel is a place-indexed "
                + "constant (within-place std " + String(r.within_place_std)
                + ") — the transport constraint is vacuous and the orientation "
                + "bit is decided by noise",
            )
            if r.landmark_r2 < lm_min:
                lm_min = r.landmark_r2
            if r.nuisance_r2 > nu_max:
                nu_max = r.nuisance_r2
            if r.u_anisotropy < aniso_min:
                aniso_min = r.u_anisotropy
            if r.within_place_std < wps_min:
                wps_min = r.within_place_std

        # ---- hypothesis 4.0, measured directly -----------------------------
        checks += 2
        assert_true(
            m.landmark_r2 > 0.9,
            "seed " + String(s) + ": the frame channel did not find the "
            + "transported subspace (landmark R^2 " + String(m.landmark_r2)
            + ") — hypothesis 4.0 fails on its own terms",
        )
        assert_true(
            m.nuisance_r2 < 0.1,
            "seed " + String(s) + ": texture leaked into the frame channel "
            + "(nuisance R^2 " + String(m.nuisance_r2) + ")",
        )

        # ---- the observable -------------------------------------------------
        if m.det_h < 0 and m.n_reflected == 1:
            mob_ok += 1
        if o.det_h > 0 and o.n_reflected == 0:
            ori_ok += 1
        if o.det_h < 0:
            false_obstructions += 1

    checks += 3
    assert_true(
        mob_ok == SEEDS,
        "Mobius: det H = -1 with exactly one reflected edge in only "
        + String(mob_ok) + "/" + String(SEEDS) + " seeds",
    )
    assert_true(
        false_obstructions == 0,
        "NEGATIVE CONTROL FAILED: the orientable world was reported obstructed "
        + "in " + String(false_obstructions) + "/" + String(SEEDS)
        + " seeds — a learned encoder is manufacturing obstructions",
    )
    assert_true(
        ori_ok == SEEDS,
        "orientable: det H = +1 with zero reflected edges in only "
        + String(ori_ok) + "/" + String(SEEDS) + " seeds",
    )

    print("seeds compared        :", SEEDS, "(mobius + orientable on the SAME binary)")
    print("mobius  det H = -1, 1 reflected edge :", mob_ok, "/", SEEDS)
    print("orient. det H = +1, 0 reflected edges:", ori_ok, "/", SEEDS)
    print("false obstructions    :", false_obstructions, "/", SEEDS)
    print("worst landmark R^2    :", lm_min, " (frame channel found the landmark)")
    print("worst nuisance R^2    :", nu_max, " (texture kept out)")
    print("worst anisotropy      :", aniso_min)
    print("worst within-place std:", wps_min)
    print("assertions compared   :", checks)
    print("PASS: G6 det H survives learned encoders (P1)")
