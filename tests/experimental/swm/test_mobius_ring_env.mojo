"""G5 — SWM Phase 2 gate: the Mobius corridor, and the nuisance that makes E1 real.

This gate welds Phase 2 to Phase 1: the environment's planted transports AND
transports recovered from its observed data must both produce `det H = -1`,
using the same `place_graph` / `procrustes` code the Phase 1 gates pinned.

It also enforces the property that makes E1 a test rather than a tautology.
The design doc's E1 is `x = g_i w` with `g_i in O(2)` and nothing else — an
orthogonal group action and no "rest", so hypothesis 4.0 (the topologically
relevant part is SEPARABLE from ordinary content) could not be falsified by it.
Here each cell also carries a texture that identifies it, is IDENTICAL at both
lap parities, and is NOT transported; observation is an overcomplete mixing of
the two. So the aliasing is sharp — texture says "same cell", landmark says
"mirrored" — and Phase 3's encoder must FIND the transported subspace.

Validates:
  - a lap returns to the same cell; an ODD lap mirrors the landmark exactly
    (`H = diag(1, -1)` to 1e-12), an EVEN lap restores it
  - the texture is bit-identical across parities, so the mirroring is carried
    ONLY by the transported subspace
  - the mixing is non-trivial: every observed coordinate blends landmark with
    texture, so nothing can be read off a coordinate
  - backward undoes forward (the transport is orthogonal, so `R^T = R^-1`)
  - planted transports -> `det H = -1`; transports RECOVERED by Procrustes from
    noisy observed landmark pairs -> `det H = -1` too
  - the reward is genuinely parity-dependent
  - determinism: same seeds, identical trajectories
  - NEGATIVE CONTROL: the orientable world must show NO mirroring, `det H = +1`
    from both planted and recovered transports, and identical observations
    across parities. Without it, code that always answered "obstructed" would
    pass everything above.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_mobius_ring_env.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.procrustes import PairBatch, procrustes_o_d
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.envs.mobius_ring import MobiusRing, MobiusConfig

comptime DT = DType.float64
comptime N = 12
comptime NUIS = 6
comptime OBS = 16
comptime EnvT = MobiusRing[N, NUIS, OBS, DT]


def ring_det_h(transports: List[SqMat[2, DT]]) raises -> Float64:
    """Build the ring in `place_graph` and read the holonomy of its one cycle."""
    var g = PlaceGraph[2, DT]()
    for _ in range(N):
        _ = g.add_place()
    for i in range(N):
        _ = g.add_edge(Edge.action_edge(i, (i + 1) % N, 0), transports[i])
    g.rebuild_gauge(0)
    var cyc = g.fundamental_cycle_edges()
    if len(cyc) != 1:
        raise Error("ring must have exactly one fundamental cycle")
    return g.holonomy_det(cyc[0])


def recovered_det_h(mobius: Bool, mut checks: Int) raises -> Float64:
    """Fit each edge from NOISY observed landmark pairs, then read the cycle.

    This is the leg that makes the gate about DATA rather than about the
    matrices we planted: the same Newton-polar Procrustes that G4 pinned
    against numpy, fed by the environment.
    """
    var cfg = (
        MobiusConfig.default_mobius(7717)
        if mobius
        else MobiusConfig.default_orientable(7717)
    )
    var env = EnvT(cfg)
    var noise = Rng(99991)
    var batches = List[PairBatch[2, DT]]()
    for _ in range(N):
        batches.append(PairBatch[2, DT]())

    comptime EPISODES = 40
    comptime LAPS = 4
    var pairs_pushed = 0
    for ep in range(EPISODES):
        env.reset(UInt64(1000 + ep))
        var prev = env.true_landmark()
        for _step in range(LAPS * N):
            var edge = env.place_id()
            env.step(0)
            var cur = env.true_landmark()
            var x = InlineArray[Scalar[DT], 2](fill=0)
            var y = InlineArray[Scalar[DT], 2](fill=0)
            for c in range(2):
                x[c] = prev[c] + Scalar[DT](noise.normal() * cfg.obs_noise)
                y[c] = cur[c] + Scalar[DT](noise.normal() * cfg.obs_noise)
            batches[edge].push(x, y)
            pairs_pushed += 1
            prev = cur.copy()
    checks += 1
    assert_true(
        pairs_pushed == EPISODES * LAPS * N,
        "expected " + String(EPISODES * LAPS * N) + " pairs, pushed "
        + String(pairs_pushed),
    )

    var fits = List[SqMat[2, DT]]()
    for e in range(N):
        fits.append(procrustes_o_d[2, DT](batches[e]))
    return ring_det_h(fits)


def check_world(mobius: Bool, mut checks: Int) raises:
    var label = "mobius" if mobius else "orientable"
    var cfg = (
        MobiusConfig.default_mobius()
        if mobius
        else MobiusConfig.default_orientable()
    )
    var env = EnvT(cfg)

    # ---- mixing is non-trivial: nothing is readable off a coordinate --------
    for r in range(OBS):
        var lm_mag = Float64(0)
        var nu_mag = Float64(0)
        for c in range(2):
            lm_mag += abs(Float64(env.mix[r * EnvT.LATENT_DIM + c]))
        for c in range(2, EnvT.LATENT_DIM):
            nu_mag += abs(Float64(env.mix[r * EnvT.LATENT_DIM + c]))
        checks += 2
        assert_true(
            lm_mag >= 0.25,
            "obs coord " + String(r) + " does not see the landmark",
        )
        assert_true(
            nu_mag >= 0.25,
            "obs coord " + String(r) + " does not see the texture — it would be "
            + "a free frame readout",
        )

    # ---- an observation is a MEASUREMENT, not a lookup ----------------------
    # Found by mutation: nothing here noticed a silently noise-free env, which
    # would make Phase 3's learning problem easier than advertised and would
    # render every residual floor meaningless.
    env.reset(8080)
    var o1 = env.observation()
    var o2 = env.observation()
    var obs_drift = Float64(0)
    for k in range(OBS):
        obs_drift += abs(Float64(o1[k] - o2[k]))
    var expected = Float64(OBS) * cfg.obs_noise
    checks += 2
    assert_true(
        obs_drift > 0.2 * expected,
        label + ": two observations of the SAME state are identical — the "
        + "observation noise is not reaching the output (drift="
        + String(obs_drift) + ")",
    )
    assert_true(
        obs_drift < 8.0 * expected,
        label + ": observation noise is far larger than configured (drift="
        + String(obs_drift) + ", expected ~" + String(expected) + ")",
    )

    # ---- laps, mirroring, and the texture that does NOT move ---------------
    env.reset(4242)
    var lm_lap0 = env.true_landmark()
    var nu_lap0 = env.nuisance_at(env.place_id())
    var start_cell = env.place_id()

    for _ in range(N):
        env.step(0)
    checks += 2
    assert_true(env.place_id() == start_cell, "a lap must return to the cell")
    assert_true(
        env.lap_parity() == 1, "one lap must cross the seam an odd number of times"
    )
    var lm_lap1 = env.true_landmark()
    var nu_lap1 = env.nuisance_at(env.place_id())

    for _ in range(N):
        env.step(0)
    checks += 1
    assert_true(env.lap_parity() == 0, "two laps must restore even parity")
    var lm_lap2 = env.true_landmark()

    # The texture is the invariant half: identical, bit for bit.
    for k in range(NUIS):
        checks += 1
        assert_true(
            nu_lap0[k] == nu_lap1[k],
            "texture moved between parities at " + label + " — the nuisance "
            + "channel must NOT be transported",
        )

    var mirror_err = abs(Float64(lm_lap1[0] - lm_lap0[0])) + abs(
        Float64(lm_lap1[1] + lm_lap0[1])
    )
    var same_err = abs(Float64(lm_lap1[0] - lm_lap0[0])) + abs(
        Float64(lm_lap1[1] - lm_lap0[1])
    )
    var restore_err = abs(Float64(lm_lap2[0] - lm_lap0[0])) + abs(
        Float64(lm_lap2[1] - lm_lap0[1])
    )
    checks += 1
    assert_true(
        restore_err <= 1e-12,
        label + ": two laps must restore the landmark, err=" + String(restore_err),
    )
    if mobius:
        checks += 2
        assert_true(
            mirror_err <= 1e-12,
            "mobius: one lap must MIRROR the landmark (H = diag(1,-1)), err="
            + String(mirror_err),
        )
        assert_true(
            same_err > 0.1,
            "mobius: one lap must CHANGE the landmark — if it does not, the "
            + "seam is not doing anything and every later gate is vacuous",
        )
    else:
        checks += 1
        assert_true(
            same_err <= 1e-12,
            "NEGATIVE CONTROL FAILED: orientable world mirrored the landmark, "
            + "err=" + String(same_err),
        )

    # ---- backward undoes forward (orthogonality: R^T = R^-1) ---------------
    env.reset(31337)
    var before = env.true_landmark()
    for _ in range(N + 5):
        env.step(0)
    for _ in range(N + 5):
        env.step(1)
    var after = env.true_landmark()
    checks += 2
    assert_true(env.place_id() == 0, "walk out and back must return to the cell")
    assert_true(
        abs(Float64(after[0] - before[0])) + abs(Float64(after[1] - before[1]))
        <= 1e-12,
        label + ": backward did not undo forward",
    )

    # ---- planted vs recovered holonomy -------------------------------------
    var planted = List[SqMat[2, DT]]()
    for i in range(N):
        planted.append(env.edge_transport(i))
    var det_planted = ring_det_h(planted)
    var det_recovered = recovered_det_h(mobius, checks)
    var want = -1.0 if mobius else 1.0
    checks += 2
    assert_true(
        abs(det_planted - want) <= 1e-9,
        label + ": planted det H = " + String(det_planted) + ", want " + String(want),
    )
    assert_true(
        abs(det_recovered - want) <= 1e-9,
        label + ": det H RECOVERED from observed pairs = "
        + String(det_recovered) + ", want " + String(want),
    )
    print(
        "  " + label + ": det H planted =", det_planted,
        " recovered from data =", det_recovered,
    )


def main() raises:
    var checks = 0

    check_world(True, checks)
    check_world(False, checks)

    # ---- the reward really does depend on parity, at EVERY seed ------------
    # An argmax over all 2N frames, so a goal always exists; the leg that
    # matters is that reaching the goal CELL at the wrong parity pays nothing.
    # (An earlier tolerance-based reward satisfied this in only ~45% of
    # episodes — an ill-posed task, not an unlucky seed.)
    var env = EnvT(MobiusConfig.default_mobius())
    var episodes_checked = 0
    var degenerate = 0
    for ep in range(50):
        env.reset(UInt64(500 + ep))
        var gc = env.goal_cell()
        var gp = env.goal_parity()
        if env.goal_margin() < 1e-9:
            degenerate += 1
            continue
        var r_even = List[Float64]()
        var r_odd = List[Float64]()
        for _ in range(N):
            r_even.append(env.reward())
            env.step(0)
        for _ in range(N):
            r_odd.append(env.reward())
            env.step(0)
        var hits = 0
        var differ = 0
        for i in range(N):
            if r_even[i] > 0 or r_odd[i] > 0:
                hits += 1
            if r_even[i] != r_odd[i]:
                differ += 1
        checks += 3
        assert_true(
            hits == 1,
            "episode " + String(ep) + ": exactly one goal state must exist, got "
            + String(hits),
        )
        assert_true(
            differ == 1,
            "episode " + String(ep) + ": the goal cell must pay at exactly ONE "
            + "parity — otherwise E1 needs no world model. differ="
            + String(differ),
        )
        assert_true(
            gc >= 0 and gc < N and (gp == 0 or gp == 1),
            "episode " + String(ep) + ": malformed goal",
        )
        episodes_checked += 1
    checks += 2
    assert_true(
        episodes_checked >= 45,
        "too few non-degenerate episodes: " + String(episodes_checked),
    )
    assert_true(degenerate == 0, "degenerate goals: " + String(degenerate))

    # NEGATIVE CONTROL on the task itself: in the ORIENTABLE world the two
    # parities are the same frame, so the goal is an exact tie and the task
    # carries no parity at all. This is what shows the difficulty above comes
    # from the SEAM and not from the way the reward is written.
    var env_o = EnvT(MobiusConfig.default_orientable())
    var ties = 0
    for ep in range(20):
        env_o.reset(UInt64(500 + ep))
        if env_o.goal_margin() <= 1e-12:
            ties += 1
    checks += 1
    assert_true(
        ties == 20,
        "NEGATIVE CONTROL FAILED: the orientable world must have NO parity "
        + "structure (goal tied across parities), tied in only "
        + String(ties) + "/20 episodes",
    )

    # ---- determinism --------------------------------------------------------
    var a = EnvT(MobiusConfig.default_mobius())
    var b = EnvT(MobiusConfig.default_mobius())
    a.reset(24601)
    b.reset(24601)
    var drift = Float64(0)
    for i in range(3 * N):
        a.step(i % 2)
        b.step(i % 2)
        var oa = a.observation()
        var ob = b.observation()
        for k in range(OBS):
            drift += abs(Float64(oa[k] - ob[k]))
    checks += 1
    assert_true(
        drift == 0.0,
        "same seed must give identical trajectories, drift=" + String(drift),
    )

    print("cells:", N, " nuisance dims:", NUIS, " observation dims:", OBS)
    print("worlds compared    : 2 (mobius + orientable control)")
    print("episodes with a unique, parity-gated goal:", episodes_checked, "of 50")
    print("orientable control: parity-tied goals:", ties, "of 20")
    print("assertions compared:", checks)
    print("PASS: G5 Mobius corridor with non-transported nuisance")
