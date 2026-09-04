"""G10 + G11 — SWM Phase 4 gates: inference as descent, and why the removal rule.

G10 pins the mechanics: descent on the sheaf Dirichlet energy is monotone and
lands on the exact stationary point.

G11 is the one that earns its place. The design doc asserts that an
identification whose cycle carries a non-trivial holonomy must be REMOVED from
the energy, because it is not a constraint but a monodromy, and including it
would force the frame channel into the cycle's fixed subspace. That is a
positive claim about a failure, and it is measured here rather than assumed.

The measurement is a 2x2: {Mobius, orientable} x {identification in energy,
out}. Only ONE cell may move. If the collapse showed up when a *consistent*
identification is added, it would be an artefact of adding a constraint rather
than of adding a contradictory one.

Measured anisotropy of the inferred frame at the revisited place:

    beta   mobius+ID   mobius-ID   orient+ID   orient-ID
    0.02     0.092       0.943       0.943       0.943
    0.10     0.157       0.944       0.943       0.944
    0.50     0.340       0.945       0.944       0.945
    1.00     0.456       0.945       0.944       0.945
    5.00     0.756       0.946       0.945       0.946

The beta dependence is the doc's §4.4 mechanism, visible: with WEAK anchors the
inference spreads the disagreement and the frame collapses toward the cycle's
one-dimensional fixed subspace (a 10x loss of anisotropy); with STRONG anchors
the anchors win and the disagreement is deposited on the identification edge
instead. Either way the pre-consensus residual is the only honest place to read
an edge from, which is why `observables` measures it before inference and never
after.

Validates:
  - the energy is monotone non-increasing under the Jacobi step
  - the iterate converges to `solve_frames_exact` (a dense solve of the
    stationary equations) to 1e-12
  - on an orientable ring, adding a consistent identification changes NOTHING
  - on Mobius, adding the contradictory identification collapses the frame, and
    the collapse deepens as the anchors weaken
  - NEGATIVE CONTROL: the three non-Mobius-with-ID cells stay flat and high at
    every beta

Run:
    pixi run mojo run -I . tests/experimental/swm/test_sheaf_inference.mojo
"""

from std.math import abs, cos, sin
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.sheaf_inference import (
    InferenceConfig,
    frame_energy,
    frame_gradient,
    infer_frames,
    solve_frames_exact,
    frame_covariance_anisotropy,
)

comptime DT = DType.float64
comptime N = 12
comptime TRIALS = 120


def build_ring(mobius: Bool, id_in_energy: Bool) raises -> PlaceGraph[2, DT]:
    """A lap with a REVISIT: N+1 places, plus an identification (N == 0).

    A new place every step, always; recognition adds an edge and never merges
    vertices (v2 §4.1). The identification is therefore always a non-tree edge
    and contributes exactly one fundamental cycle.
    """
    var rng = Rng(11)
    var g = PlaceGraph[2, DT]()
    for _ in range(N + 1):
        _ = g.add_place()
    var total = Float64(0)
    for i in range(N):
        var t = rng.uniform_range(-0.6, 0.6)
        if i == N - 1:
            t = -total
        else:
            total += t
        var m = SqMat[2, DT]()
        m[0, 0] = Scalar[DT](cos(t))
        m[0, 1] = Scalar[DT](-sin(t))
        m[1, 0] = Scalar[DT](sin(t))
        m[1, 1] = Scalar[DT](cos(t))
        if mobius and i == N - 1:
            var refl = SqMat[2, DT].identity()
            refl[1, 1] = Scalar[DT](-1)
            m = refl * m
        _ = g.add_edge(Edge.action_edge(i, i + 1, 0), m)
    var ide = Edge.identification(N, 0)
    ide.in_energy = id_in_energy
    _ = g.add_edge(ide, SqMat[2, DT].identity())
    g.rebuild_gauge(0)
    return g^


def collapse_anisotropy(
    mobius: Bool, id_in: Bool, beta: Float64
) raises -> Float64:
    """Anisotropy of the inferred frame at the revisited place, over trials."""
    var g = build_ring(mobius, id_in)
    var cfg = InferenceConfig.default()
    cfg.beta = beta
    var rng = Rng(999)
    var samples = List[Float64]()
    for _ in range(TRIALS):
        var a = rng.uniform_range(0.0, 6.283185307179586)
        var w = List[Float64](length=2, fill=0)
        w[0] = cos(a)
        w[1] = sin(a)
        var anchors = List[Float64](length=(N + 1) * 2, fill=0)
        var has = List[Bool](length=N + 1, fill=True)
        for p in range(N + 1):
            for i in range(2):
                var s = Float64(0)
                for j in range(2):
                    s += Float64(g.t_root[p][i, j]) * w[j]
                anchors[p * 2 + i] = s + rng.normal() * 0.02
        var ws = List[Float64](length=(N + 1) * 2, fill=0)
        var u = infer_frames[2, DT](g, anchors, has, cfg, ws)
        samples.append(u[0])
        samples.append(u[1])
    return frame_covariance_anisotropy(samples, TRIALS)


def main() raises:
    var checks = 0

    # =====================================================================
    # G10 — the mechanics
    # =====================================================================
    var g = build_ring(True, False)
    var cfg = InferenceConfig.default()
    var rng = Rng(4242)
    var anchors = List[Float64](length=(N + 1) * 2, fill=0)
    var has = List[Bool](length=N + 1, fill=True)
    for i in range((N + 1) * 2):
        anchors[i] = rng.uniform_range(-1.0, 1.0)

    var u = List[Float64](length=(N + 1) * 2, fill=0)
    var grad = List[Float64](length=(N + 1) * 2, fill=0)
    var prev = frame_energy[2, DT](g, u, anchors, has, cfg)
    var e0 = prev
    var rises = 0
    for _ in range(cfg.steps):
        frame_gradient[2, DT](g, u, anchors, has, cfg, grad)
        for i in range(len(u)):
            u[i] -= cfg.lr * grad[i]
        var e = frame_energy[2, DT](g, u, anchors, has, cfg)
        if e > prev + 1e-12:
            rises += 1
        prev = e
    checks += 2
    assert_true(rises == 0, "the energy rose on " + String(rises) + " steps")
    assert_true(
        prev < e0 - 1e-6,
        "the energy must actually fall: " + String(e0) + " -> " + String(prev),
    )

    var exact = solve_frames_exact[2, DT](g, anchors, has, cfg)
    var worst = Float64(0)
    for i in range(len(u)):
        var d = abs(u[i] - exact[i])
        if d > worst:
            worst = d
    checks += 1
    assert_true(
        worst < 1e-12,
        "the iterate must reach the exact stationary point, worst |diff| = "
        + String(worst),
    )
    print("G10  energy", e0, "->", prev, "  monotone steps:", cfg.steps - rises,
          "/", cfg.steps, "  |iterative - exact|:", worst)

    # =====================================================================
    # G11 — the removal rule, as a 2x2. Only one cell may move.
    # =====================================================================
    var betas: List[Float64] = [0.02, 0.1, 0.5, 1.0, 5.0]
    print("beta | mobius+ID | mobius-ID | orient+ID | orient-ID")
    var prev_collapse = Float64(0)
    for k in range(len(betas)):
        var b = betas[k]
        var m_in = collapse_anisotropy(True, True, b)
        var m_out = collapse_anisotropy(True, False, b)
        var o_in = collapse_anisotropy(False, True, b)
        var o_out = collapse_anisotropy(False, False, b)
        print(b, "|", m_in, "|", m_out, "|", o_in, "|", o_out)

        checks += 4
        # The three control cells must stay flat and high.
        assert_true(
            m_out > 0.8,
            "beta " + String(b) + ": removing the identification must leave "
            + "the frame uncollapsed, got " + String(m_out),
        )
        assert_true(
            o_out > 0.8,
            "beta " + String(b) + ": orientable without ID, got " + String(o_out),
        )
        assert_true(
            o_in > 0.8,
            "NEGATIVE CONTROL FAILED: adding a CONSISTENT identification "
            + "collapsed the frame (" + String(o_in) + ") at beta " + String(b)
            + " — then the collapse is an artefact of adding a constraint, not "
            + "of adding a contradictory one",
        )
        assert_true(
            abs(o_in - o_out) < 0.02,
            "on an orientable ring the identification must be a no-op: "
            + String(o_in) + " vs " + String(o_out),
        )

        # The one cell that moves.
        checks += 1
        assert_true(
            m_in < m_out - 0.1,
            "beta " + String(b) + ": including the contradictory "
            + "identification must collapse the frame, got " + String(m_in)
            + " vs " + String(m_out) + " without it",
        )
        if b <= 0.02:
            checks += 1
            assert_true(
                m_in < 0.2,
                "with weak anchors the collapse must be severe (the frame is "
                + "driven into the cycle's 1-D fixed subspace), got "
                + String(m_in),
            )
        # §4.4's mechanism: strong anchors deposit the disagreement on the
        # identification edge instead, so the collapse must WEAKEN with beta.
        checks += 1
        assert_true(
            m_in > prev_collapse - 1e-9,
            "the collapse must weaken as the anchors strengthen (that is the "
            + "mechanism, not a coincidence): " + String(m_in) + " after "
            + String(prev_collapse),
        )
        prev_collapse = m_in

    print()
    print("trials per cell     :", TRIALS, " betas:", len(betas))
    print("assertions compared :", checks)
    print("PASS: G10 inference = descent, G11 the removal rule is necessary")
