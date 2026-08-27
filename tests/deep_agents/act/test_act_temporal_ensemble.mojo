# +--------------------------------------------------------------------------+ #
# | M7 gate — temporal ensembling
# +--------------------------------------------------------------------------+ #
"""Gates `deep_agents/act/inference.mojo` against the reference's own
`[T, T+K, A]` buffer scheme.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo run -I . tests/deep_agents/act/test_act_temporal_ensemble.mojo

The two implementations differ deliberately — a `K x K` ring with explicit
occupancy versus a `T x (T+K)` buffer with `all(!= 0)` occupancy — so agreeing
on every timestep of a 12-step episode is a real check rather than a
tautology.

The check that matters most is the WEIGHT ORDER. `w_i = exp(-m*i)` with `i = 0`
the OLDEST chunk means the oldest prediction dominates. Reading it the other way
round is the natural mistake, it produces a perfectly plausible smoothed output,
and it inverts the behaviour ensembling exists for. So there is a direct
assertion on the direction, not only on the aggregate.
"""

from std.math import exp

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.inference import TemporalEnsemble, denormalize
from mojo_rl.deep_agents.act.refload import RefDump


comptime REF_DIR = "/tmp/act_ref"

# Must match `dump_act_reference.py:section_ensemble`.
comptime T = 12
comptime K = 4
comptime A = 3
comptime M = 0.01


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def main() raises:
    var fails = 0
    print("Temporal ensembling gate (reference: " + String(REF_DIR) + ")")
    print("")

    var d = RefDump(String(REF_DIR))
    var chunks = d.get(String("ens_chunks"))  # (T, K, A)
    var ref_out = d.get(String("ens_out"))  # (T, A)

    var te = TemporalEnsemble[A, K](m=M)
    var out = List[Scalar[DT]](length=T * A, fill=Scalar[DT](0.0))
    var worst = Float64(0.0)
    var contrib_ok = True

    for t in range(T):
        te.push(t, chunks, t * K * A)
        te.action_at(t, out, t * A)
        for j in range(A):
            worst = max(
                worst,
                abs(Float64(out[t * A + j]) - Float64(ref_out[t * A + j])),
            )
        # Warm-up: t+1 chunks cover step t until the ring is full, then K.
        var want = t + 1 if t + 1 < K else K
        if te.n_contributors(t) != want:
            contrib_ok = False

    check(
        fails,
        "ensembled action matches the reference at every step",
        worst < 1e-6,
        "max|diff| over " + String(T) + " steps = " + String(worst),
    )
    check(
        fails,
        "contributor count ramps 1..K then holds",
        contrib_ok,
        "at t=" + String(T - 1) + ": " + String(te.n_contributors(T - 1)),
    )

    # ── the weight ORDER, asserted directly ──────────────────────────────
    # At a step with a full ring, the ensembled action must sit CLOSER to the
    # oldest contributing chunk's prediction than to the newest. Constructed so
    # the two disagree strongly; an inverted `exp(-m*i)` flips this and nothing
    # else in the gate would notice.
    var te2 = TemporalEnsemble[A, K](m=1.0)  # m=1 makes the ordering stark
    var synth = List[Scalar[DT]](length=K * A, fill=Scalar[DT](0.0))
    for i in range(K):
        # Query i predicts the value `i` for EVERY step it covers, so the
        # contributions to a given step are exactly 0, 1, 2, 3 by query age.
        for c in range(K):
            for j in range(A):
                synth[c * A + j] = Scalar[DT](i)
        te2.push(i, synth, 0)
    var o2 = List[Scalar[DT]](length=A, fill=Scalar[DT](0.0))
    te2.action_at(K - 1, o2)

    # Weights at m=1 over ages 0..3 (oldest first): exp(0), e^-1, e^-2, e^-3.
    # Values by age: oldest query is i=0 (value 0), newest is i=3 (value 3).
    var wsum = Float64(0.0)
    var want = Float64(0.0)
    for r in range(K):
        var w = exp(-1.0 * Float64(r))
        wsum += w
        want += w * Float64(r)  # rank r == query i == value i
    want /= wsum
    check(
        fails,
        "w_i = exp(-m*i) with i=0 the OLDEST chunk",
        abs(Float64(o2[0]) - want) < 1e-5,
        "got " + String(Float64(o2[0])) + ", want " + String(want)
        + " (an inverted weighting would give "
        + String(Float64(K - 1) - want + 0.0) + ")",
    )
    check(
        fails,
        "the ensemble leans OLD, not new",
        Float64(o2[0]) < Float64(K - 1) / 2.0,
        String(Float64(o2[0])) + " < midpoint " + String(Float64(K - 1) / 2.0),
    )

    # ── reset clears the ring ────────────────────────────────────────────
    te2.reset()
    var raised = False
    try:
        te2.action_at(K - 1, o2)
    except:
        raised = True
    check(
        fails,
        "reset() empties the ring (action_at raises rather than returning 0)",
        raised,
    )

    # ── denormalize round-trips ──────────────────────────────────────────
    var mean = List[Scalar[DT]](length=A, fill=Scalar[DT](0.0))
    var std = List[Scalar[DT]](length=A, fill=Scalar[DT](0.0))
    for j in range(A):
        mean[j] = Scalar[DT](2.5 * Float64(j) - 1.0)
        std[j] = Scalar[DT](0.5 + 0.25 * Float64(j))
    var dn = List[Scalar[DT]](length=A, fill=Scalar[DT](0.0))
    denormalize(out, 0, mean, std, dn, 0, A)
    var dworst = Float64(0.0)
    for j in range(A):
        var back = (Float64(dn[j]) - Float64(mean[j])) / Float64(std[j])
        dworst = max(dworst, abs(back - Float64(out[j])))
    check(
        fails,
        "denormalize inverts to the normalized action",
        dworst < 1e-5,
        "max|diff| = " + String(dworst),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act temporal ensemble gate failed")
