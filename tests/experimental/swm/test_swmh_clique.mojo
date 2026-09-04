"""G18 — SWM Phase 7: which identifications to believe, without an oracle.

G17 (6c) found that under texture aliasing a false identification reads
`det H = -1` as readily as a true one, and offered PCM as a partial defence at
73 % / 11 %. Re-deriving the composition showed the 73 % was a GAUGE ARTEFACT:
in the spanning-tree gauge the tree is flat, so the PCM composition of two
closures is the product of their holonomies, and the corrected criterion
accepts 100 % of true pairs (`closure_pair_composition`). This gate builds on
that in three legs, each with its negative control.

**A. The maximal clique, with the group bootstrapped from the closures.** No
lap length, no true monodromy: every `det = -1` closure is tried as the
candidate reflection `M`, and the largest set within tolerance of `{I, M}`
wins. Gated: the clique holds >= 95 % of the true closures, its `M` matches the
oracle monodromy (checked, never used), the answer does not depend on scan
order, and the identity-only (textbook) clique is visibly smaller. The false
closures that SURVIVE inside the clique are counted by base cell — they are the
gauge-coincident residual a one-action ring cannot resolve, and the reason the
2D gate exists.

**B. The orientable aliased twin — the control 6c did not run.** A false
identification's holonomy is the transport along the walk between its two
endpoints. On an orientable world every transport is a rotation, so NO walk
can produce `det = -1`: a false identification cannot manufacture a reflection
the world does not have. It can only mis-attribute a reflection the world DOES
have to a spurious cycle. That narrows 6c's finding from "`det H` inherits the
recogniser's reliability" to "the GRAPH inherits it; the `Z/2` class of a walk
is a fact about the walk". Gated as a count: false identifications must be
offered (> 20) and zero of them may read `det = -1`.

**C. The oracle leak in the transport table.** Phases 3-6c indexed transports
by the ORACLE cell even while the recogniser was under test, so the transport
for `(forward, place)` never had to serve two cells. With the table indexed by
the texture label — the label a content recogniser can deliver — a merged label
must fit two different rotations with one matrix, and the PRE-CONSENSUS residual
on that entry is the signal the design's §7 promised ("a false identification
creates an aberrant edge"). Measured against the oracle-labelled table as the
nominal floor; gated: every merged entry is ABERRANT, no closure through one is
filed as an OBSTRUCTION, and the oracle-labelled control still reaches
OBSTRUCTION on the odd-lap closures — else the ABERRANT verdicts would be
proving nothing.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_clique.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import (
    MobiusRing,
    MobiusConfig,
    ACTION_FORWARD,
)
from mojo_rl.experimental.swm.place_recognition import PlaceMemory, MATCH_NONE
from mojo_rl.experimental.swm.observables import (
    maximal_clique_z2,
    Z2Clique,
    classify,
    class_name,
    CLASS_OBSTRUCTION,
    CLASS_ABERRANT,
)

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]
comptime THRESH = 2.0
comptime TOL = 0.3
comptime EPISODES = 20


struct Closures(Copyable, Movable):
    """What the recogniser proposed, with the truth kept beside it for scoring."""

    var h: List[SqMat[2, DT]]
    var correct: List[Bool]
    var base_cell: List[Int]
    var r_local: List[Float64]
    """Largest pre-consensus residual (norm) over the walk spanned by the
    closure — the edge-level reading `classify` consumes."""
    var n_true: Int
    var n_false: Int

    def __init__(out self):
        self.h = List[SqMat[2, DT]]()
        self.correct = List[Bool]()
        self.base_cell = List[Int]()
        self.r_local = List[Float64]()
        self.n_true = 0
        self.n_false = 0

    def __init__(out self, *, copy: Self):
        self.h = copy.h.copy()
        self.correct = copy.correct.copy()
        self.base_cell = copy.base_cell.copy()
        self.r_local = copy.r_local.copy()
        self.n_true = copy.n_true
        self.n_false = copy.n_false

    def __init__(out self, *, deinit move: Self):
        self.h = move.h^
        self.correct = move.correct^
        self.base_cell = move.base_cell^
        self.r_local = move.r_local^
        self.n_true = move.n_true
        self.n_false = move.n_false


def enc(mut env: EnvT, m: TrainerT.ModelT) raises -> List[Scalar[DT]]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    m.enc.forward(o, hid, lat)
    return lat^


def entry_residual(m: TrainerT.ModelT, label: Int) -> Float64:
    """Pre-consensus residual NORM of the selected branch at `(forward, label)`."""
    var e = m.table.index(ACTION_FORWARD, label)
    var sq = m.table.res_plus[e] if m.table.sigma[e] > 0 else m.table.res_minus[e]
    return sqrt(sq)


def collect(
    env_cfg: MobiusConfig,
    m: TrainerT.ModelT,
    n_labels: Int,
    modelled_confusion: Bool = False,
) raises -> Closures:
    """Content-only recognition over `EPISODES` episodes of three laps, with the
    transport table read by LABEL (`cell % n_labels`). Holonomies are read in
    root gauge, `T_t^T T_s`, the same reading G17 uses.

    `modelled_confusion` adds, for every query, a second closure to the
    memory entry of the ALIASED partner cell (`cell + N/2`). It exists for leg
    B: on the orientable twin the learned content channel is free to absorb
    frame information (nothing flips there) and disambiguates aliased cells on
    its own — measured, 1 false identification in 480 — so the learned
    recogniser offers nothing for the leg to reject. The leg's claim is about
    the holonomy of the WALK a false closure spans, which does not depend on
    how the closure was proposed, so the confusion is modelled instead."""
    var trs = List[SqMat[2, DT]]()
    var res = List[Float64]()
    for i in range(N):
        trs.append(m.table.transport_for(ACTION_FORWARD, i % n_labels))
        res.append(entry_residual(m, i % n_labels))
    var tr = List[SqMat[2, DT]]()
    tr.append(SqMat[2, DT].identity())
    for k in range(3 * N):
        tr.append(trs[k % N] * tr[k])

    var out = Closures()
    for ep in range(EPISODES):
        var mem = PlaceMemory[10, 2, DT]()
        var env = EnvT(env_cfg)
        env.reset(UInt64(3000 + ep))
        var ms = List[Int]()
        for t in range(N):
            mem.add(enc(env, m), env.place_id(), env.lap_parity())
            ms.append(t)
            env.step(ACTION_FORWARD)
        for t in range(N, 3 * N):
            var e = enc(env, m)
            var idx = mem.query(e, THRESH, True)
            var n_prop = 1
            if modelled_confusion:
                n_prop = 2
            for prop in range(n_prop):
                if prop == 1:
                    idx = (env.place_id() + N // 2) % N  # memory index == cell
                if idx == MATCH_NONE:
                    continue
                var s = ms[idx]
                out.h.append(tr[t].transpose() * tr[s])
                var ok = mem.truth_place[idx] == env.place_id()
                out.correct.append(ok)
                out.base_cell.append(mem.truth_place[idx])
                var worst = Float64(0)
                for k in range(s, t):
                    if res[k % N] > worst:
                        worst = res[k % N]
                out.r_local.append(worst)
                if ok:
                    out.n_true += 1
                else:
                    out.n_false += 1
            env.step(ACTION_FORWARD)
    return out^


def per_cell_residual(
    env_cfg: MobiusConfig, m: TrainerT.ModelT, n_labels: Int
) raises -> List[Float64]:
    """Mean transport residual NORM per TRUE cell, using the transport of the
    cell's LABEL — the splitting signal a label-merging recogniser leaves."""
    var sum = List[Float64](length=N, fill=0)
    var cnt = List[Float64](length=N, fill=0)
    for ep in range(EPISODES):
        var env = EnvT(env_cfg)
        env.reset(UInt64(5000 + ep))
        var prev = enc(env, m)
        for _ in range(3 * N):
            var c = env.place_id()
            env.step(ACTION_FORWARD)
            var cur = enc(env, m)
            var r = m.table.transport_for(ACTION_FORWARD, c % n_labels)
            var sq = Float64(0)
            for i in range(2):
                var pred = Scalar[DT](0)
                for j in range(2):
                    pred += r[i, j] * prev[j]
                var d = Float64(pred - cur[i])
                sq += d * d
            sum[c] += sqrt(sq)
            cnt[c] += 1
            prev = cur^
    for c in range(N):
        if cnt[c] > 0:
            sum[c] /= cnt[c]
    return sum^


def count_negative(c: Closures, only_false: Bool) -> Int:
    var n = 0
    for i in range(len(c.h)):
        if only_false and c.correct[i]:
            continue
        if Float64(c.h[i].det()) < 0:
            n += 1
    return n


def main() raises:
    var checks = 0
    var cfg = Phase3Config.with_content()
    cfg.seed = 20260904

    # =====================================================================
    # A. Mobius, aliased, ORACLE-labelled table (G17's setting): the clique.
    # =====================================================================
    var acfg = MobiusConfig.aliased_mobius(2)
    var amodel = TrainerT.train(acfg, cfg)
    var mob = collect(acfg, amodel, N)
    print("A | Mobius aliased(2): closures", len(mob.h), " true", mob.n_true,
          " false", mob.n_false, " det=-1 among false:",
          count_negative(mob, True))

    var fwd = maximal_clique_z2[2, DT](mob.h, TOL, False)
    var rev = maximal_clique_z2[2, DT](mob.h, TOL, True)
    var true_in = 0
    var false_in = 0
    var false_by_cell = List[Int](length=N, fill=0)
    for i in range(len(fwd.members)):
        var idx = fwd.members[i]
        if mob.correct[idx]:
            true_in += 1
        else:
            false_in += 1
            false_by_cell[mob.base_cell[idx]] += 1
    print("A | clique: members", len(fwd.members), " (identity", fwd.n_identity,
          ", reflection", fwd.n_reflection, ")  true in", true_in, "/",
          mob.n_true, "  false in", false_in, "/", mob.n_false)
    var cells = String("")
    for c in range(N):
        if false_by_cell[c] > 0:
            cells += " cell " + String(c) + ":" + String(false_by_cell[c])
    print("A | false members by base cell:" + cells)

    # The bootstrapped M against the oracle monodromy — a CHECK, never an input.
    var oracle_m = SqMat[2, DT].identity()
    for i in range(N):
        oracle_m = amodel.table.transport_for(ACTION_FORWARD, i) * oracle_m
    var m_err = 2.0
    var seed_err = 2.0
    if fwd.reference >= 0:
        var m_ref = SqMat[2, DT]()
        for i in range(2):
            for j in range(2):
                m_ref[i, j] = Scalar[DT](fwd.refined_reference[i * 2 + j])
        m_err = Float64((m_ref - oracle_m).frobenius_norm())
        seed_err = Float64((mob.h[fwd.reference] - oracle_m).frobenius_norm())
    print("A | bootstrapped M vs oracle lap holonomy: refined |M - T_N|_F =",
          m_err, "  (seed closure", seed_err, ", member spread", fwd.spread, ")")

    checks += 5
    assert_true(
        mob.n_false > 20,
        "aliasing must produce false identifications or the clique has nothing "
        + "to reject. got " + String(mob.n_false),
    )
    assert_true(
        Float64(true_in) >= 0.95 * Float64(mob.n_true),
        "the clique must keep >= 95% of TRUE closures, got " + String(true_in)
        + "/" + String(mob.n_true),
    )
    assert_true(
        fwd.same_members(rev),
        "the clique must not depend on which reflection is examined first",
    )
    assert_true(
        m_err < 0.1 and m_err < seed_err,
        "the bootstrapped reflection must BE the world's monodromy (no lap "
        + "length was used to find it), and refining must move it closer than "
        + "the seed closure: |M - T_N|_F = " + String(m_err) + " vs seed "
        + String(seed_err),
    )
    assert_true(
        Float64(fwd.n_identity) < 0.75 * Float64(mob.n_true),
        "NEGATIVE CONTROL: the identity-only (textbook) clique must be visibly "
        + "smaller than the true closure count, else the group-aware form "
        + "bought nothing. got " + String(fwd.n_identity) + " vs "
        + String(mob.n_true),
    )

    # =====================================================================
    # B. The orientable aliased twin: can a false identification manufacture
    #    a reflection the world does not have?
    # =====================================================================
    var ocfg = MobiusConfig.aliased_orientable(2)
    var omodel = TrainerT.train(ocfg, cfg)
    var ori_learned = collect(ocfg, omodel, N)
    print("B | orientable aliased(2), LEARNED recogniser: closures",
          len(ori_learned.h), " true", ori_learned.n_true, " false",
          ori_learned.n_false, " -- the content channel disambiguates "
          + "aliased cells here (nothing flips), so confusion is MODELLED below")
    var ori = collect(ocfg, omodel, N, True)
    var ori_neg_all = count_negative(ori, False)
    var oclique = maximal_clique_z2[2, DT](ori.h, TOL, False)
    var o_true_in = 0
    for i in range(len(oclique.members)):
        if ori.correct[oclique.members[i]]:
            o_true_in += 1
    print("B | orientable aliased(2), modelled confusion: closures", len(ori.h), " true", ori.n_true,
          " false", ori.n_false, " det=-1 among ALL:", ori_neg_all)
    print("B | clique: members", len(oclique.members), " reference",
          oclique.reference, " true in", o_true_in, "/", ori.n_true)
    checks += 3
    assert_true(
        ori.n_false > 20,
        "the orientable twin must be offered false identifications too, got "
        + String(ori.n_false),
    )
    assert_true(
        ori_neg_all == 0,
        "THE CONTROL 6c LACKED: on an orientable world no walk carries a "
        + "reflection, so a false identification cannot read det = -1. got "
        + String(ori_neg_all),
    )
    assert_true(
        oclique.reference < 0 and Float64(o_true_in) >= 0.95 * Float64(ori.n_true),
        "orientable: the clique must be identity-only and keep the true "
        + "closures",
    )

    # =====================================================================
    # C. Close the oracle leak: transports indexed by the TEXTURE LABEL.
    # =====================================================================
    var lcfg = cfg
    lcfg.place_labels_from_texture = True
    var lmodel = TrainerT.train(acfg, lcfg)
    var n_labels = N // 2
    var lab = collect(acfg, lmodel, n_labels)

    var floor = Float64(0)
    for i in range(N):
        floor += entry_residual(amodel, i)
    floor /= Float64(N)
    var merged_min = 1e300
    var merged_max = Float64(0)
    var n_reflected_entries = 0
    for l in range(n_labels):
        var r = entry_residual(lmodel, l)
        if r < merged_min:
            merged_min = r
        if r > merged_max:
            merged_max = r
        if lmodel.table.sigma[lmodel.table.index(ACTION_FORWARD, l)] < 0:
            n_reflected_entries += 1
    var lap_l = SqMat[2, DT].identity()
    for i in range(N):
        lap_l = lmodel.table.transport_for(ACTION_FORWARD, i % n_labels) * lap_l
    print("C | pre-consensus residual (EWMA per entry): oracle-labelled floor",
          floor, "  merged labels min", merged_min, " max", merged_max,
          "  ratio min", merged_min / floor, " max", merged_max / floor)
    print("C | label table: reflected entries", n_reflected_entries, "/",
          n_labels, "  det of the 12-step lap product", lap_l.det())

    var env_l = EnvT(acfg)
    var steps = cfg.laps * N
    var ev = TrainerT._evaluate(env_l, lmodel.enc, lmodel.table, lcfg, steps, steps + 1)
    var env_o = EnvT(acfg)
    var ev_o = TrainerT._evaluate(env_o, amodel.enc, amodel.table, cfg, steps, steps + 1)
    print("C | frame channel, label-indexed : landmark R^2", ev.landmark_r2,
          " nuisance R^2", ev.nuisance_r2, " anisotropy", ev.u_anisotropy)
    print("C | frame channel, oracle-labelled: landmark R^2", ev_o.landmark_r2,
          " nuisance R^2", ev_o.nuisance_r2, " anisotropy", ev_o.u_anisotropy)

    # The splitting signal: residual per TRUE cell under the merged transport.
    var pc_l = per_cell_residual(acfg, lmodel, n_labels)
    var pc_o = per_cell_residual(acfg, amodel, N)
    var pairs_split = 0
    var worst_pair_ratio = 1e300
    var line = String("")
    for c in range(n_labels):
        var a = pc_l[c]
        var b = pc_l[c + n_labels]
        var hi = a if a > b else b
        var lo = b if a > b else a
        var ratio = hi / (lo + 1e-12)
        if ratio < worst_pair_ratio:
            worst_pair_ratio = ratio
        if ratio > 2.0:
            pairs_split += 1
        line += " (" + String(c) + "," + String(c + n_labels) + "): "
        line += String(Int(a * 1000)) + "/" + String(Int(b * 1000))
    var pc_o_mean = Float64(0)
    for c in range(N):
        pc_o_mean += pc_o[c]
    pc_o_mean /= Float64(N)
    print("C | per-cell residual x1000 under the MERGED transport, by aliased "
          + "pair:" + line)
    print("C | per-cell residual, oracle-labelled mean", pc_o_mean,
          "  pairs with one side > 2x the other:", pairs_split, "/", n_labels,
          " worst pair ratio", worst_pair_ratio)

    var verdicts_l = List[Int](length=5, fill=0)
    for i in range(len(lab.h)):
        var v = classify(
            lab.r_local[i], floor, Float64(lab.h[i].det()),
            Float64(lab.h[i].dist_to_identity()), TOL, False,
        )
        verdicts_l[Int(v)] += 1
    var verdicts_o = List[Int](length=5, fill=0)
    var obstr_true_odd = 0
    for i in range(len(mob.h)):
        var v = classify(
            mob.r_local[i], floor, Float64(mob.h[i].det()),
            Float64(mob.h[i].dist_to_identity()), TOL, False,
        )
        verdicts_o[Int(v)] += 1
        if v == CLASS_OBSTRUCTION and mob.correct[i]:
            obstr_true_odd += 1
    print("C | verdicts, label-indexed  : NOMINAL", verdicts_l[0], "ABERRANT",
          verdicts_l[1], "OBSTRUCTION", verdicts_l[2], "UNDECIDED",
          verdicts_l[3], "  of", len(lab.h), "closures")
    print("C | verdicts, oracle-labelled: NOMINAL", verdicts_o[0], "ABERRANT",
          verdicts_o[1], "OBSTRUCTION", verdicts_o[2], "UNDECIDED",
          verdicts_o[3], "  of", len(mob.h), "closures (true odd-lap "
          + "obstructions " + String(obstr_true_odd) + ")")

    checks += 5
    assert_true(
        merged_min > 1.5 * floor and merged_min < 10.0 * floor,
        "the merged entries must show an ELEVATED residual that stays BELOW "
        + "the 10x ABERRANT threshold — the doc's §7 mechanism half-fires: "
        + "min ratio " + String(merged_min / floor)
        + ". If this ever reaches 10x, the aberrant-edge story holds and the "
        + "docstring is wrong; if it drops to 1x, co-adaptation is complete.",
    )
    assert_true(
        verdicts_l[Int(CLASS_OBSTRUCTION)] == 0 and Float64(lap_l.det()) > 0,
        "with merged labels the seam is UNREPRESENTABLE (one entry must be a "
        + "rotation for one cell and a reflection for its alias), so the "
        + "obstruction must be LOST, not manufactured: obstructions "
        + String(verdicts_l[Int(CLASS_OBSTRUCTION)]) + ", lap det "
        + String(lap_l.det()),
    )
    assert_true(
        verdicts_l[Int(CLASS_ABERRANT)] == 0 and len(lab.h) > 20,
        "...and the diluted residual never reaches ABERRANT either: "
        + String(verdicts_l[Int(CLASS_ABERRANT)]) + "/" + String(len(lab.h)),
    )
    assert_true(
        ev.landmark_r2 < ev_o.landmark_r2 - 0.1,
        "co-adaptation must be VISIBLE as a loss of landmark fidelity in the "
        + "frame channel, else the residual dilution has no explanation: R^2 "
        + String(ev.landmark_r2) + " vs oracle-labelled " + String(ev_o.landmark_r2),
    )
    assert_true(
        obstr_true_odd > 100 and verdicts_o[Int(CLASS_ABERRANT)] == 0,
        "CONTROL: with oracle labels the odd-lap closures must still reach "
        + "OBSTRUCTION and nothing is ABERRANT, else leg C's verdicts prove "
        + "nothing. got obstructions " + String(obstr_true_odd)
        + ", aberrant " + String(verdicts_o[Int(CLASS_ABERRANT)]),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G18 the clique keeps the true closures, an orientable world "
          + "cannot be made to reflect, and merged labels LOSE the seam")
