"""G25 — SWM Phase 8c: the P5 comparison, clones from the graph against
clones from EM.

P5 was the design doc's riskiest prediction and the one deferred longest: at
an equal sample budget, SWM-H's structural reading reaches a correct map
before a Clone-Structured Cognitive Graph (George et al. 2021) stabilises its
clones by EM. This gate measures it on the one setting where the two answer
the same question — perceptual aliasing of places — which is CSCG's home
ground, not SWM-H's (parity through a seam is not something a clone HMM
represents, and it is not asked to).

Same input to both: the online label sequence from the learned content
channel (29 symbols, one aliased pair) and the action sequence, on the flat
Klein world with LOCAL aliasing. SWM splits by successor context (G24). CSCG
runs Baum-Welch with a clone budget of 3 per symbol, from several EM seeds,
and is decoded by Viterbi. Both maps are scored the same way: purity against
the true place, merged and split counts, effective clones; and both get the
same frame channel — the transports are re-fitted per clone on each map, so
CSCG's vertices are read with SWM-H's holonomy too.

Two criteria per method, because they come apart: "aliasing resolved" (a pure
map with no merged label) and "exact map" (that, with exactly the true number
of clones and no place split across clones). The frame reading (5 reversing
cycles) is a third line: it needs enough pairs per edge and is the same for
both maps.

Measured (flat Klein, local aliasing, 29 symbols):

  visits   SWM clones   CSCG clones   aliasing resolved      exact map
   121        34            35        both (over-split)      neither
   242        30            30        both                   both
   484        30            39        both                   SWM only
  1936        30            56        both                   SWM only

So P5 as written — SWM before CSCG stabilises its two clones — is NOT
confirmed: the two are level, aliasing resolved at 121 visits (over-split)
and the exact 30-clone map at 242 visits for both. What
separates them is STABILITY: a fixed-budget EM keeps splitting places into
extra pure clones as data grows (purity stays 1.0, the map inflates), while
the context rule is pinned at 30 by construction — a label with one context
is never split. Recorded as measured, gated so it cannot silently reverse.

Gated mechanics (a vacuous baseline would flatter SWM): CSCG's mean
log-likelihood must rise under EM, and at the full budget it must resolve the
aliased symbol with a pure map.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_cscg_comparison.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.klein_grid import (
    KleinWorld,
    KleinWorldConfig,
    ALIAS_LOCAL,
)
from mojo_rl.experimental.swm.map_builder import (
    WalkRecord,
    label_walk,
    count_labels,
    split_until_stable,
    score_map,
    clone_graph,
    count_reversing,
)
from mojo_rl.experimental.swm.cscg import Cscg, dense_labels

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TrainerT = SwmPhase3[NP, 6, 16, 32, 8, DT]
comptime WorldT = KleinWorld[W, H, 6, 16]
comptime EPISODES = 16
comptime STEPS = 4 * NP
comptime LABEL_THRESHOLD = 2.0
comptime CLONES = 3
comptime EM_ITERS = 60
comptime EM_SEEDS = 3


def reversing_of(rec: WalkRecord, labels: List[Int]) -> Int:
    try:
        var g = clone_graph(rec, labels, 2)
        return count_reversing(g)
    except:
        return -1


def main() raises:
    var checks = 0
    var wcfg = KleinWorldConfig.flat_klein().with_alias(ALIAS_LOCAL)
    var cfg = Phase3Config.with_content()
    cfg.seed = 31000
    var env = WorldT(wcfg)
    var m = TrainerT.train_world(env, cfg, NP)
    var rec = TrainerT.record_walks(env, m, EPISODES, STEPS, 41000)

    var sizes: List[Int] = [121, 242, 484, 968, 1936]
    var first_swm_res = -1
    var first_swm_exact = -1
    var first_cscg_res = -1
    var first_cscg_exact = -1
    var first_frame = -1
    var ll_first = Float64(0)
    var ll_last = Float64(0)
    var swm_full_clones = 0
    var cscg_full_clones = 0
    var cscg_full_purity = Float64(0)
    var cscg_full_merged = 1
    print("visits | SWM clones purity merged split reversing | CSCG(best of "
          + String(EM_SEEDS) + ") clones purity merged split reversing  LL")
    for si in range(len(sizes)):
        var sub = rec.prefix(sizes[si])
        var labels = label_walk(sub, LABEL_THRESHOLD)
        var n_sym = count_labels(labels)
        # ---- SWM: split by context
        var clones = split_until_stable(labels, sub.action, 2)
        var s_sc = score_map(clones, sub.true_place, NP)
        var s_rev = reversing_of(sub, clones)
        var s_res = s_sc.n_merged == 0 and s_sc.purity > 0.97
        var s_exact = s_res and s_sc.n_labels == NP and s_sc.n_true_split == 0
        if first_swm_res < 0 and s_res:
            first_swm_res = sizes[si]
        if first_swm_exact < 0 and s_exact:
            first_swm_exact = sizes[si]
        if first_frame < 0 and s_exact and s_rev == 5:
            first_frame = sizes[si]
        # ---- CSCG: EM, best log-likelihood over seeds
        var best_ll = -1e300
        var best_dec = List[Int]()
        for es in range(EM_SEEDS):
            var model = Cscg(n_sym, CLONES, 2, UInt64(777 + es * 101))
            var ll0 = model.em(labels, sub.action, 1)
            var ll = model.em(labels, sub.action, EM_ITERS)
            if si == len(sizes) - 1 and es == 0:
                ll_first = ll0
                ll_last = ll
            if ll > best_ll:
                best_ll = ll
                best_dec = dense_labels(model.decode(labels, sub.action))
        var c_sc = score_map(best_dec, sub.true_place, NP)
        var c_rev = reversing_of(sub, best_dec)
        var c_res = c_sc.n_merged == 0 and c_sc.purity > 0.97
        var c_exact = c_res and c_sc.n_labels == NP and c_sc.n_true_split == 0
        if first_cscg_res < 0 and c_res:
            first_cscg_res = sizes[si]
        if first_cscg_exact < 0 and c_exact:
            first_cscg_exact = sizes[si]
        if si == len(sizes) - 1:
            swm_full_clones = s_sc.n_labels
            cscg_full_clones = c_sc.n_labels
            cscg_full_purity = c_sc.purity
            cscg_full_merged = c_sc.n_merged
        print(sizes[si], "|", s_sc.n_labels, s_sc.purity, s_sc.n_merged,
              s_sc.n_true_split, s_rev, "|", c_sc.n_labels, c_sc.purity,
              c_sc.n_merged, c_sc.n_true_split, c_rev, " ", best_ll)
    print("aliasing resolved : SWM at", first_swm_res, " CSCG at", first_cscg_res, " visits")
    print("exact map (30)    : SWM at", first_swm_exact, " CSCG at", first_cscg_exact, " visits")
    print("frame reads 5 reversing on the exact SWM map at", first_frame, "visits")
    print("CSCG EM at full budget, seed 0: mean LL/visit", ll_first, "->", ll_last)
    if first_cscg_res > 0 and first_swm_res < first_cscg_res:
        print("P5 verdict: CONFIRMED on aliasing resolution")
    elif first_cscg_res > 0 and first_swm_res == first_cscg_res:
        print("P5 verdict: NOT confirmed — level on aliasing resolution; SWM wins on clone-count stability only")
    else:
        print("P5 verdict: NOT confirmed — CSCG resolves the aliasing first")

    checks += 5
    assert_true(
        ll_last > ll_first + 0.05,
        "CSCG's EM must actually learn (log-likelihood up): " + String(ll_first)
        + " -> " + String(ll_last),
    )
    assert_true(
        cscg_full_merged == 0 and cscg_full_purity > 0.95,
        "at the full budget CSCG must resolve the aliasing (else the baseline "
        + "is broken, not beaten): clones " + String(cscg_full_clones)
        + " purity " + String(cscg_full_purity) + " merged "
        + String(cscg_full_merged),
    )
    assert_true(
        first_swm_exact > 0 and first_swm_exact <= 484,
        "SWM must reach the exact map within 484 visits, got " + String(first_swm_exact),
    )
    assert_true(
        first_swm_res == first_cscg_res,
        "RECORDED: P5 is NOT confirmed — both resolve the aliasing at the same "
        + "budget. If this changes, the docstring and README are stale: SWM "
        + String(first_swm_res) + " vs CSCG " + String(first_cscg_res),
    )
    assert_true(
        swm_full_clones == NP and cscg_full_clones > NP + 10,
        "RECORDED: the separation is STABILITY — fixed-budget EM fragments "
        + "places into extra pure clones as data grows while the context rule "
        + "stays at the true count: SWM " + String(swm_full_clones) + " CSCG "
        + String(cscg_full_clones) + " at " + String(sizes[len(sizes) - 1]),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G25 the P5 comparison is measured")
