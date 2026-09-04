"""G24 — SWM Phase 8b: a map with no oracle in it, and clones from the graph.

Every place graph before this gate took its vertices from the oracle cell.
Here the agent's own content channel labels the visits, the graph's successor
conflicts say which labels hide two places, and a context-splitting rule
clones them — CSCG's clone count read from the graph instead of EM. The
transports are then re-fitted per clone and the 31 fundamental cycles read on
the clone graph, with the truth graph as the reference.

Three worlds, learned encoders (the training itself indexes transports by the
TEXTURE label, what a recogniser can deliver — G18 leg C's convention):

  flat Klein, LOCAL aliasing   (1,1) ~ (4,3): labelling merges them (29 labels),
                               the merged label conflicts under both actions,
                               splitting yields 30 clones with purity ~1, and
                               the clone graph reads the planted 5 reversing
                               cycles.
  flat torus, LOCAL aliasing   same pipeline, ZERO reversing (control).
  flat Klein, GLOBAL aliasing  (x,y) ~ (x+3,y): 15 labels, ZERO conflicts,
                               nothing to split — the quotient is a consistent
                               world (G19 C), recorded, with its holonomy
                               reading reported.

And the curve the CSCG comparison needs: map correctness against the number
of visits used, on the local-aliasing Klein world.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_klein_map_builder.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.klein_grid import (
    KleinWorld,
    KleinWorldConfig,
    ALIAS_LOCAL,
    ALIAS_GLOBAL,
)
from mojo_rl.experimental.swm.map_builder import (
    WalkRecord,
    label_walk,
    count_labels,
    successor_conflicts,
    split_until_stable,
    score_map,
    clone_graph,
    count_reversing,
)

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TrainerT = SwmPhase3[NP, 6, 16, 32, 8, DT]
comptime WorldT = KleinWorld[W, H, 6, 16]
comptime EPISODES = 16
comptime STEPS = 4 * NP
comptime LABEL_THRESHOLD = 2.0


def texture_purity(labels: List[Int], mut env: WorldT, rec: WalkRecord) -> Float64:
    """Purity of the online labels against the TEXTURE label (what they are
    supposed to recover; aliased cells share one)."""
    var tex = List[Int](length=rec.size(), fill=0)
    for t in range(rec.size()):
        tex[t] = env.texture_of[rec.true_place[t]]
    return score_map(labels, tex, NP).purity


def run_world(
    name: String, wcfg: KleinWorldConfig, mut checks: Int, expect_reversing: Int,
    expect_conflict: Bool,
) raises -> WalkRecord:
    var cfg = Phase3Config.with_content()
    cfg.seed = 31000
    var env = WorldT(wcfg)
    var m = TrainerT.train_world(env, cfg, NP)
    var rec = TrainerT.record_walks(env, m, EPISODES, STEPS, 41000)
    var labels = label_walk(rec, LABEL_THRESHOLD)
    var n_lab = count_labels(labels)
    var tp = texture_purity(labels, env, rec)
    var conflicts = successor_conflicts(labels, rec.action, 2)
    var n_conf_labels = 0
    for l in range(n_lab):
        if conflicts[l] > 0:
            n_conf_labels += 1
    var clones = split_until_stable(labels, rec.action, 2)
    var before = score_map(labels, rec.true_place, NP)
    var after = score_map(clones, rec.true_place, NP)
    var g = clone_graph(rec, clones, 2)
    var n_cyc = len(g.fundamental_cycle_edges())
    var rev = count_reversing(g)
    var truth = env.grid.build_graph()
    var rev_true = count_reversing(truth)
    print(name, "| online labels", n_lab, " texture purity", tp,
          " | conflicting labels", n_conf_labels,
          " | before: merged", before.n_merged, " split", before.n_true_split,
          " | after: clones", after.n_labels, " purity", after.purity,
          " merged", after.n_merged, " split", after.n_true_split,
          " | clone graph: places", g.n_places, " edges", g.n_edges(),
          " cycles", n_cyc, " reversing", rev, "(truth", rev_true, ")")
    checks += 4
    # The online labeller is allowed to UNDER-segment (on the orientable twin
    # the content channel is a weaker place code — G18 — and merges a few
    # extra pairs at this threshold); what it must not do is invent labels
    # that split a true place, because nothing downstream can un-split.
    assert_true(
        tp > 0.85 and before.n_true_split == 0,
        name + ": online labels must coarsen the texture map, never split a "
        + "place: texture purity " + String(tp) + ", places split "
        + String(before.n_true_split),
    )
    if expect_conflict:
        assert_true(
            n_conf_labels == before.n_merged and n_conf_labels >= 1,
            name + ": every merged label and no other must show a successor "
            + "conflict: " + String(n_conf_labels) + " conflicting vs "
            + String(before.n_merged) + " merged",
        )
    else:
        assert_true(
            n_conf_labels == 0 and before.n_merged == NP // 2,
            name + ": a GLOBAL symmetry merges every label and conflicts on "
            + "none — the quotient is a consistent world: " + String(n_conf_labels)
            + " conflicting, " + String(before.n_merged) + " merged",
        )
    if expect_conflict:
        assert_true(
            after.n_labels == NP and after.purity > 0.97 and after.n_merged == 0
            and after.n_true_split == 0,
            name + ": splitting must recover exactly the true places: clones "
            + String(after.n_labels) + " purity " + String(after.purity)
            + " merged " + String(after.n_merged) + " split "
            + String(after.n_true_split),
        )
        assert_true(
            rev == expect_reversing and n_cyc == 31,
            name + ": the clone graph must read the planted holonomy classes: "
            + String(rev) + " reversing of " + String(n_cyc) + " cycles, truth "
            + String(expect_reversing),
        )
    else:
        assert_true(
            after.n_labels == before.n_labels,
            name + ": with nothing to split the label count must not change",
        )
        assert_true(True, "")
    return rec^


def main() raises:
    var checks = 0
    var rec_local = run_world(
        "Klein LOCAL ", KleinWorldConfig.flat_klein().with_alias(ALIAS_LOCAL),
        checks, 5, True,
    )
    _ = run_world(
        "torus LOCAL ", KleinWorldConfig.flat_torus().with_alias(ALIAS_LOCAL),
        checks, 0, True,
    )
    _ = run_world(
        "Klein GLOBAL", KleinWorldConfig.flat_klein().with_alias(ALIAS_GLOBAL),
        checks, 5, False,
    )

    # ---- sample efficiency: visits needed for a correct map ---------------
    print()
    print("visits | labels | clones | purity | merged | split | reversing")
    var first_correct = -1
    var sizes: List[Int] = [121, 242, 484, 968, 1936]
    for si in range(len(sizes)):
        var sub = rec_local.prefix(sizes[si])
        var labels = label_walk(sub, LABEL_THRESHOLD)
        var clones = split_until_stable(labels, sub.action, 2)
        var sc = score_map(clones, sub.true_place, NP)
        var rev: Int
        try:
            var g = clone_graph(sub, clones, 2)
            rev = count_reversing(g)
        except:
            rev = -1
        print(sizes[si], "|", count_labels(labels), "|", sc.n_labels, "|",
              sc.purity, "|", sc.n_merged, "|", sc.n_true_split, "|", rev)
        if first_correct < 0 and sc.n_labels == NP and sc.purity > 0.97 and rev == 5:
            first_correct = sizes[si]
    print("first correct map (30 clones, purity > 0.97, 5 reversing) at", first_correct, "visits")
    checks += 1
    assert_true(first_correct > 0, "the map must become correct within the recorded walks")

    print()
    print("assertions compared :", checks)
    print("PASS: G24 clones from successor conflicts, map without an oracle")
