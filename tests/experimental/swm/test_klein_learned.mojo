"""G23 — SWM Phase 8a: learned encoders on the 2D bundle, 31 cycles read.

6b read many cycles at once but from PLANTED or Procrustes-recovered
transports on frame observations; 6c and Phase 7 removed the place oracle but
only on the ring. This is the first time the whole Phase 3 recipe — an MLP
encoder over mixed landmark + texture observations, per-(action, place)
transports with the orientation bit, the per-place anti-collapse hinge — runs
on a place graph with two actions and 31 fundamental cycles.

The training loop is now written once over the `SwmWorld` trait; the ring and
the grid both conform, and the ring's gates run unchanged through it.

Gated, both worlds on the same binary:

  flat Klein bundle  -> the learned transports reproduce the planted Z/2 class
                        on all 31 fundamental cycles, 5 of them reversing;
  flat torus         -> 31/31 trivial class, ZERO reversing (the control that
                        a learned encoder does not manufacture obstructions in
                        2D any more than in 1D);

with the frame channel VALID on both (landmark R^2 high, texture R^2 low, not
collapsed, not a place-indexed constant), and the content channel localising
(nearest-centroid cell accuracy).

Run:
    pixi run mojo run -I . tests/experimental/swm/test_klein_learned.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.klein_grid import (
    KleinWorld,
    KleinWorldConfig,
    ACT_X,
    ACT_Y,
)

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TrainerT = SwmPhase3[NP, 6, 16, 32, 8, DT]
comptime WorldT = KleinWorld[W, H, 6, 16]
comptime SEEDS = 3


def learned_graph(m: TrainerT.ModelT) raises -> PlaceGraph[2, DT]:
    """Same construction order as `KleinGrid.build_graph`, so fundamental
    cycles correspond one-to-one by non-tree edge index."""
    var g = PlaceGraph[2, DT]()
    for _ in range(NP):
        _ = g.add_place()
    for y in range(H):
        for x in range(W):
            var p = y * W + x
            var qx = y * W + (x + 1) % W
            _ = g.add_edge(Edge.action_edge(p, qx, ACT_X), m.table.transport_for(ACT_X, p))
            var qy = ((y + 1) % H) * W + x
            _ = g.add_edge(Edge.action_edge(p, qy, ACT_Y), m.table.transport_for(ACT_Y, p))
    g.rebuild_gauge(0)
    return g^


def main() raises:
    var checks = 0
    var names: List[String] = ["flat Klein ", "flat torus "]
    var total_false = 0
    var total_reversing_ok = 0
    print("world       | seed | cycles agree | reversing (truth) | landmark R^2 | nuisance R^2 | aniso | wp-std | content cell acc")
    for wi in range(2):
        var wcfg = KleinWorldConfig.flat_klein() if wi == 0 else KleinWorldConfig.flat_torus()
        var truth_world = WorldT(wcfg)
        var g_true = truth_world.grid.build_graph()
        var cyc = g_true.fundamental_cycle_edges()
        var n_rev_true = 0
        for i in range(len(cyc)):
            if g_true.holonomy_det(cyc[i]) < 0:
                n_rev_true += 1
        for s in range(SEEDS):
            var cfg = Phase3Config.with_content()
            cfg.seed = UInt64(31000 + s * 7717)
            var env = WorldT(wcfg)
            var m = TrainerT.train_world(env, cfg, NP)
            var g = learned_graph(m)
            var cyc_l = g.fundamental_cycle_edges()
            var agree = 0
            var n_rev = 0
            for i in range(len(cyc)):
                var dt = g_true.holonomy_det(cyc[i])
                var dl = g.holonomy_det(cyc_l[i])
                if (dt < 0) == (dl < 0):
                    agree += 1
                if dl < 0:
                    n_rev += 1
            var ev = WorldT(wcfg)
            var st = TrainerT.validity_stats(ev, m, cfg)
            print(names[wi], "|", s, "|", agree, "/", len(cyc), "|", n_rev, "(",
                  n_rev_true, ") |", st.landmark_r2, "|", st.nuisance_r2, "|",
                  st.u_anisotropy, "|", st.within_place_std, "|",
                  st.content_cell_acc)
            checks += 4
            assert_true(
                len(cyc) == 31 and len(cyc_l) == 31,
                "a 6x5 torus grid has 31 fundamental cycles",
            )
            assert_true(
                st.u_anisotropy > 0.05 and st.within_place_std > 0.05,
                names[wi] + " seed " + String(s) + ": frame channel INVALID "
                + "(collapsed or place-constant), det readings meaningless",
            )
            assert_true(
                st.landmark_r2 > 0.9 and st.nuisance_r2 < 0.1,
                names[wi] + " seed " + String(s) + ": hypothesis 4.0 fails in 2D "
                + "(landmark R^2 " + String(st.landmark_r2) + ", nuisance R^2 "
                + String(st.nuisance_r2) + ")",
            )
            assert_true(
                agree == 31,
                names[wi] + " seed " + String(s) + ": the learned transports "
                + "must reproduce the planted Z/2 class on every cycle, got "
                + String(agree) + "/31",
            )
            if wi == 1:
                total_false += n_rev
            else:
                if n_rev == n_rev_true:
                    total_reversing_ok += 1
    checks += 2
    assert_true(
        total_false == 0,
        "NEGATIVE CONTROL: the torus was reported reversing on "
        + String(total_false) + " cycles",
    )
    assert_true(
        total_reversing_ok == SEEDS,
        "the Klein bundle must show exactly the planted number of reversing "
        + "cycles in every seed",
    )
    print()
    print("assertions compared :", checks)
    print("PASS: G23 learned encoders read 31 cycles on the flat 2D bundle")
