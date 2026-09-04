"""G4 — SWM Phase 1 gate: per-edge O(2) fit against the pinned numpy oracle.

This reproduces row A of the §1.2 table of docs/SHEAF_WORLD_MODELS_V2.md, and
it is the gate that makes the oracle worth having.

numpy's PCG64 draws cannot be reproduced here, so re-running the same
*generative process* in Mojo could only ever compare statistics. Instead the
oracle carries the exact 1920 observation pairs numpy saw
(`tools/swm/dump_mobius_reference.py`), and this gate demands numpy's transports
back from them. numpy fits by SVD; `procrustes.mojo` fits by Newton polar
decomposition. Two algorithms, one answer — a shared bug cannot make this pass,
which a transcription of the numpy source could not have promised.

Validates:
  - all 12 per-edge transports match the oracle to 1e-9
  - all 12 per-edge residuals match the oracle to 1e-9 relative
  - residuals sit AT the declared noise floor (2 * 0.02^2 = 8e-4) and are
    UNIFORM (max/median ~ 1.14): no edge is locally suspect...
  - ...yet det H = -1 and ||H - I||_F = 2. The obstruction is a property of the
    CYCLE, not of any edge. That contrast is the entire claim of the document.
  - closure error after k = 1..4 laps matches the oracle
  - NEGATIVE CONTROL: the translation ablation (model B) fit from the SAME
    pairs must show the 7-10x odd/even parity gap. Without this leg, a closure
    metric that returned a constant would pass every assertion above.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_procrustes_vs_oracle.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.procrustes import (
    PairBatch,
    procrustes_o_d,
    mean_squared_residual,
)
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.reference_io import (
    load_reference,
    ref_scalar,
    ref_int,
    ref_indexed,
    ref_vector,
    ref_count,
)

comptime DT = DType.float64
comptime D = 2
comptime REF_DIR = "tests/experimental/swm/reference/"


def median_of(values: List[Float64]) -> Float64:
    var v = values.copy()
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var n = len(v)
    if n % 2 == 1:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def main() raises:
    var oracle = load_reference(REF_DIR + "mobius_proto.txt")
    var pair_rows = load_reference(REF_DIR + "mobius_pairs.txt")
    var seq_rows = load_reference(REF_DIR + "mobius_seqs.txt")

    var n_ring = ref_int(oracle, "ring_size")
    var per_edge = ref_int(oracle, "pairs_per_edge")
    var n_loops = ref_int(oracle, "n_loops")
    var n_ep = ref_int(oracle, "n_episodes")
    var noise_floor = ref_scalar(oracle, "noise_floor")
    var checks = 0

    if n_ring != 12:
        raise Error("oracle ring_size changed; this gate assumes 12")

    # ---- vacuity guard: the oracle must actually contain what we claim ------
    var pairs_seen = ref_count(pair_rows, "pair")
    var seqs_seen = ref_count(seq_rows, "seq")
    checks += 2
    assert_true(
        pairs_seen == n_ring * per_edge,
        "expected " + String(n_ring * per_edge) + " pairs, oracle has "
        + String(pairs_seen),
    )
    assert_true(
        seqs_seen == n_ep * (n_loops * n_ring + 1),
        "expected " + String(n_ep * (n_loops * n_ring + 1))
        + " trajectory samples, oracle has " + String(seqs_seen),
    )

    # ---- load the pairs, edge by edge --------------------------------------
    var batches = List[PairBatch[D, DT]]()
    for _ in range(n_ring):
        batches.append(PairBatch[D, DT]())
    for i in range(len(pair_rows)):
        if pair_rows[i].key != "pair":
            continue
        ref nums = pair_rows[i].nums
        var e = Int(nums[0])
        batches[e].xs.append(Scalar[DT](nums[2]))
        batches[e].xs.append(Scalar[DT](nums[3]))
        batches[e].ys.append(Scalar[DT](nums[4]))
        batches[e].ys.append(Scalar[DT](nums[5]))
    for e in range(n_ring):
        checks += 1
        assert_true(
            batches[e].count() == per_edge,
            "edge " + String(e) + " has " + String(batches[e].count())
            + " pairs, expected " + String(per_edge),
        )

    # ---- fit + compare against numpy ---------------------------------------
    var fits = List[SqMat[D, DT]]()
    var residuals = List[Float64]()
    var worst_r = Float64(0)
    var worst_res_rel = Float64(0)
    for e in range(n_ring):
        var r = procrustes_o_d[D, DT](batches[e])
        var want = ref_indexed(oracle, "fit_a_r", e)
        var diff = Float64(0)
        for i in range(D):
            for j in range(D):
                var d = abs(Float64(r[i, j]) - want[i * D + j])
                if d > diff:
                    diff = d
        if diff > worst_r:
            worst_r = diff
        checks += 1
        assert_true(
            diff <= 1e-9,
            "edge " + String(e) + ": Newton polar fit differs from the numpy "
            + "SVD fit by " + String(diff),
        )

        var res = Float64(mean_squared_residual[D, DT](batches[e], r))
        var want_res = ref_indexed(oracle, "fit_a_residual", e)[0]
        var rel = abs(res - want_res) / want_res
        if rel > worst_res_rel:
            worst_res_rel = rel
        checks += 1
        assert_true(
            rel <= 1e-9,
            "edge " + String(e) + ": residual " + String(res) + " vs oracle "
            + String(want_res),
        )
        fits.append(r)
        residuals.append(res)

    # ---- "no edge is suspect": residuals at the floor, and uniform ----------
    var res_med = median_of(residuals)
    var res_max = Float64(0)
    for i in range(len(residuals)):
        if residuals[i] > res_max:
            res_max = residuals[i]
    var ratio = res_max / res_med
    checks += 3
    assert_true(
        res_med <= 1.5 * noise_floor and res_med >= 0.5 * noise_floor,
        "median residual " + String(res_med) + " is not at the noise floor "
        + String(noise_floor),
    )
    assert_true(
        ratio < 2.0,
        "residuals are not uniform: max/median = " + String(ratio),
    )
    assert_true(
        abs(ratio - ref_scalar(oracle, "fit_a_residual_max_over_median")) <= 1e-9,
        "max/median ratio disagrees with the oracle",
    )

    # ---- ...but the cycle is obstructed ------------------------------------
    var g = PlaceGraph[D, DT]()
    for _ in range(n_ring):
        _ = g.add_place()
    for i in range(n_ring):
        _ = g.add_edge(Edge.action_edge(i, (i + 1) % n_ring, 0), fits[i])
    g.rebuild_gauge(0)
    var cyc = g.fundamental_cycle_edges()
    checks += 1
    assert_true(len(cyc) == 1, "ring must have one fundamental cycle")
    var det_h = g.holonomy_det(cyc[0])
    var fro_h = g.holonomy_dist_to_identity(cyc[0])
    checks += 2
    assert_true(
        abs(det_h - ref_scalar(oracle, "fit_a_det_h")) <= 1e-9,
        "det H = " + String(det_h) + " vs oracle "
        + String(ref_scalar(oracle, "fit_a_det_h")),
    )
    assert_true(
        abs(fro_h - ref_scalar(oracle, "fit_a_h_minus_i_fro")) <= 1e-9,
        "||H-I||_F = " + String(fro_h) + " vs oracle "
        + String(ref_scalar(oracle, "fit_a_h_minus_i_fro")),
    )

    # ---- loop-closure error, k = 1..4 --------------------------------------
    var steps = n_loops * n_ring + 1
    var traj = List[Float64](length=n_ep * steps * D, fill=0.0)
    for i in range(len(seq_rows)):
        if seq_rows[i].key != "seq":
            continue
        ref nums = seq_rows[i].nums
        var ep = Int(nums[0])
        var t = Int(nums[1])
        traj[(ep * steps + t) * D + 0] = nums[2]
        traj[(ep * steps + t) * D + 1] = nums[3]

    var lce_a = List[Float64](length=n_loops, fill=0.0)
    var lce_b = List[Float64](length=n_loops, fill=0.0)

    # Model B, fit from the SAME pairs: a single global frame whose transitions
    # are translations. It can LOCATE the seam but not represent it.
    var trans = List[Float64](length=n_ring * D, fill=0.0)
    for e in range(n_ring):
        for k in range(batches[e].count()):
            for c in range(D):
                trans[e * D + c] += Float64(
                    batches[e].ys[k * D + c] - batches[e].xs[k * D + c]
                )
        for c in range(D):
            trans[e * D + c] /= Float64(batches[e].count())

    for ep in range(n_ep):
        var xa = List[Float64](length=D, fill=0.0)
        var xb = List[Float64](length=D, fill=0.0)
        for c in range(D):
            xa[c] = traj[(ep * steps + 0) * D + c]
            xb[c] = xa[c]
        for k in range(1, n_loops + 1):
            for i in range(n_ring):
                var nxt = List[Float64](length=D, fill=0.0)
                for r in range(D):
                    var s = Float64(0)
                    for c in range(D):
                        s += Float64(fits[i][r, c]) * xa[c]
                    nxt[r] = s
                for c in range(D):
                    xa[c] = nxt[c]
                    xb[c] = xb[c] + trans[i * D + c]
            var da = Float64(0)
            var db = Float64(0)
            for c in range(D):
                var ta = xa[c] - traj[(ep * steps + k * n_ring) * D + c]
                var tb = xb[c] - traj[(ep * steps + k * n_ring) * D + c]
                da += ta * ta
                db += tb * tb
            lce_a[k - 1] += sqrt(da)
            lce_b[k - 1] += sqrt(db)
    for k in range(n_loops):
        lce_a[k] /= Float64(n_ep)
        lce_b[k] /= Float64(n_ep)

    for k in range(1, n_loops + 1):
        var want = ref_indexed(oracle, "fit_a_closure_error", k)[0]
        checks += 1
        assert_true(
            abs(lce_a[k - 1] - want) <= 1e-9,
            "closure error k=" + String(k) + ": " + String(lce_a[k - 1])
            + " vs oracle " + String(want),
        )
        var want_b = ref_indexed(oracle, "fit_b_closure_error", k)[0]
        checks += 1
        assert_true(
            abs(lce_b[k - 1] - want_b) <= 1e-9,
            "ablation B closure error k=" + String(k) + ": "
            + String(lce_b[k - 1]) + " vs oracle " + String(want_b),
        )

    # ---- NEGATIVE CONTROL: the metric must be able to discriminate ----------
    # Model B fails by PARITY. If the closure metric were degenerate, model A
    # and model B would look alike and every assertion above would be hollow.
    var odd_b = 0.5 * (lce_b[0] + lce_b[2])
    var even_b = 0.5 * (lce_b[1] + lce_b[3])
    var gap = odd_b / even_b
    checks += 2
    assert_true(
        gap >= 7.0 and gap <= 10.0,
        "NEGATIVE CONTROL FAILED: ablation B odd/even parity gap = "
        + String(gap) + ", expected 7-10x",
    )
    assert_true(
        lce_b[0] / lce_a[0] >= 3.0,
        "NEGATIVE CONTROL FAILED: ablation B is not worse than A at k=1",
    )

    print("edges compared      :", n_ring)
    print("pairs compared      :", pairs_seen, "(", per_edge, "per edge )")
    print("trajectory samples  :", seqs_seen)
    print("worst |R - R_numpy| :", worst_r)
    print("worst residual rel  :", worst_res_rel)
    print("residual median     :", res_med, " noise floor:", noise_floor)
    print("residual max/median :", ratio, "  (no edge is locally suspect)")
    print("det H               :", det_h, "  ||H-I||_F:", fro_h)
    print(
        "closure A k=1..4    :", lce_a[0], lce_a[1], lce_a[2], lce_a[3]
    )
    print(
        "closure B k=1..4    :", lce_b[0], lce_b[1], lce_b[2], lce_b[3],
        " parity gap:", gap,
    )
    print("assertions compared :", checks)
    print("PASS: G4 per-edge O(2) fit reproduces the numpy oracle")
