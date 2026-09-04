"""G17 — SWM Phase 6c: the last oracle, and what it costs to remove it.

Every result from Phase 3 onward assumed an ORACLE place identity. This removes
it, and the answer is not simply "it works".

**The design doc's predicted failure, isolated and confirmed.** §4.1 says that
after an odd lap the same place is seen MIRRORED, so a naive similarity on the
whole encoding misses exactly the revisits that create the informative cycle.
Holding the landmark direction fixed so that mirroring is the ONLY thing that
changes:

    naive full-latent   parity 0: 360/360      parity 1: 190/360
    content only        parity 0: 360/360      parity 1: 360/360

The content channel supplies the frame-invariance the doc asks for, and it does
so for a structural reason: under the reflection `u -> H u`, and the only O(2)
invariant of a vector is its norm, which is near-constant here. The frame simply
cannot carry frame-invariant place information.

**Which makes E1 too easy unless the textures alias.** With one distinctive
texture per cell, content-based recognition is correct by construction. Under
`aliased_mobius(2)` — cells `c` and `c + 6` perceptually identical — it produces
291 wrong matches out of 480.

**And the false matches are not caught by the holonomy.** A false identification
spanning the same number of edges as a true one produces the SAME reading:
measured, 182/182 true and 134/134 false identifications both give `det H = -1`.
The holonomy depends on the graph span, not on whether the two places genuinely
match, so `det H` inherits the reliability of place recognition WHOLESALE. The
doc's §7 says a false identification "creates an aberrant edge (handled)"; that
is too optimistic — under perceptual aliasing it manufactures an obstruction
indistinguishable from a real one.

**The defence is PCM, with a correction.** The textbook criterion asks whether
composing two closures returns the IDENTITY, which assumes a global frame
exists. Here it does not, and the composition of two genuine closures at
different parities is the REFLECTION. Asking instead whether the composition
lies in the world's HOLONOMY GROUP lifts true-pair consistency from 44% to 89%
while false pairs stay low (3% -> 11%).

Run:
    pixi run mojo run -I . tests/experimental/swm/test_place_recognition.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import (
    MobiusRing,
    MobiusConfig,
    ACTION_FORWARD,
)
from mojo_rl.experimental.swm.place_recognition import (
    PlaceMemory,
    score_recogniser,
    MATCH_NONE,
)
from mojo_rl.experimental.swm.observables import pairwise_consistent_in_group

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]
comptime THRESH = 2.0


def enc(mut env: EnvT, m: TrainerT.ModelT) raises -> List[Scalar[DT]]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    m.enc.forward(o, hid, lat)
    return lat^


def main() raises:
    var checks = 0
    var cfg = Phase3Config.with_content()
    cfg.seed = 20260904

    # =====================================================================
    # 1. The doc's mechanism, isolated: same landmark, so mirroring is the
    #    only difference between a parity-0 and a parity-1 revisit.
    # =====================================================================
    var ecfg = MobiusConfig.default_mobius()
    var model = TrainerT.train(ecfg, cfg)
    var res = List[Int](length=8, fill=0)  # [c0,n0,c1,n1] x 2 modes
    for mode in range(2):
        for ep in range(20):
            var mem = PlaceMemory[10, 2, DT]()
            var env = EnvT(ecfg)
            env.reset(UInt64(3000 + ep))
            for _ in range(N):
                mem.add(enc(env, model), env.place_id(), env.lap_parity())
                env.step(ACTION_FORWARD)
            var q = List[Scalar[DT]]()
            var qp = List[Int]()
            var qr = List[Int]()
            for _ in range(2 * N):
                var e = enc(env, model)
                for i in range(10):
                    q.append(e[i])
                qp.append(env.place_id())
                qr.append(env.lap_parity())
                env.step(ACTION_FORWARD)
            var st = score_recogniser[10, 2, DT](
                mem, q, qp, qr, THRESH, mode == 1
            )
            res[mode * 4 + 0] += st.correct_parity_0
            res[mode * 4 + 1] += st.n_parity_0
            res[mode * 4 + 2] += st.correct_parity_1
            res[mode * 4 + 3] += st.n_parity_1
    var naive_p0 = Float64(res[0]) / Float64(res[1])
    var naive_p1 = Float64(res[2]) / Float64(res[3])
    var cont_p0 = Float64(res[4]) / Float64(res[5])
    var cont_p1 = Float64(res[6]) / Float64(res[7])
    print("within-episode | naive full-latent  parity0", res[0], "/", res[1],
          " parity1", res[2], "/", res[3])
    print("               | content only       parity0", res[4], "/", res[5],
          " parity1", res[6], "/", res[7])
    checks += 4
    assert_true(
        naive_p0 > 0.95,
        "the naive recogniser must work on SAME-parity revisits, else the "
        + "parity-1 failure below is not about mirroring: " + String(naive_p0),
    )
    assert_true(
        naive_p1 < 0.75,
        "the design doc's predicted failure: a whole-latent similarity must "
        + "MISS mirrored revisits. got " + String(naive_p1),
    )
    assert_true(
        cont_p1 > 0.95,
        "the content channel must be frame-invariant, got " + String(cont_p1),
    )
    assert_true(
        abs(cont_p0 - cont_p1) < 0.05,
        "frame-invariance means the two parities must score the SAME: "
        + String(cont_p0) + " vs " + String(cont_p1),
    )

    # =====================================================================
    # 2. Under texture aliasing the content channel produces FALSE matches,
    #    and the holonomy cannot tell them from true ones.
    # =====================================================================
    var acfg = MobiusConfig.aliased_mobius(2)
    var amodel = TrainerT.train(acfg, cfg)
    var trs = List[SqMat[2, DT]]()
    for i in range(N):
        trs.append(amodel.table.transport_for(ACTION_FORWARD, i))
    var tr = List[SqMat[2, DT]]()
    tr.append(SqMat[2, DT].identity())
    for k in range(3 * N):
        tr.append(trs[k % N] * tr[k])

    var true_pos = 0
    var false_pos = 0
    var true_neg_det = 0
    var false_neg_det = 0
    var s_list = List[Int]()
    var t_list = List[Int]()
    var ok_list = List[Bool]()
    for ep in range(20):
        var mem = PlaceMemory[10, 2, DT]()
        var env = EnvT(acfg)
        env.reset(UInt64(3000 + ep))
        var ms = List[Int]()
        for t in range(N):
            mem.add(enc(env, amodel), env.place_id(), env.lap_parity())
            ms.append(t)
            env.step(ACTION_FORWARD)
        for t in range(N, 3 * N):
            var e = enc(env, amodel)
            var idx = mem.query(e, THRESH, True)
            if idx != MATCH_NONE:
                var h = tr[t].transpose() * tr[ms[idx]]
                var correct = mem.truth_place[idx] == env.place_id()
                s_list.append(ms[idx])
                t_list.append(t)
                ok_list.append(correct)
                if correct:
                    true_pos += 1
                    if Float64(h.det()) < 0:
                        true_neg_det += 1
                else:
                    false_pos += 1
                    if Float64(h.det()) < 0:
                        false_neg_det += 1
            env.step(ACTION_FORWARD)
    print("aliased | true identifications", true_pos, "of which det H = -1:",
          true_neg_det)
    print("        | FALSE identifications", false_pos,
          "of which det H = -1:", false_neg_det)
    checks += 2
    assert_true(
        false_pos > 20,
        "texture aliasing must actually produce false identifications, else "
        + "the rest of this gate is vacuous. got " + String(false_pos),
    )
    assert_true(
        false_neg_det > 0,
        "THE FINDING: a false identification must be shown to manufacture a "
        + "det H = -1 reading. If it never did, the doc's claim that a false "
        + "identification is 'handled' as an aberrant edge would stand",
    )

    # =====================================================================
    # 3. PCM defends — but the textbook criterion must be made group-aware.
    # =====================================================================
    var group = List[SqMat[2, DT]]()
    group.append(SqMat[2, DT].identity())
    var mono = tr[N].copy()
    group.append(mono^)
    var only_i = List[SqMat[2, DT]]()
    only_i.append(SqMat[2, DT].identity())

    var tt_g = 0
    var tt_n = 0
    var tf_g = 0
    var tf_n = 0
    var tt_i = 0
    var tf_i = 0
    for a in range(len(s_list)):
        for b in range(a + 1, len(s_list)):
            var c = (tr[t_list[a]].transpose() * tr[t_list[b]]) * (
                tr[s_list[b]].transpose() * tr[s_list[a]]
            )
            var same = ok_list[a] and ok_list[b]
            var mixed = ok_list[a] != ok_list[b]
            if not (same or mixed):
                continue
            var g_ok = pairwise_consistent_in_group[2, DT](c, group, 0.3)
            var i_ok = pairwise_consistent_in_group[2, DT](c, only_i, 0.3)
            if same:
                tt_n += 1
                if g_ok:
                    tt_g += 1
                if i_ok:
                    tt_i += 1
            else:
                tf_n += 1
                if g_ok:
                    tf_g += 1
                if i_ok:
                    tf_i += 1
    print("PCM  textbook (composition = I) : true-true", tt_i, "/", tt_n,
          "  true-false", tf_i, "/", tf_n)
    print("PCM  group-aware (in {I, H})    : true-true", tt_g, "/", tt_n,
          "  true-false", tf_g, "/", tf_n)
    var tt_rate_g = Float64(tt_g) / Float64(tt_n)
    var tf_rate_g = Float64(tf_g) / Float64(tf_n)
    var tt_rate_i = Float64(tt_i) / Float64(tt_n)
    checks += 3
    assert_true(
        tt_rate_g > 3.0 * tf_rate_g,
        "consistency must discriminate true pairs from mixed ones: "
        + String(tt_rate_g) + " vs " + String(tf_rate_g),
    )
    assert_true(
        tt_rate_g > tt_rate_i + 0.2,
        "THE CORRECTION: the group-aware criterion must accept far more TRUE "
        + "closures than the textbook 'composition = I', which assumes a global "
        + "frame exists and therefore rejects genuine closures at opposite "
        + "parities. got " + String(tt_rate_g) + " vs " + String(tt_rate_i),
    )
    assert_true(
        tt_rate_i < 0.75,
        "...and the textbook criterion must be SHOWN to reject good closures, "
        + "else there is nothing to correct. got " + String(tt_rate_i),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G17 place recognition needs the content channel, and det H "
          + "inherits its reliability")
