"""Milestone 1 — FB end to end on `point_mass easy`, offline.

`docs/BFM_ZERO_SHOT_RL.md` §13. This is the run that says whether the
implementation is CORRECT, not whether it is good. Four phases, none of which
involves an environment during training:

    collect   random policy -> HDF5 TrajectoryStore of (qpos, qvel, action)
    train     FB on that store, no env in the loop
    infer     z = E_rho[B(s)·r(s)] with r RELABELLED offline (`reward_at`)
    evaluate  return of pi_z vs the random policy that produced the data

**Why point_mass and not walker.** §12: `nq = 2` is the only domain in the
suite where the successor measure is traceable by hand. On walker a collapsed
`B` and a correct `B` produce the same loss curve, so a walker run cannot tell
you whether the code works — which is the only question milestone 1 asks.

**Why the dataset stores qpos/qvel and not observations.** Because step 3
relabels. `z` is computed from a reward the collection run never saw, which is
only possible because the state is stored in generalized coordinates and
`reward_at` can replay it. That is the whole architectural bet of component 0,
and this script is where it pays off — the collection policy is random and
carries no notion of "reach the target".

Note that for `point_mass` the observation IS `[qpos, qvel]`, so the storage
choice looks free here. It is not free on walker (18 stored floats vs 24
observed, and `xmat`/`subtree_linvel` unrecoverable from the observation); this
domain is the one where the two coincide, which is convenient for a first
validation and misleading as a general lesson.

⚠ **The bar is "beats random", not "solves the task".** point_mass's reward is
~1e-245 more than 10 cm from the target, so a random-policy dataset is mostly
zeros and `z` is determined by a thin slice of it. Beating random on that is
evidence the machinery is wired correctly. It is not evidence FB is trained.

Run:
    pixi run mojo run -I . examples/fb/point_mass_fb_milestone1.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh

from mojo_rl.data.column import ColumnSpec
from mojo_rl.data.store import TrajectoryStore, TrajectoryStoreWriter
from mojo_rl.data.resident import ResidentColumn
from mojo_rl.data.sampler import UniformSampler

from mojo_rl.envs.dm_control.point_mass import DMPointMassEasy

from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb import sample_z, z_from_reward


comptime NQ: Int = 2
comptime NV: Int = 2
comptime NACT: Int = 2
comptime OBS: Int = NQ + NV          # point_mass obs = [qpos, qvel]
comptime D: Int = 50                 # §12: 50 at milestone 1, 128 later
comptime BATCH: Int = 64
comptime HID: Int = 256

comptime N_EPISODES: Int = 60
comptime EP_LEN: Int = 200
comptime TRAIN_STEPS: Int = 40000    # §13 budgets 500k; the loss plateaus by ~16k here
comptime RELABEL_ROWS: Int = 2000
comptime EVAL_EPISODES: Int = 20
comptime EVAL_LEN: Int = 200
comptime SEED: Int = 20260805

comptime STORE_PATH: StaticString = (
    "/tmp/mojo_rl_fb_point_mass_milestone1.h5"
)

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
# ⚠⚠ NO normalisation on B here, and the M2 walker run DIVERGED without one
# (`L_ortho` positive and growing 8x, `|B|` climbing, measure loss unbounded
# below). Meta Motivo sets `"b": {"norm": true}` and §6's table names
# `layer_norm.mojo`; `examples/fb/fb_train_gpu.mojo` now ends `BNet` in a
# `LayerNorm[D]`.
#
# It is left OFF here on purpose: this script's recorded M1 result (pi_z 0.617
# vs random 2e-11) was produced by this architecture, and its `L_ortho` stayed
# NEGATIVE and `|B|` stable at ~1.9 for 40 k steps — point_mass at d=50 does
# not hit the failure. Changing it silently would invalidate a recorded number
# without re-running it. Add the LayerNorm and re-measure before trusting a
# comparison against the M2 configuration.
comptime BNet = Sequential[Linear[OBS, HID], ReLU[HID], Linear[HID, D]]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, NACT, D, BATCH]
comptime Env = DMPointMassEasy[DType.float64]


# ══════════════════════════════════════════════════════════════════════
# 1. Collect
# ══════════════════════════════════════════════════════════════════════


def collect() raises -> Int:
    """Random policy into a TrajectoryStore. Returns the row count.

    Stores the state AFTER each step together with the action that produced
    it — the same convention `test_reward_relabel.mojo` gates, so the rows can
    be scored by `reward_at(qpos, qvel, action)` with no further bookkeeping.
    """
    print("[1] collecting", N_EPISODES, "x", EP_LEN, "steps ...")
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, NV))
    cols.append(ColumnSpec(String("action"), DType.float32, NACT))

    var w = TrajectoryStoreWriter(
        String(STORE_PATH), cols^, env_id=String("dm_control/point_mass-easy"),
        seed=SEED,
    )
    var env = Env()
    seed(SEED)

    var qbuf = List[Float32](length=NQ, fill=Float32(0))
    var vbuf = List[Float32](length=NV, fill=Float32(0))
    var abuf = List[Float32](length=NACT, fill=Float32(0))

    var rows = 0
    for _ep in range(N_EPISODES):
        _ = env.reset()
        for _t in range(EP_LEN):
            var act = Env.ActionType()
            for k in range(NACT):
                var v = random_float64() * 2.0 - 1.0
                act.data[k] = v
                abuf[k] = Float32(v)
            _ = env.step(act)
            for i in range(NQ):
                qbuf[i] = Float32(Float64(env.d.qpos.data[i]))
            for i in range(NV):
                vbuf[i] = Float32(Float64(env.d.qvel.data[i]))
            w.append[DType.float32](
                String("qpos"), qbuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            w.append[DType.float32](
                String("qvel"), vbuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            w.append[DType.float32](
                String("action"), abuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            rows += 1
        w.end_episode()
    w.close()
    print("      ", rows, "rows ->", STORE_PATH)
    return rows


# ══════════════════════════════════════════════════════════════════════
# 2. Train
# ══════════════════════════════════════════════════════════════════════


def _gather_obs(
    ref qpos: ResidentColumn[DType.float32],
    ref qvel: ResidentColumn[DType.float32],
    ref idx: List[Int],
    mut dst: Tensor,
) raises:
    """`[BATCH, OBS]` from row indices — obs = [qpos | qvel]."""
    dst.ensure(len(idx) * OBS)
    for b in range(len(idx)):
        var r = idx[b]
        for k in range(NQ):
            dst.data[b * OBS + k] = Scalar[DT](Float64(qpos.host[r * NQ + k]))
        for k in range(NV):
            dst.data[b * OBS + NQ + k] = Scalar[DT](
                Float64(qvel.host[r * NV + k])
            )


def _gather_act(
    ref action: ResidentColumn[DType.float32],
    ref idx: List[Int],
    mut dst: Tensor,
) raises:
    dst.ensure(len(idx) * NACT)
    for b in range(len(idx)):
        var r = idx[b]
        for k in range(NACT):
            dst.data[b * NACT + k] = Scalar[DT](
                Float64(action.host[r * NACT + k])
            )


def _draw(ref s: UniformSampler, n: Int) raises -> List[Int]:
    var d = s.draw(n)
    var out = List[Int]()
    for i in range(n):
        out.append(Int(d.host[i]))
    return out^


def _dist_to_target(ref e: Env) -> Float64:
    """‖qpos‖ — the mass sits at `qpos` (two slide joints) and the target at
    the origin, so this is exact and, unlike the reward, does not underflow."""
    var acc = Float64(0)
    for k in range(NQ):
        var v = Float64(e.d.qpos.data[k])
        acc += v * v
    return sqrt(acc)


def main() raises:
    var n_rows = collect()

    print("[2] training FB for", TRAIN_STEPS, "steps (d =", D, ") ...")
    var store = TrajectoryStore(String(STORE_PATH))
    var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
    var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
    var action = ResidentColumn[DType.float32].load(store, String("action"))
    var sampler = UniformSampler(n_rows)

    var t = Trainer.make(lr=3e-4, gamma=0.98, tau=0.01)
    seed(SEED + 1)

    var s = Tensor()
    var a = Tensor()
    var s_next = Tensor()
    var s_plus = Tensor()
    var z = Tensor()

    for step in range(TRAIN_STEPS):
        # Two INDEPENDENT draws. `s_plus` is not `s_next` — see loss.mojo.
        var i0 = _draw(sampler, BATCH)
        var i1 = _draw(sampler, BATCH)
        # `s'` is the row after `s` within the same episode. Rows at an episode
        # boundary would pair a terminal state with the next episode's start,
        # so they are pulled back by one; with EP_LEN=200 that biases a
        # half-percent of draws and keeps the transition valid.
        var i0n = List[Int]()
        for b in range(BATCH):
            var r = i0[b]
            var nxt = r + 1
            if nxt >= n_rows or (nxt % EP_LEN) == 0:
                nxt = r
            i0n.append(nxt)

        _gather_obs(qpos, qvel, i0, s)
        _gather_act(action, i0, a)
        _gather_obs(qpos, qvel, i0n, s_next)
        _gather_obs(qpos, qvel, i1, s_plus)

        # z: half uniform on the sphere, half from B(s+) — the mixture the
        # sampler owns, renormalised inside it.
        var b_sp = Tensor()
        t.backward_embed[BATCH](s_plus, b_sp)
        var b_list = List[Scalar[DT]](length=BATCH * D, fill=Scalar[DT](0))
        for i in range(BATCH * D):
            b_list[i] = b_sp.data[i]
        var zl = sample_z[D](BATCH, b_list, BATCH, uniform_frac=0.5)
        z.ensure(BATCH * D)
        for i in range(BATCH * D):
            z.data[i] = zl[i]

        t.load_batch(s, a, s_next, s_plus, z)
        var l = t.train_step()
        if step % 250 == 0 or step == TRAIN_STEPS - 1:
            print(
                "       step", step, " measure", l.measure, " ortho", l.ortho,
                " |B|", l.b_norm,
            )

    # ══════════════════════════════════════════════════════════════════
    # 3. Zero-shot inference — z from a reward the collection never saw
    # ══════════════════════════════════════════════════════════════════
    print("[3] relabelling", RELABEL_ROWS, "rows and computing z ...")
    var scorer = Env()
    _ = scorer.reset()
    var rel_idx = _draw(sampler, RELABEL_ROWS)

    var b_in = Tensor()
    _gather_obs(qpos, qvel, rel_idx, b_in)
    var b_out = Tensor()
    t.backward_embed[RELABEL_ROWS](b_in, b_out)

    var b_flat = List[Scalar[DT]](
        length=RELABEL_ROWS * D, fill=Scalar[DT](0)
    )
    for i in range(RELABEL_ROWS * D):
        b_flat[i] = b_out.data[i]

    var rewards = List[Scalar[DT]](length=RELABEL_ROWS, fill=Scalar[DT](0))
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    var ac = List[Float64](length=NACT, fill=0.0)
    var r_sum = Float64(0)
    var r_max = Float64(0)
    for i in range(RELABEL_ROWS):
        var r = rel_idx[i]
        for k in range(NQ):
            q[k] = Float64(qpos.host[r * NQ + k])
        for k in range(NV):
            v[k] = Float64(qvel.host[r * NV + k])
        for k in range(NACT):
            ac[k] = Float64(action.host[r * NACT + k])
        var got = scorer.reward_at(q, v, ac)
        var rv = Float64(got[0])
        rewards[i] = Scalar[DT](rv)
        r_sum += rv
        if rv > r_max:
            r_max = rv
    print(
        "       relabelled reward: mean", r_sum / Float64(RELABEL_ROWS),
        " max", r_max,
    )

    var zl = z_from_reward[D](b_flat, rewards, RELABEL_ROWS)
    var z_eval = Tensor.alloc(D)
    for k in range(D):
        z_eval.data[k] = zl[k]

    # ══════════════════════════════════════════════════════════════════
    # 4. Evaluate — pi_z vs the random policy that produced the data
    # ══════════════════════════════════════════════════════════════════
    print("[4] evaluating", EVAL_EPISODES, "episodes ...")
    var ret_pi = Float64(0)
    var ret_rand = Float64(0)
    # ⚠ The RETURN is not the primary metric here, and reporting it alone would
    # be dishonest. point_mass scores ~1e-245 more than 10 cm out, so 200 steps
    # from a random spawn give both policies a return in the 1e-10 range: a
    # ratio between them is a ratio of two underflowed numbers and would read
    # as a large win no matter what the policy did.
    #
    # DISTANCE TO TARGET is the quantity that discriminates. The target sits at
    # the origin and `qpos` IS the mass position (two slide joints), so
    # ‖qpos‖ is exact, never underflows, and answers the actual question:
    # does pi_z move the mass toward the target. `min_dist` over the episode is
    # reported alongside `final_dist` because a policy that reaches the target
    # and overshoots is still doing the right thing.
    var fin_pi = Float64(0)
    var fin_rand = Float64(0)
    var min_pi = Float64(0)
    var min_rand = Float64(0)

    for ep in range(EVAL_EPISODES):
        # Same reset seed for both policies, so the comparison is over the
        # SAME start states. Comparing returns from different starts on a
        # reward this sharp would be dominated by where the mass happened to
        # spawn, not by the policy.
        seed(SEED + 1000 + ep)
        var e1 = Env()
        _ = e1.reset()
        var obs1 = Tensor.alloc(OBS)
        var act_out = Tensor()
        var mn1 = Float64(1e30)
        for _t in range(EVAL_LEN):
            for k in range(NQ):
                obs1.data[k] = Scalar[DT](Float64(e1.d.qpos.data[k]))
            for k in range(NV):
                obs1.data[NQ + k] = Scalar[DT](Float64(e1.d.qvel.data[k]))
            t.act[1](obs1, z_eval, act_out)
            var a1 = Env.ActionType()
            for k in range(NACT):
                a1.data[k] = Float64(act_out.data[k])
            var o = e1.step(a1)
            ret_pi += Float64(o[1])
            var d1 = _dist_to_target(e1)
            if d1 < mn1:
                mn1 = d1
        fin_pi += _dist_to_target(e1)
        min_pi += mn1

        seed(SEED + 1000 + ep)
        var e2 = Env()
        _ = e2.reset()
        var mn2 = Float64(1e30)
        for _t in range(EVAL_LEN):
            var a2 = Env.ActionType()
            for k in range(NACT):
                a2.data[k] = random_float64() * 2.0 - 1.0
            var o = e2.step(a2)
            ret_rand += Float64(o[1])
            var d2 = _dist_to_target(e2)
            if d2 < mn2:
                mn2 = d2
        fin_rand += _dist_to_target(e2)
        min_rand += mn2

    var n = Float64(EVAL_EPISODES)
    var mp = ret_pi / n
    var mr = ret_rand / n
    var fp = fin_pi / n
    var fr = fin_rand / n
    var np_ = min_pi / n
    var nr = min_rand / n
    print("")
    print("  ================================================================")
    print("   PRIMARY — distance to target (m), lower is better")
    print("     pi_z    final", fp, "  best-in-episode", np_)
    print("     random  final", fr, "  best-in-episode", nr)
    print("")
    print("   SECONDARY — episode return. Both are ~0 because point_mass")
    print("   scores ~1e-245 outside a few cm; a RATIO of these is a ratio of")
    print("   two underflowed numbers and means nothing on its own.")
    print("     pi_z  ", mp, "   random", mr)
    print("  ================================================================")

    # ── the verdict ──────────────────────────────────────────────────
    # WHICH metric decides depends on whether the return has left the
    # underflow regime. Below ~1e-3 the reward is numerically dead and a
    # comparison of two such numbers says nothing (see the note above), so
    # distance is all there is. Above it, the RETURN is the task's own metric
    # and the right one to judge by.
    var return_is_real = mp > 1e-3
    if return_is_real and mp > 1000.0 * mr:
        print(
            "  pi_z BEATS random on the task's own metric: return", mp,
            "vs", mr, ". That comparison is meaningful here — pi_z's return is"
            " O(1), not an underflowed ~1e-10 — so the offline pipeline"
            " (random-policy data -> relabel -> z -> policy) reaches a target"
            " the collection run had no notion of."
        )
        if fp > fr:
            print(
                "  ⚠ But it does NOT HOLD the target: best-in-episode distance",
                np_, "vs random's", nr, ", while the FINAL distance is worse (",
                fp, "vs", fr, "). pi_z reaches and overshoots. That is a"
                " policy-quality gap, not a wiring bug — expect it to close"
                " with the full 500k budget and a larger d."
            )
    elif return_is_real:
        print(
            "  pi_z's return (", mp, ") is real but not clearly above random's"
            " (", mr, "). Inconclusive."
        )
    else:
        print(
            "  Both returns are in the underflow regime (", mp, "vs", mr,
            "), so only distance can be read: pi_z", fp, "vs random", fr, "."
        )
        if fp >= fr - 0.005:
            print(
                "  pi_z is not closer to the target. Check |B| above before"
                " assuming a defect: a value drifting toward 0 is the collapse"
                " L_ortho exists to prevent, and the measure loss descends"
                " anyway. If |B| is stable and the loss has PLATEAUED, the run"
                " is converged and the problem is not the step count."
            )
