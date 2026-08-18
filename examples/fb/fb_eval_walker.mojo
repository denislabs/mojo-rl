"""Zero-shot evaluation of an FB checkpoint on walker — the decisive test.

`docs/BFM_ZERO_SHOT_RL.md` §13 step 4a, for M2. Loads a checkpoint written by
`fb_train_gpu.mojo`, computes `z` from a task's TRUE reward by relabelling
dataset rows, and rolls out `pi_z` against the random policy that produced the
data.

**Why this and not the loss curves.** FB's losses are not a convergence signal
you can read: the measure loss descends whether `B` collapses, expands or
behaves, and its absolute scale is set by `||F||` and `||B||` rather than by
anything about the representation. Two separate readings of those curves have
already been wrong on this project. The zero-shot return is the quantity the
whole component exists to produce, and it either beats random or it does not.

**Walker's reward is safe to compare directly**, unlike point_mass. `stand`,
`walk` and `run` all return O(0.1 - 1) per step through `rewards.tolerance`, so
a 1000-step return lands in the hundreds and a ratio between two policies means
what it looks like. point_mass scores ~1e-245 more than 10 cm out, which is why
the M1 script reports DISTANCE and this one reports return.

**All three tasks from ONE checkpoint.** That is the entire claim of zero-shot
RL: `stand`, `walk` and `run` are three different rewards, and `z` is computed
per task by relabelling with `reward_at`. Nothing is retrained between them. If
FB works, one set of weights covers all three; if `z` were doing nothing, all
three would score alike and match random.

⚠ The three `z` vectors must DIFFER. A `z` that ignored its reward would still
be renormalised onto the sphere and still produce a plausible policy — the run
would look fine and mean nothing. Checked explicitly below.

Run (CPU: dm_control envs are CPU-only, gap G10):
    pixi run mojo run -I . examples/fb/fb_eval_walker.mojo

⚠⚠ **THIS FILE HAS NEVER BEEN COMPILED.** It was written while
`physics3d/model/model_renderer.mojo` had an uncommitted in-progress change
calling `MODEL_DEF.render_skin(...)`, a method not yet declared on
`ModelDefLike` — which breaks every build that touches `Phyics3dEnv`, and this
script needs one to relabel rewards. Committed so the work is not lost, NOT
because it is verified. Build it once that trait method lands, and expect the
usual first-compile corrections before trusting a number out of it.
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.sys import argv

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import ResidentColumn
from mojo_rl.data.sampler import UniformSampler

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb import z_from_reward
from mojo_rl.deep_agents.fb.obs_norm import ObsNorm


comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
comptime OBS: Int = NQ + NV
comptime D: Int = 128
comptime HID: Int = 1024
comptime BATCH: Int = 1024          # must match the trained checkpoint

# Checkpoints are STEP-STAMPED (`<path>.<step>`, plus `.final`) — point this at
# the step you want. ⚠ Prefer a checkpoint from a HEALTHY stretch over the
# latest one: FB's measure loss cycles, and the last file written is not
# necessarily the best model. Read the log for a run of consistently NEGATIVE
# measure with a small, non-growing `actor` before choosing.
# ⚠ These defaults were STALE and dangerous: they pointed at
# `fb_walker_d128.ckpt.100000`, which is from the LEARNABLE-`LayerNorm` run.
# That BNet carries gamma/beta Params this architecture does not have, so the
# load either fails or silently skips them — and a silently-skipped norm layer
# is exactly the failure this file's own BNet comment warns about.
comptime CKPT: StaticString = "checkpoints/fb_walker_all_d128.ckpt.1200000"
# ⚠ MUST be the store the checkpoint TRAINED on. §13 records an eval that
# computed z from a local 10 k store while the checkpoint had trained on 1 M —
# the numbers were not wrong so much as unattributable.
comptime STORE: StaticString = "fb_walker_all_sac.h5"
# ⚠ `z = E[B(s)·r(s)]` is NOT invariant to a constant offset in `r`: adding `c`
# adds `c·E[B(s)]` — the dataset-mean direction, which carries no task
# information. It should be near-harmless in exact FB (a constant reward shifts
# every action's return equally) and is not with a learned B.
#
# MEASURED on the 500 k checkpoint over 4096 rows, cosine between task z's:
#
#            stand-walk   walk-run   stand-run
#   uncentered   0.885      0.949      0.905
#   centered     0.618      0.846      0.648
#
# So the three "tasks" currently query policies 88-95% aligned in z-space, and
# centering genuinely separates them. But this was measured against a BANG-BANG
# actor, where z barely reaches the behaviour at all — so it is a HYPOTHESIS
# with support, not a fix. Flip the flag and compare once a non-saturated
# policy exists; do not change the default on the strength of the table above.
#
# MEASURED 2026-08-13 on the fixed 500 k checkpoint, 64 episodes, paired:
#
#             uncentered           centered
#   stand   ratio 1.152 t 1.23   ratio 1.323 t 3.37  p 0.0013
#   walk    ratio 1.007 t 0.07   ratio 1.233 t 2.62  p 0.0110
#   run     ratio 1.135 t 1.84   ratio 1.244 t 3.54  p 0.0008
#
# Centering turns three null results into three positive ones and more than
# doubles the task separation (|z_stand - z_walk|_1: 44 -> 101). THAT is why
# the default is now True — a policy-level A/B, not the cosine proxy that
# suggested it.
comptime Z_CENTER: Bool = True
comptime RELABEL_ROWS: Int = 4096
# ⚠ 10 episodes CANNOT resolve these effects. At the measured spreads the
# episode counts needed for t=2 are ~49 (stand), ~97 (run) and ~916 (walk) —
# walk's effect is small against a large per-episode variance. 64 makes stand
# and run decidable and leaves walk honestly reported as underpowered rather
# than silently called "within noise" at a sample size that could never say
# otherwise.
comptime EVAL_EPISODES: Int = 64

# ⚠⚠ **ONE CHECKPOINT IS NOT A MEASUREMENT OF THE METHOD.** Swept across 11
# rungs of the same 1.22 M run (64 episodes each, centered z, deterministic
# eval — the random baseline is bit-identical at every rung, so these are real
# weight differences, not eval noise):
#
#     step   stand   walk    run
#     100k   1.154  0.927  1.066
#     300k   1.430  1.273  1.671
#     500k   1.323  1.233  1.244
#     800k   1.415  1.696  1.570
#     900k   1.417  1.523  1.160
#     950k   1.621  1.633  1.277
#    1000k   1.333  0.942  1.121   <- walk goes NULL
#    1050k   1.376  1.236  1.344
#    1100k   1.525  1.519  1.488
#    1150k   1.399  1.544  1.457
#    1200k   1.609  2.413  1.998   <- best of 11
#
# Late-region mean (8 checkpoints, 800k-1200k):
#     stand 1.462 (sd 0.11)   walk 1.563 (sd 0.42)   run 1.427 (sd 0.28)
#
# So walk swings 0.94 -> 2.41 between checkpoints 200 k apart, and the last
# rung is 1.54x / 1.40x the TYPICAL late checkpoint on walk / run. Quoting the
# final checkpoint reports the best of eleven draws.
#
# ⚠ Two consequences for anyone reading a number out of this file:
#   * Report the MEAN over several late rungs, not the last one. The sweep is
#     ~90 s per rung with the runtime checkpoint argument.
#   * A single rung can say "walk does not work" (1000k) or "walk works at
#     2.4x" (1200k) about the SAME run.
#
# ⚠ And the curve is FLAT past ~300 k: 300-500k means 1.377 / 1.253 / 1.458 vs
# 800-1200k 1.462 / 1.563 / 1.427. More gradient steps are not what closes the
# remaining gap to the SAC experts in the data (~19% / ~5% / ~5% of expert at
# the late mean) — that is what the FB-CPR milestone is for.
comptime _EVAL_EPISODES_NOTE: Int = 0
comptime EVAL_LEN: Int = 1000       # dm_control's own episode length
comptime SEED: Int = 20260805

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

# ⚠ MUST match `fb_train_gpu.mojo` exactly, the norm layer included — the
# checkpoint is keyed by parameter name and a shape mismatch raises, but a
# MISSING norm would load the Linears fine and silently drop B's output scale.
#
# ⚠⚠ `LayerNormNoAffine`, NOT `LayerNorm`: the learnable gamma let ||B|| drift
# 11.31 -> 17.54 over 100 k steps with `L_ortho` rising 8.8x behind it. The
# no-affine layer carries NO params, so a checkpoint written against one and
# loaded into the other would restore every Linear without complaint and
# differ only in B's scale — exactly the silent failure this comment warns
# about, one layer deeper.
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, NACT, D, BATCH, "cpu"]


def _draw(ref s: UniformSampler, n: Int) raises -> List[Int]:
    var d = s.draw(n)
    var out = List[Int]()
    for i in range(n):
        out.append(Int(d.host[i]))
    return out^


def _z_for_task[
    CONFIG: Phyics3dEnvConfig
](
    mut t: Trainer,
    ref qpos: ResidentColumn[DType.float32],
    ref qvel: ResidentColumn[DType.float32],
    ref action: ResidentColumn[DType.float32],
    ref idx: List[Int],
    ref b_flat: List[Scalar[DT]],
    name: String,
) raises -> Tensor:
    """`z = E_rho[B(s)·r(s)]` with `r` RELABELLED offline for this task.

    The collection run was a random policy with no notion of standing, walking
    or running. Every reward here is invented after the fact and recomputed
    from stored generalized coordinates — which is the whole architectural bet
    of component 0, and the reason the dataset stores qpos/qvel.
    """
    comptime Env = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]
    var scorer = Env()
    _ = scorer.reset()

    var n = len(idx)
    var rewards = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    var ac = List[Float64](length=NACT, fill=0.0)
    var lo = Float64(1e30)
    var hi = Float64(-1e30)
    var sm = Float64(0)
    for i in range(n):
        var r = idx[i]
        for k in range(NQ):
            q[k] = Float64(qpos.host[r * NQ + k])
        for k in range(NV):
            v[k] = Float64(qvel.host[r * NV + k])
        for k in range(NACT):
            ac[k] = Float64(action.host[r * NACT + k])
        var got = scorer.reward_at(q, v, ac)
        var rv = Float64(got[0])
        rewards[i] = Scalar[DT](rv)
        sm += rv
        if rv < lo:
            lo = rv
        if rv > hi:
            hi = rv
    print(
        "      ", name, "relabelled reward: mean", sm / Float64(n),
        " range [", lo, ",", hi, "]",
    )

    comptime if Z_CENTER:
        var mu = sm / Float64(n)
        for i in range(n):
            rewards[i] = Scalar[DT](Float64(rewards[i]) - mu)
    var zl = z_from_reward[D](b_flat, rewards, n)
    var z = Tensor.alloc(D)
    for k in range(D):
        z.data[k] = zl[k]
    return z^


def _rollout[
    CONFIG: Phyics3dEnvConfig
](
    mut t: Trainer, ref z: Tensor, use_policy: Bool, ep_seed: Int,
    mut act_mean_abs: Float64, mut act_sat_frac: Float64,
    ref onorm: ObsNorm[OBS], has_norm: Bool,
) raises -> Float64:
    comptime Env = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]
    seed(ep_seed)
    var env = Env()
    _ = env.reset()

    var obs = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    for k in range(D):
        z1.data[k] = z.data[k]
    var act_out = Tensor()
    var ret = Float64(0)
    # ⚠ What the policy DOES, not just what it scores. A Tanh actor that has
    # saturated emits +-1 constantly: that is bang-bang torque, which on walker
    # is reliably WORSE than random jitter, and it looks identical to
    # "undertrained" in the return alone.
    var abs_sum = Float64(0)
    var sat = 0
    var n_act = 0
    for _ in range(EVAL_LEN):
        var a = Env.ActionType()
        if use_policy:
            for k in range(NQ):
                obs.data[k] = Scalar[DT](Float64(env.d.qpos.data[k]))
            for k in range(NV):
                obs.data[NQ + k] = Scalar[DT](Float64(env.d.qvel.data[k]))
            # ⚠⚠ The SAME standardisation the actor was trained under, taken
            # from `<ckpt>.norm`. Skipping it here would feed the policy a
            # distribution it never saw; it would still emit actions in
            # [-1, 1] and the run would still produce a number.
            if has_norm:
                onorm.apply_row(obs)
            t.act[1](obs, z1, act_out)
            for k in range(NACT):
                var av = Float64(act_out.data[k])
                a.data[k] = av
                abs_sum += abs(av)
                if abs(av) > 0.99:
                    sat += 1
                n_act += 1
        else:
            for k in range(NACT):
                a.data[k] = random_float64() * 2.0 - 1.0
        var out = env.step(a)
        ret += Float64(out[1])
    if use_policy and n_act > 0:
        act_mean_abs = abs_sum / Float64(n_act)
        act_sat_frac = Float64(sat) / Float64(n_act)
    return ret


def _eval_task[
    CONFIG: Phyics3dEnvConfig
](
    mut t: Trainer, ref z: Tensor, name: String,
    ref onorm: ObsNorm[OBS], has_norm: Bool,
) raises:
    # ⚠ PAIRED per episode. Both policies see the SAME reset seed, so the
    # difference is taken within an episode and the (large) spread across start
    # states cancels. Comparing two independent means would need far more
    # episodes to see the same effect.
    var diffs = List[Float64]()
    var rp = Float64(0)
    var rr = Float64(0)
    var mean_abs = Float64(0)
    var sat_frac = Float64(0)
    for ep in range(EVAL_EPISODES):
        var ama = Float64(0)
        var asf = Float64(0)
        var a = _rollout[CONFIG](
            t, z, True, SEED + 1000 + ep, ama, asf, onorm, has_norm
        )
        var dummy_a = Float64(0)
        var dummy_b = Float64(0)
        var b = _rollout[CONFIG](
            t, z, False, SEED + 1000 + ep, dummy_a, dummy_b, onorm, has_norm
        )
        mean_abs += ama
        sat_frac += asf
        rp += a
        rr += b
        diffs.append(a - b)
    var n = Float64(EVAL_EPISODES)
    var mp = rp / n
    var mr = rr / n

    # Standard error of the PAIRED difference. Without it a ratio of 1.15 is
    # not a claim — walker returns vary by tens across start states, and the
    # whole margin can sit inside one standard error.
    var md = Float64(0)
    for i in range(len(diffs)):
        md += diffs[i]
    md /= n
    var sq = Float64(0)
    for i in range(len(diffs)):
        var d = diffs[i] - md
        sq += d * d
    var sd = sqrt(sq / (n - 1.0)) if EVAL_EPISODES > 1 else 0.0
    var se = sd / sqrt(n)
    var tstat = md / se if se > 1e-12 else 0.0
    print(
        "   ", name, ": pi_z", mp, "  random", mr,
        "  ratio", mp / mr if mr > 1e-9 else 0.0,
    )
    print(
        "            paired diff", md, "+-", se, " (t =", tstat, ")",
        " SIGNAL" if abs(tstat) > 2.0 else " within noise",
    )
    print(
        "            pi_z action: mean|a| =", mean_abs / n,
        " saturated(|a|>0.99) =", sat_frac / n,
    )


def main() raises:
    # ⚠ Checkpoint path is a RUNTIME argument when one is given. `CKPT` is
    # comptime, so sweeping rungs would otherwise cost one full rebuild per
    # checkpoint — which is how "does more training help?" becomes an
    # afternoon instead of ten minutes.
    var ck = String(CKPT)
    var av = argv()
    if len(av) > 1:
        ck = String(av[1])
    print("[1] loading checkpoint", ck, "...")
    var t = Trainer.make(lr=3e-4, ctx=None)
    t.load_state(ck)

    # ⚠⚠ Whether this run standardised its observations is read off the
    # CHECKPOINT, never off a flag in this file. `fb_train_gpu.mojo` writes
    # `<ckpt>.norm` beside every rung it saves; a run trained on raw inputs
    # writes none and this finds none. See `fb/obs_norm.mojo` for why a flag
    # here would be the wrong shape: train-normalised/eval-raw raises nothing,
    # scores plausibly, and reads as "that arm did not help".
    var maybe_norm = ObsNorm[OBS].try_load(ck + ".norm")
    var has_norm = Bool(maybe_norm)
    var onorm = ObsNorm[OBS]()
    if has_norm:
        onorm = maybe_norm.take()
        print("      obs standardisation: ON (from", ck + ".norm)")
    else:
        print("      obs standardisation: off (no", ck + ".norm)")

    print("[2] loading", STORE, "for relabelling ...")
    var store = TrajectoryStore(String(STORE))
    var n_rows = store.n_rows()
    var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
    var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
    var action = ResidentColumn[DType.float32].load(store, String("action"))
    var sampler = UniformSampler(n_rows)
    seed(SEED)
    var idx = _draw(sampler, RELABEL_ROWS)

    # B(s) on the relabel rows — computed ONCE, shared by all three tasks.
    var b_in = Tensor.alloc(RELABEL_ROWS * OBS)
    for i in range(RELABEL_ROWS):
        var r = idx[i]
        for k in range(NQ):
            b_in.data[i * OBS + k] = Scalar[DT](Float64(qpos.host[r * NQ + k]))
        for k in range(NV):
            b_in.data[i * OBS + NQ + k] = Scalar[DT](
                Float64(qvel.host[r * NV + k])
            )
    if has_norm:
        onorm.apply_rows(b_in, RELABEL_ROWS)
    var b_out = Tensor()
    t.backward_embed[RELABEL_ROWS](b_in, b_out)
    var b_flat = List[Scalar[DT]](
        length=RELABEL_ROWS * D, fill=Scalar[DT](0)
    )
    for i in range(RELABEL_ROWS * D):
        b_flat[i] = b_out.data[i]

    print("[3] computing z per task from RELABELLED rewards ...")
    var z_stand = _z_for_task[DMWalkerConfig[0.0]](
        t, qpos, qvel, action, idx, b_flat, String("stand")
    )
    var z_walk = _z_for_task[DMWalkerConfig[1.0]](
        t, qpos, qvel, action, idx, b_flat, String("walk ")
    )
    var z_run = _z_for_task[DMWalkerConfig[8.0]](
        t, qpos, qvel, action, idx, b_flat, String("run  ")
    )

    # ⚠ The three z MUST differ. A z that ignored its reward would still be
    # renormalised onto the sphere and still drive a plausible-looking policy;
    # the whole evaluation would then be three copies of one number.
    var d_sw = Float64(0)
    var d_wr = Float64(0)
    for k in range(D):
        d_sw += abs(Float64(z_stand.data[k]) - Float64(z_walk.data[k]))
        d_wr += abs(Float64(z_walk.data[k]) - Float64(z_run.data[k]))
    print("      |z_stand - z_walk|_1 =", d_sw, "  |z_walk - z_run|_1 =", d_wr)
    if d_sw < 1e-3 or d_wr < 1e-3:
        print(
            "  ⚠⚠ The task z vectors are (nearly) IDENTICAL. z_from_reward is"
            " not responding to its reward argument, and the three numbers"
            " below are three copies of one experiment."
        )

    print("[4] rolling out", EVAL_EPISODES, "x", EVAL_LEN, "steps per task ...")
    _eval_task[DMWalkerConfig[0.0]](t, z_stand, String("stand"), onorm, has_norm)
    _eval_task[DMWalkerConfig[1.0]](t, z_walk, String("walk "), onorm, has_norm)
    _eval_task[DMWalkerConfig[8.0]](t, z_run, String("run  "), onorm, has_norm)
    print("")
    print("  A ratio > 1 means one set of weights, given only that task's")
    print("  reward AFTER training, beat a uniform-random policy. That is the")
    print("  zero-shot claim.")
    print("")
    print("  ⚠ The baseline is UNIFORM RANDOM, not the data-generating policy.")
    print("  This store came from SAC ladders whose top rungs score ~983 /")
    print("  ~965 / ~720, so beating random is the floor, not the ceiling:")
    print("  pi_z currently recovers ~17% / ~4% / ~4% of expert. The earlier")
    print("  wording here said \'the random policy that produced the data\',")
    print("  which was true of the M1 store and is not true of this one.")
