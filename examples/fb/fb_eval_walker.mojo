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

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm import LayerNorm

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import ResidentColumn
from mojo_rl.data.sampler import UniformSampler

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb import z_from_reward


comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
comptime OBS: Int = NQ + NV
comptime D: Int = 128
comptime HID: Int = 1024
comptime BATCH: Int = 1024          # must match the trained checkpoint

comptime CKPT: StaticString = "fb_walker_d128.ckpt"
comptime STORE: StaticString = "/tmp/fb_walker_wide.h5"
comptime RELABEL_ROWS: Int = 4096
comptime EVAL_EPISODES: Int = 10
comptime EVAL_LEN: Int = 1000       # dm_control's own episode length
comptime SEED: Int = 20260805

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

# ⚠ MUST match `fb_train_gpu.mojo` exactly, LayerNorm included — the checkpoint
# is keyed by parameter name and a shape mismatch raises, but a MISSING
# LayerNorm would load the Linears fine and silently drop B's output scale.
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNorm[D]
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

    var zl = z_from_reward[D](b_flat, rewards, n)
    var z = Tensor.alloc(D)
    for k in range(D):
        z.data[k] = zl[k]
    return z^


def _rollout[
    CONFIG: Phyics3dEnvConfig
](mut t: Trainer, ref z: Tensor, use_policy: Bool, ep_seed: Int) raises -> Float64:
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
    for _ in range(EVAL_LEN):
        var a = Env.ActionType()
        if use_policy:
            for k in range(NQ):
                obs.data[k] = Scalar[DT](Float64(env.d.qpos.data[k]))
            for k in range(NV):
                obs.data[NQ + k] = Scalar[DT](Float64(env.d.qvel.data[k]))
            t.act[1](obs, z1, act_out)
            for k in range(NACT):
                a.data[k] = Float64(act_out.data[k])
        else:
            for k in range(NACT):
                a.data[k] = random_float64() * 2.0 - 1.0
        var out = env.step(a)
        ret += Float64(out[1])
    return ret


def _eval_task[
    CONFIG: Phyics3dEnvConfig
](mut t: Trainer, ref z: Tensor, name: String) raises:
    # ⚠ PAIRED per episode. Both policies see the SAME reset seed, so the
    # difference is taken within an episode and the (large) spread across start
    # states cancels. Comparing two independent means would need far more
    # episodes to see the same effect.
    var diffs = List[Float64]()
    var rp = Float64(0)
    var rr = Float64(0)
    for ep in range(EVAL_EPISODES):
        var a = _rollout[CONFIG](t, z, True, SEED + 1000 + ep)
        var b = _rollout[CONFIG](t, z, False, SEED + 1000 + ep)
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


def main() raises:
    print("[1] loading checkpoint", CKPT, "...")
    var t = Trainer.make(lr=3e-4, ctx=None)
    t.load_state(String(CKPT))

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
    _eval_task[DMWalkerConfig[0.0]](t, z_stand, String("stand"))
    _eval_task[DMWalkerConfig[1.0]](t, z_walk, String("walk "))
    _eval_task[DMWalkerConfig[8.0]](t, z_run, String("run  "))
    print("")
    print("  A ratio > 1 on a task means one set of weights, given only that")
    print("  task's reward AFTER training, beat the random policy that")
    print("  produced the data. That is the zero-shot claim.")
