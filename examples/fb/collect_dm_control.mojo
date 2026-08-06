"""Collect an FB dataset for one dm_control domain, and MEASURE its coverage.

`docs/BFM_ZERO_SHOT_RL.md` component 1:

> Coverage metrics to compute BEFORE training FB [...] Otherwise you discover
> the dataset is poor after 2 M gradient steps.

So this script does not just collect. It collects TWICE — once with the suite's
own reset, once with `WideResetConfig` — and then asks RND a question with a
directional answer:

    fit the predictor on the NARROW dataset,
    then measure novelty on WIDE rows.

High novelty there means the widened reset visits states the suite reset never
produces, i.e. the lever added coverage. The reverse direction (fit wide,
measure narrow) is the control: it should be LOW, because the wide distribution
contains the narrow one. Reporting only one direction would not distinguish
"added coverage" from "moved somewhere else entirely", and moving is not what
is wanted — the narrow states are the useful ones for `stand`.

## Choosing the domain

`DOMAIN` below is a compile-time constant, and only the selected branch is
instantiated. That is deliberate: quadruped alone takes minutes to compile, and
building all three for a run that uses one is how a laptop ends up thermally
throttled. Edit the line, rebuild, run.

⚠ Quadruped uses `HEIGHT_OFF`. It sets `RESET_FIND_HEIGHT`, so a height range
would be discarded by `_find_non_contacting_height` after the reset hook
returns — `WideResetConfig` refuses the combination at compile time. Its
diversity comes from the per-episode orientation the suite already draws, plus
`QVEL_SCALE`.

Run:
    pixi run mojo run -I . examples/fb/collect_dm_control.mojo
"""

from std.math import sqrt
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import ResidentColumn
from mojo_rl.data.sampler import UniformSampler

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig

from mojo_rl.deep_agents.fb.collect import collect_random
from mojo_rl.deep_agents.fb.rnd import RND

from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel, DMCheetahConfig
from mojo_rl.envs.dm_control.wide_reset import (
    WideResetConfig,
    WALKER_ROOTZ_ADR, WALKER_Z_LO, WALKER_Z_HI,
    CHEETAH_ROOTZ_ADR, CHEETAH_Z_LO, CHEETAH_Z_HI,
)


# ── pick ONE ─────────────────────────────────────────────────────────────
comptime DOMAIN: StaticString = "walker"      # "walker" | "cheetah"

comptime N_EPISODES: Int = 40
comptime EP_LEN: Int = 250
comptime SEED: Int = 20260805
comptime RND_FIT_STEPS: Int = 400
comptime RND_BATCH: Int = 64
comptime RND_FEAT: Int = 32
comptime RND_HID: Int = 128
comptime PROBE_ROWS: Int = 512

comptime OUT_DIR: StaticString = "/tmp/"


def _obs_dim[NQ: Int, NV: Int]() -> Int:
    return NQ + NV


def _load_obs(
    path: String,
    mut store_rows: Int,
    mut qpos: ResidentColumn[DType.float32],
    mut qvel: ResidentColumn[DType.float32],
) raises:
    """Out-params rather than a tuple: two elements of ONE tuple share an
    origin, so passing both to `_gather` trips Mojo's aliasing check. Separate
    variables give separate origins."""
    var st = TrajectoryStore(path)
    store_rows = st.n_rows()
    qpos = ResidentColumn[DType.float32].load(st, String("qpos"))
    qvel = ResidentColumn[DType.float32].load(st, String("qvel"))


def _gather[
    NQ: Int, NV: Int
](
    ref qpos: ResidentColumn[DType.float32],
    ref qvel: ResidentColumn[DType.float32],
    ref idx: List[Int],
    mut dst: Tensor,
) raises:
    comptime OBS = NQ + NV
    dst.ensure(len(idx) * OBS)
    for b in range(len(idx)):
        var r = idx[b]
        for k in range(NQ):
            dst.data[b * OBS + k] = Scalar[DT](Float64(qpos.host[r * NQ + k]))
        for k in range(NV):
            dst.data[b * OBS + NQ + k] = Scalar[DT](
                Float64(qvel.host[r * NV + k])
            )


def _draw_early(n_rows: Int, n: Int, ep_len: Int, window: Int) raises -> List[Int]:
    """Rows from the FIRST `window` steps of each episode.

    A reset lever only writes the START state, so the obvious worry is that a
    contractive rollout washes it out and a whole-dataset average understates
    what the lever did.

    ⚠ On walker that worry is WRONG, and this probe is what showed it: early
    rows come out at 1.78x against 2.10x over the whole dataset — LOWER, not
    higher. Dropping the torso from 1.5 m produces a tumble whose high-velocity
    states the suite reset never reaches, so the widening propagates through
    the velocities instead of decaying with the pose. Kept as a diagnostic, and
    as a reminder that the intuition was checkable and false.
    """
    var out = List[Int]()
    var eps = n_rows // ep_len
    if eps <= 0 or window <= 0:
        raise Error("_draw_early: bad episode geometry")
    for i in range(n):
        var e = Int(random_float64() * Float64(eps))
        if e >= eps:
            e = eps - 1
        var t = Int(random_float64() * Float64(window))
        if t >= window:
            t = window - 1
        out.append(e * ep_len + t)
    return out^


def _draw(ref s: UniformSampler, n: Int) raises -> List[Int]:
    var d = s.draw(n)
    var out = List[Int]()
    for i in range(n):
        out.append(Int(d.host[i]))
    return out^


def run[
    MODEL_DEF: ModelDefLike, NARROW: Phyics3dEnvConfig, WIDE: Phyics3dEnvConfig
](name: String) raises:
    comptime NQ = MODEL_DEF.NQ
    comptime NV = MODEL_DEF.NV
    comptime OBS = NQ + NV

    var narrow_path = String(OUT_DIR) + "fb_" + name + "_narrow.h5"
    var wide_path = String(OUT_DIR) + "fb_" + name + "_wide.h5"

    print("[1] collecting", name, "—", N_EPISODES, "x", EP_LEN, "steps ...")
    seed(SEED)
    var sn = collect_random[MODEL_DEF, NARROW](
        narrow_path, String("dm_control/") + name + "-narrow",
        N_EPISODES, EP_LEN, SEED,
    )
    seed(SEED + 1)
    var sw = collect_random[MODEL_DEF, WIDE](
        wide_path, String("dm_control/") + name + "-wide",
        N_EPISODES, EP_LEN, SEED + 1,
    )
    print("      narrow:", sn.rows, "rows   wide:", sw.rows, "rows")

    # ── coverage ─────────────────────────────────────────────────────────
    comptime Net = Sequential[
        Linear[OBS, RND_HID], ReLU[RND_HID], Linear[RND_HID, RND_FEAT]
    ]
    comptime Rnd = RND[Net, OBS, RND_FEAT, RND_BATCH]

    var nrows = 0
    var wrows = 0
    var nq = ResidentColumn[DType.float32](
        List[Scalar[DType.float32]](), 0, NQ, String("qpos")
    )
    var nv = ResidentColumn[DType.float32](
        List[Scalar[DType.float32]](), 0, NV, String("qvel")
    )
    var wq = ResidentColumn[DType.float32](
        List[Scalar[DType.float32]](), 0, NQ, String("qpos")
    )
    var wv = ResidentColumn[DType.float32](
        List[Scalar[DType.float32]](), 0, NV, String("qvel")
    )
    _load_obs(narrow_path, nrows, nq, nv)
    _load_obs(wide_path, wrows, wq, wv)
    var n_sampler = UniformSampler(nrows)
    var w_sampler = UniformSampler(wrows)

    print("[2] fitting RND on the NARROW dataset ...")
    seed(SEED + 2)
    var r_narrow = Rnd.make(lr=1e-3)
    var batch = Tensor()
    for _ in range(RND_FIT_STEPS):
        var idx = _draw(n_sampler, RND_BATCH)
        _gather[NQ, NV](nq, nv, idx, batch)
        _ = r_narrow.fit[RND_BATCH](batch)

    var probe_n = Tensor()
    var probe_w = Tensor()
    var idx_n = _draw(n_sampler, PROBE_ROWS)
    var idx_w = _draw(w_sampler, PROBE_ROWS)
    _gather[NQ, NV](nq, nv, idx_n, probe_n)
    _gather[NQ, NV](wq, wv, idx_w, probe_w)

    var dst = Tensor()
    var nov_nn = r_narrow.novelty[PROBE_ROWS](probe_n, dst)   # in-distribution
    var nov_nw = r_narrow.novelty[PROBE_ROWS](probe_w, dst)   # wide, unseen

    print("[3] fitting RND on the WIDE dataset (the control) ...")
    seed(SEED + 3)
    var r_wide = Rnd.make(lr=1e-3)
    for _ in range(RND_FIT_STEPS):
        var idx = _draw(w_sampler, RND_BATCH)
        _gather[NQ, NV](wq, wv, idx, batch)
        _ = r_wide.fit[RND_BATCH](batch)
    var nov_wn = r_wide.novelty[PROBE_ROWS](probe_n, dst)
    var nov_ww = r_wide.novelty[PROBE_ROWS](probe_w, dst)

    print("")
    print("  ================= coverage,", name, "=================")
    print("   fitted on NARROW:  narrow rows", nov_nn, "  wide rows", nov_nw)
    print("   fitted on WIDE  :  narrow rows", nov_wn, "  wide rows", nov_ww)
    var gain = nov_nw / nov_nn if nov_nn > 0 else 0.0
    var leak = nov_wn / nov_ww if nov_ww > 0 else 0.0
    print("   wide-is-novel-to-narrow ratio :", gain, " (want >> 1)")
    print("   narrow-is-novel-to-wide ratio :", leak, " (want ~1)")

    # ── the same question restricted to EARLY rows ───────────────────────
    comptime EARLY = 25
    var e_idx_n = _draw_early(nrows, PROBE_ROWS, EP_LEN, EARLY)
    var e_idx_w = _draw_early(wrows, PROBE_ROWS, EP_LEN, EARLY)
    var e_probe_n = Tensor()
    var e_probe_w = Tensor()
    _gather[NQ, NV](nq, nv, e_idx_n, e_probe_n)
    _gather[NQ, NV](wq, wv, e_idx_w, e_probe_w)
    var e_nn = r_narrow.novelty[PROBE_ROWS](e_probe_n, dst)
    var e_nw = r_narrow.novelty[PROBE_ROWS](e_probe_w, dst)
    var e_gain = e_nw / e_nn if e_nn > 0 else 0.0
    print("   first", EARLY, "steps only    :", e_gain,
          " (reported, not asserted — see the note below)")
    print("  ==================================================")
    # The CONTROL is the correctness check, not the gain. `leak ~ 1` says the
    # wide distribution CONTAINS the narrow one; a large `leak` would mean the
    # lever moved the distribution rather than widening it, and the suite's own
    # states are the useful ones for `stand`.
    if leak > 3.0:
        print(
            "  ⚠ The widened reset MOVED rather than widened (narrow rows are",
            leak, "x novel to it). A distribution that abandons the suite's own"
            " states is worse, not better — narrow the band."
        )
    elif gain < 1.3:
        print(
            "  ⚠ The widened reset barely changed the distribution (", gain,
            "x). Check ROOT_Z_ADR against"
            " tests/dm_control/test_wide_reset.mojo for this domain: a write"
            " to the wrong coordinate produces exactly this."
        )
    else:
        print(
            "  The widened reset added coverage (", gain, "x) while still"
            " containing the suite's own states (control", leak, "~ 1). That"
            " is the shape a working collection lever has."
        )
        print(
            "  ⚠ It is a MODEST gain, and RND novelty is a proxy, not a"
            " measure of how useful the added states are. Component 1 ranks"
            " start states above curiosity but ranks DUMPING FULL SAC REPLAY"
            " BUFFERS above both — the falls from early training are where the"
            " diversity is. Neither that nor the FB feedback loop is"
            " implemented; treat this number as the baseline they must beat."
        )
    # ⚠ The early-row figure above is REPORTED, not asserted, and it refuted
    # the obvious hypothesis when it was added: on walker it comes out BELOW
    # the whole-dataset gain (1.78 vs 2.10), so the rollout is not washing the
    # lever out. A walker dropped from 1.5 m tumbles into high-velocity states
    # the suite reset never reaches, so the widening propagates through the
    # VELOCITIES rather than decaying with the pose. Do not "fix" collection by
    # shortening episodes on the strength of an intuition this number
    # contradicts.


def main() raises:
    comptime if DOMAIN == "walker":
        run[
            DMWalkerModel,
            DMWalkerConfig[0.0],
            WideResetConfig[
                DMWalkerConfig[0.0], WALKER_ROOTZ_ADR,
                WALKER_Z_LO, WALKER_Z_HI,
            ],
        ](String("walker"))
    elif DOMAIN == "cheetah":
        run[
            DMCheetahModel,
            DMCheetahConfig,
            WideResetConfig[
                DMCheetahConfig, CHEETAH_ROOTZ_ADR,
                CHEETAH_Z_LO, CHEETAH_Z_HI,
            ],
        ](String("cheetah"))
    else:
        comptime assert False, "DOMAIN must be 'walker' or 'cheetah'"
