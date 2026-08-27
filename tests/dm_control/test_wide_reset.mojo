"""Coverage gate for `WideResetConfig` — does the widened reset widen anything?

`docs/BFM_ZERO_SHOT_RL.md` component 1: "compute coverage metrics BEFORE
training FB, otherwise you discover the dataset is poor after 2 M gradient
steps." This is the smallest such metric, applied to the one lever that is
supposed to produce it.

The failure this guards against is specific and silent. `WideResetConfig`
writes `qpos[ROOT_Z_ADR]` — an index handed in as a compile-time parameter. Get
it wrong and the write lands on some other coordinate: nothing raises, the
episodes still run, and the dataset comes out with exactly the narrow height
distribution it was created to avoid. So the assertion is not "the config was
constructed" but "the torso height actually spreads, and the baseline actually
does not".

Both halves matter. A gate that only checked the wide config would pass if the
BASE reset had been spreading heights all along, which would mean the wrapper
buys nothing.

Run with:
    pixi run mojo run -I . tests/dm_control/test_wide_reset.mojo
"""

from std.math import sqrt
from std.random import seed
from std.testing import assert_true

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig
from mojo_rl.envs.dm_control.walker.walker_xml import TORSO_BODY_IDX
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel, DMCheetahConfig
from mojo_rl.envs.dm_control.cheetah.cheetah_xml import (
    TORSO_BODY_IDX as CHEETAH_TORSO_BODY_IDX,
)
from mojo_rl.envs.dm_control.wide_reset import (
    WideResetConfig,
    WALKER_ROOTZ_ADR,
    WALKER_Z_LO,
    WALKER_Z_HI,
    WALKER_TORSO_NOMINAL_Z,
    CHEETAH_ROOTZ_ADR,
    CHEETAH_Z_LO,
    CHEETAH_Z_HI,
    CHEETAH_TORSO_NOMINAL_Z,
)


comptime N_RESETS: Int = 400
comptime SEED: Int = 20260805

comptime NarrowCfg = DMWalkerConfig[0.0]
comptime WideCfg = WideResetConfig[DMWalkerConfig[0.0], WALKER_ROOTZ_ADR]

comptime NarrowEnv = Phyics3dEnv[
    DMWalkerModel, NarrowCfg, DType.float64, False
]
comptime WideEnv = Phyics3dEnv[DMWalkerModel, WideCfg, DType.float64, False]


struct Spread(Movable & Deinitable):
    var lo: Float64
    var hi: Float64
    var mean: Float64
    var std: Float64

    def __init__(out self, lo: Float64, hi: Float64, mean: Float64, std: Float64):
        self.lo = lo
        self.hi = hi
        self.mean = mean
        self.std = std

    def __init__(out self, *, deinit move: Self):
        self.lo = move.lo
        self.hi = move.hi
        self.mean = move.mean
        self.std = move.std


def _height_spread[
    CONFIG: Phyics3dEnvConfig
](label: String) raises -> Spread:
    """Torso height over N_RESETS resets.

    Read from `d.xpos` — the FK product the reward reads — not from `qpos`.
    Reading back the coordinate we just wrote would confirm the write and
    nothing else; going through FK is what shows the height reached the body
    the task cares about.
    """
    comptime Env = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]
    seed(SEED)
    var env = Env()

    var lo = Float64(1e30)
    var hi = Float64(-1e30)
    var s = Float64(0)
    var s2 = Float64(0)
    for _ in range(N_RESETS):
        _ = env.reset()
        var h = Float64(env.d.xpos.data[TORSO_BODY_IDX * 3 + 2])
        if h < lo:
            lo = h
        if h > hi:
            hi = h
        s += h
        s2 += h * h
    var n = Float64(N_RESETS)
    var mean = s / n
    var var_ = s2 / n - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "   ", label, " torso height: [", lo, ",", hi, "]  mean", mean,
        " std", sqrt(var_),
    )
    return Spread(lo, hi, mean, sqrt(var_))


def test_baseline_is_narrow() raises:
    print("[1] suite reset (baseline) ...")
    var narrow = _height_spread[NarrowCfg]("narrow")
    # The suite reset draws the limb pose and the torso PITCH, so the torso
    # height is not literally constant — a folded, pitched walker sits lower.
    # What it never does is put the torso a metre up or flat on the ground,
    # and that is what the bound below pins.
    assert_true(
        narrow.hi - narrow.lo < 0.6,
        "the baseline walker reset already spreads torso height by "
        + String(narrow.hi - narrow.lo) + " m. If that is genuinely the case,"
        " WideResetConfig buys nothing for this domain and the comparison"
        " below is meaningless — re-derive the lever before trusting it.",
    )


def test_wide_reset_spreads_height() raises:
    print("[2] widened reset ...")
    var narrow = _height_spread[NarrowCfg]("narrow")
    var wide = _height_spread[WideCfg]("wide  ")

    assert_true(
        wide.std > 0.2,
        "widened reset std is only " + String(wide.std) + " m — ROOT_Z_ADR ("
        + String(WALKER_ROOTZ_ADR) + ") probably does not name the root height"
        " coordinate, in which case the write landed on another joint and"
        " nothing raised. (An assertion phrased RELATIVE to the baseline would"
        " not catch this: the baseline std is exactly 0, so any positive"
        " spread beats any multiple of it.)",
    )

    # ⚠ The band, not just its width. `rootz` is a joint coordinate offset from
    # the torso's XML pos, so a range that looks right in joint space can put
    # the walker between 1.4 m and 2.8 m — a perfect-width distribution of
    # nothing but free fall. That was the real first version of this config,
    # and only an ABSOLUTE bound on the world height caught it.
    var want_lo = WALKER_TORSO_NOMINAL_Z + WALKER_Z_LO
    var want_hi = WALKER_TORSO_NOMINAL_Z + WALKER_Z_HI
    assert_true(
        wide.lo < want_lo + 0.15,
        "widened torso height never gets below " + String(wide.lo)
        + " m (wanted ~" + String(want_lo) + "). The 'lying down' half of the"
        " distribution is missing.",
    )
    assert_true(
        wide.hi > want_hi - 0.15,
        "widened torso height never exceeds " + String(wide.hi)
        + " m (wanted ~" + String(want_hi) + ")",
    )
    # And it must actually cover STANDING — a dataset that never sees the
    # walker upright cannot support a `stand` reward at zero shot.
    assert_true(
        wide.lo < 1.2 and wide.hi > 1.2,
        "the widened band [" + String(wide.lo) + ", " + String(wide.hi)
        + "] does not contain _STAND_HEIGHT = 1.2",
    )
    print("      std ", narrow.std, "->", wide.std, "  band contains 1.2  OK")


def test_reward_still_matches_base() raises:
    """The wrapper must forward the REWARD untouched.

    A wrapper that silently changed the reward would be far worse than one that
    failed to widen: every downstream number — the zero-shot return, the SAC
    baseline — would be computed against a task that is not walker-stand.
    """
    print("[3] wrapper forwards the reward ...")
    seed(SEED)
    var a = NarrowEnv()
    _ = a.reset()
    seed(SEED)
    var b = WideEnv()
    _ = b.reset()

    # Score the SAME state under both configs. Identical qpos/qvel in, so any
    # difference is the wrapper's doing.
    var q = List[Float64]()
    for i in range(DMWalkerModel.NQ):
        q.append(0.13 * Float64(i) - 0.4)
    q[WALKER_ROOTZ_ADR] = 1.05
    var v = List[Float64]()
    for i in range(DMWalkerModel.NV):
        v.append(0.07 * Float64(i) - 0.2)
    var act = List[Float64]()
    for i in range(DMWalkerModel.ACTION_DIM):
        act.append(0.3 - 0.1 * Float64(i))

    var ra = a.reward_at(q, v, act)
    var rb = b.reward_at(q, v, act)
    assert_true(
        Float64(ra[0]) == Float64(rb[0]),
        "WideResetConfig changed the reward: base " + String(Float64(ra[0]))
        + " vs wrapped " + String(Float64(rb[0])),
    )
    assert_true(
        Float64(ra[0]) > 1e-6,
        "the probe state scores ~0 under both configs, so an equality check"
        " between them proves nothing — pick a state the reward responds to",
    )
    print("      identical reward", Float64(ra[0]), " OK")


def test_cheetah_uses_its_own_root_index() raises:
    """Cheetah declares `rootx, rootz, rooty`; walker declares `rootz, rootx,
    rooty`. The height coordinate is therefore at a DIFFERENT index, and the
    only thing that catches a copy-pasted `WALKER_ROOTZ_ADR` is measuring the
    world height of the body the reward reads.
    """
    print("[4] cheetah: ROOT_Z_ADR =", CHEETAH_ROOTZ_ADR, "(walker uses",
          WALKER_ROOTZ_ADR, ") ...")
    comptime CheetahWide = WideResetConfig[
        DMCheetahConfig, CHEETAH_ROOTZ_ADR, CHEETAH_Z_LO, CHEETAH_Z_HI
    ]
    comptime CEnv = Phyics3dEnv[
        DMCheetahModel, CheetahWide, DType.float64, False
    ]
    assert_true(
        CHEETAH_ROOTZ_ADR != WALKER_ROOTZ_ADR,
        "cheetah and walker resolved to the same root index — one of them is"
        " wrong; the two models declare their root joints in different orders",
    )

    seed(SEED)
    var env = CEnv()
    var lo = Float64(1e30)
    var hi = Float64(-1e30)
    for _ in range(N_RESETS):
        _ = env.reset()
        var h = Float64(env.d.xpos.data[CHEETAH_TORSO_BODY_IDX * 3 + 2])
        if h < lo:
            lo = h
        if h > hi:
            hi = h
    var want_lo = CHEETAH_TORSO_NOMINAL_Z + CHEETAH_Z_LO
    var want_hi = CHEETAH_TORSO_NOMINAL_Z + CHEETAH_Z_HI
    print("       cheetah torso height: [", lo, ",", hi, "]  wanted ~[",
          want_lo, ",", want_hi, "]")
    assert_true(
        lo < want_lo + 0.15 and hi > want_hi - 0.15,
        "cheetah torso height spans [" + String(lo) + ", " + String(hi)
        + "] but the configured band is [" + String(want_lo) + ", "
        + String(want_hi) + "]. If the span is right but SHIFTED, ROOT_Z_ADR"
        " is writing the wrong coordinate.",
    )


def main() raises:
    test_baseline_is_narrow()
    test_wide_reset_spreads_height()
    test_reward_still_matches_base()
    test_cheetah_uses_its_own_root_index()
    print("\n[PASS] widened reset gate")
