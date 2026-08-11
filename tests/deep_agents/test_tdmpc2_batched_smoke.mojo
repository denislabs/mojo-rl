"""TD-MPC2 BATCHED driver smoke (CPU, MPC-off) — 4 × PendulumV2.

The batched sibling of `test_tdmpc2_agent_smoke.mojo`. That one proves the
single-env stack learns; this one proves `train_batched` — N envs stepped in
lockstep, one acting pass over [N, ·], and the per-env strided replay walk —
runs end to end and still drives the world-model loss down.

Gates:
  1. `train_batched` completes and the WM loss is finite and DECREASES.
  2. The replay's strided sampler returns windows that are ONE env's
     trajectory. This is checked directly rather than through the loss,
     because a window that hops between envs still trains, still produces a
     falling loss, and is still garbage — there is no signal to read.

Gate 2 works by recording a synthetic batch where env `e`'s observations all
carry `e` in channel 0 and a per-env step counter in channel 1, then asserting
every sampled window is constant in channel 0 and consecutive in channel 1.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_batched_smoke.mojo`
"""

from std.math import isfinite
from std.random import seed
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ENC = 32
comptime ACT = 1
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 4
comptime H = 3
comptime N_ENVS = 4
comptime CAP = 4096          # multiple of N_ENVS (enforced by the driver)

comptime Ag = TDMPC2Agent[
    "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]
comptime Env = BatchedCpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]


def test_strided_windows_stay_within_one_env() raises:
    """A sampled window must be one env's own consecutive frames."""
    comptime T = 3
    comptime STEPS = 64
    var buf = SequenceReplay[OBS, ACT, CAP].new()
    buf.set_env_stride(N_ENVS)

    # Lockstep record, exactly as the driver does: env 0 … N-1 per iteration.
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var a = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    for t in range(STEPS):
        for e in range(N_ENVS):
            o[0] = Scalar[DT](e)      # env id
            o[1] = Scalar[DT](t)      # per-env step counter
            o[2] = Scalar[DT](0)
            a[0] = Scalar[DT](e)
            buf.record(
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=o[0])),
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=a[0])),
                Scalar[DT](e),
                Scalar[DT](0),
            )

    assert_true(buf.can_sample[T](), "should be able to sample")

    var ob = List[Scalar[DT]](length=B * (T + 1) * OBS, fill=Scalar[DT](0))
    var ab = List[Scalar[DT]](length=B * T * ACT, fill=Scalar[DT](0))
    var rb = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var db = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))

    for _trial in range(20):
        buf.sample_batch[B, T](
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=ob[0])),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=ab[0])),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=rb[0])),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=db[0])),
        )
        for b in range(B):
            var e0 = ob[b * (T + 1) * OBS + 0]
            var t0 = ob[b * (T + 1) * OBS + 1]
            for k in range(T + 1):
                var base = b * (T + 1) * OBS + k * OBS
                assert_true(
                    ob[base + 0] == e0,
                    "window frame came from a DIFFERENT env — the strided"
                    " walk is crossing lanes",
                )
                assert_true(
                    ob[base + 1] == t0 + Scalar[DT](k),
                    "window frames are not consecutive in time",
                )
            # Reward/action carry the env id too — same lane, all T frames.
            for k in range(T):
                assert_true(
                    rb[b * T + k] == e0, "reward came from another env"
                )
                assert_true(
                    ab[b * T * ACT + k * ACT] == e0,
                    "action came from another env",
                )
    print("  strided windows: 20 batches x", B, "rows — all single-env ✓")


def test_contiguous_path_unchanged() raises:
    """stride=1 keeps the original contiguous walk (single-env parity)."""
    comptime T = 2
    var buf = SequenceReplay[OBS, ACT, CAP].new()
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var a = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    for t in range(32):
        o[0] = Scalar[DT](t)
        buf.record(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=o[0])),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=a[0])),
            Scalar[DT](t),
            Scalar[DT](0),
        )
    var ob = List[Scalar[DT]](length=B * (T + 1) * OBS, fill=Scalar[DT](0))
    var ab = List[Scalar[DT]](length=B * T * ACT, fill=Scalar[DT](0))
    var rb = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var db = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    buf.sample_batch[B, T](
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=ob[0])),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=ab[0])),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=rb[0])),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](Pointer(to=db[0])),
    )
    for b in range(B):
        var t0 = ob[b * (T + 1) * OBS]
        for k in range(T + 1):
            assert_true(
                ob[b * (T + 1) * OBS + k * OBS] == t0 + Scalar[DT](k),
                "stride-1 windows must stay contiguous",
            )
    print("  contiguous (stride=1) walk unchanged ✓")


def test_train_batched_runs_and_learns() raises:
    """`train_batched` end to end on 4 CPU pendulums."""
    seed(7)
    var env = Env(PendulumV2[DT]())
    var ag = Ag.make(
        lr=Scalar[DT](1e-3),
        action_scale=Scalar[DT](2.0),
        learning_starts=64,
    )

    # 64 warmup + 192 learning env-steps across 4 envs = 64 iterations.
    var wm0: Scalar[DT] = 0.0
    var _best = ag.train_batched[Env, N_ENVS](
        env,
        256,
        rng_seed=UInt64(11),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    wm0 = ag.last_wm_loss()
    assert_true(isfinite(Float64(wm0)), "WM loss must be finite")
    assert_true(wm0 > Scalar[DT](0.0), "WM loss must be positive")

    var _best2 = ag.train_batched[Env, N_ENVS](
        env,
        512,
        rng_seed=UInt64(12),
        updates_per_step=1,
        print_every=0,
        verbose=False,
        base_step=256,
    )
    var wm1 = ag.last_wm_loss()
    assert_true(isfinite(Float64(wm1)), "WM loss must stay finite")
    assert_true(
        wm1 < wm0,
        "WM loss should fall with more training (wm0="
        + String(wm0) + ", wm1=" + String(wm1) + ")",
    )
    # The buffer holds one frame per env per iteration.
    assert_true(
        ag.replay.count() == 768 // N_ENVS * N_ENVS,
        "replay should hold every recorded transition",
    )
    print("  train_batched: wm", wm0, "->", wm1, " replay",
          ag.replay.count(), "✓")


def test_single_env_train_on_batched_replay_raises() raises:
    """Mixing collection modes in one replay must FAIL, not corrupt.

    A single-env `train` after `train_batched` would keep filling a ring whose
    existing frames are interleaved N ways; both halves then decode as
    nonsense. `set_env_stride` refuses instead — the whole point being that
    nothing downstream could have noticed."""
    seed(5)
    var benv = Env(PendulumV2[DT]())
    var ag = Ag.make(
        lr=Scalar[DT](1e-3),
        action_scale=Scalar[DT](2.0),
        learning_starts=10_000,   # collection only, no updates
    )
    var _b = ag.train_batched[Env, N_ENVS](
        benv, 64, rng_seed=UInt64(5), print_every=0, verbose=False
    )
    assert_true(ag.replay.count() == 64, "batched frames recorded")

    var senv = PendulumV2[DT]()
    var raised = False
    try:
        var _s = ag.train[PendulumV2[DT]](
            senv, 8, print_every=0, verbose=False
        )
    except:
        raised = True
    assert_true(
        raised,
        "single-env train on a batched replay must raise, not silently"
        " interleave two layouts in one ring",
    )
    print("  single-env train on a batched replay raises ✓")


def test_greedy_eval_batched_finite() raises:
    seed(3)
    var eval_env = Env(PendulumV2[DT]())
    var ag = Ag.make(
        lr=Scalar[DT](1e-3),
        action_scale=Scalar[DT](2.0),
        learning_starts=0,
    )
    var ret = ag.evaluate_batched[Env, N_ENVS](eval_env, max_steps=205)
    assert_true(isfinite(Float64(ret)), "eval return must be finite")
    print("  evaluate_batched return =", ret, "✓")


def main() raises:
    print("=" * 70)
    print("TD-MPC2 batched driver smoke (CPU, MPC-off) — 4 x Pendulum")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("BATCHED SMOKE PASSED")
    print("=" * 70)
