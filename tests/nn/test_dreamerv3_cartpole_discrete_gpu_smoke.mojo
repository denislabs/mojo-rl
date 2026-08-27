"""DreamerV3 discrete GPU smoke — device-resident categorical actor (CartPole).

GPU counterpart of `test_dreamerv3_cartpole_discrete_smoke.mojo`. Drives the
`DreamerV3Agent["gpu", ..., DISCRETE=True]` on the real CartPole loop and
exercises the device-resident `_ac_gpu_disc` path BOTH ways: `want_diag=False`
(the fast training path — no host diagnostic readout) on most train_steps and
`want_diag=True` (the gated host readout) periodically. Confirms the fast path
runs, trains (WM decreases), select_action emits valid one-hots, and the gated
diagnostics produce finite losses.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_cartpole_discrete_gpu_smoke.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.cartpole import CartPoleEnv

comptime OBS = 4
comptime ACT = 2
comptime DETER = 64
comptime H = 16
comptime STOCH = 8
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 16
comptime DEC_U = 16
comptime HU = 16
comptime VU = 16
comptime PU = 16
comptime BINS = 51
comptime B = 4
comptime T = 8
comptime T_IMAG = 6
comptime CAP = 4096

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True, target="gpu"
]


def _is_onehot(a: Pointer[Scalar[DT], MutAnyOrigin]) -> Bool:
    var ones = 0
    var zeros = 0
    for i in range(ACT):
        if a[i] == Scalar[DT](1.0):
            ones += 1
        elif a[i] == Scalar[DT](0.0):
            zeros += 1
    return ones == 1 and zeros == ACT - 1


def _argmax(a: Pointer[Scalar[DT], MutAnyOrigin]) -> Int:
    var k = 0
    var best = a[0]
    for i in range(1, ACT):
        if a[i] > best:
            best = a[i]
            k = i
    return k


def main() raises:
    print("=" * 70)
    print("DreamerV3 discrete GPU smoke — device-resident categorical actor")
    print("=" * 70)
    seed(11)
    var ctx = DeviceContext()
    var env = CartPoleEnv[DT]()
    var ag = Ag.make(ctx=ctx, lr=Scalar[DT](1e-3), learning_starts=64, warmup_steps=0)

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

    comptime TOTAL = 260
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0
    var all_onehot = True
    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        var idx: Int
        if step < LEARN_START:
            idx = Int(random_float64() * 2.0)
            if idx >= ACT:
                idx = ACT - 1
            for a in range(ACT):
                actbuf[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
        else:
            ag.select_action(obsbuf.as_unsafe_any_origin(), actbuf, explore=True)
            if not _is_onehot(actbuf.as_unsafe_any_origin()):
                all_onehot = False
            idx = _argmax(actbuf)
        var res = env.step_obs(idx)
        ag.record(obsbuf, actbuf, res[1], Scalar[DT](1.0) if res[2] else Scalar[DT](0.0))
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            # toggle want_diag: most steps fast path, every 5th the gated readout.
            var wd = (n_train % 5 == 0)
            if ag.train_step(want_diag=wd):
                if wd:
                    var wm = ag.last_wm_loss()
                    var ac = ag.last_ac_loss()
                    assert_true(wm == wm, "WM finite")
                    assert_true(ac == ac, "AC finite")
                    if first_wm == Scalar[DT](0.0):
                        first_wm = wm
                    last_wm = wm
                n_train += 1

    print("  trained", n_train, "steps (diag every 5th); WM:", first_wm, "->", last_wm)
    assert_true(all_onehot, "GPU select_action emits valid one-hot actions")
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")

    ag.reset_belief()
    var ev_obs = env.reset_obs_list()
    var ret: Scalar[DT] = 0.0
    for _s in range(500):
        for i in range(OBS):
            obsbuf[i] = ev_obs[i]
        ag.select_greedy_action(obsbuf, actbuf)
        assert_true(_is_onehot(actbuf), "greedy action is one-hot")
        var r = env.step_obs(_argmax(actbuf))
        ret += r[1]
        ev_obs = r[0].copy()
        if r[2]:
            break
    print("  greedy eval return (1 ep) =", ret)
    assert_true(ret == ret, "eval return finite")
    obsbuf.free(); actbuf.free()
    print("=" * 70)
    print("DISCRETE GPU SMOKE PASSED — device-resident _ac_gpu_disc fast+diag paths")
    print("=" * 70)
