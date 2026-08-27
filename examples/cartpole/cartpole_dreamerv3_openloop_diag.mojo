"""DreamerV3 (nn) — open-loop world-model fidelity diagnostic (CartPole, CPU).

Follow-up to the convergence investigation. The CartPole lighthouse does not
solve because the actor gets no advantage signal: imagined returns collapse to
a near-constant (value head → constant, ret_sd≈0) while the latent stays
healthy (feat_sd≈0.40). The leading explanation is that imagination
OVER-SURVIVES — imagined rollouts don't reproduce the pole falling, so every
state looks equally valuable.

This probe tests that directly. After training, it collects a real greedy
episode that TERMINATES (the pole falls at frame L), seeds the posterior belief
from CTX real steps, then rolls forward to the terminal frame two ways and
prints the decoded POLE ANGLE (obs[2]) + the continue-head probability:

  * open-loop      — prior dynamics (`imagine`), fed the real recorded actions
                     but NO observations. This is what imagination uses.
  * teacher-forced — re-observes each real obs (posterior). The 1-step floor.

Read the printout (CartPole fails when |pole angle| ≳ 0.2095 rad):
  - If the REAL pole angle grows toward ±0.21 (the fall) but the OPEN-LOOP
    angle stays small AND ol_con stays ≈1 ⇒ imagination does NOT reproduce
    termination → the dynamics are the bottleneck (model-exploitation gap),
    not a code bug. This explains the advantage collapse.
  - If open-loop tracks the real fall (angle grows, ol_con drops) ⇒ the WM is
    faithful; look elsewhere (return scaling / actor).

Run (CPU):
  pixi run mojo run -I . examples/cartpole/cartpole_dreamerv3_openloop_diag.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.cartpole import CartPoleEnv

# ── config: mirrors examples/cartpole/cartpole_dreamerv3_nn.mojo ──
comptime OBS = 4
comptime ACT = 2
comptime DETER = 128
comptime H = 32
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 10
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
    OUT_INIT=Kaiming,  # full reward/critic output init (positive-reward optimism)
]

comptime TOTAL = 20_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime PROBE_EVERY = 10_000
comptime POLE_IDX = 2          # CartPole obs = [cart_x, cart_v, pole_angle, pole_v]
comptime CTX = 4               # context steps observed before the open-loop roll
comptime HORCAP = 80           # max stored episode length / horizon buffer size


def _argmax(a: Pointer[Scalar[DT], MutAnyOrigin]) -> Int:
    var k = 0
    var best = a[0]
    for i in range(1, ACT):
        if a[i] > best:
            best = a[i]
            k = i
    return k


def _probe(mut ag: Ag, step: Int) raises:
    """Collect a terminating greedy episode (pole falls), then open-loop trace
    through the fall and print pole-angle + continue probability."""
    var env = CartPoleEnv[DT]()
    var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    # buffers for the chosen episode (frames 0..L, actions 0..L-1)
    var robs = alloc[Scalar[DT]]((HORCAP + 1) * OBS).as_unsafe_any_origin()
    var ract = alloc[Scalar[DT]](HORCAP * ACT).as_unsafe_any_origin()
    var best_len = 0
    var best_terminated = False

    # try several greedy episodes; keep the LONGEST terminating one that fits
    # (longest → horizon reaches deepest toward the fall; must terminate so the
    #  window actually spans the failure).
    var tmp_obs = alloc[Scalar[DT]]((HORCAP + 1) * OBS)
    var tmp_act = alloc[Scalar[DT]](HORCAP * ACT)
    for _ep in range(60):
        ag.reset_belief()
        var o = env.reset_obs_list()
        for i in range(OBS):
            tmp_obs[i] = o[i]
        var L = 0
        var term = False
        for t in range(HORCAP):
            for i in range(OBS):
                ob[i] = o[i]
            ag.select_greedy_action(ob, ac)
            var idx = _argmax(ac)
            for a in range(ACT):
                tmp_act[t * ACT + a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
            var r = env.step_obs(idx)
            for i in range(OBS):
                tmp_obs[(t + 1) * OBS + i] = r[0][i]
            L = t + 1
            if r[2]:
                term = True
                break
        # prefer a terminating episode; among those, the longest
        var better = (term and not best_terminated) or (
            term == best_terminated and L > best_len
        )
        if better:
            best_len = L
            best_terminated = term
            for i in range((L + 1) * OBS):
                robs[i] = tmp_obs[i]
            for i in range(L * ACT):
                ract[i] = tmp_act[i]

    var eff_hor = best_len - CTX
    print("── open-loop WM trace @ step", step, " (episode_len=", best_len,
          " terminated=", best_terminated, " CTX=", CTX, " hor=", eff_hor, ") ──")
    if eff_hor < 2:
        print("   episodes too short to trace (need len >= CTX+2). skip.")
        ob.free(); ac.free(); robs.free(); ract.free()
        tmp_obs.free(); tmp_act.free()
        return

    var ol_obs = alloc[Scalar[DT]](HORCAP * OBS).as_unsafe_any_origin()
    var tf_obs = alloc[Scalar[DT]](HORCAP * OBS).as_unsafe_any_origin()
    var ol_con = alloc[Scalar[DT]](HORCAP).as_unsafe_any_origin()
    var tf_con = alloc[Scalar[DT]](HORCAP).as_unsafe_any_origin()
    ag.trainer.openloop_trace(
        robs, ract, CTX, eff_hor, ol_obs, tf_obs, ol_con, tf_con
    )
    print("   h | real_pole | ol_pole | tf_pole | ol_con | tf_con")
    for h in range(eff_hor):
        var real_pole = robs[(CTX + h) * OBS + POLE_IDX]
        print(
            "  ", h, " | ", real_pole, " | ", ol_obs[h * OBS + POLE_IDX],
            " | ", tf_obs[h * OBS + POLE_IDX], " | ", ol_con[h], " | ", tf_con[h],
        )
    ob.free(); ac.free(); robs.free(); ract.free()
    tmp_obs.free(); tmp_act.free()
    ol_obs.free(); tf_obs.free(); ol_con.free(); tf_con.free()


def main() raises:
    print("=" * 70)
    print("DreamerV3 CartPole open-loop WM fidelity diagnostic — train", TOTAL, "steps")
    print("=" * 70)
    seed(42)
    var env = CartPoleEnv[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](3e-4), learning_starts=LEARN_START, warmup_steps=500,
    )
    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

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
            ag.select_action(obsbuf, actbuf, explore=True)
            idx = _argmax(actbuf)
        var res = env.step_obs(idx)
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            for i in range(OBS):
                obsbuf[i] = res[0][i]
            ag.record_terminal(obsbuf)
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()
        if step > 0 and step % PROBE_EVERY == 0:
            print("  [train] step", step, " WM=", ag.last_wm_loss(),
                  " real_rew=", ag.dbg_real_rew(), " con_m=", ag.dbg_con_mean(),
                  " val_sd=", ag.dbg_val_std())
            _probe(ag, step)
            obs = env.reset_obs_list()
            ag.reset_belief()

    print("── FINAL probe ──")
    _probe(ag, TOTAL)
    obsbuf.free(); actbuf.free()
