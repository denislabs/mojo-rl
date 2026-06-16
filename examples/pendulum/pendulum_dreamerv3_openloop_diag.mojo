"""DreamerV3 (nn) — open-loop world-model accuracy diagnostic (Pendulum, CPU).

Confirms whether the remaining non-convergence is a world-model FIDELITY
(capacity / latent-resolution) issue rather than a code bug. After training,
it builds the posterior belief from CTX real steps, then rolls HOR steps
forward two ways and compares decoded predictions to the real trajectory:

  * open-loop      — prior dynamics (`imagine`), fed the real recorded actions
                     but NO observations. This is what imagination uses.
  * teacher-forced — re-observes each real obs (posterior). The 1-step floor.

Read the printout:
  - If `ol_obs` (open-loop obs MSE) climbs steeply with horizon while `tf_obs`
    stays small  ⇒  the DYNAMICS are the bottleneck (capacity / `CLASSES`),
    not the decoder and not a bug. Imagined rollouts diverge from reality, so
    the actor optimizes a fantasy → the policy can't transfer.
  - If `ol_obs` ≈ `tf_obs` and both small ⇒ the WM predicts well; look
    elsewhere (actor/return scaling).
  - `ol_rew` shows whether predicted reward tracks the real reward downrange.

CONFIRMATION PROTOCOL: run once with CLASSES=4 (the current lighthouse value)
and once with CLASSES=32. If the open-loop curve is much flatter/lower with
CLASSES=32, the latent resolution is the wall.

The WM forward is bit-identical CPU↔GPU (see test_dreamerv3_ac_parity), so this
CPU diagnosis transfers to the GPU lighthouse.

Run:
  pixi run mojo run -I . examples/pendulum/pendulum_dreamerv3_openloop_diag.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.pendulum import PendulumV2

# ── config (fast CPU; mirrors the working CPU example except CLASSES toggle) ──
comptime OBS = 3
comptime ACT = 1
comptime DETER = 128
comptime H = 64
comptime STOCH = 32
comptime CLASSES = 4          # ◀── TOGGLE: 4 (lighthouse) vs 32 (reference). Compare the curves.
comptime BLOCKS = 8
comptime TOKEN = 64
comptime DEC_U = 64
comptime HU = 64
comptime VU = 64
comptime PU = 64
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]

comptime TOTAL = 6000
comptime LEARN_START = 600
comptime GRAD_STEPS_PER = 2
comptime PROBE_EVERY = 2500
comptime CTX = 8         # context steps to observe before open-loop
comptime HOR = 30        # open-loop horizon
comptime ACT_SCALE = Scalar[DT](2.0)


def _probe(mut ag: Ag, step: Int) raises:
    """Collect a fresh greedy episode, run the open-loop report, print curve."""
    var env = PendulumV2[DT]()
    var n = CTX + HOR
    var robs = alloc[Scalar[DT]]((n + 1) * OBS)
    var ract = alloc[Scalar[DT]](n * ACT)
    var rrew = alloc[Scalar[DT]](n)
    var ob = alloc[Scalar[DT]](OBS)
    var ac = alloc[Scalar[DT]](ACT)
    ag.reset_belief()
    var o = env.reset_obs_list()
    for t in range(n + 1):
        for i in range(OBS):
            robs[t * OBS + i] = o[i]
            ob[i] = o[i]
        if t < n:
            ag.select_greedy_action(ob, ac)          # normalized [-1,1] action
            for j in range(ACT):
                ract[t * ACT + j] = ac[j]
            var al = List[Scalar[DT]]()
            al.append(ac[0] * ACT_SCALE)             # driver scales for env
            var r = env.step_continuous_vec[DT](al)
            rrew[t] = r[1]
            o = r[0].copy()

    var ol_obs = alloc[Scalar[DT]](HOR)
    var tf_obs = alloc[Scalar[DT]](HOR)
    var ol_rew = alloc[Scalar[DT]](HOR)
    var tf_rew = alloc[Scalar[DT]](HOR)
    ag.trainer.openloop_report(
        robs, ract, rrew, CTX, HOR, ol_obs, tf_obs, ol_rew, tf_rew
    )
    print("── open-loop WM probe @ step", step, " (CTX=", CTX, " HOR=", HOR,
          " CLASSES=", CLASSES, ") ──")
    print("   h | ol_obs_MSE | tf_obs_MSE | ol_rew_err | tf_rew_err")
    for h in range(HOR):
        if h % 4 == 0 or h == HOR - 1:
            print("  ", h, " | ", ol_obs[h], " | ", tf_obs[h], " | ",
                  ol_rew[h], " | ", tf_rew[h])
    robs.free(); ract.free(); rrew.free(); ob.free(); ac.free()
    ol_obs.free(); tf_obs.free(); ol_rew.free(); tf_rew.free()


def main() raises:
    print("======================================================================")
    print("DreamerV3 open-loop WM diagnostic — CLASSES=", CLASSES, " DETER=", DETER,
          " STOCH=", STOCH, " | train", TOTAL, "steps")
    print("======================================================================")
    seed(42)
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](4e-5), learning_starts=LEARN_START, action_scale=ACT_SCALE,
    )
    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)   # normalized
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        al.append(actbuf[0] * ACT_SCALE)
        var res = env.step_continuous_vec[DT](al)
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START:
            for _g in range(GRAD_STEPS_PER):
                _ = ag.train_step()
        if step >= LEARN_START and step % PROBE_EVERY == 0:
            print("  [train] step", step, " WM=", ag.last_wm_loss(),
                  " rew_pred=", ag.dbg_rew_pred(), " real_rew=", ag.dbg_real_rew())
            _probe(ag, step)
            obs = env.reset_obs_list()
            ag.reset_belief()

    print("── FINAL probe ──")
    _probe(ag, TOTAL)
    obsbuf.free(); actbuf.free()
