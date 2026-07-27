"""Diagnostic: can the DreamerV3 world-model PRIOR overfit a tiny fixed Pong
dataset? Decides "bug vs config" for the imagination collapse.

Collects ONE short Pong episode (OBS_MODE=3, gray-96, C=1) into a tiny WM, trains
the real GPU train_step on it many times (overfit), then:
  1. reports whether recon (obs_loss) falls → decoder/posterior can memorize.
  2. reports whether dyn_kl falls toward the free-bits floor (~1 nat) → the PRIOR
     can learn even MEMORIZED (near-deterministic) transitions.
  3. runs open-loop decode on the memorized sequence and reports IMAGINED-vs-REAL
     MSE per horizon step vs the RECON (teacher-forced) control.

Interpretation:
  * recon→low, dyn_kl→~1, open-loop reproduces  ⇒ machinery CORRECT → the full
    failure is capacity/signal (single-frame) → fix = frame-stacking (C=4).
  * recon→low BUT dyn_kl stuck high AND open-loop collapses on MEMORIZED data
    ⇒ real BUG in the dyn-loss / prior / imagination path (hunt it).

Tiny model — memorizing one episode's dynamics needs little capacity, so a stuck
dyn_kl here is NOT a capacity excuse; it isolates the prior.

Run: pixi run -e apple  mojo run -I . tests/atari/diag_pong_wm_overfit.mojo
     pixi run -e nvidia mojo run -I . tests/atari/diag_pong_wm_overfit.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from mojo_rl.envs.atari import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame

# ── tiny model (real CNN pixel path; small so 1500 steps run in minutes) ──
comptime C = 1
comptime IMG = 96
comptime BASE = 16
comptime OBS = C * IMG * IMG  # 9216
comptime ACT = 6
comptime DETER = 256
comptime H = 64
comptime STOCH = 16
comptime CLASSES = 16
comptime BLOCKS = 4
comptime TOKEN = 128
comptime DEC_U = 128
comptime HU = 64
comptime VU = 64
comptime PU = 64
comptime BINS = 51
comptime B = 4
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 512

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True, ENC, DEC, RECON_SIGMOID=True,
]
comptime Env = AtariEnv[3, DT]

comptime N_COLLECT = 220  # transitions from one episode (fixed overfit set)
comptime N_TRAIN = 800  # overfit gradient steps (enough to see recon+dyn_kl trend)
comptime CTX = 5
comptime HOR = 20


def main() raises:
    seed(0)
    print("=" * 66)
    print("DreamerV3 Pong WM overfit — can the PRIOR learn memorized dynamics?")
    print("  tiny model  BASE", BASE, " DETER", DETER, " B", B, " T", T)
    print("=" * 66)

    with DeviceContext() as ctx:
        var agent = Ag.make(
            ctx=ctx, lr=Scalar[DT](3e-4), learning_starts=0, warmup_steps=0
        )
        var env = Env(AtariGame.PONG)

        var robs = alloc[Scalar[DT]]((N_COLLECT + 1) * OBS).as_unsafe_any_origin()
        var ract = alloc[Scalar[DT]](N_COLLECT * ACT).as_unsafe_any_origin()
        var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

        # ── collect ONE episode with random actions (guarantees ball + paddle
        # motion) → the fixed overfit dataset ──
        agent.reset_belief()
        var obs = env.reset_obs_list()
        var collected = 0
        for t in range(N_COLLECT):
            for i in range(OBS):
                ob[i] = obs[i]
                robs[t * OBS + i] = obs[i]
            var idx = Int(random_float64() * Float64(ACT))
            if idx >= ACT:
                idx = ACT - 1
            for a in range(ACT):
                ac[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
                ract[t * ACT + a] = ac[a]
            var res = env.step_obs(idx)
            agent.record(
                ob, ac, res[1].cast[DT](),
                Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
            )
            obs = res[0].copy()
            collected = t + 1
            if res[2]:
                for i in range(OBS):
                    ob[i] = res[0][i].cast[DT]()
                agent.record_terminal(ob)
                break
        for i in range(OBS):
            robs[collected * OBS + i] = obs[i]
        print("collected", collected, "transitions (one episode)")
        if collected < CTX + HOR + 1:
            raise Error("episode too short for open-loop check")

        # ── overfit: train the real GPU WM+AC train_step on this fixed set ──
        print("-" * 66)
        print("overfitting", N_TRAIN, "steps (recon should fall; watch dyn_kl):")
        for step in range(N_TRAIN):
            _ = agent.train_step(want_diag=True)
            if step % 100 == 0 or step == N_TRAIN - 1:
                print(
                    "  step", step,
                    " wm=", agent.last_wm_loss(),
                    " obs_loss=", agent.dbg_obs_loss(),
                    " dyn_kl=", agent.dbg_dyn_kl(),
                )
        var final_obs_loss = agent.dbg_obs_loss()
        var final_dyn_kl = agent.dbg_dyn_kl()

        # ── open-loop reproduction on the MEMORIZED sequence ──
        print("-" * 66)
        print("open-loop decode on the memorized sequence (CTX", CTX, "HOR", HOR, "):")
        var ol = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        var tf = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        agent.trainer.openloop_decode_gpu(robs, ract, CTX, HOR, ol, tf)

        var ol_h1 = Scalar[DT](0.0)
        var ol_hn = Scalar[DT](0.0)
        var tf_hn = Scalar[DT](0.0)
        print("  h   IMAGINED_mse   RECON_mse   (vs REAL, summed over 9216 px)")
        for h in range(HOR):
            var ol_mse = Scalar[DT](0.0)
            var tf_mse = Scalar[DT](0.0)
            var rb = (CTX + h) * OBS
            var ob_ = h * OBS
            for k in range(OBS):
                var d1 = ol[ob_ + k] - robs[rb + k]
                var d2 = tf[ob_ + k] - robs[rb + k]
                ol_mse += d1 * d1
                tf_mse += d2 * d2
            if h == 1:
                ol_h1 = ol_mse
            if h == HOR - 1:
                ol_hn = ol_mse
                tf_hn = tf_mse
            if h < 6 or h == HOR - 1:
                print("  ", h, "  ", ol_mse, "   ", tf_mse)

        # ── verdict ──
        print("=" * 66)
        print("SUMMARY")
        print("  final obs_loss (recon) =", final_obs_loss)
        print("  final dyn_kl           =", final_dyn_kl, " (free-bits floor ~1.0)")
        print("  open-loop IMAGINED mse  h1 =", ol_h1, "  h", HOR - 1, "=", ol_hn)
        print("  RECON mse (control)     h", HOR - 1, "=", tf_hn)
        print("-" * 66)
        if final_obs_loss > Scalar[DT](50.0):
            print("  → recon did NOT overfit — decoder/capacity issue (unexpected).")
        elif final_dyn_kl > Scalar[DT](2.0):
            print("  → recon overfit BUT dyn_kl stuck high on MEMORIZED data:")
            print("    the PRIOR cannot learn even memorized transitions → BUG signal.")
        elif ol_hn > Scalar[DT](5.0) * (ol_h1 + Scalar[DT](1e-3)):
            print("  → dyn_kl fell but open-loop still collapses on memorized data:")
            print("    imagination-rollout path suspect → BUG signal.")
        else:
            print("  → recon overfit, dyn_kl fell, open-loop reproduces memorized")
            print("    dynamics → machinery CORRECT → full failure = signal/capacity")
            print("    (single-frame) → fix = frame-stacking (C=4).")
        print("=" * 66)

        ob.free()
        ac.free()
        robs.free()
        ract.free()
        ol.free()
        tf.free()
        env.close()
        _ = env^
        _ = agent^
