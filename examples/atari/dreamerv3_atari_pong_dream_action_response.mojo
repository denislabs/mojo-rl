"""DreamerV3 Atari Pong — DREAM action-response probe (counterfactual).

`tests/atari/diag_pong_action_response.mojo` checks the REAL env: hold each
action, watch the agent paddle move. This probe asks the same question of the
WORLD MODEL: seed the posterior belief from CTX real frames, then roll the
RSSM PRIOR forward holding each of the 6 actions constant (no observations),
decode every imagined latent, and track the Y-centroid of the imagined RIGHT
(agent) paddle per branch.

Why this is the decisive Pong diagnostic: the actor learns EXCLUSIVELY from
imagined rollouts. If the imagined paddle does not respond to the action fed
into the dynamics, the advantage w.r.t. the action is flat and the policy can
never improve — eval stays -21 forever even with a pixel-perfect imagination
of ball/opponent (which only proves the ACTION-INDEPENDENT dynamics).

  * spread < ~3 px  → WM action-conditioning is the gap (actor is blind).
  * spread ≥ ~3 px  → WM is fully functional; look at the AC side
                      (policy collapse / exploration / training budget).

Also writes a 6-panel GIF (one panel per held action) for visual confirmation.

Run (checkpoint from the training run; C below MUST match it):
    pixi run -e apple  mojo run -I . examples/atari/dreamerv3_atari_pong_dream_action_response.mojo
    pixi run -e nvidia mojo run -I . examples/atari/dreamerv3_atari_pong_dream_action_response.mojo
"""

from std.memory import alloc
from std.random import seed
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
from mojo_rl.render.image_writer import save_frame_sequence_gif

# ── arch (MUST match the training run that WROTE the checkpoint) ──
# C=1 ↔ AtariEnv OBS_MODE=3 (gray-96 single frame); C=4 ↔ OBS_MODE=4 (stack).
comptime C = 1
comptime IMG = 96
comptime BASE = 48
comptime OBS = C * IMG * IMG
comptime ACT = 6
comptime DETER = 2048
comptime H = 256
comptime STOCH = 32
comptime CLASSES = 32
comptime BLOCKS = 8
comptime TOKEN = 1024
comptime DEC_U = 1024
comptime HU = 256
comptime VU = 256
comptime PU = 256
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 256  # never trains here → tiny replay (params load from ckpt)

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "gpu",
    OBS,
    ACT,
    DETER,
    H,
    STOCH,
    CLASSES,
    BLOCKS,
    TOKEN,
    DEC_U,
    HU,
    VU,
    PU,
    BINS,
    B,
    T,
    T_IMAG,
    CAP,
    True,  # DISCRETE=True
    ENC,
    DEC,
    RECON_SIGMOID=True,  # must match training (decode = sigmoid)
]
comptime OBS_MODE = 3 if C == 1 else 4
comptime Env = AtariEnv[OBS_MODE, DT]

comptime CHECKPOINT_PATH = "dreamerv3_atari_pong_gpu.ckpt"
comptime GIF_PATH = "dreamerv3_atari_pong_dream_actions.gif"

comptime CTX = 8  # real context frames to seed the belief
comptime HOR = 20  # imagination horizon per action branch
comptime NEWCH = (C - 1) * IMG * IMG  # newest-frame offset within OBS
comptime NACT = 6  # branches: each Pong action held constant

# GIF layout: one panel per action branch
comptime SEP = 2
comptime WC = NACT * IMG + (NACT - 1) * SEP
comptime HC = IMG


def _centroid_y(
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin], x0: Int, x1: Int
) -> Tuple[Scalar[DT], Int]:
    """Weighted Y centroid of bright (>0.5) pixels in column band [x0,x1),
    rows [19,92) — EXCLUDES the top wall (rows 14-18) and bottom wall (92-95),
    which are always-bright in this band and would dominate the paddle's
    ~6-12 px (measured via the gray-96 band scan). The ball crossing the band
    adds ≤2 px of noise. n==0 → paddle out of band / vanished."""
    var sy = Scalar[DT](0.0)
    var n = 0
    for y in range(19, 92):
        for x in range(x0, x1):
            if obs[y * IMG + x] > Scalar[DT](0.5):
                sy += Scalar[DT](y)
                n += 1
    if n == 0:
        return (Scalar[DT](-1.0), 0)
    return (sy / Scalar[DT](n), n)


def _argmax(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Int:
    var best = 0
    var bv = p[0]
    for i in range(1, n):
        if p[i] > bv:
            bv = p[i]
            best = i
    return best


def main() raises:
    print("=" * 70)
    print("DreamerV3 Pong — DREAM action response (counterfactual imagination)")
    print("  CTX", CTX, " HOR", HOR, " per-branch; OBS", OBS)
    print("=" * 70)
    seed(42)

    with DeviceContext() as ctx:
        var agent = Ag.make(ctx=ctx)
        print("loading checkpoint", CHECKPOINT_PATH, "...")
        agent.load(CHECKPOINT_PATH)

        # ── collect CTX real steps for the belief context (sampled actor) ──
        var env = Env(AtariGame.PONG)
        var robs = alloc[Scalar[DT]](
            (CTX + HOR + 1) * OBS
        ).as_unsafe_any_origin()
        var ract = alloc[Scalar[DT]]((CTX + HOR) * ACT).as_unsafe_any_origin()
        var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
        # zero-fill: the teacher-forced panel of openloop_decode_gpu reads obs
        # beyond CTX — we ignore that output, but keep the reads defined.
        for i in range((CTX + HOR + 1) * OBS):
            robs[i] = 0.0
        for i in range((CTX + HOR) * ACT):
            ract[i] = 0.0

        agent.reset_belief()
        var obs = env.reset_obs_list()
        for i in range(OBS):
            robs[i] = obs[i]
        print("collecting", CTX, "context steps...")
        for t in range(CTX):
            for i in range(OBS):
                ob[i] = obs[i]
            agent.select_action(ob, ac, explore=True)
            for a in range(ACT):
                ract[t * ACT + a] = ac[a]
            var res = env.step_obs(_argmax(ac, ACT))
            obs = res[0].copy()
            for i in range(OBS):
                robs[(t + 1) * OBS + i] = obs[i]
            if res[2]:
                raise Error("episode ended inside the context window — rerun")

        # real paddle position at the end of the context (the branch origin).
        var oc = _centroid_y(ob, 80, 92)
        print("  agent-paddle Y at branch origin =", oc[0], "(n=", oc[1], ")")

        # ── per-action branch: force the action from the last context step on ──
        var ol = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        var tf = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        var frames = alloc[Scalar[DT]](NACT * HOR * OBS).as_unsafe_any_origin()
        var endy = List[Scalar[DT]]()
        print("-" * 70)
        print("  per-step imagined right-paddle Y (walls excluded; -1 = out of")
        print("  band / not detected). Real-env mapping: 2/4 = UP, 3/5 = DOWN.")
        for a in range(NACT):
            # openloop_decode_gpu rolls the prior on ract[(CTX-1+h)] — force the
            # branch action from index CTX-1 onward (one-hot), keep the real
            # context actions before it.
            for t in range(CTX - 1, CTX + HOR):
                for k in range(ACT):
                    ract[t * ACT + k] = (
                        Scalar[DT](1.0) if k == a else Scalar[DT](0.0)
                    )
            agent.trainer.openloop_decode_gpu(robs, ract, CTX, HOR, ol, tf)
            for i in range(HOR * OBS):
                frames[a * HOR * OBS + i] = ol[i]
            print("  action", a, ":")
            var last = Scalar[DT](-1.0)
            for h in range(HOR):
                var ch = _centroid_y(ol + h * OBS + NEWCH, 80, 92)
                print("      h=", h, "  y=", ch[0], "  n=", ch[1])
                if ch[0] >= Scalar[DT](0.0):
                    last = ch[0]
            endy.append(last)  # last DETECTED position along the branch

        # spread of the FINAL imagined paddle position across action branches.
        var mn = Scalar[DT](1e9)
        var mx = Scalar[DT](-1e9)
        var seen = 0
        for i in range(len(endy)):
            if endy[i] >= Scalar[DT](0.0):
                if endy[i] < mn:
                    mn = endy[i]
                if endy[i] > mx:
                    mx = endy[i]
                seen += 1
        print("-" * 70)
        if seen < 2:
            print("  paddle not detected in the decoded dream (n<2 branches)")
            print("  ⇒ the imagined paddle VANISHES → WM drops the agent sprite.")
        else:
            var spread = mx - mn
            print("  imagined agent-paddle Y spread across actions =", spread, "px")
            if spread < Scalar[DT](3.0):
                print("  → imagined paddle does NOT respond to the action input")
                print("    ⇒ WM action-conditioning is the gap; the actor is blind.")
            else:
                print("  → imagined paddle RESPONDS to actions (WM fully functional)")
                print("    ⇒ look at the AC side: policy collapse / exploration /")
                print("      training budget — not the world model.")

        # ── 6-panel GIF: [ NOOP | a1 | a2 | a3 | a4 | a5 ] imagined frames ──
        var comp = alloc[Scalar[DType.float32]](HOR * HC * WC)
        var sepval = Float32(0.12)
        for h in range(HOR):
            var fbase = h * HC * WC
            for p in range(HC * WC):
                comp[fbase + p] = sepval
            for a in range(NACT):
                var pbase = a * (IMG + SEP)
                for y in range(IMG):
                    var row = fbase + y * WC + pbase
                    var sb = a * HOR * OBS + h * OBS + NEWCH + y * IMG
                    for x in range(IMG):
                        comp[row + x] = Float32(frames[sb + x])

        save_frame_sequence_gif(
            GIF_PATH,
            comp,
            HOR,
            HC,
            WC,
            channels=1,
            fps=6,
            loop=True,
            vmin=0.0,
            vmax=1.0,
        )

        ob.free()
        ac.free()
        robs.free()
        ract.free()
        ol.free()
        tf.free()
        frames.free()
        comp.free()
        print("=" * 70)
        print("DONE — open ", GIF_PATH)
        print("  panels: one per held action 0..5 (same order as the env diag)")
        print("=" * 70)
        _ = env^
        _ = agent^
