"""Dreamer 4 pixel-CarRacing — imagination accuracy GIF (GPU decode).

Loads a checkpoint from the ONLINE CarRacing run
(`dreamer4_car_racing_online.mojo`, which writes `<base>.tok.ckpt` +
`<base>.{dyn,te,ph,rh,vh,ch}.ckpt`), collects a short greedy segment, then
produces a triptych GIF, one frame per horizon step:

    [ REAL | RECON | IMAGINED ]

  * REAL      — ground-truth frame from the env (downscaled to the tokenizer's
                TGT×TGT grayscale target, newest stacked channel).
  * RECON     — tokenizer autoencode of the SAME real frame (encode→decode, MAE
                masking OFF). This is the DECODE UPPER BOUND: if RECON is blurry
                the tokenizer is the limit, not the dynamics. THE priority panel
                for answering "is the tokenizer good / undertrained?".
  * IMAGINED  — open-loop latent rollout. Seed the first NCTX context latents by
                encoding the real context frame(s), then autoregressively roll
                the FROZEN dynamics transformer forward conditioned on the
                RECORDED greedy actions (flow-matching ODE denoise, K=K_IMAG
                substeps), and decode each generated latent with the tokenizer
                decoder. This is ACTION-conditioned (recorded actions), NOT
                policy-sampled. If IMAGINED tracks REAL the world model is
                faithful; if it drifts/blurs while RECON stays sharp, the
                dynamics are the bottleneck.

Because the dynamics transformer has a fixed time window T, the imagined rollout
covers exactly one window (NCTX clean context frames + T−NCTX generated frames),
so the GIF shows up to T frames.

The dynamics latent (NSP·DSP flat) and the tokenizer bottleneck z (L·D_BOT flat)
are the same 256-D object (the online example enforces ND == L·D_BOT, NSP = L,
DSP = D_BOT), so a generated dynamics latent feeds `tok.dec` directly.

GIF encoding is pure Mojo (`save_frame_sequence_gif`) — no Python, no SDL.

Run (NVIDIA, after the online run has written a checkpoint into the cwd):
    pixi run -e nvidia mojo run -I . \\
        examples/car_racing/dreamer4_car_racing_imagination_gif.mojo
"""

from std.memory import alloc
from std.math import max
from std.random import seed
from std.gpu.host import DeviceContext, HostBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.checkpoint import load_params

from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.patchify import (
    downscale_box, temporal_patchify, temporal_unpatchify,
)
from mojo_rl.deep_agents.dreamer4.imag_rollout import _fwd_window
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao, _ilog2
from mojo_rl.deep_agents.dreamer4.online import (
    OnlineRng, _encode, _push_frame, _step_repeat,
)
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB
from mojo_rl.render.image_writer import save_frame_sequence_gif


# ── checkpoint base (overridable) ──
comptime CKPT_BASE = "dreamer4_carracing_online"
comptime GIF_PATH = "dreamer4_carracing_imagination.gif"


def main() raises:
    # ── image / patch dims (MUST match dreamer4_car_racing_online.mojo) ──
    comptime IN_CH = 4
    comptime IMG = 84                       # CarRacing PIX_RES
    comptime TGT = 64                       # tokenizer target resolution
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 64 patches
    comptime DP = PATCH * PATCH                     # 64
    comptime CAP = 100_000

    # ── sequence / batch ──
    comptime T = 10
    comptime B = 8
    comptime B_SELF = 2

    # ── tokenizer ──
    comptime L = 16
    comptime D_BOT = 16
    comptime TOK_D = 128
    comptime TOK_NH = 4
    comptime TOK_HID = 256
    comptime TOK_DEPTH = 2
    comptime DROP = 0.5
    comptime TOK_SEED = 7

    # ── agent / dynamics ──
    comptime NSP = L
    comptime DSP = D_BOT
    comptime ND = NSP * DSP                 # 256 (== tokenizer L·D_BOT)
    comptime D_DYN = 128
    comptime NH = 4
    comptime NREG = 2
    comptime HID_DYN = 256
    comptime DEPTH_DYN = 2
    comptime KMAX = 4

    # ── heads / imagination ──
    comptime NAGENT = 1
    comptime NTASK = 1
    comptime HHID = 128
    comptime NACT = 5                       # noop/left/right/gas/brake
    comptime NBINS = 41
    comptime NMTP = 2
    comptime ADIM = NACT
    comptime AHID = 2 * D_DYN
    comptime K_IMAG = 2
    comptime NCTX = 1

    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime EMAX = _ilog2(KMAX)            # cleanest sig index
    comptime K = K_IMAG                     # ODE substeps
    comptime E_K = _ilog2(K)
    comptime SCALE = KMAX // K
    comptime FRAME_REPEAT = 4

    comptime Ag = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX, "gpu",
    ]
    comptime Tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, NSP, NP, DSP, TOK_HID, TOK_DEPTH,
        DROP, DROP, TOK_SEED,
    ]
    comptime Env = CarRacingMB[DT, PIXEL_OBS=True, PIX_RES=IMG]
    comptime DYNT = Ag.DYN
    comptime AGD = Ag.AGD                    # NAGENT · D_DYN

    # ── derived batch dims for the B=1 imagination window ──
    comptime BF = T                         # B' = 1 ⇒ BF = T

    # ── triptych layout (all panels at TGT resolution) ──
    comptime SEP = 2
    comptime HC = TGT
    comptime WC = 3 * TGT + 2 * SEP

    print("=" * 70)
    print("Dreamer 4 pixel-CarRacing — imagination GIF (GPU decode)")
    print("  T", T, " NCTX", NCTX, " TGT", TGT, " IN_CH", IN_CH)
    print("=" * 70)
    seed(42)

    with DeviceContext() as ctx:
        var dctx = Optional(ctx)

        # ── build + load ────────────────────────────────────────────────
        var agent = Ag.make["cpu", Xavier](dctx)
        var tok = Tok.make["gpu", Xavier](dctx)
        print("loading checkpoint base", CKPT_BASE, "...")
        load_params["gpu"](tok, String(CKPT_BASE) + ".tok.ckpt", dctx)
        agent.load(String(CKPT_BASE), dctx)
        tok.set_mae_p(0.0, 0.0)   # DISABLE MAE masking for clean recon/encode
        print("loaded.")

        var env = Env()

        # ── collect a short greedy segment (mirrors the online eval loop) ──
        # rolling window of the last ≤T encoded obs frames (front-aligned).
        var win_patch = Tensor.alloc(T * NP * DP)
        var win_z = Tensor.alloc(T * ND)
        var fr1 = Tensor.alloc(TGT * TGT)
        var pa1 = Tensor.alloc(NP * DP)
        var win_act = List[Int](length=T, fill=-1)
        var act_hist = List[Scalar[DT]](length=T * ADIM, fill=Scalar[DT](0.0))
        var cur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))

        # per-frame display buffers (downscaled TGT×TGT, one channel)
        var real_t = Tensor.alloc(T * TGT * TGT)
        var recon_t = Tensor.alloc(T * TGT * TGT)
        var imag_t = Tensor.alloc(T * TGT * TGT)
        # recorded greedy action per step (action taken at state = frame idx)
        var ract = List[Int](length=T, fill=0)

        var ob0 = env.reset_obs_list()
        for i in range(IMG_DIM):
            cur[i] = Scalar[DT](Float64(ob0[i]))
        var win_n = 0
        var last_action = -1
        var collected = 0
        print("collecting greedy segment (up to", T, "frames)...")
        for step in range(T):
            # store the downscaled newest-channel real frame for display
            downscale_box[IMG, IMG, TGT, TGT](
                _mao(cur.unsafe_ptr() + (IN_CH - 1) * IMG * IMG),
                _mao(real_t.data.unsafe_ptr() + step * TGT * TGT),
            )
            win_n = _push_frame[IN_CH, IMG, TGT, PATCH, T, NP, DP](
                _mao(cur.unsafe_ptr()), fr1, pa1, win_patch, win_act,
                win_n, last_action,
            )
            for i in range(win_n * NP * DP, T * NP * DP):
                win_patch.data[i] = Scalar[DT](0.0)
            _encode[T, "gpu"](tok.enc, win_patch, win_z, dctx)
            for i in range(T * ADIM):
                act_hist[i] = Scalar[DT](0.0)
            for fr in range(1, win_n):
                var ap = win_act[fr]
                if ap >= 0 and ap < ADIM:
                    act_hist[fr * ADIM + ap] = Scalar[DT](1.0)
            var a = agent.act_from_latents(
                _mao(win_z.data.unsafe_ptr()), win_n,
                _mao(act_hist.unsafe_ptr()), 0, False, 0.0, dctx,
            )
            ract[step] = a
            collected = step + 1
            var rd = _step_repeat[Env, IMG_DIM](env, a, FRAME_REPEAT, cur)
            last_action = a
            if rd[1]:  # env terminated early
                print("  episode terminated early at frame", collected)
                break
        print("  collected", collected, "frames")
        if collected < NCTX + 1:
            raise Error("segment too short to imagine (need NCTX+1 frames)")

        # ── RECON: tokenizer autoencode over the whole T-window ───────────
        # (the tokenizer's causal TIME attention needs nn-batch = a multiple of
        # T, so the recon runs on the full window at once, not per-frame.)
        var recwin_patch = Tensor.alloc(T * NP * DP)
        var recwin_out = Tensor.alloc(T * NP * DP)
        print("decoding RECON (tokenizer autoencode)...")
        for i in range(T * NP * DP):
            recwin_patch.data[i] = Scalar[DT](0.0)
        for f in range(collected):
            temporal_patchify[1, 1, TGT, TGT, PATCH](
                _mao(real_t.data.unsafe_ptr() + f * TGT * TGT),
                _mao(recwin_patch.data.unsafe_ptr() + f * NP * DP),
            )
        recwin_patch.ensure_gpu(ctx, T * NP * DP)
        recwin_patch.upload(ctx)
        recwin_out.ensure_gpu(ctx, T * NP * DP)
        tok.forward["gpu", T](TensorRefs[1](recwin_patch), recwin_out, dctx)
        recwin_out.download(ctx)
        for f in range(collected):
            temporal_unpatchify[1, 1, TGT, TGT, PATCH](
                _mao(recwin_out.data.unsafe_ptr() + f * NP * DP),
                _mao(recon_t.data.unsafe_ptr() + f * TGT * TGT),
            )

        # ── IMAGINED: open-loop action-conditioned latent rollout ─────────
        # 1. encode the NCTX context frame(s) → context latent(s).
        var ctxwin_patch = Tensor.alloc(T * NP * DP)
        var ctxwin_z = Tensor.alloc(T * ND)
        for i in range(T * NP * DP):
            ctxwin_patch.data[i] = Scalar[DT](0.0)
        for c in range(NCTX):
            temporal_patchify[1, 1, TGT, TGT, PATCH](
                _mao(real_t.data.unsafe_ptr() + c * TGT * TGT),
                _mao(ctxwin_patch.data.unsafe_ptr() + c * NP * DP),
            )
        _encode[T, "gpu"](tok.enc, ctxwin_patch, ctxwin_z, dctx)

        # 2. window buffers for the B=1 frozen-dynamics rollout.
        var packed = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0.0))
        var zhat = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0.0))
        var h_host = List[Scalar[DT]](length=BF * AGD, fill=Scalar[DT](0.0))
        var act_oh = List[Scalar[DT]](length=BF * ADIM, fill=Scalar[DT](0.0))
        var act_mask = List[Scalar[DT]](length=BF * ADIM, fill=Scalar[DT](1.0))
        var sig = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.0))
        var step_l = List[Scalar[DT]](
            length=BF, fill=Scalar[DT](Float64(EMAX))
        )
        var agent_in = List[Scalar[DT]](length=BF * AGD, fill=Scalar[DT](0.0))
        var task_l = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))

        # ODE τ=0 noise seeds (deterministic gaussian).
        var noise_rng = OnlineRng(20260706)
        var znoise = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0.0))
        for i in range(BF * ND):
            znoise[i] = Scalar[DT](noise_rng.gauss())

        # boundary tensors for the dynamics forward (host + device).
        var in_t = Tensor.alloc(BF * ND)
        var out_t = Tensor.alloc(BF * ND)
        in_t.ensure_gpu(ctx, BF * ND)
        out_t.ensure_gpu(ctx, BF * ND)
        var h_ag = Optional[HostBuffer[DT]](
            ctx.enqueue_create_host_buffer[DT](BF * AGD)
        )

        # seed context frames 0..NCTX-1 clean.
        for c in range(NCTX):
            sig[c] = Scalar[DT](Float64(KMAX - 1))
            step_l[c] = Scalar[DT](Float64(EMAX))
            for i in range(ND):
                packed[c * ND + i] = ctxwin_z.data[c * ND + i]

        # task embedding → agent tokens [BF, AGD].
        agent.te.embed_into["cpu", 1, T](
            _mao(task_l.unsafe_ptr()), _mao(agent_in.unsafe_ptr()), None
        )

        var dt = 1.0 / Float64(K)
        print("rolling FROZEN dynamics open-loop on recorded actions...")
        for tgt in range(NCTX, T):
            var state_i = tgt - 1
            var a = ract[state_i]           # recorded greedy action at state i
            for aa in range(ADIM):
                act_oh[tgt * ADIM + aa] = Scalar[DT](1.0 if aa == a else 0.0)
            # seed frame tgt at τ=0 with gaussian noise.
            for kk in range(ND):
                packed[tgt * ND + kk] = znoise[tgt * ND + kk]
            step_l[tgt] = Scalar[DT](Float64(E_K))
            # flow-matching ODE denoise over K substeps.
            for isub in range(K):
                var tau = Float64(isub) / Float64(K)
                var sig_i = isub * SCALE
                sig[tgt] = Scalar[DT](Float64(sig_i))
                _fwd_window[DYNT, "gpu", BF, ND, AGD](
                    agent.dyn,
                    _mao(sig.unsafe_ptr()), _mao(step_l.unsafe_ptr()),
                    _mao(act_oh.unsafe_ptr()), _mao(act_mask.unsafe_ptr()),
                    _mao(agent_in.unsafe_ptr()),
                    _mao(packed.unsafe_ptr()), _mao(zhat.unsafe_ptr()),
                    _mao(h_host.unsafe_ptr()),
                    in_t, out_t, dctx, h_ag,
                )
                var denom = max(1e-4, 1.0 - tau)
                var fb = tgt * ND
                for kk in range(ND):
                    var x1 = Float64(zhat[fb + kk])
                    var zv = Float64(packed[fb + kk])
                    packed[fb + kk] = Scalar[DT](zv + (x1 - zv) / denom * dt)
            # frame tgt is now clean for subsequent reads.
            sig[tgt] = Scalar[DT](Float64(KMAX - 1))
            step_l[tgt] = Scalar[DT](Float64(EMAX))

        # ── decode the whole imagined window (context + generated) at once ──
        # (batch = T for the decoder's causal time attention). packed holds the
        # context latent(s) at frames 0..NCTX-1 and the generated latents after.
        var decwin_in = Tensor.alloc(T * ND)
        var decwin_out = Tensor.alloc(T * NP * DP)
        for i in range(T * ND):
            decwin_in.data[i] = packed[i]
        decwin_in.ensure_gpu(ctx, T * ND)
        decwin_in.upload(ctx)
        decwin_out.ensure_gpu(ctx, T * NP * DP)
        tok.dec.forward["gpu", T](TensorRefs[1](decwin_in), decwin_out, dctx)
        decwin_out.download(ctx)
        for f in range(collected):
            temporal_unpatchify[1, 1, TGT, TGT, PATCH](
                _mao(decwin_out.data.unsafe_ptr() + f * NP * DP),
                _mao(imag_t.data.unsafe_ptr() + f * TGT * TGT),
            )

        # ── compose [ REAL | RECON | IMAGINED ] grayscale triptych ────────
        var nrend = collected
        var comp = alloc[Scalar[DType.float32]](nrend * HC * WC)
        var sepval = Float32(0.12)
        for f in range(nrend):
            var fbase = f * HC * WC
            for p in range(HC * WC):
                comp[fbase + p] = sepval
            for y in range(TGT):
                var row = fbase + y * WC
                var rb = f * TGT * TGT + y * TGT
                for x in range(TGT):
                    comp[row + x] = Float32(real_t.data[rb + x])
                    comp[row + TGT + SEP + x] = Float32(recon_t.data[rb + x])
                    comp[row + 2 * TGT + 2 * SEP + x] = Float32(
                        imag_t.data[rb + x]
                    )

        save_frame_sequence_gif(
            GIF_PATH,
            comp,
            nrend,
            HC,
            WC,
            channels=1,
            fps=10,
            loop=True,
            vmin=0.0,
            vmax=1.0,
        )
        comp.free()

        print("=" * 70)
        print("DONE — open", GIF_PATH)
        print("  panels: [ REAL | RECON | IMAGINED ]  (TGT×TGT grayscale)")
        print("=" * 70)
        _ = env^
        _ = agent^
        _ = tok^
