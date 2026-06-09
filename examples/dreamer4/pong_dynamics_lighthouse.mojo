"""Dreamer 4 dynamics lighthouse — learn to predict the next Pong frame (CPU).

    pixi run mojo run -I . examples/dreamer4/pong_dynamics_lighthouse.mojo

End-to-end Phase 2 validation of the interactive dynamics + shortcut-forcing
flow-matching loss on the *frozen* tokenizer:

  Phase A  train the causal tokenizer on Pong frames (as in the tokenizer
           lighthouse), then FREEZE it (mae p=0 ⇒ clean latents).
  Phase B  each step: stream a Pong window, encode all T frames → clean
           latents z1 (packed k=1 ⇒ n_spatial=L, d_spatial=D_BOT); train the
           dynamics with `dynamics_pretrain_loss` (empirical flow + bootstrap).
  Eval     autoregressive one-step rollout: take frames 0..T-2 as clean
           context, ODE-sample (K steps) the T-1-th frame's latents, and
           score them against the true last-frame latents (latent MSE) — plus
           splice+decode for an informational pixel PSNR.

GATE = **latent-MSE of the ODE-sampled frame** (sampled vs ground-truth
latents). At init the zero-init flow head drives the ODE integration to ~0
(MSE ≈ mean(z²)); a trained dynamics predicts the true next-frame latents, so
the MSE collapses. This also validates the K-step ODE SAMPLER, a distinct
code path from the single-step training loss.

Pixel PSNR is reported but NOT gated: Pong frames are mostly-black and
near-identical frame-to-frame, so one-step next-frame PSNR saturates at the
(frozen) tokenizer's recon ceiling (~23 dB) regardless of the predicted frame
— a property of the sparse content + tokenizer, not the dynamics. Pure CPU.
"""

from std.memory import alloc
from std.math import sqrt, log, log10, cos
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler, ScriptedPongPolicy,
)
from mojo_rl.experimental.lewm2.pong_data import WindowSource
from mojo_rl.deep_agents2.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics
from mojo_rl.deep_agents2.dreamer4.shortcut_loss import dynamics_pretrain_loss
from mojo_rl.deep_agents2.dreamer4.ode_sampler import sample_one_timestep
from mojo_rl.deep_agents2.dreamer4.recon_loss import (
    masked_recon_loss, full_recon_psnr,
)
from mojo_rl.deep_agents2.dreamer4.patchify import downscale_box, temporal_patchify


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


# ── tiny deterministic RNG (xorshift64* + Box-Muller) ──────────────────
struct Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed | 1

    def u64(mut self) -> UInt64:
        var x = self.s
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.s = x
        return x * 0x2545F4914F6CDD1D

    def uniform(mut self) -> Float64:
        return Float64(self.u64() >> 11) * (1.0 / 9007199254740992.0)

    def gauss(mut self) -> Float64:
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


def main() raises:
    print("=" * 70)
    print("Dreamer 4 dynamics lighthouse — next-frame prediction (CPU)")
    print("=" * 70)

    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime ACT = 3
    comptime T = 4
    comptime B = 4
    comptime BATCH = B * T

    comptime TGT = 32
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                      # 64
    comptime D = 64
    comptime NH = 4
    comptime L = 8
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5

    comptime NSP = L                  # packing k=1
    comptime DSP = D_BOT
    comptime ND = NSP * DSP
    comptime KMAX = 4
    comptime KEVAL = 4
    comptime NREG = 2
    comptime D_DYN = 64
    comptime HID_DYN = 128
    comptime DEPTH_DYN = 2
    comptime B_SELF = 2
    comptime B_EMP = B - B_SELF
    comptime EMAX = 2                 # log2(KMAX)

    comptime STEPS_TOK = 150
    comptime STEPS_DYN = 220
    comptime EVAL_EVERY = 40
    comptime LR_TOK = Scalar[DT](2e-3)
    comptime LR_DYN = Scalar[DT](1e-3)

    comptime FRAME_N = BATCH * TGT * TGT
    comptime PATCH_N = BATCH * NP * DP
    comptime ZN = BATCH * ND

    comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]
    var src = WindowSource[IMG_DIM, ACT, T, B, "cpu", OnlineBuf].make(
        OnlineBuf.make(ScriptedPongPolicy(eps=0.3))
    )

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make[target="cpu", INIT=Xavier]()
    var topt = Adam.make["cpu", M=type_of(tok)](tok)
    topt.lr = LR_TOK

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX
    ].make[target="cpu", INIT=Xavier]()
    var dopt = Adam.make["cpu", M=type_of(dyn)](dyn)
    dopt.lr = LR_DYN

    var frames = _alloc(FRAME_N)
    var patches = _alloc(PATCH_N)
    var pred = _alloc(PATCH_N)
    var gpred = _alloc(PATCH_N)
    var gin = _alloc(PATCH_N)
    var z1 = _alloc(ZN)
    var gz = _alloc(ZN)
    var z0n = _alloc(ZN)
    var sigma = _alloc(BATCH)
    var sig_idx = _alloc(BATCH)
    var step_idx = _alloc(BATCH)
    var grad_zhat = _alloc(ZN)
    var zhat = _alloc(ZN)
    var ctx_lat = _alloc(B * (T - 1) * ND)
    var z_init = _alloc(B * ND)
    var pred_last = _alloc(B * ND)
    var zwin = _alloc(ZN)
    var rec = _alloc(PATCH_N)

    var pt = TileTensor(patches, row_major[BATCH, NP * DP]())
    var prt = TileTensor(pred, row_major[BATCH, NP * DP]())
    var git = TileTensor(gin, row_major[BATCH, NP * DP]())
    var z1_t = TileTensor(z1, row_major[BATCH, ND]())
    var zwin_t = TileTensor(zwin, row_major[BATCH, ND]())
    var rec_t = TileTensor(rec, row_major[BATCH, NP * DP]())
    var rng = Rng(20260606)

    # ── Phase A: train tokenizer ────────────────────────────────────────
    print("- Phase A: tokenizer")
    for step in range(STEPS_TOK):
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + bt * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)

        topt.zero_grad["cpu"](tok)
        tok.forward["cpu", BATCH](pt, output=prt)
        var mask = tok.mae_mask_ptr()
        var loss = masked_recon_loss[NP, DP, BATCH](pred, patches, mask, gpred)
        var got = TileTensor(gpred, row_major[BATCH, NP * DP]())
        tok.vjp["cpu", BATCH](got, git)
        topt.step["cpu"](tok)
        tok.advance_rng()
        if step % 50 == 0:
            tok.set_mae_p(0.0, 0.0)
            tok.forward["cpu", BATCH](pt, output=prt)
            print("   tok step", step, " recon PSNR =",
                  full_recon_psnr[NP, DP, BATCH](pred, patches), "dB")
            tok.set_mae_p(DROP, DROP)
    tok.set_mae_p(0.0, 0.0)            # FREEZE: clean latents from here on

    # ── Phase B: train dynamics on frozen-tokenizer latents ─────────────
    print("- Phase B: dynamics (shortcut forcing)")
    var first_psnr: Float64 = 0.0
    var last_psnr: Float64 = 0.0
    var first_lmse: Float64 = 0.0
    var last_lmse: Float64 = 0.0
    for step in range(-1, STEPS_DYN):
        var is_eval = step < 0 or step % EVAL_EVERY == 0 or step == STEPS_DYN - 1

        # stream a fresh Pong window + patchify
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + bt * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)

        # encode clean latents (tokenizer frozen, p=0)
        tok.enc.forward["cpu", BATCH](pt, output=z1_t)

        if step >= 0:
            # sample per-(b,t) step/sigma + noise, then one shortcut step
            for b in range(B):
                var is_self = b >= B_EMP
                for t in range(T):
                    var bt = b * T + t
                    var stp = EMAX
                    if is_self:
                        stp = Int(rng.uniform() * Float64(EMAX))   # [0,EMAX)
                    var K = 1 << stp
                    var j = Int(rng.uniform() * Float64(K))
                    if j >= K:
                        j = K - 1
                    var scale = KMAX // K
                    sigma[bt] = Scalar[DT](Float64(j) / Float64(K))
                    sig_idx[bt] = Scalar[DT](Float64(j * scale))
                    step_idx[bt] = Scalar[DT](Float64(stp))
            for i in range(ZN):
                z0n[i] = Scalar[DT](rng.gauss())

            var do_boot = step >= 30
            dopt.zero_grad["cpu"](dyn)
            var loss = dynamics_pretrain_loss[
                type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX
            ](dyn, z1, z0n, sigma, sig_idx, step_idx, do_boot, grad_zhat, zhat)
            var gzt = TileTensor(grad_zhat, row_major[BATCH, ND]())
            var gzi = TileTensor(gz, row_major[BATCH, ND]())
            dyn.vjp["cpu", BATCH](gzt, gzi)
            dopt.step["cpu"](dyn)
            if step % EVAL_EVERY == 0:
                print("   dyn step", step, " loss =", loss)

        # ── rollout eval (uses the just-encoded window) ─────────────────
        if is_eval:
            for b in range(B):
                for t in range(T - 1):
                    for i in range(ND):
                        ctx_lat[(b * (T - 1) + t) * ND + i] = z1[(b * T + t) * ND + i]
            for b in range(B):
                for i in range(ND):
                    z_init[b * ND + i] = Scalar[DT](rng.gauss())
            sample_one_timestep[type_of(dyn), B, T, NSP, DSP, KMAX, KEVAL](
                dyn, ctx_lat, z_init, pred_last
            )
            # latent-MSE: ODE-sampled last frame vs true last-frame latents
            var lse: Float64 = 0.0
            for b in range(B):
                for i in range(ND):
                    var dl = (
                        Float64(pred_last[b * ND + i])
                        - Float64(z1[(b * T + (T - 1)) * ND + i])
                    )
                    lse += dl * dl
            var lmse = lse / Float64(B * ND)
            for b in range(B):
                for t in range(T):
                    for i in range(ND):
                        var wi = (b * T + t) * ND + i
                        if t < T - 1:
                            zwin[wi] = z1[wi]
                        else:
                            zwin[wi] = pred_last[b * ND + i]
            tok.dec.forward["cpu", BATCH](zwin_t, output=rec_t)
            var sse: Float64 = 0.0
            for b in range(B):
                var base = (b * T + (T - 1)) * NP * DP
                for k in range(NP * DP):
                    var dd = Float64(rec[base + k]) - Float64(patches[base + k])
                    sse += dd * dd
            var mse = sse / Float64(B * NP * DP)
            var psnr = 120.0 if mse <= 1e-12 else -10.0 * log10(mse)
            if step < 0:
                first_psnr = psnr
                first_lmse = lmse
                print("   untrained dynamics: latent MSE =", lmse,
                      " (rollout PSNR =", psnr, "dB, saturated)")
            else:
                last_psnr = psnr
                last_lmse = lmse
                print("   dyn step", step, " latent MSE =", lmse,
                      " (rollout PSNR =", psnr, "dB)")

    print("-" * 70)
    print("  latent MSE   first =", first_lmse, "  final =", last_lmse)
    print("  rollout PSNR first =", first_psnr, "dB  final =", last_psnr,
          "dB  (informational — saturated at tokenizer ceiling)")
    assert_true(last_lmse < 0.3 * first_lmse,
                "ODE-sampled latent MSE must collapse (dynamics learns)")
    print("=" * 70)
    print("DYNAMICS LIGHTHOUSE PASSED — shortcut-forcing dynamics + ODE")
    print("sampler predict next-frame latents on frozen-tokenizer Pong (CPU)")
    print("=" * 70)
