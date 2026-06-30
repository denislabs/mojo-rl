"""Dreamer4 tokenizer encoder read-out correctness gate (CPU).

Regression gate for the encoder `LearnedTokens` wiring (`dreamer4/encoder.mojo`).
The encoder must PREPEND `L` latent register tokens to the `NP` masked patch
tokens — sequence `[ L latents | NP patches ]` — so the tanh bottleneck (the
first `L` tokens, read by the `Slice`) is a function of ALL `NP` patches via the
modality-space attention. A prior bug used `LearnedTokens[L, NP, ...]` (args
swapped), which structurally KEEPS only the first `L` of the `NP` input patches
and drops the rest before the transformer ever sees them.

The existing storage smoke (`test_dreamer4_storage_smoke.mojo`) cannot catch
this: it uses `L=2, NP=3` and asserts only `Σ|grad| > 0`, and a plain
overfit/PSNR gate does NOT discriminate either (the decoder can memorize a small
fixed frame set even when most patches are dropped, because the few surviving
patches still identify the frame).

The discriminating, deterministic check is PATCH SENSITIVITY with `L != NP`:
  • Correct wiring → perturbing ANY single input patch changes the bottleneck z
    (every patch feeds the latents through attention) → min‖Δz‖ > 0.
  • Buggy wiring  → patches at index ≥ L never enter the transformer → those
    perturbations give Δz == 0 exactly → min‖Δz‖ == 0 → gate FAILS.

A second part trains the full tokenizer a few steps with `L != NP` to confirm the
forward/vjp/optimizer path actually reduces recon error end-to-end (train-path
sanity; not a discriminator).

Run: pixi run mojo run -I . tests/nn/test_dreamer4_encoder_readout.mojo
"""

from std.math import sqrt
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.optimizer.adam import Adam

from mojo_rl.deep_agents.dreamer4.encoder import Dreamer4Encoder
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.recon_loss import (
    masked_recon_loss, full_recon_psnr,
)
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


# ── Part 1: patch-sensitivity discriminator (L != NP) ──────────────────────
def _encoder_patch_sensitivity() raises -> Bool:
    comptime DP = 4
    comptime D = 16
    comptime NH = 2
    comptime T = 1
    comptime L = 3        # latents  ≠  patches  → exercises the swap
    comptime NP = 8       # the buggy wiring drops patches [L, NP) = [3, 8)
    comptime D_BOT = 8
    comptime HID = 16
    comptime DEPTH = 2
    comptime BATCH = 1    # T = 1, one frame → space attention only
    comptime NPDP = NP * DP
    comptime ZN = L * D_BOT

    # P_MIN = P_MAX = 0 → MAE keeps every patch → z is a deterministic
    # function of the input (no random dropout to confound the sensitivity).
    comptime ENC = Dreamer4Encoder[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, 0.0, 0.0, 0, True
    ]
    var enc = ENC.make["cpu", Deterministic](None)

    var inp = Tensor.alloc(BATCH * NPDP)
    for i in range(BATCH * NPDP):
        inp.data[i] = Scalar[DT]((i % 5) - 2) * 0.13 + 0.2

    var z0 = Tensor.alloc(BATCH * ZN)
    enc.forward["cpu", BATCH](TensorRefs[1](inp), z0, None)

    var zp = Tensor.alloc(BATCH * ZN)
    var min_dz = 1.0e30
    var max_dz = 0.0
    for p in range(NP):
        # perturb every channel of patch p, re-encode, restore.
        for d in range(DP):
            inp.data[p * DP + d] += Scalar[DT](0.5)
        enc.forward["cpu", BATCH](TensorRefs[1](inp), zp, None)
        for d in range(DP):
            inp.data[p * DP + d] -= Scalar[DT](0.5)

        var sse = 0.0
        for k in range(BATCH * ZN):
            var diff = Float64(zp.data[k]) - Float64(z0.data[k])
            sse += diff * diff
        var dz = sqrt(sse)
        print("    patch", p, " ‖Δz‖ =", dz)
        if dz < min_dz:
            min_dz = dz
        if dz > max_dz:
            max_dz = dz

    # Every patch must move the bottleneck. The buggy wiring leaves patches
    # [L, NP) with Δz == 0 → min_dz == 0.
    print("    min‖Δz‖ =", min_dz, " max‖Δz‖ =", max_dz)
    return min_dz > 1.0e-4


# ── Part 2: train-path sanity (full tokenizer, L != NP) ────────────────────
def _tokenizer_train_path() raises -> Bool:
    comptime DP = 4
    comptime D = 16
    comptime NH = 2
    comptime T = 1
    comptime L = 3
    comptime NP = 8
    comptime D_BOT = 8
    comptime HID = 16
    comptime DEPTH = 2
    comptime B = 2        # few distinct frames → overfit to a solid PSNR floor
    comptime BATCH = B * T
    comptime NPDP = NP * DP
    comptime DROP = 0.5

    comptime TOK = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 0, True
    ]
    var tok = TOK.make["cpu", Deterministic](None)

    # A small FIXED batch of distinct frames to overfit.
    var patches = Tensor.alloc(BATCH * NPDP)
    for i in range(BATCH * NPDP):
        patches.data[i] = Scalar[DT]((i * 7 + 3) % 11) / 11.0  # in [0,1)

    var pred = Tensor.alloc(BATCH * NPDP)
    var gpred = Tensor.alloc(BATCH * NPDP)
    var gin = Tensor.alloc(BATCH * NPDP)
    var opt = Adam(lr=2.0e-3)

    # clean-pass PSNR (run with MAE off)
    tok.set_mae_p(0.0, 0.0)
    tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
    var psnr0 = full_recon_psnr[NP, DP, BATCH](
        _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
    )
    tok.set_mae_p(DROP, DROP)

    for _ in range(800):
        opt.zero_grad["cpu"](tok, None)
        tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
        _ = masked_recon_loss[NP, DP, BATCH](
            _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr()),
            tok.mae_mask_ptr(), _mao(gpred.data.unsafe_ptr()),
        )
        tok.vjp["cpu", BATCH](
            TensorRefs[1](patches), gpred, TensorRefs[1](gin), None
        )
        opt.step["cpu"](tok, None)
        tok.advance_rng()

    tok.set_mae_p(0.0, 0.0)
    tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
    var psnr1 = full_recon_psnr[NP, DP, BATCH](
        _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
    )

    print("    recon PSNR:", psnr0, "→", psnr1, "dB")
    # Train-path sanity: the real masked-MAE objective must clearly reduce recon
    # error through the L != NP encoder (the absolute ceiling is low because the
    # overfit targets are random-content patches with no spatial structure; the
    # IMPROVEMENT is the signal, not the floor).
    return (psnr1 > psnr0 + 4.0) and (psnr1 > 10.0)


def main() raises:
    print("Dreamer4 encoder read-out gate (CPU)")
    var sens_ok = _encoder_patch_sensitivity()
    print("  every patch feeds the bottleneck (L != NP):",
          "OK" if sens_ok else "FAIL")
    var train_ok = _tokenizer_train_path()
    print("  tokenizer trains end-to-end (L != NP):",
          "OK" if train_ok else "FAIL")
    assert_true(sens_ok, "encoder patch sensitivity (read-out wiring)")
    assert_true(train_ok, "tokenizer recon improves with L != NP")
    print("DREAMER4 ENCODER READOUT GATE OK")
