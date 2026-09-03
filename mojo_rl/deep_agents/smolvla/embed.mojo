# +--------------------------------------------------------------------------+ #
# | SmolVLA — prefix and suffix assembly
# +--------------------------------------------------------------------------+ #
"""Turning cameras, an instruction, a joint state and a noisy action chunk into
the two token streams the fused loop consumes.

    prefix  [images…, language…, state]   ->  [B, P * 960]   (the VLM stream)
    suffix  [action ⊕ time]               ->  [B, S * 720]   (the expert stream)

## ⚠ The sqrt(D) scaling is not optional

`embed_prefix` multiplies BOTH the image and the language embeddings by
`sqrt(hidden)` = sqrt(960) ≈ 30.98:

    img_emb  = img_emb  * img_emb_dim ** 0.5
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)

Dropping it leaves every prefix token ~31x too small. Nothing crashes, no NaN
appears, and the policy is simply wrong — the single easiest catastrophic
omission in this file. The state embedding is NOT scaled.

## ⚠ The time embedding is split-halves, and computed in float64

`create_sinusoidal_pos_embedding` (openpi's, copied exactly) is

    fraction = linspace(0, 1, D/2)          # BOTH endpoints included
    period   = min_period * (max_period/min_period) ** fraction
    out      = concat([sin(t / period * 2π), cos(t / period * 2π)])

`concat([sin, cos])`, so the first D/2 entries are sines and the last D/2
cosines — NOT interleaved, and not the same convention as RoPE's channel pairs
two files over. The reference computes it in float64 and casts, so this does
too: at `min_period = 0.004` the scaling factor is ~1570, and fp32 rounding of
`t * 1570` is visible in the low bits.

## ⚠ Concat order is [action, time]

`torch.cat([action_emb, time_emb], dim=2)` — action first, giving the 1440 that
`action_time_mlp_in` takes (2 x 720). Reversed, it is the same shape and a
different model. The MLP is `in -> SiLU -> out`, a plain two-layer MLP; it is NOT
the SwiGLU used inside the decoder layers.

⚠ `add_image_special_tokens = False` in this checkpoint, so there are no
image-start/end tokens around each camera. The branch exists in the reference;
enabling it would change the prefix length and every mask derived from it.
"""

from std.math import sin, cos, exp, log, sqrt
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


comptime SMOLVLA_MIN_PERIOD: Float64 = 0.004
comptime SMOLVLA_MAX_PERIOD: Float64 = 4.0


def sinusoidal_time_embedding[
    D: Int,
    MIN_PERIOD: Float64 = SMOLVLA_MIN_PERIOD,
    MAX_PERIOD: Float64 = SMOLVLA_MAX_PERIOD,
](t: Float64) raises -> List[Scalar[DT]]:
    """`[sin(t/period · 2π) … , cos(t/period · 2π) …]`, length `D`.

    Accumulated in Float64 and cast once at the end, matching the reference's
    `get_safe_dtype(torch.float64, …)`.
    """
    comptime assert D % 2 == 0, "sinusoidal_time_embedding: D must be even"
    comptime H = D // 2
    comptime assert H > 1, "D/2 must exceed 1 for linspace's two endpoints"
    var two_pi = 6.283185307179586
    var out = List[Scalar[DT]](unsafe_uninit_length=D)
    var ratio = log(MAX_PERIOD / MIN_PERIOD)
    for i in range(H):
        # linspace(0, 1, H) includes BOTH endpoints -> i / (H - 1)
        var frac = Float64(i) / Float64(H - 1)
        var period = MIN_PERIOD * exp(ratio * frac)
        var arg = t / period * two_pi
        out[i] = Scalar[DT](sin(arg))
        out[H + i] = Scalar[DT](cos(arg))
    return out^


def scale_in_place[
    target: StaticString
](mut t: Tensor, n: Int, factor: Scalar[DT], ctx: Optional[DeviceContext] = None) raises:
    """Multiply the first `n` elements by `factor`, on the host.

    Used for the sqrt(D) prefix scaling. Host-side because assembly happens once
    per inference, before anything is uploaded — a kernel here would be
    ceremony for one pass over a few hundred thousand floats.
    """
    for i in range(n):
        t.data[i] = t.data[i] * factor


def embed_language_tokens[
    VOCAB: Int, DIM: Int
](
    mut weight: Tensor, ref ids: List[Int], mut out: Tensor, scale: Bool = True
) raises:
    """Gather embedding rows for `ids` and (by default) apply the sqrt(DIM) scale.

    A row gather, not a one-hot matmul: the instruction is at most 48 tokens and
    a `[48, 49280]` one-hot would be 9.4 M floats to express 48 integers.
    """
    var n = len(ids)
    out.ensure(n * DIM)
    var f = sqrt(Scalar[DT](DIM)) if scale else Scalar[DT](1)
    for t in range(n):
        var id = ids[t]
        if id < 0 or id >= VOCAB:
            raise Error(
                "embed_language_tokens: id " + String(id) + " out of vocab "
                + String(VOCAB)
            )
        for d in range(DIM):
            out.data[t * DIM + d] = weight.data[id * DIM + d] * f
    _ = n
