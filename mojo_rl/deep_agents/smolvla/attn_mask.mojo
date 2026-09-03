# +--------------------------------------------------------------------------+ #
# | SmolVLA — the prefix-LM block attention mask
# +--------------------------------------------------------------------------+ #
"""`make_att_2d_masks`, the rule that decides who attends to whom.

Transcribed from `lerobot/policies/common/vla_utils.py`, itself "copied from
big_vision" and shared with openpi. One 1-D vector `ar` drives everything:

    token i may attend to token j  iff  cumsum(ar)[j] <= cumsum(ar)[i]

`ar[k] == 1` starts a new block; `ar[k] == 0` keeps the token in the previous
one. Within a block attention is BIDIRECTIONAL, across blocks it is backwards
only. The reference's own examples, which the gate re-checks:

    [1 1 1 1 1 1]           pure causal
    [0 0 0 1 1 1]           prefix-LM: first 3 mutual, last 3 causal
    [1 0 1 0 1 0 0 1 0 0]   causal between 4 blocks

⚠ **This is not a causal mask, and SmolVLA is not a causal model.** Its `ar` is
`0` across BOTH the image and the language spans, so the whole visual+text
prefix is one bidirectional block. Assuming causality there — the obvious guess
for anything built on an LLM — silently changes what every prefix token sees.

## One builder, three uses

The window arguments exist because the same `ar` serves the square prefill mask
and the two rectangular denoising ones, and deriving each separately is how they
drift apart:

    prefill      rows = all,    cols = all           [P, P]
    denoise self rows = suffix, cols = all           [S, P+S]
    denoise cross rows = suffix, cols = prefix       [S, P]

Output is ADDITIVE (`0.0` allow, `MASK_NEG` deny), the convention
`MaskedAttention` and `CrossAttention` already take.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.masked_attention import MASK_NEG


def cumsum_blocks(ref ar: List[Int]) -> List[Int]:
    """Running block index — `cumsum(ar)`."""
    var out = List[Int](unsafe_uninit_length=len(ar))
    var acc = 0
    for i in range(len(ar)):
        acc += ar[i]
        out[i] = acc
    return out^


def att_2d_mask(
    ref ar: List[Int], q_lo: Int, q_hi: Int, kv_lo: Int, kv_hi: Int
) raises -> List[Scalar[DT]]:
    """Additive `[q_hi-q_lo, kv_hi-kv_lo]` mask over the window of `ar`.

    Rows and columns index the SAME token sequence; the windows select which
    queries and which keys this particular call is about. Both bounds are
    half-open.
    """
    if q_lo < 0 or q_hi > len(ar) or q_lo > q_hi:
        raise Error("att_2d_mask: bad query window")
    if kv_lo < 0 or kv_hi > len(ar) or kv_lo > kv_hi:
        raise Error("att_2d_mask: bad key window")
    var cs = cumsum_blocks(ar)
    var m = List[Scalar[DT]]()
    for i in range(q_lo, q_hi):
        for j in range(kv_lo, kv_hi):
            m.append(
                Scalar[DT](0.0) if cs[j] <= cs[i] else Scalar[DT](MASK_NEG)
            )
    return m^


def att_2d_mask_square(ref ar: List[Int]) raises -> List[Scalar[DT]]:
    """The full `[N, N]` mask — the prefill case."""
    return att_2d_mask(ar, 0, len(ar), 0, len(ar))


def smolvla_ar(
    n_image: Int, n_lang: Int, n_state: Int, chunk: Int
) raises -> List[Int]:
    """SmolVLA's own `ar`, from `embed_prefix` + `embed_suffix`.

        images   0 …   \\  one bidirectional block: the whole visual+text prefix
        language 0 …   /
        state    1 …      its own block — sees the prefix, the prefix cannot see it
        actions  1 × chunk   one block PER TOKEN, i.e. causal within the chunk

    ⚠ The action span is `[1] * chunk_size`, not `[1] + [0]*(chunk-1)`. Each
    action token opens its own block, so the chunk is CAUSAL within itself even
    though flow matching denoises it jointly. The `[1] + [0]*…` variant — a
    single bidirectional chunk — is what several pi-0 implementations use, and
    picking it here would be shape-identical and a different model.
    """
    if n_image < 0 or n_lang < 0 or n_state < 0 or chunk < 0:
        raise Error("smolvla_ar: negative span")
    var ar = List[Int]()
    for _ in range(n_image + n_lang):
        ar.append(0)
    for _ in range(n_state):
        ar.append(1)
    for _ in range(chunk):
        ar.append(1)
    return ar^
