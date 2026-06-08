"""GPT construction ops for `GPTDropTied` — the nanoGPT recipe pieces that
live *outside* the forward graph (init-time weight surgery + the one-time
weight-tie wiring).

`GPTDropTied` (in `composites.mojo`) is the forward graph: a `GPTDrop`
whose LM head is a bias-less `TiedLinear` borrowing the token embedding's
table. Two pieces still happen outside the graph because they touch
specific weights by position in the child tree:

  - **c_proj scaled init** (`gpt_scale_residual_proj`): divide each
    residual *output* projection (attention-out + FFN-out) weight by
    1/√(2L) after the generic INIT, bounding residual-stream variance as
    depth grows (GPT-2 / nanoGPT). Call once, after `make`.

  - **weight-tie wiring** (`gpt_wire_tie`): point the `TiedLinear` head at
    the embedding's value + grad buffers (nanoGPT's `lm_head.weight =
    wte.weight`). Call ONCE after the model reaches its final home (e.g.
    inside the `Trainer`) and after any model load — it captures raw
    buffer pointers, which must be live and stationary. After wiring, the
    standard `Trainer.train_*` loop trains the tied model with **no**
    per-step tying code: the shared weight accumulates one gradient (both
    the embedding's lookup-vjp and the head's matmul-vjp `+=` into it) and
    the optimizer updates it once (reflection sees it only via the
    embedding — the head owns no `Param`). Bias-less head ⇒ nothing to
    freeze. See `primitives/tied_linear.mojo`.

Both take the concrete `GPTDropTied[...]` model so the child-tree walk
type-checks. Pass GPTDropTied's *full* param list explicitly — Mojo can
neither reverse-infer a parametric alias's args from the expanded
`Sequential` type (it won't solve `3*embed_dim = 1152`) nor fold the
params' defaults into the dependent `net` argument type. A param that
doesn't match the passed `net`'s type is a loud compile error.

GPTDropTied child tree (see `composites.mojo`):
    [0] Tokenwise[Embedding]      ← embedding W (the shared weight)
    [1] BiasAdd
    [2] Dropout
    [3] Repeat[TransformerBlockDrop]
          .children[L] = block L:
            [0] Residual(LN + MHADrop)
                  .inner.children[1] = MHADrop =
                    Seq[Tok[Lin d,3d], QKVToMajor, Attn,
                        Tok[Lin d,d] (c_proj @ .children[3]), Dropout]
            [1] Residual(LN + FFNDrop)
                  .inner.children[1] = FFNDrop =
                    Seq[Tok[Lin d,ff], GELU,
                        Tok[Lin ff,d] (c_proj @ .children[2]), Dropout]
    [4] Tokenwise[LayerNorm]
    [5] Tokenwise[TiedLinear]     ← LM head (borrows [0]) ; = N-1
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext

from .constants import DT, TPB
from .core.module import mptr
from .composites import GPTDropTied


# ──────────────────────────────────────────────────────────────────────
# GPU kernel.
# ──────────────────────────────────────────────────────────────────────


def _scale_kernel(
    buf: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, s: Scalar[DT]
):
    """`buf[i] *= s`. One thread per element."""
    var i = Int(global_idx.x)
    if i < n:
        buf[i] = buf[i] * s


# ──────────────────────────────────────────────────────────────────────
# Public ops — concrete `GPTDropTied[...]`, full param list explicit.
# ──────────────────────────────────────────────────────────────────────


def gpt_scale_residual_proj[
    vocab: Int, seq_len: Int, embed_dim: Int, n_heads: Int, n_layers: Int,
    ff_mult: Int, causal: Bool, dropout_p: Float64, seed_base: UInt64,
    use_max: Bool,
](
    mut net: GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers, ff_mult, causal,
        dropout_p, seed_base, use_max,
    ],
    ctx: DeviceContext,
) raises:
    """nanoGPT/GPT-2 scaled init: divide each residual output projection
    (attention-out + FFN-out) weight by 1/√(2L). Call once after `make`."""
    var s = Scalar[DT](1.0 / sqrt(Float64(2 * n_layers)))
    comptime DD = embed_dim * embed_dim              # attn-out Linear[D, D]
    comptime FD = (ff_mult * embed_dim) * embed_dim  # FFN-out  Linear[F, D]
    comptime db = (DD + TPB - 1) // TPB
    comptime fb = (FD + TPB - 1) // TPB
    for L in range(n_layers):
        var a = mptr(
            net.children[3].children[L].children[0].inner
            .children[1].children[3].inner.weight.val.dev.value().unsafe_ptr()
        )
        ctx.enqueue_function[_scale_kernel](
            a, DD, s, grid_dim=db, block_dim=TPB
        )
        var f = mptr(
            net.children[3].children[L].children[1].inner
            .children[1].children[2].inner.weight.val.dev.value().unsafe_ptr()
        )
        ctx.enqueue_function[_scale_kernel](
            f, FD, s, grid_dim=fb, block_dim=TPB
        )


def gpt_wire_tie[
    target: StaticString,
    vocab: Int, seq_len: Int, embed_dim: Int, n_heads: Int, n_layers: Int,
    ff_mult: Int, causal: Bool, dropout_p: Float64, seed_base: UInt64,
    use_max: Bool,
](
    mut net: GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers, ff_mult, causal,
        dropout_p, seed_base, use_max,
    ],
) raises:
    """Point the `TiedLinear` LM head at the embedding's value + grad
    buffers. Call ONCE after the model settles in its final home (and
    after any load). Idempotent. `target` selects which storage (device
    buffers on 'gpu', host lists on 'cpu')."""
    comptime LM_IDX = GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers, ff_mult, causal,
        dropout_p, seed_base, use_max,
    ].N - 1
    # Capture the embedding (child 0) buffer pointers as plain values first,
    # releasing the borrow before mutating the head (child N-1).
    comptime if target == "gpu":
        var val_ptr = mptr(
            net.children[0].inner.weight.val.dev.value().unsafe_ptr()
        )
        var grd_ptr = mptr(
            net.children[0].inner.weight.grd.dev.value().unsafe_ptr()
        )
        net.children[LM_IDX].inner.tie_to(val_ptr, grd_ptr)
    else:
        var val_ptr = net.children[0].inner.weight.val.cpu_ptr()
        var grd_ptr = net.children[0].inner.weight.grd.cpu_ptr()
        net.children[LM_IDX].inner.tie_to(val_ptr, grd_ptr)
