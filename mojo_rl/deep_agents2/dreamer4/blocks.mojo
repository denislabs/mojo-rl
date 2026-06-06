"""Dreamer 4 block-causal transformer — comptime combinator aliases.

With the two custom leaves in place — `ModalitySpaceAttention` (modality-gated
space attention) and `TimeAttentionLatents` (causal time attention over the
latent tokens) — the Dreamer 4 block-causal transformer is a pure combinator
stack at nn2-BATCH = B·T (one frame per sample, sequence S, channel D). No
bespoke module is needed for the transformer itself.

Sublayers are pre-RMSNorm residual branches:

    SpaceSub : x + Linear∘ModalitySpaceAttn∘QKV(RMSNorm(x))
    TimeSub  : x + TimeAttentionLatents(RMSNorm(x))   (latents only)
    FFNSub   : x + Linear∘SwiGLU∘Linear(RMSNorm(x))

`Dreamer4Block` puts time attention in *every* block (time_every = 1, the
tokenizer config). Other `time_every` schedules can be built by composing
`Dreamer4BlockNoTime` and `Dreamer4Block` explicitly. The full stack is
`Repeat[DEPTH, Dreamer4Block]` (independent params per layer).

Per-token sublayers use `Tokenwise[S, ...]` (shared weights across the S
tokens); attention spans the sequence. `SwiGLU[HID]` is wrapped per-token so
each token's 2·HID projection is split into (u, v) correctly.
"""

from mojo_rl.nn2.combinators import Sequential, Residual, Repeat, Tokenwise
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.primitives.qkv_to_major import QKVToMajor
from mojo_rl.nn2.primitives.swiglu import SwiGLU
from mojo_rl.nn2.primitives.modality_space_attention import ModalitySpaceAttention
from mojo_rl.nn2.primitives.time_attention_latents import TimeAttentionLatents


comptime Dreamer4SpaceSub[
    D: Int, NH: Int, S: Int, L: Int, MODE: StaticString, USE_MAX: Bool = True
] = Residual[
    Sequential[
        Tokenwise[S, RMSNorm[D]],
        Tokenwise[S, Linear[D, 3 * D]],
        QKVToMajor[S, D],
        ModalitySpaceAttention[D, NH, S, L, MODE, USE_MAX],
        Tokenwise[S, Linear[D, D]],
    ]
]


comptime Dreamer4TimeSub[
    D: Int, NH: Int, T: Int, S: Int, L: Int
] = Residual[
    Sequential[
        Tokenwise[S, RMSNorm[D]],
        TimeAttentionLatents[D, NH, T, S, L],
    ]
]


comptime Dreamer4FFNSub[D: Int, S: Int, HID: Int] = Residual[
    Sequential[
        Tokenwise[S, RMSNorm[D]],
        Tokenwise[S, Linear[D, 2 * HID]],
        Tokenwise[S, SwiGLU[HID]],
        Tokenwise[S, Linear[HID, D]],
    ]
]


# Full block with time attention (time_every = 1).
comptime Dreamer4Block[
    D: Int, NH: Int, T: Int, S: Int, L: Int, HID: Int,
    MODE: StaticString, USE_MAX: Bool = True,
] = Sequential[
    Dreamer4SpaceSub[D, NH, S, L, MODE, USE_MAX],
    Dreamer4TimeSub[D, NH, T, S, L],
    Dreamer4FFNSub[D, S, HID],
]


# Space + FFN only (for non-time-attention layers in a time_every > 1 schedule).
comptime Dreamer4BlockNoTime[
    D: Int, NH: Int, S: Int, L: Int, HID: Int,
    MODE: StaticString, USE_MAX: Bool = True,
] = Sequential[
    Dreamer4SpaceSub[D, NH, S, L, MODE, USE_MAX],
    Dreamer4FFNSub[D, S, HID],
]


# Depth-stacked block-causal transformer (time every block).
comptime Dreamer4Stack[
    D: Int, NH: Int, T: Int, S: Int, L: Int, HID: Int, DEPTH: Int,
    MODE: StaticString, USE_MAX: Bool = True,
] = Repeat[DEPTH, Dreamer4Block[D, NH, T, S, L, HID, MODE, USE_MAX]]
