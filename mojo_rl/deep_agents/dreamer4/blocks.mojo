"""Dreamer 4 block-causal transformer — comptime combinator aliases.

With the two custom leaves in place — `ModalitySpaceAttention` (modality-gated
space attention) and `TimeAttentionLatents` (causal time attention over the
latent tokens) — the Dreamer 4 block-causal transformer is a pure combinator
stack at nn-BATCH = B·T (one frame per sample, sequence S, channel D). No
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

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.residual import Residual
from mojo_rl.nn.combinators.repeat import Repeat
from mojo_rl.nn.combinators.tokenwise import Tokenwise
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.rms_norm import RMSNorm
from mojo_rl.nn.primitives.qkv_to_major import QKVToMajor
from mojo_rl.nn.primitives.swiglu import SwiGLU
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.nn.primitives.activations import Sigmoid
from mojo_rl.nn.primitives.slice import Slice
from mojo_rl.nn.primitives.learned_tokens import LearnedTokens
from mojo_rl.nn.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from mojo_rl.nn.primitives.modality_space_attention import ModalitySpaceAttention
from mojo_rl.nn.primitives.dynamics_space_attention import DynamicsSpaceAttention
from mojo_rl.nn.primitives.time_attention_latents import TimeAttentionLatents


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


# ──────────────────────────────────────────────────────────────────────
# Dynamics-transformer blocks (model.py:Dynamics). Same sublayer structure as
# the generic block, but the SPACE attention uses `DynamicsSpaceAttention` so
# the modality mask follows the real per-frame dynamics layout
#   [ action | signal | step | spatial×NSP | register×NREG | agent×NAGENT ]
# (S = 3 + NSP + NREG + NAGENT). The TIME attention runs over ALL S token
# positions (L = S): time attention is position-wise across frames, so agent
# tokens never leak into other positions through it — the space mask alone
# enforces agent isolation. With NAGENT = 0 the `wm_agent_bc` mask collapses to
# full mixing (no token carries the agent id), so this is bit-identical to the
# unconditional `Dreamer4Stack[..., L=S, "wm_agent"]` path.
comptime Dreamer4DynSpaceSub[
    D: Int, NH: Int, NSP: Int, NREG: Int, NAGENT: Int,
    MODE: StaticString, USE_MAX: Bool = True,
] = Residual[
    Sequential[
        Tokenwise[3 + NSP + NREG + NAGENT, RMSNorm[D]],
        Tokenwise[3 + NSP + NREG + NAGENT, Linear[D, 3 * D]],
        QKVToMajor[3 + NSP + NREG + NAGENT, D],
        DynamicsSpaceAttention[D, NH, NSP, NREG, NAGENT, MODE, USE_MAX],
        Tokenwise[3 + NSP + NREG + NAGENT, Linear[D, D]],
    ]
]


comptime Dreamer4DynBlock[
    D: Int, NH: Int, T: Int, NSP: Int, NREG: Int, NAGENT: Int, HID: Int,
    MODE: StaticString, USE_MAX: Bool = True,
] = Sequential[
    Dreamer4DynSpaceSub[D, NH, NSP, NREG, NAGENT, MODE, USE_MAX],
    Dreamer4TimeSub[
        D, NH, T, 3 + NSP + NREG + NAGENT, 3 + NSP + NREG + NAGENT
    ],
    Dreamer4FFNSub[D, 3 + NSP + NREG + NAGENT, HID],
]


comptime Dreamer4DynStack[
    D: Int, NH: Int, T: Int, NSP: Int, NREG: Int, NAGENT: Int, HID: Int,
    DEPTH: Int, MODE: StaticString, USE_MAX: Bool = True,
] = Repeat[
    DEPTH, Dreamer4DynBlock[D, NH, T, NSP, NREG, NAGENT, HID, MODE, USE_MAX]
]


# ──────────────────────────────────────────────────────────────────────
# Tokenizer decoder (model.py:Decoder). Input is the per-frame bottleneck
# z (L latents × D_BOT); output is the reconstructed patch tokens (NP × DP)
# in [0, 1]. With S = L + NP this is a near-pure Sequential — the learned
# patch queries are `LearnedTokens` (append), the latent→patch read-out is a
# `Slice` of the transformer output. Runs at nn-BATCH = B·T.
#
#   up_proj(tanh) → append patch queries → +positions → decoder transformer
#   → slice patch tokens → patch_head → sigmoid
# ──────────────────────────────────────────────────────────────────────
comptime Dreamer4Decoder[
    D_BOT: Int, D: Int, NH: Int, T: Int, L: Int, NP: Int, DP: Int,
    HID: Int, DEPTH: Int, USE_MAX: Bool = True,
] = Sequential[
    Tokenwise[L, Linear[D_BOT, D]],                 # up_proj
    Tanh[L * D],
    LearnedTokens[L, NP, D, False],                 # append patch queries → S=L+NP
    SinusoidalPosAddBT[T, L + NP, D],
    Dreamer4Stack[D, NH, T, L + NP, L, HID, DEPTH, "decoder", USE_MAX],
    Slice[(L + NP) * D, L * D, (L + NP) * D],        # patch tokens → NP·D
    Tokenwise[NP, Linear[D, DP]],                    # patch_head
    Sigmoid[NP * DP],
]
