"""ActionEmbedder — LeWM action encoding head.

Maps a per-step action vector `(B, T, ACT)` into a per-step embedding
`(B, T, EMB)` that conditions the ARPredictor's AdaLN-zero blocks.

Reference (`references/le-wm-main/module.py:189-214`):

    self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
    self.embed = nn.Sequential(
        nn.Linear(smoothed_dim, mlp_scale * emb_dim),
        nn.SiLU(),
        nn.Linear(mlp_scale * emb_dim, emb_dim),
    )

Conv1d k=1 over `(B, ACT, T)` is per-timestep Linear, so the whole stack is
three per-token Linears with SiLU between the last two. Mojo-rl maps this
to `Tokenwise[T, ...]` composition over the existing `Linear` / `LinearSwish`
Models — no new code, just composition.

Usage:

    comptime EMB = ActionEmbedder[
        T=4, ACT=1, smoothed=32, emb=128, mlp_scale=4,
    ]

The result is a `Model`, so it plugs into `NetworkState`, `Trainer`, etc.

Shapes: IN_DIM = T * ACT, OUT_DIM = T * EMB.
"""

from ...nn.model import Sequential, Linear, LinearSwish, Tokenwise


comptime ActionEmbedder[
    T: Int,
    ACT: Int,
    smoothed: Int,
    emb: Int,
    mlp_scale: Int = 4,
] = Sequential[
    # Step 1: Conv1d k=1 = per-timestep Linear[ACT, smoothed].
    Tokenwise[T, Linear[ACT, smoothed]],
    # Step 2: Linear[smoothed, mlp_scale*emb] + SiLU (fused).
    Tokenwise[T, LinearSwish[smoothed, mlp_scale * emb]],
    # Step 3: Linear[mlp_scale*emb, emb].
    Tokenwise[T, Linear[mlp_scale * emb, emb]],
]
