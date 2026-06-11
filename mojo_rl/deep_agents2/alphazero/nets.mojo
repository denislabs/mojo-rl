"""AlphaZero networks (nn2) — shared torso fanning out to a policy head and a
value head, concatenated into one ``[policy_logits | raw_value]`` output.

Layout (matches the planner's ``search_gpu_alphazero`` contract):
  * ``PRED_OUT_DIM = ACT + 1``
  * policy logits at columns ``[0, ACT)`` — the MCTS init/expand kernels softmax
    (+ legal-mask) this slice into the prior.
  * raw value at column ``ACT`` — **un-squashed**. The masked expand/backup
    kernel applies the tanh squash itself (``VALUE_SQUASH=True`` in
    ``GenericGPUMCTS.search_gpu_alphazero``), so the net must emit the raw
    pre-tanh scalar.

Three torso families are provided — all share the same external contract
(``IN_DIMS[0] == OBS``, ``OUT_DIM == ACT + 1``, raw value), so the MCTS
adapter / self-play / eval / checkpoint paths treat them interchangeably:

  * ``AZMLPNet``    — flat-obs MLP (small boards: TicTacToe / Connect4).
  * ``AZCNNNet``    — Conv→BN→ReLU stack → flatten → FC heads (alpha-zero-general
                       style). The flat obs is interpreted as ``PLANES × R × C``
                       (the env emits plane-major obs, i.e. CHW flat), so it
                       feeds ``Conv2D`` directly with no reshape.
  * ``AZResNetNet`` — conv stem → ``NUM_BLOCKS`` identity-skip ResBlocks →
                       reduce-to-1×1 conv → flatten → FC heads (closer to the
                       original AlphaZero backbone).

The CNN / ResNet torsos carry ``BatchNorm2D``, so callers must toggle
``set_attr["training"]`` (1.0 train / 0.0 eval) around MCTS inference vs the
training step — the self-play driver and eval harness do this; it is a no-op
for ``AZMLPNet``.

``Parallel[A, B]`` concatenates ``[A(x) | B(x)]`` column-wise, so
``Parallel[Linear[H, ACT], Linear[H, 1]]`` produces exactly the
``[policy(ACT) | value(1)]`` packing the planner expects.
"""

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.parallel import Parallel
from mojo_rl.nn2.combinators.repeat import Repeat
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn2.models.resnet import ResBlockConv2DBN
from mojo_rl.nn2.primitives.batch_norm_2d import BN2D_DEFAULT_EPS


# ──────────────────────────────────────────────────────────────────────
# MLP torso
# ──────────────────────────────────────────────────────────────────────


comptime AZMLPNet[OBS: Int, ACT: Int, H: Int] = Sequential[
    LinearReLU[OBS, H],
    LinearReLU[H, H],
    Parallel[Linear[H, ACT], Linear[H, 1]],
]


# ──────────────────────────────────────────────────────────────────────
# CNN torso (alpha-zero-general style)
# ──────────────────────────────────────────────────────────────────────


# 4× Conv→BN→ReLU: three "same" (R×C preserved) then one "valid" 3×3 that
# reduces R×C → (R-2)×(C-2), then flatten → FC trunk → policy/value heads.
# Requires R >= 3 and C >= 3.  ``F`` = conv filters, ``FC`` = head width.
comptime AZCNNNet[
    OBS: Int, ACT: Int, PLANES: Int, R: Int, C: Int, F: Int, FC: Int,
] = Sequential[
    Conv2DBatchNormReLU[PLANES, F, 3, 1, 1, R, C],   # planes→F, R×C → R×C
    Conv2DBatchNormReLU[F, F, 3, 1, 1, R, C],
    Conv2DBatchNormReLU[F, F, 3, 1, 1, R, C],
    Conv2DBatchNormReLU[F, F, 3, 1, 0, R, C],         # R×C → (R-2)×(C-2)
    Flatten[F * (R - 2) * (C - 2)],
    LinearReLU[F * (R - 2) * (C - 2), FC],
    Parallel[Linear[FC, ACT], Linear[FC, 1]],
]


# ──────────────────────────────────────────────────────────────────────
# ResNet torso (closer to the original AlphaZero backbone)
# ──────────────────────────────────────────────────────────────────────


# Conv stem → NUM_BLOCKS identity-skip ResBlocks (R×C preserved) → a "valid"
# 3×3 conv reducing R×C → (R-2)×(C-2) → flatten → FC trunk → heads.
# ``ResBlockConv2DBN`` needs P=(K-1)//2=1 for K=3 to preserve spatial dims.
comptime AZResNetNet[
    OBS: Int, ACT: Int, PLANES: Int, R: Int, C: Int,
    F: Int, NUM_BLOCKS: Int, FC: Int,
    EPS: Float64 = BN2D_DEFAULT_EPS,   # BatchNorm epsilon (caps train-mode
    #   inv_std amplification; raise to e.g. 1e-3 if a float32 torso overflows
    #   the policy logits on low-diversity self-play batches).
] = Sequential[
    Conv2DBatchNormReLU[PLANES, F, 3, 1, 1, R, C, EPS],   # stem: planes→F
    Repeat[NUM_BLOCKS, ResBlockConv2DBN[F, 3, 1, R, C, EPS], shared=False],
    Conv2DBatchNormReLU[F, F, 3, 1, 0, R, C, EPS],    # reduce R×C → (R-2)×(C-2)
    Flatten[F * (R - 2) * (C - 2)],
    LinearReLU[F * (R - 2) * (C - 2), FC],
    Parallel[Linear[FC, ACT], Linear[FC, 1]],
]


# ──────────────────────────────────────────────────────────────────────
# Board presets (fill in geometry; caller picks widths)
# ──────────────────────────────────────────────────────────────────────


# TicTacToe: 27D obs = 3 planes × 3×3, 9 actions.
comptime AZTicTacToeCNN[F: Int = 32, FC: Int = 64] = AZCNNNet[27, 9, 3, 3, 3, F, FC]
comptime AZTicTacToeResNet[F: Int = 32, NB: Int = 3, FC: Int = 64] = AZResNetNet[
    27, 9, 3, 3, 3, F, NB, FC
]

# Connect4: 126D obs = 3 planes × 6×7, 7 actions.
comptime AZConnectFourCNN[F: Int = 64, FC: Int = 128] = AZCNNNet[
    126, 7, 3, 6, 7, F, FC
]
comptime AZConnectFourResNet[
    F: Int = 64, NB: Int = 5, FC: Int = 128,
    EPS: Float64 = BN2D_DEFAULT_EPS,
] = AZResNetNet[126, 7, 3, 6, 7, F, NB, FC, EPS]
