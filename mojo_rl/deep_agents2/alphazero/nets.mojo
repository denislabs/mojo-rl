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

``AZMLPNet`` is the small-board torso (TicTacToe / Connect4 flat obs). A ResNet
torso variant lands alongside it once the MLP path converges.

``Parallel[A, B]`` concatenates ``[A(x) | B(x)]`` column-wise, so
``Parallel[Linear[H, ACT], Linear[H, 1]]`` produces exactly the
``[policy(ACT) | value(1)]`` packing the planner expects.
"""

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.parallel import Parallel
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU


comptime AZMLPNet[OBS: Int, ACT: Int, H: Int] = Sequential[
    LinearReLU[OBS, H],
    LinearReLU[H, H],
    Parallel[Linear[H, ACT], Linear[H, 1]],
]
