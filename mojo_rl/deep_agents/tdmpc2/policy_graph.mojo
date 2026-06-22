"""TD-MPC2 policy (actor) loss as a single ComputeGraph.

Reference `update_pi` (`references/tdmpc2-main/tdmpc2/tdmpc2.py:209-240`):

    a ~ π(z);  q = avg of 2 random Q(z, a) (DETACHED, two-hot decoded);  q /= scale
    pi_loss = mean( entropy_coef·ACT·log_prob(a|z) − q/scale )

Structurally SAC's actor loss with three swaps (vs `sac/actor_loss.mojo`):
  * min-of-2 critics → **avg-of-2** (both two-hot **decoded** to scalars);
  * Q scaled by 1/RunningScale (a `Scale` node, multiplier set per step);
  * α = `entropy_coef·ACT` (constant, not a learned temperature).

The two Q heads' params must stay detached (reference `detach=True`) — only
the action gets gradient. Storage has no per-node `MODE="input_only"`; instead
(mirroring the storage SAC actor loss) the Q heads are threaded externals whose
param grads from this backward are simply discarded — the value/TD-target step
`zero_grad`s the Q nets before its own update, so the policy loss never moves
them. POLICY, RSample, and the two Q heads are external (trainer-owned),
threaded into `forward`/`vjp` in node order.

Output `[B, 1]`: per-sample policy loss = α·log_prob − 0.5·(q1d+q2d)/scale,
where the `qscaled` Scale multiplier folds `0.5/scale` and `alpha_lp` folds
`entropy_coef·ACT`.
"""

from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node, ExternalNode
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat
from mojo_rl.nn.storage.primitives.add import Add
from mojo_rl.nn.storage.primitives.scale import Scale
from mojo_rl.nn.storage.primitives.binary_elementwise import BinarySub
from mojo_rl.nn.storage.primitives.rsample import RSample

from .nets import TDMPC2Policy, TDMPC2QNet
from .losses import TwoHotDecode


comptime TDMPC2PolicyGraph[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    QP: Float64 = 0.0,
] = ComputeGraph[
    InputSlot["z", LATENT],
    ExternalNode["pi_out", TDMPC2Policy[LATENT, ACT, MLP], "z"],
    ExternalNode["alp", RSample[ACT], "pi_out"],
    Node["action", Slice[ACT + 1, 0, ACT], "alp"],
    Node["log_prob", Slice[ACT + 1, ACT, ACT + 1], "alp"],
    Node["za", Concat[LATENT, ACT], "z", "action"],
    ExternalNode["q1", TDMPC2QNet[LATENT, ACT, MLP, BINS, QP], "za"],
    ExternalNode["q2", TDMPC2QNet[LATENT, ACT, MLP, BINS, QP], "za"],
    Node["q1d", TwoHotDecode[BINS, VMIN, VMAX], "q1"],
    Node["q2d", TwoHotDecode[BINS, VMIN, VMAX], "q2"],
    Node["qsum", Add[1], "q1d", "q2d"],
    Node["qscaled", Scale[1], "qsum"],
    Node["alpha_lp", Scale[1], "log_prob"],
    Node["loss", BinarySub[1], "alpha_lp", "qscaled"],
]
