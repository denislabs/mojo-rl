"""TD-MPC2 policy (actor) loss as a single ComputeGraph.

Reference `update_pi` (`references/tdmpc2-main/tdmpc2/tdmpc2.py:209-240`):

    a ~ π(z);  q = avg of 2 random Q(z, a) (DETACHED, two-hot decoded);  q /= scale
    pi_loss = mean( entropy_coef·ACT·log_prob(a|z) − q/scale )

Structurally SAC's actor loss with three swaps (vs `sac/actor_loss.mojo`):
  * min-of-2 critics → **avg-of-2** (both two-hot **decoded** to scalars);
  * Q scaled by 1/RunningScale (a `Scale` node, multiplier set per step);
  * α = `entropy_coef·ACT` (constant, not a learned temperature).

The two Q heads run with `MODE="input_only"` — Q params get no gradient
(reference `detach=True`), only the action does. POLICY, RSample, and the
two Q heads are external (trainer-owned); bound per step via `set_external`.

Output `[B, 1]`: per-sample policy loss = α·log_prob − 0.5·(q1d+q2d)/scale,
where the `qscaled` Scale multiplier folds `0.5/scale` and `alpha_lp` folds
`entropy_coef·ACT`.
"""

from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.concat import Concat
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.deep_agents2.primitives.rsample import RSample

from .nets import TDMPC2Policy, TDMPC2QNet
from .losses import TwoHotDecode


comptime TDMPC2PolicyGraph[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
] = ComputeGraph[
    1,
    InputSlot["z", LATENT],
    ExternalNode["pi_out", TDMPC2Policy[LATENT, ACT, MLP], "z"],
    ExternalNode["alp", RSample[ACT], "pi_out"],
    Node["action", Slice[ACT + 1, 0, ACT], "alp"],
    Node["log_prob", Slice[ACT + 1, ACT, ACT + 1], "alp"],
    Node["za", Concat[LATENT, ACT], "z", "action"],
    ExternalNode["q1", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za", MODE="input_only"],
    ExternalNode["q2", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za", MODE="input_only"],
    Node["q1d", TwoHotDecode[BINS, VMIN, VMAX], "q1"],
    Node["q2d", TwoHotDecode[BINS, VMIN, VMAX], "q2"],
    Node["qsum", Add[1, 2], "q1d", "q2d"],
    Node["qscaled", Scale[1], "qsum"],
    Node["alpha_lp", Scale[1], "log_prob"],
    Node["loss", BinarySub[1], "alpha_lp", "qscaled"],
]
