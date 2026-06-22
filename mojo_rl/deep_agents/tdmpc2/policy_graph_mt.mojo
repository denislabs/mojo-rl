"""TD-MPC2 multi-task policy (actor) loss ComputeGraph (item C, §14.3).

Clone of `policy_graph.mojo` with a `task_emb [TASK_EMB]` input slot threaded
into BOTH the policy input (`zt = Concat[LATENT, TASK_EMB]("z","task_emb")`) and
the detached-Q input (`za = Concat[LATENT, MAX_ACT, TASK_EMB]`). After
`graph.vjp`, `grad_input["task_emb"]` carries the actor-loss gradient w.r.t.
the embedding (collected into the table — site 3 of the embedding grad flow).

As in the single-task `policy_graph.mojo`, RSample is an internal `Node` (the
storage ComputeGraph caches per-forward leaf state only for graph-owned
children, so a threaded-external RSample's reparam noise wouldn't survive
forward→vjp), and the detached-Q semantics is the SAC pattern: the two Q heads
are threaded externals whose param grads from this backward are discarded (the
value/TD-target step zero_grads them before its own update).

See `policy_graph.mojo` for the loss structure.
"""

from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node, ExternalNode
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat
from mojo_rl.nn.storage.primitives.add import Add
from mojo_rl.nn.storage.primitives.scale import Scale
from mojo_rl.nn.storage.primitives.binary_elementwise import BinarySub
from mojo_rl.nn.storage.primitives.rsample import RSample

from .nets_mt import TDMPC2PolicyMT, TDMPC2QNetMT
from .losses import TwoHotDecode


comptime TDMPC2PolicyGraphMT[
    LATENT: Int,
    MAX_ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
] = ComputeGraph[
    InputSlot["z", LATENT],
    InputSlot["task_emb", TASK_EMB],
    Node["zt", Concat[LATENT, TASK_EMB], "z", "task_emb"],
    ExternalNode["pi_out", TDMPC2PolicyMT[LATENT, MAX_ACT, MLP, TASK_EMB], "zt"],
    Node["alp", RSample[MAX_ACT], "pi_out"],
    Node["action", Slice[MAX_ACT + 1, 0, MAX_ACT], "alp"],
    Node["log_prob", Slice[MAX_ACT + 1, MAX_ACT, MAX_ACT + 1], "alp"],
    Node["za", Concat[LATENT, MAX_ACT, TASK_EMB], "z", "action", "task_emb"],
    ExternalNode["q1", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"],
    ExternalNode["q2", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"],
    Node["q1d", TwoHotDecode[BINS, VMIN, VMAX], "q1"],
    Node["q2d", TwoHotDecode[BINS, VMIN, VMAX], "q2"],
    Node["qsum", Add[1], "q1d", "q2d"],
    Node["qscaled", Scale[1], "qsum"],
    Node["alpha_lp", Scale[1], "log_prob"],
    Node["loss", BinarySub[1], "alpha_lp", "qscaled"],
]
