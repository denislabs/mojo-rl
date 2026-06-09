"""TD-MPC2 multi-task policy (actor) loss ComputeGraph (item C, §14.3).

Clone of `policy_graph.mojo` with a `task_emb [TASK_EMB]` input slot threaded
into BOTH the policy input (`zt = Concat[LATENT, TASK_EMB]("z","task_emb")`) and
the detached-Q input (`za = Concat[LATENT, MAX_ACT, TASK_EMB]`). After
`graph.vjp`, `grad_input_ptr["task_emb"]` carries the actor-loss gradient w.r.t.
the embedding (collected into the table — site 3 of the embedding grad flow).
See `policy_graph.mojo` for the loss structure.
"""

from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.concat import Concat
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.deep_agents2.primitives.rsample import RSample

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
    1,
    InputSlot["z", LATENT],
    InputSlot["task_emb", TASK_EMB],
    Node["zt", Concat[LATENT, TASK_EMB], "z", "task_emb"],
    ExternalNode["pi_out", TDMPC2PolicyMT[LATENT, MAX_ACT, MLP, TASK_EMB], "zt"],
    ExternalNode["alp", RSample[MAX_ACT], "pi_out"],
    Node["action", Slice[MAX_ACT + 1, 0, MAX_ACT], "alp"],
    Node["log_prob", Slice[MAX_ACT + 1, MAX_ACT, MAX_ACT + 1], "alp"],
    Node["za", Concat[LATENT, MAX_ACT, TASK_EMB], "z", "action", "task_emb"],
    ExternalNode[
        "q1", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za",
        MODE="input_only",
    ],
    ExternalNode[
        "q2", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za",
        MODE="input_only",
    ],
    Node["q1d", TwoHotDecode[BINS, VMIN, VMAX], "q1"],
    Node["q2d", TwoHotDecode[BINS, VMIN, VMAX], "q2"],
    Node["qsum", Add[1, 2], "q1d", "q2d"],
    Node["qscaled", Scale[1], "qsum"],
    Node["alpha_lp", Scale[1], "log_prob"],
    Node["loss", BinarySub[1], "alpha_lp", "qscaled"],
]
