"""TD-MPC2 multi-task world-model per-step ComputeGraph (item C, §14.3).

Clone of `wm_graph.mojo` with one extra input slot `task_emb [TASK_EMB]` and a
3-way `za = Concat[LATENT, ACT, TASK_EMB]("z","a","task_emb")` so the dynamics /
reward / Q / termination externals are conditioned on the task embedding. The
external module types are the `*MT` variants (first-layer fan-in widened by
TASK_EMB). Loss-column layout, `NLOSS`, `TERM_COL`, and `OUT_DIM = 8 + LATENT`
are unchanged — the embedding only routes gradient back to its slot (read via
`grad_input_ptr["task_emb"]`), it never appears in the output.

See `wm_graph.mojo` for the full column documentation.
"""

from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.nn.primitives.concat import Concat

from .nets_mt import (
    TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT, TDMPC2TerminationMT,
)
from .losses import MSELossPlain, TDMPC2TwoHotLoss, BCEWithLogitsLoss
from .wm_graph import NQ, NLOSS, TERM_COL


comptime TDMPC2WMGraphMT[
    LATENT: Int,
    MAX_ACT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
] = ComputeGraph[
    8 + LATENT,
    InputSlot["z", LATENT],
    InputSlot["a", MAX_ACT],
    InputSlot["task_emb", TASK_EMB],
    InputSlot["z_enc_next", LATENT],
    InputSlot["r", 1],
    InputSlot["td", 1],
    InputSlot["done", 1],
    Node["za", Concat[LATENT, MAX_ACT, TASK_EMB], "z", "a", "task_emb"],
    ExternalNode[
        "znext", TDMPC2DynamicsMT[LATENT, MAX_ACT, MLP, SN, TASK_EMB], "za"
    ],
    Node["cons", MSELossPlain[LATENT], "znext", "z_enc_next"],
    ExternalNode[
        "rlog", TDMPC2RewardMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB], "za"
    ],
    Node["rloss", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "rlog", "r"],
    ExternalNode[
        "q0", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"
    ],
    ExternalNode[
        "q1", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"
    ],
    ExternalNode[
        "q2", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"
    ],
    ExternalNode[
        "q3", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"
    ],
    ExternalNode[
        "q4", TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, TASK_EMB, QP], "za"
    ],
    ExternalNode[
        "term", TDMPC2TerminationMT[LATENT, MAX_ACT, MLP, TASK_EMB], "za"
    ],
    Node["v0", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q0", "td"],
    Node["v1", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q1", "td"],
    Node["v2", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q2", "td"],
    Node["v3", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q3", "td"],
    Node["v4", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q4", "td"],
    Node["tloss", BCEWithLogitsLoss, "term", "done"],
    Node["lv0", Concat[1, 1, 1, 1], "cons", "rloss", "v0", "v1"],
    Node["lv1", Concat[1, 1, 1, 1], "v2", "v3", "v4", "tloss"],
    Node["lossvec", Concat[4, 4], "lv0", "lv1"],
    Node["out", Concat[8, LATENT], "lossvec", "znext"],
]
