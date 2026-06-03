"""TD-MPC2 world-model per-step ComputeGraph (fixed 5 Q-heads).

One graph instance, driven `HORIZON` times by `WMStep`. All losses derive
from the step inputs (`z`, `a`) so a single `vjp` accumulates every head's
param grad AND routes the carry gradient back to `z` for BPTT.

Inputs:
  "z"          [LATENT]  carry-in (z_t; z_0 = encode(obs[0]))
  "a"          [ACT]     action a_t (data)
  "z_enc_next" [LATENT]  stop-grad encode(obs[t+1]) — consistency target
  "r"          [1]       scalar reward r_t (data; symlog+two-hot inside op)
  "td"         [1]       stop-grad scalar TD target (data) — value target

Output `[B, 7 + LATENT]`:
  col 0        consistency loss      (MSE(znext, z_enc_next))
  col 1        reward loss           (two-hot CE)
  cols 2..6    value losses v0..v4   (two-hot CE per Q head)
  cols 7..end  znext  (carry passthrough for BPTT — routed, not computed)

`znext` is consumed by BOTH the consistency loss (col 0) and the carry
passthrough (cols 7..), so at backward it accumulates the consistency
gradient + the next-step carry gradient — exactly the BPTT contract.

NQ is fixed at 5 (reference `num_q`); the graph node list is written out
(ComputeGraph node packs are not variadic).
"""

from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.nn2.primitives.concat import Concat

from .nets import TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet
from .losses import MSELossPlain, TDMPC2TwoHotLoss


comptime NQ = 5


comptime TDMPC2WMGraph[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
] = ComputeGraph[
    7 + LATENT,
    InputSlot["z", LATENT],
    InputSlot["a", ACT],
    InputSlot["z_enc_next", LATENT],
    InputSlot["r", 1],
    InputSlot["td", 1],
    Node["za", Concat[LATENT, ACT], "z", "a"],
    ExternalNode["znext", TDMPC2Dynamics[LATENT, ACT, MLP, SN], "za"],
    Node["cons", MSELossPlain[LATENT], "znext", "z_enc_next"],
    ExternalNode["rlog", TDMPC2Reward[LATENT, ACT, MLP, BINS], "za"],
    Node["rloss", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "rlog", "r"],
    ExternalNode["q0", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za"],
    ExternalNode["q1", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za"],
    ExternalNode["q2", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za"],
    ExternalNode["q3", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za"],
    ExternalNode["q4", TDMPC2QNet[LATENT, ACT, MLP, BINS], "za"],
    Node["v0", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q0", "td"],
    Node["v1", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q1", "td"],
    Node["v2", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q2", "td"],
    Node["v3", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q3", "td"],
    Node["v4", TDMPC2TwoHotLoss[BINS, VMIN, VMAX], "q4", "td"],
    Node["lv0", Concat[1, 1, 1, 1], "cons", "rloss", "v0", "v1"],
    Node["lv1", Concat[1, 1, 1], "v2", "v3", "v4"],
    Node["lossvec", Concat[4, 3], "lv0", "lv1"],
    Node["out", Concat[7, LATENT], "lossvec", "znext"],
]
