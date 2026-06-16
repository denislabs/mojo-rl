"""LeWMLossGraph — the full JEPA objective as one ComputeGraph (parameterized).

All components are owned Nodes (LeWM has a single objective — no nets shared
across graphs as in SAC — so one Adam iterates `graph.for_each_param`). The
encoder bridges effective BATCH = B·T → B via `Tokenwise[T, LeWMEncoder]`.

  pixels  ─Tokenwise[T,Encoder]→ emb ─┬─Slice[0:H]→ ctx_x ─BiasAdd→ x_pe ─┐
                                      ├─Slice[Np:Np+H]→ tgt               ├ARPredictor→PredProj→pred
  actions ─ActionEmbedder→ act_emb ───┴─Slice[0:H]→ ctx_a ────────────────┘
  loss = MSEPerSample(pred, tgt) + λ·SIGReg(emb)         (per-sample (B,1))

NO stop-gradient on the target — the reference flows gradients through
`tgt_emb = emb[:, n_preds:]` (the paper's headline: end-to-end, no SG/EMA;
gradient through the target pulls the encoder toward predictable
representations, with SIGReg as the sole anti-collapse term). Our original
port detached `tgt` — a deviation removed 2026-06-10 (reference audit).

λ is the `sig_s` Scale multiplier — `set_node_attr["sig_s","multiplier"](λ)`.
The collapse probes read the `emb` node output via `node_out_ptr["emb"]`.
"""

from ...nn.core.module import Module
from ...nn.combinators import ComputeGraph, InputSlot, Node, Tokenwise
from ...nn.primitives.slice import Slice
from ...nn.primitives.bias_add import BiasAdd
from ...nn.primitives.scale import Scale
from ...nn.primitives.add import Add
from ...nn.primitives.mse_per_sample import MSEPerSample
from ...nn.primitives.sigreg import SIGReg
from .encoder import LeWMEncoder, ActionEmbedder, ARPredictor, PredProj


comptime LeWMLossGraph[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    ENC_PROJ_H: Int,
    ENC_FF_MULT: Int,
    T: Int,
    ACT: Int,
    SMOOTHED: Int,
    AE_MLP: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
    DEPTH: Int,
    PRED_PROJ_H: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
    PRED_DIM_HEAD: Int = 0,
    # Encoder type — defaults to the mean-pooled `LeWMEncoder`. Pass
    # `LeWMEncoderCLS[...same dims...]` to train the CLS-token variant
    # (image→(B,EMB) interface is identical, so the graph is unchanged
    # elsewhere). A trailing type param with a dims-derived default keeps
    # every existing caller (which omits it) bit-identical.
    ENC: Module = LeWMEncoder[
        IN_CH, IMG, PATCH, (IMG // PATCH) * (IMG // PATCH),
        HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H, ENC_FF_MULT,
    ],
] = ComputeGraph[
    1,
    InputSlot["pixels", T * IN_CH * IMG * IMG],
    InputSlot["actions", T * ACT],
    Node["emb", Tokenwise[T, ENC], "pixels"],
    Node["act_emb", ActionEmbedder[T, ACT, SMOOTHED, EMB, AE_MLP], "actions"],
    Node["ctx_x", Slice[T * EMB, 0, H * EMB], "emb"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["tgt", Slice[T * EMB, N_PREDS * EMB, (N_PREDS + H) * EMB], "emb"],
    Node["x_pe", BiasAdd[H * EMB], "ctx_x"],
    Node[
        "pred_raw",
        ARPredictor[EMB, PRED_HEADS, H, PRED_FF, DEPTH, PRED_DIM_HEAD],
        "x_pe", "ctx_a",
    ],
    Node["pred", PredProj[H, EMB, PRED_PROJ_H], "pred_raw"],
    Node["pl", MSEPerSample[H * EMB], "pred", "tgt"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "emb"],
    Node["sig_s", Scale[1], "sig"],
    Node["loss", Add[1, 2], "pl", "sig_s"],
]
