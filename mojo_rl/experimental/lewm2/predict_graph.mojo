"""LeWMPredictGraph — predictor-from-latents (the MPC inference path).

The loss graph couples encoder→predictor (its context always comes from
encoding pixels). Autoregressive MPC needs the predictor to run on ARBITRARY
latent context (the rolling buffer of predicted latents), so this is a
separate graph:

  latent_ctx (B, H·EMB) ─BiasAdd(pos)→ x_pe ─┐
  actions (B, T·ACT) ─ActionEmbedder→ ─Slice[0:H]→ ctx_a ─┴ARPredictor→PredProj→ pred

Its param-bearing nodes — `act_emb`, `x_pe`, `pred_raw`, `pred` — are the
SAME module types, with the SAME `for_each_param` names, as the matching
nodes in `LeWMLossGraph` (a suffix of its param set, minus the encoder
`emb`). So `LeWMPredictor.sync_from_named` copies the trained weights by
NAME from the trainer's exported dict — exact, order-independent.

The graph output is `pred` (B, H·EMB), so `forward` writes it straight to
the caller's output tile (no node_out_ptr needed). Forward-only; no params
of its own beyond the synced snapshot.
"""

from std.collections import Dict
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ...nn2.constants import DT
from ...nn2.core import ParamVisitor
from ...nn2.core.target_storage import TargetStorage, assert_tag_for
from ...nn2.initializer import Kaiming
from ...nn2.combinators import ComputeGraph, InputSlot, Node
from ...nn2.primitives.slice import Slice
from ...nn2.primitives.bias_add import BiasAdd
from .encoder import ActionEmbedder, ARPredictor, PredProj


comptime LeWMPredictGraph[
    EMB: Int, T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int, PRED_PROJ_H: Int,
] = ComputeGraph[
    H * EMB,
    InputSlot["latent_ctx", H * EMB],
    InputSlot["actions", T * ACT],
    Node["act_emb", ActionEmbedder[T, ACT, SMOOTHED, EMB, AE_MLP], "actions"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["x_pe", BiasAdd[H * EMB], "latent_ctx"],
    Node[
        "pred_raw",
        ARPredictor[EMB, PRED_HEADS, H, PRED_FF, DEPTH],
        "x_pe", "ctx_a",
    ],
    Node["pred", PredProj[H, EMB, PRED_PROJ_H], "pred_raw"],
]


# Import visitor: copy each param from a name→values dict (CPU direct /
# GPU H2D). Missing names are left at their init (shouldn't happen if the
# predict graph node names match the trainer's).
struct _NamedImportVisitor(ParamVisitor):
    var d: Dict[String, List[Scalar[DT]]]
    var ctx: Optional[DeviceContext]
    var missing: Int

    def __init__(
        out self, var d: Dict[String, List[Scalar[DT]]],
        ctx: Optional[DeviceContext] = None,
    ):
        self.d = d^
        self.ctx = ctx
        self.missing = 0

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        if name not in self.d:
            self.missing += 1
            return
        ref vals = self.d[name]
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        if self.ctx:
            var c = self.ctx.value()
            var host = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
            for i in range(n_elems):
                host[i] = vals[i]
            var dev = DeviceBuffer[DT](c, p, n_elems, owning=False)
            c.enqueue_copy(dev, host.unsafe_ptr())
            c.synchronize()
        else:
            for i in range(n_elems):
                p[i] = vals[i]


struct LeWMPredictor[
    EMB: Int, T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int, PRED_PROJ_H: Int,
    BATCH: Int, target: StaticString = "cpu",
](Movable & ImplicitlyDestructible):
    comptime PG = LeWMPredictGraph[
        Self.EMB, Self.T, Self.ACT, Self.SMOOTHED, Self.AE_MLP,
        Self.H, Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH, Self.PRED_PROJ_H,
    ]
    comptime HE = Self.H * Self.EMB
    comptime ACTIN = Self.T * Self.ACT

    var graph: Self.PG
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.PG()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make(ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.graph = Self.PG.make[target = Self.target, INIT=Kaiming](ctx=ctx)
        p.ts = TargetStorage.make[Self.target](ctx=ctx)
        return p^

    def sync_from_named(
        mut self, var d: Dict[String, List[Scalar[DT]]],
    ) raises:
        """Overwrite the predictor's params with the trainer's snapshot,
        matched by `for_each_param` name."""
        var v = _NamedImportVisitor(d^, ctx=self.ts.ctx)
        self.graph.for_each_param[Self.target, _NamedImportVisitor]("", v)

    def forward(
        mut self,
        latent_ctx: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        actions: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut pred_out: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        """Run the predictor: (latent_ctx, actions) → pred (B, H·EMB),
        written to `pred_out`."""
        assert_tag_for["LeWMPredictor", Self.target](self.ts.target_tag)
        self.graph.set_input["latent_ctx", Self.BATCH](latent_ctx)
        self.graph.set_input["actions", Self.BATCH](actions)
        self.graph.forward[Self.target, Self.BATCH](pred_out)
