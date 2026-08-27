"""LeWM JEPA loss graph — forward+backward smoke (storage, CPU).

The whole objective as one storage ComputeGraph (the SAC-actor-loss analogue),
imported from the library `LeWMLossGraph` so this gates the migrated graph:

  pixels ─Tokenwise[T,Encoder]→ emb ─┬─Slice[0:H]→ ctx_x ─BiasAdd→ x_pe ─┐
                                     ├─Slice[Np:Np+H]→ tgt                ├ARPredictor→PredProj→pred
  actions ─ActionEmbedder→ act_emb ──┴─Slice[0:H]→ ctx_a ────────────────┘
  loss = MSEPerSample(pred, tgt) + λ·SIGReg(emb)        (per-sample, (B,1))

Validates: graph compiles, forward finite, vjp delivers non-zero grads to a
strict majority of parameter groups (AdaLN scale/shift projections are gated
to zero at init by design).
"""

from std.math import isnan, isinf
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn import Tensor, ParamVisitor, Kaiming
from mojo_rl.experimental.lewm.loss_graph import LeWMLossGraph


# toy config
comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime FF_MULT = 2

comptime T = 4
comptime ACT = 3
comptime SMOOTHED = 8
comptime AE_MLP = 2

comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 16
comptime DEPTH = 2
comptime PRED_PROJ_H = 16

comptime SIG_PROJ = 8
comptime SIG_KNOTS = 5

comptime B = 2
comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT


comptime LossGraph = LeWMLossGraph[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H, FF_MULT,
    T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF, DEPTH,
    PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
]


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


struct GradStats(ParamVisitor):
    var n_params: Int
    var n_nonzero: Int

    def __init__(out self):
        self.n_params = 0
        self.n_nonzero = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.n_params += 1
        for i in range(N):
            if grad.data[i] != Scalar[DT](0.0):
                self.n_nonzero += 1
                return


def test_loss_graph_smoke() raises:
    print("test_loss_graph_smoke ...")
    var g = LossGraph.make["cpu", Kaiming]()
    g.set_node_attr["sig_s", "multiplier"](Scalar[DT](0.09))

    var pix = Tensor.alloc(B * PIX)
    var act = Tensor.alloc(B * ACTIN)
    for k in range(B * PIX):
        pix.data[k] = _det(k + 1, 1.0)
    for k in range(B * ACTIN):
        act.data[k] = _det(k + 7, 1.0)

    g.set_input["pixels", B](pix, None)
    g.set_input["actions", B](act, None)
    var loss = Tensor.alloc(B * 1)
    g.forward[B, "cpu"](loss, None)

    var mean_loss: Scalar[DT] = 0.0
    for b in range(B):
        assert_true(not (isnan(loss.data[b]) or isinf(loss.data[b])),
                    "loss must be finite")
        mean_loss += loss.data[b]
    mean_loss /= Scalar[DT](B)
    print("   mean loss =", mean_loss)
    assert_true(mean_loss >= Scalar[DT](0.0), "loss >= 0 (MSE + λ·SIGReg)")

    # seed grad = 1/B, backward.
    var gseed = Tensor.alloc(B * 1)
    for b in range(B):
        gseed.data[b] = Scalar[DT](1.0 / Float64(B))
    g.vjp[B, "cpu"](gseed, None)

    var gs = GradStats()
    g.for_each_param["cpu", GradStats](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "graph has params")
    # AdaLN-zero init gates the per-block scale/shift modulation projections to
    # zero grad at step 0 (they activate as the gates move off zero). The rest
    # (encoder, AE, gates, pred-proj, pos-embed, SIGReg path) is grad-bearing.
    assert_true(gs.n_nonzero * 2 >= gs.n_params,
                "≥50% of params receive grad at init (AdaLN scale/shift gated)")
    _ = g^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM JEPA loss graph smoke (storage, CPU)")
    print("=" * 70)
    test_loss_graph_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
