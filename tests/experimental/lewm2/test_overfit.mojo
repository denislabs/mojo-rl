"""LeWM JEPA — overfit a fixed batch (Phase D, CPU).

Drives the full training loop over the JEPA loss graph:
  zero_grad → set_input → forward → mean-reduce → seed 1/B → vjp → Adam.step
on a single fixed (pixels, actions) batch, and asserts the loss drops
substantially. This validates the complete trainer wiring (and that the
AdaLN scale/shift projections activate once the gates move off zero — the
loss can't keep falling otherwise). Real-data training swaps the fixed
batch for an offline-buffer sampler.
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.combinators import ComputeGraph, InputSlot, Node, Tokenwise
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.stop_grad import StopGrad
from mojo_rl.nn2.primitives.bias_add import BiasAdd
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.primitives.mse_per_sample import MSEPerSample
from mojo_rl.nn2.primitives.sigreg import SIGReg
from mojo_rl.experimental.lewm2.encoder import (
    LeWMEncoder, ActionEmbedder, ARPredictor, PredProj,
)


comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
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
comptime B = 4
comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime TE = T * EMB
comptime HE = H * EMB

comptime Encoder = LeWMEncoder[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS,
    EMB, ENC_PROJ_H, FF_MULT
]

comptime LossGraph = ComputeGraph[
    1,
    InputSlot["pixels", PIX],
    InputSlot["actions", ACTIN],
    Node["emb", Tokenwise[T, Encoder], "pixels"],
    Node["act_emb", ActionEmbedder[T, ACT, SMOOTHED, EMB, AE_MLP], "actions"],
    Node["ctx_x", Slice[TE, 0, HE], "emb"],
    Node["ctx_a", Slice[TE, 0, HE], "act_emb"],
    Node["tgt_raw", Slice[TE, N_PREDS * EMB, (N_PREDS + H) * EMB], "emb"],
    Node["tgt", StopGrad[HE], "tgt_raw"],
    Node["x_pe", BiasAdd[HE], "ctx_x"],
    Node["pred_raw", ARPredictor[EMB, PRED_HEADS, H, PRED_FF, DEPTH],
         "x_pe", "ctx_a"],
    Node["pred", PredProj[H, EMB, PRED_PROJ_H], "pred_raw"],
    Node["pl", MSEPerSample[HE], "pred", "tgt"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "emb"],
    Node["sig_s", Scale[1], "sig"],
    Node["loss", Add[1, 2], "pl", "sig_s"],
]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def test_overfit() raises:
    print("test_overfit ...")
    var g = LossGraph.make[target="cpu", INIT=Kaiming]()
    g.set_node_attr["sig_s", "multiplier"](Scalar[DT](0.09))
    var opt = Adam.make_graph["cpu"](g)
    opt.lr = Scalar[DT](1e-3)

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var loss = _a(B * 1); var gseed = _a(B * 1)
    for k in range(B * PIX):
        pix[k] = _det(k + 1, 1.0)
    for k in range(B * ACTIN):
        act[k] = _det(k + 7, 1.0)
    for b in range(B):
        gseed[b] = Scalar[DT](1.0 / Float64(B))

    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())
    var loss_t = TileTensor(loss, row_major[B, 1]())
    var gseed_t = TileTensor(gseed, row_major[B, 1]())
    g.set_input["pixels", B](pix_t)
    g.set_input["actions", B](act_t)

    comptime STEPS = 150
    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    for s in range(STEPS):
        opt.zero_grad_graph["cpu"](g)
        g.forward["cpu", B](loss_t)
        var m: Scalar[DT] = 0.0
        for b in range(B):
            m += loss[b]
        m /= Scalar[DT](B)
        if s == 0:
            first = m
        last = m
        if s % 30 == 0 or s == STEPS - 1:
            print("   step", s, " loss=", m)
        g.vjp["cpu", B](gseed_t)
        opt.step_graph["cpu"](g)

    print("   first=", first, " last=", last, " ratio=", last / first)
    assert_true(last < first, "loss must decrease")
    assert_true(last < Scalar[DT](0.6) * first,
                "loss must drop >40% overfitting a fixed batch")
    pix.free(); act.free(); loss.free(); gseed.free()
    _ = g^
    _ = opt^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM JEPA overfit (Phase D, CPU)")
    print("=" * 70)
    test_overfit()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
