"""LeWM JEPA — overfit a fixed batch (storage, CPU).

Drives the full training loop over the JEPA loss graph (the library
`LeWMLossGraph`):
  zero_grad → set_input → forward → mean-reduce → seed 1/B → vjp → Adam.step
on a single fixed (pixels, actions) batch, and asserts the loss drops
substantially — validating the complete graph/optimizer wiring on the storage
surface (and that the AdaLN scale/shift projections activate once the gates
move off zero).
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn import Tensor, Adam, Kaiming
from mojo_rl.experimental.lewm.loss_graph import LeWMLossGraph


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
comptime B = 4
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


def test_overfit() raises:
    print("test_overfit ...")
    var g = LossGraph.make["cpu", Kaiming]()
    g.set_node_attr["sig_s", "multiplier"](Scalar[DT](0.09))
    var opt = Adam(lr=Scalar[DT](1e-3))

    var pix = Tensor.alloc(B * PIX)
    var act = Tensor.alloc(B * ACTIN)
    for k in range(B * PIX):
        pix.data[k] = _det(k + 1, 1.0)
    for k in range(B * ACTIN):
        act.data[k] = _det(k + 7, 1.0)
    g.set_input["pixels", B](pix, None)
    g.set_input["actions", B](act, None)

    var loss = Tensor.alloc(B * 1)
    var gseed = Tensor.alloc(B * 1)
    for b in range(B):
        gseed.data[b] = Scalar[DT](1.0 / Float64(B))

    comptime STEPS = 150
    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    for s in range(STEPS):
        g.zero_grad["cpu"](None)
        g.forward[B, "cpu"](loss, None)
        var m: Scalar[DT] = 0.0
        for b in range(B):
            m += loss.data[b]
        m /= Scalar[DT](B)
        if s == 0:
            first = m
        last = m
        if s % 30 == 0 or s == STEPS - 1:
            print("   step", s, " loss=", m)
        g.vjp[B, "cpu"](gseed, None)
        opt.begin_step()
        g.for_each_param["cpu"](opt, None)

    print("   first=", first, " last=", last, " ratio=", last / first)
    assert_true(last < first, "loss must decrease")
    assert_true(last < Scalar[DT](0.6) * first,
                "loss must drop >40% overfitting a fixed batch")
    _ = g^
    _ = opt^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM JEPA overfit (storage, CPU)")
    print("=" * 70)
    test_overfit()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
