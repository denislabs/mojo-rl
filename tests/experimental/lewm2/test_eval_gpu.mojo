"""LeWM2 teacher-forced eval — GPU path smoke (Apple, toy scale).

Same as test_eval but target="gpu": validates the eval's device path
(scorer GPU pixel upload + action H2D, forward_into D2H readout, CEM /
shooter driving the device forward) at toy dims so it runs cheaply on
Apple — NOT the 84×84 model. Asserts the pipeline runs + scores finite.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm2/test_eval_gpu.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.eval import lewm2_action_awareness_eval


comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime ENC_FF_MULT = 2
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

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def main() raises:
    print("=" * 70)
    print("LeWM2 teacher-forced eval — GPU smoke (toy)")
    print("=" * 70)
    var ctx = DeviceContext()
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)

    # one fixed synthetic batch, device-resident, train a few steps
    var pix_d = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_d = ctx.enqueue_create_buffer[DT](B * ACTIN)
    var pix_h = ctx.enqueue_create_host_buffer[DT](B * PIX)
    var act_h = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    ctx.synchronize()
    for k in range(B * PIX):
        pix_h.unsafe_ptr()[k] = _det(k + 1)
    # one-hot actions
    for b in range(B):
        for t in range(T):
            for a in range(ACT):
                act_h.unsafe_ptr()[(b * T + t) * ACT + a] = Scalar[DT](
                    1.0 if a == ((b + t) % ACT) else 0.0
                )
    ctx.enqueue_copy(pix_d, pix_h)
    ctx.enqueue_copy(act_d, act_h)
    ctx.synchronize()
    var pix_t = TileTensor(_p(pix_d), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_d), row_major[B, ACTIN]())
    for _ in range(40):
        _ = tr.train_step(pix_t, act_t)

    # eval expects HOST pixels + HOST actions (it uploads internally).
    var pix_host = alloc[Scalar[DT]](B * PIX)
    var act_host = alloc[Scalar[DT]](B * ACTIN)
    for k in range(B * PIX):
        pix_host[k] = pix_h.unsafe_ptr()[k]
    for k in range(B * ACTIN):
        act_host[k] = act_h.unsafe_ptr()[k]

    print("eval (gpu) ...")
    var r = lewm2_action_awareness_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
    ](tr, pix_host, act_host, num_random=12, cem_iters=2, cem_samples=16,
      cem_topk=4, ctx=ctx)

    assert_true(not (isnan(r[0]) or isinf(r[0])), "expert finite")
    assert_true(not (isnan(r[2]) or isinf(r[2])), "random_min finite")
    assert_true(not (isnan(r[3]) or isinf(r[3])), "cem finite")
    assert_true(r[2] <= r[1] + 1e-9, "random_min <= random_mean")

    pix_host.free(); act_host.free()
    _ = tr^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
