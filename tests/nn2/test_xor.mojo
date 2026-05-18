"""End-to-end XOR sanity test — pulls together every Phase 1 component.

Network: Sequential2(Sequential2(Linear[2, 4], ReLU[4]), Linear[4, 2])
Loss:    CrossEntropyLoss[2] (one-hot targets)
Optim:   Adam (default hyperparams except lr=0.05)
Data:    XOR truth table — 4 samples, 2-class output

Goal: 100% classification accuracy after a few hundred steps. This is
the canonical "does the framework actually train" sanity check. If this
passes, MNIST is just a bigger version of the same recipe.

Init: hand-picked deterministic weights — small, broken-symmetry. We
don't have an Initializer abstraction yet (Phase 1 deferred), and a
fully-zero init would deadlock under ReLU (no gradient flow).
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential2
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam


def test_xor_converges() raises:
    comptime IN = 2
    comptime HID = 4
    comptime OUT = 2
    comptime BATCH = 4   # train on all 4 samples per step
    comptime N_STEPS = 1500
    comptime LR: Scalar[DT] = 0.05

    # ── Construct network ─────────────────────────────────────────────
    var lin0 = Linear[IN, HID]()
    var lin1 = Linear[HID, OUT]()

    # Hand-picked init — broken-symmetry, small magnitude.
    # Linear[2, 4]: weight is 2x4.
    var w0 = TileTensor(lin0.weight, row_major[IN, HID]())
    w0[0, 0] =  0.50; w0[0, 1] = -0.30; w0[0, 2] =  0.70; w0[0, 3] = -0.40
    w0[1, 0] = -0.20; w0[1, 1] =  0.40; w0[1, 2] = -0.60; w0[1, 3] =  0.50

    # Linear[4, 2]: weight is 4x2.
    var w1 = TileTensor(lin1.weight, row_major[HID, OUT]())
    w1[0, 0] =  0.30; w1[0, 1] = -0.40
    w1[1, 0] = -0.50; w1[1, 1] =  0.60
    w1[2, 0] =  0.40; w1[2, 1] = -0.30
    w1[3, 0] = -0.60; w1[3, 1] =  0.50

    # Bias init: small nonzero values for lin0 so ReLU pre-activation
    # at input (0, 0) doesn't sit on the boundary x=0 (gradient is 0
    # there by ReLU convention, which would kill learning for that
    # sample). Hand-picked distinct values further break symmetry.
    var b0 = TileTensor(lin0.bias, row_major[HID]())
    b0[0] = 0.10
    b0[1] = -0.05
    b0[2] = 0.15
    b0[3] = -0.10

    var net = Sequential2(Sequential2(lin0^, ReLU[HID]()), lin1^)
    var loss_fn = CrossEntropyLoss[OUT]()
    var optim = Adam.make(net, lr=LR)

    # ── XOR data ──────────────────────────────────────────────────────
    # (0, 0) → 0;  (0, 1) → 1;  (1, 0) → 1;  (1, 1) → 0
    var in_buf:    UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var tgt_buf:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    in_buf[0] = 0.0; in_buf[1] = 0.0
    in_buf[2] = 0.0; in_buf[3] = 1.0
    in_buf[4] = 1.0; in_buf[5] = 0.0
    in_buf[6] = 1.0; in_buf[7] = 1.0
    # One-hot targets — true class is given by XOR(x0, x1).
    for k in range(BATCH * OUT):
        tgt_buf[k] = 0.0
    tgt_buf[0 * OUT + 0] = 1.0   # class 0
    tgt_buf[1 * OUT + 1] = 1.0   # class 1
    tgt_buf[2 * OUT + 1] = 1.0   # class 1
    tgt_buf[3 * OUT + 0] = 1.0   # class 0

    var input   = TileTensor(in_buf,  row_major[BATCH, IN]())
    var targets = TileTensor(tgt_buf, row_major[BATCH, OUT]())

    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var output     = TileTensor(out_buf, row_major[BATCH, OUT]())
    var grad_out   = TileTensor(go_buf,  row_major[BATCH, OUT]())
    var grad_input = TileTensor(gi_buf,  row_major[BATCH, IN]())

    # ── Training loop ─────────────────────────────────────────────────
    var initial_loss: Scalar[DT] = 0.0
    var final_loss: Scalar[DT] = 0.0
    for step_i in range(N_STEPS):
        net.first.first.zero_grad()
        net.second.zero_grad()

        net.forward[BATCH](input, output)
        var L = loss_fn.forward[BATCH](output, targets)
        if step_i == 0:
            initial_loss = L
        final_loss = L

        loss_fn.backward[BATCH](targets, grad_out)
        net.backward[BATCH](grad_out, grad_input)
        optim.step(net)

    # Loss should drop substantially.
    assert_true(final_loss < initial_loss * Scalar[DT](0.5),
        "Expected loss to drop >2x. initial=" + String(initial_loss)
        + " final=" + String(final_loss))

    # ── Inference: check accuracy ────────────────────────────────────
    net.forward[BATCH](input, output)
    var n_correct = 0
    var expected_class = List[Int]()
    expected_class.append(0)
    expected_class.append(1)
    expected_class.append(1)
    expected_class.append(0)
    for b in range(BATCH):
        # argmax over OUT
        var best_c: Int = 0
        var best_v: Scalar[DT] = output[b, 0]
        for c in range(1, OUT):
            if output[b, c] > best_v:
                best_v = output[b, c]
                best_c = c
        if best_c == expected_class[b]:
            n_correct += 1
    print("  XOR accuracy: " + String(n_correct) + "/" + String(BATCH)
          + " | initial_loss=" + String(initial_loss)
          + " final_loss=" + String(final_loss))
    assert_equal(n_correct, BATCH)

    in_buf.free()
    tgt_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_xor_converges PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 XOR end-to-end (CPU, Phase 1)")
    print("=" * 60)
    test_xor_converges()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
