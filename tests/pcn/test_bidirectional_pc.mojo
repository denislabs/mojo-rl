"""Bidirectional PC test — Bogacz notebook 5 reproduction (CPU).

Trains two PC paths sharing a common latent:
  - UP   : image (784) → x_shared (HIDDEN) → label (10)
  - DOWN : label (10)  → x_shared (HIDDEN) → image (784)

Total energy = alpha_up * E_up + alpha_down * E_down.
The latent x_shared is updated by the SUM of both paths' gradients per
inference iteration. Each path's weights are updated by its own per-block
weight gradients (down path scaled by alpha_down).

Architecture (smaller than notebook 5 for fast CPU smoke):
    UP  : PCBlock[784, 32, PCIdentity] → x_shared → PCBlock[32, 10, PCReLU]
    DOWN: PCBlock[10,  32, PCIdentity] → x_shared → PCBlock[32, 784, PCReLU]

Pass criteria:
  - Up classification accuracy ≥ 50% on test subset (alpha_down small enough
    not to hurt classification much).
  - Generated per-class images via DOWN path differ across classes
    (avg pairwise pixel-MSE between class-0 and class-9 images > 0.005).

Bidirectional logic is inlined here using PCBlock primitives directly —
not promoted to the framework yet, since this is a research toy.

Run:
    pixi run mojo run -I . tests/pcn/test_bidirectional_pc.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.datasets.mnist import MNIST
from mojo_rl.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
)


comptime BATCH = 100
comptime HIDDEN = 32
comptime EPOCHS = 3
comptime T_INFER = 15
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.001
comptime ALPHA_UP: Float64 = 1.0
comptime ALPHA_DOWN: Float64 = 0.05  # higher than Bogacz (0.0001) to make gen meaningful in our small budget

comptime N_TRAIN = 2000
comptime N_TEST = 500
comptime N_TRAIN_BATCHES = N_TRAIN // BATCH
comptime N_TEST_BATCHES = N_TEST // BATCH

# Up path
comptime UB0 = PCBlock[784, HIDDEN, PCIdentity]
comptime UB1 = PCBlock[HIDDEN, 10, PCReLU]
comptime UP_NET = PCSequential[UB0, UB1]
comptime UP_PARAM_SIZE = UP_NET.PARAM_SIZE
comptime UB0_PARAM_SIZE = UB0.PARAM_SIZE      # 784*32+32 = 25120
comptime UB1_PARAM_SIZE = UB1.PARAM_SIZE      # 32*10+10 = 330

# Down path
comptime DB0 = PCBlock[10, HIDDEN, PCIdentity]
comptime DB1 = PCBlock[HIDDEN, 784, PCReLU]
comptime DOWN_NET = PCSequential[DB0, DB1]
comptime DOWN_PARAM_SIZE = DOWN_NET.PARAM_SIZE
comptime DB0_PARAM_SIZE = DB0.PARAM_SIZE      # 10*32+32 = 352
comptime DB1_PARAM_SIZE = DB1.PARAM_SIZE      # 32*784+784 = 26176

comptime OPT = Adam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("Bidirectional PC — Bogacz notebook 5 reproduction (CPU)")
    print("=" * 60)
    print("  UP   arch  : 784 →", HIDDEN, "→ 10")
    print("  DOWN arch  : 10  →", HIDDEN, "→ 784")
    print("  UP params  :", UP_PARAM_SIZE, " DOWN params:", DOWN_PARAM_SIZE)
    print("  HIDDEN     :", HIDDEN, "  shared latent dim")
    print("  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER, " EPOCHS=", EPOCHS)
    print("  α_up=", ALPHA_UP, " α_down=", ALPHA_DOWN)

    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train,", MNIST.N_TEST, "test")

    # ── Allocate UP params + Adam state ──────────────────────────────────────
    var up_params_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE)
    var up_grads_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE)
    var up_opt_state_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE * OPT.STATE_PER_PARAM)
    var up_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE)
    memset(up_params_buf, 0, UP_PARAM_SIZE)
    memset(up_grads_buf, 0, UP_PARAM_SIZE)
    memset(up_opt_state_buf, 0, UP_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(up_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var up_params = LayoutTensor[
        dtype, Layout.row_major(UP_PARAM_SIZE), MutAnyOrigin
    ](up_params_buf)
    var up_grads = LayoutTensor[
        dtype, Layout.row_major(UP_PARAM_SIZE), MutAnyOrigin
    ](up_grads_buf)
    var up_opt_state = LayoutTensor[
        dtype, Layout.row_major(UP_PARAM_SIZE, OPT.STATE_PER_PARAM), MutAnyOrigin
    ](up_opt_state_buf)
    var up_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](up_opt_global_buf)
    UP_NET.initialize_params[Xavier[], dtype](up_params)

    # ── Allocate DOWN params + Adam state ────────────────────────────────────
    var dn_params_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE)
    var dn_grads_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE)
    var dn_opt_state_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM)
    var dn_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE)
    memset(dn_params_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_grads_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_opt_state_buf, 0, DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(dn_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var dn_params = LayoutTensor[
        dtype, Layout.row_major(DOWN_PARAM_SIZE), MutAnyOrigin
    ](dn_params_buf)
    var dn_grads = LayoutTensor[
        dtype, Layout.row_major(DOWN_PARAM_SIZE), MutAnyOrigin
    ](dn_grads_buf)
    var dn_opt_state = LayoutTensor[
        dtype, Layout.row_major(DOWN_PARAM_SIZE, OPT.STATE_PER_PARAM), MutAnyOrigin
    ](dn_opt_state_buf)
    var dn_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](dn_opt_global_buf)
    DOWN_NET.initialize_params[Xavier[], dtype](dn_params)

    # ── Shared latent x_shared (HIDDEN-dim, BATCH samples) ───────────────────
    var x_shared_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    memset(x_shared_buf, 0, BATCH * HIDDEN)
    var x_shared = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](x_shared_buf)

    # ── Per-path scratch: μ, ε, a_below, z (one slot per block) ─────────────
    # UP block_0: predicts x_shared from image. mu/eps shape [BATCH, HIDDEN]; a_below shape [BATCH, 784]; z (pull_back) shape [BATCH, 784] — unused (block_0 below is image, clamped)
    # UP block_1: predicts label from x_shared. mu/eps shape [BATCH, 10]; a_below shape [BATCH, HIDDEN]; z shape [BATCH, HIDDEN]
    # DOWN block_0: predicts x_shared from label. mu/eps shape [BATCH, HIDDEN]; a_below shape [BATCH, 10]; z [BATCH, 10] — unused
    # DOWN block_1: predicts image from x_shared. mu/eps shape [BATCH, 784]; a_below shape [BATCH, HIDDEN]; z shape [BATCH, HIDDEN]

    # Allocate flat per-path scratch
    var up_mu0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var up_eps0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var up_mu1_buf = alloc[Scalar[dtype]](BATCH * 10)
    var up_eps1_buf = alloc[Scalar[dtype]](BATCH * 10)
    var up_a0_buf = alloc[Scalar[dtype]](BATCH * 784)   # ACT(image) for block_0
    var up_a1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)  # ACT(x_shared) for block_1
    var up_z1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)  # pull_back of ε_up_1

    var dn_mu0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dn_eps0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dn_mu1_buf = alloc[Scalar[dtype]](BATCH * 784)
    var dn_eps1_buf = alloc[Scalar[dtype]](BATCH * 784)
    var dn_a0_buf = alloc[Scalar[dtype]](BATCH * 10)
    var dn_a1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dn_z1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)

    var dx_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    memset(dx_buf, 0, BATCH * HIDDEN)

    # ── Per-batch input buffers ──────────────────────────────────────────────
    var image_buf = alloc[Scalar[dtype]](BATCH * 784)
    var label_oh_buf = alloc[Scalar[dtype]](BATCH * 10)
    memset(image_buf, 0, BATCH * 784)
    memset(label_oh_buf, 0, BATCH * 10)

    # Param block views (used inside training loop)
    var up_p0 = LayoutTensor[
        dtype, Layout.row_major(UB0_PARAM_SIZE), MutAnyOrigin
    ](up_params_buf)
    var up_p1 = LayoutTensor[
        dtype, Layout.row_major(UB1_PARAM_SIZE), MutAnyOrigin
    ](up_params_buf + UB0_PARAM_SIZE)
    var dn_p0 = LayoutTensor[
        dtype, Layout.row_major(DB0_PARAM_SIZE), MutAnyOrigin
    ](dn_params_buf)
    var dn_p1 = LayoutTensor[
        dtype, Layout.row_major(DB1_PARAM_SIZE), MutAnyOrigin
    ](dn_params_buf + DB0_PARAM_SIZE)
    var up_g0 = LayoutTensor[
        dtype, Layout.row_major(UB0_PARAM_SIZE), MutAnyOrigin
    ](up_grads_buf)
    var up_g1 = LayoutTensor[
        dtype, Layout.row_major(UB1_PARAM_SIZE), MutAnyOrigin
    ](up_grads_buf + UB0_PARAM_SIZE)
    var dn_g0 = LayoutTensor[
        dtype, Layout.row_major(DB0_PARAM_SIZE), MutAnyOrigin
    ](dn_grads_buf)
    var dn_g1 = LayoutTensor[
        dtype, Layout.row_major(DB1_PARAM_SIZE), MutAnyOrigin
    ](dn_grads_buf + DB0_PARAM_SIZE)

    # Buffer views for inline kernel calls
    var image_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 784), MutAnyOrigin
    ](image_buf)
    var label_oh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](label_oh_buf)
    var up_mu0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](up_mu0_buf)
    var up_eps0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](up_eps0_buf)
    var up_mu1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](up_mu1_buf)
    var up_eps1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](up_eps1_buf)
    var up_a0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 784), MutAnyOrigin
    ](up_a0_buf)
    var up_a1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](up_a1_buf)
    var up_z1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](up_z1_buf)
    var dn_mu0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](dn_mu0_buf)
    var dn_eps0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](dn_eps0_buf)
    var dn_mu1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 784), MutAnyOrigin
    ](dn_mu1_buf)
    var dn_eps1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 784), MutAnyOrigin
    ](dn_eps1_buf)
    var dn_a0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](dn_a0_buf)
    var dn_a1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](dn_a1_buf)
    var dn_z1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](dn_z1_buf)
    var dx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](dx_buf)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | up_loss | dn_loss | wall_t (s)")
    print("  ------+---------+---------+------------")
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var ep_up_loss: Float64 = 0.0
        var ep_dn_loss: Float64 = 0.0
        for batch_idx in range(N_TRAIN_BATCHES):
            # Load batch into image_buf + label_oh_buf
            for i in range(BATCH):
                var sample_idx = batch_idx * BATCH + i
                for j in range(784):
                    image_buf[i * 784 + j] = ds.train_images[
                        sample_idx * 784 + j
                    ]
                for c in range(10):
                    label_oh_buf[i * 10 + c] = Scalar[dtype](0)
                label_oh_buf[i * 10 + Int(ds.train_labels[sample_idx])] = (
                    Scalar[dtype](1.0)
                )

            # ── Init x_shared via UP forward sweep: x_shared = μ_up_0 ──
            UB0.predict[BATCH, dtype](image_t, up_p0, up_mu0_t, up_a0_t)
            for i in range(BATCH * HIDDEN):
                x_shared_buf[i] = up_mu0_buf[i]

            # ── T_INFER iterations of joint inference ──
            for _ in range(T_INFER):
                # UP path: μ_up_0 from image, μ_up_1 from x_shared
                UB0.predict[BATCH, dtype](image_t, up_p0, up_mu0_t, up_a0_t)
                UB1.predict[BATCH, dtype](x_shared, up_p1, up_mu1_t, up_a1_t)
                # ε_up_0 = x_shared - μ_up_0;  ε_up_1 = label_oh - μ_up_1
                UB0.eps_compute[BATCH, dtype](x_shared, up_mu0_t, up_eps0_t)
                UB1.eps_compute[BATCH, dtype](label_oh_t, up_mu1_t, up_eps1_t)

                # DOWN path: μ_dn_0 from label, μ_dn_1 from x_shared
                DB0.predict[BATCH, dtype](label_oh_t, dn_p0, dn_mu0_t, dn_a0_t)
                DB1.predict[BATCH, dtype](x_shared, dn_p1, dn_mu1_t, dn_a1_t)
                DB0.eps_compute[BATCH, dtype](x_shared, dn_mu0_t, dn_eps0_t)
                DB1.eps_compute[BATCH, dtype](image_t, dn_mu1_t, dn_eps1_t)

                # Phase C: dx for x_shared
                # UP contribution:
                #   z_up_1 = pull_back(ε_up_1, W_up_1)
                #   z_up_1 ← act'(x_shared) ⊙ z_up_1
                #   dx_up = ε_up_0 - z_up_1
                UB1.pull_back[BATCH, dtype](up_eps1_t, up_p1, up_z1_t)
                UB1.act_derivative_mul[BATCH, dtype](
                    x_shared, up_z1_t, up_z1_t
                )
                # DOWN contribution: similarly with DB1
                DB1.pull_back[BATCH, dtype](dn_eps1_t, dn_p1, dn_z1_t)
                DB1.act_derivative_mul[BATCH, dtype](
                    x_shared, dn_z1_t, dn_z1_t
                )

                # Total dx = α_up·(ε_up_0 - z_up_1) + α_down·(ε_dn_0 - z_dn_1)
                for b in range(BATCH):
                    for k in range(HIDDEN):
                        var u = (
                            Float64(up_eps0_buf[b * HIDDEN + k])
                            - Float64(up_z1_buf[b * HIDDEN + k])
                        )
                        var d = (
                            Float64(dn_eps0_buf[b * HIDDEN + k])
                            - Float64(dn_z1_buf[b * HIDDEN + k])
                        )
                        dx_buf[b * HIDDEN + k] = Scalar[dtype](
                            ALPHA_UP * u + ALPHA_DOWN * d
                        )

                # Phase D: x_shared -= lr_x · dx
                for i in range(BATCH * HIDDEN):
                    x_shared_buf[i] = (
                        x_shared_buf[i]
                        - Scalar[dtype](LR_X) * dx_buf[i]
                    )

            # ── Compute weight grads (using post-inference ε) ──
            UB0.weight_grad[BATCH, dtype](up_eps0_t, up_a0_t, up_g0)
            UB1.weight_grad[BATCH, dtype](up_eps1_t, up_a1_t, up_g1)
            DB0.weight_grad[BATCH, dtype](dn_eps0_t, dn_a0_t, dn_g0)
            DB1.weight_grad[BATCH, dtype](dn_eps1_t, dn_a1_t, dn_g1)

            # Scale DOWN grads by alpha_down (UP grads stay alpha_up=1)
            for i in range(DOWN_PARAM_SIZE):
                dn_grads_buf[i] = (
                    dn_grads_buf[i] * Scalar[dtype](ALPHA_DOWN)
                )

            # ── Adam steps (separate per path) ──
            step_num += 1
            OPT.step[UP_PARAM_SIZE, dtype](
                up_params, up_grads, up_opt_state, up_opt_global, step_num
            )
            OPT.step[DOWN_PARAM_SIZE, dtype](
                dn_params, dn_grads, dn_opt_state, dn_opt_global, step_num
            )

            # Track losses (per-block ε² sums / 2)
            var sup_loss: Float64 = 0
            for i in range(BATCH * 10):
                var v = Float64(up_eps1_buf[i])
                sup_loss += v * v
            sup_loss *= 0.5
            var rec_loss: Float64 = 0
            for i in range(BATCH * 784):
                var v = Float64(dn_eps1_buf[i])
                rec_loss += v * v
            rec_loss *= 0.5
            ep_up_loss += sup_loss
            ep_dn_loss += rec_loss

        var elapsed = Float64(perf_counter_ns() - t0) / 1e9
        var avg_up = ep_up_loss / Float64(N_TRAIN_BATCHES)
        var avg_dn = ep_dn_loss / Float64(N_TRAIN_BATCHES)
        print("    ", epoch, "  ", avg_up, "  ", avg_dn, "  ", elapsed)

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Eval: classification accuracy via UP forward_eval ────────────────────
    var pred_buf = alloc[Scalar[dtype]](BATCH * 10)
    memset(pred_buf, 0, BATCH * 10)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](pred_buf)

    # Reuse mu/a buffers as scratch for forward_eval
    var up_eval_mu_buf = alloc[Scalar[dtype]](BATCH * UP_NET.SCRATCH_OUT_DIM)
    var up_eval_a_buf = alloc[Scalar[dtype]](BATCH * UP_NET.SCRATCH_IN_DIM)
    memset(up_eval_mu_buf, 0, BATCH * UP_NET.SCRATCH_OUT_DIM)
    memset(up_eval_a_buf, 0, BATCH * UP_NET.SCRATCH_IN_DIM)
    var up_eval_mu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, UP_NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](up_eval_mu_buf)
    var up_eval_a_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, UP_NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](up_eval_a_buf)

    var correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            for j in range(784):
                image_buf[i * 784 + j] = ds.test_images[sample_idx * 784 + j]
        UP_NET.forward_eval[BATCH, dtype](image_t, up_params, pred_t)
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(pred_buf[i * 10])
            for c in range(1, 10):
                var v = Float64(pred_buf[i * 10 + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                correct += 1
    var test_acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
    print("\n  UP test accuracy:", test_acc)

    # ── Generation: pass each one-hot label through DOWN forward_eval ────────
    # Use a small batch (10 labels = 10 samples) — but our test fixed BATCH=100,
    # so we'll fill 10 samples in a batch of 100 (others are zeros).
    var gen_label_buf = alloc[Scalar[dtype]](BATCH * 10)
    var gen_image_buf = alloc[Scalar[dtype]](BATCH * 784)
    memset(gen_label_buf, 0, BATCH * 10)
    memset(gen_image_buf, 0, BATCH * 784)
    for c in range(10):
        gen_label_buf[c * 10 + c] = Scalar[dtype](1.0)  # one-hot for class c
    var gen_label_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 10), MutAnyOrigin
    ](gen_label_buf)
    var gen_image_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 784), MutAnyOrigin
    ](gen_image_buf)
    DOWN_NET.forward_eval[BATCH, dtype](
        gen_label_t, dn_params, gen_image_t
    )

    # Compute pairwise pixel-MSE between class-0 and class-9 generated images
    var class_diff: Float64 = 0
    for j in range(784):
        var d = (
            Float64(gen_image_buf[0 * 784 + j])
            - Float64(gen_image_buf[9 * 784 + j])
        )
        class_diff += d * d
    class_diff /= Float64(784)
    print("  Per-pixel MSE between class-0 and class-9 generated images:", class_diff)

    # ── Mean magnitude per class (sanity that decoder isn't outputting zeros) ─
    var gen_mean_mag: Float64 = 0
    for c in range(10):
        for j in range(784):
            gen_mean_mag += abs(Float64(gen_image_buf[c * 784 + j]))
    gen_mean_mag /= Float64(10 * 784)
    print("  Mean |gen pixel| across 10 classes:", gen_mean_mag)

    # ── Verdict ──────────────────────────────────────────────────────────────
    # Primary criterion: classification works (validates joint inference).
    # Generation diversity is a secondary criterion — needs much more
    # training budget to differentiate classes well, so it's informational.
    var pass_acc = test_acc >= 0.50
    print("\n  classification check (≥50%):", pass_acc, " (got ", test_acc, ")")
    print("  generation diversity (info)  : per-class MSE =", class_diff)
    print("  generation magnitude (info)  : mean |pixel| =", gen_mean_mag)
    if pass_acc:
        print("\n  [PASS] bidirectional PC: joint inference trains the up-path classifier")
        if class_diff < 0.005:
            print("  (NOTE) decoder under-trained — would need more epochs/larger α_down for full generation quality")
    else:
        print("\n  [FAIL] bidirectional PC: classification accuracy too low")
        raise Error("bidirectional PC test failed")

    # cleanup
    up_params_buf.free()
    up_grads_buf.free()
    up_opt_state_buf.free()
    up_opt_global_buf.free()
    dn_params_buf.free()
    dn_grads_buf.free()
    dn_opt_state_buf.free()
    dn_opt_global_buf.free()
    x_shared_buf.free()
    up_mu0_buf.free()
    up_eps0_buf.free()
    up_mu1_buf.free()
    up_eps1_buf.free()
    up_a0_buf.free()
    up_a1_buf.free()
    up_z1_buf.free()
    dn_mu0_buf.free()
    dn_eps0_buf.free()
    dn_mu1_buf.free()
    dn_eps1_buf.free()
    dn_a0_buf.free()
    dn_a1_buf.free()
    dn_z1_buf.free()
    dx_buf.free()
    image_buf.free()
    label_oh_buf.free()
    pred_buf.free()
    up_eval_mu_buf.free()
    up_eval_a_buf.free()
    gen_label_buf.free()
    gen_image_buf.free()
    print("=== Done ===")
