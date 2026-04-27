"""Three-piece OFENet training proof.

Validates the Phase-3b integration pattern: OFE is split into three
independent Models (StateBranch, ActionBranch, Linear predictor), each
with its own NetworkState. Aux MSE loss on predicted next-state is
computed by manually chaining forward / backward across the three
pieces; a single shared aux-optimizer step then updates all three.

This proves we don't need a single `OFENetPredictor` Sequential to
train the extractor — we can train it as three chained pieces. In the
REDQ-OFE agent this chain runs once per env step, while actor / critics
consume intermediate features (phi_s from StateBranch,
phi_sa from ActionBranch) as stop-gradient inputs.
"""

from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import Linear
from mojo_rl.nn.composites_ofenet import StateBranch6, ActionBranch6


def test_three_piece_trains() raises:
    """Chain forward/backward through 3 pieces, confirm loss decreases."""
    print("=" * 60)
    print("TEST: 3-piece OFE (StateBranch + ActionBranch + Linear) trains")
    print("=" * 60)

    comptime SD = 4
    comptime AD = 2
    comptime PU = 3
    comptime PhiS = SD + 6 * PU       # 22
    comptime PhiSA_in = PhiS + AD     # 24  (concat of phi_s + a)
    comptime PhiSA = PhiSA_in + 6 * PU  # 42  (out of ActionBranch)

    comptime SB = StateBranch6[SD, PU]
    comptime AB = ActionBranch6[PhiS, AD, PU]
    comptime PR = Linear[PhiSA, SD]

    comptime BS = 8
    comptime STEPS = 200

    var sb = NetworkState[SB, Adam[LR=0.001]]()
    var ab = NetworkState[AB, Adam[LR=0.001]]()
    var pr = NetworkState[PR, Adam[LR=0.001]]()
    sb.initialize[Xavier[]]()
    ab.initialize[Xavier[]]()
    pr.initialize[Xavier[]]()

    # Allocate forward buffers
    var s_ptr = alloc[Scalar[dtype]](BS * SD)
    var a_ptr = alloc[Scalar[dtype]](BS * AD)
    var target_ptr = alloc[Scalar[dtype]](BS * SD)

    var phi_s_ptr = alloc[Scalar[dtype]](BS * PhiS)
    var phi_sa_in_ptr = alloc[Scalar[dtype]](BS * PhiSA_in)  # concat(phi_s, a)
    var phi_sa_ptr = alloc[Scalar[dtype]](BS * PhiSA)
    var pred_ptr = alloc[Scalar[dtype]](BS * SD)

    var sb_cache = alloc[Scalar[dtype]](BS * SB.CACHE_SIZE)
    var ab_cache = alloc[Scalar[dtype]](BS * AB.CACHE_SIZE)
    var pr_cache = alloc[Scalar[dtype]](BS * PR.CACHE_SIZE if PR.CACHE_SIZE > 0 else 1)

    # Backward buffers
    var grad_pred_ptr = alloc[Scalar[dtype]](BS * SD)
    var grad_phi_sa_ptr = alloc[Scalar[dtype]](BS * PhiSA)
    var grad_phi_sa_in_ptr = alloc[Scalar[dtype]](BS * PhiSA_in)
    var grad_phi_s_ptr = alloc[Scalar[dtype]](BS * PhiS)
    var grad_s_ptr = alloc[Scalar[dtype]](BS * SD)

    # Toy data: next_state = 0.5 * s + 0.3 * a[0]
    for b in range(BS):
        for i in range(SD):
            s_ptr[b * SD + i] = Scalar[dtype](
                Float64(b % 5) * 0.2 + Float64(i) * 0.1
            )
        for i in range(AD):
            a_ptr[b * AD + i] = Scalar[dtype](
                Float64(b % 3) * 0.3 - Float64(i) * 0.2
            )
        var a0 = Float64(a_ptr[b * AD])
        for i in range(SD):
            target_ptr[b * SD + i] = Scalar[dtype](
                0.5 * Float64(s_ptr[b * SD + i]) + 0.3 * a0
            )

    var s_t = LayoutTensor[dtype, Layout.row_major(BS, SD), MutAnyOrigin](s_ptr)
    var phi_s_t = LayoutTensor[dtype, Layout.row_major(BS, PhiS), MutAnyOrigin](phi_s_ptr)
    var phi_sa_in_t = LayoutTensor[dtype, Layout.row_major(BS, PhiSA_in), MutAnyOrigin](phi_sa_in_ptr)
    var phi_sa_t = LayoutTensor[dtype, Layout.row_major(BS, PhiSA), MutAnyOrigin](phi_sa_ptr)
    var pred_t = LayoutTensor[dtype, Layout.row_major(BS, SD), MutAnyOrigin](pred_ptr)
    var sb_cache_t = LayoutTensor[dtype, Layout.row_major(BS, SB.CACHE_SIZE), MutAnyOrigin](sb_cache)
    var ab_cache_t = LayoutTensor[dtype, Layout.row_major(BS, AB.CACHE_SIZE), MutAnyOrigin](ab_cache)
    var pr_cache_t = LayoutTensor[dtype, Layout.row_major(BS, PR.CACHE_SIZE), MutAnyOrigin](pr_cache)

    var grad_pred_t = LayoutTensor[dtype, Layout.row_major(BS, SD), MutAnyOrigin](grad_pred_ptr)
    var grad_phi_sa_t = LayoutTensor[dtype, Layout.row_major(BS, PhiSA), MutAnyOrigin](grad_phi_sa_ptr)
    var grad_phi_sa_in_t = LayoutTensor[dtype, Layout.row_major(BS, PhiSA_in), MutAnyOrigin](grad_phi_sa_in_ptr)
    var grad_phi_s_t = LayoutTensor[dtype, Layout.row_major(BS, PhiS), MutAnyOrigin](grad_phi_s_ptr)
    var grad_s_t = LayoutTensor[dtype, Layout.row_major(BS, SD), MutAnyOrigin](grad_s_ptr)

    var init_loss: Float64 = 0.0
    var final_loss: Float64 = 0.0

    for step in range(STEPS):
        # ─── Forward chain ───
        memset(phi_s_ptr, 0, BS * PhiS)
        memset(sb_cache, 0, BS * SB.CACHE_SIZE)
        SB.forward[BS](s_t, phi_s_t, sb.params_view(), sb.model_state_view(), sb_cache_t)

        # Manual concat: phi_sa_in = [phi_s | a]
        for b in range(BS):
            for i in range(PhiS):
                phi_sa_in_ptr[b * PhiSA_in + i] = phi_s_ptr[b * PhiS + i]
            for i in range(AD):
                phi_sa_in_ptr[b * PhiSA_in + PhiS + i] = a_ptr[b * AD + i]

        memset(phi_sa_ptr, 0, BS * PhiSA)
        memset(ab_cache, 0, BS * AB.CACHE_SIZE)
        AB.forward[BS](phi_sa_in_t, phi_sa_t, ab.params_view(), ab.model_state_view(), ab_cache_t)

        memset(pred_ptr, 0, BS * SD)
        memset(pr_cache, 0, BS * PR.CACHE_SIZE)
        PR.forward[BS](phi_sa_t, pred_t, pr.params_view(), pr.model_state_view(), pr_cache_t)

        # ─── MSE loss + gradient ───
        var loss: Float64 = 0.0
        var inv = 1.0 / Float64(BS * SD)
        for i in range(BS * SD):
            var diff = Float64(pred_ptr[i]) - Float64(target_ptr[i])
            loss += diff * diff
            grad_pred_ptr[i] = Scalar[dtype](2.0 * diff * inv)
        loss *= inv
        if step == 0:
            init_loss = loss

        # ─── Backward chain ───
        sb.zero_grads()
        ab.zero_grads()
        pr.zero_grads()

        # 1. Linear predictor backward → grad_phi_sa
        memset(grad_phi_sa_ptr, 0, BS * PhiSA)
        var pr_grads = pr.grads_view()
        PR.backward[BS](grad_pred_t, grad_phi_sa_t, pr.params_view(), pr.model_state_view(), pr_cache_t, pr_grads)

        # 2. ActionBranch backward → grad_phi_sa_in (split into grad_phi_s | grad_a)
        memset(grad_phi_sa_in_ptr, 0, BS * PhiSA_in)
        var ab_grads = ab.grads_view()
        AB.backward[BS](grad_phi_sa_t, grad_phi_sa_in_t, ab.params_view(), ab.model_state_view(), ab_cache_t, ab_grads)

        # 3. Split: first PhiS dims are grad_phi_s, rest are grad_a (discarded)
        for b in range(BS):
            for i in range(PhiS):
                grad_phi_s_ptr[b * PhiS + i] = grad_phi_sa_in_ptr[b * PhiSA_in + i]

        # 4. StateBranch backward → grad_s (unused)
        memset(grad_s_ptr, 0, BS * SD)
        var sb_grads = sb.grads_view()
        SB.backward[BS](grad_phi_s_t, grad_s_t, sb.params_view(), sb.model_state_view(), sb_cache_t, sb_grads)

        # ─── Optimizer step (aux) ───
        sb.optimizer_step()
        ab.optimizer_step()
        pr.optimizer_step()

        if step == STEPS - 1:
            final_loss = loss

        if step == 0 or step == STEPS // 2 or step == STEPS - 1:
            print("  step", step, "  loss=", loss)

    print("Init loss:", init_loss, " Final loss:", final_loss)
    if final_loss < init_loss * 0.2:
        print("PASS: 3-piece chain trains (>=5x loss reduction)")
    else:
        print("FAIL: insufficient loss reduction")

    s_ptr.free(); a_ptr.free(); target_ptr.free()
    phi_s_ptr.free(); phi_sa_in_ptr.free(); phi_sa_ptr.free(); pred_ptr.free()
    sb_cache.free(); ab_cache.free(); pr_cache.free()
    grad_pred_ptr.free(); grad_phi_sa_ptr.free(); grad_phi_sa_in_ptr.free()
    grad_phi_s_ptr.free(); grad_s_ptr.free()


def main() raises:
    test_three_piece_trains()
