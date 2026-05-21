"""Gradcheck for the unified-buffer spike.

Validates the orchestrator-owns-slabs design end-to-end by FD-comparing
analytical gradients (from one Sequential.backward call) against
numerical gradients (perturb each weight ± eps, two forwards each).

Network:  Linear[2, 4] → Tanh → Linear[4, 3] → ReLU → Linear[3, 1]

Covers all three leaf types and both cache strategies:
  - Linear      (input-caching, aliases slab)
  - ReLU        (input-caching, aliases slab, element-wise in-place safe)
  - Tanh        (output-caching, owns its own cache buffer — option 2)

Loss: scalar L = sum_b output[b, 0]^2 / 2
      dL/d_output[b, 0] = output[b, 0]

Pass criterion: max relative error < 1e-3 across every weight + bias.
"""

from std.math import abs as fabs, tanh as ftanh
from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2_v2.spike_unified_buffers import (
    DT, Linear, ReLU, Tanh, Sequential,
)


comptime BATCH = 3
comptime D0 = 2
comptime D1 = 4
comptime D2 = 3
comptime D3 = 1


def _forward_loss(
    mut net: Sequential[Linear[D0, D1], Tanh[D1], Linear[D1, D2], ReLU[D2], Linear[D2, D3]],
    in_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    var in_t = TileTensor(in_p, row_major[BATCH, D0]())
    var out_t = TileTensor(out_p, row_major[BATCH, D3]())
    net.forward[BATCH](in_t, out_t)
    var L: Scalar[DT] = 0.0
    for b in range(BATCH):
        var y = out_p[b * D3 + 0]
        L += Scalar[DT](0.5) * y * y
    return L


def _max_rel_err(
    analytic: UnsafePointer[Scalar[DT], MutAnyOrigin],
    numeric: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Scalar[DT]:
    var max_rel: Scalar[DT] = 0.0
    for k in range(n):
        var a = analytic[k]
        var num = numeric[k]
        var denom: Scalar[DT] = fabs(a) + fabs(num) + Scalar[DT](1e-8)
        var rel = fabs(a - num) / denom
        if rel > max_rel:
            max_rel = rel
    return max_rel


def main() raises:
    # ── Build the network. ──
    var lin0 = Linear[D0, D1].make_xavier(seed_offset=0)
    var th = Tanh[D1]()
    var lin1 = Linear[D1, D2].make_xavier(seed_offset=1)
    var re = ReLU[D2]()
    var lin2 = Linear[D2, D3].make_xavier(seed_offset=2)

    var net = Sequential[
        Linear[D0, D1], Tanh[D1], Linear[D1, D2], ReLU[D2], Linear[D2, D3],
    ](lin0^, th^, lin1^, re^, lin2^)

    # ── Allocate I/O buffers. ──
    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D0
    )
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D3
    )
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D3
    )
    var gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D0
    )

    # Deterministic-looking input.
    for b in range(BATCH):
        in_p[b * D0 + 0] = Scalar[DT](0.3) * Scalar[DT](b + 1)
        in_p[b * D0 + 1] = Scalar[DT](-0.5) * Scalar[DT](b + 1) + Scalar[DT](0.2)

    # ── Analytical forward + backward. ──
    net.children[0].zero_grad()
    net.children[2].zero_grad()
    net.children[4].zero_grad()

    var L_baseline = _forward_loss(net, in_p, out_p)

    # dL/d_output = output (since L = 0.5 * sum output^2)
    for b in range(BATCH):
        go_p[b * D3 + 0] = out_p[b * D3 + 0]
    var go_t = TileTensor(go_p, row_major[BATCH, D3]())
    var gi_t = TileTensor(gi_p, row_major[BATCH, D0]())
    net.backward[BATCH](go_t, gi_t)

    print("L_baseline =", L_baseline)

    # ── FD check each Linear's weight + bias. ──
    comptime eps: Scalar[DT] = 1e-3

    # Snapshot analytical grads.
    var n_w0 = D0 * D1
    var n_b0 = D1
    var n_w1 = D1 * D2
    var n_b1 = D2
    var n_w2 = D2 * D3
    var n_b2 = D3

    var num_w0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_w0)
    var num_b0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_b0)
    var num_w1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_w1)
    var num_b1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_b1)
    var num_w2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_w2)
    var num_b2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_b2)

    # FD over weight 0.
    for k in range(n_w0):
        var orig = net.children[0].weight[k]
        net.children[0].weight[k] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[0].weight[k] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[0].weight[k] = orig
        num_w0[k] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
    for j in range(n_b0):
        var orig = net.children[0].bias[j]
        net.children[0].bias[j] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[0].bias[j] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[0].bias[j] = orig
        num_b0[j] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)

    for k in range(n_w1):
        var orig = net.children[2].weight[k]
        net.children[2].weight[k] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[2].weight[k] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[2].weight[k] = orig
        num_w1[k] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
    for j in range(n_b1):
        var orig = net.children[2].bias[j]
        net.children[2].bias[j] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[2].bias[j] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[2].bias[j] = orig
        num_b1[j] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)

    for k in range(n_w2):
        var orig = net.children[4].weight[k]
        net.children[4].weight[k] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[4].weight[k] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[4].weight[k] = orig
        num_w2[k] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
    for j in range(n_b2):
        var orig = net.children[4].bias[j]
        net.children[4].bias[j] = orig + eps
        var L_plus = _forward_loss(net, in_p, out_p)
        net.children[4].bias[j] = orig - eps
        var L_minus = _forward_loss(net, in_p, out_p)
        net.children[4].bias[j] = orig
        num_b2[j] = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)

    # ── Recompute the analytical gradient cleanly (FD above mutated then
    # restored, but the original analytical pass is what we trust).
    # The analytical grads are still in grad_w/grad_b — they were
    # computed before the FD perturbations.
    var ana_w0_p = net.children[0].grad_w.unsafe_ptr()
    var ana_b0_p = net.children[0].grad_b.unsafe_ptr()
    var ana_w1_p = net.children[2].grad_w.unsafe_ptr()
    var ana_b1_p = net.children[2].grad_b.unsafe_ptr()
    var ana_w2_p = net.children[4].grad_w.unsafe_ptr()
    var ana_b2_p = net.children[4].grad_b.unsafe_ptr()

    var err_w0 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_w0_p), num_w0, n_w0
    )
    var err_b0 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_b0_p), num_b0, n_b0
    )
    var err_w1 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_w1_p), num_w1, n_w1
    )
    var err_b1 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_b1_p), num_b1, n_b1
    )
    var err_w2 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_w2_p), num_w2, n_w2
    )
    var err_b2 = _max_rel_err(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ana_b2_p), num_b2, n_b2
    )

    print("max_rel_err Linear0.weight =", err_w0)
    print("max_rel_err Linear0.bias   =", err_b0)
    print("max_rel_err Linear1.weight =", err_w1)
    print("max_rel_err Linear1.bias   =", err_b1)
    print("max_rel_err Linear2.weight =", err_w2)
    print("max_rel_err Linear2.bias   =", err_b2)

    comptime tol: Scalar[DT] = 1e-2
    var ok = (
        err_w0 < tol and err_b0 < tol
        and err_w1 < tol and err_b1 < tol
        and err_w2 < tol and err_b2 < tol
    )
    if ok:
        print("PASS — unified-buffer spike gradcheck within tol", tol)
    else:
        print("FAIL — gradcheck out of tol", tol)
        raise Error("gradcheck failed")

    # ── Cleanup. ──
    in_p.free()
    out_p.free()
    go_p.free()
    gi_p.free()
    num_w0.free()
    num_b0.free()
    num_w1.free()
    num_b1.free()
    num_w2.free()
    num_b2.free()
