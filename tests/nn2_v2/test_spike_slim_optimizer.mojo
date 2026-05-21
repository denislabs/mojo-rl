"""Train the spike network on XOR for a few epochs using the slim Adam.

Validates the slim Optimizer trait end-to-end:
  - `Adam.make[M](model)` with no hyperparams.
  - User overrides `opt.lr = 0.05` after make — public mut field.
  - `opt.zero_grad(net)` / forward / backward / `opt.step(net)` loop.
  - Loss decreases (XOR is small enough to overfit in <500 steps).

Network: Linear[2,4] → Tanh → Linear[4,1]
Loss:    0.5 * mean((output - target)^2)
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2_v2.spike_unified_buffers import (
    DT, Linear, Tanh, Sequential,
)
from mojo_rl.nn2_v2.spike_slim_optimizer import Adam


comptime BATCH = 4   # all four XOR rows
comptime D0 = 2
comptime D1 = 4
comptime D2 = 1


def main() raises:
    # Build network.
    var l0 = Linear[D0, D1].make_xavier(seed_offset=0)
    var th = Tanh[D1]()
    var l1 = Linear[D1, D2].make_xavier(seed_offset=1)
    var net = Sequential[Linear[D0, D1], Tanh[D1], Linear[D1, D2]](
        l0^, th^, l1^,
    )

    # Build Adam via the slim trait factory — no hyperparams.
    var opt = Adam.make(net)
    # Override the public mut hyperparam field. This is the slim-trait
    # idiom — `opt.lr = ...` instead of `Adam.make[M](net, lr=..., ...)`.
    opt.lr = Scalar[DT](0.05)

    # XOR data.
    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D0
    )
    var tg_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D2
    )
    in_p[0*D0+0] = 0.0;  in_p[0*D0+1] = 0.0;  tg_p[0] = 0.0
    in_p[1*D0+0] = 0.0;  in_p[1*D0+1] = 1.0;  tg_p[1] = 1.0
    in_p[2*D0+0] = 1.0;  in_p[2*D0+1] = 0.0;  tg_p[2] = 1.0
    in_p[3*D0+0] = 1.0;  in_p[3*D0+1] = 1.0;  tg_p[3] = 0.0

    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D2
    )
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D2
    )
    var gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * D0
    )

    var L0: Scalar[DT] = Scalar[DT](0.0)
    var L_final: Scalar[DT] = Scalar[DT](0.0)

    for step in range(500):
        opt.zero_grad(net)

        # Forward.
        var in_t = TileTensor(in_p, row_major[BATCH, D0]())
        var out_t = TileTensor(out_p, row_major[BATCH, D2]())
        net.forward[BATCH](in_t, out_t)

        # Loss = 0.5 * mean((output - target)^2)
        var L: Scalar[DT] = Scalar[DT](0.0)
        for b in range(BATCH):
            var d = out_p[b] - tg_p[b]
            L += Scalar[DT](0.5) * d * d
        L /= Scalar[DT](BATCH)
        if step == 0:
            L0 = L
        L_final = L

        # Backward: dL/d_out[b] = (out[b] - tg[b]) / BATCH
        for b in range(BATCH):
            go_p[b] = (out_p[b] - tg_p[b]) / Scalar[DT](BATCH)
        var go_t = TileTensor(go_p, row_major[BATCH, D2]())
        var gi_t = TileTensor(gi_p, row_major[BATCH, D0]())
        net.backward[BATCH](go_t, gi_t)

        opt.step(net)

        if step % 100 == 0 or step == 499:
            print("step", step, "loss", L)

    print("L_initial =", L0)
    print("L_final   =", L_final)

    # XOR with a 4-unit hidden Tanh layer should solve to L < 0.02 in 500 steps.
    if L_final < Scalar[DT](0.02) and L_final < L0 * Scalar[DT](0.1):
        print("PASS — slim Adam trained XOR (final loss 10× smaller, <0.02)")
    else:
        print("FAIL — slim Adam did not train XOR adequately")
        raise Error("slim Adam training did not converge")

    in_p.free()
    tg_p.free()
    out_p.free()
    go_p.free()
    gi_p.free()
