"""`rsi_inject_kernel` on device, and the lie-down rule agreeing across the split.

Two reasons this file exists.

1. NOTHING ELSE INSTANTIATES THE KERNEL where it can be built. Its only caller
   is `examples/g1/bfm_zero_train_gpu.mojo`, which does not compile for Metal
   (the G1 batched env, §12.2), and a generic Mojo kernel is only type-checked
   when instantiated — a green package build proves nothing about it. So an
   edit to the kernel body could reach the box as a 15-minute compile error.

2. THE LIE-DOWN RULE IS READ FROM TWO SIDES. The kernel applies the transform
   on device; the driver's reset diagnostic reports what FRACTION of lanes got
   it, counted on the host from the same uniforms. Both go through
   `lie_down_selected` so the rule is written once — this gates that they
   actually agree, because a diagnostic that drifts from the simulation reports
   a number nobody is running.

Checks, on a lane sweep of u2 across [0, 1) at `lie_prob = 0.3`:
  * the host predicate and the lanes the kernel ACTUALLY transformed (z
    overwritten with `G1_LIE_DOWN_Z`) are the same count, and the same lanes;
  * every lane got a row inside its episode, and qpos/qvel were written;
  * NON-VACUITY: the sweep must straddle the threshold — some lanes lie down
    and some do not — else "the counts agree" observes nothing.

Run: pixi run -e apple mojo run -I . tests/robots/test_g1_rsi_inject_kernel_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.envs.robots.unitree_g1_rsi import (
    rsi_inject_kernel, lie_down_selected, G1_RSI_NQ, G1_RSI_NV, G1_LIE_DOWN_Z,
)

comptime LANES = 64
comptime NQ = G1_RSI_NQ
comptime NV = G1_RSI_NV
comptime W = NQ + NV
comptime NROW = 8
comptime NEP = 2
comptime P = Scalar[DT](0.3)


def main() raises:
    var c = DeviceContext()
    print("G1 rsi_inject_kernel + lie-down rule across the host/device split")

    var rows = c.enqueue_create_buffer[DT](NROW * W)
    var h_rows = c.enqueue_create_host_buffer[DT](NROW * W)
    for r in range(NROW):
        for i in range(W):
            # distinct per (row, field) so a wrong row is visible in qpos
            h_rows[r * W + i] = Scalar[DT](Float64(r) + Float64(i) * 0.01)
    c.enqueue_copy(rows, h_rows)

    var epo = c.enqueue_create_buffer[DT](NEP)
    var epl = c.enqueue_create_buffer[DT](NEP)
    var h_epo = c.enqueue_create_host_buffer[DT](NEP)
    var h_epl = c.enqueue_create_host_buffer[DT](NEP)
    h_epo[0] = 0; h_epo[1] = 4
    h_epl[0] = 4; h_epl[1] = 4
    c.enqueue_copy(epo, h_epo); c.enqueue_copy(epl, h_epl)

    var u = c.enqueue_create_buffer[DT](LANES * 3)
    var h_u = c.enqueue_create_host_buffer[DT](LANES * 3)
    for l in range(LANES):
        h_u[l * 3 + 0] = Scalar[DT](Float64(l % NEP) / Float64(NEP))
        h_u[l * 3 + 1] = Scalar[DT](0.5)
        h_u[l * 3 + 2] = Scalar[DT](Float64(l) / Float64(LANES))  # sweeps [0,1)
    c.enqueue_copy(u, h_u)

    var qpos = c.enqueue_create_buffer[DT](LANES * NQ)
    var qvel = c.enqueue_create_buffer[DT](LANES * NV)
    var rout = c.enqueue_create_buffer[DT](LANES)
    c.enqueue_function[rsi_inject_kernel[LANES, NQ, NV]](
        mptr(rows.unsafe_ptr()), mptr(epo.unsafe_ptr()), mptr(epl.unsafe_ptr()),
        Int32(NEP), mptr(u.unsafe_ptr()), P, Scalar[DT](1.0),
        mptr(qpos.unsafe_ptr()), mptr(qvel.unsafe_ptr()), mptr(rout.unsafe_ptr()),
        grid_dim=(LANES + TPB - 1) // TPB, block_dim=TPB,
    )
    var h_qpos = c.enqueue_create_host_buffer[DT](LANES * NQ)
    var h_qvel = c.enqueue_create_host_buffer[DT](LANES * NV)
    var h_rout = c.enqueue_create_host_buffer[DT](LANES)
    c.enqueue_copy(h_qpos, qpos)
    c.enqueue_copy(h_qvel, qvel)
    c.enqueue_copy(h_rout, rout)
    c.synchronize()

    var host_n = 0
    var kernel_n = 0
    var disagree = 0
    var bad_row = 0
    for l in range(LANES):
        var want = lie_down_selected(h_u[l * 3 + 2], P)
        # what the kernel ACTUALLY did: z replaced by the lie-down height
        var got = abs(Float64(h_qpos[l * NQ + 2]) - Float64(G1_LIE_DOWN_Z)) < 1e-9
        if want:
            host_n += 1
        if got:
            kernel_n += 1
        if want != got:
            disagree += 1
            print("    lane", l, "u2 =", h_u[l * 3 + 2], " host says", want, " kernel did", got)
        var r = Int(Float64(h_rout[l]))
        var e = l % NEP
        if r < Int(Float64(h_epo[e])) or r >= Int(Float64(h_epo[e])) + Int(Float64(h_epl[e])):
            bad_row += 1
        # qvel must have been written from the row (non-lie fields are untouched
        # by the transform, so this also proves the copy ran)
        if abs(Float64(h_qvel[l * NV]) - (Float64(r) + Float64(NQ) * 0.01)) > 1e-5:
            bad_row += 1

    print("  lanes:", LANES, " host predicate:", host_n, " kernel applied:", kernel_n)
    print("  fraction:", Float64(kernel_n) / Float64(LANES), " lie_prob:", Float64(P))

    assert_true(
        host_n > 0 and host_n < LANES,
        "vacuous: the u2 sweep did not straddle lie_prob, so every lane fell on"
        " one side and the agreement check observes nothing",
    )
    assert_true(disagree == 0, "the host lie-down predicate disagrees with the kernel")
    assert_true(bad_row == 0, "a lane got a row outside its episode, or qvel was not written")
    print("G1_RSI_INJECT_KERNEL OK")
