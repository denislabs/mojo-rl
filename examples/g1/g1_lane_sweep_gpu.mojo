"""Unitree G1 batched-env throughput at one lane count — the G0 lane sweep.

    pixi run -e nvidia mojo run -I . examples/g1/g1_lane_sweep_gpu.mojo
    # then edit `LANES` below (64, 128, 256, 512, 1024) and run again; or
    for n in 64 128 256 512 1024; do
        sed "s/^comptime LANES = .*/comptime LANES = $n/" examples/g1/g1_lane_sweep_gpu.mojo \
            > /tmp/g1_sweep_$n.mojo && pixi run -e nvidia mojo run -I . /tmp/g1_sweep_$n.mojo
    done

Prints control steps per second (all lanes), physics substeps per second,
and microseconds per control step per lane, for LANES lanes of the G1 under
BFM-Zero's torque PD at 50 Hz over 200 Hz physics with a driven action.

WHY ONE LANE COUNT PER RUN. `N_ENVS` is a compile-time parameter of
`Phyics3dBatchedEnv`, and every distinct value is a separate instantiation
of the blocked Newton kernel — ~15 minutes of compile each on the 5090
(`feedback_a_rented_gpu_box_prices_every_gate_in_compile_minutes`). A
`comptime for` over five counts would be a 75-minute build; one constant
per run makes each count its own short job and lets a failing count (out
of per-thread stack, out of memory) fail alone. (This nightly's `std.sys`
has no `env_get_int`, so the count is an edited constant rather than a
`-D` define.)

WHAT THE NUMBER IS FOR. BFM-Zero's own pipeline runs at ~764 control frames
per second TOTAL on one H200 with 1024 envs (UFO, Table 1) — the learner is
the wall-clock, not the simulator — so the bar for the simulator is only
~3 000 physics substeps per second. This sweep tells us which lane count
clears that with room, and where the per-lane cost stops falling.

MEASURED (RTX 5090, 2026-09-09, 200 timed control steps after 20 warmup,
driven action, the reference's 4 substeps of 1/200 s):

    lanes   control steps/s   physics substeps/s   us / control step / lane
      64          4 386             17 545                 228
     128          8 666             34 664                 115
     256         16 015             64 061                  62
     512         25 480            101 920                  39
    1024         36 494            145 977                  27

At 1024 lanes that is 48x BFM-Zero's whole-pipeline rate and 49x the bar;
sixteen gradient steps per batched control step at this rate would need
570 learner steps per second, so the learner sets G3's pace, as it did
theirs. The per-lane cost is still falling at 1024 (27 vs 39 at 512):
2048 is worth one run when a config wants it. Each lane count compiled and
ran in ~114 s on the box — the kernel instantiation is two minutes, not
the fifteen the blocked-kernel gates cost.

⚠ TWO PHASES ARE TIMED SEPARATELY. The first `WARMUP` steps include kernel
launch and any lazy allocation; only the `MEASURE` steps after them are
reported. Do not build anything else on the box while this runs — a
concurrent `mojo build` inflated a Menagerie row 9x once
(`docs/menagerie_fidelity_harnesses/README.md`, `bigsweep.py`).
"""

from max.gpu.host import DeviceContext, HostBuffer
from std.time import perf_counter_ns
from std.math import sin

from mojo_rl.nn.constants import DT
from mojo_rl.envs.robots import UnitreeG1Batched
from mojo_rl.envs.robots.unitree_g1_xml import UnitreeG1Model
from mojo_rl.envs.robots.unitree_g1_pd import G1_CONTROL_DECIMATION


comptime LANES = 256
comptime WARMUP = 20
comptime MEASURE = 200


def _drive[ACT_DIM: Int](
    ctx: DeviceContext,
    mut env: UnitreeG1Batched[LANES],
    mut h_act: HostBuffer[DT],
    t: Int,
) raises:
    """The parity gate's drive — every joint at its own phase, 0.3 amplitude —
    on every lane, then one batched control step."""
    for j in range(ACT_DIM):
        var u = 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)
        for e in range(LANES):
            h_act[e * ACT_DIM + j] = Scalar[DT](u)
    ctx.enqueue_copy(env._action, h_act)
    env.step_batch[LANES](Optional(ctx), 0)


def main() raises:
    comptime ACT_DIM = UnitreeG1Model.ACTION_DIM
    with DeviceContext() as ctx:
        var env = UnitreeG1Batched[LANES](ctx)
        env.reset_batch[LANES](Optional(ctx), UInt64(1))
        var h_act = ctx.enqueue_create_host_buffer[DT](LANES * ACT_DIM)
        ctx.synchronize()

        for t in range(WARMUP):
            _drive[ACT_DIM](ctx, env, h_act, t)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for t in range(MEASURE):
            _drive[ACT_DIM](ctx, env, h_act, WARMUP + t)
        ctx.synchronize()
        var dt_s = Float64(perf_counter_ns() - t0) * 1e-9

        var ctrl_per_s = Float64(MEASURE * LANES) / dt_s
        var sub_per_s = ctrl_per_s * Float64(G1_CONTROL_DECIMATION)
        var us_per_ctrl_lane = dt_s * 1e6 / Float64(MEASURE * LANES)
        print("unitree_g1 lane sweep")
        print("  lanes               =", LANES)
        print("  control steps       =", MEASURE, "(after", WARMUP, "warmup)")
        print("  wall                =", dt_s, "s")
        print("  control steps / s   =", ctrl_per_s, "(all lanes)")
        print("  physics substeps / s=", sub_per_s)
        print("  us / control step / lane =", us_per_ctrl_lane)
        print("  BFM-Zero's own rate for reference: ~764 control steps/s at 1024 envs on one H200")
