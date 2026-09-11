"""The G1 collision kernel at the RL operating point — PERFORMANCE.md §13.51.

    pixi run -e nvidia mojo build -I . benchmarks/physics3d_gpu/bench_g1_collision.mojo -o /tmp/g1coll
    /tmp/g1coll [rollouts=128] [ctrl_steps=40] [first_snap=24] [rounds=2] [cpu_check=1] [diag_lanes=0]

    pixi run -e apple  mojo build -I . benchmarks/physics3d_gpu/bench_g1_collision.mojo -o /tmp/g1coll

WHAT IT MEASURES. §13.51 found `_detect_contacts_sap_fields_kernel` to be
47% of a BFM-Zero G1 training run on the RTX 5090: 1024 lanes of a 29-DoF
humanoid whose untrained policies fall over, so that by control step ~25
of every episode the lanes are sprawled on the floor with limbs folded
into each other, and the kernel's per-launch time climbs from ~58 ms to a
~131 ms plateau. The driver is 25 s of setup before the first step and the
learner is half the GPU, so the kernel cannot be iterated on through it.
This file reproduces the WORKLOAD without the learner or the solver:

  1. `rollouts` CPU rollouts of `UnitreeG1[float32]` (the single-env
     facade, any platform) under the driven action of the parity gate with
     a random per-joint amplitude and phase, saturating the PD targets so
     the body falls. `qpos` is snapshotted at every control step from
     `first_snap` on — the plateau regime of §13.51.
  2. A batched fields `Data` of `BATCH` lanes (edit the constant, as
     `g1_lane_sweep_gpu.mojo` does: every count is its own instantiation),
     lane `e` carrying rollout `e % rollouts`. For every snapshot in order:
     upload `qpos`, run FK on the device, then time ONE
     `detect_contacts_sap["gpu"]` launch between two synchronizes. The
     snapshots are consecutive control steps of the same lanes, so the
     hill climb's cross-step warm slots (§13.49) see what training sees;
     `rounds` repeats the whole snapshot sequence, and the FIRST round's
     first snapshot is the cold one.
  3. `cpu_check=1`: the same poses through `detect_contacts_sap["cpu"]` on
     a batched CPU `Data`, contact by contact — `ncon` per lane and every
     field within `CPU_TOL`. ⚠ THE CHECK PRINTS FAIL ON THE G1 FOR BOTH
     KERNELS, AND THE NUMBER BESIDE IT IS THE RESULT: a float32 mesh-mesh
     manifold on a knife edge clips one vertex more or fewer on one side
     (§13.52: on the 5090 BOTH kernels sit the SAME 264 lanes of 16384 off
     the CPU, worst 0.057; on Apple 208 for the block kernel and ~750 for
     the serial one). `diag_lanes=N` prints N mismatched lanes' contact
     lists from both sides, which is how a knife edge (same body pair, ±1
     manifold point, shared points 1e-8 apart) is told from a missing
     contact (a pair on one side only). Read the count and the dump, not
     the verdict.

READ THE `csum` LINE. It is a checksum of the whole contact array (every
lane, every record, every field) per snapshot, in float64. Two builds of
this file that print the same `csum` on every snapshot produced the same
contacts to the last bit. That claim holds STATELESS ONLY —
`HILL_WARM_ACROSS_STEPS=False` in both builds — because with the warm
start on, a tie on a flat face lands where the seed sent it (§13.48) and
the block kernel's lanes race on a hash slot; and it holds on NVIDIA only:
on Apple the serial kernel is the side that disagrees with the CPU
(§13.52, `feedback_metal_wide_per_thread_inlinearray_miscompute`), so
`csum(block) != csum(serial)` there says nothing about the block kernel.

THE NUMBERS THAT MATTER. `ms/launch` on the LAST round, averaged over the
snapshots, is the plateau figure of §13.51 for this lane count; the first
round's first snapshot is the cold launch. `ncon` mean/max and the
`saturated` count (lanes at `MAX_CONTACTS`) say whether the workload is
the training one: §13.51's mechanism is contact COUNT.

⚠ ONE LANE COUNT PER BINARY, like the lane sweep. ⚠ The G1 is NVIDIA-only
on the batched SOLVER path (nv 35 broke Metal's per-thread stack); the FK
and collision kernels alone compile and run on Apple, which is what makes
this file usable on a laptop — but the decision numbers are the 5090's
(`_a_terms_repeat_cost_is_not_its_removal_saving`: Apple got the sign
wrong once on a collision A/B).
"""

from std.math import abs, sin
from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import UnitreeG1Model
from mojo_rl.envs.robots.unitree_g1_config import UnitreeG1Config
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.collision.ccd_workspace import (
    COLL_BLOCK_KERNEL, COLL_TPB, COLL_CCD_LANES, COLL_NCAND_CAP,
    COLL_NO_FALLBACK, HILL_WARM_ACROSS_STEPS,
)
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE, METADATA_SIZE, META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B, CONTACT_IDX_DIST,
    CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
)


comptime DT = DType.float32
comptime BATCH: Int = 1024
comptime NQ = UnitreeG1Model.NQ
comptime ACT_DIM = UnitreeG1Model.ACTION_DIM
comptime MAXC = UnitreeG1Model.MAX_CONTACTS
comptime MD = ModelDims[
    UnitreeG1Model,
    nmesh_verts = UnitreeG1Config.NMESH_VERTS,
    nhfield_data = UnitreeG1Config.NHFIELD_DATA,
    nmesh_tri = UnitreeG1Config.NMESH_TRI,
]
comptime Dat = Data[DT, MD, BATCH]
comptime Mod = Model[DT, MD]
# The float32 GPU-vs-CPU band of the parity gates (`test_contact_pair_gpu_parity`).
comptime CPU_TOL: Float64 = 1e-4


def _fmt(v: Float64, places: Int = 3) -> String:
    var mul = 1.0
    for _ in range(places):
        mul *= 10.0
    var scaled = Int(v * mul + (0.5 if v >= 0 else -0.5))
    var whole = scaled // Int(mul)
    var frac = scaled % Int(mul)
    if frac < 0:
        frac = -frac
    var f = String(frac)
    while f.byte_length() < places:
        f = "0" + f
    return String(whole) + "." + f


def _rollouts(
    n_roll: Int, ctrl_steps: Int, first_snap: Int
) raises -> List[Float32]:
    """`[(ctrl_steps - first_snap), n_roll, NQ]` qpos snapshots from the CPU env."""
    var n_snap = ctrl_steps - first_snap
    var snaps = List[Float32](length=n_snap * n_roll * NQ, fill=0)
    var env = UnitreeG1[DT]()
    var t0 = perf_counter_ns()
    var ncon_sum = 0.0
    var ncon_n = 0
    seed(7)
    for r in range(n_roll):
        _ = env.reset()
        # Per-joint amplitude in [0.6, 1.0] with a random sign and phase, at
        # ~0.2 rad/step: saturating PD targets that swing the body over.
        var amp = List[Float64](capacity=ACT_DIM)
        var phase = List[Float64](capacity=ACT_DIM)
        for _j in range(ACT_DIM):
            var a = 0.6 + 0.4 * random_float64()
            if random_float64() < 0.5:
                a = -a
            amp.append(a)
            phase.append(random_float64() * 6.283185307179586)
        var act = ContAction[ACT_DIM]()
        for t in range(ctrl_steps):
            for j in range(ACT_DIM):
                act.data[j] = amp[j] * sin(Float64(t) * 0.2 + phase[j])
            _ = env.step(act)
            if t >= first_snap:
                var s = t - first_snap
                for i in range(NQ):
                    snaps[(s * n_roll + r) * NQ + i] = Float32(env.d.qpos.data[i])
                ncon_sum += Float64(env.d.meta.data[META_IDX_NUM_CONTACTS])
                ncon_n += 1
    var dt_s = Float64(perf_counter_ns() - t0) * 1e-9
    print(
        "rollouts: ", n_roll, " x ", ctrl_steps, " control steps on the CPU in ",
        _fmt(dt_s, 1), " s; CPU env ncon over the snapshots mean ",
        _fmt(ncon_sum / Float64(ncon_n), 2), sep="",
    )
    return snaps^


def _ncon_stats(d: Dat) -> Tuple[Float64, Int, Int, Int, Int]:
    """(mean, max, min, saturated, flagged) over the lanes' downloaded
    `meta`. `flagged` counts lanes at `ncon = -1`: the block kernel's
    serial-fallback mark, visible only with `COLL_NO_FALLBACK` on."""
    var s = 0.0
    var mx = 0
    var mn = 1 << 30
    var sat = 0
    var flagged = 0
    for e in range(BATCH):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if n < 0:
            flagged += 1
            continue
        s += Float64(n)
        if n > mx:
            mx = n
        if n < mn:
            mn = n
        if n >= MAXC:
            sat += 1
    return (s / Float64(BATCH - flagged if flagged < BATCH else 1), mx, mn, sat, flagged)


def _csum(d: Dat) -> Float64:
    """Every lane, every record, every field, weighted by position."""
    var s = 0.0
    for e in range(BATCH):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if n < 0 or n > MAXC:
            return -1.0e30
        for k in range(n):
            var b = (e * MAXC + k) * CONTACT_SIZE
            for f in range(CONTACT_SIZE):
                s += Float64(d.contacts.data[b + f]) * Float64((k + 1) * (f + 1))
    return s


def main() raises:
    var args = argv()
    var n_roll = Int(atol(String(args[1]))) if len(args) > 1 else 128
    var ctrl_steps = Int(atol(String(args[2]))) if len(args) > 2 else 40
    var first_snap = Int(atol(String(args[3]))) if len(args) > 3 else 24
    var rounds = Int(atol(String(args[4]))) if len(args) > 4 else 2
    var cpu_check = (Int(atol(String(args[5]))) if len(args) > 5 else 1) != 0
    # `diag_lanes` > 0 prints that many mismatched lanes' contact lists,
    # both sides: (body_a, body_b) dist pos — to tell a knife-edge pair
    # (dist within float32 noise of the cutoff) from a missing contact.
    var diag_lanes = Int(atol(String(args[6]))) if len(args) > 6 else 0
    if n_roll > BATCH:
        n_roll = BATCH
    var n_snap = ctrl_steps - first_snap
    print("bench_g1_collision: BATCH", BATCH, " rollouts", n_roll,
          " snapshots", n_snap, " (control steps", first_snap, "..",
          ctrl_steps - 1, ") rounds", rounds)
    print("  COLL_BLOCK_KERNEL", COLL_BLOCK_KERNEL, " COLL_TPB", COLL_TPB,
          " COLL_CCD_LANES", COLL_CCD_LANES, " COLL_NCAND_CAP", COLL_NCAND_CAP,
          " COLL_NO_FALLBACK", COLL_NO_FALLBACK, " HILL_WARM_ACROSS_STEPS",
          HILL_WARM_ACROSS_STEPS, " MAX_CONTACTS", MAXC)

    var snaps = _rollouts(n_roll, ctrl_steps, first_snap)

    var ctx = DeviceContext()
    var mf = Mod()
    UnitreeG1Model.init_fields[DT](ctx, mf)
    var d = Dat()
    d.upload_all(ctx)
    var dc = Dat()
    ctx.synchronize()

    var plateau_ms = 0.0
    var plateau_n = 0
    var worst_cpu = 0.0
    var ncon_mismatch = 0
    var lanes_compared = 0
    var records_compared = 0
    for rnd in range(rounds):
        for s in range(n_snap):
            for e in range(BATCH):
                var r = e % n_roll
                for i in range(NQ):
                    d.qpos.data[e * NQ + i] = snaps[(s * n_roll + r) * NQ + i]
            d.qpos.upload(ctx)
            forward_kinematics["gpu", DT, BATCH=BATCH](d, mf, ctx)
            ctx.synchronize()
            var t0 = perf_counter_ns()
            detect_contacts_sap["gpu", DT, BATCH=BATCH](d, mf, ctx)
            ctx.synchronize()
            var ms = Float64(perf_counter_ns() - t0) * 1e-6
            d.contacts.download(ctx)
            d.meta.download(ctx)
            ctx.synchronize()
            var st = _ncon_stats(d)
            var cs = _csum(d)
            print(
                "round ", rnd, " snap ", s, ": ", _fmt(ms, 3), " ms/launch",
                "  ncon mean ", _fmt(st[0], 2), " max ", st[1], " min ", st[2],
                " saturated ", st[3], " flagged ", st[4], "  csum ", cs, sep="",
            )
            if rnd == rounds - 1:
                plateau_ms += ms
                plateau_n += 1
            if cpu_check and rnd == rounds - 1:
                for e in range(BATCH):
                    for i in range(NQ):
                        dc.qpos.data[e * NQ + i] = d.qpos.data[e * NQ + i]
                forward_kinematics["cpu", DT, BATCH=BATCH](dc, mf)
                detect_contacts_sap["cpu", DT, BATCH=BATCH](dc, mf)
                for e in range(BATCH):
                    var ng = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
                    var nc = Int(dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
                    lanes_compared += 1
                    if ng != nc:
                        ncon_mismatch += 1
                        if diag_lanes > 0:
                            diag_lanes -= 1
                            print("  DIAG snap", s, "lane", e, "ncon gpu", ng, "cpu", nc)
                            for side in range(2):
                                var n_side = ng if side == 0 else nc
                                var line = String("    gpu:" if side == 0 else "    cpu:")
                                for k in range(n_side):
                                    var b = (e * MAXC + k) * CONTACT_SIZE
                                    var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A]) if side == 0 else Int(dc.contacts.data[b + CONTACT_IDX_BODY_A])
                                    var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B]) if side == 0 else Int(dc.contacts.data[b + CONTACT_IDX_BODY_B])
                                    var dist = Float64(d.contacts.data[b + CONTACT_IDX_DIST]) if side == 0 else Float64(dc.contacts.data[b + CONTACT_IDX_DIST])
                                    var pz = Float64(d.contacts.data[b + CONTACT_IDX_POS_Z]) if side == 0 else Float64(dc.contacts.data[b + CONTACT_IDX_POS_Z])
                                    line += " (" + String(ba) + "," + String(bb) + ")d=" + String(dist) + "z=" + _fmt(pz, 4)
                                print(line)
                        continue
                    for k in range(ng):
                        var b = (e * MAXC + k) * CONTACT_SIZE
                        records_compared += 1
                        for f in range(CONTACT_SIZE):
                            var diff = abs(
                                Float64(d.contacts.data[b + f])
                                - Float64(dc.contacts.data[b + f])
                            )
                            if diff > worst_cpu:
                                worst_cpu = diff
    print("RESULT g1_collision BATCH=", BATCH, " block=", COLL_BLOCK_KERNEL,
          " plateau_ms_per_launch=", _fmt(plateau_ms / Float64(plateau_n), 3),
          " env_steps_per_s_if_collision_alone=",
          _fmt(Float64(BATCH) * 1000.0 / (plateau_ms / Float64(plateau_n)), 0),
          sep="")
    if cpu_check:
        print("CPU check: lanes compared ", lanes_compared, " ncon mismatches ",
              ncon_mismatch, " records compared ", records_compared,
              " worst |gpu-cpu| ", worst_cpu,
              "  -> ", "PASS" if (ncon_mismatch == 0 and worst_cpu <= CPU_TOL
                                  and records_compared > 0) else "FAIL",
              sep="")
