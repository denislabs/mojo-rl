"""The block collision kernel at LIBERO's operating point — PERFORMANCE.md §13.55.

    pixi run -e nvidia mojo build -I . -o /tmp/libero_coll \\
        benchmarks/physics3d_gpu/bench_libero_collision.mojo
    /tmp/libero_coll [--windows 5,30,45] [--snaps 8] [--rounds 2]
                     [--cpu-check 1] [--demo-offset 0]

    # the arms, built and run interleaved, then the table:
    pixi run -e nvidia bash scripts/libero_collision_arms.sh build
    pixi run -e nvidia bash scripts/libero_collision_arms.sh run
    pixi run python scripts/libero_collision_arms.py

WHY. After §13.54 moved the elliptic Newton solve onto the blocked kernel,
LIBERO's physics step on the RTX 5090 (256 lanes) is 63% ONE kernel:
`_detect_contacts_sap_block_kernel`, 9.47 ms per substep, launches spanning
7.1-14.0 ms. The G1 work of §13.52 tuned that kernel on 40 geoms; LIBERO has
240 (156 colliding, mostly boxes, 11 meshes), a table under everything and a
144-contact budget. This file reproduces the WORKLOAD without the solver, the
controller or the env, so one build answers one question.

WHAT IT FOUND FIRST (4 lanes, Apple, the `report` arm): on EVERY lane the
sweep passes 377-500 AABB pairs against the 256-candidate cap, so every lane
overflows, is marked, and the serial per-env kernel does its collision —
the block kernel never runs its narrow phase on LIBERO. Only 25-127 of those
pairs survive the narrow phase's first rejects, which `COLL_PREFILTER`
applies at listing — production on NVIDIA since the 5090 measured it exact
and 10-19x on this launch (PERFORMANCE.md §13.55). ⚠ Off on Metal: there
the block kernel drops the box/box contacts (`test_box_box_sap_gpu_parity`,
a Metal miscompute), which the overflow had been hiding.

WHAT IT DOES.

  1. POSES FROM THE RECORDED DEMONSTRATIONS. Lane `e` carries task `e % 10`,
     its `(e // 10)`-th demo (`libero_demo_batched`'s mapping). For every
     WINDOW start `w` and snapshot `s < snaps`, the lane's pose is the demo's
     recorded state at control step `w + s` (clamped to its last step),
     remapped into our `qpos` order (`load_state_remap`). The default windows
     are the solver log's regimes: 5 (settled on the table), 30 (grasp and
     carry: the collision count's peak, up to 89 contacts), 45 (placing:
     the heaviest solves). Consecutive snapshots are consecutive control
     steps, so the hill climb's cross-step warm slots see a moving scene.
     ⚠ These are MuJoCo's recorded poses in OUR scene, whose fixtures sit at
     the centre of LIBERO's per-episode draw — up to a centimetre from the
     recording (`libero_demo_batched`'s header). The WORKLOAD (which geoms
     overlap, what is grasped) is the demonstration's; a knife-edge contact
     may differ from the replayed env's. `libero_goal` activates every slot
     for every task, so no inactive prop is parked and the poses are the
     env's.
  2. The collision instantiation is the batched env's: `LiberoGoalModel`
     under `ModelDims[..., nmesh_verts / nhfield_data / nmesh_tri]` from
     `LiberoGoalOscConfig`, float32, `LANES` lanes (edit the constant; the
     arm script `sed`s it). Per snapshot: upload `qpos`, FK on the device,
     then ONE `detect_contacts_sap["gpu"]` timed between two synchronizes
     (the block kernel, plus the flagged-only serial launch unless
     `COLL_NO_FALLBACK`). The last of `rounds` passes over a window is the
     one reported; one untimed launch first absorbs module load.
  3. `--cpu-check 1`: the same poses through `detect_contacts_sap["cpu"]`,
     `ncon` per lane and every field within `CPU_TOL`. On meshes this is
     the float32 knife-edge band (§13.52: hundreds of lanes of 16 384 on
     the G1, same body pair, ±1 manifold point) — read the COUNT and compare
     it ACROSS ARMS: an arm that changes the contact set moves it.
  4. `COLL_CAND_REPORT = True` builds decode the kernel's per-lane report
     (`ccd_workspace.mojo`): candidates, ROUNDS of `COLL_TPB`, kinds in round
     order, sweep length, fallback reasons. Not a timing build.

READ. `RESULT` lines (one per window) are what the arm script tabulates:
`ms_mean` over the last round's snapshots. The `csum` on each snapshot line
is a float64 checksum of every lane's contacts; two builds with equal `csum`
produced the same contacts to the bit — stateless only
(`HILL_WARM_ACROSS_STEPS = False`), because a warm seed can move a tie on a
flat face (§13.48) and the block kernel's lanes race on a hash slot.
"""

from std.math import abs
from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.hdf5.reader import H5File, H5Dataset
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.broadphase_sap import (
    detect_contacts_sap, COLL_STOP_AFTER, COLL_REPORT_BASE, _KIND_PLANE_BASE,
)
from noeira.physics3d.collision.ccd_workspace import (
    COLL_BLOCK_KERNEL, COLL_TPB, COLL_NCAND_CAP, COLL_NO_FALLBACK,
    COLL_STAGE_SLOTS, HILL_WARM_ACROSS_STEPS, COLL_CAND_REPORT,
    COLL_REPORT_HDR, COLL_PREFILTER,
)
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE, METADATA_SIZE, META_IDX_NUM_CONTACTS,
)
from noeira.tasks.libero_state_remap import load_state_remap
from noeira.tasks.libero_goal_xml import LiberoGoalModel
from noeira.tasks.libero_goal_config import LiberoGoalOscConfig


comptime DT = DType.float32
comptime H = DType.float64
comptime LANES: Int = 256
"""The lane count — a compile-time constant, `sed`ed per build like
`libero_demo_batched`'s."""
comptime FAMILY = "libero_goal"
comptime TASK_DIR = "noeira/tasks/tasks/"
comptime DEMO_DIR = "references/libero_demos/libero_goal"
comptime N_TASKS = 10
comptime DEMOS_PER_TASK = (LANES + N_TASKS - 1) // N_TASKS
comptime DEMOS_IN_FILE = 50
comptime N_PAIRS = N_TASKS * DEMOS_PER_TASK
comptime MD = ModelDims[
    LiberoGoalModel,
    nmesh_verts = LiberoGoalOscConfig.NMESH_VERTS,
    nhfield_data = LiberoGoalOscConfig.NHFIELD_DATA,
    nmesh_tri = LiberoGoalOscConfig.NMESH_TRI,
]
comptime NQ = LiberoGoalModel.NQ
comptime MAXC = LiberoGoalModel.MAX_CONTACTS
comptime Dat = Data[DT, MD, LANES]
comptime Mod = Model[DT, MD]
comptime STAGE_ROW = COLL_STAGE_SLOTS * CONTACT_SIZE
# The float32 GPU-vs-CPU band of the parity gates.
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
    var sign = String("-") if (v < 0 and whole == 0) else String("")
    return sign + String(whole) + "." + f


def _task_names() raises -> List[String]:
    var out = List[String]()
    var want = String(FAMILY) + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _rank_name(r: Int) -> String:
    """`mj_geom_type_rank`'s order: MuJoCo's `mjtGeom`."""
    if r == 0:
        return "plane"
    if r == 1:
        return "hfield"
    if r == 2:
        return "sphere"
    if r == 3:
        return "capsule"
    if r == 4:
        return "ellipsoid"
    if r == 5:
        return "cylinder"
    if r == 6:
        return "box"
    if r == 7:
        return "mesh"
    return "?"


def _kind_name(k: Int) -> String:
    if k >= _KIND_PLANE_BASE:
        return "plane-" + _rank_name(k - _KIND_PLANE_BASE)
    return _rank_name(k // 8) + "-" + _rank_name(k % 8)


def _sorted(xs: List[Int]) -> List[Int]:
    var out = xs.copy()
    for i in range(1, len(out)):
        var v = out[i]
        var j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    return out^


def _pct(sorted_xs: List[Int], q: Float64) -> Int:
    if len(sorted_xs) == 0:
        return 0
    var k = Int(Float64(len(sorted_xs) - 1) * q + 0.5)
    return sorted_xs[k]


def _mean(xs: List[Int]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = 0.0
    for i in range(len(xs)):
        s += Float64(xs[i])
    return s / Float64(len(xs))


def _load_poses(
    windows: List[Int], snaps: Int, demo_offset: Int, mut pair_T: List[Int]
) raises -> List[Float32]:
    """`[n_windows, snaps, N_PAIRS, NQ]` qpos from the recorded demo states."""
    var names = _task_names()
    if len(names) != N_TASKS:
        raise Error(
            String(len(names)) + " " + FAMILY + " tasks on disk, this bench"
            " is built for " + String(N_TASKS)
        )
    var remap = load_state_remap(String(FAMILY))
    if remap.nq != NQ:
        raise Error(
            "the state remap maps " + String(remap.nq) + " qpos words, the"
            " model has " + String(NQ)
        )
    var row_words = remap.row_words()
    var raw = unsafe_alloc[Scalar[H]](row_words).as_unsafe_any_origin()
    var row = List[Float64](length=row_words, fill=0.0)
    var qo = List[Float64](length=remap.nq, fill=0.0)
    var vo = List[Float64](length=remap.nv, fill=0.0)
    var nw = len(windows)
    var out = List[Float32](length=nw * snaps * N_PAIRS * NQ, fill=0)
    var t0 = perf_counter_ns()
    for ti in range(N_TASKS):
        var stem = String(
            String(names[ti])[byte = String(FAMILY).byte_length() + 2 : String(names[ti]).byte_length()]
        )
        var path = String(DEMO_DIR) + "/" + stem + "_demo.hdf5"
        if not exists(path):
            raise Error(
                "no demo file at " + path + " (HF `yifengzhu-hf/LIBERO-datasets`)"
            )
        var h5 = H5File(path)
        for k in range(DEMOS_PER_TASK):
            var di = (demo_offset + k) % DEMOS_IN_FILE
            var ds: H5Dataset
            try:
                ds = h5.open_dataset(String("data/demo_") + String(di) + "/states")
            except e:
                raise Error(
                    "cannot open data/demo_" + String(di) + "/states in " + path
                    + " (" + String(e) + ") — `pixi run python"
                    " tools/tasks/check_libero_demos.py`"
                )
            var T = Int(ds.dims[0])
            var pi = ti * DEMOS_PER_TASK + k
            pair_T[pi] = T
            for w in range(nw):
                for s in range(snaps):
                    var t = windows[w] + s
                    if t > T - 1:
                        t = T - 1
                    ds.read_range[H](t, t + 1, raw)
                    for r in range(row_words):
                        row[r] = Float64(raw[unsafe_offset=r])
                    remap.convert_into(row, qo, vo)
                    var base = ((w * snaps + s) * N_PAIRS + pi) * NQ
                    for i in range(NQ):
                        out[base + i] = Float32(qo[i])
    print(
        "poses: ", N_PAIRS, " demos x ", nw, " windows x ", snaps,
        " snapshots from ", DEMO_DIR, " in ",
        _fmt(Float64(perf_counter_ns() - t0) * 1e-9, 1), " s", sep="",
    )
    return out^


def _lane_pair(e: Int) -> Int:
    return (e % N_TASKS) * DEMOS_PER_TASK + e // N_TASKS


def _ncon_stats(d: Dat) -> Tuple[Float64, Int, Int, Int]:
    """(mean over unflagged lanes, max, saturated, flagged)."""
    var s = 0.0
    var mx = 0
    var sat = 0
    var flagged = 0
    for e in range(LANES):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if n < 0:
            flagged += 1
            continue
        s += Float64(n)
        if n > mx:
            mx = n
        if n >= MAXC:
            sat += 1
    var live = LANES - flagged
    return (s / Float64(live if live > 0 else 1), mx, sat, flagged)


def _csum(d: Dat) -> Float64:
    var s = 0.0
    for e in range(LANES):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if n < 0 or n > MAXC:
            continue
        for k in range(n):
            var b = (e * MAXC + k) * CONTACT_SIZE
            for f in range(CONTACT_SIZE):
                s += Float64(d.contacts.data[b + f]) * Float64((k + 1) * (f + 1))
    return s


struct ReportAgg(Movable):
    """`COLL_CAND_REPORT` over one window's last round: every lane of every
    snapshot."""

    var ncand: List[Int]
    var rounds: List[Int]
    var planes: List[Int]
    var sap_n: List[Int]
    var tests: List[Int]
    var shifts: List[Int]
    var aabb_all: List[Int]
    var survive: List[Int]
    var overflow: Int
    var fallback: Int
    var kinds: List[Int]
    var last_kinds: List[Int]

    def __init__(out self):
        self.ncand = List[Int]()
        self.rounds = List[Int]()
        self.planes = List[Int]()
        self.sap_n = List[Int]()
        self.tests = List[Int]()
        self.shifts = List[Int]()
        self.aabb_all = List[Int]()
        self.survive = List[Int]()
        self.overflow = 0
        self.fallback = 0
        self.kinds = List[Int](length=_KIND_PLANE_BASE + 16, fill=0)
        self.last_kinds = List[Int](length=_KIND_PLANE_BASE + 16, fill=0)


def _decode_report(d: Dat, mut agg: ReportAgg) -> Tuple[Int, Int]:
    """Fold one snapshot's per-lane reports into `agg`; returns (max rounds,
    max candidates) over the lanes."""
    var max_r = 0
    var max_c = 0
    for e in range(LANES):
        var rb = e * STAGE_ROW + COLL_REPORT_BASE
        var nc = Int(d.coll_stage.data[rb + 0])
        var tpb = Int(d.coll_stage.data[rb + 7])
        if tpb <= 0:
            tpb = COLL_TPB
        var r = (nc + tpb - 1) // tpb
        agg.ncand.append(nc)
        agg.rounds.append(r)
        agg.sap_n.append(Int(d.coll_stage.data[rb + 4]))
        agg.tests.append(Int(d.coll_stage.data[rb + 5]))
        agg.shifts.append(Int(d.coll_stage.data[rb + 6]))
        agg.aabb_all.append(Int(d.coll_stage.data[rb + 8]))
        agg.survive.append(Int(d.coll_stage.data[rb + 9]))
        if Int(d.coll_stage.data[rb + 1]) != 0:
            agg.overflow += 1
        if Int(d.coll_stage.data[rb + 2]) != 0:
            agg.fallback += 1
        var np = 0
        for p in range(nc):
            var key = Int(d.coll_stage.data[rb + COLL_REPORT_HDR + p])
            if key < 0 or key >= len(agg.kinds):
                continue
            agg.kinds[key] += 1
            if key >= _KIND_PLANE_BASE:
                np += 1
            if r > 0 and p >= (r - 1) * tpb:
                agg.last_kinds[key] += 1
        agg.planes.append(np)
        if r > max_r:
            max_r = r
        if nc > max_c:
            max_c = nc
    return (max_r, max_c)


def _top_kinds(counts: List[Int], n_top: Int) -> String:
    var total = 0
    for k in range(len(counts)):
        total += counts[k]
    if total == 0:
        return String("(none)")
    var used = List[Bool](length=len(counts), fill=False)
    var line = String("")
    for _ in range(n_top):
        var best = -1
        for k in range(len(counts)):
            if used[k] or counts[k] == 0:
                continue
            if best < 0 or counts[k] > counts[best]:
                best = k
        if best < 0:
            break
        used[best] = True
        line += (
            _kind_name(best) + " " + _fmt(100.0 * Float64(counts[best]) / Float64(total), 1)
            + "%  "
        )
    return line^


def _print_report(w: Int, agg: ReportAgg):
    var sc = _sorted(agg.ncand)
    var st = _sorted(agg.tests)
    var ss = _sorted(agg.shifts)
    var max_r = 0
    for i in range(len(agg.rounds)):
        if agg.rounds[i] > max_r:
            max_r = agg.rounds[i]
    var hist = String("")
    for r in range(max_r + 1):
        var n = 0
        for i in range(len(agg.rounds)):
            if agg.rounds[i] == r:
                n += 1
        if n > 0:
            hist += String(r) + ":" + String(n) + " "
    print("REPORT window=", w, " lane-snapshots=", len(agg.ncand), sep="")
    print(
        "  candidates per lane  mean ", _fmt(_mean(agg.ncand), 1), "  p50 ",
        _pct(sc, 0.5), "  p90 ", _pct(sc, 0.9), "  max ", _pct(sc, 1.0),
        "   (cap ", COLL_NCAND_CAP, ")", sep="",
    )
    print("  rounds of COLL_TPB=", COLL_TPB, " (rounds:lanes)  ", hist, sep="")
    var sa = _sorted(agg.aabb_all)
    var sv = _sorted(agg.survive)
    print(
        "  sweep AABB passes, UNCAPPED  mean ", _fmt(_mean(agg.aabb_all), 1),
        "  p90 ", _pct(sa, 0.9), "  max ", _pct(sa, 1.0),
        "  | surviving the pair/body/mask filters  mean ",
        _fmt(_mean(agg.survive), 1), "  p90 ", _pct(sv, 0.9), "  max ",
        _pct(sv, 1.0), sep="",
    )
    print(
        "  plane candidates mean ", _fmt(_mean(agg.planes), 1),
        "  | sweep: geoms ", _fmt(_mean(agg.sap_n), 1), "  AABB tests mean ",
        _fmt(_mean(agg.tests), 0), " p90 ", _pct(st, 0.9), " max ",
        _pct(st, 1.0), "  | sort shifts mean ", _fmt(_mean(agg.shifts), 0),
        " max ", _pct(ss, 1.0), sep="",
    )
    print(
        "  serial fallback: list overflow ", agg.overflow,
        "  any reason ", agg.fallback, sep="",
    )
    print("  kinds over all candidates:  ", _top_kinds(agg.kinds, 10), sep="")
    print("  kinds in each lane's LAST round:  ", _top_kinds(agg.last_kinds, 6), sep="")


def main() raises:
    var args = argv()
    var windows = List[Int]()
    windows.append(5)
    windows.append(30)
    windows.append(45)
    var snaps = 8
    var rounds = 2
    var cpu_check = True
    var demo_offset = 0
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--windows" and i + 1 < len(args):
            windows = List[Int]()
            var parts = String(args[i + 1]).split(",")
            for k in range(len(parts)):
                var piece = String(String(parts[k]).strip())
                if piece.byte_length() > 0:
                    windows.append(Int(piece))
            i += 1
        elif s == "--snaps" and i + 1 < len(args):
            snaps = Int(String(args[i + 1]))
            i += 1
        elif s == "--rounds" and i + 1 < len(args):
            rounds = Int(String(args[i + 1]))
            i += 1
        elif s == "--cpu-check" and i + 1 < len(args):
            cpu_check = Int(String(args[i + 1])) != 0
            i += 1
        elif s == "--demo-offset" and i + 1 < len(args):
            demo_offset = Int(String(args[i + 1]))
            i += 1
        else:
            # ⚠ REFUSED, NOT IGNORED — a dropped argument runs the wrong
            # experiment and says nothing.
            raise Error("bench_libero_collision: unknown argument '" + s + "'")
        i += 1
    if len(windows) == 0 or snaps < 1 or rounds < 1:
        raise Error("bench_libero_collision: need windows, snaps >= 1, rounds >= 1")

    print("bench_libero_collision: LANES", LANES, " windows", len(windows),
          " snaps", snaps, " rounds", rounds, " MAX_CONTACTS", MAXC)
    print(
        "ARM COLL_BLOCK_KERNEL=", COLL_BLOCK_KERNEL, " COLL_TPB=", COLL_TPB,
        " COLL_NO_FALLBACK=", COLL_NO_FALLBACK, " COLL_PREFILTER=",
        COLL_PREFILTER, " COLL_STOP_AFTER=",
        COLL_STOP_AFTER, " COLL_CAND_REPORT=", COLL_CAND_REPORT,
        " HILL_WARM_ACROSS_STEPS=", HILL_WARM_ACROSS_STEPS, sep="",
    )
    comptime if COLL_CAND_REPORT:
        print("  ⚠ COLL_CAND_REPORT build: the times below are NOT the kernel's")

    var pair_T = List[Int](length=N_PAIRS, fill=0)
    var poses = _load_poses(windows, snaps, demo_offset, pair_T)

    var ctx = DeviceContext()
    var mf = Mod()
    LiberoGoalModel.init_fields[DT](ctx, mf)
    var d = Dat()
    d.upload_all(ctx)
    var dc = Dat()
    ctx.synchronize()

    # One untimed launch: module load and first-touch allocations.
    for e in range(LANES):
        var b = _lane_pair(e) * NQ
        for q in range(NQ):
            d.qpos.data[e * NQ + q] = poses[b + q]
    d.qpos.upload(ctx)
    forward_kinematics["gpu", DT, BATCH=LANES](d, mf, ctx)
    detect_contacts_sap["gpu", DT, BATCH=LANES](d, mf, ctx)
    ctx.synchronize()

    for w in range(len(windows)):
        var ended = 0
        for pi in range(N_PAIRS):
            if pair_T[pi] <= windows[w]:
                ended += 1
        print(
            "=== window ", windows[w], ": control steps ", windows[w], "..",
            windows[w] + snaps - 1, "  (", ended, " of ", N_PAIRS,
            " demos already ended; those lanes hold their last pose)", sep="",
        )
        var sum_ms = 0.0
        var min_ms = 1.0e30
        var max_ms = 0.0
        var n_ms = 0
        var ncon_sum = 0.0
        var ncon_max = 0
        var sat_total = 0
        var flagged_total = 0
        var cpu_lanes = 0
        var cpu_mismatch = 0
        var cpu_worst = 0.0
        var agg = ReportAgg()
        for rnd in range(rounds):
            for s in range(snaps):
                for e in range(LANES):
                    var b = ((w * snaps + s) * N_PAIRS + _lane_pair(e)) * NQ
                    for q in range(NQ):
                        d.qpos.data[e * NQ + q] = poses[b + q]
                d.qpos.upload(ctx)
                forward_kinematics["gpu", DT, BATCH=LANES](d, mf, ctx)
                ctx.synchronize()
                var t0 = perf_counter_ns()
                detect_contacts_sap["gpu", DT, BATCH=LANES](d, mf, ctx)
                ctx.synchronize()
                var ms = Float64(perf_counter_ns() - t0) * 1e-6
                d.contacts.download(ctx)
                d.meta.download(ctx)
                comptime if COLL_CAND_REPORT:
                    d.coll_stage.download(ctx)
                ctx.synchronize()
                var st = _ncon_stats(d)
                var extra = String("")
                var last = rnd == rounds - 1
                comptime if COLL_CAND_REPORT:
                    if last:
                        var mr = _decode_report(d, agg)
                        extra = "  rounds max " + String(mr[0]) + " candidates max " + String(mr[1])
                print(
                    "  round ", rnd, " snap ", s, ": ", _fmt(ms, 3), " ms/launch",
                    "  ncon mean ", _fmt(st[0], 1), " max ", st[1],
                    " saturated ", st[2], " flagged ", st[3],
                    "  csum ", _csum(d), extra, sep="",
                )
                if not last:
                    continue
                sum_ms += ms
                n_ms += 1
                if ms < min_ms:
                    min_ms = ms
                if ms > max_ms:
                    max_ms = ms
                ncon_sum += st[0]
                if st[1] > ncon_max:
                    ncon_max = st[1]
                sat_total += st[2]
                flagged_total += st[3]
                if cpu_check and COLL_STOP_AFTER == 0:
                    for e in range(LANES):
                        for q in range(NQ):
                            dc.qpos.data[e * NQ + q] = d.qpos.data[e * NQ + q]
                    forward_kinematics["cpu", DT, BATCH=LANES](dc, mf)
                    detect_contacts_sap["cpu", DT, BATCH=LANES](dc, mf)
                    for e in range(LANES):
                        var ng = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
                        var nc = Int(dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
                        if ng < 0:
                            continue
                        cpu_lanes += 1
                        if ng != nc:
                            cpu_mismatch += 1
                            continue
                        for k in range(ng):
                            var b = (e * MAXC + k) * CONTACT_SIZE
                            for f in range(CONTACT_SIZE):
                                var diff = abs(
                                    Float64(d.contacts.data[b + f])
                                    - Float64(dc.contacts.data[b + f])
                                )
                                if diff > cpu_worst:
                                    cpu_worst = diff
        comptime if COLL_CAND_REPORT:
            _print_report(windows[w], agg)
        else:
            _ = agg
        print(
            "RESULT window=", windows[w], " ms_mean=", _fmt(sum_ms / Float64(n_ms), 3),
            " ms_min=", _fmt(min_ms, 3), " ms_max=", _fmt(max_ms, 3),
            " ncon_mean=", _fmt(ncon_sum / Float64(n_ms), 2), " ncon_max=", ncon_max,
            " saturated=", sat_total, " flagged=", flagged_total,
            " cpu_lanes=", cpu_lanes, " cpu_ncon_mismatch=", cpu_mismatch,
            " cpu_worst=", cpu_worst,
            " tpb=", COLL_TPB, " nofb=", COLL_NO_FALLBACK,
            " stop=", COLL_STOP_AFTER, " report=", COLL_CAND_REPORT,
            " prefilter=", COLL_PREFILTER, sep="",
        )
