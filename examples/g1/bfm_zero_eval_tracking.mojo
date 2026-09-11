"""Tracking evaluation of OUR checkpoints with OUR B — G3.4, on the CPU.

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_eval_tracking.mojo -o /tmp/g1eval
    pixi run /tmp/g1eval --ckpt runs/<id>/checkpoints/step_2000.ckpt \
        [--clips 7 25] [--segments 3] [--out step_2000.csv]

⚠ CHECKPOINTS MOVED (2026-09-09). The trainer used to write `g3_priv.2000`
in the working directory; it now writes `runs/<id>/checkpoints/step_2000.ckpt`
and prints its run directory on the first line. This file needs no change —
`--ckpt` has always taken a path, and `<ckpt>.norm` is read beside it.
    pixi run /tmp/g1eval --random                     # the null baseline: a fresh init

The reference's `tracking_inference` with the checkpoint's networks in place
of the released ones: for a ten-second segment, `z_t = project(B(row t+1))`
on the store's `[state | privileged]` rows (single step, as `tracking_inference`
— the training rollouts use the mean of eight, this does not), reset to
row 0, T − 1 mean-action steps of `π_z` in the CPU G1 env, the metrics of
`_calc_metrics` through the G2 protocol module (`Episode.record` /
`metrics`), so the numbers sit on the same scale as the released actor's
G2 column and the released CSV. The normaliser sidecar `<ckpt>.norm` is
applied to every input of B and π when present (a checkpoint trained with
`normalize_obs`); absent, inputs are raw.

Prints per segment `distance` / `emd` / `proximity` (and the released
Isaac number beside it where the CSV has the segment), per clip means, and
the overall mean over the segments scored — the paper's "tracking"
(1.079 for the released model in MuJoCo; 0.989 overall for the retrained
release in Isaac at 440 M; Fig. 13's 60 M point ≈ 0.91). `--random` is the
null the plan asks for: a random-init net must score like a random policy,
far from any of those.

⚠ SAME DIMS AS THE DRIVER. `H`, `L`, `HB`, `D` must equal
`bfm_zero_train_gpu.mojo`'s — a checkpoint of another size fails to load.
⚠ `-Xlinker -ld_classic` ON macOS (the trainer's nested generics exceed
Apple's ld symbol-name limit; see the agent smoke).
⚠ Five-rung scoring: score several checkpoints of a run, never one
(`fb_online_cpr_walker_gpu.mojo` header: a single late rung read 2.53
where five read 1.74).
"""

from std.math import sqrt, abs
from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.core.cont_action import ContAction
from mojo_rl.core.assignment import emd_uniform
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb.obs_norm import ObsNorm
from mojo_rl.deep_agents.fb.bfm_towers import BFMFTower, BFMActorTower, BFMBNet
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM, UNITREE_G1_PRIV_DIM,
)


comptime OBS: Int = UNITREE_G1_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = 256
comptime H: Int = 1024
comptime L: Int = 3
comptime HB: Int = 256
comptime BATCH: Int = 64        # the trainer's batch — unused at eval, kept small
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime SEG_ROWS: Int = 499

comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = BFMBNet[OBS, D, HB]
comptime ANet = BFMActorTower[OBS, D, H, L, ACT]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _flag(name: String, default: String) -> String:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name and i + 1 < len(args):
            return String(args[i + 1])
    return default


def _has(name: String) -> Bool:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name:
            return True
    return False


def _flag_ints(name: String) -> List[Int]:
    """`--clips 7 25`: every integer after the flag until the next flag."""
    var out = List[Int]()
    var args = argv()
    var on = False
    for i in range(len(args)):
        var a = String(args[i])
        if a == name:
            on = True
            continue
        if on:
            if a.byte_length() > 0 and a.startswith("--"):
                break
            try:
                out.append(atol(a))
            except:
                break
    return out^


def _py_list(builtins: PythonObject, xs: List[Float64]) raises -> PythonObject:
    var out = builtins.list()
    for i in range(len(xs)):
        _ = out.append(xs[i])
    return out


def _project[Dz: Int](mut z: Tensor, row: Int):
    var s = 0.0
    for k in range(Dz):
        var v = Float64(z.data[row * Dz + k])
        s += v * v
    var scale = sqrt(Float64(Dz)) / sqrt(s + 1e-12)
    for k in range(Dz):
        z.data[row * Dz + k] = Scalar[DT](Float64(z.data[row * Dz + k]) * scale)


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    var random_init = _has("--random")
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var max_segments = atol(_flag(String("--segments"), String(1 << 30)))
    var out_csv = _flag(String("--out"), String(""))
    var clips = _flag_ints(String("--clips"))
    if ckpt == "" and not random_init:
        raise Error("pass --ckpt <path> or --random")

    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var builtins = Python.import_module("builtins")
    var oracle = Python.import_module("bfm_zero_tracking_oracle")
    var proto = oracle.Protocol(released=oracle.RELEASED, with_actor=False)
    var tally = oracle.Tally(proto)
    if len(clips) == 0:
        var all = proto.all_clips()
        var n = Int(Float64(py=builtins.len(all)))
        for i in range(n):
            clips.append(Int(Float64(py=all[i])))

    # ── the networks ──────────────────────────────────────────────────
    var t = Trainer.make(lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None, seed=UInt64(7))
    var norm: Optional[ObsNorm[OBS]] = None
    if random_init:
        print("random-init networks (the null baseline)")
    else:
        print("loading", ckpt)
        t.load_state(ckpt)
        norm = ObsNorm[OBS].try_load(ckpt + ".norm")
        if norm:
            print("  normaliser sidecar applied")
        else:
            print("  no .norm sidecar: raw inputs")

    # ── the store's [state | privileged] rows for B ───────────────────
    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    # ⚠ The reference's target is `qpos_ref[:, 7:]` — the 29 joint angles after
    # the 7-wide free joint (`Episode.__init__`). Loading the column here is
    # what lets `emd` be computed natively instead of in the oracle.
    var qpos_col = store.load_column[DType.float32](String("qpos"))

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var b_in = Tensor.alloc(SEG_ROWS * OBS)
    var b_out = Tensor()
    var z_seg = Tensor.alloc(SEG_ROWS * D)
    var t0 = perf_counter_ns()
    var n_scored = 0
    var sum_distance = 0.0
    var sum_emd = 0.0
    # The native EMD is gated against the oracle's on EVERY segment scored, so
    # the agreement is measured on real trajectories rather than only on the
    # synthetic cases of `tests/core/test_assignment_emd.mojo`.
    var worst_emd_gap = 0.0
    var ach = List[Float64](length=SEG_ROWS * ACT, fill=0.0)
    var tgt = List[Float64](length=SEG_ROWS * ACT, fill=0.0)

    for ci in range(len(clips)):
        var clip = clips[ci]
        var n_seg = Int(Float64(py=proto.n_segments(clip)))
        if n_seg > max_segments:
            n_seg = max_segments
        print("clip", clip, String(proto.keys[clip]), ":", n_seg, "segments")
        for seg in range(n_seg):
            var ep = proto.episode(clip, seg)
            var r0 = Int(Float64(py=ep.first_row()))
            var T = Int(Float64(py=ep.T))
            # z_t = project(B(row t+1)), rows 1..T-1 of the segment
            for j in range(T):
                for k in range(UNITREE_G1_STATE_DIM):
                    b_in.data[j * OBS + k] = Scalar[DT](st[(r0 + j) * UNITREE_G1_STATE_DIM + k])
                for k in range(UNITREE_G1_PRIV_DIM):
                    b_in.data[j * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](pv[(r0 + j) * UNITREE_G1_PRIV_DIM + k])
            if norm:
                norm.value().apply_rows(b_in, T)
            t.backward_embed[SEG_ROWS](b_in, b_out)
            var zlist = List[Float64](capacity=(T - 1) * D)
            for j in range(T - 1):
                for k in range(D):
                    z_seg.data[j * D + k] = b_out.data[(j + 1) * D + k]
                _project[D](z_seg, j)
                for k in range(D):
                    zlist.append(Float64(z_seg.data[j * D + k]))
            _ = ep.set_z(_py_list(builtins, zlist))

            # reset and roll
            var init = ep.init_state()
            for i in range(NQ):
                qp[i] = Float64(py=init[0][i])
            for i in range(NV):
                qv[i] = Float64(py=init[1][i])
            env.set_state(qp, qv)
            for i in range(NQ):
                qp[i] = Float64(env.d.qpos.data[i])
            _ = ep.record(_py_list(builtins, qp))
            var nrec = 0
            # ⚠ TWO RECORD SITES, AND THEY MUST STAY IN STEP. The `nrec == T`
            # assert below is the guard: add a `record` without its
            # accumulation and the count no longer matches.
            for k in range(ACT):
                ach[nrec * ACT + k] = qp[7 + k]
            nrec += 1
            for step in range(T - 1):
                var o = env.get_obs_list()
                for k in range(OBS):
                    obs_t.data[k] = Scalar[DT](Float64(o[k]))
                if norm:
                    norm.value().apply_row(obs_t)
                for k in range(D):
                    z1.data[k] = z_seg.data[step * D + k]
                t.act[1](obs_t, z1, act_out)
                var a = ContAction[ACT]()
                for k in range(ACT):
                    var v = Float64(act_out.data[k])
                    if v > 1.0:
                        v = 1.0
                    elif v < -1.0:
                        v = -1.0
                    a.data[k] = v
                _ = env.step(a)
                for i in range(NQ):
                    qp[i] = Float64(env.d.qpos.data[i])
                _ = ep.record(_py_list(builtins, qp))
                for k in range(ACT):
                    ach[nrec * ACT + k] = qp[7 + k]
                nrec += 1
            if nrec != T:
                raise Error(
                    "achieved rows " + String(nrec) + " != T " + String(T)
                    + " — a `record` site lost its native accumulation"
                )
            # target = the store's qpos rows for this segment, joints only
            for j in range(T):
                for k in range(ACT):
                    tgt[j * ACT + k] = Float64(
                        qpos_col[(r0 + j) * NQ + 7 + k]
                    )
            var emd_native = emd_uniform(ach, tgt, T, ACT)
            var m = ep.metrics()
            var emd_gap = abs(emd_native - Float64(py=m["emd"]))
            if emd_gap > worst_emd_gap:
                worst_emd_gap = emd_gap
            sum_emd += emd_native
            print(String(tally.add(clip, seg, m)))
            sum_distance += Float64(py=m["distance"])
            n_scored += 1
        print(String(tally.report(clip)))
    var el = Float64(perf_counter_ns() - t0) * 1e-9
    print("OVERALL: mean distance", sum_distance / Float64(max(n_scored, 1)), "over", n_scored, "segments in", el, "s")
    print("OVERALL: mean emd (native)", sum_emd / Float64(max(n_scored, 1)))
    print("  native-vs-oracle EMD: worst |diff| over", n_scored, "segments =",
          worst_emd_gap)
    # ⚠ NOT A TOLERANCE THAT WAS TUNED. The solve is combinatorial, so the two
    # agree EXACTLY once the costs do; this band is float64 round-off on a sum
    # of 499 terms, and anything above it is a real disagreement.
    if n_scored > 0 and worst_emd_gap > 1e-9:
        raise Error(
            "native EMD disagrees with the oracle by " + String(worst_emd_gap)
            + " — `core/assignment.mojo` and `_emd` have diverged"
        )
    if out_csv != "":
        _ = tally.write_csv(out_csv)
        print("wrote", out_csv)
