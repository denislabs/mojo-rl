"""The RELEASED BFM-Zero actor in OUR G1 — the G2 rung's gate.

    pixi run mojo run -I . examples/g1/bfm_zero_tracking_gate.mojo
    G2_MUJOCO_COLUMN=/path/to/mujoco_column.csv pixi run mojo run -I . examples/g1/bfm_zero_tracking_gate.mojo
    G2_CLIPS=25 G2_MAX_SEGMENTS=3 pixi run mojo run -I . examples/g1/bfm_zero_tracking_gate.mojo   # a smoke

Three simulators, one policy, one protocol. `tools/g1/bfm_zero_tracking_oracle.py`
holds everything that is not physics — the segment data, the reset state,
the actor-side observation (state, last action, 4-step history in the
reference's layout), the ONNX actor call through onnxruntime, and the
metrics of the reference's `_calc_metrics` — and is imported here through
Python interop. This file only does what a simulator does: set the state,
step 4 substeps of 1/200 s under the reference's PD, hand back `qpos` and
`qvel`. So the columns of the table differ by physics and nothing else:

    ours     — `UnitreeG1[float64]`, this engine, noise off, DR off
    MuJoCo   — the same module's `--mujoco` driver on the reference's own
               scene (`scene_29dof_freebase_noadditional_actuators.xml`),
               noise off, DR off: the paper's sim-to-sim row, re-run
    Isaac    — the released `humanoidverse_tracking_eval.csv`, the model's
               own evaluation at its last training step, 1024 envs, with
               the observation noise and domain randomisation it trained
               under still on

The released z exist for two full clips (`zs_7`: dance1_subject2, 13
ten-second segments; `zs_25`: walk1_subject1, 26 segments), so 39 segments
are scored. The metric is the paper's "tracking": mean over rows of the
L2 joint-angle error against the reference segment (`distance`), plus the
EMD and proximity the CSV carries.

WHAT PASSES. Per segment the rollouts are chaotic — a 500-step contact
rollout diverges from itself under 1e-11 (`docs/BFM_ZERO_G1_REPRODUCTION.md`
§8) — so the gate is on MEANS over a clip's segments, and the band is the
spread the reference itself shows between its two simulators on the same
model. Ours must land within `TOL_REL` of the MuJoCo column's mean
`distance` on each clip (both noise-free, both DR-free: the closest pair),
and the obs pipeline is cross-checked on every segment: the engine's own
64-D observation (`custom_extract_obs_cpu`) against the module's
`state_from(qpos, qvel)`, which must agree to `TOL_OBS` on the LIVE state
after the first step. That band is the float32 rounding of the default
joint-angle table: the module subtracts the store's `default_dof_pos`,
the reference's own float32 tensor, and the engine subtracts the yaml's
float64 value — 1.2e-8 on a 0.3 rad default, measured. On the reset row
itself the two differ by up to ~1.5e-4: the store's root quaternion is
the reference's slerp output, unit to only ~4e-5 on some rows (§10 of the
doc), and the engine's projected gravity uses the unit-quaternion form
while the module uses the reference's `a - b + c`; both are exact for a
unit quaternion, which every live row has. Reported, bounded by
`TOL_OBS_RESET`, not gated at the float32 floor.

⚠ ONE CALL PER STEP THROUGH PYTHON. The actor ONNX is exported at batch 1;
the loop is ~3 ms per control step on an M1 (2 ms actor, 0.4 ms physics,
the rest interop), two minutes for the 39 segments.
"""

from std.math import abs
from std.os import getenv
from std.python import Python, PythonObject

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import UnitreeG1Model


comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime ACT = UnitreeG1Model.ACTION_DIM
comptime STATE_DIM = 64
comptime TOL_REL = 0.10     # |ours - MuJoCo| / MuJoCo on a clip's mean distance
comptime TOL_OBS = 1e-7         # engine obs vs the module's state_from on a LIVE state (float32 default table)
comptime TOL_OBS_RESET = 5e-4   # same on the reset row: the store's non-unit root quaternion


def _py_list(builtins: PythonObject, xs: List[Float64]) raises -> PythonObject:
    var out = builtins.list()
    for i in range(len(xs)):
        _ = out.append(xs[i])
    return out


def _read_qpos_qvel(
    env: UnitreeG1[DType.float64], mut qp: List[Float64], mut qv: List[Float64]
):
    for i in range(NQ):
        qp[i] = Float64(env.d.qpos.data[i])
    for i in range(NV):
        qv[i] = Float64(env.d.qvel.data[i])


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name)
    if s == "":
        return default
    try:
        return Int(s)
    except:
        return default


def main() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var builtins = Python.import_module("builtins")
    var oracle = Python.import_module("bfm_zero_tracking_oracle")
    var proto = oracle.Protocol()
    var mujoco_csv = getenv("G2_MUJOCO_COLUMN")
    var tally = oracle.Tally(proto, mujoco_csv) if mujoco_csv != "" else oracle.Tally(proto)

    var only_clip = _env_int("G2_CLIPS", -1)
    var max_segments = _env_int("G2_MAX_SEGMENTS", 1 << 30)

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)

    print("BFM-Zero released actor in OUR G1 (noise off, DR off)")
    var clips = proto.clips()
    var n_clips = Int(Float64(py=builtins.len(clips)))
    var worst_obs = 0.0
    var worst_obs_reset = 0.0
    var failed = 0
    for ci in range(n_clips):
        var clip = Int(Float64(py=clips[ci]))
        if only_clip >= 0 and clip != only_clip:
            continue
        var n_seg = Int(Float64(py=proto.n_segments(clip)))
        if n_seg > max_segments:
            n_seg = max_segments
        print("clip", clip, String(proto.keys[clip]), ":", n_seg, "segments")
        for seg in range(n_seg):
            var ep = proto.episode(clip, seg)
            var init = ep.init_state()
            for i in range(NQ):
                qp[i] = Float64(py=init[0][i])
            for i in range(NV):
                qv[i] = Float64(py=init[1][i])
            env.set_state(qp, qv)
            _read_qpos_qvel(env, qp, qv)
            _ = ep.record(_py_list(builtins, qp))
            var T = Int(Float64(py=ep.T))

            for t in range(T - 1):
                # obs pipeline cross-check: the engine's own 64-D observation
                # against the module's state_from — reset row reported, the
                # first live row gated
                if t <= 1:
                    var ours_obs = env.get_obs_list()
                    var ref_obs = ep.state_from(_py_list(builtins, qp), _py_list(builtins, qv))[0]
                    for i in range(STATE_DIM):
                        var e = abs(Float64(ours_obs[i]) - Float64(py=ref_obs[i]))
                        if t == 0 and e > worst_obs_reset:
                            worst_obs_reset = e
                        if t == 1 and e > worst_obs:
                            worst_obs = e
                var a = ep.act(_py_list(builtins, qp), _py_list(builtins, qv))
                var act = ContAction[ACT]()
                for j in range(ACT):
                    act.data[j] = Float64(py=a[j])
                _ = env.step(act)
                _read_qpos_qvel(env, qp, qv)
                _ = ep.record(_py_list(builtins, qp))
            var m = ep.metrics()
            print(String(tally.add(clip, seg, m)))
        print(String(tally.report(clip)))
        var ours = Float64(py=tally.mean(clip))
        var mj = Float64(py=tally.mean(clip, "distance", "mujoco"))
        if mj == mj:  # a MuJoCo column was given
            var rel = abs(ours - mj) / mj
            print("  ours vs MuJoCo mean distance: rel", rel, "(band", TOL_REL, ")")
            if rel > TOL_REL:
                failed += 1
    print("obs pipeline: engine obs vs state_from, live row worst |d| =", worst_obs, "(band", TOL_OBS, ")")
    print("              reset row worst |d| =", worst_obs_reset, "(band", TOL_OBS_RESET, ": the store's non-unit root quaternion)")
    if worst_obs > TOL_OBS or worst_obs_reset > TOL_OBS_RESET:
        failed += 1
    if failed > 0:
        print("G2 GATE FAILED:", failed, "checks")
    else:
        print("G2 GATE PASSED")
