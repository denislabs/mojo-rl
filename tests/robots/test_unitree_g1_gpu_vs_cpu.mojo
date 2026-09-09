"""Unitree G1: batched GPU vs CPU, ONE STEP AT A TIME from the same state.

    pixi run -e nvidia mojo run -I . tests/robots/test_unitree_g1_gpu_vs_cpu.mojo
    pixi run -e apple  mojo run -I . tests/robots/test_unitree_g1_gpu_vs_cpu.mojo   # SKIPS

Three envs walk the same driven rollout. The float64 CPU env is the
trajectory; before EVERY control step its `qpos`/`qvel` are injected into
the GPU lanes and into a float32 CPU env, all three take one step, and the
64-D observations are compared. Three columns come out:

    GPU   vs CPU64   — what a trainer on the device sees against the gate
                       the G1 was proved on (`test_unitree_g1_vs_mujoco`)
    CPU32 vs CPU64   — the PRECISION floor: same code, same device, the
                       dtype alone
    GPU   vs CPU32   — the DEVICE column: same dtype, same algorithm; this
                       is the one that is gated

⚠⚠ WHY NOT A FREE-RUNNING LOCKSTEP LIKE `test_humanoid_gpu_vs_cpu`. The
first run of this file on the 5090 (2026-09-09) did that: both lanes
identical to every digit, the first ten steps inside 1e-4 + 1e-3|cpu|,
then 522 elements out of band from step 11, peaking at 0.07 on a joint
velocity of ~5 and falling back to 1e-4 before rising again at step 52.
That shape is the reference's own behaviour under any perturbation:
MuJoCo perturbed by 1e-11 in one joint is 1e-3 from itself at step 99 on
this rollout, non-monotone in the perturbation size. A free-running
comparison of a float32 solve against a float64 one on a contact-rich
35-dof body measures the Lyapunov exponent, not the device. Re-injecting
the state every step is what made the MuJoCo gate decidable
(`docs/menagerie_fidelity_harnesses/g1/inject.py`), and it is what makes
this one decidable.

WHAT THIS GATES THAT THE HUMANOID GATE DOES NOT: `custom_apply_actions_gpu`
(the torque PD per lane per substep, with the effort clip THEN the motor
ctrlrange clamp read from the actuator records), the observation layout on
the device, and mesh-vs-plane / mesh-vs-mesh multicontact on the batched
path at nv 35 with 40 collidable meshes.

⚠⚠ NVIDIA-ONLY. nv 35 is above the nv 24 at which the SO-101 family
exhausted Metal's per-thread stack in the Newton solver; on Apple this
SKIPS with a message. Until it has run green on the 5090 the G1's GPU
hooks are compiled but NOT VALUE-GATED.

⚠ THE BOUND IS PROVISIONAL AND MEASURED, NOT INHERITED. The device column
is gated at `ATOL_DEV + RTOL_DEV * |cpu32|`; the first run prints all
three columns and the bound is pinned from that print. A TF32 path would
show up as a device column 1e3 wider than the precision floor
(`_a_gpu_vs_cpu_band_written_on_metal_is_a_tf32_trap_on_cuda`).
"""

from max.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.robots import UnitreeG1, UnitreeG1Batched
from mojo_rl.envs.robots.unitree_g1_xml import UnitreeG1Model


comptime N_ENVS = 2
comptime N_STEPS = 60
# The device column (GPU vs CPU32): provisional, pinned from the first
# NVIDIA print.
comptime ATOL_DEV = 1e-3
comptime RTOL_DEV = 1e-2


def _action(t: Int, j: Int) -> Float64:
    """The parity gate's drive: every joint at its own phase, 0.3 amplitude."""
    return 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _run(ctx: DeviceContext) raises:
    comptime NQ = UnitreeG1Model.NQ
    comptime NV = UnitreeG1Model.NV
    comptime OBS_DIM = UnitreeG1Model.OBS_DIM
    comptime ACT_DIM = UnitreeG1Model.ACTION_DIM

    var cpu64 = UnitreeG1[DType.float64]()
    var cpu32 = UnitreeG1[DType.float32]()
    var gpu = UnitreeG1Batched[N_ENVS](ctx)
    _ = cpu64.reset()
    _ = cpu32.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(3))

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    ctx.synchronize()

    var worst_gpu64 = 0.0
    var worst_3264 = 0.0
    var worst_dev = 0.0
    var worst_dev_step = -1
    var worst_dev_k = -1
    var worst_lane = 0.0
    var n_bad = 0
    var obs_lo = 1e30
    var obs_hi = -1e30

    for t in range(N_STEPS):
        # ── inject the float64 trajectory's state into the other two ────
        var qp = List[Float64](capacity=NQ)
        var qv = List[Float64](capacity=NV)
        for i in range(NQ):
            qp.append(Float64(cpu64.d.qpos.data[i]))
        for i in range(NV):
            qv.append(Float64(cpu64.d.qvel.data[i]))
        cpu32.set_state(qp, qv)
        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            for i in range(NQ):
                gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qp[i])
            for i in range(NV):
                gpu.d.qvel.data[e * NV + i] = Scalar[DT](qv[i])
        gpu.d.qpos.upload(ctx)
        gpu.d.qvel.upload(ctx)
        ctx.synchronize()

        # ── one control step on all three ────────────────────────────────
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = _action(t, j)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.synchronize()
        var r64 = cpu64.step(act)
        var r32 = cpu32.step(act)

        # ── the three columns, plus lane agreement ───────────────────────
        var step_gpu64 = 0.0
        var step_3264 = 0.0
        var step_dev = 0.0
        for k in range(OBS_DIM):
            var v64 = Float64(r64[0].data[k])
            var v32 = Float64(r32[0].data[k])
            var vg = Float64(h_obs[k])
            if v64 < obs_lo:
                obs_lo = v64
            if v64 > obs_hi:
                obs_hi = v64
            var d_gpu64 = abs(vg - v64)
            var d_3264 = abs(v32 - v64)
            var d_dev = abs(vg - v32)
            if d_gpu64 > step_gpu64:
                step_gpu64 = d_gpu64
            if d_3264 > step_3264:
                step_3264 = d_3264
            if d_dev > step_dev:
                step_dev = d_dev
            if d_dev > worst_dev:
                worst_dev = d_dev
                worst_dev_step = t
                worst_dev_k = k
            if d_dev > ATOL_DEV + RTOL_DEV * abs(v32):
                print(
                    "  DEVICE MISMATCH step=", t, " k=", k, " gpu=", vg,
                    " cpu32=", v32, " cpu64=", v64, " |gpu-cpu32|=", d_dev,
                )
                n_bad += 1
            for e in range(1, N_ENVS):
                var dl = abs(Float64(h_obs[e * OBS_DIM + k]) - vg)
                if dl > worst_lane:
                    worst_lane = dl
        if step_gpu64 > worst_gpu64:
            worst_gpu64 = step_gpu64
        if step_3264 > worst_3264:
            worst_3264 = step_3264
        if t < 3 or t % 10 == 9:
            print(
                "      step", t, " |gpu-cpu64|", step_gpu64,
                " |cpu32-cpu64|", step_3264, " |gpu-cpu32|", step_dev,
            )

    print(
        "  unitree_g1 one-step GPU vs CPU over", N_STEPS, "steps x", N_ENVS,
        "lanes:",
    )
    print("    GPU   vs CPU64 (trainer vs the proved gate) worst", worst_gpu64)
    print("    CPU32 vs CPU64 (the precision floor)        worst", worst_3264)
    print(
        "    GPU   vs CPU32 (the device column, GATED)    worst", worst_dev,
        " at step", worst_dev_step, " k", worst_dev_k, "  bound", ATOL_DEV,
        "+", RTOL_DEV, "*|cpu32|",
    )
    print("    lane-to-lane worst", worst_lane, "   cpu64 obs range [",
          obs_lo, ",", obs_hi, "]")
    assert_true(
        worst_lane == 0.0,
        "the lanes disagree by " + String(worst_lane)
        + " on identical inputs — a race or a per-lane indexing defect",
    )
    assert_true(
        obs_hi - obs_lo > 1.0,
        "the observation never moved — the drive is not driving; the gate"
        " is vacuous",
    )
    assert_true(
        n_bad == 0,
        String(n_bad) + " element(s) of the device column outside"
        " atol+rtol*|cpu32| — see the DEVICE MISMATCH lines above",
    )


def test_unitree_g1_gpu_matches_cpu() raises:
    if not has_nvidia_gpu_accelerator():
        print(
            "  unitree_g1 GPU vs CPU: SKIPPED on Apple — nv 35 exceeds Metal's"
            " per-thread stack in the Newton solver (SO-101 broke it at nv"
            " 24). UNGATED until run on NVIDIA."
        )
        return
    with DeviceContext() as ctx:
        _run(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
