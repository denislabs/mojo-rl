"""Unitree G1: batched GPU vs CPU, per step — the G1's own GPU-vs-CPU gate.

    pixi run -e nvidia mojo run -I . tests/robots/test_unitree_g1_gpu_vs_cpu.mojo
    pixi run -e apple  mojo run -I . tests/robots/test_unitree_g1_gpu_vs_cpu.mojo   # expected to SKIP

Read `tests/dm_control/test_pendulum_gpu_vs_cpu.mojo`'s header for why the
comparison has to be per-step, and `test_humanoid_gpu_vs_cpu.mojo` for the
protocol this copies: the same driven action on every lane, the CPU env
stepped beside the batch, the 64-D observation compared element-wise inside
`ATOL + RTOL * |cpu|`.

WHAT THIS GATES THAT THE HUMANOID GATE DOES NOT:

  * `custom_apply_actions_gpu` — the torque PD runs per lane per substep on
    the device (`HAS_CUSTOM_ACTUATION_GPU`) and once per substep on the CPU
    through `CUSTOM_ACTIONS_EVERY_SUBSTEP`. The two cadences are supposed to
    be identical now; this is the measurement.
  * the double clip — effort limit then `<motor ctrlrange>` — read from the
    actuator records on both devices.
  * mesh-vs-plane and mesh-vs-mesh multicontact on the batched path at nv 35
    with 40 collidable meshes (the walker and humanoid gates have no meshes).

⚠⚠ NVIDIA-ONLY. nv 35 is above the nv 24 at which the SO-101 family already
exhausted Metal's per-thread stack in the Newton solver, so on Apple this
SKIPS with a message rather than turning a hardware limit into a red test.
Until it has run on the 5090 the G1's GPU hooks are compiled but NOT
VALUE-GATED — do not read a green Apple run as covering them.

⚠ THE BOUND IS THE HUMANOID GATE'S AND IS PROVISIONAL. fp32 on the device
against fp64 on the host over 60 control steps of contact-rich dynamics; the
humanoid gate's `ATOL 1e-4 / RTOL 1e-3` is the starting point and the first
NVIDIA run sets it from the measurement (print, then pin). A TF32 matmul
path would show up here as a band 1e3 wider than Metal's
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
comptime ATOL = 1e-4
comptime RTOL = 1e-3


def _run(ctx: DeviceContext) raises:
    comptime OBS_DIM = UnitreeG1Model.OBS_DIM
    comptime ACT_DIM = UnitreeG1Model.ACTION_DIM

    var cpu = UnitreeG1[DType.float64]()
    var gpu = UnitreeG1Batched[N_ENVS](ctx)
    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(3))
    # Both resets are the deterministic stand pose, so no state injection is
    # needed — asserted, not assumed.
    gpu.d.qpos.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        for i in range(UnitreeG1Model.NQ):
            assert_true(
                abs(Float64(gpu.d.qpos.data[e * UnitreeG1Model.NQ + i])
                    - Float64(cpu.d.qpos.data[i])) < 1e-6,
                "lane " + String(e) + " reset qpos[" + String(i)
                + "] differs from the CPU reset",
            )

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    ctx.synchronize()

    var max_obs = 0.0
    var n_bad = 0
    var worst_step = -1
    var worst_k = -1
    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            # The parity test's drive: per-joint phase, 0.3 amplitude.
            var u = 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.synchronize()
        var res = cpu.step(act)
        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var cpu_v = res[0].data[k]
                var d = abs(Float64(h_obs[e * OBS_DIM + k]) - cpu_v)
                if d > max_obs:
                    max_obs = d
                    worst_step = t
                    worst_k = k
                if d > ATOL + RTOL * abs(cpu_v):
                    print(
                        "  OBS MISMATCH step=", t, " env=", e, " k=", k,
                        " gpu=", h_obs[e * OBS_DIM + k], " cpu=", cpu_v,
                        " diff=", d,
                    )
                    n_bad += 1
    print(
        "  unitree_g1 GPU vs CPU: ", N_STEPS, " steps x ", N_ENVS,
        " lanes — max |obs diff| = ", max_obs, " (step ", worst_step,
        ", k ", worst_k, ")  bound ", ATOL, " + ", RTOL, "*|cpu|",
    )
    assert_true(
        n_bad == 0,
        String(n_bad) + " element(s) outside atol+rtol*|cpu| — see the"
        " MISMATCH lines above",
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
