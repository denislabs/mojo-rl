"""dm_control `pendulum-swingup`: batched GPU path vs the CPU path, per step.

THE GATE THAT DID NOT EXIST. Every other dm_control test in this directory
runs the CPU (`Phyics3dEnv`, float64, one env) path against MuJoCo. That says
nothing about `Phyics3dBatchedEnv`, and two of the three defects the G10 work
turned up are invisible to it:

  * `SYNC_FK_AFTER_STEP` was honoured by `Phyics3dEnv` and IGNORED by
    `Phyics3dBatchedEnv` — so every derived quantity (here: both observation
    entries and the reward, all read off `xmat`) would have been one control
    step stale on the GPU with the CPU gate still green.
  * the GPU hooks are a hand-written SECOND implementation of the reward and
    observation. Nothing forces them to agree with the CPU hooks they mirror.

So this drives BOTH paths from the same injected state with the same action
sequence and diffs obs and reward EVERY step. A per-episode check, or a "does
it learn" check, would hide a one-step lag completely.

⚠ To confirm the gate is not vacuous, flip SANITY_BREAK_SYNC to True: it makes
the CPU side read its obs one step late, which is precisely the defect above.
The run must FAIL. If it passes, the comparison is not measuring what it says.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_pendulum_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_pendulum_gpu_vs_cpu.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs, cos
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.dm_control.pendulum import (
    DMPendulum,
    DMPendulumBatched,
    DMPendulumModel,
)

comptime N_ENVS = 4
comptime NQ = DMPendulumModel.NQ
comptime NV = DMPendulumModel.NV
comptime OBS_DIM = DMPendulumModel.OBS_DIM
comptime ACT_DIM = DMPendulumModel.ACTION_DIM

# Control steps to compare. Long enough that a one-step staleness is
# unmistakable (the pole swings, so xmat_zz moves O(1e-2) per step) and that
# float32 drift shows its real growth rate rather than its first step.
comptime N_STEPS = 60

# The pole's starting angle, rad off vertical. Chosen away from the reward's
# 8-degree bound so the hard indicator is not sitting on its discontinuity,
# where a 1e-7 float32 difference would flip the reward between 0 and 1 and
# the test would be measuring the bound, not the port.
comptime START_ANGLE: Float64 = 0.6

# Agreement bound, float64 CPU vs float32 GPU.
#
# ⚠ Deliberately LOOSE, and NOT the discriminating part of this test. A
# one-step lag shows up at O(1e-2) and a wrong hook (wrong xmat column, wrong
# bound) at O(1) — both orders above this. What the bound absorbs is float32
# rounding compounding through the Euler steps of an unstable pendulum.
comptime TOL: Float64 = 5e-3

# Set True to check the gate can fail — see the module docstring.
comptime SANITY_BREAK_SYNC = False


def test_pendulum_gpu_matches_cpu() raises:
    with DeviceContext() as ctx:
        var cpu = DMPendulum[DType.float64]()
        var gpu = DMPendulumBatched[N_ENVS](ctx)

        _ = cpu.reset()
        gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(7))

        # Force a SHARED start state. The two reset randomisers are different
        # by construction (host `random_float64` vs a per-lane Philox stream),
        # so without this the test would compare two unrelated episodes and
        # assert nothing.
        var qpos0 = List[Float64](length=NQ, fill=0.0)
        var qvel0 = List[Float64](length=NV, fill=0.0)
        qpos0[0] = START_ANGLE
        cpu.set_state(qpos0, qvel0)

        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            for i in range(NQ):
                gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qpos0[i])
            for i in range(NV):
                gpu.d.qvel.data[e * NV + i] = Scalar[DT](qvel0[i])
        gpu.d.qpos.upload(ctx)
        gpu.d.qvel.upload(ctx)
        ctx.synchronize()

        var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
        var h_rew = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        ctx.synchronize()

        var max_obs = 0.0
        var max_rew = 0.0
        var prev_obs = List[Float64](length=OBS_DIM, fill=0.0)

        for t in range(N_STEPS):
            # A slow sinusoidal torque, so the pole actually swings through
            # vertical. A zero or constant action would leave xmat_zz pinned
            # near -1, where the reward is 0 throughout and the observation
            # barely moves — the test would gate nothing.
            var u = 0.9 * cos(Float64(t) * 0.21)

            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    h_act[e * ACT_DIM + j] = Scalar[DT](u)
            ctx.enqueue_copy(gpu._action, h_act)
            gpu.step_batch[N_ENVS](Optional(ctx), 0)
            ctx.enqueue_copy(h_obs, gpu._obs)
            ctx.enqueue_copy(h_rew, gpu._reward)
            ctx.synchronize()

            var act = ContAction[ACT_DIM]()
            for j in range(ACT_DIM):
                act.data[j] = u
            var res = cpu.step(act)
            var cpu_obs = List[Float64](capacity=OBS_DIM)
            for k in range(OBS_DIM):
                cpu_obs.append(res[0].data[k])
            var cpu_rew = Float64(res[1])

            comptime if SANITY_BREAK_SYNC:
                # Emulate the one-control-step staleness: compare against the
                # PREVIOUS CPU observation. Must make this test fail.
                var stale = prev_obs.copy()
                prev_obs = cpu_obs.copy()
                cpu_obs = stale^

            # Every lane, not just lane 0: they start identical, so a kernel
            # that indexed a lane wrongly would show up here rather than
            # silently agreeing.
            for e in range(N_ENVS):
                for k in range(OBS_DIM):
                    var d = abs(
                        Float64(h_obs[e * OBS_DIM + k]) - cpu_obs[k]
                    )
                    if d > max_obs:
                        max_obs = d
                    if d > TOL:
                        print(
                            "OBS MISMATCH step=", t, " env=", e, " k=", k,
                            " gpu=", h_obs[e * OBS_DIM + k],
                            " cpu=", cpu_obs[k], " diff=", d,
                        )
                    assert_true(
                        d <= TOL, "pendulum GPU obs diverges from CPU"
                    )
                var dr = abs(Float64(h_rew[e]) - cpu_rew)
                if dr > max_rew:
                    max_rew = dr
                if dr > TOL:
                    print(
                        "REWARD MISMATCH step=", t, " env=", e,
                        " gpu=", h_rew[e], " cpu=", cpu_rew, " diff=", dr,
                    )
                assert_true(
                    dr <= TOL, "pendulum GPU reward diverges from CPU"
                )

        # Only read under SANITY_BREAK_SYNC; keeps the compiler quiet in the
        # normal build without moving the declaration out of the loop's scope.
        _ = prev_obs

        print(
            "pendulum GPU vs CPU: ", N_STEPS, " steps x ", N_ENVS,
            " lanes — max |obs diff| = ", max_obs,
            ", max |reward diff| = ", max_rew, " (bound ", TOL, ")",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
