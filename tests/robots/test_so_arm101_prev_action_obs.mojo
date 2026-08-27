"""GATE the previous action in the observation, on BOTH devices.

`SoArmReachConfig.RECORD_PREV_ACTION` puts the last control step's action into
`obs[21..26]`, so the policy can be charged for its ACTION RATE and actually
see the quantity it is charged for. Three things have to hold, and each has a
specific way of failing silently:

1. **the value is the action just applied** — the env writes it at
   action-application time, before physics;
2. **a reset CLEARS it** — `_reset_env_lane` does not touch the TASK_PARAM
   range, so a stale slot means step 0 of every episode carries the previous
   episode's last command and an enormous fake action rate;
3. **CPU and GPU agree** — ⚠⚠ THE TWO PATHS RUN OBS AND REWARD IN OPPOSITE
   ORDER (`Phyics3dEnv`: reward then obs; `Phyics3dBatchedEnv`: obs then
   reward), which is exactly why the write does NOT live in the reward hook.
   A one-step skew here is a policy trained on one observation and evaluated
   on another, with nothing raising.

    pixi run mojo run -I . tests/robots/test_so_arm101_prev_action_obs.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.utils.fmt import col, fixed

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime N_ENVS = 4
comptime BatchedEnvT = Phyics3dBatchedEnv[
    SoArm101Model, SoArm101ReachConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM
comptime ACT_DIM = EnvT.ACTION_DIM
comptime PREV = 21
comptime TOL = 1e-6


def probe(i: Int, j: Int) -> Float64:
    """A distinctive value per (step, joint) — a permutation or an off-by-one
    in the tail shows up as a mismatch rather than as plausible numbers."""
    return -0.9 + 0.11 * Float64(j) + 0.03 * Float64(i)


def main() raises:
    var fails = 0
    print("=" * 72)
    print("PREVIOUS ACTION IN THE OBSERVATION —", OBS_DIM, "dims, prev at",
          PREV)
    print("=" * 72)

    # ── CPU ───────────────────────────────────────────────────────────────
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var s0 = env.reset()
    var zero_ok = True
    for j in range(6):
        if abs(Float64(s0.data[PREV + j])) > TOL:
            zero_ok = False
    print("  [cpu] reset clears the slots:", "PASS" if zero_ok else "FAIL")
    if not zero_ok:
        fails += 1

    for i in range(3):
        var a = EnvT.ActionType()
        for j in range(ACT_DIM):
            a.data[j] = probe(i, j)
        var out = env.step(a)
        var ok = True
        var worst = 0.0
        for j in range(6):
            var e = abs(Float64(out[0].data[PREV + j]) - probe(i, j))
            if e > worst:
                worst = e
            if e > TOL:
                ok = False
        print(
            "  [cpu] step", i, "obs[21..26] == the action applied:",
            "PASS" if ok else "FAIL", " worst", fixed(worst, 8),
        )
        if not ok:
            fails += 1

    # a reset mid-run must clear again
    var s1 = env.reset()
    var re_ok = True
    for j in range(6):
        if abs(Float64(s1.data[PREV + j])) > TOL:
            re_ok = False
    print("  [cpu] reset AFTER stepping clears:", "PASS" if re_ok else "FAIL")
    if not re_ok:
        fails += 1

    # ── GPU, the same contract ────────────────────────────────────────────
    #
    # ⚠ `obs_ptr()` / `action_ptr()` ARE DEVICE POINTERS on this path and
    # dereferencing them from the host CRASHES — the first version of this
    # test did exactly that. Staging goes through `enqueue_copy` over a
    # non-owning `DeviceBuffer` view, the same idiom `driver_offpolicy` uses.
    with DeviceContext() as gctx:
        var benv = BatchedEnvT(gctx)
        benv.reset_batch[N_ENVS](gctx, UInt64(1))
        var host_a = gctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var host_o = gctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
        gctx.synchronize()
        for i in range(3):
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    host_a[e * ACT_DIM + j] = Scalar[DT](probe(i, j))
            var act_view = DeviceBuffer[DT](
                gctx, benv.action_ptr(), N_ENVS * ACT_DIM, owning=False
            )
            gctx.enqueue_copy(act_view, host_a)
            benv.step_batch[N_ENVS](gctx, UInt64(1))
            var obs_view = DeviceBuffer[DT](
                gctx, benv.obs_ptr(), N_ENVS * OBS_DIM, owning=False
            )
            gctx.enqueue_copy(host_o, obs_view)
            gctx.synchronize()
            var ok = True
            var worst = 0.0
            for e in range(N_ENVS):
                for j in range(6):
                    var got = Float64(host_o[e * OBS_DIM + PREV + j])
                    var err = abs(got - probe(i, j))
                    if err > worst:
                        worst = err
                    if err > TOL:
                        ok = False
            print(
                "  [gpu] step", i, "obs[21..26] == the action applied:",
                "PASS" if ok else "FAIL", " worst", fixed(worst, 8),
            )
            if not ok:
                fails += 1

    print("=" * 72)
    if fails == 0:
        print("  ALL PASS — the previous action reaches the observation on")
        print("  both devices, at the same step index, and a reset clears it.")
    else:
        print("  ", fails, "FAILURE(S)")
