"""Facade production-default gate: Phyics3dBatchedEnv' default
physics bundle (SOLVER="newton" + PARALLEL_GPU=True + CRBA_TREEWALK=True,
the legacy GPU production configuration) vs the same facade with the
dense CRBA.

The chain below the facade is already gated piecewise: fields RK4+Newton
== legacy serial Newton BIT-EXACT (test_rk4_newton_fields), parallel _mt
ops == serial BIT-EXACT (test_fields_mt_parity), fields treewalk ==
legacy treewalk BIT-EXACT + dense-vs-treewalk within legacy tolerances
(test_crba_treewalk_fields), hooks/sync machinery == legacy BIT-EXACT
(test_batched_env_fields, pinned to PGS/serial). What remains for the
facade default is exactly the treewalk-vs-dense delta: this gate steps
the two configs in lockstep on walker2d and requires their trajectories
to (a) actually diverge (treewalk provably active) and (b) stay within a
tight tolerance (~1e-8/eval M difference amplified over 16 contact
steps), with identical done timing.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_batched_env_fields_production.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS, METADATA_SIZE

comptime BATCH = 2
comptime N_STEPS = 16
comptime OBS_DIM = Walker2dModel.OBS_DIM
comptime ACT_DIM = Walker2dModel.ACTION_DIM
comptime RESET_SEED = UInt64(123)

# The facade DEFAULT — the legacy production bundle.
comptime EnvProd = Phyics3dBatchedEnv[
    Walker2dModel, Walker2dConfig, BATCH, TERMINATE_ON_UNHEALTHY=True
]
# Same config with dense CRBA — the only knob that is not bit-identical.
comptime EnvDense = Phyics3dBatchedEnv[
    Walker2dModel,
    Walker2dConfig,
    BATCH,
    TERMINATE_ON_UNHEALTHY=True,
    SOLVER="newton",
    PARALLEL_GPU=True,
    CRBA_TREEWALK=False,
]


def _action_val(t: Int, e: Int, j: Int) -> Scalar[DT]:
    return Scalar[DT]((t * 5 + e * 3 + j * 7) % 9 - 4) / 8.0


def main() raises:
    print("--- Facade production default (newton+parallel+treewalk) ---")
    var ctx = DeviceContext()

    var env_p = EnvProd(ctx)
    var env_d = EnvDense(ctx)
    env_p.reset_batch[BATCH](Optional(ctx), RESET_SEED)
    env_d.reset_batch[BATCH](Optional(ctx), RESET_SEED)

    var h_act = ctx.enqueue_create_host_buffer[DT](BATCH * ACT_DIM)
    var h_obs_p = ctx.enqueue_create_host_buffer[DT](BATCH * OBS_DIM)
    var h_obs_d = ctx.enqueue_create_host_buffer[DT](BATCH * OBS_DIM)
    var h_done_p = ctx.enqueue_create_host_buffer[DT](BATCH)
    var h_done_d = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    # Same reset seed + solver-independent reset hooks => identical starts.
    ctx.enqueue_copy(h_obs_p, env_p._obs)
    ctx.enqueue_copy(h_obs_d, env_d._obs)
    ctx.synchronize()
    for i in range(BATCH * OBS_DIM):
        if h_obs_p[i] != h_obs_d[i]:
            raise Error("reset obs differ — reset must be solver-independent")
    print("  reset: obs identical across configs")

    var worst = Float64(0)
    var diverged = False
    var total_ncon = 0
    for t in range(N_STEPS):
        for e in range(BATCH):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = _action_val(t, e, j)
        ctx.enqueue_copy(env_p._action, h_act)
        ctx.enqueue_copy(env_d._action, h_act)
        env_p.step_batch[BATCH](Optional(ctx), 0)
        env_d.step_batch[BATCH](Optional(ctx), 0)

        ctx.enqueue_copy(h_obs_p, env_p._obs)
        ctx.enqueue_copy(h_obs_d, env_d._obs)
        ctx.enqueue_copy(h_done_p, env_p._done)
        ctx.enqueue_copy(h_done_d, env_d._done)
        env_p.d.meta.download(ctx)
        ctx.synchronize()
        for i in range(BATCH * OBS_DIM):
            var err = abs(Float64(h_obs_p[i]) - Float64(h_obs_d[i]))
            if err > worst:
                worst = err
            if h_obs_p[i] != h_obs_d[i]:
                diverged = True
        for e in range(BATCH):
            if h_done_p[e] != h_done_d[e]:
                raise Error("done timing differs at step " + String(t))
            total_ncon += Int(
                env_p.d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )

    print(
        "  worst obs |treewalk - dense| over", N_STEPS, "steps:", worst,
        " contacts seen:", total_ncon,
    )
    if total_ncon == 0:
        raise Error("no contacts over the run — gate is vacuous")
    if not diverged:
        raise Error(
            "trajectories identical — treewalk CRBA provably not active"
        )
    if worst > 1e-3:
        raise Error("treewalk-vs-dense divergence beyond budget")
    print("  PASS: treewalk active, within 1e-3 of dense, done timing exact")
    print("test_batched_env_fields_production: ALL PASS")
