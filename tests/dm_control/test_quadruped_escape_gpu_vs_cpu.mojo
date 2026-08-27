"""dm_control `quadruped-escape`: batched GPU vs CPU, per step.

The first suite gate for a model whose TERRAIN IS STATE, and the first for
`ray_model` running inside a kernel. Three GPU hooks land together here and
each of them fails silently on its own:

  · `init_hfield_gpu`            — a lane with no terrain collides against a
    FLAT PLANE. Both engines then agree beautifully about a task that is not
    escape. `test_the_gpu_terrain_is_a_bowl_per_lane` is the guard.
  · `custom_extract_obs_ray_gpu` — a missing override leaves the last 23 dims
    at whatever the observation buffer last held. Worse, a rangefinder that
    hits NOTHING reads exactly 1.0 on both sides, so a sweep of misses is a
    perfect agreement that proves nothing.
    `test_the_gpu_rangefinders_actually_hit` is the guard.
  · `compute_reward_and_done_gpu` — the default returns 0.0, and so does a CPU
    quadruped that never leaves the bowl's centre.

⚠⚠ THE TERRAIN IS INJECTED, NOT COMPARED, and it has to be. The two
generators draw from DIFFERENT streams by construction — host `random_float64`
in sequence on the CPU, per-lane Philox addressed by bump index on the GPU —
and they compute in different precisions (Float64 is banned on device). So
this file copies the CPU's grid into every lane before stepping, exactly as
`test_quadruped_escape_vs_dm_control` copies ours into MuJoCo. What gates the
GPU generator is its SHAPE, in its own test below, plus the fact that two lanes
must NOT agree — a per-lane draw that returned the same bowl to every lane
would pass every other check in this file.

⚠ THE WINDOW IS AIRBORNE, for the reason `test_quadruped_gpu_vs_cpu.mojo`
spells out at length: a stiff quadruped in contact puts a toe on either side
of the threshold depending on float32-vs-float64 rounding within a substep, so
the two paths disagree on the contact SET rather than on any number. Escape
adds nothing to that argument and inherits it. What is new is that the
rangefinders DO see the terrain from the air — they are what makes an airborne
window a real test of the ray path rather than a test of empty space.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_quadruped_escape_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_quadruped_escape_gpu_vs_cpu.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs, sin, min, max
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
)
from mojo_rl.envs.dm_control.quadruped import N_HINGE
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    DMQuadrupedEscapeModel,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_escape_config import (
    ESCAPE_N_RF,
    DMQuadrupedEscapeConfig,
)

comptime N_ENVS = 2
comptime RES = 201
comptime NHF = RES * RES

comptime NQ = DMQuadrupedEscapeModel.NQ
comptime NV = DMQuadrupedEscapeModel.NV
comptime OBS_DIM = DMQuadrupedEscapeModel.OBS_DIM
comptime ACT_DIM = DMQuadrupedEscapeModel.ACTION_DIM

# The three blocks escape's observation is made of.
comptime O_ORIGIN_0: Int = OBS_DIM - 3 - ESCAPE_N_RF
comptime O_RF_0: Int = OBS_DIM - ESCAPE_N_RF

# 16 control steps of fall, same window as `test_quadruped_gpu_vs_cpu`.
comptime N_STEPS = 16
# ⚠ ABOVE THE BOWL'S RIM, NOT ABOVE A PLANE. The terrain runs to `size[2]` =
# 5 m at the rim, and the drop is over the centre where the bowl is nearly
# flat — but the rangefinders point outward and MUST reach the slope, which is
# the whole reason this window is worth stepping. `ncon == 0` is asserted every
# step, so a mis-sized window fails loudly rather than becoming a contact test.
comptime DROP_Z: Float64 = 1.5

comptime ATOL: Float64 = 5e-3
comptime RTOL: Float64 = 5e-3


def _pose() -> Tuple[List[Float64], List[Float64]]:
    """The shared state both engines are driven from."""
    var qpos = List[Float64](length=NQ, fill=0.0)
    var qvel = List[Float64](length=NV, fill=0.0)
    qpos[2] = DROP_Z
    qpos[3] = 1.0  # free-joint quat is W-FIRST
    for i in range(7, NQ):
        qpos[i] = 0.1
    return (qpos^, qvel^)


def test_the_gpu_terrain_is_a_bowl_per_lane() raises:
    """`init_hfield_gpu`'s SHAPE, and that the lanes DISAGREE.

    ⚠ THE SECOND HALF IS THE LOAD-BEARING ONE. A hook that generated one bowl
    and broadcast it to every lane would pass every shape check here and every
    parity check below — the parity tests overwrite the grid anyway. Per-lane
    variation is the property that cannot be faked, and it is the entire
    reason the terrain lives in `Data` rather than `Model`.
    """
    var ctx = DeviceContext()
    var gpu = Phyics3dBatchedEnv[
        DMQuadrupedEscapeModel, DMQuadrupedEscapeConfig, N_ENVS,
        TERMINATE_ON_UNHEALTHY=False,
    ](ctx)
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(11))

    gpu.d.hfield_data.download(ctx)
    ctx.synchronize()
    var adr = Int(gpu.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    var nrow = Int(gpu.mf.hfield_meta.data[HFIELD_META_IDX_NROW])
    assert_true(
        nrow == RES, "grid is " + String(nrow) + ", expected " + String(RES)
    )

    var diff_lanes = 0.0
    for e in range(N_ENVS):
        var b = e * NHF + adr
        var lo = 1e30
        var hi = -1e30
        for i in range(NHF):
            var v = Float64(gpu.d.hfield_data.data[b + i])
            lo = min(lo, v)
            hi = max(hi, v)
        var centre = Float64(
            gpu.d.hfield_data.data[b + (RES // 2) * RES + RES // 2]
        )
        var rim = Float64(
            gpu.d.hfield_data.data[b + (RES // 2) * RES + (RES * 3) // 4]
        )
        print(
            "  lane", e, " lo", lo, " hi", hi, " centre", centre, " rim", rim
        )
        assert_true(lo >= 0.0 and hi <= 1.0, "terrain outside [0, 1]")
        assert_true(hi > 0.5, "the bowl has no rim — hi is only " + String(hi))
        assert_true(
            centre < 0.1,
            "the bowl's centre is " + String(centre) + ", expected near flat",
        )
        assert_true(
            rim > centre + 0.2,
            "the rim (" + String(rim) + ") is not above the centre ("
            + String(centre) + ") — this is not a bowl",
        )

    for i in range(NHF):
        var a = Float64(gpu.d.hfield_data.data[0 * NHF + adr + i])
        var b2 = Float64(gpu.d.hfield_data.data[1 * NHF + adr + i])
        diff_lanes = max(diff_lanes, abs(a - b2))
    print("  worst |lane0 - lane1|", diff_lanes)
    assert_true(
        diff_lanes > 1e-3,
        "every lane drew the SAME terrain (worst |lane0 - lane1| "
        + String(diff_lanes)
        + "). `init_hfield_gpu` is not per-lane — the whole point of putting"
        + " the grid in `Data` is that a lane resets on its own.",
    )


def test_escape_obs_and_reward_gpu_vs_cpu() raises:
    """The 101 observation dims and the reward, per step, against the CPU."""
    var ctx = DeviceContext()
    var cpu = Phyics3dEnv[
        DMQuadrupedEscapeModel, DMQuadrupedEscapeConfig, DType.float64, False
    ]()
    var gpu = Phyics3dBatchedEnv[
        DMQuadrupedEscapeModel, DMQuadrupedEscapeConfig, N_ENVS,
        TERMINATE_ON_UNHEALTHY=False,
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(11))

    # ── the shared terrain ────────────────────────────────────────────────
    var cadr = Int(cpu.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    var gadr = Int(gpu.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    gpu.d.hfield_data.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        for i in range(NHF):
            gpu.d.hfield_data.data[e * NHF + gadr + i] = Scalar[DT](
                cpu.d.hfield_data.data[cadr + i]
            )
    gpu.d.hfield_data.upload(ctx)
    ctx.synchronize()

    # ── the shared pose ───────────────────────────────────────────────────
    var p = _pose()
    cpu.set_state(p[0], p[1])
    gpu.d.qpos.download(ctx)
    gpu.d.qvel.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        for i in range(NQ):
            gpu.d.qpos.data[e * NQ + i] = Scalar[DT](p[0][i])
        for i in range(NV):
            gpu.d.qvel.data[e * NV + i] = Scalar[DT](p[1][i])
    gpu.d.qpos.upload(ctx)
    gpu.d.qvel.upload(ctx)
    ctx.synchronize()

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var h_rew = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.synchronize()

    var max_abs = 0.0
    var max_rel = 0.0
    var worst_k = -1
    var max_rew = 0.0
    # Non-vacuity, per block.
    var origin_hi = 0.0
    var rf_hits = 0
    var rf_lo = 1e30
    var rf_hi = -1e30
    var rew_lo = 1e30
    var rew_hi = -1e30

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = 0.6 * sin(Float64(t) * 0.23 + Float64(j) * 0.7)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_rew, gpu._reward)
        ctx.synchronize()

        var res = cpu.step(act)
        var cpu_rew = Float64(res[1])
        rew_lo = min(rew_lo, cpu_rew)
        rew_hi = max(rew_hi, cpu_rew)

        gpu.d.meta.download(ctx)
        ctx.synchronize()
        assert_true(
            Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]) == 0
            and Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]) == 0,
            "the window reached the terrain at step "
            + String(t)
            + " (ncon cpu "
            + String(Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]))
            + ", gpu "
            + String(Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]))
            + "). N_STEPS/DROP_Z need re-sizing — this gate is only valid"
            + " while contact plays no part.",
        )

        for k in range(O_ORIGIN_0, O_RF_0):
            origin_hi = max(origin_hi, abs(Float64(res[0].data[k])))
        for k in range(O_RF_0, OBS_DIM):
            var v = Float64(res[0].data[k])
            rf_lo = min(rf_lo, v)
            rf_hi = max(rf_hi, v)
            # ⚠ A MISS IS EXACTLY 1.0 — the sentinel substitution, not a
            # saturated hit. `tanh` reaches 1.0 only asymptotically, so an
            # exact 1.0 is a miss and anything else is a real intersection.
            if v != 1.0:
                rf_hits += 1

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var c_v = Float64(res[0].data[k])
                var g_v = Float64(h_obs[e * OBS_DIM + k])
                var d = abs(c_v - g_v)
                var bound = ATOL + RTOL * abs(c_v)
                max_abs = max(max_abs, d)
                var rel = d / bound
                if rel > max_rel:
                    max_rel = rel
                    worst_k = k
            max_rew = max(max_rew, abs(cpu_rew - Float64(h_rew[e])))

    print("  worst obs |d|        ", max_abs, " at dim", worst_k)
    print("  worst obs / bound    ", max_rel)
    print("  worst reward |d|     ", max_rew)
    print("  origin block peak    ", origin_hi)
    print(
        "  rangefinder hits     ",
        rf_hits,
        "of",
        N_STEPS * ESCAPE_N_RF,
        " range [",
        rf_lo,
        ",",
        rf_hi,
        "]",
    )
    print("  reward range         [", rew_lo, ",", rew_hi, "]")

    # ── non-vacuity, before the comparison verdict ────────────────────────
    assert_true(
        origin_hi > 0.1,
        "the origin block never left zero (peak " + String(origin_hi)
        + ") — dims " + String(O_ORIGIN_0) + ".." + String(O_RF_0 - 1)
        + " are not being written.",
    )
    assert_true(
        rf_hits > N_STEPS * ESCAPE_N_RF // 4,
        "only " + String(rf_hits) + " of " + String(N_STEPS * ESCAPE_N_RF)
        + " rangefinder readings were a HIT. A miss reads exactly 1.0 on both"
        + " sides, so a sweep of misses agrees perfectly and tests nothing."
        + " Re-pose the window so the rays reach the terrain.",
    )
    assert_true(
        rf_hi - rf_lo > 1e-3,
        "every rangefinder reading was the same value (spread "
        + String(rf_hi - rf_lo) + ") — the block is a constant, not a sensor.",
    )
    assert_true(
        rew_hi - rew_lo > 0.0,
        "the reward never moved over " + String(N_STEPS) + " steps; a"
        + " constant agrees with a constant.",
    )

    assert_true(
        max_rel <= 1.0,
        "observation mismatch: worst " + String(max_abs) + " at dim "
        + String(worst_k) + ", " + String(max_rel) + "x the bound.",
    )
    assert_true(
        max_rew <= ATOL + RTOL * max(abs(rew_lo), abs(rew_hi)),
        "reward mismatch: worst " + String(max_rew),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
