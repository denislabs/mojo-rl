"""`quadruped escape` — the 49th suite task, against dm_control.

    pixi run mojo run -I . tests/dm_control/test_quadruped_escape_vs_dm_control.mojo

⚠⚠ THE TERRAIN IS WRITTEN INTO BOTH ENGINES, NOT GENERATED TWICE. Our reset
builds the bowl with BILINEAR interpolation where dm_control uses
`scipy.ndimage.zoom(order=3)` — a labelled deviation recorded in
`quadruped_escape_config.custom_reset_full_cpu`. A per-episode terrain
therefore cannot be compared, and the honest thing is to say so and gate
everything downstream: this takes OUR grid, writes it into MuJoCo, and
compares the observation and the reward computed on the same surface. What is
NOT gated here is the generator, and `test_the_terrain_is_a_bowl` checks only
its SHAPE.

⚠ THE POSES ARE DRIVEN, NOT RESET. `Escape.initialize_episode` draws a random
orientation, so two resets never agree; the gate sets qpos on both sides
instead. ⚠ It uses several poses INCLUDING ones deep in the bowl, because a
rangefinder that misses everything reads 1.0 on both sides and proves nothing —
`test_the_rangefinders_actually_hit` is the guard on that.

⚠ Every hardcoded index in the config (`ESCAPE_WORKSPACE_SITE`,
`ESCAPE_RF_SITE_0`, ...) is re-resolved by NAME here. A task fragment that
inserted a site would otherwise shift all twenty rangefinders silently.
"""

from std.math import abs, sqrt, tanh
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.quadruped import DMQuadrupedEscape
from mojo_rl.envs.dm_control.quadruped.quadruped_escape_config import (
    ESCAPE_WORKSPACE_SITE,
    ESCAPE_RF_SITE_0,
    ESCAPE_N_RF,
    ESCAPE_TERRAIN_GEOM,
    ESCAPE_TERRAIN_RADIUS,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    TORSO_BODY_IDX,
    QUADRUPED_ESCAPE_OBS_DIM,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
)

comptime DT = DType.float64
comptime RES = 201
comptime NPOSE = 4


def _ref() raises -> PythonObject:
    var sys = Python.import_module("sys")
    if Int(py=sys.path.count("tests/dm_control")) == 0:
        _ = sys.path.insert(0, "tests/dm_control")
    return Python.import_module("quadruped_escape_ref")


def _pose(p: Int, i: Int) -> Float64:
    """Free-root qpos, then twelve hinges. Poses chosen to be INSIDE the bowl
    at varying radius, so the rangefinders have terrain to find."""
    if i < 3:
        # x, y, z — the bowl's wall peaks near radius 15 m.
        if i == 0:
            return 0.0 if p == 0 else (6.0 if p == 1 else (12.0 if p == 2 else -9.0))
        if i == 1:
            return 0.0 if p == 0 else (-4.0 if p == 1 else (3.0 if p == 2 else 7.0))
        return 0.7 + 0.15 * Float64(p)
    if i < 7:
        # A mostly-upright quaternion, leaned a little differently each pose.
        if i == 3:
            return 0.98
        if i == 4:
            return 0.05 * Float64(p)
        if i == 5:
            return -0.04 * Float64(p)
        return 0.02 * Float64(p)
    # The twelve hinges.
    return 0.15 * Float64((i + p) % 5) - 0.3


def test_the_indices_the_config_hardcodes() raises:
    """Every constant, resolved by NAME against the compiled model."""
    var R = _ref()
    var md = R.load()
    var m = md[0]
    var ids = R.site_ids(m)
    assert_true(
        Int(py=ids["workspace"]) == ESCAPE_WORKSPACE_SITE,
        "workspace site is " + String(Int(py=ids["workspace"])),
    )
    assert_true(
        Int(py=ids["rf_first"]) == ESCAPE_RF_SITE_0,
        "first rangefinder site is " + String(Int(py=ids["rf_first"])),
    )
    assert_true(Int(py=ids["n_rf"]) == ESCAPE_N_RF, "rangefinder count")
    assert_true(
        Int(py=ids["torso_body"]) == TORSO_BODY_IDX, "torso body index"
    )
    assert_true(
        Int(py=ids["terrain_geom"]) == ESCAPE_TERRAIN_GEOM, "terrain geom"
    )
    assert_true(
        abs(Float64(py=m.hfield_size[0][0]) - ESCAPE_TERRAIN_RADIUS) < 1e-12,
        "terrain radius",
    )
    print("  workspace 2, rf 3..22, torso body 1, terrain geom 1 — all by name")


def test_the_terrain_is_a_bowl() raises:
    """The generator's SHAPE, which is all this file can gate about it.

    ⚠ NOT a comparison against dm_control: our upsampling kernel differs by
    design. What is asserted is what the bowl formula guarantees regardless of
    kernel — a rim that is high, a centre that is low, and every sample inside
    `[0, 1]` because `.5 - cos(.)/2` is and the bumps multiply by at most 1.
    """
    var env = DMQuadrupedEscape[DT]()
    _ = env.reset()
    var base = HFIELD_META_IDX_ADR
    var adr = Int(env.mf.hfield_meta.data[base])
    var nrow = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_NROW])
    assert_true(nrow == RES, "grid is " + String(nrow) + ", expected 201")

    var lo = 1e30
    var hi = -1e30
    for i in range(RES * RES):
        var v = Float64(env.mf.hfield_data.data[adr + i])
        lo = min(lo, v)
        hi = max(hi, v)
    # The bowl peaks at radius 0.5 of the half-extent, i.e. a RIM, and the
    # centre is nearly flat (radius clipped to .04).
    var centre = Float64(
        env.mf.hfield_data.data[adr + (RES // 2) * RES + RES // 2]
    )
    var rim = Float64(
        env.mf.hfield_data.data[adr + (RES // 2) * RES + (RES * 3) // 4]
    )
    print("  terrain lo", lo, " hi", hi, " centre", centre, " rim", rim)
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


def test_the_rangefinders_actually_hit() raises:
    """A miss reads 1.0 on BOTH sides, so a sweep of misses proves nothing."""
    var R = _ref()
    var env = DMQuadrupedEscape[DT]()
    _ = env.reset()
    var md = R.load()
    var m = md[0]
    var d = md[1]
    var mujoco = Python.import_module("mujoco")

    var grid = Python.list()
    var adr = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    for i in range(RES * RES):
        _ = grid.append(Float64(env.mf.hfield_data.data[adr + i]))
    _ = R.set_terrain(m, grid)

    var n_hit = 0
    for p in range(NPOSE):
        for i in range(env.MODEL_DEF.NQ):
            d.qpos[i] = _pose(p, i)
        for i in range(env.MODEL_DEF.NV):
            d.qvel[i] = 0.0
        _ = mujoco.mj_forward(m, d)
        var rf = R.rangefinder(m, d)
        for k in range(ESCAPE_N_RF):
            if Float64(py=rf[k]) != 1.0:
                n_hit += 1
    print("  MuJoCo rangefinder readings that are NOT a miss:", n_hit,
          "of", NPOSE * ESCAPE_N_RF)
    assert_true(
        n_hit > NPOSE * ESCAPE_N_RF // 4,
        "only " + String(n_hit) + " readings hit anything — every pose is"
        " above the terrain and the parity sweep below would compare 1.0"
        " against 1.0 twenty times",
    )


def test_escape_obs_and_reward_vs_dm_control() raises:
    var R = _ref()
    var mujoco = Python.import_module("mujoco")
    var env = DMQuadrupedEscape[DT]()
    _ = env.reset()

    var md = R.load()
    var m = md[0]
    var d = md[1]

    # ⚠ OUR grid into MuJoCo, so both sides ray the same surface.
    var grid = Python.list()
    var adr = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    for i in range(RES * RES):
        _ = grid.append(Float64(env.mf.hfield_data.data[adr + i]))
    _ = R.set_terrain(m, grid)

    var worst_origin = 0.0
    var worst_rf = 0.0
    var worst_reward = 0.0
    var worst_common = 0.0

    for p in range(NPOSE):
        var qp = List[Float64]()
        var qv = List[Float64]()
        for i in range(env.MODEL_DEF.NQ):
            var v = _pose(p, i)
            qp.append(v)
            d.qpos[i] = v
        for i in range(env.MODEL_DEF.NV):
            qv.append(0.0)
            d.qvel[i] = 0.0
        _ = mujoco.mj_forward(m, d)

        var obs = env.obs_at(qp, qv)

        # ── origin (3) ────────────────────────────────────────────────────
        var org = R.origin(m, d)
        for k in range(3):
            worst_origin = max(
                worst_origin,
                abs(Float64(obs.data[78 + k]) - Float64(py=org[k])),
            )

        # ── rangefinder (20) ──────────────────────────────────────────────
        var rf = R.rangefinder(m, d)
        for k in range(ESCAPE_N_RF):
            worst_rf = max(
                worst_rf,
                abs(Float64(obs.data[81 + k]) - Float64(py=rf[k])),
            )

        # ── the common block's torso_upright, as a cheap cross-check ─────
        worst_common = max(
            worst_common,
            abs(Float64(obs.data[47]) - Float64(py=R.torso_upright(m, d))),
        )

        # ── reward ────────────────────────────────────────────────────────
        var rw = env.reward_at(qp, qv, List[Float64](), Scalar[DT](0), 1)
        worst_reward = max(
            worst_reward,
            abs(Float64(rw[0]) - Float64(py=R.escape_reward(m, d))),
        )

    print("  worst |d origin|      ", worst_origin)
    print("  worst |d rangefinder| ", worst_rf)
    print("  worst |d reward|      ", worst_reward)
    print("  torso_upright cross   ", worst_common)

    assert_true(worst_origin < 1e-9, "origin " + String(worst_origin))
    # ⚠⚠ THIS WAS 1.1e-07 AND IS NOW 2.0e-15, AND THE SEVEN ORDERS WERE A
    # STORAGE TYPE. `mjModel.hfield_data` is `float*`; our terrain generator
    # wrote float64, so the two engines were intersecting DIFFERENT SURFACES —
    # an elevation difference of ~5e-08 m amplified by shallow incidence into
    # a tenth of a micrometre of range. Rounding the generator's output to
    # float32 (`custom_reset_full_cpu`) collapsed it. Keep this tolerance
    # TIGHT: at 1e-8 it passed with the defect in place.
    assert_true(worst_rf < 1e-12, "rangefinder " + String(worst_rf))
    assert_true(worst_reward < 1e-9, "reward " + String(worst_reward))
    assert_true(worst_common < 1e-9, "torso_upright " + String(worst_common))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
