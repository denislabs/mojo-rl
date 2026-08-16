"""dm_control `hopper` parity: our env vs MuJoCo + the reference task.

The last Tier A domain, and the first to need `<touch>` sensors — which appear
in hopper's OBSERVATION (`np.log1p(sensordata[['touch_toe','touch_heel']])`),
not only in a reward term, so getting them wrong corrupts the policy input.

Three regimes, three standards of proof, because hopper spends its life in
contact and our contact solver is the known dominant error term:

  1. `test_hopper_model_matches_mujoco` — exact. Dims, masses, inertias, and
     the SITE RECORDS (body / type / size / local pos), which is the direct
     gate on site default-class inheritance. Hopper declares both touch sites
     entirely through `<default class="hopper">`, so before that landed the
     sites had no radius and the sensor zone was a degenerate point.

  2. `test_hopper_airborne_matches_mujoco` — tight. With the hopper thrown
     clear of the floor there are no contacts, so this is the same
     smooth-dynamics gate as the other domains.

  3. `test_hopper_touch_tracks_mujoco` — AGGREGATE, not per-step, and here is
     why, because the reason is not obvious and the weaker gate is not laziness.

     A foot at rest penetrates the floor by ~1e-5 m while carrying the
     hopper's full weight, so penetration depth says nothing about force. Our
     state agrees with MuJoCo's to ~1e-3 on this settling drop — which is
     TWO ORDERS COARSER than that penetration. Whether either engine sees a
     contact at a given instant is therefore undetermined: measured, the
     contact count differs on 12 of 40 steps and MuJoCo reports a loaded site
     on 7 site-steps where we report none. That is the pre-existing contact
     solver residual (see docs/DM_CONTROL_PORT.md) expressing itself through
     a threshold, not a bug in the sensor or in collision detection.

     So this gates what the residual does NOT wash out: both engines must load
     the same SITES, must load them on a comparable FRACTION of the window,
     and where both fire at once the magnitudes must sit inside a wide band.
     A sensor reading the wrong site, the wrong body, or a dead zone fails all
     three; the per-step flicker is printed, not asserted.

     PRACTICAL CONSEQUENCE, worth knowing before training on this env:
     hopper's touch observation is noisier here than in dm_control.

See the pendulum test for why MuJoCo is driven directly rather than through
the dm_control package.

Run with:
    pixi run mojo run -I . tests/dm_control/test_hopper_vs_dm_control.mojo
"""

from std.math import abs, sin, log, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.hopper import (
    DMHopperStand,
    DMHopperHop,
    DMHopperModel,
    TORSO_BODY_IDX,
    FOOT_BODY_IDX,
    TOUCH_TOE_SITE_IDX,
    TOUCH_HEEL_SITE_IDX,
    STAND_HEIGHT,
    HOP_SPEED,
)
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.physics3d.constants import GEOM_SPHERE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_SIZE_0,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    META_IDX_NUM_CONTACTS,
)


comptime Env = DMHopperStand[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/hopper.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 7
comptime NV: Int = 7
comptime NBODY: Int = 6
comptime NGEOM: Int = 7
comptime NSITE: Int = 2
comptime NACT: Int = 4
# hopper.py sets _CONTROL_TIMESTEP = .02 over a .005 model step.
comptime FRAME_SKIP: Int = 4

# MuJoCo's sensordata layout: subtreelinvel occupies 0..2, then the two touches.
comptime REF_TOUCH_TOE_ADR: Int = 3
comptime REF_TOUCH_HEEL_ADR: Int = 4

comptime STATE_TOL_SMOOTH: Float64 = 1e-8
comptime OBS_TOL_SMOOTH: Float64 = 1e-8
comptime MIN_SMOOTH_STEPS: Int = 20
comptime LIMIT_MARGIN: Float64 = 0.02

comptime AMP_AIR: Float64 = 0.05
comptime N_STEPS: Int = 60

# Grounded regime: how far apart the two solvers' contact forces may sit.
# Measured 2026-07-29 on the settling drop below: ratios 0.44..0.84.
comptime TOUCH_RATIO_LO: Float64 = 0.25
comptime TOUCH_RATIO_HI: Float64 = 4.0
comptime N_SETTLE: Int = 40
# A "loaded" foot, well clear of numerical noise.
comptime FORCE_FLOOR: Float64 = 10.0
comptime MIN_LOADED_FIRINGS: Int = 4
# How far our loaded-fraction may sit from MuJoCo's over the window. Measured
# 2026-07-29: 28/40 vs 35/40 per site-step, i.e. 0.80 of MuJoCo's.
comptime FRACTION_TOL: Float64 = 0.35


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    return Python.tuple(mujoco, model, data)


def _build_model() raises -> Model[DType.float64, Dims[nv=DMHopperModel.NV, nbody=DMHopperModel.NBODY, njoint=DMHopperModel.NJOINT, ngeom=DMHopperModel.NGEOM, nequality=DMHopperModel.MAX_EQUALITY, ntendon=DMHopperModel.MAX_TENDON, nsite=DMHopperModel.NSITE, nexclude=DMHopperModel.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=DMHopperModel.NV, nbody=DMHopperModel.NBODY, njoint=DMHopperModel.NJOINT, ngeom=DMHopperModel.NGEOM, nequality=DMHopperModel.MAX_EQUALITY, ntendon=DMHopperModel.MAX_TENDON, nsite=DMHopperModel.NSITE, nexclude=DMHopperModel.NEXCLUDE, nmesh_verts=0]]()
    DMHopperModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_hopper_model_matches_mujoco() raises:
    """Dims, per-body mass / CoM / inertia, and the full site records."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/hopper.xml")

    assert_true(Int(py=m.nbody) == DMHopperModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMHopperModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMHopperModel.NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == DMHopperModel.NV, "nv mismatch")
    assert_true(Int(py=m.ngeom) == DMHopperModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nsite) == DMHopperModel.NSITE, "nsite mismatch")
    assert_true(Int(py=m.nu) == DMHopperModel.nact, "nu mismatch")

    var torso_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    )
    var foot_id = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot"))
    assert_true(torso_id == TORSO_BODY_IDX, "torso body index mismatch")
    assert_true(foot_id == FOOT_BODY_IDX, "foot body index mismatch")

    var mf = _build_model()
    var worst = 0.0
    for b in range(NBODY):
        var base = b * MODEL_BODY_SIZE
        var dm = abs(
            mf.bodies.data[base + BODY_IDX_MASS] - Float64(py=m.body_mass[b])
        )
        if dm > worst:
            worst = dm
        for k in range(3):
            var dp = abs(
                mf.bodies.data[base + BODY_IDX_IPOS_X + k]
                - Float64(py=m.body_ipos[b][k])
            )
            if dp > worst:
                worst = dp
            var di = abs(
                mf.bodies.data[base + BODY_IDX_IXX + k]
                - Float64(py=m.body_inertia[b][k])
            )
            if di > worst:
                worst = di
    print("hopper model build: max |d(mass,ipos,inertia)| =", worst)
    assert_true(worst <= 1e-12, "hopper model differs from MuJoCo")

    # Site records — the gate on site default-class inheritance. Both sites get
    # their type AND size only from <default class="hopper">, so a regression
    # here shows up as radius 0 (a sensor that never fires) rather than a crash.
    var names = ["touch_toe", "touch_heel"]
    for s in range(NSITE):
        var base = s * MODEL_SITE_SIZE
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, names[s])
        )
        assert_true(ref_id == s, "site index mismatch for " + names[s])
        assert_true(
            Int(mf.sites.data[base + SITE_IDX_BODY])
            == Int(py=m.site_bodyid[s]),
            "site body mismatch",
        )
        # Type codes are OUR enum, not MuJoCo's (mjGEOM_SPHERE is 2 there,
        # GEOM_SPHERE is 1 here) — compare against ours, and separately assert
        # MuJoCo also calls it a sphere.
        assert_true(
            Int(mf.sites.data[base + SITE_IDX_TYPE]) == GEOM_SPHERE,
            "site type is not sphere — did the class stop being inherited?",
        )
        assert_true(
            Int(py=m.site_type[s]) == 2, "MuJoCo no longer calls this a sphere"
        )
        var dr = abs(
            Float64(mf.sites.data[base + SITE_IDX_SIZE_0])
            - Float64(py=m.site_size[s][0])
        )
        assert_true(dr <= 1e-15, "site radius differs from MuJoCo")
        assert_true(
            Float64(mf.sites.data[base + SITE_IDX_SIZE_0]) > 0.0,
            "site radius is zero — the touch zone would never fire",
        )
        for k in range(3):
            var dp = abs(
                Float64(mf.sites.data[base + SITE_IDX_POS_X + k])
                - Float64(py=m.site_pos[s][k])
            )
            assert_true(dp <= 1e-15, "site local pos differs from MuJoCo")
    print("  site records match (body, sphere type, radius, local pos)")


def test_hopper_airborne_matches_mujoco() raises:
    """Smooth dynamics with the hopper thrown clear of the floor."""
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]

    var max_state_smooth = 0.0
    var max_obs_smooth = 0.0
    var min_smooth_steps = N_STEPS

    # qpos = [rootx, rootz, rooty, waist, hip, knee, ankle]. rootz lifts the
    # whole hopper; the leg angles start inside their (degree-stated) ranges.
    var inits = [
        [0.0, 2.0, 0.0, 0.0, -0.5, 1.0, 0.0],
        [0.0, 2.5, 0.2, 0.2, -1.0, 1.5, 0.2],
        [0.0, 3.0, -0.3, -0.3, -0.8, 0.8, -0.2],
    ]

    for init in inits:
        mujoco.mj_resetData(model, data)
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        var env = Env()
        _ = env.reset()
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(init[i])
        for _ in range(NV):
            vs.append(0.0)
        env.set_state(qs, vs)

        var smooth = True
        var smooth_steps = 0

        for step in range(N_STEPS):
            var act = Env.ActionType()
            for k in range(NACT):
                var a = AMP_AIR * sin(0.21 * Float64(step) + 0.8 * Float64(k))
                data.ctrl[k] = a
                act.data[k] = a
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            var out = env.step(act)

            # Leave the smooth regime as soon as a contact appears OR a joint
            # reaches a bound — checked BEFORE accumulating, since either acts
            # during the substeps we just ran.
            if Int(py=data.ncon) > 0:
                smooth = False
            for j in range(NQ):
                var lo = Float64(py=model.jnt_range[j][0])
                var hi = Float64(py=model.jnt_range[j][1])
                if lo == hi:
                    continue
                var q = Float64(py=data.qpos[j])
                if q < lo + LIMIT_MARGIN or q > hi - LIMIT_MARGIN:
                    smooth = False

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if smooth and dq > max_state_smooth:
                    max_state_smooth = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if smooth and dv > max_state_smooth:
                    max_state_smooth = dv
            if smooth:
                smooth_steps = step + 1

            # observation: qpos[1:], qvel, then the two touch terms. Airborne
            # both touches are 0 on either side, which the touch test covers
            # properly — here they just have to agree.
            var obs = out[0]
            var oi = 0
            for i in range(1, NQ):
                var d_o = abs(
                    Float64(py=data.qpos[i]) - Float64(obs.data[oi])
                )
                if smooth and d_o > max_obs_smooth:
                    max_obs_smooth = d_o
                oi += 1
            for i in range(NV):
                var d_o = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[oi])
                )
                if smooth and d_o > max_obs_smooth:
                    max_obs_smooth = d_o
                oi += 1
            # ⚠⚠ THE REFERENCE COMES FROM NUMPY, NOT FROM MOJO. This used to
            # be `log(1.0 + ref_toe)` — the same expression the config under
            # test computed — so any error in our log1p arithmetic appeared on
            # BOTH sides and cancelled exactly. The gate was structurally
            # blind to the whole class: measured, `log(1.0 + x)` carries up to
            # 1.02e-09 absolute against `np.log1p` on real touch forces, and
            # `std.math.log1p` up to 3.70e-07, and this leg could not see
            # either. `np.log1p` is what `hopper.py` actually calls.
            var ref_toe = Float64(py=data.sensordata[REF_TOUCH_TOE_ADR])
            var ref_heel = Float64(py=data.sensordata[REF_TOUCH_HEEL_ADR])
            var np_ = Python.import_module("numpy")
            var d_t0 = abs(
                Float64(py=np_.log1p(ref_toe)) - Float64(obs.data[oi])
            )
            var d_t1 = abs(
                Float64(py=np_.log1p(ref_heel)) - Float64(obs.data[oi + 1])
            )
            if smooth and d_t0 > max_obs_smooth:
                max_obs_smooth = d_t0
            if smooth and d_t1 > max_obs_smooth:
                max_obs_smooth = d_t1

        if smooth_steps < min_smooth_steps:
            min_smooth_steps = smooth_steps

    print("hopper (airborne) vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print("  shortest contact-free, limit-free prefix =", min_smooth_steps)
    print(
        "  smooth: max |d(state)| =", max_state_smooth,
        " |d(obs)| =", max_obs_smooth,
    )
    assert_true(
        min_smooth_steps >= MIN_SMOOTH_STEPS,
        "smooth prefix too short — the tight bounds prove nothing",
    )
    assert_true(max_state_smooth <= STATE_TOL_SMOOTH, "physics deviated")
    assert_true(max_obs_smooth <= OBS_TOL_SMOOTH, "observation deviated")


def test_hopper_touch_tracks_mujoco() raises:
    """Touch sensors on a settling drop: same steps, same sites, loose values.

    Deliberately NOT a numeric parity test — read the module docstring. The
    contact forces themselves disagree between the two solvers, so pinning the
    sensor tightly would be pinning that disagreement.
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var mf = _build_model()

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    var env = Env()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for _ in range(NQ):
        qs.append(0.0)
    for _ in range(NV):
        vs.append(0.0)
    env.set_state(qs, vs)

    var ref_loaded = 0
    var our_loaded = 0
    var both_loaded = 0
    var flicker_steps = 0
    var worst_ratio_lo = 1e9
    var worst_ratio_hi = 0.0
    var max_ours = 0.0
    var max_ref = 0.0
    var sane = True
    # Per-site loaded counts — a sensor wired to the wrong site would still
    # match in total while getting these backwards.
    var ref_per_site = [0, 0]
    var our_per_site = [0, 0]

    for _step in range(N_SETTLE):
        var act = Env.ActionType()
        for k in range(NACT):
            act.data[k] = 0.0
            data.ctrl[k] = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        _ = env.step(act)

        if Int(py=data.ncon) != Int(env.d.meta.data[META_IDX_NUM_CONTACTS]):
            flicker_steps += 1

        var refs = [
            Float64(py=data.sensordata[REF_TOUCH_TOE_ADR]),
            Float64(py=data.sensordata[REF_TOUCH_HEEL_ADR]),
        ]
        var ours = [
            touch_sphere_site(env.d, mf.sites.data, TOUCH_TOE_SITE_IDX, 1.0),
            touch_sphere_site(env.d, mf.sites.data, TOUCH_HEEL_SITE_IDX, 1.0),
        ]

        for s in range(NSITE):
            if refs[s] > max_ref:
                max_ref = refs[s]
            if ours[s] > max_ours:
                max_ours = ours[s]
            if not (ours[s] >= 0.0 and ours[s] < 1e9):
                sane = False

            var r_hot = refs[s] > FORCE_FLOOR
            var o_hot = ours[s] > FORCE_FLOOR
            if r_hot:
                ref_loaded += 1
                ref_per_site[s] += 1
            if o_hot:
                our_loaded += 1
                our_per_site[s] += 1
            if r_hot and o_hot:
                both_loaded += 1
                var r = ours[s] / refs[s]
                if r < worst_ratio_lo:
                    worst_ratio_lo = r
                if r > worst_ratio_hi:
                    worst_ratio_hi = r

    var frac = 0.0
    if ref_loaded > 0:
        frac = Float64(our_loaded) / Float64(ref_loaded)

    print("hopper touch over", N_SETTLE, "settling steps:")
    print(
        "  loaded site-steps (>", FORCE_FLOOR, "N ):  MuJoCo", ref_loaded,
        " ours", our_loaded, " both", both_loaded,
        " (ours/MuJoCo =", frac, ")",
    )
    print(
        "  per site — MuJoCo toe/heel", ref_per_site[0], ref_per_site[1],
        "  ours", our_per_site[0], our_per_site[1],
    )
    print(
        "  contact count differed on", flicker_steps, "of", N_SETTLE, "steps",
        "(grazing contacts — not gated, see the docstring)",
    )
    print(
        "  force ratio ours/ref in [", worst_ratio_lo, ",", worst_ratio_hi, "]"
    )
    print("  peak force  ours =", max_ours, " ref =", max_ref)

    assert_true(max_ref > 1.0, "MuJoCo never registered a touch — test vacuous")
    assert_true(sane, "touch sensor produced a negative or absurd value")
    assert_true(
        ref_loaded >= MIN_LOADED_FIRINGS and both_loaded
        >= MIN_LOADED_FIRINGS,
        "too few loaded site-steps for this gate to mean anything",
    )
    # Same sites, both non-trivially loaded — catches a sensor pointed at the
    # wrong site or the wrong body.
    for s in range(NSITE):
        assert_true(
            (ref_per_site[s] > 0) == (our_per_site[s] > 0),
            "a site loaded in one engine and never in the other",
        )
    assert_true(
        abs(frac - 1.0) <= FRACTION_TOL,
        "our loaded fraction is far from MuJoCo's — the sensor is not just"
        " noisy, it is systematically firing too much or too little",
    )
    assert_true(
        worst_ratio_lo >= TOUCH_RATIO_LO and worst_ratio_hi <= TOUCH_RATIO_HI,
        "touch magnitudes outside the contact-solver band — suspect the sensor",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
