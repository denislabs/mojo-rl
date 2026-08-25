"""`dm_control` `quadruped escape` — the 49th and last suite task.

`suite/quadruped.py::Escape`. Everything `Move` observes, plus where the origin
is and what twenty rangefinders see; the reward is uprightness times how far
the robot has escaped a bowl-shaped heightfield.

⚠⚠ THIS IS THE ONLY SUITE TASK WHOSE TERRAIN IS STATE. `<hfield nrow ncol>`
with no `file` is a grid of zeros that `initialize_episode` overwrites on every
reset — see `custom_reset_cpu`. Every other model in the suite is fixed once
built.

WHAT ESCAPE ADDS TO `_common_observations`, in order:

  · `origin` (3) — the world origin expressed in the TORSO frame, i.e.
    `-torso_pos . torso_xmat`. ⚠ NOT the torso's position: it is the vector
    the robot would have to travel to get back, in its own frame, so it
    rotates with the robot.
  · `rangefinder` (20) — `where(reading == -1, 1.0, tanh(reading))`.
    ⚠⚠ A MISS BECOMES 1.0, NOT A NEGATIVE. `-1` is the sentinel `mj_ray`
    returns for no intersection and dm_control replaces it BEFORE the `tanh`,
    so a miss reads as MAXIMUM range — the same value a very distant hit
    saturates to. Feeding -1 through `tanh` instead gives -0.76 on exactly the
    states where the robot has escaped, which is where the reward is highest.

THE REWARD is `_upright_reward(deviation_angle=20) * escape_reward`.
⚠ `deviation_angle=20` here and `0` in `Move` — the bound is
`cos(20 deg) = 0.9397` rather than 1, so the torso may lean 20 degrees before
the term starts falling. Copying `Move`'s upright term would make the task
harder in a way no gate on the walk model can see.
`escape_reward` is `tolerance(|site_xpos['workspace']|, bounds=(30, inf),
margin=30, value_at_margin=0, linear)` — 0 at the centre of the bowl, 1 once
the workspace site is `hfield_size[0]` metres out.
"""

from std.math import sqrt, tanh, inf, cos, pi
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.fields.dims import DimsLike
from mojo_rl.physics3d.ray.model import ray_model
from mojo_rl.physics3d.kinematics.site_frame import site_world_quat_list
from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from mojo_rl.physics3d.sensors.rangefinder import rangefinder_site
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZZ
from mojo_rl.physics3d.gpu.constants import (
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
)
from ..rewards import tolerance, SIGMOID_LINEAR
from .quadruped_config import (
    _upright_reward_at,
    _common_obs_cpu,
    _random_root_orientation,
    QUADRUPED_FRAME_SKIP,
    QUADRUPED_MAX_STEPS,
)
from .quadruped_xml import (
    TORSO_BODY_IDX,
    TORSO_SITE_IDX,
    TOE_SITE_0,
)
from .quadruped_dims import DM_QUADRUPED_ESCAPE_DIMS as qep
from ...phyics3d_env_config import Phyics3dEnvConfig


# ── Model indices, resolved by NAME against the asset ─────────────────────
# ⚠ Pinned in `test_quadruped_escape_vs_dm_control` by looking each one up in
# MuJoCo rather than trusting these; a task fragment that inserted a site
# would otherwise shift every rangefinder silently.
comptime ESCAPE_WORKSPACE_SITE: Int = 2
comptime ESCAPE_RF_SITE_0: Int = 3
comptime ESCAPE_N_RF: Int = 20
comptime ESCAPE_TERRAIN_GEOM: Int = 1

# `hfield_size[0]`, the terrain's x radius. Both the reward's bound and its
# margin.
comptime ESCAPE_TERRAIN_RADIUS: Float64 = 30.0

# `_upright_reward(deviation_angle=20)` — cos(20 deg).
comptime ESCAPE_UPRIGHT_DEVIATION: Float64 = 0.9396926207859084

# `_HEIGHTFIELD_ID`, `_TERRAIN_SMOOTHNESS`, `_TERRAIN_BUMP_SCALE`.
comptime ESCAPE_HFIELD_ID: Int = 0
comptime ESCAPE_TERRAIN_SMOOTHNESS: Float64 = 0.15
comptime ESCAPE_TERRAIN_BUMP_SCALE: Float64 = 2.0


struct DMQuadrupedEscapeConfig(Phyics3dEnvConfig):
    """`Escape` — bowl terrain, twenty rangefinders, distance-from-origin."""

    comptime FRAME_SKIP: Int = QUADRUPED_FRAME_SKIP
    comptime MAX_STEPS: Int = QUADRUPED_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime RNE_POST: Bool = True
    comptime RESET_FIND_HEIGHT: Bool = True
    # ⚠ NO GPU HOOKS. `ray_model` is a host-side linear scan over every geom,
    # and the terrain rewrite is a CPU write followed by an upload — neither
    # has a batched form yet. Claiming `True` here would silently run the
    # batched path with a zero observation for the last 23 dims.
    comptime HAS_GPU_HOOKS: Bool = False

    # ⚠ THE HEIGHTFIELD GRID IS 201x201 AND MUST BE ALLOCATED EXACTLY.
    # `<hfield nrow ncol>` with NO `file` is a grid of zeros that
    # `custom_reset_full_cpu` overwrites every reset — the only model in the
    # suite whose terrain is STATE rather than an asset. Left at the default 0
    # the grid is one element wide and the terrain collides as a flat plane.
    comptime NHFIELD_DATA: Int = 201 * 201

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(qep.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        # `Escape.initialize_episode` randomizes the ORIENTATION only.
        return 0.0

    @staticmethod
    def custom_extract_obs_ray_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) raises -> Bool:
        """`_common_observations` + `origin` + `rangefinder`."""
        if not _common_obs_cpu[DTYPE, TORSO_SITE_IDX, TOE_SITE_0](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            act, obs,
        ):
            return False

        # ── origin, in the torso frame ────────────────────────────────────
        # `-torso_pos.dot(torso_frame)`; numpy's `v.dot(M)` is `M^T v`, so
        # this is the torso rotation's INVERSE applied to `-torso_pos`.
        var bq = QuatGeneric[DTYPE](
            d.xquat.data[TORSO_BODY_IDX * 4 + 3],
            d.xquat.data[TORSO_BODY_IDX * 4 + 0],
            d.xquat.data[TORSO_BODY_IDX * 4 + 1],
            d.xquat.data[TORSO_BODY_IDX * 4 + 2],
        )
        var tp = Vec3Generic[DTYPE](
            d.xpos.data[TORSO_BODY_IDX * 3 + 0],
            d.xpos.data[TORSO_BODY_IDX * 3 + 1],
            d.xpos.data[TORSO_BODY_IDX * 3 + 2],
        )
        var org = bq.rotate_vec_inverse(
            Vec3Generic[DTYPE](-tp.x, -tp.y, -tp.z)
        )
        obs.append(org.x)
        obs.append(org.y)
        obs.append(org.z)

        # ── the twenty rangefinders ───────────────────────────────────────
        # ⚠ `comptime if` RATHER THAN A CONSTRAINT ON THE HOOK. The trait
        # declares this over an unconstrained `DTYPE` and `ray_model` needs
        # floating-point evidence; putting `where DTYPE.is_floating_point()`
        # on the trait member would impose it on all fifty implementers.
        comptime if DTYPE.is_floating_point():
            for i in range(ESCAPE_N_RF):
                var raw = rangefinder_site[DTYPE, D, 1](
                    d, mf, ESCAPE_RF_SITE_0 + i
                )
                # `np.where(rf == -1.0, 1.0, np.tanh(rf))` — A MISS IS 1.0.
                obs.append(
                    Scalar[DTYPE](1.0) if raw == -1.0
                    else Scalar[DTYPE](tanh(raw))
                )
            return True
        else:
            # An integer-typed physics model is not a thing this engine
            # builds; returning False would silently fall back to the default
            # observation, which is the WRONG LENGTH.
            return False

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`orientation = randn(4); orientation /= norm(orientation)`.

        The same draw `Move.initialize_episode` makes; the height is found
        afterwards by `RESET_FIND_HEIGHT`. Shared verbatim with walk/run
        rather than re-derived — see `DMQuadrupedConfig.custom_reset_cpu`.
        """
        _random_root_orientation[DTYPE, D](d)

    @staticmethod
    def custom_reset_full_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
    ) raises:
        """Rewrite the heightfield — `Escape.initialize_episode`.

        ⚠⚠ THIS RUNS BEFORE `_find_non_contacting_height`, and the order is
        load-bearing: that routine raises the robot in 1 cm steps until nothing
        touches, so it must see the terrain it will stand on. Generating the
        terrain afterwards would spawn the quadruped inside a hill.

            radius     = clip(hypot(col, row), .04, 1)   over [-1, 1]^2
            bowl       = .5 - cos(2*pi*radius)/2
            bumps      = uniform(0.15, 1, (bump_res, bump_res))
            terrain    = bowl * zoom(bumps, res/bump_res)

        `bump_res = int(2*hfield_size[0] / 2) = 30` and `res = 201`.

        ⚠⚠ ONE LABELLED DEVIATION: THE UPSAMPLING KERNEL. dm_control calls
        `scipy.ndimage.zoom(..., order=3)` — a cubic B-spline with a recursive
        prefilter and scipy's own boundary handling. This uses BILINEAR
        interpolation on the same grid at the same sample positions. The bowl,
        the bump distribution, the grid, the product and the `[0, 1]`
        normalisation MuJoCo applies on load are all identical; what differs is
        the shape of the interpolation between bump centres, so a given seed
        produces a different-but-equivalent terrain rather than a different
        KIND of terrain.
        ⚠ The consequence, stated rather than buried: a per-episode terrain
        cannot be compared against dm_control's, so
        `test_quadruped_escape_vs_dm_control` writes the SAME grid into both
        engines and gates everything downstream of it. Nothing here gates the
        generator itself beyond its shape.
        """
        var base = ESCAPE_HFIELD_ID * MODEL_HFIELD_META_SIZE
        var adr = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_ADR])
        var nrow = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_NROW])
        var ncol = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_NCOL])
        if nrow != ncol:
            raise Error(
                "escape: the heightfield must be square, got "
                + String(nrow) + "x" + String(ncol)
            )
        var res = nrow
        var sx = Float64(mf.hfield_meta.data[base + HFIELD_META_IDX_SIZE_X])
        var bump_res = Int((2.0 * sx) / ESCAPE_TERRAIN_BUMP_SCALE)
        if bump_res < 2:
            bump_res = 2

        # The bumps, drawn once per reset.
        var bumps = List[Float64](capacity=bump_res * bump_res)
        for _ in range(bump_res * bump_res):
            bumps.append(
                ESCAPE_TERRAIN_SMOOTHNESS
                + random_float64() * (1.0 - ESCAPE_TERRAIN_SMOOTHNESS)
            )

        # `np.ogrid[-1:1:res*1j]` — res samples INCLUSIVE of both ends.
        var step = 2.0 / Float64(res - 1)
        for r in range(res):
            var ry = -1.0 + Float64(r) * step
            for c in range(res):
                var cx = -1.0 + Float64(c) * step
                var rad = sqrt(cx * cx + ry * ry)
                # `np.clip(radius, .04, 1)` — the floor keeps the bowl's
                # centre flat instead of cusped, the ceiling stops the cosine
                # turning back up outside the unit disc.
                if rad < 0.04:
                    rad = 0.04
                if rad > 1.0:
                    rad = 1.0
                var bowl = 0.5 - cos(2.0 * pi * rad) / 2.0

                # Bilinear sample of `bumps` at the same position `zoom` uses
                # with `grid_mode=False`: input index = out * (n_in-1)/(n_out-1).
                var fy = Float64(r) * Float64(bump_res - 1) / Float64(res - 1)
                var fx = Float64(c) * Float64(bump_res - 1) / Float64(res - 1)
                var y0 = Int(fy)
                var x0 = Int(fx)
                var y1 = y0 + 1 if y0 + 1 < bump_res else y0
                var x1 = x0 + 1 if x0 + 1 < bump_res else x0
                var ty = fy - Float64(y0)
                var tx = fx - Float64(x0)
                var b00 = bumps[y0 * bump_res + x0]
                var b01 = bumps[y0 * bump_res + x1]
                var b10 = bumps[y1 * bump_res + x0]
                var b11 = bumps[y1 * bump_res + x1]
                var smooth = (
                    b00 * (1.0 - tx) * (1.0 - ty)
                    + b01 * tx * (1.0 - ty)
                    + b10 * (1.0 - tx) * ty
                    + b11 * tx * ty
                )
                # ⚠⚠ ROUNDED TO float32, BECAUSE `mjModel.hfield_data` IS
                # `float*`. MuJoCo computes the terrain in double and stores
                # it as float, and every ray it casts reads those floats.
                # Keeping our copy in double makes the two engines intersect
                # DIFFERENT SURFACES — measured at 1.1e-07 on the rangefinder
                # parity gate before this cast, from an elevation difference
                # of ~5e-08 m amplified by shallow incidence. Same rule as
                # `load_mesh_hull`'s hull vertices.
                d.hfield_data.data[adr + r * res + c] = Scalar[DTYPE](
                    Float64(bowl * smooth).cast[DType.float32]()
                )

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """`_upright_reward(deviation_angle=20) * escape_reward`."""
        var zz = xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)
        var upright = _upright_reward_at(zz, ESCAPE_UPRIGHT_DEVIATION)

        # `origin_distance()` = |site_xpos['workspace']|, the FULL 3-vector
        # norm and not a horizontal one.
        var wx = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 0])
        var wy = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 1])
        var wz = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 2])
        var dist = sqrt(wx * wx + wy * wy + wz * wz)

        var escape = tolerance[SIGMOID_LINEAR, 0.0](
            dist,
            ESCAPE_TERRAIN_RADIUS,
            inf[DType.float64](),
            ESCAPE_TERRAIN_RADIUS,
        )
        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](upright * escape), False)

