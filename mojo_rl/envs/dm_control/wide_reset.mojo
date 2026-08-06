"""Widened start-state distribution for unsupervised data collection.

`docs/BFM_ZERO_SHOT_RL.md` component 1 ranks four diversity levers and calls
varied start states "the most underrated — the best diversity-per-line ratio in
the whole component". This is that lever.

**What the suite already does.** dm_control resets through
`randomizers.randomize_limited_and_rotational_joints`: every LIMITED hinge or
slide is drawn uniformly inside its range, every UNLIMITED hinge uniformly in
[-pi, pi]. Unlike the Gym envs, the joint POSE is therefore already random at
every reset, and re-randomising it here would be redundant.

**What it does not do**, exactly: unlimited SLIDES are left alone, and
velocities are left at zero. For walker those unlimited slides are `rootz` and
`rootx` — the torso height and its horizontal position. So the suite reset
gives a walker with random limbs and a random pitch (`rooty` is an unlimited
hinge, hence drawn) parked at its nominal height with zero velocity. "Lying
down", "airborne" and "already moving" are precisely the states it never
produces, and they are the ones a behavioural foundation model needs to have
seen.

**Why a wrapper and not a fork of each config.** The randomisation is a
property of the COLLECTION run, not of the task: the same walker must reset
narrowly when its reward is being gated against dm_control and widely when it
is filling a dataset. Wrapping keeps one reward implementation with one parity
test, and makes the widening a compile-time opt-in that no evaluation path can
pick up by accident.

    comptime WideWalker = Phyics3dEnv[
        DMWalkerModel,
        WideResetConfig[DMWalkerConfig[0.0], WALKER_ROOTZ_ADR],
        DType.float64, False,
    ]

⚠⚠ **`Z_LO`/`Z_HI` are root JOINT coordinates, not world heights.** A slide
joint displaces its body from the pose declared in the XML, so for walker —
whose torso sits at `pos="0 0 1.3"` — `qpos[rootz] = 0` already means a world
height of 1.3 m, and drawing on [0.1, 1.5] puts the torso between 1.4 m and
2.8 m: permanently airborne, never once lying down. That was the first version
of this file, and it is exactly the failure the header warns about; the numbers
below are the corrected ones.

The range is per-domain and must be read off the model, not guessed. Too low
spawns the torso inside the floor, and the first step then spends itself
resolving a deep penetration — a garbage transition that still gets written to
the dataset. Too high wastes the episode in free fall.
`tests/dm_control/test_wide_reset.mojo` asserts the resulting WORLD height
band rather than the joint range, because the world height is the quantity the
reward reads and the only one that catches this class of mistake.
`RESET_FIND_HEIGHT` is the alternative for models whose floor clearance
depends on the drawn pose (quadruped uses it).
"""

from std.random import random_float64
from std.math import sqrt

from mojo_rl.physics3d.fields import Data

from ..phyics3d_env_config import Phyics3dEnvConfig


# ── walker root DoFs ─────────────────────────────────────────────────────
# qpos layout is joint declaration order: rootz, rootx, rooty, then the six
# leg hinges (walker_xml.mojo:53-76).
comptime WALKER_ROOTZ_ADR: Int = 0

# Torso XML pos is z = 1.3, so these offsets map to world heights
# [0.1, 1.5] m. `_STAND_HEIGHT` is 1.2, which sits inside the band: the
# distribution must cover standing, not just the failure modes around it,
# or `stand`'s reward is never seen anywhere near its maximum.
comptime WALKER_TORSO_NOMINAL_Z: Float64 = 1.3
comptime WALKER_Z_LO: Float64 = 0.1 - WALKER_TORSO_NOMINAL_Z
comptime WALKER_Z_HI: Float64 = 1.5 - WALKER_TORSO_NOMINAL_Z


struct WideResetConfig[
    BASE: Phyics3dEnvConfig,
    ROOT_Z_ADR: Int,
    Z_LO: Float64 = WALKER_Z_LO,
    Z_HI: Float64 = WALKER_Z_HI,
    QVEL_SCALE: Float64 = 0.0,
](Phyics3dEnvConfig):
    """`BASE` in every respect except the reset, which also draws the root
    height and (optionally) the velocities.

    `QVEL_SCALE > 0` draws every velocity coordinate uniformly in
    [-QVEL_SCALE, QVEL_SCALE]. It is off by default because it is the more
    dangerous of the two knobs: a large qvel on a joint the model cannot
    actually reach that fast produces states no policy will ever visit, which
    is anti-diversity — the dataset gets wider while the USEFUL support does
    not. Turn it on with a value read off the model's own rollouts, not a
    round number.
    """

    # ── forwarded, unchanged ─────────────────────────────────────────────
    comptime FRAME_SKIP: Int = Self.BASE.FRAME_SKIP
    comptime MAX_STEPS: Int = Self.BASE.MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = Self.BASE.INTEGRATOR_WS_EXTRA
    comptime INTEGRATOR: StaticString = Self.BASE.INTEGRATOR
    comptime SYNC_FK_AFTER_STEP: Bool = Self.BASE.SYNC_FK_AFTER_STEP
    comptime RNE_POST: Bool = Self.BASE.RNE_POST
    comptime RESET_FIND_HEIGHT: Bool = Self.BASE.RESET_FIND_HEIGHT

    @staticmethod
    def get_timestep() -> Float64:
        return Self.BASE.get_timestep()

    @staticmethod
    def get_reset_noise() -> Float64:
        return Self.BASE.get_reset_noise()

    @staticmethod
    def pre_step_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        Self.BASE.pre_step_cpu(d, prev_x)

    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return Self.BASE.compute_reward_and_done_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites, prev_x, actions,
            step_count, frame_skip,
        )

    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        return Self.BASE.custom_extract_obs_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites, act, obs
        )

    @staticmethod
    def custom_reset_model_cpu[
        DTYPE: DType
    ](
        mut m_bodies: List[Scalar[DTYPE]],
        mut m_joints: List[Scalar[DTYPE]],
        mut m_geoms: List[Scalar[DTYPE]],
        mut m_sites: List[Scalar[DTYPE]],
        mut m_tendons: List[Scalar[DTYPE]],
    ):
        Self.BASE.custom_reset_model_cpu(
            m_bodies, m_joints, m_geoms, m_sites, m_tendons
        )

    @staticmethod
    def custom_apply_actions_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        m_tendons: List[Scalar[DTYPE]],
        actions: List[Float64],
    ) -> Bool:
        return Self.BASE.custom_apply_actions_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites, m_tendons, actions
        )

    # ── the one hook that differs ────────────────────────────────────────
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """The suite reset, then the root DoFs it leaves alone.

        BASE runs FIRST and is not skipped: it is what draws the limb pose and
        the torso pitch, and re-implementing it here would fork a routine that
        is gated against dm_control's own randomizer.
        """
        Self.BASE.custom_reset_cpu(d, m_bodies, m_joints, m_geoms, m_sites)

        comptime assert Self.ROOT_Z_ADR >= 0, (
            "WideResetConfig: ROOT_Z_ADR must name the root height coordinate"
        )
        comptime assert Self.Z_HI > Self.Z_LO, (
            "WideResetConfig: Z_HI must exceed Z_LO"
        )
        if Self.ROOT_Z_ADR < NQ:
            d.qpos.data[Self.ROOT_Z_ADR] = Scalar[DTYPE](
                Self.Z_LO + random_float64() * (Self.Z_HI - Self.Z_LO)
            )

        comptime if Self.QVEL_SCALE > 0.0:
            for i in range(NV):
                d.qvel.data[i] = Scalar[DTYPE](
                    (random_float64() * 2.0 - 1.0) * Self.QVEL_SCALE
                )
