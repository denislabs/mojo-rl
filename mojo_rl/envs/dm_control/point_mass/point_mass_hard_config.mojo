"""dm_control `point_mass-hard` task config — port of `suite/point_mass.py`.

Identical to `easy` in observation, reward, episode length and model. The ONE
difference is `initialize_episode`, which additionally randomizes the mapping
from controls to joints (`PointMass.__init__(randomize_gains=True)`):

    dir1 = random.randn(2); dir1 /= norm(dir1)
    parallel = True
    while parallel:                       # reject a near-parallel second axis
      dir2 = random.randn(2); dir2 /= norm(dir2)
      parallel = abs(dot(dir1, dir2)) > 0.9
    physics.model.wrap_prm[[0, 1]] = dir1
    physics.model.wrap_prm[[2, 3]] = dir2

so each control drives a random linear combination of `root_x`/`root_y`, and
the policy has to infer the mixing from experience. `easy`'s identity mixing is
the special case.

WHERE THE COEFS LIVE. `wrap_prm` for a fixed tendon is the per-joint `coef` of
each `<joint>` wrap, in declaration order: entries 0,1 are `t1`'s coefs on
(root_x, root_y) and 2,3 are `t2`'s. Here that is
`Model.tendons[t, TENDON_IDX_COEF_0/1]` — the RUNTIME records, written by
`custom_reset_model_cpu` below.

WHY THIS CONFIG ALSO OWNS ACTUATION. `MODEL_DEF.apply_actions` reads its
transmission from the COMPTIME tables (`_acd.motor_trn_coef`), baked from the
XML at build time and therefore blind to per-episode writes. Inheriting it
would silently keep the identity mixing and turn `hard` back into `easy` — a
task that trains perfectly well and is simply the wrong one. So the config
returns True from `custom_apply_actions_cpu` and redoes the (short) motor
transmission against the runtime records instead.

Everything else is `easy`'s, reused rather than restated: the joint randomizer,
the sparse-ish `tolerance` reward and the timestep all come from
`DMPointMassConfig`.
"""

from std.random import random_float64
from std.math import sqrt, log, cos, pi, abs

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_DOF_ADR,
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
)

from .point_mass_xml import DMPointMassModel
from .point_mass_config import DMPointMassConfig

from ...phyics3d_env_config import Phyics3dEnvConfig


# `abs(np.dot(dir1, dir2)) > 0.9` — the rejection threshold on the cosine
# between the two actuation directions (both are unit vectors).
comptime PARALLEL_COS: Float64 = 0.9

# Bail-out for the rejection loop. The accepted set is |cos| <= .9, i.e. ~2/pi
# * arccos(.9) ~= 71% of the circle, so 64 draws miss it with probability
# ~1e-35 — the guard is against a wedged RNG, not against bad luck.
comptime MAX_REJECT_TRIES: Int = 64


def _randn() -> Float64:
    """One standard normal, Box-Muller. `np.random.randn` in the reference.

    Only the cosine branch is kept: the sine partner would have to be cached
    across calls to be used, and this is called four times per episode.
    """
    var u1 = random_float64()
    # log(0) is -inf; nudge off the open end rather than resampling, which
    # would bias nothing but reads as if it might.
    if u1 <= 0.0:
        u1 = 1e-12
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


struct DMPointMassHardConfig(Phyics3dEnvConfig):
    # === Physics === (identical to easy — one XML, one timestep, one horizon)
    comptime FRAME_SKIP: Int = DMPointMassConfig.FRAME_SKIP
    comptime MAX_STEPS: Int = DMPointMassConfig.MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = DMPointMassConfig.INTEGRATOR_WS_EXTRA
    comptime SYNC_FK_AFTER_STEP: Bool = DMPointMassConfig.SYNC_FK_AFTER_STEP
    comptime INTEGRATOR: StaticString = DMPointMassConfig.INTEGRATOR

    # === CPU: Observation === (`easy`'s: qpos then qvel, both whole)
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
        return DMPointMassConfig.custom_extract_obs_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites, act, obs
        )

    # === CPU: Reset (state) === (`easy`'s joint randomizer)
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
        DMPointMassConfig.custom_reset_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites
        )

    # === CPU: Reset (model) — THE task difference ===
    @staticmethod
    def custom_reset_model_cpu[
        DTYPE: DType,
    ](
        mut m_bodies: List[Scalar[DTYPE]],
        mut m_joints: List[Scalar[DTYPE]],
        mut m_geoms: List[Scalar[DTYPE]],
        mut m_sites: List[Scalar[DTYPE]],
        mut m_tendons: List[Scalar[DTYPE]],
    ):
        """Randomize the control-to-joint mixing: two unit directions that are
        not too parallel, one per tendon.

        The reference's own draw sequence is NOT reproducible here (different
        RNG), so a rollout cannot be matched episode-for-episode against
        dm_control. The parity test writes the same coefs into both engines and
        gates the physics instead — the same split the ball_in_cup test uses
        for its rejection-sampled qpos.
        """
        var d1x = _randn()
        var d1y = _randn()
        var n1 = sqrt(d1x * d1x + d1y * d1y)
        d1x /= n1
        d1y /= n1

        var d2x = 0.0
        var d2y = 1.0
        for _ in range(MAX_REJECT_TRIES):
            var cx = _randn()
            var cy = _randn()
            var n2 = sqrt(cx * cx + cy * cy)
            cx /= n2
            cy /= n2
            if abs(d1x * cx + d1y * cy) <= PARALLEL_COS:
                d2x = cx
                d2y = cy
                break

        # `wrap_prm[[0, 1]] = dir1` / `wrap_prm[[2, 3]] = dir2`: the two joint
        # wraps of tendon 0, then those of tendon 1, in declaration order.
        var t0 = 0 * MODEL_TENDON_SIZE
        m_tendons[t0 + TENDON_IDX_COEF_0 + 0] = Scalar[DTYPE](d1x)
        m_tendons[t0 + TENDON_IDX_COEF_0 + 1] = Scalar[DTYPE](d1y)
        var t1 = 1 * MODEL_TENDON_SIZE
        m_tendons[t1 + TENDON_IDX_COEF_0 + 0] = Scalar[DTYPE](d2x)
        m_tendons[t1 + TENDON_IDX_COEF_0 + 1] = Scalar[DTYPE](d2y)

    # === CPU: Actuation against the RUNTIME tendon records ===
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
        """`qfrc[dof] += gear * coef * ctrl`, with `coef` read per episode.

        The same arithmetic as `ModelDefFromXML.apply_actions`' motor branch
        (both are `moment^T force` over the transmission, and a `<motor>`'s
        force is just its clamped ctrl), differing only in reading `coef` and
        the DOF address from the runtime records rather than the comptime
        tables. Gear and ctrlrange still come from the comptime tables: `hard`
        randomizes the mixing only, so those are as constant as the XML.

        ACTUATOR a DRIVES TENDON a. The model declares one `<motor tendon=>`
        per `<fixed>`, in the same order, and nothing at runtime records that
        pairing — the parity test pins it by checking each actuator's comptime
        transmission against its tendon's XML coefs.
        """
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)

        comptime nact = DMPointMassModel.nact
        for a in range(nact):
            if a >= len(actions):
                break
            var ctrl = actions[a]
            if ctrl > DMPointMassModel._acd.motor_ctrl_max[a]:
                ctrl = DMPointMassModel._acd.motor_ctrl_max[a]
            elif ctrl < DMPointMassModel._acd.motor_ctrl_min[a]:
                ctrl = DMPointMassModel._acd.motor_ctrl_min[a]
            var gear = DMPointMassModel._acd.motor_gears[a]

            var to = a * MODEL_TENDON_SIZE
            var njnt = Int(m_tendons[to + TENDON_IDX_NUM_JOINTS])
            for k in range(njnt):
                var jid = Int(m_tendons[to + TENDON_IDX_JOINT_0 + k])
                var coef = Float64(m_tendons[to + TENDON_IDX_COEF_0 + k])
                var dadr = Int(
                    m_joints[jid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR]
                )
                if dadr < 0 or dadr >= NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](gear * coef * ctrl)
        return True

    # === CPU: Reward === (`easy`'s, unchanged)
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
        return DMPointMassConfig.compute_reward_and_done_cpu(
            d,
            m_bodies,
            m_joints,
            m_geoms,
            m_sites,
            prev_x,
            actions,
            step_count,
            frame_skip,
        )

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPointMassModel.TIMESTEP)
