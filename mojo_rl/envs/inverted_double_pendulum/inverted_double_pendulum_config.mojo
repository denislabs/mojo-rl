"""InvertedDoublePendulum environment configuration for generic Phyics3dEnv."""

from std.math import sin, cos
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    qvel_offset,
    rk4_extra_workspace_size,
)

from .inverted_double_pendulum_xml import InvertedDoublePendulumModel

from ..phyics3d_env_config import Phyics3dEnvConfig


# Pole segment length (each segment 0.6 m)
comptime _POLE_LEN = 0.6


struct InvertedDoublePendulumConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        InvertedDoublePendulumModel.NQ, InvertedDoublePendulumModel.NV
    ]()

    # Termination threshold: height of tip must be > 1.0 m
    comptime MIN_TIP_HEIGHT = 1.0

    # === CPU: Custom obs extraction (9D with sin/cos encoding) ===
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        # OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2),
        #              clip(qvel[0:3], -10, 10), 0.0]
        obs.append(data.qpos[0])
        obs.append(Scalar[DTYPE](sin(Float64(data.qpos[1]))))
        obs.append(Scalar[DTYPE](sin(Float64(data.qpos[2]))))
        obs.append(Scalar[DTYPE](cos(Float64(data.qpos[1]))))
        obs.append(Scalar[DTYPE](cos(Float64(data.qpos[2]))))
        for i in range(3):
            var v = data.qvel[i]
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs.append(v)
        obs.append(Scalar[DTYPE](0.0))  # qfrc_constraint placeholder
        return True

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        mut prev_x: Scalar[DTYPE],
    ):
        # Save cart x position (qpos[0]) — unused for reward but required by trait
        prev_x = data.qpos[0]

    # === CPU: Reward + termination ===
    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        var q0 = data.qpos[0]  # cart x
        var q1 = data.qpos[1]  # pole1 angle
        var q2 = data.qpos[2]  # pole2 angle

        # Tip position (analytical from joint angles)
        var pole_len = Scalar[DTYPE](_POLE_LEN)
        var x_tip = (
            q0
            + pole_len * Scalar[DTYPE](sin(Float64(q1)))
            + pole_len * Scalar[DTYPE](sin(Float64(q1) + Float64(q2)))
        )
        var z_tip = pole_len * Scalar[DTYPE](
            cos(Float64(q1))
        ) + pole_len * Scalar[DTYPE](cos(Float64(q1) + Float64(q2)))

        var terminated = z_tip <= Scalar[DTYPE](Self.MIN_TIP_HEIGHT)

        var dist_penalty = Scalar[DTYPE](0.01) * x_tip * x_tip + (
            z_tip - Scalar[DTYPE](2.0)
        ) * (z_tip - Scalar[DTYPE](2.0))
        var v1 = data.qvel[1]
        var v2 = data.qvel[2]
        var vel_penalty = (
            Scalar[DTYPE](1e-3) * v1 * v1 + Scalar[DTYPE](5e-3) * v2 * v2
        )

        var alive_bonus = Scalar[DTYPE](0.0) if terminated else Scalar[DTYPE](
            10.0
        )
        var reward = alive_bonus - dist_penalty - vel_penalty

        return (reward, terminated)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(InvertedDoublePendulumModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.1

    # === GPU inline: Pre-step hook ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        meta_offset: Int,
    ):
        # Save cart x position (qpos[0]) into META_IDX_PREV_X
        comptime QPOS_OFF = qpos_offset[
            InvertedDoublePendulumModel.NQ, InvertedDoublePendulumModel.NV
        ]()
        states[env, meta_offset + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]

    # === GPU inline: Reward + termination ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
        MODEL_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
        qpos_off: Int,
        xpos_off: Int,
        xipos_off: Int,
        cfrc_ext_off: Int,
        cvel_off: Int,
        meta_offset: Int,
        curriculum_offset: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"

        var q0 = rebind[Scalar[DTYPE]](states[env, qpos_off + 0])  # cart x
        var q1 = rebind[Scalar[DTYPE]](states[env, qpos_off + 1])  # pole1 angle
        var q2 = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])  # pole2 angle

        # Tip position: analytical from joint angles
        var pole_len = Scalar[DTYPE](_POLE_LEN)
        var s1 = Scalar[DTYPE](sin(q1))
        var s12 = Scalar[DTYPE](sin(q1 + q2))
        var c1 = Scalar[DTYPE](cos(q1))
        var c12 = Scalar[DTYPE](cos(q1 + q2))

        var x_tip = q0 + pole_len * s1 + pole_len * s12
        var z_tip = pole_len * c1 + pole_len * c12

        var terminated = z_tip <= Scalar[DTYPE](1.0)

        var dist_penalty = Scalar[DTYPE](0.01) * x_tip * x_tip + (
            z_tip - Scalar[DTYPE](2.0)
        ) * (z_tip - Scalar[DTYPE](2.0))

        comptime QVEL_OFF = qvel_offset[
            InvertedDoublePendulumModel.NQ, InvertedDoublePendulumModel.NV
        ]()
        var v1 = rebind[Scalar[DTYPE]](states[env, QVEL_OFF + 1])
        var v2 = rebind[Scalar[DTYPE]](states[env, QVEL_OFF + 2])
        var vel_penalty = (
            Scalar[DTYPE](1e-3) * v1 * v1 + Scalar[DTYPE](5e-3) * v2 * v2
        )

        var alive_bonus = Scalar[DTYPE](0.0) if terminated else Scalar[DTYPE](
            10.0
        )
        var reward = alive_bonus - dist_penalty - vel_penalty

        return (reward, terminated)

    # === GPU inline: Custom obs extraction (9D with sin/cos encoding) ===
    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
        qpos_off: Int,
        qvel_off: Int,
        xpos_off: Int,
    ) -> Bool:
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        # OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2),
        #              clip(qvel[0],-10,10), clip(qvel[1],-10,10), clip(qvel[2],-10,10),
        #              0.0]  # qfrc_constraint[0] not in state buffer → 0
        var q0 = rebind[Scalar[DTYPE]](states[env, qpos_off + 0])
        var q1 = rebind[Scalar[DTYPE]](states[env, qpos_off + 1])
        var q2 = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])

        obs[env, 0] = q0
        obs[env, 1] = Scalar[DTYPE](sin(q1))
        obs[env, 2] = Scalar[DTYPE](sin(q2))
        obs[env, 3] = Scalar[DTYPE](cos(q1))
        obs[env, 4] = Scalar[DTYPE](cos(q2))

        comptime for i in range(3):
            var v = rebind[Scalar[DTYPE]](states[env, qvel_off + i])
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs[env, 5 + i] = v

        # qfrc_constraint[0] is not stored in the state buffer; use 0.0
        obs[env, 8] = Scalar[DTYPE](0.0)

        return True
