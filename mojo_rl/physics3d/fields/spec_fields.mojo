"""The runtime replacement for `_acd` (phases 1a.2 / 1a.4).

`ModelDefFromXML._acd` (`ComptimeActData`) is a comptime struct of ~20
`InlineArray`s that the model's XML is interpreted into at struct-elaboration
time. Every actuator value the engine uses arrives through it: `apply_actions`
materializes fourteen of those arrays per call, and `apply_actions_kernel_gpu`
bakes them into the kernel as literals via a comptime-unrolled loop.

`SpecFields` is the same data as packed record tensors, filled by
`fields_build.build_spec_fields` from the runtime `FlatModelDef`:

    actuators    [NACT, MODEL_ACTUATOR_SIZE]      gains, ranges, transmission
    act_tendons  [NTEN, MODEL_ACT_TENDON_SIZE]    fixed-tendon springs
    qpos0        [NQ]                             the reference pose
    pose_meta    [POSE_META_SIZE]                 qpos0_nq, free_joint_qpos_adr
    key_meta     [NKEY, KEY_META_SIZE]            time + the three lengths
    key_qpos / key_qvel / key_ctrl                one row per <key>

Column indices are `ACT_IDX_*` / `ACTTEN_IDX_*` / `POSE_IDX_*` / `KEY_IDX_*` in
`gpu/constants.mojo`, beside every other record layout.

The pose half is not actuation and is here anyway: this struct exists to be
`_acd`'s runtime replacement, and `_acd` carried both. Splitting them would
mean two builds, two parses and two arguments at every consumer.

⚠ WHY NOT A FAMILY ON `fields.Model`. `Model` is the operand bundle the
integrator, solver and collision kernels bind. Actuation is read by exactly one
function per target and by nothing else, so it is a different bundle. It is
also a practical matter: `Model` is named with an explicit parameter list in 48
files, and adding a fifteenth parameter would have meant threading `NACT`
through every one of them to keep them compiling — churn with no bearing on the
change.

⚠ THE DIMS ARE STILL COMPTIME HERE, and that is not an oversight. Phase 1a
moves the DATA off the comptime interpreter; the DIMS move in 1b. Each is
floored at 1 (`*_F`) so the tensors are always bindable — a zero-extent operand
aborts at bind (same reason `fields.Model._at_least_one` exists).
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl

from ..gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    ACT_IDX_KP,
    ACT_IDX_ACT_ADR,
    ACT_IDX_TENDON_ID,
    ACT_IDX_JOINT_ID,
    ACT_IDX_TRN_QADR_0,
    ACT_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_QADR_0,
    ACTTEN_IDX_TRN_DADR_0,
    TENDON_MAX_WRAPS,
    POSE_META_SIZE,
    POSE_IDX_FREE_JOINT_QPOS_ADR,
    KEY_META_SIZE,
    JLIM_SIZE,
)


struct SpecFields[
    DTYPE: DType,
    NACT: Int,
    NTEN: Int,
    # ⚠ APPENDED, AND WITHOUT DEFAULTS ON PURPOSE. A defaulted `NQ = 0` would
    # give a caller who forgot it a zero-length `qpos0` that reads as an
    # all-zero reference pose — a legal-looking model that resets to the wrong
    # place, with nothing to fail. Every construction goes through
    # `ModelDefFromXML.make_spec_fields` / `init_spec_fields`, which supply
    # them, so there is no site that WANTS a default.
    NQ: Int,
    NV: Int,
    NKEY: Int,
    NJOINT: Int,
](Movable):
    """The runtime twin of `_acd`: actuation records + the reference pose."""

    comptime NACT_F: Int = Self.NACT if Self.NACT > 0 else 1
    comptime NTEN_F: Int = Self.NTEN if Self.NTEN > 0 else 1
    comptime NQ_F: Int = Self.NQ if Self.NQ > 0 else 1
    comptime NV_F: Int = Self.NV if Self.NV > 0 else 1
    comptime NKEY_F: Int = Self.NKEY if Self.NKEY > 0 else 1
    comptime NJOINT_F: Int = Self.NJOINT if Self.NJOINT > 0 else 1

    var actuators: TensorImpl[Self.DTYPE]  # [NACT, MODEL_ACTUATOR_SIZE]
    var act_tendons: TensorImpl[Self.DTYPE]  # [NTEN, MODEL_ACT_TENDON_SIZE]
    # Reference pose (`mj_resetData`) and `<keyframe>`
    # (`mj_resetDataKeyframe`). ⚠ THESE ARE NOT ACTUATION, and they live here
    # because they are the rest of what `_acd` carried — this struct's job is
    # to be its runtime replacement, not to be a taxonomy. `qpos0` is indexed
    # by qpos ADDRESS, so a free joint occupies 7 of its slots and a `<custom>
    # <numeric name="init_qpos">` overrides the lot; that last case is why it
    # cannot be folded into `Model.joints[JOINT_IDX_QPOS0]`, which is one
    # scalar per JOINT.
    var qpos0: TensorImpl[Self.DTYPE]  # [NQ]
    var pose_meta: TensorImpl[Self.DTYPE]  # [POSE_META_SIZE]
    var key_meta: TensorImpl[Self.DTYPE]  # [NKEY, KEY_META_SIZE]
    var key_qpos: TensorImpl[Self.DTYPE]  # [NKEY, NQ]
    var key_qvel: TensorImpl[Self.DTYPE]  # [NKEY, NV]
    var key_ctrl: TensorImpl[Self.DTYPE]  # [NKEY, NACT]
    # The `enforce_limits` clamp — see `JLIM_*`. Separate from
    # `Model.joints`' limit columns because that record has no LIMITED flag.
    var joint_limits: TensorImpl[Self.DTYPE]  # [NJOINT, JLIM_SIZE]

    def __init__(out self) raises:
        self.actuators = TensorImpl[Self.DTYPE].alloc(
            Self.NACT_F * MODEL_ACTUATOR_SIZE
        )
        self.act_tendons = TensorImpl[Self.DTYPE].alloc(
            Self.NTEN_F * MODEL_ACT_TENDON_SIZE
        )
        self.qpos0 = TensorImpl[Self.DTYPE].alloc(Self.NQ_F)
        self.pose_meta = TensorImpl[Self.DTYPE].alloc(POSE_META_SIZE)
        # -1 = no free joint. Zero is a VALID qpos address (quadruped's free
        # joint IS at 0), so an unwritten slot would put an identity
        # quaternion into qpos[3] of a model that has no free joint at all.
        self.pose_meta.data[POSE_IDX_FREE_JOINT_QPOS_ADR] = Scalar[
            Self.DTYPE
        ](-1)
        self.key_meta = TensorImpl[Self.DTYPE].alloc(
            Self.NKEY_F * KEY_META_SIZE
        )
        self.key_qpos = TensorImpl[Self.DTYPE].alloc(Self.NKEY_F * Self.NQ_F)
        self.key_qvel = TensorImpl[Self.DTYPE].alloc(Self.NKEY_F * Self.NV_F)
        self.key_ctrl = TensorImpl[Self.DTYPE].alloc(
            Self.NKEY_F * Self.NACT_F
        )
        self.joint_limits = TensorImpl[Self.DTYPE].alloc(
            Self.NJOINT_F * JLIM_SIZE
        )
        # ⚠⚠ THE SENTINELS ARE -1 AND THE GAIN IS 1, AND THE ZERO `alloc`
        # LEAVES IS WRONG FOR BOTH. `TensorImpl.alloc` DOES zero-fill, so
        # unwritten rows are genuinely 0 — which is exactly the problem:
        #   * a zero `act_adr` / `trn_qadr` / `trn_dadr` is a VALID INDEX, so
        #     an unfilled slot would silently read qpos[0] / drive dof 0 /
        #     borrow another actuator's activation rather than be skipped;
        #   * a zero `kp` is MuJoCo's `gainprm[0]` default of 1 misread, and
        #     `force = kp * u` for EVERY kind — so it silently zeroes the
        #     force of every bare `<motor>`. That exact default cost a fix in
        #     1a.1 (`8fa068b5`) when it was 0.0 on `ActuatorData`.
        # The columns where zero is a legitimate "absent" — `stiffness`,
        # `trn_n` — are left as `alloc` wrote them and skipped by the readers,
        # which is what makes iterating the padded `NTEN_F` rows safe.
        # Only the slots the fill does not always write are seeded; the rest
        # are overwritten unconditionally by `build_spec_fields`.
        for i in range(Self.NACT_F):
            var o = i * MODEL_ACTUATOR_SIZE
            self.actuators.data[o + ACT_IDX_KP] = Scalar[Self.DTYPE](1)
            self.actuators.data[o + ACT_IDX_ACT_ADR] = Scalar[Self.DTYPE](-1)
            self.actuators.data[o + ACT_IDX_TENDON_ID] = Scalar[Self.DTYPE](-1)
            self.actuators.data[o + ACT_IDX_JOINT_ID] = Scalar[Self.DTYPE](-1)
            for k in range(TENDON_MAX_WRAPS):
                self.actuators.data[o + ACT_IDX_TRN_QADR_0 + k] = Scalar[
                    Self.DTYPE
                ](-1)
                self.actuators.data[o + ACT_IDX_TRN_DADR_0 + k] = Scalar[
                    Self.DTYPE
                ](-1)
        for t in range(Self.NTEN_F):
            var o = t * MODEL_ACT_TENDON_SIZE
            for k in range(TENDON_MAX_WRAPS):
                self.act_tendons.data[o + ACTTEN_IDX_TRN_QADR_0 + k] = Scalar[
                    Self.DTYPE
                ](-1)
                self.act_tendons.data[o + ACTTEN_IDX_TRN_DADR_0 + k] = Scalar[
                    Self.DTYPE
                ](-1)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Host -> device. Static config: called once at model build, like
        `Model.upload_all`."""
        self.actuators.upload(ctx)
        self.act_tendons.upload(ctx)
        # ⚠ The pose/keyframe tensors are CPU-ONLY consumers today
        # (`reset_data`, `key_*_at`) and are uploaded anyway so a future GPU
        # reset hook cannot read an empty device buffer. They are a few
        # hundred bytes.
        self.qpos0.upload(ctx)
        self.pose_meta.upload(ctx)
        self.key_meta.upload(ctx)
        self.key_qpos.upload(ctx)
        self.key_qvel.upload(ctx)
        self.key_ctrl.upload(ctx)
        self.joint_limits.upload(ctx)


# =============================================================================
# Column accessors
# =============================================================================
#
# One record column as a `List[Float64]`. The records are packed and strided,
# so a caller that wants "every actuator's kp" would otherwise write the
# `i * MODEL_ACTUATOR_SIZE + ACT_IDX_KP` arithmetic itself at every site — and
# these exist mostly for the MuJoCo gates, which compare a whole column
# against `mjModel` and used to read a flat `ComptimeActData` array.
#
# ⚠ `Float64` REGARDLESS OF `DTYPE`, deliberately: the caller is comparing
# against MuJoCo's own float64 values, and widening at the read keeps the
# comparison honest about what the record actually holds.


def actuator_column[
    DT: DType, NA: Int, NT: Int, NQ: Int, NV: Int, NK: Int, NJ: Int
](
    sf: SpecFields[DT, NA, NT, NQ, NV, NK, NJ], col: Int, n: Int
) raises -> List[Float64]:
    """`sf.actuators[:n, col]`. `col` is an `ACT_IDX_*`."""
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(sf.actuators.data[i * MODEL_ACTUATOR_SIZE + col]))
    return out^


def act_tendon_column[
    DT: DType, NA: Int, NT: Int, NQ: Int, NV: Int, NK: Int, NJ: Int
](
    sf: SpecFields[DT, NA, NT, NQ, NV, NK, NJ], col: Int, n: Int
) raises -> List[Float64]:
    """`sf.act_tendons[:n, col]`. `col` is an `ACTTEN_IDX_*`."""
    var out = List[Float64](capacity=n)
    for t in range(n):
        out.append(
            Float64(sf.act_tendons.data[t * MODEL_ACT_TENDON_SIZE + col])
        )
    return out^


def joint_limit_column[
    DT: DType, NA: Int, NT: Int, NQ: Int, NV: Int, NK: Int, NJ: Int
](
    sf: SpecFields[DT, NA, NT, NQ, NV, NK, NJ], col: Int, n: Int
) raises -> List[Float64]:
    """`sf.joint_limits[:n, col]`. `col` is a `JLIM_IDX_*`."""
    var out = List[Float64](capacity=n)
    for j in range(n):
        out.append(Float64(sf.joint_limits.data[j * JLIM_SIZE + col]))
    return out^
