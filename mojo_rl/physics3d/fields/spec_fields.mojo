"""Actuation record tensors — the runtime replacement for `_acd` (phase 1a.2).

`ModelDefFromXML._acd` (`ComptimeActData`) is a comptime struct of ~20
`InlineArray`s that the model's XML is interpreted into at struct-elaboration
time. Every actuator value the engine uses arrives through it: `apply_actions`
materializes fourteen of those arrays per call, and `apply_actions_kernel_gpu`
bakes them into the kernel as literals via a comptime-unrolled loop.

`SpecFields` is the same data as packed record tensors, filled by
`fields_build.build_spec_fields` from the runtime `FlatModelDef`. Two tensors:

    actuators    [NACT, MODEL_ACTUATOR_SIZE]      gains, ranges, transmission
    act_tendons  [NTEN, MODEL_ACT_TENDON_SIZE]    fixed-tendon springs

Column indices are `ACT_IDX_*` / `ACTTEN_IDX_*` in `gpu/constants.mojo`, beside
every other record layout.

⚠ WHY NOT A FAMILY ON `fields.Model`. `Model` is the operand bundle the
integrator, solver and collision kernels bind. Actuation is read by exactly one
function per target and by nothing else, so it is a different bundle. It is
also a practical matter: `Model` is named with an explicit parameter list in 48
files, and adding a fifteenth parameter would have meant threading `NACT`
through every one of them to keep them compiling — churn with no bearing on the
change.

⚠ `NACT`/`NTEN` ARE STILL COMPTIME HERE, and that is not an oversight. Phase
1a.2 moves the DATA off the comptime interpreter; the DIMS move in 1b. Both are
floored at 1 so the tensors are always bindable — a zero-extent operand aborts
at bind (same reason `fields.Model._at_least_one` exists).
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
)


struct SpecFields[DTYPE: DType, NACT: Int, NTEN: Int](Movable):
    """Actuator + fixed-tendon-spring records as two packed tensors."""

    comptime NACT_F: Int = Self.NACT if Self.NACT > 0 else 1
    comptime NTEN_F: Int = Self.NTEN if Self.NTEN > 0 else 1

    var actuators: TensorImpl[Self.DTYPE]  # [NACT, MODEL_ACTUATOR_SIZE]
    var act_tendons: TensorImpl[Self.DTYPE]  # [NTEN, MODEL_ACT_TENDON_SIZE]

    def __init__(out self) raises:
        self.actuators = TensorImpl[Self.DTYPE].alloc(
            Self.NACT_F * MODEL_ACTUATOR_SIZE
        )
        self.act_tendons = TensorImpl[Self.DTYPE].alloc(
            Self.NTEN_F * MODEL_ACT_TENDON_SIZE
        )
        # ⚠⚠ THE SENTINELS ARE -1 AND THE GAIN IS 1, AND ZERO IS WRONG FOR
        # BOTH. `alloc` does not promise zeroed memory, and even if it did:
        #   * a zero `act_adr` / `trn_qadr` / `trn_dadr` is a VALID INDEX, so
        #     an unfilled slot would silently read qpos[0] / drive dof 0 /
        #     borrow another actuator's activation rather than be skipped;
        #   * a zero `kp` is MuJoCo's `gainprm[0]` default of 1 misread, and
        #     `force = kp * u` for EVERY kind — so it silently zeroes the
        #     force of every bare `<motor>`. That exact default cost a fix in
        #     1a.1 (`8fa068b5`) when it was 0.0 on `ActuatorData`.
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
