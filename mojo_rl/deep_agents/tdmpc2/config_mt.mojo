"""TD-MPC2 multi-task named preset (item C, §14.3) — Design-F facade.

Sugar over `TDMPC2MultiTaskAgent[...]`, mirroring `config.mojo` for the
single-task agent. Adds `MAX_OBS / MAX_ACT / NUM_TASKS / TASK_EMB` to the
config surface; the single-task `config.mojo` is untouched.

    var agent = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP
    ](ctx=ctx, lr=..., bce_coef=...)
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT

from .agent_mt import TDMPC2MultiTaskAgent


def TDMPC2MultiTask[
    target: StaticString,
    MAX_OBS: Int, MAX_ACT: Int, NUM_TASKS: Int, TASK_EMB: Int,
    B: Int, CAP: Int,
    ENC: Int = 256,
    LATENT: Int = 512,
    MLP: Int = 512,
    BINS: Int = 101,
    SN: Int = 8,
    VMIN: Int = -10,
    VMAX: Int = 10,
    H: Int = 3,
    QP: Float64 = 0.0,
    # MPPI planning budget — read only by `select_action_mpc` / the
    # `USE_MPC=True` drivers. Reference TD-MPC2 is 512/24/64/6.
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    NUM_ITERS: Int = 8 if MAX_ACT >= 20 else 6,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = Scalar[DT](3e-4),
    gamma: Scalar[DT] = Scalar[DT](0.99),
    tau: Scalar[DT] = Scalar[DT](0.01),
    action_scale: Scalar[DT] = Scalar[DT](1.0),
    learning_starts: Int = 1_000,
    enc_lr_scale: Scalar[DT] = Scalar[DT](0.3),
    temperature: Scalar[DT] = Scalar[DT](0.5),
    bce_coef: Scalar[DT] = Scalar[DT](0.0),
) raises -> TDMPC2MultiTaskAgent[
    target, MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
    NUM_TASKS, TASK_EMB, QP, NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
]:
    """Multi-task TD-MPC2: one task-conditioned world model + Q-ensemble over a
    set of envs (obs padded to MAX_OBS, actions to MAX_ACT, a learned per-task
    embedding concatenated into every net). Acting is MPC-off by default;
    `select_action_mpc` (and the `USE_MPC=True` drivers) plan with the
    task-conditioned MPPI callback on the GPU target. Dims default to the
    reference config.yaml; scalars are overridable."""
    return TDMPC2MultiTaskAgent[
        target, MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
        CAP, NUM_TASKS, TASK_EMB, QP, NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES,
        NUM_ITERS,
    ].make(
        lr=lr, gamma=gamma, tau=tau, action_scale=action_scale,
        learning_starts=learning_starts, enc_lr_scale=enc_lr_scale,
        temperature=temperature, bce_coef=bce_coef, ctx=ctx,
    )
