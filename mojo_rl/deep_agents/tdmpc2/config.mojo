"""TD-MPC2 named preset — config descriptor + factory (Design F).

Additive sugar over the primitive `TDMPC2Agent[target, OBS, ENC, ACT,
LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP, NUM_SAMPLES, NUM_PI_TRAJS,
NUM_ELITES, NUM_ITERS]`. The primitive stays the source of truth for
arbitrary dims; this module names the canonical TD-MPC2 setup (reference
config.yaml dims: latent 512, mlp 512, enc 256, num_bins 101, simnorm 8,
horizon 3, MPPI 512/24/64/6) and bundles its tuned scalar defaults.

Same shape as `sac/config.mojo`:

  1. `TDMPC2ConfigT` — a trait bundling the FULL compile-time identity of
     the algorithm: the deployment `TARGET`, every architecture dimension,
     the MPPI planning budget, plus tuned scalar defaults (`DEF_*`).
     TD-MPC2's nets are derived inside the agent from the dims (encoder,
     dynamics, reward, Q-ensemble, policy), so — unlike SAC — the config
     carries DIMS rather than `Module` types.

  2. `TDMPC2Config` — a zero-field conformer struct parametrized by
     `target` + dims. ONE config covers both CPU and GPU (the agent's
     blocks are target-generic).

  3. `agent_from_config` + the capitalized preset `TDMPC2`. The preset is
     a SINGLE function taking `target` as a parameter and reads like a
     constructor at the call site:

         var agent = TDMPC2["cpu", OBS, ACT, B, CAP]()
         var agent = TDMPC2["gpu", OBS, ACT, B, CAP](ctx=ctx)

Acting defaults to MPC-off (`a = π(encode(obs))`) via `select_action`;
`select_action_mpc` (GPU only) plans with the MPPI budget the config
carries. Checkpoints written from a `TDMPC2[...]` agent on one target
load on any other `TDMPC2[...]` agent of matching dims.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT

from .agent import TDMPC2Agent


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait TDMPC2ConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a TD-MPC2-family algorithm. Conformers
    are zero-field comptime tags — never instantiated at runtime; only
    their comptime members are read."""

    comptime TARGET: StaticString
    comptime OBS: Int
    comptime ENC: Int
    comptime ACT: Int
    comptime LATENT: Int
    comptime MLP: Int
    comptime BINS: Int
    comptime SN: Int
    comptime VMIN: Int
    comptime VMAX: Int
    comptime B: Int
    comptime H: Int
    comptime CAP: Int
    # MPPI planning budget (used only by select_action_mpc on GPU).
    comptime NUM_SAMPLES: Int
    comptime NUM_PI_TRAJS: Int
    comptime NUM_ELITES: Int
    comptime NUM_ITERS: Int
    # Q-trunk dropout prob (item D). 0.0 = always-on no-op (bit-identical).
    comptime QP: Float64

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_ENC_LR_SCALE: Scalar[DT]
    comptime DEF_TEMPERATURE: Scalar[DT]
    # Termination BCE coefficient (item B): 0 = non-episodic (default,
    # bit-identical); >0 trains the termination head for episodic envs.
    comptime DEF_BCE_COEF: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer — one struct, parametrized by `target` + dims.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct TDMPC2Config[
    target: StaticString,
    obs: Int, act: Int, b: Int, cap: Int,
    enc: Int = 256,
    latent: Int = 512,
    mlp: Int = 512,
    bins: Int = 101,
    sn: Int = 8,
    vmin: Int = -10,
    vmax: Int = 10,
    h: Int = 3,
    num_samples: Int = 512,
    num_pi_trajs: Int = 24,
    num_elites: Int = 64,
    # Reference `tdmpc2.py:35` bumps MPPI iterations by 2 for high-dim action
    # spaces (harder planning). HalfCheetah (act=6) keeps 6; Humanoid (act>=20)
    # auto-gets 8 without the caller thinking about it.
    num_iters: Int = 8 if act >= 20 else 6,
    qp: Float64 = 0.0,
](TDMPC2ConfigT):
    """TD-MPC2 (Hansen et al. 2024) — implicit world model (encoder +
    latent dynamics + reward + Q-ensemble), two-hot distributional
    value/reward, MPPI planning. Architecture dims default to the
    reference config.yaml; only obs/act/b/cap are mandatory. One config
    covers cpu + gpu via the agent's target-generic blocks.

    Struct params are lowercase; the uppercase trait members alias them
    (so they fold to the raw params for return-type unification, the same
    pattern SAC uses for `target`→`TARGET`)."""

    comptime TARGET = Self.target
    comptime OBS = Self.obs
    comptime ENC = Self.enc
    comptime ACT = Self.act
    comptime LATENT = Self.latent
    comptime MLP = Self.mlp
    comptime BINS = Self.bins
    comptime SN = Self.sn
    comptime VMIN = Self.vmin
    comptime VMAX = Self.vmax
    comptime B = Self.b
    comptime H = Self.h
    comptime CAP = Self.cap
    comptime NUM_SAMPLES = Self.num_samples
    comptime NUM_PI_TRAJS = Self.num_pi_trajs
    comptime NUM_ELITES = Self.num_elites
    comptime NUM_ITERS = Self.num_iters
    comptime QP = Self.qp

    comptime DEF_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.01)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_ENC_LR_SCALE = Scalar[DT](0.3)
    comptime DEF_TEMPERATURE = Scalar[DT](0.5)
    comptime DEF_BCE_COEF = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any TDMPC2ConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: TDMPC2ConfigT,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = CONFIG.DEF_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    enc_lr_scale: Scalar[DT] = CONFIG.DEF_ENC_LR_SCALE,
    temperature: Scalar[DT] = CONFIG.DEF_TEMPERATURE,
    bce_coef: Scalar[DT] = CONFIG.DEF_BCE_COEF,
) raises -> TDMPC2Agent[
    CONFIG.TARGET,
    CONFIG.OBS, CONFIG.ENC, CONFIG.ACT, CONFIG.LATENT, CONFIG.MLP,
    CONFIG.BINS, CONFIG.SN, CONFIG.VMIN, CONFIG.VMAX, CONFIG.B, CONFIG.H,
    CONFIG.CAP, CONFIG.NUM_SAMPLES, CONFIG.NUM_PI_TRAJS, CONFIG.NUM_ELITES,
    CONFIG.NUM_ITERS, CONFIG.QP,
]:
    """Build the primitive `TDMPC2Agent` from any `TDMPC2ConfigT`. Every
    scalar defaults to the config's tuned value but stays overridable. The
    deployment target, dims, and planning budget are read off the config,
    so this one function serves cpu and gpu."""
    return TDMPC2Agent[
        CONFIG.TARGET,
        CONFIG.OBS, CONFIG.ENC, CONFIG.ACT, CONFIG.LATENT, CONFIG.MLP,
        CONFIG.BINS, CONFIG.SN, CONFIG.VMIN, CONFIG.VMAX, CONFIG.B, CONFIG.H,
        CONFIG.CAP, CONFIG.NUM_SAMPLES, CONFIG.NUM_PI_TRAJS,
        CONFIG.NUM_ELITES, CONFIG.NUM_ITERS, CONFIG.QP,
    ].make(
        lr=lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        learning_starts=learning_starts,
        enc_lr_scale=enc_lr_scale,
        temperature=temperature,
        bce_coef=bce_coef,
        ctx=ctx,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# Reads like a constructor. Full tuning surface, defaults from the config.
# ──────────────────────────────────────────────────────────────────────


def TDMPC2[
    target: StaticString,
    OBS: Int, ACT: Int, B: Int, CAP: Int,
    ENC: Int = 256,
    LATENT: Int = 512,
    MLP: Int = 512,
    BINS: Int = 101,
    SN: Int = 8,
    VMIN: Int = -10,
    VMAX: Int = 10,
    H: Int = 3,
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    # Reference `tdmpc2.py:35`: +2 MPPI iterations for high-dim action spaces.
    # act<20 (HalfCheetah) → 6; act>=20 (Humanoid) → 8, applied automatically.
    NUM_ITERS: Int = 8 if ACT >= 20 else 6,
    QP: Float64 = 0.0,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_LR,
    gamma: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_GAMMA,
    tau: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_TAU,
    action_scale: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_ACTION_SCALE,
    learning_starts: Int = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_LEARNING_STARTS,
    enc_lr_scale: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_ENC_LR_SCALE,
    temperature: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_TEMPERATURE,
    bce_coef: Scalar[DT] = TDMPC2Config[
        target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS,
    ].DEF_BCE_COEF,
) raises -> TDMPC2Agent[
    target, OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
    NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS, QP,
]:
    """TD-MPC2 with the canonical implicit world model + Q-ensemble + MPPI
    planning. `target` selects cpu/gpu; architecture dims default to the
    reference config.yaml (latent 512 / mlp 512 / enc 256 / bins 101 /
    simnorm 8 / horizon 3 / MPPI 512·24·64·6); all scalars default to the
    tuned config value but stay overridable. Acting is MPC-off by default
    (`select_action`); `select_action_mpc` plans on GPU."""
    return agent_from_config[
        TDMPC2Config[
            target, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX,
            H, NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES, NUM_ITERS, QP,
        ]
    ](
        ctx=ctx,
        lr=lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        learning_starts=learning_starts,
        enc_lr_scale=enc_lr_scale,
        temperature=temperature,
        bce_coef=bce_coef,
    )
