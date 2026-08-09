"""MuZeroBatchedAgent — facade over the batched device-replay self-play driver.

The batched, GPU-only sibling of `MuZeroAgent`. Where `MuZeroAgent.train` wires
the single-env, host-replay `run_muzero_gumbel_selfplay_gpu`, this facade wires
`run_muzero_gumbel_selfplay_gpu_batched_devreplay`: ``N_ENVS`` envs stepped in
parallel through a `BatchedGpuDiscreteEnv`, searched in one batched Gumbel
launch, with the obs ring resident on the GPU (`GPUMCTSSequenceReplay`).

Same shape as `MuZeroAgent`: construct with the optimizer/value config
(``lr`` / ``gamma`` / ``v_min`` / ``v_max`` / ``value_coef`` / ``max_grad_norm``),
then call ``train(env, ...)`` — session-local Adam optimizers are recreated each
call (the nets keep their weights), reproducing the convergence config (e.g.
clip 10).

``OBS_STORE_DT`` selects the device obs ring dtype: ``uint8`` (default) for pixel
obs (lossless ``k/255``), ``DT`` for vector/state obs (lossless rebind — pixel
quantization would corrupt physical state values). Device-ring constraints:
``CAP % N_ENVS == 0`` and ``CAP >= N_ENVS · max_ep_steps``.

    comptime Agent = MuZeroBatchedAgent[
        BatchedEnvT, Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        N_ENVS, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N, OBS_STORE_DT=DT,
    ]
    var agent = Agent(ctx=ctx, lr=3e-4, v_min=-20, v_max=20,
                      value_coef=1.0, max_grad_norm=10.0)
    var loss = agent.train[L=RemoteLogger](env, iterations=..., eval_env=...)
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.core.logger import Logger, NoOpLogger

from ..training.batched_env import BatchedEnv
from .selfplay_gpu_batched import run_muzero_gumbel_selfplay_gpu_batched_devreplay


struct MuZeroBatchedAgent[
    BENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    OBS_STORE_DT: DType = DType.uint8,
](Deinitable, Movable):
    var ctx: DeviceContext
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]
    var max_grad_norm: Scalar[DT]
    var weight_decay: Scalar[DT]

    def __init__(
        out self,
        ctx: DeviceContext,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        weight_decay: Scalar[DT] = Scalar[DT](0.0),
    ) raises:
        self.ctx = ctx
        self.rep = Self.REP.make["gpu", Kaiming](Optional(ctx))
        self.dyn = Self.DYN.make["gpu", Kaiming](Optional(ctx))
        self.pred = Self.PRED.make["gpu", Kaiming](Optional(ctx))
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        # Decoupled (AdamW-style) L2 weight decay, applied to `APPLY_DECAY`
        # weight Params only (biases/norms excluded). 0.0 = off. The muzero-general
        # Atari recipe uses 1e-4 as an overfitting guard for long training runs.
        self.weight_decay = weight_decay

    def train[
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: Self.BENV,
        iterations: Int,
        learning_starts: Int = 256,
        train_per_iter: Int = Self.N_ENVS,   # default UTD 1:1
        seed: UInt64 = 0,
        max_ep_steps: Int = 27000,
        temperature_decay_steps: Int = 0,
        reanalyze_every: Int = 0,
        reanalyze_batch: Int = Self.N_ENVS,
        target_sync_interval: Int = 0,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        eval_horizon: Int = 0,
        eval_env: Optional[Pointer[Self.BENV, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
        use_per: Bool = False,
        per_alpha: Scalar[DT] = Scalar[DT](1.0),
        per_beta: Scalar[DT] = Scalar[DT](1.0),
        lr_decay_rate: Scalar[DT] = Scalar[DT](1.0),
        lr_decay_steps: Int = 0,
    ) raises -> Float64:
        """Batched device-replay self-play. ``learning_starts`` is in stored
        steps; each iteration advances ``N_ENVS`` env steps and runs
        ``train_per_iter`` gradient steps. ``train_per_iter`` defaults to
        ``N_ENVS`` → **UTD 1:1** (one gradient step per env step, matching the
        single-env driver's sample efficiency); set it to 1 for UTD 1:N_ENVS
        (fastest wall-clock, ~N_ENVS× more env steps to converge).
        ``reanalyze_batch`` (when ``reanalyze_every > 0``) is how many stored
        positions get fresh-net targets per trigger — default ``N_ENVS`` (low
        coverage); set ≈ ``B`` so each training batch is mostly fresh targets
        (the EfficientZero coverage lever for sample efficiency).
        ``target_sync_interval`` gates target-net reanalyze: 0 (default) searches
        with the live nets (bit-identical to before); set > 0 to reanalyze
        through lagging copies refreshed every that-many grad steps — the
        standard target-net stabiliser, worth pairing with high
        ``reanalyze_batch`` (matches EZv2 / official MuZero's delayed reanalyze
        model).

        ``lr_decay_rate`` / ``lr_decay_steps`` apply muzero-general's exponential
        LR schedule ``lr = lr_init · lr_decay_rate^(grad_step / lr_decay_steps)``
        across all three optimizers (reference: rate 0.1 over ~⅓ of the grad-step
        budget). Defaults (rate 1.0 / steps 0) = constant LR, bit-identical to
        before. Returns the last training loss; the nets keep their weights
        across ``train`` calls."""
        var orep = Adam(lr=self.lr, wd=self.weight_decay)
        var odyn = Adam(lr=self.lr, wd=self.weight_decay)
        var opred = Adam(lr=self.lr, wd=self.weight_decay)
        # NOTE: legacy Adam.max_grad_norm clipping is dropped in the storage path
        # (storage Adam has no such field; the migrated unroll does not clip).
        # Re-add via clip_grad_norm in mz_unroll_train_step_* if convergence needs
        # it (matches the AlphaZero storage driver pattern).
        return run_muzero_gumbel_selfplay_gpu_batched_devreplay[
            Self.BENV, Self.REP, Self.DYN, Self.PRED,
            Self.N_ENVS, Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
            Self.NUM_SIMS, Self.MAX_NODES, Self.MAX_K,
            Self.CAP, Self.B, Self.K, Self.N,
            OBS_STORE_DT = Self.OBS_STORE_DT,
            L=L,
        ](
            self.ctx, env, self.rep, self.dyn, self.pred, orep, odyn, opred,
            iterations=iterations,
            learning_starts=learning_starts,
            train_per_iter=train_per_iter,
            gamma=self.gamma,
            v_min=self.v_min,
            v_max=self.v_max,
            seed=seed,
            max_ep_steps=max_ep_steps,
            value_coef=self.value_coef,
            temperature_decay_steps=temperature_decay_steps,
            reanalyze_every=reanalyze_every,
            reanalyze_batch=reanalyze_batch,
            target_sync_interval=target_sync_interval,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_horizon=eval_horizon,
            eval_env=eval_env,
            diag_every=diag_every,
            report_every=report_every,
            logger=logger,
            verbose=verbose,
            use_per=use_per,
            per_alpha=per_alpha,
            per_beta=per_beta,
            lr_decay_rate=lr_decay_rate,
            lr_decay_steps=lr_decay_steps,
        )
