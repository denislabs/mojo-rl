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

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
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
](ImplicitlyDestructible, Movable):
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

    def __init__(
        out self,
        ctx: DeviceContext,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises:
        self.ctx = ctx
        self.rep = Self.REP.make["gpu", INIT=Kaiming](ctx=ctx)
        self.dyn = Self.DYN.make["gpu", INIT=Kaiming](ctx=ctx)
        self.pred = Self.PRED.make["gpu", INIT=Kaiming](ctx=ctx)
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

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
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        eval_horizon: Int = 0,
        eval_env: Optional[UnsafePointer[Self.BENV, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
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
        (the EfficientZero coverage lever for sample efficiency). Returns the
        last training loss; the nets keep their weights across ``train`` calls."""
        var orep = Adam.make["gpu", M = Self.REP](self.rep, self.ctx)
        var odyn = Adam.make["gpu", M = Self.DYN](self.dyn, self.ctx)
        var opred = Adam.make["gpu", M = Self.PRED](self.pred, self.ctx)
        orep.lr = self.lr; odyn.lr = self.lr; opred.lr = self.lr
        orep.max_grad_norm = self.max_grad_norm
        odyn.max_grad_norm = self.max_grad_norm
        opred.max_grad_norm = self.max_grad_norm
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
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_horizon=eval_horizon,
            eval_env=eval_env,
            diag_every=diag_every,
            report_every=report_every,
            logger=logger,
            verbose=verbose,
        )
