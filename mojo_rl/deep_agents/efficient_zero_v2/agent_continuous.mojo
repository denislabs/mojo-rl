"""EZv2ContinuousAgent — facade over the continuous self-play driver + greedy
eval + checkpointing (GPU).

The continuous sibling of `EZv2DiscreteAgent`. Where the discrete agent runs on
CPU (decision D1: discrete CPU uses vanilla PUCT MCTS), the continuous agent is
**GPU-only** — the `SampledGumbelGPUMCTS` planner (Gumbel-Top-k over sampled
continuous action vectors) exists only on the device. It carries the same five
nets as the discrete agent (MuZero ``rep`` / ``dyn`` + the squashed-Gaussian
``pred`` head + SimSiam ``proj`` / ``predh``) and trains with
`ezv2_unroll_train_step_continuous_gpu` (MuZero BPTT + consistency + Gaussian
policy NLL).

  * `__init__(ctx, lr, gamma, v_min, v_max, ...policy hyperparams...)` — build
    fresh Kaiming nets on the device. The squashed-Gaussian parameters
    (``max_action`` / ``min_std`` / ``std_magnification`` / ``soft_clamp`` /
    ``init_std`` / ``ent_scale``) are stored once and fed to **both** the planner
    sampler and the training loss so the policy parameterization matches.
  * `train(env, iterations, ...)` — GPU sampled-Gumbel self-play; optimizers are
    session-local (recreated per call), nets persist across calls.
  * `eval_greedy(env, episodes)` — the deployed-agent metric: deterministic
    (argmax-visit) sampled-Gumbel rollout. The projector/predictor are
    consistency-only, unused here.
  * `save` / `load` — one-file `nn-ckpt v2` envelope packing all five nets
    (sections `rep` / `dyn` / `pred` / `proj` / `predh`), GPU body visitors.
"""

from std.memory import alloc
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module, mptr
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.checkpoint import (
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
)
from mojo_rl.deep_agents.core.checkpoint_helpers import (
    split_lines_v2,
    read_file_v2,
    expect_v2_header,
)
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from mojo_rl.planners.tree_search import SampledGumbelGPUMCTS, SinglePlayer

from mojo_rl.core.logger import Logger, NoOpLogger
from .selfplay_gpu_continuous import run_ezv2_sampled_selfplay_gpu
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZContPredGPU


@fieldwise_init
struct EZv2ContinuousAgent[
    ENV: BoxContinuousActionEnv & ImplicitlyDeletable,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
](ImplicitlyDeletable, Movable):
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var proj: Self.PROJM
    var predh: Self.PREDH
    var ctx: DeviceContext
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]
    var consistency_coef: Scalar[DT]
    var policy_coef: Scalar[DT]
    var max_action: Scalar[DT]
    var min_std: Scalar[DT]
    var std_magnification: Scalar[DT]
    var soft_clamp: Scalar[DT]
    var init_std: Scalar[DT]
    var ent_scale: Scalar[DT]
    var c_visit: Scalar[DT]
    var c_scale: Scalar[DT]

    def __init__(
        out self,
        ctx: DeviceContext,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        v_min: Scalar[DT] = Scalar[DT](-50.0),
        v_max: Scalar[DT] = Scalar[DT](2.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        consistency_coef: Scalar[DT] = Scalar[DT](2.0),
        policy_coef: Scalar[DT] = Scalar[DT](1.0),
        max_action: Scalar[DT] = Scalar[DT](2.0),
        min_std: Scalar[DT] = Scalar[DT](0.5),
        std_magnification: Scalar[DT] = Scalar[DT](3.0),
        soft_clamp: Scalar[DT] = Scalar[DT](5.0),
        init_std: Scalar[DT] = Scalar[DT](1.0),
        ent_scale: Scalar[DT] = Scalar[DT](0.05),
        c_visit: Scalar[DT] = Scalar[DT](50.0),
        c_scale: Scalar[DT] = Scalar[DT](0.1),
    ) raises:
        self.ctx = ctx
        self.rep = Self.REP.make["gpu", INIT=Kaiming](ctx)
        self.dyn = Self.DYN.make["gpu", INIT=Kaiming](ctx)
        self.pred = Self.PRED.make["gpu", INIT=Kaiming](ctx)
        self.proj = Self.PROJM.make["gpu", INIT=Kaiming](ctx)
        self.predh = Self.PREDH.make["gpu", INIT=Kaiming](ctx)
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        self.consistency_coef = consistency_coef
        self.policy_coef = policy_coef
        self.max_action = max_action
        self.min_std = min_std
        self.std_magnification = std_magnification
        self.soft_clamp = soft_clamp
        self.init_std = init_std
        self.ent_scale = ent_scale
        self.c_visit = c_visit
        self.c_scale = c_scale

    def train[
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: Self.ENV,
        iterations: Int,
        learning_starts: Int = 2000,
        train_per_iter: Int = 1,
        seed: UInt64 = 0,
        max_ep_steps: Int = 200,
        target_sync_interval: Int = 200,
        reanalyze_interval: Int = 1,
        reanalyze_warmup: Int = 500,
        reanalyze_batch: Int = 4,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
    ) raises -> Float64:
        """GPU sampled-Gumbel self-play training (MuZero BPTT + SimSiam
        consistency + squashed-Gaussian policy NLL). Returns the last training
        loss. Optimizers recreated here; nets persist."""
        var orep = Adam.make["gpu", M = Self.REP](self.rep, self.ctx)
        var odyn = Adam.make["gpu", M = Self.DYN](self.dyn, self.ctx)
        var opred = Adam.make["gpu", M = Self.PRED](self.pred, self.ctx)
        var oproj = Adam.make["gpu", M = Self.PROJM](self.proj, self.ctx)
        var opredh = Adam.make["gpu", M = Self.PREDH](self.predh, self.ctx)
        orep.lr = self.lr
        odyn.lr = self.lr
        opred.lr = self.lr
        oproj.lr = self.lr
        opredh.lr = self.lr
        return run_ezv2_sampled_selfplay_gpu[
            Self.ENV, Self.REP, Self.DYN, Self.PRED, Self.PROJM, Self.PREDH,
            Self.OBS, Self.ACT_DIM, Self.LATENT, Self.BINS,
            Self.NUM_SIMS, Self.MAX_NODES, Self.K_ROOT, Self.K_NON_ROOT,
            Self.CAP, Self.B, Self.K, Self.N,
            L=L,
        ](
            self.ctx, env,
            self.rep, self.dyn, self.pred, self.proj, self.predh,
            orep, odyn, opred, oproj, opredh,
            iterations=iterations,
            learning_starts=learning_starts,
            train_per_iter=train_per_iter,
            gamma=self.gamma,
            v_min=self.v_min,
            v_max=self.v_max,
            seed=seed,
            max_ep_steps=max_ep_steps,
            value_coef=self.value_coef,
            consistency_coef=self.consistency_coef,
            policy_coef=self.policy_coef,
            max_action=self.max_action,
            min_std=self.min_std,
            std_magnification=self.std_magnification,
            soft_clamp=self.soft_clamp,
            init_std=self.init_std,
            ent_scale=self.ent_scale,
            c_visit=self.c_visit,
            c_scale=self.c_scale,
            target_sync_interval=target_sync_interval,
            reanalyze_interval=reanalyze_interval,
            reanalyze_warmup=reanalyze_warmup,
            reanalyze_batch=reanalyze_batch,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            diag_every=diag_every,
            report_every=report_every,
            logger=logger,
            verbose=verbose,
        )

    def eval_greedy(
        mut self,
        mut env: Self.ENV,
        episodes: Int = 10,
        max_ep_steps: Int = 200,
    ) raises -> Float64:
        """Deployed-agent metric: deterministic (argmax-visit) sampled-Gumbel
        rollout mean return — the honest measure of the policy."""
        comptime N_ENVS = 1
        var planner = SampledGumbelGPUMCTS[
            N_ENVS, Self.ACT_DIM, Self.LATENT, Self.BINS, Self.MAX_NODES,
            Self.K_ROOT, Self.K_NON_ROOT, Self.NUM_SIMS,
        ](
            self.ctx,
            gamma=Float64(self.gamma),
            v_min=Float64(self.v_min),
            v_max=Float64(self.v_max),
            reward_min=Float64(self.v_min),
            reward_max=Float64(self.v_max),
            max_action=Float64(self.max_action),
            min_std=Float64(self.min_std),
            std_magnification=Float64(self.std_magnification),
            soft_clamp=Float64(self.soft_clamp),
            init_std=Float64(self.init_std),
            c_visit=Float64(self.c_visit),
            c_scale=Float64(self.c_scale),
        )
        var rep_a = MZRepGPU[Self.OBS, Self.LATENT, Self.REP].make(self.rep)
        var dyn_a = MZDynGPU[
            Self.LATENT, Self.ACT_DIM, Self.BINS, Self.DYN
        ].make(self.dyn)
        var pred_a = MZContPredGPU[
            Self.LATENT, Self.ACT_DIM, Self.BINS, Self.PRED
        ].make(self.pred)

        var d_obs = self.ctx.enqueue_create_buffer[DT](N_ENVS * Self.OBS)
        var h_obs = mptr(alloc[Scalar[DT]](N_ENVS * Self.OBS))
        var h_act = mptr(alloc[Scalar[DT]](N_ENVS * Self.ACT_DIM))
        var mcts_seed = UInt32(0)
        var total = 0.0
        for _ in range(episodes):
            var eo = env.reset_obs_list()
            var eo_f = List[Float64]()
            for j in range(Self.OBS):
                eo_f.append(Float64(eo[j]))
            var eret = 0.0
            for _step in range(max_ep_steps):
                for j in range(Self.OBS):
                    h_obs[j] = Scalar[DT](eo_f[j])
                self.ctx.enqueue_copy(d_obs, h_obs)
                var obs_t = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
                ](mptr(d_obs.unsafe_ptr()))
                planner.search_gpu[
                    MZRepGPU[Self.OBS, Self.LATENT, Self.REP],
                    MZDynGPU[Self.LATENT, Self.ACT_DIM, Self.BINS, Self.DYN],
                    MZContPredGPU[
                        Self.LATENT, Self.ACT_DIM, Self.BINS, Self.PRED
                    ],
                ](self.ctx, rep_a, dyn_a, pred_a, obs_t,
                  deterministic=True, rng_seed=mcts_seed)
                mcts_seed += UInt32(1)
                self.ctx.enqueue_copy(h_act, planner.chosen_actions_view())
                self.ctx.synchronize()
                var ea = List[Scalar[DT]]()
                for d in range(Self.ACT_DIM):
                    ea.append(h_act[d])
                var es = env.step_continuous_vec[DT](ea)
                eret += Float64(es[1])
                eo_f = List[Float64]()
                for j in range(Self.OBS):
                    eo_f.append(Float64(es[0][j]))
                if es[2]:
                    break
            total += eret
        h_obs.free(); h_act.free()
        return total / Float64(episodes)

    def save(mut self, path: String) raises:
        var body = String("")
        save_state_v2_body_gpu(self.rep, body, String("rep"), self.ctx)
        save_state_v2_body_gpu(self.dyn, body, String("dyn"), self.ctx)
        save_state_v2_body_gpu(self.pred, body, String("pred"), self.ctx)
        save_state_v2_body_gpu(self.proj, body, String("proj"), self.ctx)
        save_state_v2_body_gpu(self.predh, body, String("predh"), self.ctx)
        var content = String("nn-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx = 1
        load_state_v2_body_gpu(self.rep, lines, idx, String("rep"), self.ctx)
        load_state_v2_body_gpu(self.dyn, lines, idx, String("dyn"), self.ctx)
        load_state_v2_body_gpu(self.pred, lines, idx, String("pred"), self.ctx)
        load_state_v2_body_gpu(self.proj, lines, idx, String("proj"), self.ctx)
        load_state_v2_body_gpu(
            self.predh, lines, idx, String("predh"), self.ctx
        )
