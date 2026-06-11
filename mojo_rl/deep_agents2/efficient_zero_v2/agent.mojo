"""EZv2DiscreteAgent — facade over the discrete self-play driver + greedy eval +
checkpointing (CPU).

Mirrors `MuZeroAgent`, but EZv2 carries two extra SimSiam nets (projector +
predictor) and trains with the temporal-consistency objective. This is the
**CPU** discrete agent (the lighthouse path, decision D1: CPU uses vanilla PUCT
MCTS). The GPU Gumbel path is a separate build.

  * `__init__(lr, gamma, v_min, v_max, value_coef, consistency_coef)` — build
    fresh Kaiming rep/dyn/pred/proj/predh nets on CPU.
  * `train(env, iterations, ...)` — single-player self-play (env → CPU MCTS over
    the learned model → sequence replay → EZv2 BPTT+consistency unroll).
    Optimizers are session-local (recreated per call); nets persist across calls.
  * `eval_greedy(env, episodes)` — the deployed-agent metric: noise-off,
    argmax-visit rollout (judge the policy by this, not the exploratory training
    return). The projector/predictor are consistency-only, unused here.
  * `save` / `load` — one-file `nn2-ckpt v2` envelope packing all five nets
    (sections `rep` / `dyn` / `pred` / `proj` / `predh`).
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body,
    load_state_v2_body,
)
from mojo_rl.deep_agents2.core.checkpoint_helpers import (
    split_lines_v2,
    read_file_v2,
    expect_v2_header,
)
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)

from mojo_rl.core.logger import Logger, NoOpLogger
from .selfplay_cpu import run_ezv2_selfplay_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU


@fieldwise_init
struct EZv2DiscreteAgent[
    ENV: BoxDiscreteActionEnv & ImplicitlyDestructible,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
](ImplicitlyDestructible, Movable):
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var proj: Self.PROJM
    var predh: Self.PREDH
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]
    var consistency_coef: Scalar[DT]

    def __init__(
        out self,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    ) raises:
        self.rep = Self.REP.make["cpu", INIT=Kaiming]()
        self.dyn = Self.DYN.make["cpu", INIT=Kaiming]()
        self.pred = Self.PRED.make["cpu", INIT=Kaiming]()
        self.proj = Self.PROJM.make["cpu", INIT=Kaiming]()
        self.predh = Self.PREDH.make["cpu", INIT=Kaiming]()
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        self.consistency_coef = consistency_coef

    def train[
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: Self.ENV,
        iterations: Int,
        learning_starts: Int = 500,
        train_per_iter: Int = 1,
        seed: UInt64 = 0,
        max_ep_steps: Int = 500,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
    ) raises -> Float64:
        """Single-player self-play training (MuZero BPTT + SimSiam consistency).
        Returns the last training loss. Optimizers recreated here; nets persist."""
        var orep = Adam.make["cpu", M = Self.REP](self.rep)
        var odyn = Adam.make["cpu", M = Self.DYN](self.dyn)
        var opred = Adam.make["cpu", M = Self.PRED](self.pred)
        var oproj = Adam.make["cpu", M = Self.PROJM](self.proj)
        var opredh = Adam.make["cpu", M = Self.PREDH](self.predh)
        orep.lr = self.lr
        odyn.lr = self.lr
        opred.lr = self.lr
        oproj.lr = self.lr
        opredh.lr = self.lr
        return run_ezv2_selfplay_cpu[
            Self.ENV, Self.REP, Self.DYN, Self.PRED, Self.PROJM, Self.PREDH,
            Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
            Self.NUM_SIMS, Self.MAX_NODES, Self.CAP,
            Self.B, Self.K, Self.N,
            L=L,
        ](
            env, self.rep, self.dyn, self.pred, self.proj, self.predh,
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
        max_ep_steps: Int = 500,
    ) raises -> Float64:
        """Deployed-agent metric: greedy (noise-off, argmax-visit) rollout mean
        return — the honest measure of the policy."""
        var mcts = GenericCPUMCTS[
            Self.ACT, Self.LATENT, Self.NUM_SIMS, Self.MAX_NODES,
            MuZeroPUCT[19652.0, 1.25],
            DirichletNoise[0.25, 0.25], SinglePlayer, 8, 3,
        ](gamma=Float64(self.gamma))
        var rep_a = MZRepCPU[Self.OBS, Self.LATENT, Self.REP](
            net=UnsafePointer(to=self.rep)
        )
        var dyn_a = MZDynCPU[Self.LATENT, Self.ACT, Self.BINS, Self.DYN](
            net=UnsafePointer(to=self.dyn),
            v_min=self.v_min, v_max=self.v_max,
        )
        var pred_a = MZPredCPU[
            Self.LATENT, Self.ACT, Self.BINS, Self.PRED
        ](
            net=UnsafePointer(to=self.pred),
            v_min=self.v_min, v_max=self.v_max,
        )
        var total = 0.0
        for _ in range(episodes):
            var eo = env.reset_obs_list()
            var eo_f = List[Float64]()
            for j in range(Self.OBS):
                eo_f.append(Float64(eo[j]))
            var eret = 0.0
            for _step in range(max_ep_steps):
                var ep = mcts.search[
                    type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                ](rep_a, dyn_a, pred_a, eo_f, add_noise=False)
                var best = 0
                for a in range(1, Self.ACT):
                    if ep[a] > ep[best]:
                        best = a
                var es = env.step_obs(best)
                eret += Float64(es[1])
                eo_f = List[Float64]()
                for j in range(Self.OBS):
                    eo_f.append(Float64(es[0][j]))
                if es[2]:
                    break
            total += eret
        return total / Float64(episodes)

    def save(mut self, path: String) raises:
        """Weights-only snapshot of the five EZv2 nets (rep / dyn / pred /
        proj / predh) in the `nn2-ckpt v2` envelope. Uses the `save`/`load`
        surface shared by every agent facade. NOTE: optimizers are
        session-local — rebuilt per `train_*` call — so only weights
        persist; this is the inference / self-play artifact, not a
        training-resume checkpoint."""
        var body = String("")
        save_state_v2_body(self.rep, body, String("rep"))
        save_state_v2_body(self.dyn, body, String("dyn"))
        save_state_v2_body(self.pred, body, String("pred"))
        save_state_v2_body(self.proj, body, String("proj"))
        save_state_v2_body(self.predh, body, String("predh"))
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load(mut self, path: String) raises:
        """Inverse of `save` — restores all five net weights. Optimizer
        state is not checkpointed (session-local; see `save`)."""
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx = 1
        load_state_v2_body(self.rep, lines, idx, String("rep"))
        load_state_v2_body(self.dyn, lines, idx, String("dyn"))
        load_state_v2_body(self.pred, lines, idx, String("pred"))
        load_state_v2_body(self.proj, lines, idx, String("proj"))
        load_state_v2_body(self.predh, lines, idx, String("predh"))
