"""MuZeroAgent — facade over the self-play driver + greedy eval + checkpointing.

Mirrors `AlphaZeroAgent`, but MuZero carries a *learned model* — three nets
(h/g/f = representation / dynamics / prediction) with three optimizers — instead
of AlphaZero's single net + true game rules. Parameterized on `TARGET`
("cpu" / "gpu") like the SAC dual-path. The GPU path is a **CPU-search /
GPU-train hybrid**: the K-step BPTT unroll runs on the device while the MCTS
search runs on a CPU mirror of the nets (downloaded from the device after each
train step) — see `selfplay_gpu.mojo`. `save` / `load` are device-aware.

  * `__init__(ctx, lr, gamma, v_min, v_max, value_coef)` — build fresh Kaiming
    rep/dyn/pred nets on `TARGET`. `v_min`/`v_max` are the **h-space** value/reward
    support shared with the targets and the planner.
  * `train(env, iterations, ...)` — single-player self-play (env → CPU MCTS over
    the learned model → sequence replay → K-step BPTT unroll). Optimizers are
    session-local (recreated per call), nets persist across calls.
  * `eval_greedy(env, episodes)` — the deployed-agent metric: noise-off,
    argmax-visit rollout. ALWAYS judge the policy by this, not the exploratory
    training return (∝-visit sampling + root Dirichlet noise badly understates it).
  * `save` / `load` — one-file `nn2-ckpt v2` envelope packing all three nets
    (sections `rep` / `dyn` / `pred`).

The three nets share `LATENT` + `BINS`; reward (`dyn`) and value (`pred`) heads
are categorical over `[v_min, v_max]` — keep those in sync with the config.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body,
    load_state_v2_body,
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
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

from .selfplay_cpu import run_muzero_selfplay_cpu
from .selfplay_gpu import run_muzero_selfplay_gpu, mz_sync_gpu_to_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU


@fieldwise_init
struct MuZeroAgent[
    TARGET: StaticString,
    ENV: BoxDiscreteActionEnv & ImplicitlyDestructible,
    REP: Module,
    DYN: Module,
    PRED: Module,
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
    var ctx: Optional[DeviceContext]
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
    ) raises:
        self.ctx = ctx
        self.rep = Self.REP.make[Self.TARGET, INIT=Kaiming](ctx=ctx)
        self.dyn = Self.DYN.make[Self.TARGET, INIT=Kaiming](ctx=ctx)
        self.pred = Self.PRED.make[Self.TARGET, INIT=Kaiming](ctx=ctx)
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef

    def train(
        mut self,
        mut env: Self.ENV,
        iterations: Int,
        learning_starts: Int = 500,
        train_per_iter: Int = 1,
        seed: UInt64 = 0,
        max_ep_steps: Int = 500,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        verbose: Bool = False,
    ) raises -> Float64:
        """Single-player self-play training over the learned model. Returns the
        last training loss. Optimizers are recreated here (session-local); the
        nets keep their weights across `train` calls."""
        comptime if Self.TARGET == "cpu":
            var orep = Adam.make["cpu", M = Self.REP](self.rep)
            var odyn = Adam.make["cpu", M = Self.DYN](self.dyn)
            var opred = Adam.make["cpu", M = Self.PRED](self.pred)
            orep.lr = self.lr
            odyn.lr = self.lr
            opred.lr = self.lr
            return run_muzero_selfplay_cpu[
                Self.ENV, Self.REP, Self.DYN, Self.PRED,
                Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
                Self.NUM_SIMS, Self.MAX_NODES, Self.CAP,
                Self.B, Self.K, Self.N,
            ](
                env, self.rep, self.dyn, self.pred, orep, odyn, opred,
                iterations=iterations,
                learning_starts=learning_starts,
                train_per_iter=train_per_iter,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
                seed=seed,
                max_ep_steps=max_ep_steps,
                value_coef=self.value_coef,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                verbose=verbose,
            )
        else:
            var c = self.ctx.value()
            var orep = Adam.make["gpu", M = Self.REP](self.rep, c)
            var odyn = Adam.make["gpu", M = Self.DYN](self.dyn, c)
            var opred = Adam.make["gpu", M = Self.PRED](self.pred, c)
            orep.lr = self.lr
            odyn.lr = self.lr
            opred.lr = self.lr
            return run_muzero_selfplay_gpu[
                Self.ENV, Self.REP, Self.DYN, Self.PRED,
                Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
                Self.NUM_SIMS, Self.MAX_NODES, Self.CAP,
                Self.B, Self.K, Self.N,
            ](
                c, env, self.rep, self.dyn, self.pred, orep, odyn, opred,
                iterations=iterations,
                learning_starts=learning_starts,
                train_per_iter=train_per_iter,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
                seed=seed,
                max_ep_steps=max_ep_steps,
                value_coef=self.value_coef,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                verbose=verbose,
            )

    def eval_greedy(
        mut self,
        mut env: Self.ENV,
        episodes: Int = 10,
        max_ep_steps: Int = 500,
    ) raises -> Float64:
        """Deployed-agent metric: greedy (noise-off, argmax-visit) rollout mean
        return. This is the honest measure of the policy — the training return is
        exploratory and understates it."""
        comptime if Self.TARGET == "cpu":
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
        else:
            # CPU-search / GPU-train hybrid: the MCTS search runs on a CPU
            # mirror of the device nets. Download the trained device params into
            # the mirror, then run the same greedy rollout as the CPU branch.
            var c = self.ctx.value()
            var rep_c = Self.REP.make["cpu", INIT=Kaiming]()
            var dyn_c = Self.DYN.make["cpu", INIT=Kaiming]()
            var pred_c = Self.PRED.make["cpu", INIT=Kaiming]()
            mz_sync_gpu_to_cpu(self.rep, rep_c, c)
            mz_sync_gpu_to_cpu(self.dyn, dyn_c, c)
            mz_sync_gpu_to_cpu(self.pred, pred_c, c)
            var mcts = GenericCPUMCTS[
                Self.ACT, Self.LATENT, Self.NUM_SIMS, Self.MAX_NODES,
                MuZeroPUCT[19652.0, 1.25],
                DirichletNoise[0.25, 0.25], SinglePlayer, 8, 3,
            ](gamma=Float64(self.gamma))
            var rep_a = MZRepCPU[Self.OBS, Self.LATENT, Self.REP](
                net=UnsafePointer(to=rep_c)
            )
            var dyn_a = MZDynCPU[Self.LATENT, Self.ACT, Self.BINS, Self.DYN](
                net=UnsafePointer(to=dyn_c),
                v_min=self.v_min, v_max=self.v_max,
            )
            var pred_a = MZPredCPU[
                Self.LATENT, Self.ACT, Self.BINS, Self.PRED
            ](
                net=UnsafePointer(to=pred_c),
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
            var result = total / Float64(episodes)
            # keep the CPU mirrors alive until *after* the rollout — the MCTS
            # adapters hold `UnsafePointer(to=rep_c)`, an indirection the
            # lifetime analyzer can't see, so without these extenders it may free
            # the mirrors right after the sync and dangle the search pointers.
            _ = rep_c^
            _ = dyn_c^
            _ = pred_c^
            return result

    def save(mut self, path: String) raises:
        var body = String("")
        comptime if Self.TARGET == "cpu":
            save_state_v2_body(self.rep, body, String("rep"))
            save_state_v2_body(self.dyn, body, String("dyn"))
            save_state_v2_body(self.pred, body, String("pred"))
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.rep, body, String("rep"), c)
            save_state_v2_body_gpu(self.dyn, body, String("dyn"), c)
            save_state_v2_body_gpu(self.pred, body, String("pred"), c)
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx = 1
        comptime if Self.TARGET == "cpu":
            load_state_v2_body(self.rep, lines, idx, String("rep"))
            load_state_v2_body(self.dyn, lines, idx, String("dyn"))
            load_state_v2_body(self.pred, lines, idx, String("pred"))
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.rep, lines, idx, String("rep"), c)
            load_state_v2_body_gpu(self.dyn, lines, idx, String("dyn"), c)
            load_state_v2_body_gpu(self.pred, lines, idx, String("pred"), c)
