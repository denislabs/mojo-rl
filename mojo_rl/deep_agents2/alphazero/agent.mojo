"""AlphaZeroAgent — facade over the self-play driver + eval + checkpointing.

Owns the (GPU-resident) prediction net and a device context, and packages the
loose functions into a usable API:

  * `make`              — build a fresh agent (Kaiming-init net).
  * `train(iterations)` — run a self-play training session, return last loss.
  * `eval_vs_random`    — greedy-policy win/draw/loss vs a random opponent.
  * `save` / `load`     — one-file `nn2-ckpt v2` checkpoint of the net (the
                          GPU saver is byte-identical to the CPU one, so a
                          GPU-trained model reloads anywhere).

The net's params persist across `train` calls (training continues from the
current weights); the optimizer + replay are session-local (recreated per
`train` call) — fine for the board-game scale here, revisit for long
incremental runs.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.deep_agents2.core.checkpoint_helpers import (
    split_lines_v2, read_file_v2, expect_v2_header,
)
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv

from .selfplay import run_alphazero_selfplay
from .eval import eval_policy_vs_random, EvalResult


@fieldwise_init
struct AlphaZeroAgent[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
](Movable, ImplicitlyDestructible):
    var ctx: DeviceContext
    var net: Self.NET
    var lr: Scalar[DT]

    @staticmethod
    def make(
        ctx: DeviceContext, lr: Scalar[DT] = Scalar[DT](0.01)
    ) raises -> Self:
        return Self(
            ctx=ctx,
            net=Self.NET.make["gpu", INIT=Kaiming](ctx=ctx),
            lr=lr,
        )

    def train(
        mut self,
        iterations: Int,
        learning_starts: Int = 20,
        train_per_iter: Int = 2,
        seed: UInt64 = 0,
    ) raises -> Float64:
        return run_alphazero_selfplay[
            Self.ENV, Self.NET, Self.N_ENVS, Self.NUM_SIMS, Self.MAX_NODES,
            Self.BATCH, Self.CAP, Self.MAX_TRAJ,
        ](
            self.ctx, self.net, iterations, learning_starts, train_per_iter,
            self.lr, seed,
        )

    def eval_vs_random[
        N_EVAL: Int, RESULT_IDX: Int, MAX_PLIES: Int
    ](
        mut self, agent_player: Int = 0, seed: UInt64 = 1
    ) raises -> EvalResult:
        return eval_policy_vs_random[
            Self.ENV, Self.NET, N_EVAL, RESULT_IDX, MAX_PLIES
        ](self.ctx, self.net, agent_player, seed)

    def save(mut self, path: String) raises:
        var body = String("")
        save_state_v2_body_gpu(self.net, body, String("net"), self.ctx)
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx = 1
        load_state_v2_body_gpu(self.net, lines, idx, String("net"), self.ctx)
