"""Smoke: the CPU AlphaZero adapters (AZRepCPU + AZPredCPU) run a storage-nn
prediction through the planner's Representation/Prediction CPU trait surface.

Validates that a storage `Module` net satisfies the CPU adapter path: AZRepCPU
snapshots the live env state into the latent, AZPredCPU loads it, runs the net
(`forward["cpu", 1]` on owned host `Tensor`s — no raw pointers), and returns a
finite legal-masked policy + tanh-squashed value.

Run: pixi run mojo run -I . tests/deep_agents/test_az_pred_adapter_cpu_smoke.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.zero.mcts_adapters_cpu import AZRepCPU, AZPredCPU
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Net = AZMLPNet[OBS, ACT, H]

    var env = Env()
    _ = env.reset()
    var net = Net.make["cpu", Kaiming]()

    var rep = AZRepCPU[Env, OBS](env=UnsafePointer(to=env))
    var pred = AZPredCPU[Env, OBS, ACT, Net](
        env=UnsafePointer(to=env), net=UnsafePointer(to=net)
    )

    # AZRepCPU snapshots the live env state into the latent the planner threads.
    var hidden = List[Float64](length=Env.SAVE_SIZE, fill=0.0)
    var dummy_obs = List[Float64](length=OBS, fill=0.0)
    rep.encode_cpu(dummy_obs, hidden)

    var policy = List[Float64](length=ACT, fill=0.0)
    var value = pred.predict_cpu(hidden, policy)

    var all_finite = value == value  # not NaN
    var psum: Float64 = 0.0
    for a in range(ACT):
        if policy[a] != policy[a]:
            all_finite = False
        psum += policy[a]
    assert_true(all_finite, "AZ CPU prediction produced NaN")
    # Non-terminal root: legal-masked policy should renormalize to ~1.
    assert_true(psum > 0.99 and psum < 1.01, "policy not normalized: " + String(psum))

    _ = env^  # keepalive: adapters hold non-owning pointers
    _ = net^
    print("AZ pred adapter CPU smoke: OK (value=", value, " psum=", psum, ")")
