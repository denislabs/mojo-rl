"""Diagnostic: What does the ConnectFour CNN output for a fresh network?"""

from std.math import exp
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.configs import (
    AlphaZeroConfig,
    AlphaZeroConnectFourCNNConfig,
    AlphaZeroConnectFourConfig,
)

comptime CNNConfig = AlphaZeroConnectFourCNNConfig[]
comptime MLPConfig = AlphaZeroConnectFourConfig[]


def test_output[Config: AlphaZeroConfig](name: String) raises:
    print("=" * 60)
    print("Testing:", name)
    print("  PARAM_SIZE:", Config.PredModel.PARAM_SIZE)
    print("  IN_DIM:", Config.PredModel.IN_DIM)
    print("  OUT_DIM:", Config.PredModel.OUT_DIM)
    print("=" * 60)

    comptime OBS = Config.obs_dim
    comptime PRED_IN = Config.PredModel.IN_DIM
    comptime ACT = Config.action_dim
    comptime PRED_OUT = Config.PredModel.OUT_DIM
    comptime CACHE_SIZE = Config.PredModel.CACHE_SIZE
    comptime BATCH = 4

    print("  PRED_IN:", PRED_IN, "(obs_dim:", OBS, ")")

    var state = NetworkState[Config.PredModel, Config.OptType]()
    state.initialize[Kaiming[]]()

    # Create obs: empty board (all cells in plane 2 = 1)
    var obs_data = alloc[Scalar[dtype]](BATCH * PRED_IN)
    memset(obs_data, 0, BATCH * PRED_IN)
    # Plane 2 (empty) starts at index 84 (2 * 42), has 42 cells
    for b in range(BATCH):
        for i in range(42):
            obs_data[b * PRED_IN + 84 + i] = Scalar[dtype](1.0)

    # Sample 1: one piece in center column (col 3, row 0)
    obs_data[1 * PRED_IN + 3] = Scalar[dtype](1.0)
    obs_data[1 * PRED_IN + 84 + 3] = Scalar[dtype](0.0)

    var pred_data = alloc[Scalar[dtype]](BATCH * PRED_OUT)
    memset(pred_data, 0, BATCH * PRED_OUT)
    var cache_data = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)
    memset(cache_data, 0, BATCH * CACHE_SIZE)

    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_IN), MutAnyOrigin](obs_data)
    var pred_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_OUT), MutAnyOrigin](pred_data)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin](cache_data)
    Config.PredModel.forward[BATCH](obs_t, pred_t, state.params_view(), state.model_state_view(), cache_t)

    for b in range(BATCH):
        print("  Sample", b, "logits:", end="")
        var max_l: Float64 = -1e18
        for a in range(ACT):
            var v = Float64(pred_data[b * PRED_OUT + a])
            print("", Float64(Int(v * 1000)) / 1000.0, end="")
            if v > max_l:
                max_l = v
        var raw_v = Float64(pred_data[b * PRED_OUT + ACT])
        print(" | val=", Float64(Int(raw_v * 1000)) / 1000.0)

        # Softmax
        var sum_e: Float64 = 0.0
        for a in range(ACT):
            sum_e += exp(Float64(pred_data[b * PRED_OUT + a]) - max_l)
        print("         probs:", end="")
        for a in range(ACT):
            var prob = exp(Float64(pred_data[b * PRED_OUT + a]) - max_l) / sum_e
            print("", Float64(Int(prob * 1000)) / 1000.0, end="")
        print()

    # Check: are logits same for different inputs?
    var same = True
    for a in range(ACT):
        var diff = Float64(pred_data[0 * PRED_OUT + a]) - Float64(pred_data[1 * PRED_OUT + a])
        if diff > 0.001 or diff < -0.001:
            same = False
            break
    if same:
        print("  WARNING: Logits identical for different inputs!")
    else:
        print("  OK: Logits differ between inputs")

    obs_data.free()
    pred_data.free()


def main() raises:
    test_output[CNNConfig]("ConnectFour CNN")
    print()
    test_output[MLPConfig]("ConnectFour MLP")
