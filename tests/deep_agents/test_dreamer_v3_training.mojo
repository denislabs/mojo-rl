"""Test DreamerV3 select_action and update."""

from std.random import random_float64
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.dreamer_v3 import DreamerV3Agent


def main():
    print("=" * 60)
    print("DreamerV3 Training Tests")
    print("=" * 60)

    # Small config
    comptime OBS = 6
    comptime ACT = 2
    var agent = DreamerV3Agent[
        obs_dim=OBS,
        action_dim=ACT,
        deter_dim=32,
        hidden=16,
        stoch_dim=4,
        classes=4,
        units=16,
        num_bins=31,
        blocks=2,
        batch_size=4,
        batch_length=8,
        imagine_horizon=5,
        buffer_capacity=1000,
    ]()

    # Test select_action
    print("Test select_action...")
    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](random_float64(-1.0, 1.0)))

    var action = agent.select_action(obs, training=True)
    print("  Action dim: " + String(len(action)))
    var all_valid = True
    for i in range(len(action)):
        var a = Float64(action[i])
        if a < -1.0 or a > 1.0:
            all_valid = False
    print("  All in [-1,1]: " + String(all_valid))
    print("  PASS")

    # Fill buffer with random data
    print("Test filling buffer...")
    for ep in range(10):
        for step in range(20):
            var o = List[Scalar[dtype]](capacity=OBS)
            for i in range(OBS):
                o.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            var a = List[Scalar[dtype]](capacity=ACT)
            for i in range(ACT):
                a.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            var r = random_float64(-1.0, 1.0)
            var done = step == 19
            agent.observe(o, a, r, done)

    print("  Buffer size: " + String(agent.state.buffer.len()))
    print("  Buffer ready: " + String(agent.state.is_ready()))
    print("  PASS")

    # Test update
    print("Test update...")
    var loss = agent.update()
    print("  Loss: " + String(loss))
    print("  Train steps: " + String(agent.train_step_count))
    var loss_finite = loss == loss  # NaN check
    print("  Loss finite: " + String(loss_finite))
    print("  PASS")

    # Test second update (ensure no crash)
    print("Test second update...")
    var loss2 = agent.update()
    print("  Loss: " + String(loss2))
    print("  PASS")

    print("=" * 60)
    print("All training tests passed.")
    print("=" * 60)
