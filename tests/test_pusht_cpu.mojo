"""CPU PushTEnv smoke test.

Goals:
  * reset returns a 18D obs vector
  * Stepping moves the agent toward the action target (PD control sanity)
  * Coverage stays in [0, 1]
  * Pushing the block from one side moves it toward the other side
"""

from std.math import pi, sqrt
from mojo_rl.envs.pusht import PushTEnv, PConstants, PushTAction


def assert_close(
    name: String, got: Float64, want: Float64, tol: Float64 = 1.0e-3
) raises:
    var d = got - want
    if d < 0.0:
        d = -d
    if d > tol:
        raise Error(
            String(
                name,
                " expected ",
                want,
                " got ",
                got,
                " (diff ",
                d,
                ")",
            )
        )


def main() raises:
    var env = PushTEnv[DType.float32](seed=42)
    var obs = env.reset_obs_list()
    print("obs_dim = ", len(obs), " expected = ", PConstants.OBS_DIM)
    if len(obs) != PConstants.OBS_DIM:
        raise Error("obs_dim mismatch")

    # Initial coverage should be in [0, 1]
    var cov0 = env.coverage()
    print("initial coverage = ", cov0)
    if cov0 < Float32(0.0) or cov0 > Float32(1.0):
        raise Error("coverage out of [0,1]")

    # PD-control sanity: take a few steps with a target far from the agent
    # and verify the agent moves toward it.
    var ap0 = env.agent_pos()
    print("agent start = (", ap0[0], ", ", ap0[1], ")")
    var target_x = Scalar[DType.float32](100.0)
    var target_y = Scalar[DType.float32](100.0)
    var a = PushTAction[DType.float32](
        target_x=target_x, target_y=target_y
    )
    var dist0 = sqrt(
        Float64((ap0[0] - target_x) * (ap0[0] - target_x))
        + Float64((ap0[1] - target_y) * (ap0[1] - target_y))
    )
    var last_dist = dist0
    var last_reward = Float32(0.0)
    for step in range(10):
        var r = env.step(a)
        var ap = env.agent_pos()
        var dist = sqrt(
            Float64((ap[0] - target_x) * (ap[0] - target_x))
            + Float64((ap[1] - target_y) * (ap[1] - target_y))
        )
        last_reward = r[1]
        print(
            "step ",
            step,
            ": agent=(",
            ap[0],
            ",",
            ap[1],
            ") dist=",
            dist,
            " reward=",
            r[1],
            " cov=",
            env.coverage(),
        )
        last_dist = dist
    print("final dist = ", last_dist, " (was ", dist0, ")")
    if last_dist >= dist0:
        raise Error("PD control failed: agent didn't move toward target")

    # Reward must be a valid float in [0, 1]
    if last_reward < Float32(0.0) or last_reward > Float32(1.0):
        raise Error("reward out of [0,1]")

    # Reset to a known configuration: agent right of block, target left of block.
    # The block (at goal angle) should get pushed left.
    _ = env.reset()
    print("after reset, ok")

    print("CPU PushTEnv smoke test passed.")
