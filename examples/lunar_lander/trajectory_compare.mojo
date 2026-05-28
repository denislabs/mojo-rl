"""Compare LunarLander trajectory: Native physics2d vs Gymnasium Box2D.

Syncs initial state from Gymnasium to native env, then applies identical
actions and compares raw physics state + observations + rewards step-by-step.

Run with:
    pixi run mojo run -I . examples/lunar_lander/trajectory_compare.mojo
"""

from std.python import Python, PythonObject
from std.math import abs, sqrt

from mojo_rl.envs.lunar_lander import LunarLander
from mojo_rl.envs.lunar_lander.constants import LLConstants
from mojo_rl.envs.lunar_lander.helpers import (
    normalize_position,
    normalize_velocity,
    normalize_angular_velocity,
    compute_shaping,
)
from mojo_rl.physics2d import dtype as phys_dtype

comptime dtype = DType.float32


def fmt(v: Float64, width: Int = 9) -> String:
    var s = String(v)
    if s.byte_length() > width:
        s = String(s[byte=:width])
    while s.byte_length() < width:
        s = s + " "
    return s


def fmt_err(v: Float64) -> String:
    if v < 1e-10:
        return "  0      "
    return fmt(v)


def sync_gym_to_native(
    mut env: LunarLander[dtype],
    gym_env: PythonObject,
) raises:
    """Copy Box2D body states from Gymnasium env into native env."""
    var lander = gym_env.unwrapped.lander
    var legs = gym_env.unwrapped.legs

    # Sync lander body (body 0)
    var lx = Float64(py=lander.position[0])
    var ly = Float64(py=lander.position[1])
    var lvx = Float64(py=lander.linearVelocity[0])
    var lvy = Float64(py=lander.linearVelocity[1])
    var la = Float64(py=lander.angle)
    var lw = Float64(py=lander.angularVelocity)
    env.physics.set_body_position(0, 0, lx, ly)
    env.physics.set_body_velocity(0, 0, lvx, lvy, lw)
    env.physics.set_body_angle(0, 0, la)

    # Sync left leg (body 1)
    var ll = legs[0]
    env.physics.set_body_position(
        0, 1, Float64(py=ll.position[0]), Float64(py=ll.position[1])
    )
    env.physics.set_body_velocity(
        0,
        1,
        Float64(py=ll.linearVelocity[0]),
        Float64(py=ll.linearVelocity[1]),
        Float64(py=ll.angularVelocity),
    )
    env.physics.set_body_angle(0, 1, Float64(py=ll.angle))

    # Sync right leg (body 2)
    var rl = legs[1]
    env.physics.set_body_position(
        0, 2, Float64(py=rl.position[0]), Float64(py=rl.position[1])
    )
    env.physics.set_body_velocity(
        0,
        2,
        Float64(py=rl.linearVelocity[0]),
        Float64(py=rl.linearVelocity[1]),
        Float64(py=rl.angularVelocity),
    )
    env.physics.set_body_angle(0, 2, Float64(py=rl.angle))

    # Reset terrain to flat helipad (match gym helipad region)
    for i in range(LLConstants.TERRAIN_CHUNKS):
        env.terrain_heights[i] = Scalar[dtype](LLConstants.HELIPAD_Y)
    var edge_heights = List[Scalar[phys_dtype]]()
    for i in range(LLConstants.TERRAIN_CHUNKS):
        edge_heights.append(Scalar[phys_dtype](LLConstants.HELIPAD_Y))
    env.edge_collision.set_terrain_from_heights(
        0,
        edge_heights,
        x_start=0.0,
        x_spacing=LLConstants.W_UNITS
        / Float64(LLConstants.TERRAIN_CHUNKS - 1),
    )

    # Recompute cached state and prev_shaping
    env._update_cached_state()
    env.prev_shaping = env._compute_shaping()
    env.step_count = 0
    env.game_over = False


def read_gym_raw(
    gym_env: PythonObject,
) raises -> InlineArray[Float64, 6]:
    """Read raw (x, y, vx, vy, angle, omega) from Gymnasium Box2D body."""
    var lander = gym_env.unwrapped.lander
    var out = InlineArray[Float64, 6](fill=0.0)
    out[0] = Float64(py=lander.position[0])
    out[1] = Float64(py=lander.position[1])
    out[2] = Float64(py=lander.linearVelocity[0])
    out[3] = Float64(py=lander.linearVelocity[1])
    out[4] = Float64(py=lander.angle)
    out[5] = Float64(py=lander.angularVelocity)
    return out


def read_native_raw(
    mut env: LunarLander[dtype],
) -> InlineArray[Float64, 6]:
    """Read raw (x, y, vx, vy, angle, omega) from native physics."""
    var out = InlineArray[Float64, 6](fill=0.0)
    out[0] = Float64(env.physics.get_body_x(0, 0))
    out[1] = Float64(env.physics.get_body_y(0, 0))
    out[2] = Float64(env.physics.get_body_vx(0, 0))
    out[3] = Float64(env.physics.get_body_vy(0, 0))
    out[4] = Float64(env.physics.get_body_angle(0, 0))
    out[5] = Float64(env.physics.get_body_omega(0, 0))
    return out


def run_phase(
    phase_name: String,
    mut env: LunarLander[dtype],
    gym_env: PythonObject,
    np: PythonObject,
    action_0: Float64,
    action_1: Float64,
    num_steps: Int,
) raises:
    """Run a phase with fixed actions and compare trajectories."""
    print()
    print("--- " + phase_name + " ---")
    print(
        "  action=[" + fmt(action_0, 4) + ", " + fmt(action_1, 4) + "]"
    )
    print(
        "Step |   dx       dy       dvx      dvy     "
        " dangle   domega  | Nat_rew  Gym_rew  dR      "
        " | Nat_obs[0:4]                              "
        " | Gym_obs[0:4]"
    )
    print("-" * 160)

    var nat_total = Float64(0)
    var gym_total = Float64(0)

    for step in range(num_steps):
        # Build action
        var action = List[Scalar[dtype]]()
        action.append(Scalar[dtype](action_0))
        action.append(Scalar[dtype](action_1))

        # Step native
        var nat_result = env.step_continuous_vec(action)
        var nat_reward = Float64(nat_result[1])
        var nat_done = nat_result[2]
        nat_total += nat_reward

        # Step gymnasium
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        _ = py_list.append(action_0)
        _ = py_list.append(action_1)
        var np_action = np.array(py_list)
        var gym_result = gym_env.step(np_action)
        var gym_reward = Float64(py=gym_result[1])
        var gym_terminated = gym_result[2].__bool__()
        var gym_truncated = gym_result[3].__bool__()
        var gym_done = gym_terminated or gym_truncated
        gym_total += gym_reward

        # Read raw physics state
        var nat = read_native_raw(env)
        var gym = read_gym_raw(gym_env)

        # Differences
        var dx = abs(nat[0] - gym[0])
        var dy = abs(nat[1] - gym[1])
        var dvx = abs(nat[2] - gym[2])
        var dvy = abs(nat[3] - gym[3])
        var da = abs(nat[4] - gym[4])
        var dw = abs(nat[5] - gym[5])

        # Read observations
        var nat_obs = env.get_observation(0)
        var gym_obs = gym_result[0]

        print(
            fmt(Float64(step), 4)
            + " | "
            + fmt_err(dx)
            + " "
            + fmt_err(dy)
            + " "
            + fmt_err(dvx)
            + " "
            + fmt_err(dvy)
            + " "
            + fmt_err(da)
            + " "
            + fmt_err(dw)
            + " | "
            + fmt(nat_reward)
            + " "
            + fmt(gym_reward)
            + " "
            + fmt_err(abs(nat_reward - gym_reward))
            + " | "
            + fmt(Float64(nat_obs[0]))
            + " "
            + fmt(Float64(nat_obs[1]))
            + " "
            + fmt(Float64(nat_obs[2]))
            + " "
            + fmt(Float64(nat_obs[3]))
            + " | "
            + fmt(Float64(py=gym_obs[0]))
            + " "
            + fmt(Float64(py=gym_obs[1]))
            + " "
            + fmt(Float64(py=gym_obs[2]))
            + " "
            + fmt(Float64(py=gym_obs[3]))
        )

        if nat_done:
            print("  >>> Native terminated <<<")
            break
        if gym_done:
            print(
                "  >>> Gymnasium terminated (terminated="
                + String(gym_terminated)
                + ") <<<"
            )
            break

    print(
        "  Total: native="
        + fmt(nat_total)
        + " gym="
        + fmt(gym_total)
        + " diff="
        + fmt_err(abs(nat_total - gym_total))
    )


def main() raises:
    print("=" * 70)
    print("LunarLander Trajectory: Native physics2d vs Gymnasium Box2D")
    print("=" * 70)

    var np = Python.import_module("numpy")
    var gym = Python.import_module("gymnasium")

    # === Create environments ===
    var env = LunarLander[dtype]()
    _ = env.reset()

    var gym_env = gym.make("LunarLander-v3", continuous=True)
    _ = gym_env.reset(seed=42)

    # === Sync native state from Gymnasium ===
    sync_gym_to_native(env, gym_env)

    # Verify sync
    var nat0 = read_native_raw(env)
    var gym0 = read_gym_raw(gym_env)
    print("\nInitial state (synced from Gymnasium):")
    print(
        "  Lander: x="
        + fmt(gym0[0])
        + " y="
        + fmt(gym0[1])
        + " vx="
        + fmt(gym0[2])
        + " vy="
        + fmt(gym0[3])
    )
    print(
        "          angle="
        + fmt(gym0[4])
        + " omega="
        + fmt(gym0[5])
    )
    print(
        "  Sync error: dx="
        + fmt_err(abs(nat0[0] - gym0[0]))
        + " dy="
        + fmt_err(abs(nat0[1] - gym0[1]))
        + " dvx="
        + fmt_err(abs(nat0[2] - gym0[2]))
        + " dvy="
        + fmt_err(abs(nat0[3] - gym0[3]))
    )

    # Compare initial observations
    var nat_obs_init = env.get_observation(0)
    var gym_obs_raw = gym_env.unwrapped.lander
    print("\nInitial observations:")
    print(
        "  Native: ["
        + fmt(Float64(nat_obs_init[0]))
        + fmt(Float64(nat_obs_init[1]))
        + fmt(Float64(nat_obs_init[2]))
        + fmt(Float64(nat_obs_init[3]))
        + fmt(Float64(nat_obs_init[4]))
        + fmt(Float64(nat_obs_init[5]))
        + fmt(Float64(nat_obs_init[6]))
        + fmt(Float64(nat_obs_init[7]))
        + "]"
    )

    # Compute what Gymnasium obs should be from same raw state
    var pos_n = normalize_position[dtype](
        Scalar[dtype](gym0[0]), Scalar[dtype](gym0[1])
    )
    var vel_n = normalize_velocity[dtype](
        Scalar[dtype](gym0[2]), Scalar[dtype](gym0[3])
    )
    var omega_n = normalize_angular_velocity[dtype](Scalar[dtype](gym0[5]))
    print(
        "  Expected (from raw): ["
        + fmt(Float64(pos_n[0]))
        + fmt(Float64(pos_n[1]))
        + fmt(Float64(vel_n[0]))
        + fmt(Float64(vel_n[1]))
        + fmt(gym0[4])
        + fmt(Float64(omega_n))
        + "...]"
    )

    # === Phase 1: Free fall ===
    run_phase("Phase 1: Free fall", env, gym_env, np, 0.0, 0.0, 30)

    # === Re-sync for phase 2 ===
    _ = gym_env.reset(seed=123)
    _ = env.reset()
    sync_gym_to_native(env, gym_env)

    # === Phase 2: Main engine full thrust ===
    run_phase("Phase 2: Main engine (action[0]=1.0)", env, gym_env, np, 1.0, 0.0, 30)

    # === Re-sync for phase 3 ===
    _ = gym_env.reset(seed=456)
    _ = env.reset()
    sync_gym_to_native(env, gym_env)

    # === Phase 3: Side engine ===
    run_phase("Phase 3: Right side engine (action[1]=0.8)", env, gym_env, np, 0.0, 0.8, 30)

    # === Re-sync for phase 4 ===
    _ = gym_env.reset(seed=789)
    _ = env.reset()
    sync_gym_to_native(env, gym_env)

    # === Phase 4: Both engines ===
    run_phase(
        "Phase 4: Main + side (action=[0.6, -0.7])", env, gym_env, np, 0.6, -0.7, 30
    )

    print()
    print("=" * 70)
    print("Done. Large dx/dy/dvx/dvy values indicate physics divergence.")
    print("=" * 70)
