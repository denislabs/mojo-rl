"""CPU CarRacingMB smoke test (multi-body physics, headless).

Validates the multi-body env end-to-end and, critically, the regression that
motivated the whole rewrite: the legacy single-body env spun in place off-track
and could NEVER reach the off-playfield -100 termination (its smoke test had to
drop that assertion). The multi-body car drives straight off-track and DOES
leave the playfield, so we assert the -100 here.

Goals:
  * reset() returns a 13-D NORMALIZED observation
  * full gas accelerates the car forward and visits tiles (Gymnasium reward)
  * driving straight off-track terminates with the -100 off-playfield penalty
"""

from mojo_rl.envs.car_racing import CarRacingMB


def fail(name: String, msg: String) raises:
    raise Error(String("[", name, "] ", msg))


def main() raises:
    print("=== CarRacingMB CPU smoke test (multi-body) ===")

    var env = CarRacingMB[DType.float32](max_steps=1000)
    var obs = env.reset()

    # ---- 1. obs shape + normalization -----------------------------------
    print("obs_dim =", len(obs), " track_length =", env.track_length())
    if len(obs) != env.obs_dim():
        fail("obs", "obs_dim mismatch")
    if env.track_length() < 20:
        fail("track", "degenerate track (<20 tiles)")
    # Normalized obs should be O(1): position/PLAYFIELD, sin/cos, vel/100, etc.
    for i in range(len(obs)):
        var v = Float64(obs[i])
        var av = v if v >= 0.0 else -v
        if av > 5.0:
            fail("obs", String("obs[", i, "]=", v, " not normalized (|v|>5)"))

    # ---- 2. full gas straight: accelerate + visit tiles -----------------
    var max_speed: Float64 = 0.0
    var tile_bonus_seen = False
    var done = False
    var steps = 0
    while not done and steps < 150:
        var r = env.step(0.0, 1.0, 0.0)  # steer 0, full gas, no brake
        done = r[2]
        steps += 1
        var sp = env.hull_speed()
        if sp > max_speed:
            max_speed = sp
        if Float64(r[1]) > 0.0:
            tile_bonus_seen = True

    var hp = env.hull_pos()
    print(
        "after", steps, "steps: max_speed=", max_speed,
        " tiles=", env.tiles_visited, "/", env.track_length(),
        " pos=(", hp[0], ",", hp[1], ")",
    )
    if max_speed <= 1.0:
        fail("accel", "car never accelerated under full gas")
    if env.tiles_visited <= 0:
        fail("tiles", "car visited no tiles driving forward")
    if not tile_bonus_seen:
        fail("reward", "never saw a positive tile-visit reward")

    # ---- 3. KILLER: straight off-track -> off-playfield -100 ------------
    # (The legacy single-body env spun in place and never reached this.)
    var env2 = CarRacingMB[DType.float32](max_steps=100000)
    _ = env2.reset()
    var term_reward: Float64 = 0.0
    var term_done = False
    var term_steps = 0
    while not term_done and term_steps < 3000:
        var r2 = env2.step(0.0, 1.0, 0.0)
        term_reward = Float64(r2[1])
        term_done = r2[2]
        term_steps += 1

    var hp2 = env2.hull_pos()
    print(
        "off-field run: done=", term_done, " after", term_steps,
        " steps, last_reward=", term_reward,
        " pos=(", hp2[0], ",", hp2[1], ")",
    )
    if not term_done:
        fail("term", "car never left the playfield driving straight (still spinning?)")
    var lapped = Float64(env2.tiles_visited) >= 0.95 * Float64(env2.track_length())
    if not lapped:
        if term_reward > -99.0:
            fail(
                "term",
                String("expected -100 off-playfield penalty, got ", term_reward),
            )

    print("=== PASS: multi-body env drives, rewards, AND reaches off-playfield -100 ===")
