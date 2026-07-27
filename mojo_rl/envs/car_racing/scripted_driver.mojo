"""Scripted pure-pursuit driver for CarRacingMB (discrete actions).

A decent, training-free data-generation policy: reads the car pose + track tile
centers straight from the env, steers toward a tile LOOKAHEAD ahead of the
nearest one, and gasses when aligned. Interleaves a gas tick while steering so
the car keeps momentum through curves. Returns a discrete action in
{0=noop, 1=left, 2=right, 3=gas, 4=brake}.

Tuned on CarRacingMB: ~+100 return / ~57 tiles per episode (vs random ~-56 / 1
tile). Used to build the decent-policy dataset for OFFLINE Dreamer 4 validation
(the paper's setting: pretrain WM+BC on good data, then imagination-RL).

    from mojo_rl.envs.car_racing.scripted_driver import scripted_car_racing_action
    var a = scripted_car_racing_action(env, step)   # env: CarRacingMB, step: Int
"""

from std.math import atan2, sqrt
from mojo_rl.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB

comptime _PI = 3.14159265358979


def _wrap(a: Float64) -> Float64:
    var x = a
    while x > _PI:
        x -= 2.0 * _PI
    while x < -_PI:
        x += 2.0 * _PI
    return x


def scripted_car_racing_action[
    DTYPE: DType,
    PIXEL_OBS: Bool,
    PIX_RES: Int, //,
    LOOKAHEAD: Int = 3,
    STEER_THRESH: Float64 = 0.06,
    TARGET_SPEED: Float64 = 9.0,
](mut env: CarRacingMB[DTYPE, PIXEL_OBS, PIX_RES], step: Int) -> Int:
    """One discrete action for the current env state. `step` drives the
    gas-while-steering interleave (pass the episode/global step counter)."""
    comptime ho = env.BODIES_OFFSET
    var st = env._state()
    var cx = Float64(rebind[Scalar[DTYPE]](st[0, ho + IDX_X]))
    var cy = Float64(rebind[Scalar[DTYPE]](st[0, ho + IDX_Y]))
    var ca = Float64(rebind[Scalar[DTYPE]](st[0, ho + IDX_ANGLE]))
    var vx = Float64(rebind[Scalar[DTYPE]](st[0, ho + IDX_VX]))
    var vy = Float64(rebind[Scalar[DTYPE]](st[0, ho + IDX_VY]))
    var spd = sqrt(vx * vx + vy * vy)

    var tl = env.track.track_length
    if tl <= 0:
        return 3                                 # no track yet → gas

    # nearest track tile to the car
    var best_i = 0
    var best_d = 1e18
    for i in range(tl):
        var dx = Float64(env.track.track[i].center_x) - cx
        var dy = Float64(env.track.track[i].center_y) - cy
        var d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    var ti = (best_i + LOOKAHEAD) % tl
    var tx = Float64(env.track.track[ti].center_x)
    var ty = Float64(env.track.track[ti].center_y)
    var desired = atan2(ty - cy, tx - cx)
    # Box2D car faces +y at angle 0 → true heading = ca + π/2; steering is
    # inverted (action 1 turns the car clockwise), hence the leading minus.
    var err = -_wrap(desired - (ca + 0.5 * _PI))

    var gas_tick = spd < TARGET_SPEED and (step % 3 == 0)
    if spd < 2.5:
        return 3                                 # kickstart: a stopped car can't turn
    elif err > STEER_THRESH:
        return 3 if gas_tick else 1
    elif err < -STEER_THRESH:
        return 3 if gas_tick else 2
    elif spd < TARGET_SPEED:
        return 3
    else:
        return 0
