"""Multi-body car dynamics validation (CPU) against Box2D ground truth.

CarDynamicsMB models the car as 5 bodies + 4 revolute joints solved with the
physics2d sequential-impulse pipeline. This test checks the two canonical
scenarios whose Box2D ground truth we measured (gen_car_reference.py), on grass:

  A: full gas, no steer -> car accelerates DEAD STRAIGHT (x~0, omega~0).
     Box2D reference at step 120: vy ~= 64.6, x ~= 0.
  B: full gas + full right steer -> bounded donut (Box2D omega ~5.7), the car
     does NOT fly off or blow up.

The legacy single-body CarDynamics spun in place off-track; this is the
regression guard that the multi-body port stays faithful + stable.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from mojo_rl.physics2d import dtype
from mojo_rl.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA
from mojo_rl.physics2d.car import CarDynamicsMB
from mojo_rl.physics2d.car.constants import (
    FRICTION_LIMIT, GRASS_FRICTION, CTRL_STEERING, CTRL_GAS, CTRL_BRAKE,
)

# Compact validation layout (one car).
comptime BOFF = 0
comptime FOFF = BOFF + CarDynamicsMB.NUM_BODIES * 13          # bodies are 13 floats
comptime JOFF = FOFF + CarDynamicsMB.NUM_BODIES * 3
comptime ROFF = JOFF + CarDynamicsMB.NUM_JOINTS * 17          # joints are 17 floats
comptime COFF = ROFF + CarDynamicsMB.NUM_WHEELS
comptime SSZ = COFF + 3
comptime DT = Scalar[dtype](0.02)


def fail(name: String, msg: String) raises:
    raise Error(String("[", name, "] ", msg))


def run_scenario(
    steer: Float64, gas: Float64, nsteps: Int
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """Run one car from (0,0,0) on grass; return final hull (x,y,angle,vx,vy,omega).
    """
    var sbuf = List[Scalar[dtype]](capacity=SSZ)
    for _ in range(SSZ):
        sbuf.append(Scalar[dtype](0.0))
    var state = LayoutTensor[dtype, Layout.row_major(1, SSZ), MutAnyOrigin](
        sbuf.unsafe_ptr()
    )

    CarDynamicsMB.init_env[1, SSZ, BOFF, FOFF, JOFF, ROFF](
        0, state, Scalar[dtype](0.0), Scalar[dtype](0.0), Scalar[dtype](0.0)
    )
    state[0, COFF + CTRL_STEERING] = Scalar[dtype](steer)
    state[0, COFF + CTRL_GAS] = Scalar[dtype](gas)
    state[0, COFF + CTRL_BRAKE] = Scalar[dtype](0.0)

    var fric = Scalar[dtype](FRICTION_LIMIT) * Scalar[dtype](GRASS_FRICTION)
    for _ in range(nsteps):
        CarDynamicsMB.step_single_env[1, SSZ, BOFF, FOFF, JOFF, ROFF, COFF](
            0, state, fric, DT
        )

    return (
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_X])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_Y])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_ANGLE])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_VX])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_VY])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_OMEGA])),
    )


def main() raises:
    print("=== CarDynamicsMB validation (vs Box2D ground truth) ===")

    # --- Scenario A: full gas, no steer -> dead straight ------------------
    var a = run_scenario(0.0, 1.0, 120)
    var ax = a[0]
    var ay = a[1]
    var avy = a[4]
    var aom = a[5]
    print("A @120: x=", ax, " y=", ay, " vy=", avy, " omega=", aom)
    var ax_abs = ax if ax >= 0.0 else -ax
    var aom_abs = aom if aom >= 0.0 else -aom
    if ax_abs > 1.0:
        fail("A", String("car drifted sideways (|x|=", ax_abs, "), expected ~0"))
    if aom_abs > 0.5:
        fail("A", String("car yawed (|omega|=", aom_abs, "), expected ~0"))
    # Box2D reference vy@120 ~= 64.6; allow generous band.
    if avy < 55.0 or avy > 72.0:
        fail("A", String("forward speed ", avy, " off Box2D reference ~64.6"))

    # --- Scenario B: full gas + full right steer -> bounded donut ---------
    var b = run_scenario(1.0, 1.0, 120)
    var bx = b[0]
    var by = b[1]
    var bom = b[5]
    print("B @120: x=", bx, " y=", by, " omega=", bom)
    var bx_abs = bx if bx >= 0.0 else -bx
    var by_abs = by if by >= 0.0 else -by
    var bom_abs = bom if bom >= 0.0 else -bom
    # Box2D donut stays near origin with bounded yaw (~5.7); must not blow up.
    if bom_abs > 12.0:
        fail("B", String("yaw rate ", bom_abs, " unbounded (Box2D ~5.7)"))
    if bx_abs > 40.0 or by_abs > 40.0:
        fail("B", String("car flew off: (", bx, ",", by, ")"))

    print("=== PASS: multi-body car matches Box2D (straight stays straight, donut bounded) ===")
