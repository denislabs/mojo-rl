"""Diagnostic: print solver internals for one env step.

Shows contacts, Jacobians, forces to compare with MuJoCo.
"""

from random import seed
from math import sqrt
from envs.half_cheetah import HalfCheetah
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahParams as P,
)
from physics3d.types import Model, Data

comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    var action = env.ActionType()
    for i in range(6):
        action[i] = 1.0

    # Run 5 env steps to get contacts established
    for step in range(6):
        _ = env.step(action)
        print("Step", step + 1, "rootx=", env.data.qpos[0],
              "rootz=", env.data.qpos[1], "rooty=", env.data.qpos[2])
        print("  vx=", env.data.qvel[0], "vz=", env.data.qvel[1],
              "vy=", env.data.qvel[2])
        print("  num_contacts=", env.data.num_contacts)
        for c in range(env.data.num_contacts):
            var ct = env.data.contacts[c]
            print("  contact", c,
                  ": body_a=", ct.body_a, "body_b=", ct.body_b,
                  " dist=", ct.dist,
                  " pos=(", ct.pos_x, ct.pos_y, ct.pos_z, ")",
                  " n=(", ct.normal_x, ct.normal_y, ct.normal_z, ")",
                  " fn=", ct.force_n,
                  " ft1=", ct.force_t1, " ft2=", ct.force_t2)
        print()

    print("\n=== Model parameters ===")
    print("cone_type =", env.model.cone_type)
    print("impratio =", env.model.impratio)
    print("friction =", env.model.friction)
    print("solref_contact =", env.model.solref_contact[0], env.model.solref_contact[1])
    print("solimp_contact =", env.model.solimp_contact[0], env.model.solimp_contact[1], env.model.solimp_contact[2])
    print("DT = 0.01 (from def)")
    print("FRAME_SKIP = 5 (from def)")
