#!/usr/bin/env python3
"""The MuJoCo half of P1c's gate — the ORACLE, not a second opinion.

Loads the composed scene that `tests/tasks/test_family_compose_vs_mujoco.mojo`
wrote and checks the two things only MuJoCo can settle:

1. **it loads at all** — MuJoCo compiles each attached asset separately and
   attaches the RESULT, where we splice TEXT. Two different routes to one
   `mjModel` is what makes this a gate rather than a restatement;
2. ⚠ **the parked slots touch NOTHING.** `ncon` at rest must equal the base
   scene's. A parked slot in contact makes every task in the family a
   different, slower problem, and the throughput curve would look fine.

    pixi run python tools/tasks/check_family.py /tmp/mojo_rl_family_compose.xml
"""
import sys
import mujoco


DEFAULT_SCENE = "mojo_rl/tasks/scenes/so101_tabletop.xml"
DEFAULT_FAMILY = "mojo_rl/tasks/families/so101_tabletop.family"


def read_family(path: str) -> dict:
    """The `.family` file, as far as this check needs it.

    ⚠ A SECOND, DELIBERATELY DUMB READER. `spec.mojo` is the real parser; this
    one exists so the oracle never reads OUR numbers — an oracle that shares
    the implementation it checks is not one
    (`feedback_a_gate_that_shares_its_reference_implementation_is_blind`).
    It is four lines and only ever reads `base=` and `slot=`.
    """
    base, slots = None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "base":
            base = v.strip()
        elif k.strip() == "slot":
            slots.append(v.strip().split(":"))
    return {"base": base, "slots": slots}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    fam = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FAMILY
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"  mujoco : nbody {m.nbody}  njnt {m.njnt}  nq {m.nq}  "
          f"nv {m.nv}  ngeom {m.ngeom}  nsite {m.nsite}")
    print(f"  contacts at rest: {d.ncon}")

    bad = 0
    # ⚠ THE PARKED-SLOT CHECK. Everything in this scene is parked far away or
    # is the robot standing on the floor, so a contact here is a slot touching
    # something it should not.
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"    contact {i}: {n1} vs {n2}  dist={c.dist:.4f}")
        bad += 1
    if bad:
        print("  FAIL: parked slots are in contact — see the park pose note in"
              " mojo_rl/tasks/family.mojo")
        return 1

    # ── the counts, derived INDEPENDENTLY through MuJoCo ─────────────────
    # ⚠ EVERY ASSET LOADED SEPARATELY AND SUMMED. This never reads our
    # parser's answer, so the two halves of this gate can only agree by both
    # being right. `nbody` sums as `1 + sum(nbody-1)` because every model
    # counts a world body and the composed scene has exactly one.
    fam_d = read_family(fam)
    exp = {"nbody": 1, "njnt": 0, "nq": 0, "nv": 0, "ngeom": 0}
    parts = [fam_d["base"]] + [s[2] for s in fam_d["slots"]]
    for a in parts:
        am = mujoco.MjModel.from_xml_path(a)
        exp["nbody"] += am.nbody - 1
        exp["njnt"] += am.njnt
        exp["nq"] += am.nq
        exp["nv"] += am.nv
        exp["ngeom"] += am.ngeom
    exp["ngeom"] += 1  # the scene's own floor, added by scene_from_base

    print(f"  expect : nbody {exp['nbody']}  njnt {exp['njnt']}  "
          f"nq {exp['nq']}  nv {exp['nv']}  ngeom {exp['ngeom']}"
          f"   ({len(parts)} assets summed independently)")
    for k, v in exp.items():
        got = getattr(m, k)
        if got != v:
            print(f"  FAIL: {k} is {got}, the assets sum to {v}")
            bad += 1
    if bad:
        print("  FAIL: the composed family does not carry every slot")
        return 1

    # ⚠ THE TREE COUNT IS THE COST MODEL, and it is why this file prints it.
    # Every free slot is its own kinematic tree, which is exactly the
    # structure the block-diagonal solver work exploits
    # (docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md).
    trees = [(int(a), int(n)) for a, n in zip(m.tree_dofadr, m.tree_dofnum)
             if n > 0]
    print(f"  kinematic trees: {len(trees)}  (dof blocks: "
          f"{[n for _, n in trees]})")
    print(f"  sparse M nC = {m.nC} vs dense nv*nv = {m.nv * m.nv}")
    print("  OK: MuJoCo loads the composed family and nothing is in contact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
