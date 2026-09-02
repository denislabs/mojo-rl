"""Generate `tests/physics3d/tree_block_goldens.mojo` — M's DIAGONAL BLOCKS.

⚠⚠ WHY THIS GATE EXISTS. `docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` rests on a
claim about the mass matrix's SHAPE: M couples a dof only with its tree
ancestors, so M's diagonal blocks are exactly the kinematic trees. Everything
downstream — a block-restricted LDL, a block-restricted `M^-1` — is only
correct if our partition is MuJoCo's partition. This dumps MuJoCo's, so ours
can be compared against it rather than against itself
(`feedback_a_gate_that_shares_its_reference_implementation_is_blind`).

TWO COLUMNS, AND THEY ARE GATED DIFFERENTLY.

  `adr`/`num` — `mjModel.tree_dofadr` / `tree_dofnum`. EXACT: a partition that
  disagrees with MuJoCo's is a wrong partition, in either direction.

  `compact` — is M restricted to the block DIAGONAL? Computed the way
  mujoco_warp computes it (`mujoco_warp/_src/io.py:186-190`): the block's nnz
  is `M_rowadr[last] + M_rownnz[last] - M_rowadr[first]`, and the block is
  compact when that equals its size. ⚠ ASYMMETRIC: our classifier is
  deliberately more conservative than MuJoCo's, so "we say DENSE, MuJoCo says
  COMPACT" is a PASS (merely slower) and "we say COMPACT, MuJoCo says DENSE" is
  a FAILURE (a silent wrong answer). The test enforces that asymmetry; this
  script only records the reference.

⚠ MuJoCo's own `dof_simplenum` is a CONTIGUOUS-SUFFIX run-length counted from
`nv-1` down (`user_model.cc:4100`), so `M_rownnz` — and therefore this column —
depends on where the simple bodies sit in the body list. That is a property of
the reference, not of the physics; it is one more reason the comparison is
one-directional.

Run: pixi run python scripts/dump_mujoco_trees.py
"""

import mujoco

# A spread, not a sample: single-tree articulated chains (the regression
# controls), multi-tree scenes with free-jointed props (the case the work is
# for), the four park scenes (the measurement this plan came from), and
# `point_mass` — COMPACT WITHOUT A FREE JOINT, two axis-aligned slides, which
# is the case a "free joint => compact" shortcut would get wrong from the
# other side.
MODELS = [
    "mojo_rl/envs/robots/assets/so101_park_k0.xml",
    "mojo_rl/envs/robots/assets/so101_park_k3.xml",
    "mojo_rl/envs/robots/assets/so101_park_k6.xml",
    "mojo_rl/envs/robots/assets/so101_park_k9.xml",
    "mojo_rl/envs/robots/assets/so_arm101.xml",
    "mojo_rl/envs/ant/assets/ant.xml",
    "mojo_rl/envs/humanoid/assets/humanoid.xml",
    "mojo_rl/envs/dm_control/assets/acrobot.xml",
    "mojo_rl/envs/dm_control/assets/ball_in_cup.xml",
    "mojo_rl/envs/dm_control/assets/cartpole3.xml",
    "mojo_rl/envs/dm_control/assets/cheetah.xml",
    "mojo_rl/envs/dm_control/assets/finger.xml",
    "mojo_rl/envs/dm_control/assets/fish.xml",
    "mojo_rl/envs/dm_control/assets/hopper.xml",
    "mojo_rl/envs/dm_control/assets/humanoid_cmu.xml",
    "mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml",
    "mojo_rl/envs/dm_control/assets/point_mass.xml",
    "mojo_rl/envs/dm_control/assets/quadruped_escape.xml",
    "mojo_rl/envs/dm_control/assets/quadruped_fetch.xml",
    "mojo_rl/envs/dm_control/assets/reacher.xml",
    "mojo_rl/envs/dm_control/assets/stacker_2.xml",
    "mojo_rl/envs/dm_control/assets/dog_fetch.xml",
    "references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml",
    "references/mujoco_menagerie-main/apptronik_apollo/scene.xml",
    "references/mujoco_menagerie-main/franka_fr3_v2/scene.xml",
    "references/mujoco_menagerie-main/aloha/scene.xml",
    "references/mujoco_menagerie-main/unitree_go2/scene.xml",
    "references/mujoco_menagerie-main/shadow_hand/scene_right.xml",
]


def mojo_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    rows = []
    for path in MODELS:
        m = mujoco.MjModel.from_xml_path(path)
        triples = []
        for adr, num in zip(m.tree_dofadr, m.tree_dofnum):
            adr, num = int(adr), int(num)
            # ⚠ `num > 0` IS WARP'S FILTER, NOT AN OPTIMISATION
            # (`io._m_blocks`). A body tree with no dofs — a fixed mount — gets
            # a `tree_*` row and no block of M.
            if num <= 0:
                continue
            last = adr + num - 1
            nnz = int(m.M_rowadr[last] + m.M_rownnz[last] - m.M_rowadr[adr])
            triples.append(f"{adr} {num} {1 if nnz == num else 0}")
        rows.append((path, int(m.nv), int(m.nC), " ".join(triples)))
        print(f"{path}: nv={m.nv} nC={m.nC} blocks={len(triples)}")

    out = [
        '"""MuJoCo 3.10.0\'s mass-matrix DIAGONAL BLOCKS — GENERATED.',
        "",
        "⚠ DO NOT EDIT. Regenerate with:",
        "    pixi run python scripts/dump_mujoco_trees.py",
        "",
        "`blk(i)` is a space-separated `dof_adr dof_num compact` triple per",
        "block, in MuJoCo's tree order. `compact` is 1 when M restricted to the",
        "block is DIAGONAL. See the script header for why `adr`/`num` are gated",
        "exactly and `compact` is gated ONE-DIRECTIONALLY.",
        "",
        "`nC` is MuJoCo's stored entry count — the SPARSE size our dense `nv*nv`",
        "is measured against.",
        '"""',
        "",
        "",
        "def blk_case_count() -> Int:",
        f"    return {len(rows)}",
        "",
        "",
        "def blk_path(i: Int) -> String:",
    ]
    for k, (p, _, _, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return String({mojo_str(p)})"]
    out += ['    return String("")', "", "", "def blk_nv(i: Int) -> Int:"]
    for k, (_, nv, _, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return {nv}"]
    out += ["    return 0", "", "", "def blk_nC(i: Int) -> Int:"]
    for k, (_, _, nc, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return {nc}"]
    out += ["    return 0", "", "", "def blk(i: Int) -> String:"]
    for k, (_, _, _, t) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return String({mojo_str(t)})"]
    out += ['    return String("")', ""]

    dst = "tests/physics3d/tree_block_goldens.mojo"
    with open(dst, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {dst} ({len(rows)} models)")


if __name__ == "__main__":
    main()
