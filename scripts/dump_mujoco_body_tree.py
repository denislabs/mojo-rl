"""Generate `tests/physics3d/body_tree_goldens.mojo` — MuJoCo's KINEMATIC TREE.

⚠⚠ WHY A SEPARATE GOLDEN FROM THE COUNTS. `test_model_dims_vs_mujoco` and
`test_model_names_vs_mujoco` both pass on a model whose tree is wrong: a
self-closing `<body .../>` (legal MJCF, used by three Menagerie models) left our
walk one level deep for the rest of the document, so every LATER SIBLING became
a descendant instead. nbody was right. Every name was right. Every geom was
present. `hello_robot_stretch_3`'s floor, table and two free-jointed objects sat
inside `base_link`, and only `studio.validate` noticed — by reporting a free
joint on a nested body, which MuJoCo forbids, on a model MuJoCo loads.

One count agreeing is not the model agreeing. This dumps the PARENT of every
body and the OWNING BODY of every geom, by name.

Run: pixi run python scripts/dump_mujoco_body_tree.py
"""

import mujoco

# ⚠ THE FIRST THREE ARE THE ONES WITH A SELF-CLOSING `<body/>`; the rest are
# controls. A table of only-affected models would pass a parser that special-
# cased them, and a table of only-controls would not have caught this at all.
MODELS = [
    "references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml",
    "references/mujoco_menagerie-main/apptronik_apollo/scene.xml",
    "references/mujoco_menagerie-main/franka_fr3_v2/scene.xml",
    "references/mujoco_menagerie-main/aloha/scene.xml",
    "references/mujoco_menagerie-main/unitree_go2/scene.xml",
    "mojo_rl/envs/ant/assets/ant.xml",
    "mojo_rl/envs/humanoid/assets/humanoid.xml",
]


def mojo_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    rows = []
    for path in MODELS:
        m = mujoco.MjModel.from_xml_path(path)

        # ⚠⚠ "world" IS BODY 0, NOT "a body with no name". The first draft
        # wrote `name or "world"`, which called every UNNAMED body the world —
        # ant has four, and the gate then reported four false mismatches on a
        # model that was correct. An unnamed body gets `#id`, which is
        # distinguishable and still stable.
        def bname(i: int) -> str:
            if i == 0:
                return "world"
            return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                    or f"#{i}")

        parents = " ".join(bname(int(m.body_parentid[i]))
                           for i in range(m.nbody))
        geom_bodies = " ".join(bname(int(m.geom_bodyid[g]))
                               for g in range(m.ngeom))
        rows.append((path, m.nbody, m.ngeom, parents, geom_bodies))
        print(f"{path}: nbody={m.nbody} ngeom={m.ngeom}")

    out = [
        '"""MuJoCo 3.10.0\'s kinematic TREE, by name — GENERATED.',
        "",
        "⚠ DO NOT EDIT. Regenerate with:",
        "    pixi run python scripts/dump_mujoco_body_tree.py",
        "",
        "`parents(i)` is a space-separated list of each body's PARENT name",
        '("world" for the worldbody and for its direct children);',
        "`geom_bodies(i)` names the body each geom belongs to.",
        '"""',
        "",
        "",
        "def tree_case_count() -> Int:",
        f"    return {len(rows)}",
        "",
        "",
        "def tree_path(i: Int) -> String:",
    ]
    for k, (p, _, _, _, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return String({mojo_str(p)})"]
    out += ['    return String("")', "", "", "def tree_nbody(i: Int) -> Int:"]
    for k, (_, nb, _, _, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return {nb}"]
    out += ["    return 0", "", "", "def tree_ngeom(i: Int) -> Int:"]
    for k, (_, _, ng, _, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return {ng}"]
    out += ["    return 0", "", "", "def tree_parents(i: Int) -> String:"]
    for k, (_, _, _, par, _) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return String({mojo_str(par)})"]
    out += ['    return String("")', "", "", "def tree_geom_bodies(i: Int) -> String:"]
    for k, (_, _, _, _, gb) in enumerate(rows):
        out += [f"    if i == {k}:", f"        return String({mojo_str(gb)})"]
    out += ['    return String("")', ""]

    path = "tests/physics3d/body_tree_goldens.mojo"
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}: {len(rows)} models")


if __name__ == "__main__":
    main()
