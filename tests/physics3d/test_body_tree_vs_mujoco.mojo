"""The kinematic TREE matches MuJoCo — not just its size. V2.7.

WHY THIS EXISTS
===============
`test_model_dims_vs_mujoco` gates the counts and `test_model_names_vs_mujoco`
gates the names. Both pass on a model whose tree is wrong, and one was:

    <body name="link_grasp_center" pos="0 0 0.23" euler="..."/>

is legal MJCF — a self-closing body, no children. `_fill_model`'s walk pushed a
level for it and had no `</body>` to pop, so it stayed one level deep **for the
rest of the document**, and every later SIBLING became a descendant.
`hello_robot_stretch_3`'s floor, its table and its two free-jointed objects all
ended up inside `base_link`; MuJoCo parents every one of them to the world.

⚠⚠ AND NBODY WAS RIGHT. So was ngeom, so was every name. The bodies were all
present and the TREE was a different robot — the exact shape of "one count
agreeing is not the model agreeing" that `<replicate>` produced from the other
direction. Three Menagerie models use a self-closing body: `stretch_3`,
`apptronik_apollo`, `franka_fr3_v2`.

⚠ IT WAS FOUND BY `studio.validate`, not by a parser gate: it reported a free
joint on a nested body and a plane inside a moving body — two rules MuJoCo
enforces — on a model MuJoCo loads. A false alarm from a validator IS a finding
when the validator's rules come from the reference.

TWO ARMS:
  1. every body's PARENT, by name, against `body_parentid`;
  2. every geom's OWNING BODY, by name, against `geom_bodyid` — the floor moved
     too, and a parent-only check would have missed which geoms went with it.

⚠ THE TABLE CARRIES BOTH THE AFFECTED MODELS AND CONTROLS. Only-affected would
pass a parser that special-cased them; only-controls is the state that let this
ship.

Regenerate: pixi run python scripts/dump_mujoco_body_tree.py
Run: pixi run mojo run -I . tests/physics3d/test_body_tree_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from tests.physics3d.body_tree_goldens import (
    tree_case_count, tree_path, tree_nbody, tree_ngeom, tree_parents,
    tree_geom_bodies,
)


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _split(s: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = String(s[byte = i : i + 1])
        if c == " ":
            out.append(cur)
            cur = String("")
        else:
            cur += c
    out.append(cur)
    return out^


def _body_label(names: List[String], i: Int) -> String:
    """`world` for body 0, the name, or `#id` when the source named nothing.

    ⚠ THE THREE ARE DISTINCT. Calling an unnamed body "world" is what the
    first golden did, and it reported four mismatches on a correct model.
    """
    if i == 0:
        return String("world")
    if i < len(names) and names[i].byte_length() > 0:
        return names[i]
    return String("#") + String(i)


def main() raises:
    var t = Tally()
    print("=== the kinematic tree vs MuJoCo 3.10.0 ===")

    var checked_parents = 0
    var checked_geoms = 0

    for c in range(tree_case_count()):
        var path = tree_path(c)
        print("---", path, "---")
        var fmd = parse_model_runtime(path)

        t.truth(len(fmd.bodies) + 1 == tree_nbody(c),
                String("nbody ", len(fmd.bodies) + 1, " (MuJoCo ",
                       tree_nbody(c), ")"))
        t.truth(len(fmd.geoms) == tree_ngeom(c),
                String("ngeom ", len(fmd.geoms), " (MuJoCo ", tree_ngeom(c),
                       ")"))

        # ── arm 1: every body's parent, BY NAME ───────────────────────────
        var want_p = _split(tree_parents(c))
        var bad_p = 0
        var n = len(fmd.bodies) + 1
        if len(want_p) != n:
            t.truth(False, String("golden has ", len(want_p),
                                  " parents for ", n, " bodies"))
        else:
            for b in range(1, n):
                var got = _body_label(fmd.body_names,
                                      fmd.bodies[b - 1].parent)
                checked_parents += 1
                if got != want_p[b]:
                    bad_p += 1
                    if bad_p <= 4:
                        print("       body", fmd.body_names[b],
                              ": parent", got, "but MuJoCo says", want_p[b])
            t.truth(bad_p == 0,
                    String(n - 1, " body parents match (", bad_p, " wrong)"))

        # ── arm 2: every geom's owning body ───────────────────────────────
        var want_g = _split(tree_geom_bodies(c))
        var bad_g = 0
        if len(want_g) != len(fmd.geoms):
            t.truth(False, String("golden has ", len(want_g),
                                  " geom owners for ", len(fmd.geoms),
                                  " geoms"))
        else:
            for g in range(len(fmd.geoms)):
                var bid = fmd.geoms[g].body_id
                var got = _body_label(fmd.body_names, bid)
                checked_geoms += 1
                if got != want_g[g]:
                    bad_g += 1
                    if bad_g <= 4:
                        print("       geom", g, fmd.geom_names[g],
                              ": body", got, "but MuJoCo says", want_g[g])
            t.truth(bad_g == 0,
                    String(len(fmd.geoms), " geom owners match (", bad_g,
                           " wrong)"))

    # ⚠ NON-VACUITY. A `_split` that returned nothing, or a table of empty
    # strings, would make every arm above trivially true.
    print("--- the comparison was not empty ---")
    t.truth(checked_parents > 100,
            String("body parents compared: ", checked_parents))
    t.truth(checked_geoms > 300,
            String("geom owners compared: ", checked_geoms))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_body_tree_vs_mujoco: " + String(t.fails) + " failed"
        )
