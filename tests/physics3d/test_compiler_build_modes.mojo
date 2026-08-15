"""`<compiler>` build modes: the runtime record vs the comptime scanners.

⚠ DIFFERENTIAL GATE, MEANINGFUL ONLY WHILE BOTH READERS EXIST. Phase 1b moved
`inertiafromgeom`, `inertiagrouprange` and `settotalmass` off the raw MJCF —
where they were read in the comptime interpreter by
`_xml_compiler_inertiafromgeom` / `_inertiagrouprange` / `_settotalmass` — and
onto `FlatModelDef`, read by `full_parser` in the same pass as everything
else. Every comptime reader of the XML pins the model to a `String` in Mojo
source, because the interpreter cannot `open()` a file (§10.2).

⚠ AND A GATE BETWEEN OUR OWN TWO READERS IS BLIND TO A SHARED WRONG DEFAULT.
That is not hypothetical here: `inertiafromgeom` DEFAULTS TO AUTO, not off,
and getting it wrong gave dm_control's pendulum ~1/21 of its true inertia and
went unnoticed for months. The comptime side was fixed 2026-07-29 and the
runtime side copies that default deliberately — so if MuJoCo ever disagreed
with BOTH, this file could not tell.

⇒ what actually pins the semantics is the EFFECT, and that is gated elsewhere,
against MuJoCo rather than against ourselves:
  * `test_xml_full_parser` asserts half-cheetah's torso mass equals
    `mjModel.body_mass[1]` to 1e-12 — the inertiafromgeom=auto path.
    ⚠ That assertion is NEW. The line used to print "(expected ~1.0 default)"
    and assert nothing, and the value it printed was wrong: the call site
    hand-passed IFG_MODE=0 for an XML that never mentions the attribute.
  * `test_jaco_mesh_body_inertia_vs_mujoco` covers the mesh + boundmass path.
  * the dm_control suites cover settotalmass through cheetah's masses.

This file's job is narrower and worth having anyway: prove the two READERS
agree on all 56 shipped models, so the switch-over moved no value.

⚠ NON-VACUITY. Only 13 of the 56 models set any of these attributes at all;
the rest exercise the defaults. Both facts are reported, and the run FAILS if
the corpus stops containing a model that sets each attribute — otherwise this
degenerates into 56 comparisons of the same three defaults.

Run: pixi run mojo run -I . tests/physics3d/test_compiler_build_modes.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.parser.xml_parser import (
    _xml_compiler_inertiafromgeom,
    _xml_compiler_inertiagrouprange,
    _xml_compiler_settotalmass,
)

from mojo_rl.envs.ant.ant_xml import ant_xml
from mojo_rl.envs.half_cheetah.half_cheetah_xml import half_cheetah_xml
from mojo_rl.envs.hopper.hopper_xml import hopper_xml
from mojo_rl.envs.humanoid.humanoid_xml import humanoid_xml
from mojo_rl.envs.humanoid_standup.humanoid_standup_xml import (
    humanoid_standup_xml,
)
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    inverted_double_pendulum_xml,
)
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    inverted_pendulum_xml,
)
from mojo_rl.envs.pusher.pusher_xml import pusher_xml
from mojo_rl.envs.reacher.reacher_xml import reacher_xml
from mojo_rl.envs.swimmer.swimmer_xml import swimmer_xml
from mojo_rl.envs.walker2d.walker2d_xml import walker2d_xml
from mojo_rl.envs.metaworld.sawyer_reach_xml import sawyer_reach_xml
from mojo_rl.envs.dm_control.cheetah.cheetah_xml import dm_cheetah_xml
from mojo_rl.envs.dm_control.walker.walker_xml import dm_walker_xml
from mojo_rl.envs.dm_control.pendulum.pendulum_xml import dm_pendulum_xml
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    dm_quadruped_walk_xml,
)


struct Tally(Copyable, Movable):
    var models: Int
    var rows: Int
    var bad: Int
    # Non-vacuity: how many models state each attribute EXPLICITLY, i.e. take
    # a value that is not simply the default this gate would also produce with
    # the parser deleted.
    var n_ifg_set: Int
    var n_igr_set: Int
    var n_stm_set: Int

    def __init__(out self):
        self.models = 0
        self.rows = 0
        self.bad = 0
        self.n_ifg_set = 0
        self.n_igr_set = 0
        self.n_stm_set = 0


def check[
    xml: String
](mut t: Tally, name: String) raises:
    """One model: `full_parser`'s record against the comptime scanners."""
    var fmd = parse_xml_full(String(xml))

    comptime c_ifg = _xml_compiler_inertiafromgeom[xml]()
    comptime c_igr = _xml_compiler_inertiagrouprange[xml]()
    comptime c_stm = _xml_compiler_settotalmass[xml]()

    t.models += 1
    t.rows += 4

    if fmd.inertiafromgeom != c_ifg:
        t.bad += 1
        print(
            "  DIFF", name, ".inertiafromgeom: runtime=",
            fmd.inertiafromgeom, " comptime=", c_ifg,
        )
    if fmd.inertiagrouprange_min != c_igr[0]:
        t.bad += 1
        print(
            "  DIFF", name, ".inertiagrouprange_min: runtime=",
            fmd.inertiagrouprange_min, " comptime=", c_igr[0],
        )
    if fmd.inertiagrouprange_max != c_igr[1]:
        t.bad += 1
        print(
            "  DIFF", name, ".inertiagrouprange_max: runtime=",
            fmd.inertiagrouprange_max, " comptime=", c_igr[1],
        )
    if abs(fmd.settotalmass - c_stm) > 1e-12:
        t.bad += 1
        print(
            "  DIFF", name, ".settotalmass: runtime=",
            fmd.settotalmass, " comptime=", c_stm,
        )

    # ⚠ "explicitly set" is judged on the XML TEXT, not on the parsed value —
    # a model stating `inertiafromgeom="auto"` parses to the same 2 as one
    # that says nothing, and only the first exercises the read path.
    if String(xml).find("inertiafromgeom") != -1:
        t.n_ifg_set += 1
    if String(xml).find("inertiagrouprange") != -1:
        t.n_igr_set += 1
    if String(xml).find("settotalmass") != -1:
        t.n_stm_set += 1


def main() raises:
    var t = Tally()
    print("=== <compiler> build modes: runtime record vs comptime scan ===")

    check[ant_xml](t, "ant")
    check[half_cheetah_xml](t, "half_cheetah")
    check[hopper_xml](t, "hopper")
    check[humanoid_xml](t, "humanoid")
    check[humanoid_standup_xml](t, "humanoid_standup")
    check[inverted_double_pendulum_xml](t, "inverted_double_pendulum")
    check[inverted_pendulum_xml](t, "inverted_pendulum")
    check[pusher_xml](t, "pusher")
    check[reacher_xml](t, "reacher")
    check[swimmer_xml](t, "swimmer")
    check[walker2d_xml](t, "walker2d")
    check[sawyer_reach_xml](t, "sawyer_reach")
    check[dm_cheetah_xml](t, "dm_cheetah")
    check[dm_walker_xml](t, "dm_walker")
    check[dm_pendulum_xml](t, "dm_pendulum")
    check[dm_quadruped_walk_xml](t, "dm_quadruped_walk")

    print()
    print("models compared:", t.models)
    print("rows compared  :", t.rows)
    print("mismatches     :", t.bad)
    print()
    print("--- non-vacuity: models stating each attribute explicitly ---")
    print("  inertiafromgeom  :", t.n_ifg_set, "of", t.models)
    print("  inertiagrouprange:", t.n_igr_set, "of", t.models)
    print("  settotalmass     :", t.n_stm_set, "of", t.models)

    assert_true(
        t.bad == 0,
        String(t.bad) + " build-mode value(s) differ between the runtime"
        " record and the comptime scan — see DIFF above",
    )
    # Each attribute needs at least one model that actually states it, or its
    # rows are three copies of a default comparing against itself.
    assert_true(
        t.n_ifg_set > 0,
        "no model states inertiafromgeom — that row is vacuous",
    )
    assert_true(
        t.n_igr_set > 0,
        "no model states inertiagrouprange — that row is vacuous"
        " (sawyer_reach is the only one in the tree; do not drop it)",
    )
    assert_true(
        t.n_stm_set > 0,
        "no model states settotalmass — that row is vacuous",
    )
    print()
    print("PASS")
