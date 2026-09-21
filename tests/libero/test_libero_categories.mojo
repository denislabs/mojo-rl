"""LIBERO's registry as data — L1's gate on the TABLE and the IMPORTER.

    pixi run mojo run -I . tests/tasks/test_libero_categories.mojo

Pure text, no assets, no corpus. What could be wrong, and what each check
is for:

* **The shipped table does not parse**, or parses to fewer categories than
  the corpus uses. The 38 names below are the categories that appear in the
  130 files (`docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md` §1.2), spelled the
  way the `.bddl` spells them; a table missing one would make that file
  refuse at import.
* **⚠⚠ An articulation direction is wrong.** `open=lt:-0.14` for the
  cabinets, `open=gt:0.10` for `short_cabinet`: the table must carry BOTH
  directions and the evaluator must honour the field, not the sign. The
  check drives real qpos values through `Threshold.holds` for the four
  classes tasks use, against LIBERO's own comparisons.
* **A malformed record is accepted.** Unknown kind, unknown comparison, a
  workspace with an asset, a duplicate — each must RAISE by name.
* **The slot yaw does not round-trip.** `slot=...:x,y,z,yaw` is new in L1;
  a yaw that parses and is not written back is a fixture silently at 0.
* **`site_pos` reads the wrong site**, or an absent one returns a default.
* **`resolve_family` emits a placeholder**, skips the workspace, or places a
  fixture at the wrong z. A one-file fixture (an inline `.bddl` and an inline
  asset XML written to a temp dir) pins the arithmetic:
  z = z_offset + off_z - bottom_site_z.
"""

from std.os import makedirs
from std.time import perf_counter_ns

from noeira.tasks.bddl import parse_bddl
from noeira.tasks.libero_categories import (
    parse_libero_table, load_libero_table, threshold_holds, Threshold,
    DEFAULT_TABLE_PATH, CMP_LT, CMP_GT, CMP_GE, KIND_WORKSPACE,
)
from noeira.tasks.libero_import import (
    resolve_family, site_pos, family_todo_count,
)
from noeira.tasks.spec import (
    parse_family, parse_slot, SLOT_STATIC, SLOT_FREE,
)


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    """Checks run and failed. Both printed: 0 failed over 0 checks is not a pass."""

    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if not ok:
            self.failures += 1
            print("  FAIL:", what)


def refuses(mut ta: Tally, text: String, what: String):
    var raised = False
    try:
        var _t = parse_libero_table(text)
    except:
        raised = True
    ta.check(raised, "must refuse " + what)


comptime CORPUS_CATEGORIES = String(
    "kitchen_table wooden_cabinet flat_stove living_room_table table"
    " study_table wine_rack desk_caddy white_cabinet wooden_two_layer_shelf"
    " floor microwave akita_black_bowl plate cream_cheese butter ketchup"
    " alphabet_soup tomato_sauce basket chocolate_pudding red_coffee_mug"
    " porcelain_mug black_book wine_bottle white_yellow_mug orange_juice milk"
    " chefmate_8_frypan moka_pot wooden_tray cookies"
    " glazed_rim_porcelain_ramekin white_bowl yellow_book salad_dressing"
    " bbq_sauce new_salad_dressing"
)

comptime BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language open the drawer)
    (:regions
      (cabinet_region
          (:target main_table)
          (:ranges (
              (-0.1 0.2 0.1 0.4)
            )
          )
          (:yaw_rotation (
              (3.141592653589793 3.141592653589793)
            )
          )
      )
      (bowl_region
          (:target main_table)
          (:ranges (
              (-0.05 -0.05 0.05 0.05)
            )
          )
      )
      (top_region
          (:target wooden_cabinet_1)
      )
    )
  (:fixtures
    main_table - table
    wooden_cabinet_1 - wooden_cabinet
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
  )
  (:obj_of_interest
    wooden_cabinet_1
  )
  (:init
    (On akita_black_bowl_1 main_table_bowl_region)
    (On wooden_cabinet_1 main_table_cabinet_region)
  )
  (:goal
    (And (Open wooden_cabinet_1_top_region))
  )
)
"""

comptime ASSET_XML = """<mujoco model="fake">
  <worldbody>
    <body>
      <body name="object">
        <geom type="box" size="0.1 0.1 0.1" group="0"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.123" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.2" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0.03 0.04 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def main() raises:
    var ta = Tally()
    print("=" * 70)
    print("LIBERO categories + resolved import — L1")
    print("=" * 70)

    # ── the shipped table ────────────────────────────────────────────────
    var t = load_libero_table(DEFAULT_TABLE_PATH)
    var names = CORPUS_CATEGORIES.split(" ")
    var n_named = 0
    for i in range(len(names)):
        var n = String(names[i])
        if n.byte_length() == 0:
            continue
        n_named += 1
        ta.check(t.has_category(n), "table carries corpus category '" + n + "'")
    ta.check(n_named == 38, "38 corpus categories listed, got " + String(n_named))
    ta.check(len(t.problems) == 5, "5 problem classes, got " + String(len(t.problems)))
    ta.check(t.category("table").kind == KIND_WORKSPACE, "table is a workspace")
    ta.check(t.category("floor").kind == KIND_WORKSPACE, "floor is a workspace")
    ta.check(
        t.category("akita_black_bowl").asset
        == "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
        "bowl asset path",
    )
    ta.check(t.problem("LIBERO_Study_Tabletop_Manipulation").off_x == -0.2,
          "study workspace offset x")
    ta.check(t.problem("LIBERO_Floor_Manipulation").z_offset == -0.025,
          "floor z_offset")
    # L2: the robot variant and where it stands, quoted from robosuite/LIBERO
    var tp = t.problem("LIBERO_Tabletop_Manipulation")
    ta.check(tp.robot == "mounted" and tp.base_z == 0.912 and tp.base_x == -0.66,
          "tabletop: MountedPanda at (-0.66, 0, 0.912)")
    var lp = t.problem("LIBERO_Living_Room_Tabletop_Manipulation")
    ta.check(lp.robot == "on_the_ground" and lp.base_z == 0.42 and lp.base_x == -0.51,
          "living room: OnTheGroundPanda at (-0.51, 0, 0.42)")
    ta.check(t.problem("LIBERO_Floor_Manipulation").robot == "on_the_ground",
          "floor: OnTheGroundPanda")
    ta.check(tp.has_table and tp.table_sx == 1.0 and tp.table_sz == 0.05,
          "tabletop re-poses a 1.0 x 1.2 x 0.05 table")
    ta.check(not lp.has_table, "living room does not re-pose its table")
    ta.check(len(tp.cameras) == 2, "two camera overrides per problem")

    # ── articulation, against LIBERO's own comparisons ──────────────────
    var cab = t.category("wooden_cabinet")
    ta.check(cab.open.op == CMP_LT and cab.open.thr == -0.14, "cabinet open=lt:-0.14")
    ta.check(cab.close.op == CMP_GT and cab.close.thr == 0.0, "cabinet close=gt:0.0")
    ta.check(cab.open.holds(-0.15) and not cab.open.holds(-0.13),
          "cabinet: qpos -0.15 open, -0.13 not")
    ta.check(cab.close.holds(0.003) and not cab.close.holds(0.0),
          "cabinet: qpos 0.003 closed, 0.0 NOT (strict >)")
    var wc = t.category("white_cabinet")
    ta.check(wc.open.thr == cab.open.thr and wc.close.thr == cab.close.thr,
          "white cabinet shares the wooden cabinet's thresholds")
    var mw = t.category("microwave")
    ta.check(mw.open.holds(-1.5) and not mw.open.holds(-1.0),
          "microwave: -1.5 open, -1.0 not")
    ta.check(mw.close.holds(-0.001) and not mw.close.holds(-0.01),
          "microwave: -0.001 closed, -0.01 not")
    var st = t.category("flat_stove")
    ta.check(st.on.op == CMP_GE and st.on.holds(0.5) and not st.on.holds(0.49),
          "stove: on at 0.5 (>=), not at 0.49")
    ta.check(st.off.holds(-0.001) and not st.off.holds(0.0),
          "stove: off below 0.0 (strict <)")
    ta.check(not st.open.present() and not cab.on.present(),
          "a stove has no open=, a cabinet has no on=")
    # ⚠ THE COUNTER-EXAMPLE: positive direction on short_cabinet.
    var sc = t.category("short_cabinet")
    ta.check(sc.open.op == CMP_GT and sc.open.holds(0.12) and not sc.open.holds(0.05),
          "short_cabinet opens POSITIVE: 0.12 open, 0.05 not")
    ta.check(not cab.open.holds(0.12), "the same qpos is NOT open on wooden_cabinet")
    ta.check(threshold_holds(CMP_LT, -0.2, -0.14) and not threshold_holds(CMP_GT, -0.2, -0.14),
          "threshold_holds honours the op")

    # ── rotation ────────────────────────────────────────────────────────
    var bk = t.category("black_book")
    ta.check(bk.axis == "y" and bk.rot_lo < bk.rot_hi, "black_book: ranged yaw about y")
    ta.check(t.category("alphabet_soup").axis == "z", "alphabet_soup rotates about z")

    # ── refusals ────────────────────────────────────────────────────────
    refuses(ta, String("schema_version=1\ncategory=a\nkind=liquid\nasset=x.xml\n"),
            "an unknown kind")
    refuses(ta, String("schema_version=1\ncategory=a\nkind=asset\n"),
            "an asset category with no asset=")
    refuses(ta, String("schema_version=1\ncategory=a\nkind=workspace\nasset=x.xml\n"),
            "a workspace WITH an asset")
    refuses(ta, String("schema_version=1\ncategory=a\nkind=asset\nasset=x.xml\nopen=near:0.1\n"),
            "an unknown comparison")
    refuses(ta, String("schema_version=1\ncategory=a\nkind=asset\nasset=x.xml\n"
                   "category=a\nkind=asset\nasset=y.xml\n"),
            "a duplicate category")
    refuses(ta, String("schema_version=1\ncategory=a\nkind=asset\nasset=x.xml\ncolour=red\n"),
            "an unknown key")
    refuses(ta, String("schema_version=1\nproblem=P\nscene=s.xml\nworkspace=table\n"),
            "a problem whose workspace is not a declared workspace category")
    refuses(ta, String("category=a\nkind=asset\nasset=x.xml\n"), "no schema_version")
    refuses(ta, String("schema_version=1\ncategory=table\nkind=workspace\n"
                   "problem=P\nscene=s.xml\nworkspace=table\n"),
            "a problem with no robot= variant")
    refuses(ta, String("schema_version=1\ncategory=table\nkind=workspace\n"
                   "problem=P\nscene=s.xml\nworkspace=table\nrobot=flying\n"),
            "an unknown robot variant")
    var raised = False
    try:
        var _c = t.category("unicorn")
    except:
        raised = True
    ta.check(raised, "an unknown category lookup RAISES, not defaults")

    # ── slot yaw round-trips ────────────────────────────────────────────
    var s4 = parse_slot(String("cab:static:a.xml:0.1,0.2,0.3,3.141592653589793"))
    ta.check(s4.has_pose and s4.yaw == 3.141592653589793, "slot parses a 4th number as yaw")
    ta.check(s4.describe() == "cab:static:a.xml:0.1,0.2,0.3,3.141592653589793",
          "slot writes the yaw back; got " + s4.describe())
    var s3 = parse_slot(String("t:static:a.xml:0.25,0.0,0.01"))
    ta.check(s3.yaw == 0.0 and s3.describe() == "t:static:a.xml:0.25,0.0,0.01",
          "a 3-number pose is unchanged on the way out")
    var bad = False
    try:
        var _b = parse_slot(String("t:static:a.xml:1,2,3,4,5"))
    except:
        bad = True
    ta.check(bad, "a 5-number pose is refused")

    # ── site_pos ────────────────────────────────────────────────────────
    var bp = site_pos(String(ASSET_XML), String("bottom_site"))
    ta.check(bp[2] == -0.123, "site_pos reads bottom_site z, got " + String(bp[2]))
    ta.check(site_pos(String(ASSET_XML), String("top_site"))[2] == 0.2,
          "site_pos reads the named site, not the first one")
    var absent = False
    try:
        var _z = site_pos(String(ASSET_XML), String("no_such_site"))
    except:
        absent = True
    ta.check(absent, "an absent site RAISES")

    # ── resolve_family on a fixture, against the quoted arithmetic ─────
    var work = String("/tmp/noeira_libero_l1_") + String(perf_counter_ns())
    makedirs(work + "/articulated_objects", exist_ok=True)
    makedirs(work + "/stable_scanned_objects/akita_black_bowl", exist_ok=True)
    makedirs(work + "/scenes", exist_ok=True)
    with open(work + "/articulated_objects/wooden_cabinet.xml", "w") as f:
        f.write(ASSET_XML)
    with open(work + "/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", "w") as f:
        f.write(ASSET_XML)
    var p = parse_bddl(String(BDDL))
    var fam = resolve_family(p, t, work)
    ta.check(family_todo_count(fam) == 0, "no TODO placeholder survives")
    ta.check(fam.base == work + "/scenes/libero_tabletop_base_style.xml",
          "base is the problem's arena; got " + fam.base)
    # ⚠ THE THIRD SITE IS IN THE FIXTURE BECAUSE `resolve_family` NOW REQUIRES
    # IT. `asset_placement_geom` refuses an asset missing any of the three
    # robosuite sites rather than defaulting a resting height, and this fixture
    # is meant to stand in for a real one — all 93 pack assets declare them.
    var fslot = -1
    for i in range(len(fam.slots)):
        if fam.slots[i].kind == SLOT_FREE:
            fslot = i
    ta.check(fslot >= 0 and fam.slots[fslot].has_geom,
             "the free slot carries the asset's placement geometry")
    ta.check(
        fslot >= 0 and fam.slots[fslot].bottom_z == -0.123
        and fam.slots[fslot].top_z == 0.2
        and abs(fam.slots[fslot].h_radius - 0.05) < 1e-12,
        "bottom_z -0.123, top_z 0.2, h_radius hypot(0.03, 0.04) = 0.05; got "
        + (fam.slots[fslot].geom_describe() if fslot >= 0 else String("none")),
    )
    ta.check(len(fam.slots) == 2, "workspace fixture is NOT a slot: 2 slots, got "
          + String(len(fam.slots)))
    ta.check(fam.slots[0].name == "wooden_cabinet_1" and fam.slots[0].kind == SLOT_STATIC,
          "fixture first, static")
    ta.check(fam.slots[0].asset == work + "/articulated_objects/wooden_cabinet.xml",
          "fixture asset resolved from the table")
    # x,y = rect centre + workspace offset (0,0); z = -0.04 + 0.90 - (-0.123)
    ta.check(abs(fam.slots[0].px) < 1e-12 and abs(fam.slots[0].py - 0.3) < 1e-12,
          "fixture x,y = rect centre; got " + String(fam.slots[0].px) + ","
          + String(fam.slots[0].py))
    var want_z = -0.04 + 0.90 + 0.123
    ta.check(abs(fam.slots[0].pz - want_z) < 1e-12,
          "fixture z = z_offset + off_z - bottom_site_z; got "
          + String(fam.slots[0].pz) + " want " + String(want_z))
    ta.check(fam.slots[0].yaw == 3.141592653589793, "fixture yaw pi from the region")
    ta.check(fam.slots[1].name == "akita_black_bowl_1" and fam.slots[1].kind == SLOT_FREE,
          "object slot free")
    ta.check(len(fam.regions) == 2, "the two ranged regions travel, the site one does not")
    ta.check(fam.regions[0].site == "robot_table_top", "regions anchor on the workspace top site")
    # ⚠ the encode/parse fixpoint must survive the yaw
    var again = parse_family(fam.encode())
    ta.check(again.slots[0].yaw == fam.slots[0].yaw, "yaw survives encode -> parse")

    # L2: with a robot dir the family is the Panda + the arena slot
    var narrowed = 0
    var fam2 = resolve_family(
        p, t, work, narrowed, String("robots"), String("arenas"),
    )
    ta.check(fam2.base == "robots/panda_robosuite.xml", "mounted Panda is the base")
    ta.check(fam2.base_x == -0.66 and fam2.base_z == 0.912, "base_pos from the problem")
    ta.check(not fam2.floor and fam2.inherit_option, "no composer floor; option inherited")
    ta.check(len(fam2.base_qpos) == 9 and fam2.base_qpos[3] == -2.44459747,
          "base_qpos is LIBERO's init pose (9)")
    ta.check(len(fam2.slots) == 3 and fam2.slots[0].name == "arena"
          and fam2.slots[0].kind == SLOT_STATIC
          and fam2.slots[0].asset == "arenas/libero_tabletop_manipulation_arena.xml",
          "arena is slot 0, static, at the generated path")
    ta.check(fam2.slots[1].asset == "noeira/tasks/libero/objects/wooden_cabinet.xml",
          "fixture uses the generated copy (no joint)")
    ta.check(fam2.slots[2].asset == "noeira/tasks/libero/objects/akita_black_bowl_free.xml",
          "object uses the generated _free copy")
    ta.check(fam2.regions[0].site == "arena_workspace", "regions anchor on arena_workspace")
    ta.check(abs(fam2.slots[1].pz - want_z) < 1e-12,
          "fixture z unchanged by the robot mode (bottom_site read from the pack)")
    var again2 = parse_family(fam2.encode())
    ta.check(again2.base_z == 0.912 and not again2.floor and again2.inherit_option
          and len(again2.base_qpos) == 9, "L2 keys survive encode -> parse")

    # a ranged fixture yaw is REFUSED, not averaged
    var ranged = String(BDDL).replace(
        "(3.141592653589793 3.141592653589793)", "(0.0 1.5707963267948966)"
    )
    var ref_raised = False
    try:
        var _r = resolve_family(parse_bddl(ranged), t, work)
    except:
        ref_raised = True
    ta.check(ref_raised, "a fixture with a RANGED yaw is refused")

    # a category the table lacks is REFUSED by name
    var unknown = String(BDDL).replace("akita_black_bowl", "unicorn_bowl")
    var unk_raised = False
    try:
        var _u = resolve_family(parse_bddl(unknown), t, work)
    except:
        unk_raised = True
    ta.check(unk_raised, "an unknown category is refused at resolve")

    print()
    print("  passed:", ta.checks - ta.failures, " failed:", ta.failures)
    if ta.failures > 0:
        raise Error(String(ta.failures) + " checks failed")
    if ta.checks < 60:
        raise Error("only " + String(ta.checks) + " checks ran — vacuous")
    print("=== PASS ===")
