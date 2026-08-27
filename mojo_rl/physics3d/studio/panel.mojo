"""The studio's UI — menu bar, left options panel, right Explorer/Inspector.

One NON-GENERIC module. Every widget line compiles ONCE, which is the studio's
whole claim: its binary does not grow with the models it can open. That is
enforced by never naming a `Model`, a `Data` or a dims provider here — the
studio flattens what it wants shown into plain `List`s and this file returns
REQUESTS.

## ⚠ WHY THIS IS NOT `viewer_core.build_sidebar`

`run_view` is parameterised on `MODEL: ModelDefLike` + `CONFIG:
Phyics3dEnvConfig` and builds a `Phyics3dEnv[MODEL, CONFIG]`, so a runtime
model cannot call it. But the type signature is the SYMPTOM, not the reason.

**An env is the RL contract, and a scene is not a task.** `Phyics3dEnv` exists
to supply obs / reward / done / action space; a composed scene has none of
those. Routing the studio through it would mean inventing an observation and a
reward for a table with two cubes on it — fabricating the very thing the user
is supposed to author later. The env belongs to the BAKE phase, where a scene
becomes a task. See `docs/PHYSICS3D_STUDIO_PLAN.md` §5.1.

Taken from `viewer_core`: the shim entry points, the widget shapes, the drive
modes, and its **KEEP IT NON-GENERIC** rule.

## Layout, following MuJoCo's `simulate`

    ┌ menu bar ─────────────────────────────────────────────┐
    │ File   Simulation   View   Help                       │
    ├────────────┬──────────────────────────┬───────────────┤
    │ Options    │        scene             │ Explorer │Insp│
    │ (left)     │                          │  (right)      │
    └────────────┴──────────────────────────┴───────────────┘

⚠ THE MENU BAR CONSUMES SPACE AT y=0, so both panels start at
`ig_frame_height_with_spacing()`. Panels placed at 0 look CLIPPED, not
offset — the symptom points at the panel rather than at the bar.
"""

from std.os import listdir, stat
from std.pathlib import Path

from mojo_rl.render.imgui import (
    ig_begin_panel, ig_begin_window, ig_end, ig_begin_child, ig_end_child,
    ig_separator, ig_separator_text, ig_text, ig_text_disabled,
    ig_text_colored, ig_spacing, ig_same_line, ig_button, ig_small_button,
    ig_selectable, ig_checkbox, ig_combo, ig_slider_float, ig_tree_node,
    ig_tree_pop, ig_push_id_int, ig_pop_id, ig_set_next_item_width,
    ig_set_next_item_open, ig_progress_bar, ig_is_item_hovered, ig_set_tooltip,
    ig_input_text, ig_collapsing_header, TextBuffer,
    ig_begin_main_menu_bar, ig_end_main_menu_bar, ig_begin_menu, ig_end_menu,
    ig_menu_item, ig_frame_height_with_spacing,
    ig_begin_tab_bar, ig_end_tab_bar, ig_begin_tab_item, ig_end_tab_item,
    ig_columns, ig_next_column, ig_set_column_width, ig_drag_float,
    ig_toggle_button,
)
from .validate import Diagnostic, SEV_INFO, SEV_WARN, SEV_ERROR

comptime SIDEBAR_W: Float32 = 300.0
comptime RIGHT_W: Float32 = 340.0

comptime SORT_NAME: Int = 0
comptime SORT_SIZE: Int = 1
comptime SORT_TIME: Int = 2

comptime SEL_NONE: Int = 0
comptime SEL_BODY: Int = 1
comptime SEL_GEOM: Int = 2


@fieldwise_init
struct StudioPanel(Movable):
    """Everything the UI owns across frames, and across a MODEL SWAP.

    ⚠ WHAT SURVIVES A SWAP IS A DECISION, not an accident. Drive mode, scale,
    pause, the browser's directory and the visibility groups persist — opening
    a second robot to compare it against the first should not silently reset
    how it is being driven or send the file browser back to the root. The
    SELECTION does not persist: an index into the old model's geoms names a
    different shape in the new one, and a stale highlight on the wrong part is
    worse than none.
    """

    var sel_kind: Int
    var sel_index: Int
    """A GEOM index here is also an `rf.geom_*` index and a `fmd.geoms` index —
    `build_render_fields` walks `fmd.geoms` in order — which is what lets a
    ray-pick result be stored unchanged."""
    var drive: Int
    var scale: Float32
    var paused: Bool
    var show_sites: Bool
    var show_hud: Bool
    var group_shown: List[Bool]
    """Visibility per geom group 0-5, MuJoCo's `mjvOption.geomgroup`."""
    var filter: TextBuffer
    var name_buf: TextBuffer
    """The Structure section's name box — one buffer, because add and rename
    are never in flight at the same moment."""
    var joint_kind: Int
    """0 hinge, 1 slide, 2 ball, 3 free — the type the Add joint button uses."""
    var want_save: Int
    """0 none, 1 scene document, 2 flattened export, 3 the EDITED MODEL.

    ⚠ THREE, NOT TWO, AND THE THIRD IS THE ONE V2 NEEDS. The scene document
    keeps the composition and cannot express a robot's edited tree; the
    flattened export goes through `writer.to_mjcf`, which REFUSES a model
    carrying `<tendon>`, `<equality>` or `<keyframe>` rather than drop them
    silently — so a structurally edited robot has no way out through either.
    Option 3 writes the studio's live document verbatim. It is already valid
    MJCF (`test_structural_edit` has MuJoCo load exactly this text after every
    edit), it is lossless by construction, and it is what is being simulated.
    """
    var want_undo: Int
    """0 none, 1 undo, 2 redo — consumed by the studio each frame."""
    var browser_open: Bool
    var browser_dir: String
    var browser_path: TextBuffer
    var browser_sort: Int
    """Which column the listing is ordered by — `SORT_NAME` / `SORT_SIZE` /
    `SORT_TIME`. On the panel, not local to the browser, so the choice
    survives closing and reopening the window (the directory already does)."""
    var browser_desc: Bool
    var recent: List[String]
    var gizmo_mode: Int
    """0 off, 1 move, 2 turn — `studio/gizmo.mojo`'s `GIZMO_*`.

    ⚠ THE PANEL DOES NOT IMPORT THOSE ALIASES, deliberately. `gizmo.mojo`
    names `FlatModelDef`, and one import of it here would make every widget
    line in this file generic — the same rule that keeps `_record` in the
    studio rather than in `ui_inspector`. The values are the contract; the
    studio is the side that knows their names."""
    var gizmo_world: Bool
    """False = the element's own axes, True = world axes.

    Defaults to LOCAL, because MJCF stores a local frame: rotating about the
    part's own axis is the edit whose numbers a user can predict."""
    var gizmo_snap: Float32
    """0 = off. Metres for move, DEGREES for turn — ImGuizmo's convention,
    which is why one scalar covers both and why the label has to say which."""

    def __init__(out self, drive: Int, scale: Float64, start_dir: String):
        self.sel_kind = SEL_NONE
        self.sel_index = -1
        self.drive = drive
        self.scale = Float32(scale)
        self.paused = False
        self.show_sites = False
        self.show_hud = False
        self.group_shown = List[Bool]()
        for g in range(6):
            # MuJoCo's default `mjvOption.geomgroup` is 1 for groups 0-2 and 0
            # for the rest (`mjv_defaultOption`), and that default is
            # load-bearing: dm_control's dog parks its collision capsules in
            # group 3 and its 162 bone meshes in group 5, so showing
            # everything draws the collision proxy as if it were the model.
            self.group_shown.append(g < 3)
        self.filter = TextBuffer()
        self.name_buf = TextBuffer()
        self.joint_kind = 0
        self.want_save = 0
        self.want_undo = 0
        self.browser_open = False
        self.browser_dir = start_dir
        self.browser_path = TextBuffer()
        self.browser_sort = SORT_NAME
        self.browser_desc = False
        self.recent = List[String]()
        self.gizmo_mode = 0
        self.gizmo_world = False
        self.gizmo_snap = 0.0

    def remember(mut self, path: String):
        """Push onto the recent list, most recent first, without duplicates."""
        var keep = List[String]()
        keep.append(path)
        for r in self.recent:
            if r != path and len(keep) < 8:
                keep.append(r)
        self.recent = keep^

    def clear_selection(mut self):
        self.sel_kind = SEL_NONE
        self.sel_index = -1


@fieldwise_init
struct PanelOut(Copyable, Movable):
    """What the UI ASKS the studio to do. No side effects on the model.

    ⚠ REQUESTS, NOT MUTATIONS, and not only for the non-generic rule. A panel
    that reset the sim itself would do it MID-FRAME, between the step and the
    draw, and the frame would render a pose that never existed. The studio
    applies these at the top of the next iteration.
    """

    var reset: Bool
    var step_once: Bool
    var reframe: Bool
    var quit: Bool
    var add_prop: Int
    """A prop KIND to drop into the scene, or -1. The studio owns the
    SceneDoc and the rebuild; the panel only asks."""
    var dup_prop: Bool
    var del_prop: Bool

    var new_name: String
    """The name typed in the Structure box — "" when it is empty."""
    var add_body_here: Bool
    """Add a child body to the selected body (or to the world when nothing
    is selected)."""
    var add_joint_here: Int
    """-1 none, else a joint TYPE index to add to the selected body."""
    var rename_here: Bool
    """Rename the selected element to `new_name`."""
    var reparent_here: Bool
    """Move the selected BODY under the body in the name box ("" = world)."""

    var del_element: Bool
    """Delete the SELECTED body or geom from the model itself — V2.1.

    ⚠ NOT `del_prop`, AND THE DIFFERENCE IS THE WHOLE OF V1 vs V2.
    `del_prop` removes an INSTANCE from the scene document — a `<frame>` and
    an `<attach>` — and never touches an asset. This edits the robot's own
    tree, so it goes through `structure.delete_element`, prunes every
    reference to what it removed, and can leave a model MuJoCo refuses.
    """

    var edit_field: Int
    """Which inspector field was dragged this frame, or -1. The STUDIO applies
    it — see `PanelOut`'s note on requests: the panel must not touch a `Model`
    (it would stop compiling once), and an edit applied mid-frame between the
    step and the draw would render a pose that never existed."""
    var edit_value: Float64

    var open_path: String
    """A model to LOAD, or "" for none. The swap is the studio's to perform:
    it owns the renderer handoff and every container that has to be rebuilt."""
    var browser_rows: Int
    """How many rows the file browser listed this frame, -1 when it is closed.

    ⚠ IT EXISTS SO THE HEADLESS SMOKE IS NOT VACUOUS. Drawing the browser
    proves the ImGui call sequence does not assert; it proves nothing about
    whether the listing found anything, and "did not crash" reads identically
    on a directory that enumerated zero entries."""

    def __init__(out self):
        self.reset = False
        self.step_once = False
        self.reframe = False
        self.quit = False
        self.add_prop = -1
        self.dup_prop = False
        self.del_prop = False
        self.del_element = False
        self.new_name = String("")
        self.add_body_here = False
        self.add_joint_here = -1
        self.rename_here = False
        self.reparent_here = False
        self.edit_field = -1
        self.edit_value = 0.0
        self.open_path = String("")
        self.browser_rows = -1


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════


def _label(names: List[String], i: Int, kind: String) -> String:
    """A name, or a bracketed index when the MJCF gave none.

    ⚠ THE BRACKETS ARE THE TELL. `FlatModelDef` stores "" for an unnamed
    element rather than synthesising `geom3`, so an export cannot claim a name
    the source never had — MuJoCo itself returns NULL for walker2d's six
    actuators. The invention therefore happens HERE, at display, where it is
    visibly a display choice.
    """
    if i >= 0 and i < len(names) and names[i].byte_length() > 0:
        return names[i].copy()
    return String("<", kind, " ", i, ">")


def _contains(hay: String, needle: String) -> Bool:
    if needle.byte_length() == 0:
        return True
    return hay.find(needle) != -1


def _f(v: Float64, places: Int = 4) -> String:
    """Fixed-point, for DISPLAY. Never for a file — see `SceneDoc.to_mjcf`.

    ⚠⚠ THE MAGNITUDE IS FORMATTED AND THE SIGN PREPENDED, because Mojo's `//`
    and `%` FLOOR. The obvious version splits the scaled integer directly, and
    for -0.3 that is `-3000 // 10000 == -1` with `-3000 % 10000 == 7000`:
    **it printed "-1.7000"**. Every negative coordinate in the inspector was
    wrong, and wrong in a way that still looks like a number — a body at
    x = -0.3 read as -1.7. Found only by writing a scene to disk and comparing
    the file with what was asked for.
    """
    var neg = v < 0
    var mag = -v if neg else v
    var mul = 1.0
    for _ in range(places):
        mul *= 10.0
    var scaled = Int(mag * mul + 0.5)
    var whole = scaled // Int(mul)
    var frac = scaled % Int(mul)
    var f = String(frac)
    while f.byte_length() < places:
        f = "0" + f
    return ("-" if neg else "") + String(whole) + "." + f


def _row(k: String, v: String) raises:
    """One key/value line of the inspector, in the two-column layout."""
    ig_text_disabled(k)
    ig_next_column()
    ig_text(v)
    ig_next_column()


def _parent_dir(p: String) -> String:
    var cut = p.rfind("/")
    if cut <= 0:
        return String(".")
    return String(p[byte=0:cut])


# ═══════════════════════════════════════════════════════════════════════════
# menu bar
# ═══════════════════════════════════════════════════════════════════════════


def ui_menu_bar(
    mut p: StudioPanel, mut out: PanelOut,
    can_undo: Bool = False, can_redo: Bool = False,
    undo_label: String = String(""), redo_label: String = String(""),
) raises -> Float32:
    """The top strip. Returns the height it took, which both panels must skip.

    ⚠ RETURNING THE HEIGHT rather than hardcoding it: the bar's height follows
    the font scale, and a constant here would leave a gap or a clipped row the
    first time anyone changes it.
    """
    var h = Float32(0.0)
    if ig_begin_main_menu_bar():
        h = ig_frame_height_with_spacing()
        if ig_begin_menu(String("File")):
            if ig_menu_item(String("Open model...")):
                p.browser_open = True
            if ig_begin_menu(String("Open recent")):
                if len(p.recent) == 0:
                    ig_text_disabled(String("nothing yet"))
                for i in range(len(p.recent)):
                    ig_push_id_int(i)
                    if ig_menu_item(p.recent[i]):
                        out.open_path = p.recent[i]
                    ig_pop_id()
                ig_end_menu()
            ig_separator()
            if ig_menu_item(String("Save scene")) and p.want_save == 0:
                p.want_save = 1
            if ig_menu_item(String("Save edited model")) and p.want_save == 0:
                p.want_save = 3
            if ig_menu_item(String("Export flattened MJCF")) \
                    and p.want_save == 0:
                p.want_save = 2
            ig_separator()
            if ig_menu_item(String("Quit")):
                out.quit = True
            ig_end_menu()
        if ig_begin_menu(String("Edit")):
            # ⚠ ENABLED FROM THE STACK, AND LABELLED WITH WHAT IT WILL TAKE
            # BACK. `can_undo`/`can_redo` were passed in from the first
            # version of this panel and read by nothing: both items were
            # always live, so clicking Undo on a fresh file looked like an
            # undo that did nothing rather than like an empty stack. Naming
            # the edit matters more here than in most editors — a structural
            # undo can restore a whole subtree, and "Undo deleted 'bthigh'"
            # is the difference between confidence and a guess.
            var ul = String("Undo")
            if undo_label.byte_length() > 0:
                ul += " " + undo_label
            var rl = String("Redo")
            if redo_label.byte_length() > 0:
                rl += " " + redo_label
            if ig_menu_item(ul, String(""), False, can_undo) \
                    and p.want_undo == 0:
                p.want_undo = 1
            if ig_menu_item(rl, String(""), False, can_redo) \
                    and p.want_undo == 0:
                p.want_undo = 2
            ig_end_menu()
        if ig_begin_menu(String("Simulation")):
            if ig_menu_item(String("Reset")):
                out.reset = True
            if ig_menu_item(String("Pause"), String(""), p.paused):
                p.paused = not p.paused
            if ig_menu_item(String("Step"), String(""), False, p.paused):
                out.step_once = True
            ig_end_menu()
        if ig_begin_menu(String("View")):
            if ig_menu_item(String("Reframe camera")):
                out.reframe = True
            _ = ig_checkbox(String("Sites"), p.show_sites)
            _ = ig_checkbox(String("Built-in HUD"), p.show_hud)
            ig_separator()
            ig_text_disabled(String("visibility groups"))
            for g in range(6):
                ig_push_id_int(g)
                var v = p.group_shown[g]
                if ig_checkbox(String("group ", g), v):
                    p.group_shown[g] = v
                ig_pop_id()
            ig_end_menu()
        if ig_begin_menu(String("Help")):
            ig_text_disabled(String("drag: orbit"))
            ig_text_disabled(String("shift+drag: pan"))
            ig_text_disabled(String("wheel: zoom"))
            ig_text_disabled(String("1-9: model cameras, 0: free"))
            ig_text_disabled(String("click a geom to select it"))
            ig_end_menu()
        ig_end_main_menu_bar()
    return h


# ═══════════════════════════════════════════════════════════════════════════
# file browser
# ═══════════════════════════════════════════════════════════════════════════


struct _Entry(Copyable, Movable):
    """One row of the browser: what `listdir` gave plus what `stat` knows."""

    var name: String
    var size: Int
    var mtime: Int
    """Seconds since the epoch, UTC. 0 when `stat` refused."""
    var is_dir: Bool

    def __init__(out self, name: String, size: Int, mtime: Int, is_dir: Bool):
        self.name = name
        self.size = size
        self.mtime = mtime
        self.is_dir = is_dir


def _fold(s: String) -> String:
    """ASCII lowercase, for a CASE-INSENSITIVE name order.

    ⚠ FINDER IS CASE-INSENSITIVE AND BYTE ORDER IS NOT. Raw `<` puts every
    capitalised name before every lowercase one, so `Panda.xml` sorts above
    `ant.xml` — which reads as "not sorted" rather than as "sorted by a rule
    you did not expect". Names here are ASCII paths; a full Unicode fold would
    be a different and much larger promise.
    """
    var out = String("")
    for i in range(s.byte_length()):
        var c = ord(String(s[byte = i : i + 1]))
        if c >= 65 and c <= 90:
            out += chr(c + 32)
        else:
            out += chr(c)
    return out^


def _p2(n: Int) -> String:
    return String("0", n) if n < 10 else String(n)


def _fmt_time(secs: Int) -> String:
    """`YYYY-MM-DD HH:MM`, UTC, from a Unix timestamp.

    ⚠ UTC, AND THE COLUMN SAYS SO. There is no timezone database reachable
    from here, and quietly showing a UTC clock under a heading that reads
    "modified" would have people comparing it against Finder and finding it
    hours out.

    Days-to-civil is Howard Hinnant's `civil_from_days` — exact for the whole
    range, no lookup tables, and the same algorithm `<chrono>` uses.
    """
    if secs <= 0:
        return String("--")
    var days = secs // 86400
    var rem = secs - days * 86400
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1
    return String(y, "-", _p2(m), "-", _p2(d), " ",
                  _p2(rem // 3600), ":", _p2((rem % 3600) // 60))


def _fmt_size(n: Int, is_dir: Bool) -> String:
    if is_dir:
        return String("--")
    if n < 1024:
        return String(n, " B")
    if n < 1024 * 1024:
        return String((n + 512) // 1024, " KB")
    return String((n + 524288) // (1024 * 1024), " MB")


def _entry_before(a: _Entry, b: _Entry, key: Int, desc: Bool) -> Bool:
    """Is `a` ordered before `b`?

    ⚠⚠ DIRECTORIES FIRST, ALWAYS, AND THE DIRECTION DOES NOT FLIP THEM. A
    reversed sort that also moved the folders to the bottom would put the
    `.. up` target and the subdirectories below a hundred files — the browser
    would look broken rather than reversed. Finder's "Folders on top" is the
    same decision.

    ⚠ AND NAME IS THE TIE-BREAK FOR EVERY KEY. Two files of the same size in
    arbitrary order re-shuffle whenever the OS hands `listdir` a different
    sequence, and a listing that changes under a stationary cursor is worse
    than one sorted by something you did not pick.
    """
    if a.is_dir != b.is_dir:
        return a.is_dir
    var na = _fold(a.name)
    var nb = _fold(b.name)
    if key == SORT_SIZE and a.size != b.size:
        return (a.size > b.size) if desc else (a.size < b.size)
    if key == SORT_TIME and a.mtime != b.mtime:
        return (a.mtime > b.mtime) if desc else (a.mtime < b.mtime)
    if na == nb:
        return False
    return (na > nb) if (desc and key == SORT_NAME) else (na < nb)


def _sort_entries(mut e: List[_Entry], key: Int, desc: Bool):
    """Insertion sort. ⚠ O(n^2) ON PURPOSE — the directories this browser
    opens hold tens of entries, and a hand-rolled comparator sort would be
    more code to be wrong in than the thing it replaces."""
    for i in range(1, len(e)):
        var j = i
        while j > 0 and _entry_before(e[j], e[j - 1], key, desc):
            e.swap_elements(j, j - 1)
            j -= 1


def _sort_header(mut p: StudioPanel, label: String, key: Int) raises:
    """One clickable column heading. Marks the active key and its direction.

    ⚠ ASCII ARROWS. ImGui's default font atlas is Latin-1; a `\u25b2` renders
    as a box, which looks like a missing glyph rather than like a sort marker.

    ⚠ SIZE AND TIME OPEN **DESCENDING**, NAME ASCENDING. Picking "size" almost
    always means "show me the big ones" and picking "modified" means "what did
    I touch last"; opening those ascending makes the first click useless and
    the second click the real one.
    """
    var l = label.copy()
    if p.browser_sort == key:
        l += " v" if p.browser_desc else " ^"
    if ig_small_button(l):
        if p.browser_sort == key:
            p.browser_desc = not p.browser_desc
        else:
            p.browser_sort = key
            p.browser_desc = key != SORT_NAME


def ui_file_browser(mut p: StudioPanel, mut out: PanelOut) raises:
    """Pick an MJCF without relaunching. A floating window, not a panel.

    ⚠ IT IS A DIRECTORY LISTING, NOT A NATIVE DIALOG. ImGui has no file
    dialog and SDL3's is a platform callback that would have to cross the FFI
    boundary and re-enter the frame loop — a much larger commitment than
    `listdir` plus two filters, for a tool whose models all live in this
    repository.

    ⚠ THE TYPED PATH IS THE ESCAPE HATCH and it is deliberately FIRST: a
    browser that can only reach what it can enumerate is useless the moment
    someone keeps their models elsewhere.
    """
    if not p.browser_open:
        return
    if not ig_begin_window(String("open model"), 340.0, 120.0, 560.0, 460.0):
        ig_end()
        return

    ig_text_disabled(p.browser_dir)
    ig_set_next_item_width(-70.0)
    _ = ig_input_text(String("##path"), p.browser_path,
                      String("path to an .xml, or pick below"))
    ig_same_line()
    if ig_button(String("open")):
        var typed = p.browser_path.value()
        if typed.byte_length() > 0:
            out.open_path = typed
            p.browser_open = False

    ig_separator()
    if ig_button(String(".. up")):
        p.browser_dir = _parent_dir(p.browser_dir)

    # ── the column headings, which are also the sort control ─────────────
    # ⚠ THE HEADER ROW IS OUTSIDE THE SCROLL CHILD so it stays put — with 65
    # Menagerie directories the sort control has to be reachable after
    # scrolling. The cost is that its content region starts a few pixels
    # inside the child's, so `border=False` here and `True` below: one set of
    # vertical rules, drawn where the rows are, rather than two sets a few
    # pixels apart. The horizontal `ig_separator` is what reads as the header
    # rule.
    ig_columns(3, False)
    ig_set_column_width(0, 300.0)
    ig_set_column_width(1, 80.0)
    _sort_header(p, String("name"), SORT_NAME)
    ig_next_column()
    _sort_header(p, String("size"), SORT_SIZE)
    ig_next_column()
    _sort_header(p, String("modified (UTC)"), SORT_TIME)
    ig_next_column()
    ig_columns(1)
    ig_separator()

    if ig_begin_child(String("entries"), 0.0, 276.0, True):
        # ⚠ ONE `listdir` PER FRAME, and it is fine: this window is open for
        # seconds, not for a session, and caching it would need invalidation
        # the moment the user creates a file. Measured directories here are
        # tens of entries. The `stat` per row rides along on the same
        # reasoning — `is_dir` was already one syscall per entry per frame.
        var rows = List[_Entry]()
        var readable = True
        try:
            var entries = listdir(Path(p.browser_dir))
            for e in entries:
                var nm = String(e)
                if nm.startswith("."):
                    continue
                var full = p.browser_dir + "/" + nm
                var is_x = nm.endswith(".xml")
                var is_d = False
                if not is_x:
                    is_d = Path(full).is_dir()
                    if not is_d:
                        continue
                var sz: Int
                var mt: Int
                try:
                    var st = stat(Path(full))
                    sz = Int(st.st_size)
                    # ⚠⚠ THE FIELD IS `st_mtimespec`, NOT `st_mtime`.
                    # `stat_result.write_to` PRINTS the label "st_mtime=",
                    # so the repr names an attribute that does not exist —
                    # reading the printout and typing what it said is a
                    # compile error with a confusing message.
                    mt = Int(st.st_mtimespec.tv_sec)
                except:
                    # ⚠ A ROW THAT CANNOT BE STATTED STILL LISTS. A broken
                    # symlink or a permission hole must not remove a file
                    # from a browser whose whole job is to show what is there.
                    sz = 0
                    mt = 0
                rows.append(_Entry(nm, sz, mt, is_d))
        except:
            readable = False
            ig_text_disabled(String("cannot read this directory"))

        _sort_entries(rows, p.browser_sort, p.browser_desc)
        out.browser_rows = len(rows)

        ig_columns(3, True)
        ig_set_column_width(0, 300.0)
        ig_set_column_width(1, 80.0)
        for i in range(len(rows)):
            ig_push_id_int(i)
            var label = String("[dir] ", rows[i].name) if rows[i].is_dir \
                else rows[i].name.copy()
            if ig_selectable(label):
                if rows[i].is_dir:
                    p.browser_dir = p.browser_dir + "/" + rows[i].name
                else:
                    out.open_path = p.browser_dir + "/" + rows[i].name
                    p.browser_open = False
            ig_next_column()
            ig_text_disabled(_fmt_size(rows[i].size, rows[i].is_dir))
            ig_next_column()
            ig_text_disabled(_fmt_time(rows[i].mtime))
            ig_next_column()
            ig_pop_id()
        ig_columns(1)
        if readable and len(rows) == 0:
            ig_text_disabled(String("no .xml here"))
    ig_end_child()

    if ig_button(String("cancel")):
        p.browser_open = False
    ig_end()


# ═══════════════════════════════════════════════════════════════════════════
# left: options
# ═══════════════════════════════════════════════════════════════════════════


def ui_options(
    mut p: StudioPanel,
    mut out: PanelOut,
    path: String,
    step_i: Int,
    ncon: Int,
    contact_budget: Int,
    step_us: Float64,
    y0: Float32,
    h: Float32,
    gizmo_hint: String,
) raises:
    ig_begin_panel(String("Options"), 0.0, y0, SIDEBAR_W, h)

    if ig_collapsing_header(String("Simulation"), True):
        if ig_button(String("Reset"), 88.0):
            out.reset = True
        ig_same_line()
        if ig_button(String("Pause") if not p.paused else String("Run"),
                     88.0):
            p.paused = not p.paused
        ig_same_line()
        if ig_button(String("Step"), 88.0):
            out.step_once = True
        ig_text_disabled(String("step ", step_i, "   ", _f(step_us, 1),
                                " us"))

        # ⚠⚠ THE CONTACT BUDGET IS A BAR, NOT A LOG LINE. Overflowing
        # `max_contacts` DROPS contacts with no error and no crash — the model
        # just gets quietly softer — and a scene composer raises the count
        # every time a prop lands. §1.2 names this as one of the two
        # silent-failure surfaces the composer creates, so the number that
        # would warn you lives where the eye already is.
        var frac = Float32(0.0)
        if contact_budget > 0:
            frac = Float32(ncon) / Float32(contact_budget)
        ig_progress_bar(frac, -1.0, 0.0,
                        String("contacts ", ncon, "/", contact_budget))
        if ig_is_item_hovered():
            ig_set_tooltip(String(
                "contacts in flight vs the workspace budget.\n"
                "at 100% the solver DROPS contacts silently."
            ))

    if ig_collapsing_header(String("Drive"), True):
        var modes = List[String]()
        modes.append(String("zero"))
        modes.append(String("random"))
        modes.append(String("sweep"))
        var cur = Int32(p.drive)
        ig_set_next_item_width(-60.0)
        if ig_combo(String("mode"), cur, modes):
            p.drive = Int(cur)
        ig_set_next_item_width(-60.0)
        _ = ig_slider_float(String("scale"), p.scale, 0.0, 1.0)

    # ── the transform gizmo — V2.10 ──────────────────────────────────────
    # ⚠ MODAL, AND OFF BY DEFAULT. A gizmo permanently on top of the
    # selection swallows clicks meant for the geom BEHIND it — the pick and
    # the gizmo hit-test the same pixels, and the gizmo wins by design. Off
    # is therefore the state in which the studio still behaves the way S1
    # shipped it, and the mode buttons are the opt-in.
    if ig_collapsing_header(String("Transform"), True):
        if ig_toggle_button(String("off"), p.gizmo_mode == 0, 58.0):
            p.gizmo_mode = 0
        ig_same_line()
        if ig_toggle_button(String("move"), p.gizmo_mode == 1, 58.0):
            p.gizmo_mode = 1
        ig_same_line()
        if ig_toggle_button(String("turn"), p.gizmo_mode == 2, 58.0):
            p.gizmo_mode = 2
        if p.gizmo_mode != 0:
            var w = p.gizmo_world
            if ig_checkbox(String("world axes"), w):
                p.gizmo_world = w
            var on = p.gizmo_snap > 0.0
            if ig_checkbox(String("snap"), on):
                # The defaults are per-MODE because the UNIT is per-mode:
                # 1 cm and 15 degrees. A single number would be one of the
                # two silently misread.
                p.gizmo_snap = 0.0 if not on \
                    else (Float32(15.0) if p.gizmo_mode == 2 else Float32(0.01))
            if p.gizmo_snap > 0.0:
                var st = p.gizmo_snap
                ig_set_next_item_width(-70.0)
                if ig_drag_float(String("deg") if p.gizmo_mode == 2
                                 else String("m"), st, 0.005, 0.0001, 90.0):
                    p.gizmo_snap = st
            if gizmo_hint.byte_length() > 0:
                ig_text_colored(gizmo_hint, 1.0, 0.75, 0.2, 1.0)
        else:
            ig_text_disabled(String("select a body or geom, then move/turn"))

    if ig_collapsing_header(String("Visibility groups")):
        # MuJoCo's `mjvOption.geomgroup`, six checkboxes. Not cosmetic: dog
        # parks its collision capsules in group 3 and its bone meshes in
        # group 5, so this is how you see either.
        for g in range(6):
            ig_push_id_int(g)
            var v = p.group_shown[g]
            if ig_checkbox(String("group ", g), v):
                p.group_shown[g] = v
            ig_pop_id()

    if ig_collapsing_header(String("Rendering")):
        _ = ig_checkbox(String("sites"), p.show_sites)
        _ = ig_checkbox(String("built-in HUD"), p.show_hud)
        if ig_button(String("reframe camera")):
            out.reframe = True

    if ig_collapsing_header(String("Props"), True):
        # ⚠ A DROPPED PROP IS A STRUCTURAL EDIT — it changes nbody, nq and nv,
        # so it goes down the SLOW path (regenerate the scene, rebuild
        # everything) rather than being patched into the live model. Measured
        # at 0.2-14 ms per asset, which is a click. The fast path is for
        # dims-preserving edits only; see `studio/edit.mojo`.
        if ig_button(String("box"), 62.0):
            out.add_prop = 0
        ig_same_line()
        if ig_button(String("sphere"), 62.0):
            out.add_prop = 1
        ig_same_line()
        if ig_button(String("capsule"), 62.0):
            out.add_prop = 2
        ig_same_line()
        if ig_button(String("cylinder"), 62.0):
            out.add_prop = 3
        ig_spacing()
        if ig_button(String("duplicate selected"), 140.0):
            out.dup_prop = True
        ig_same_line()
        if ig_button(String("delete"), 70.0):
            out.del_prop = True
        ig_text_disabled(String("props drop in front of the camera"))

    if ig_collapsing_header(String("Structure"), True):
        # ⚠ THIS EDITS THE MODEL, NOT THE SCENE. The Props buttons above add
        # and remove INSTANCES; this removes a body or a geom from the robot's
        # own tree, taking every reference to it — the actuator on its joint,
        # the exclude that names it, the tendon routed through its sites.
        ig_set_next_item_width(-1.0)
        _ = ig_input_text(String("##newname"), p.name_buf,
                          String("name for add / rename"))
        out.new_name = p.name_buf.value()

        if ig_button(String("add body"), 92.0):
            out.add_body_here = True
        ig_same_line()
        # ⚠ THE JOINT TYPE IS A CHOICE, and `free` is one of them even though
        # MuJoCo refuses a free joint on a nested body. Hiding it would be a
        # second opinion on a rule `validate_model` already owns and gates;
        # the Problems tab names the refusal by code.
        ig_set_next_item_width(110.0)
        var kinds = List[String]()
        for k in ["hinge", "slide", "ball", "free"]:
            kinds.append(String(k))
        var jk = Int32(p.joint_kind)
        if ig_combo(String("##jt"), jk, kinds):
            p.joint_kind = Int(jk)
        ig_same_line()
        if ig_button(String("add joint"), 92.0):
            out.add_joint_here = p.joint_kind

        if p.sel_kind == SEL_NONE:
            ig_text_disabled(String(
                "nothing selected — a new body goes into the world"
            ))
        else:
            var what = String("geom") if p.sel_kind == SEL_GEOM \
                else String("body")
            if ig_button(String("rename selected ") + what, 200.0):
                out.rename_here = True
            if p.sel_kind == SEL_BODY:
                # ⚠ THE NAME BOX MEANS THE NEW PARENT HERE, and empty means
                # the world. One box for three verbs is fine because they are
                # never in flight together, but the label has to say which.
                if ig_button(String("reparent under (name, or world)"), 200.0):
                    out.reparent_here = True
            if ig_button(String("delete selected ") + what, 200.0):
                out.del_element = True
            # ⚠ SAID OUT LOUD, BEFORE THE CLICK. Deleting the only geom of a
            # moving body gives a model MuJoCo refuses — a legitimate step in
            # a repair, and one the user should not discover by surprise. The
            # Problems tab names it afterwards; this names it beforehand.
            ig_text_disabled(String(
                "delete removes it and everything that referenced it"
            ))

    if ig_collapsing_header(String("Model")):
        ig_text_disabled(path)
        if ig_button(String("Open another model...")):
            p.browser_open = True

    ig_end()


# ═══════════════════════════════════════════════════════════════════════════
# right: Explorer | Inspector
# ═══════════════════════════════════════════════════════════════════════════


def ui_explorer(
    mut p: StudioPanel,
    body_names: List[String],
    geom_names: List[String],
    joint_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    joint_body: List[Int],
    height: Float32,
) raises:
    """The body tree, selectable — MuJoCo's Explorer tab.

    ⚠ A FLAT LIST WHEN FILTERED, a tree otherwise. Making the user expand
    branches to reach something whose name they just typed defeats the search,
    and dog has 62 bodies and 128 geoms.

    ⚠ IDs ARE PUSHED PER ROW. ImGui identifies a widget by its LABEL, and a
    geom named after its body — extremely common — would collide with it,
    leaving one of the two permanently unclickable. That reads as "selection
    is flaky", not as an id bug.
    """
    ig_set_next_item_width(-1.0)
    _ = ig_input_text(String("##filter"), p.filter, String("filter..."))
    var query = p.filter.value()

    if ig_begin_child(String("explorer"), 0.0, height, True):
        if query.byte_length() > 0:
            var shown = 0
            for b in range(len(body_names)):
                var nm = _label(body_names, b, "body")
                if _contains(nm, query):
                    shown += 1
                    ig_push_id_int(b)
                    if ig_selectable(
                        String("body  ", nm),
                        p.sel_kind == SEL_BODY and p.sel_index == b,
                    ):
                        p.sel_kind = SEL_BODY
                        p.sel_index = b
                    ig_pop_id()
            for g in range(len(geom_names)):
                var nm = _label(geom_names, g, "geom")
                if _contains(nm, query):
                    shown += 1
                    ig_push_id_int(100000 + g)
                    if ig_selectable(
                        String("geom  ", nm),
                        p.sel_kind == SEL_GEOM and p.sel_index == g,
                    ):
                        p.sel_kind = SEL_GEOM
                        p.sel_index = g
                    ig_pop_id()
            if shown == 0:
                ig_text_disabled(String("no match"))
        else:
            for b in range(len(body_names)):
                ig_push_id_int(b)
                # The world and the root open; everything below stays closed,
                # or a humanoid fills the panel before the inspector is
                # reachable.
                if b <= 1:
                    ig_set_next_item_open(True)
                if ig_tree_node(_label(body_names, b, "body")):
                    if ig_small_button(String("select body")):
                        p.sel_kind = SEL_BODY
                        p.sel_index = b
                    if p.sel_kind == SEL_BODY and p.sel_index == b:
                        ig_same_line()
                        ig_text_disabled(String("(selected)"))
                    for j in range(len(joint_body)):
                        if joint_body[j] != b:
                            continue
                        ig_text_disabled(
                            String("  joint  ", _label(joint_names, j, "joint"))
                        )
                    for g in range(len(geom_body)):
                        if geom_body[g] != b:
                            continue
                        ig_push_id_int(100000 + g)
                        if ig_selectable(
                            String("  geom  ", _label(geom_names, g, "geom")),
                            p.sel_kind == SEL_GEOM and p.sel_index == g,
                        ):
                            p.sel_kind = SEL_GEOM
                            p.sel_index = g
                        ig_pop_id()
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()


def ui_inspector(
    p: StudioPanel,
    mut out: PanelOut,
    body_names: List[String],
    geom_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    keys: List[String],
    vals: List[Float64],
    editable: List[Int],
) raises:
    """The selected element's record — READ ONLY in S1.

    `keys`/`vals` are the studio's flattened view of whatever is selected. The
    STUDIO builds them because it is the side that knows the dims; passing a
    `Model` or a `Data` here would make every widget line generic and cost the
    binary its "compiles once" property.

    ⚠ KEYS AND VALUES ARE PARALLEL LISTS, so a mismatch shows as a wrong
    number rather than a compile error. That is a real trade, and it is worth
    it at ONE call site — it would not be at ten.
    """
    if p.sel_kind == SEL_NONE or p.sel_index < 0:
        ig_text_disabled(String("nothing selected"))
        ig_spacing()
        ig_text_disabled(String("click a geom in the scene,"))
        ig_text_disabled(String("or pick one in the Explorer"))
        return

    if p.sel_kind == SEL_BODY:
        ig_text(String("body (", p.sel_index, ")  ",
                       _label(body_names, p.sel_index, "body")))
        var par = -1
        if p.sel_index > 0 and p.sel_index - 1 < len(body_parent):
            par = body_parent[p.sel_index - 1]
        ig_text_disabled(String("parent: ", _label(body_names, par, "body")))
    else:
        ig_text(String("geom (", p.sel_index, ")  ",
                       _label(geom_names, p.sel_index, "geom")))
        var b = -1
        if p.sel_index < len(geom_body):
            b = geom_body[p.sel_index]
        ig_text_disabled(String("body: ", _label(body_names, b, "body")))
    ig_separator()

    ig_columns(2, True)
    ig_set_column_width(0, 130.0)
    var n = len(keys) if len(keys) < len(vals) else len(vals)
    for i in range(n):
        ig_text_disabled(keys[i])
        ig_next_column()
        # ⚠ `editable[i]` IS THE EDIT FIELD ID OR -1, decided by the STUDIO.
        # The panel cannot know which record slot a row maps to — that is
        # exactly the knowledge that would make this file generic — so the
        # caller hands it a parallel list and gets back which row moved.
        if i < len(editable) and editable[i] >= 0:
            ig_push_id_int(i)
            var v = Float32(vals[i])
            ig_set_next_item_width(-1.0)
            if ig_drag_float(String("##e"), v, 0.005):
                out.edit_field = editable[i]
                out.edit_value = Float64(v)
            ig_pop_id()
        else:
            ig_text(_f(vals[i]))
        ig_next_column()
    ig_columns(1)
    ig_spacing()
    ig_text_disabled(String("drag a white value to edit it"))


def ui_problems(diags: List[Diagnostic], h: Float32) raises:
    """The diagnostics list — V2.0's half of "a marker, not an abort".

    ⚠ THE TAB IS ALWAYS PRESENT, AND SAYS SO WHEN THERE IS NOTHING. A panel
    that appears only when something is wrong cannot be used to confirm that
    nothing is: "no Problems tab" and "I did not look" are the same picture.

    ⚠ ERRORS FIRST, THEN WARNINGS, THEN INFO — and not because it is tidier.
    An error is the reason the model will not load; a warning below it is
    noise until that is fixed, and a list in discovery order buries the one
    line the user needs under twelve `zero-gear`s.
    """
    if ig_begin_child(String("problems"), 0.0, h):
        var n_err = 0
        var n_warn = 0
        for d in diags:
            if d.severity >= SEV_ERROR:
                n_err += 1
            elif d.severity == SEV_WARN:
                n_warn += 1
        if len(diags) == 0:
            ig_text_colored(String("No problems."), 0.55, 0.85, 0.55)
            ig_text_disabled(
                String(
                    "Checked against what MuJoCo refuses, not against taste."
                )
            )
        else:
            ig_text(
                String(n_err) + " error(s), " + String(n_warn) + " warning(s)"
            )
        ig_separator()

        # ⚠ THREE PASSES, NOT A SORT. `List` has no stable sort here and the
        # order WITHIN a severity is the order the checks ran, which is
        # the model order the user is reading — worth keeping.
        for sev in [SEV_ERROR, SEV_WARN, SEV_INFO]:
            for d in diags:
                if d.severity != Int(sev):
                    continue
                if Int(sev) >= SEV_ERROR:
                    ig_text_colored(
                        String("[error] ") + d.subject, 0.95, 0.4, 0.4
                    )
                elif Int(sev) == SEV_WARN:
                    ig_text_colored(
                        String("[warn]  ") + d.subject, 0.95, 0.8, 0.35
                    )
                else:
                    ig_text_disabled(String("[info]  ") + d.subject)
                if ig_is_item_hovered():
                    ig_set_tooltip(d.code + "\n\n" + d.message)
                # ⚠ THE MESSAGE IS SHOWN, NOT ONLY TOOLTIPPED. A
                # diagnostic whose text only appears on hover is one the
                # user has to guess is there; the tooltip carries the
                # CODE, which is what they would search the source for.
                ig_text_disabled(String("        ") + d.message)
                ig_spacing()
    ig_end_child()


def ui_right_panel(
    mut p: StudioPanel,
    body_names: List[String],
    geom_names: List[String],
    joint_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    joint_body: List[Int],
    keys: List[String],
    vals: List[Float64],
    editable: List[Int],
    diags: List[Diagnostic],
    mut out: PanelOut,
    x0: Float32,
    y0: Float32,
    h: Float32,
) raises:
    ig_begin_panel(String("Explorer/Inspector"), x0, y0, RIGHT_W, h)
    if ig_begin_tab_bar(String("right")):
        # ⚠ `ig_end_tab_item` ONLY WHEN `begin` RETURNED TRUE — unlike the
        # panel pairs, whose End is unconditional. Getting it wrong corrupts
        # the id stack silently in a release shim, and the symptom is a
        # DIFFERENT widget going dead.
        if ig_begin_tab_item(String("Explorer")):
            ui_explorer(p, body_names, geom_names, joint_names,
                        body_parent, geom_body, joint_body, h - 120.0)
            ig_end_tab_item()
        if ig_begin_tab_item(String("Inspector")):
            ui_inspector(p, out, body_names, geom_names, body_parent,
                         geom_body, keys, vals, editable)
            ig_end_tab_item()
        # ⚠ THE COUNT IS IN THE TAB LABEL. The whole point of the panel is a
        # state the user can WORK IN, so the badge has to be visible from the
        # tab they are already on.
        var badge = String("Problems")
        var nerr = 0
        for d in diags:
            if d.severity >= SEV_ERROR:
                nerr += 1
        if nerr > 0:
            badge = String("Problems (") + String(nerr) + ")"
        if ig_begin_tab_item(badge):
            ui_problems(diags, h - 120.0)
            ig_end_tab_item()
        ig_end_tab_bar()
    ig_end()


def build_ui(
    mut p: StudioPanel,
    path: String,
    step_i: Int,
    ncon: Int,
    contact_budget: Int,
    step_us: Float64,
    win_w: Float32,
    win_h: Float32,
    body_names: List[String],
    geom_names: List[String],
    joint_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    joint_body: List[Int],
    keys: List[String],
    vals: List[Float64],
    editable: List[Int],
    diags: List[Diagnostic],
    can_undo: Bool = False,
    can_redo: Bool = False,
    undo_label: String = String(""),
    redo_label: String = String(""),
    gizmo_hint: String = String(""),
) raises -> PanelOut:
    """The whole UI for one frame. See the module header for the layout."""
    var out = PanelOut()
    var y0 = ui_menu_bar(p, out, can_undo, can_redo, undo_label, redo_label)
    var h = win_h - y0
    ui_options(p, out, path, step_i, ncon, contact_budget, step_us, y0, h,
               gizmo_hint)
    ui_right_panel(p, body_names, geom_names, joint_names, body_parent,
                   geom_body, joint_body, keys, vals, editable, diags, out,
                   win_w - RIGHT_W, y0, h)
    ui_file_browser(p, out)
    return out^
