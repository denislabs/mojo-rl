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

from std.os import listdir
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
    ig_columns, ig_next_column, ig_set_column_width,
)

comptime SIDEBAR_W: Float32 = 300.0
comptime RIGHT_W: Float32 = 340.0

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
    var browser_open: Bool
    var browser_dir: String
    var browser_path: TextBuffer
    var recent: List[String]

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
        self.browser_open = False
        self.browser_dir = start_dir
        self.browser_path = TextBuffer()
        self.recent = List[String]()

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
    var open_path: String
    """A model to LOAD, or "" for none. The swap is the studio's to perform:
    it owns the renderer handoff and every container that has to be rebuilt."""

    def __init__(out self):
        self.reset = False
        self.step_once = False
        self.reframe = False
        self.quit = False
        self.open_path = String("")


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


def ui_menu_bar(mut p: StudioPanel, mut out: PanelOut) raises -> Float32:
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
            if ig_menu_item(String("Quit")):
                out.quit = True
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

    if ig_begin_child(String("entries"), 0.0, 300.0, True):
        # ⚠ ONE `listdir` PER FRAME, and it is fine: this window is open for
        # seconds, not for a session, and caching it would need invalidation
        # the moment the user creates a file. Measured directories here are
        # tens of entries.
        var dirs = List[String]()
        var files = List[String]()
        try:
            var entries = listdir(Path(p.browser_dir))
            for e in entries:
                var nm = String(e)
                if nm.startswith("."):
                    continue
                var full = p.browser_dir + "/" + nm
                if nm.endswith(".xml"):
                    files.append(nm)
                elif Path(full).is_dir():
                    dirs.append(nm)
        except:
            ig_text_disabled(String("cannot read this directory"))

        for i in range(len(dirs)):
            ig_push_id_int(i)
            if ig_selectable(String("[dir] ", dirs[i])):
                p.browser_dir = p.browser_dir + "/" + dirs[i]
            ig_pop_id()
        for i in range(len(files)):
            ig_push_id_int(100000 + i)
            if ig_selectable(files[i]):
                out.open_path = p.browser_dir + "/" + files[i]
                p.browser_open = False
            ig_pop_id()
        if len(dirs) == 0 and len(files) == 0:
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
    body_names: List[String],
    geom_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    keys: List[String],
    vals: List[Float64],
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
        _row(keys[i], _f(vals[i]))
    ig_columns(1)
    ig_spacing()
    ig_text_disabled(String("read-only in S1 — editing is S3"))


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
            ui_inspector(p, body_names, geom_names, body_parent, geom_body,
                         keys, vals)
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
) raises -> PanelOut:
    """The whole UI for one frame. See the module header for the layout."""
    var out = PanelOut()
    var y0 = ui_menu_bar(p, out)
    var h = win_h - y0
    ui_options(p, out, path, step_i, ncon, contact_budget, step_us, y0, h)
    ui_right_panel(p, body_names, geom_names, joint_names, body_parent,
                   geom_body, joint_body, keys, vals,
                   win_w - RIGHT_W, y0, h)
    ui_file_browser(p, out)
    return out^
