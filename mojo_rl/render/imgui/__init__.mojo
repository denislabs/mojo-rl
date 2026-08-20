"""Dear ImGui bindings — a composable, immediate-mode UI for the 3D viewer.

    pixi run build-imgui        # ONCE, builds the C++ shim this binds to
    from mojo_rl.render.imgui import ig_begin_panel, ig_button, ig_end

Every function here is a thin call into `imgui_shim.cpp`, which is a flat C
API over Dear ImGui. Widgets are composed by CALLING them in order — there is
no scene graph, no retained widget tree, and no state to keep in sync:

    ig_begin_panel("sidebar", 0.0, 0.0, 320.0, 720.0)
    ig_text("dm_control")
    ig_separator()
    if ig_button("reset episode"):
        step_i = 0
    ig_end()

⚠ THIS REPLACES `mojo_rl/render/ui.mojo`, it does not extend it. The two are
not interchangeable: `ui.mojo` widgets RECORD into a command buffer that the
renderer paints later, because an application cannot draw between
`render_frame`'s begin/end. ImGui builds its own draw list, so widgets here
draw themselves and the deferral disappears. Do not mix them in one panel —
they hit-test against different notions of where the pointer is.

⚠ REQUIRES A BUILT SHIM. `pixi run build-imgui` produces
`libmojo_imgui.dylib` beside this file. It is NOT tracked in git, so a fresh
clone has to build it, and the failure mode is a dlopen ABORT at the first
call rather than a compile error. `imgui_shim_available()` answers the
question without touching FFI, so a caller can print something useful first.

ID RULES, the one piece of ImGui that surprises people: widgets are identified
by their LABEL, so two buttons reading "reset" in the same window are ONE
widget and only one of them responds. Wrap loop bodies in
`ig_push_id_int(i)` / `ig_pop_id()`, or give labels a "##unique" suffix (the
part after "##" is part of the id but is not drawn).

⚠ A LABEL THAT CHANGES EVERY FRAME IS A NEW WIDGET EVERY FRAME, and the
symptom is not a wrong id — it is a button that CANNOT BE CLICKED. ImGui
completes a click across a press and a release; if the id changes in between,
the two land on different widgets and nothing fires. This shipped once: the
viewer's record button displayed a live frame count ("rec 1", "rec 2", ...), so
it would start a recording and then refuse to stop it.

Use "###" for any caption that varies:

    "rec 12###rec"    drawn: "rec 12"    id: "rec"     stable ✓
    "rec 12##rec"     drawn: "rec 12"    id: whole string — STILL CHANGES ✗

`ig_push_id_*` does NOT rescue this: the pushed id is combined with the
widget's own label, so a varying label still yields a varying id.
"""

from std.os import abort, getenv
from std.sys import CompilationTarget
from std.pathlib import Path
from std.ffi import (
    _Global,
    OwnedDLHandle,
    _get_dylib_function,
    c_char,
    c_int,
    c_float,
)

from mojo_rl.render.sdl import Ptr, untracked


# ═══════════════════════════════════════════════════════════════════════════
# dylib loading
# ═══════════════════════════════════════════════════════════════════════════


def _imgui_lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libmojo_imgui.dylib")
    elif CompilationTarget.is_linux():
        return String("libmojo_imgui.so")
    else:
        comptime assert False, "OS is not supported"


def _imgui_candidates() -> List[String]:
    """Where to look for the shim, most explicit first.

    Kept separate from `_init_imgui_handle` so `imgui_shim_available()` can
    check the SAME list without dlopening anything — a probe that consulted a
    different list would answer a different question than the loader asks.
    """
    var name = _imgui_lib_name()
    var candidates = List[String]()
    var override = getenv("MOJO_RL_IMGUI_LIB")
    if override.byte_length() > 0:
        candidates.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        candidates.append(root + "/mojo_rl/render/imgui/" + name)
    # Relative to CWD, which for this project is the repo root (every documented
    # command is `pixi run mojo run -I . ...` from there).
    candidates.append("mojo_rl/render/imgui/" + name)
    candidates.append(name)
    return candidates^


def imgui_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    Exists because `_Global` aborts the process on a missing library, which is
    the right behaviour for a hard dependency and the wrong first impression
    for an optional one. Call this before the first ImGui call to print
    "run `pixi run build-imgui`" instead of dying with a dlopen trace.
    """
    var candidates = _imgui_candidates()
    for i in range(len(candidates)):
        if Path(candidates[i]).exists():
            return True
    return False


def _init_imgui_handle() -> OwnedDLHandle:
    """Non-raising, as `_Global` demands. Aborts with the attempted paths
    rather than returning an uninitialised handle — the same reasoning as
    `_init_sdl_handle`, where returning garbage turned a missing library into
    a segfault at the first unrelated call."""
    var candidates = _imgui_candidates()
    for i in range(len(candidates)):
        try:
            return OwnedDLHandle(candidates[i])
        except:
            pass

    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    abort(
        "Dear ImGui shim not found. Tried:"
        + tried
        + "\nBuild it with `pixi run build-imgui`, or set"
        + " MOJO_RL_IMGUI_LIB=/path/to/"
        + _imgui_lib_name()
    )


comptime lib = _Global["MOJO_RL_IMGUI", _init_imgui_handle]()


@always_inline
def _c(mut s: String) -> Ptr[c_char, MutUntrackedOrigin]:
    """Borrow a String's NUL-terminated bytes as a `const char*`.

    ⚠ THE POINTER IS ONLY VALID WHILE `s` IS. Every use below passes it
    straight into a call that copies or consumes it before returning, which is
    why this can hand back an untracked origin at all. Do not stash it.
    """
    return untracked(s.as_c_string_slice().unsafe_ptr())


# ═══════════════════════════════════════════════════════════════════════════
# lifecycle — called by Renderer3D, not by application code
# ═══════════════════════════════════════════════════════════════════════════


def ig_init[
    W: AnyType, D: AnyType, ow: Origin, od: Origin
](
    window: Ptr[W, ow],
    device: Ptr[D, od],
    color_format: UInt32,
) raises -> Bool:
    """Attach ImGui to an EXISTING window and GPU device.

    Parameterised on the handle types so this module needs no dependency on
    `sdl_gpu`'s structs; the C side takes both as `void*` regardless.
    """
    return _get_dylib_function[
        lib,
        "mrl_ig_init",
        def(
            Ptr[NoneType, MutUntrackedOrigin],
            Ptr[NoneType, MutUntrackedOrigin],
            UInt32,
        ) thin -> Bool,
    ]()(
        untracked(window.unsafe_bitcast[NoneType]()),
        untracked(device.unsafe_bitcast[NoneType]()),
        color_format,
    )


def ig_shutdown() raises:
    _get_dylib_function[lib, "mrl_ig_shutdown", def() thin -> None]()()


def ig_new_frame() raises:
    """Open a frame. All widget calls must fall between this and `ig_prepare`.
    """
    _get_dylib_function[lib, "mrl_ig_new_frame", def() thin -> None]()()


def ig_prepare[
    T: AnyType, o: Origin
](cmd_buf: Ptr[T, o]) raises:
    """Close the frame and upload its geometry.

    ⚠ MUST RUN BEFORE ANY RENDER PASS IS OPEN on this command buffer — the
    SDL_GPU backend records buffer uploads here, and SDL_GPU forbids copy
    passes while a render pass is active.
    """
    _get_dylib_function[
        lib, "mrl_ig_prepare",
        def(Ptr[NoneType, MutUntrackedOrigin]) thin -> None,
    ]()(untracked(cmd_buf.unsafe_bitcast[NoneType]()))


def ig_render[
    T: AnyType, P: AnyType, oc: Origin, op: Origin
](cmd_buf: Ptr[T, oc], render_pass: Ptr[P, op]) raises:
    """Draw the prepared frame INSIDE an open render pass."""
    _get_dylib_function[
        lib, "mrl_ig_render",
        def(
            Ptr[NoneType, MutUntrackedOrigin],
            Ptr[NoneType, MutUntrackedOrigin],
        ) thin -> None,
    ]()(
        untracked(cmd_buf.unsafe_bitcast[NoneType]()),
        untracked(render_pass.unsafe_bitcast[NoneType]()),
    )


def ig_process_event[
    E: AnyType, o: Origin
](event: Ptr[E, o]) raises:
    """Feed one SDL event to ImGui. Every event, or input desynchronises."""
    _get_dylib_function[
        lib, "mrl_ig_process_event",
        def(Ptr[NoneType, MutUntrackedOrigin]) thin -> None,
    ]()(untracked(event.unsafe_bitcast[NoneType]()))


def ig_want_mouse() raises -> Bool:
    """True when the pointer is over ImGui. The host must NOT orbit then."""
    return _get_dylib_function[
        lib, "mrl_ig_want_mouse", def() thin -> Bool
    ]()()


def ig_want_keyboard() raises -> Bool:
    """True when ImGui is taking typed input, so "s" is a letter and not the
    screenshot shortcut. This is what makes an explicit text-input MODE
    unnecessary."""
    return _get_dylib_function[
        lib, "mrl_ig_want_keyboard", def() thin -> Bool
    ]()()


def ig_framerate() raises -> Float32:
    return _get_dylib_function[
        lib, "mrl_ig_framerate", def() thin -> c_float
    ]()()


def ig_set_ini_filename(var path: String) raises:
    """Persist window layout to `path`; "" disables persistence (the default —
    ImGui would otherwise write `imgui.ini` into the repo root)."""
    _get_dylib_function[
        lib, "mrl_ig_set_ini_filename",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(path))


# ═══════════════════════════════════════════════════════════════════════════
# windows and layout
# ═══════════════════════════════════════════════════════════════════════════


def ig_begin_panel(var name: String, x: Float32, y: Float32, w: Float32,
                   h: Float32) raises:
    """A borderless panel pinned to an exact rect — the sidebar shape.

    Position and size are re-applied every frame, so the caller's layout stays
    authoritative across window resizes. Always pair with `ig_end()`.
    """
    _get_dylib_function[
        lib, "mrl_ig_begin_panel",
        def(
            Ptr[c_char, MutUntrackedOrigin], c_float, c_float, c_float, c_float
        ) thin -> None,
    ]()(_c(name), x, y, w, h)


def ig_begin_window(var name: String, x: Float32 = 60.0, y: Float32 = 60.0,
                    w: Float32 = 320.0, h: Float32 = 240.0) raises -> Bool:
    """A movable, resizable window; the geometry applies on FIRST use only.

    Returns False when collapsed — skip the contents, but call `ig_end()`
    either way.
    """
    return _get_dylib_function[
        lib, "mrl_ig_begin_window",
        def(
            Ptr[c_char, MutUntrackedOrigin], c_float, c_float, c_float, c_float
        ) thin -> Bool,
    ]()(_c(name), x, y, w, h)


def ig_end() raises:
    _get_dylib_function[lib, "mrl_ig_end", def() thin -> None]()()


def ig_begin_child(var id: String, w: Float32 = 0.0, h: Float32 = 0.0,
                   border: Bool = True) raises -> Bool:
    """A scrolling sub-region. 0 for a dimension means "fill the parent".

    This is what makes an 85-task list a non-problem: the child scrolls, so
    the list no longer has to fit the window.
    """
    return _get_dylib_function[
        lib, "mrl_ig_begin_child",
        def(
            Ptr[c_char, MutUntrackedOrigin], c_float, c_float, Bool
        ) thin -> Bool,
    ]()(_c(id), w, h, border)


def ig_end_child() raises:
    _get_dylib_function[lib, "mrl_ig_end_child", def() thin -> None]()()


def ig_separator() raises:
    _get_dylib_function[lib, "mrl_ig_separator", def() thin -> None]()()


def ig_separator_text(var label: String) raises:
    """A horizontal rule with an embedded caption — a section heading."""
    _get_dylib_function[
        lib, "mrl_ig_separator_text",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(label))


def ig_same_line(offset_x: Float32 = 0.0, spacing: Float32 = -1.0) raises:
    """Put the NEXT widget on the same line as the previous one."""
    _get_dylib_function[
        lib, "mrl_ig_same_line", def(c_float, c_float) thin -> None,
    ]()(offset_x, spacing)


def ig_spacing() raises:
    _get_dylib_function[lib, "mrl_ig_spacing", def() thin -> None]()()


def ig_indent(w: Float32 = 0.0) raises:
    _get_dylib_function[
        lib, "mrl_ig_indent", def(c_float) thin -> None
    ]()(w)


def ig_unindent(w: Float32 = 0.0) raises:
    _get_dylib_function[
        lib, "mrl_ig_unindent", def(c_float) thin -> None
    ]()(w)


def ig_set_next_item_width(w: Float32) raises:
    """Width of the next widget. NEGATIVE means "fill minus that much" —
    -1.0 is the idiom for "full width"."""
    _get_dylib_function[
        lib, "mrl_ig_set_next_item_width", def(c_float) thin -> None
    ]()(w)


def ig_content_width() raises -> Float32:
    """Horizontal space left in the current window or child."""
    return _get_dylib_function[
        lib, "mrl_ig_content_width", def() thin -> c_float
    ]()()


def ig_push_id_int(id: Int) raises:
    """Disambiguate identical labels inside a loop. See the ID RULES note at
    the top of this module — without it, N buttons sharing a label are one
    button."""
    _get_dylib_function[
        lib, "mrl_ig_push_id_int", def(c_int) thin -> None
    ]()(c_int(id))


def ig_push_id_str(var id: String) raises:
    _get_dylib_function[
        lib, "mrl_ig_push_id_str",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(id))


def ig_pop_id() raises:
    _get_dylib_function[lib, "mrl_ig_pop_id", def() thin -> None]()()


# ═══════════════════════════════════════════════════════════════════════════
# text
# ═══════════════════════════════════════════════════════════════════════════


def ig_text(var s: String) raises:
    _get_dylib_function[
        lib, "mrl_ig_text",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(s))


def ig_text_colored(var s: String, r: Float32, g: Float32, b: Float32,
                    a: Float32 = 1.0) raises:
    """Colour components are 0..1 floats, NOT the 0..255 bytes `Color` uses."""
    _get_dylib_function[
        lib, "mrl_ig_text_colored",
        def(
            Ptr[c_char, MutUntrackedOrigin], c_float, c_float, c_float, c_float
        ) thin -> None,
    ]()(_c(s), r, g, b, a)


def ig_text_disabled(var s: String) raises:
    _get_dylib_function[
        lib, "mrl_ig_text_disabled",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(s))


def ig_text_wrapped(var s: String) raises:
    _get_dylib_function[
        lib, "mrl_ig_text_wrapped",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(s))


# ═══════════════════════════════════════════════════════════════════════════
# widgets — each returns whether it was activated THIS frame
# ═══════════════════════════════════════════════════════════════════════════


def ig_button(var label: String, w: Float32 = 0.0, h: Float32 = 0.0) raises -> Bool:
    """0 sizes auto-fit the label; a negative width fills the region."""
    return _get_dylib_function[
        lib, "mrl_ig_button",
        def(Ptr[c_char, MutUntrackedOrigin], c_float, c_float) thin -> Bool,
    ]()(_c(label), w, h)


def ig_small_button(var label: String) raises -> Bool:
    return _get_dylib_function[
        lib, "mrl_ig_small_button",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> Bool,
    ]()(_c(label))


def ig_toggle_button(var label: String, active: Bool, w: Float32 = 0.0,
                     h: Float32 = 0.0) raises -> Bool:
    """A button drawn "pressed" when `active` — the radio-group idiom."""
    return _get_dylib_function[
        lib, "mrl_ig_toggle_button",
        def(
            Ptr[c_char, MutUntrackedOrigin], Bool, c_float, c_float
        ) thin -> Bool,
    ]()(_c(label), active, w, h)


def ig_selectable(var label: String, selected: Bool = False) raises -> Bool:
    """A full-width list row, highlighted when `selected`."""
    return _get_dylib_function[
        lib, "mrl_ig_selectable",
        def(Ptr[c_char, MutUntrackedOrigin], Bool) thin -> Bool,
    ]()(_c(label), selected)


def ig_checkbox(var label: String, mut v: Bool) raises -> Bool:
    """Mutates `v` in place; returns True on the frame it changed."""
    return _get_dylib_function[
        lib, "mrl_ig_checkbox",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[Bool, MutUntrackedOrigin]
        ) thin -> Bool,
    ]()(_c(label), untracked(Ptr(to=v)))


def ig_radio(var label: String, active: Bool) raises -> Bool:
    return _get_dylib_function[
        lib, "mrl_ig_radio",
        def(Ptr[c_char, MutUntrackedOrigin], Bool) thin -> Bool,
    ]()(_c(label), active)


def ig_slider_float(var label: String, mut v: Float32, lo: Float32, hi: Float32,
                    var fmt: String = String("%.3f")) raises -> Bool:
    """A DRAGGABLE slider — it streams values for the whole gesture, which is
    the capability the hand-rolled widget layer could not provide at all."""
    return _get_dylib_function[
        lib, "mrl_ig_slider_float",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_float, MutUntrackedOrigin],
            c_float, c_float, Ptr[c_char, MutUntrackedOrigin],
        ) thin -> Bool,
    ]()(_c(label), untracked(Ptr(to=v)), lo, hi, _c(fmt))


def ig_slider_int(var label: String, mut v: Int32, lo: Int32, hi: Int32) raises -> Bool:
    return _get_dylib_function[
        lib, "mrl_ig_slider_int",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_int, MutUntrackedOrigin],
            c_int, c_int,
        ) thin -> Bool,
    ]()(_c(label), untracked(Ptr(to=v)), lo, hi)


def ig_drag_float(var label: String, mut v: Float32, speed: Float32 = 0.01,
                  lo: Float32 = 0.0, hi: Float32 = 0.0,
                  var fmt: String = String("%.3f")) raises -> Bool:
    """Unbounded drag. lo == hi == 0 means no clamping."""
    return _get_dylib_function[
        lib, "mrl_ig_drag_float",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_float, MutUntrackedOrigin],
            c_float, c_float, c_float, Ptr[c_char, MutUntrackedOrigin],
        ) thin -> Bool,
    ]()(_c(label), untracked(Ptr(to=v)), speed, lo, hi, _c(fmt))


def ig_combo(var label: String, mut current: Int32, items: List[String]) raises -> Bool:
    """A dropdown. `items` is joined into ImGui's NUL-separated form here, so
    callers pass an ordinary list."""
    var packed = String("")
    for i in range(len(items)):
        packed += items[i]
        packed += String(chr(0))
    packed += String(chr(0))
    return _get_dylib_function[
        lib, "mrl_ig_combo",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_int, MutUntrackedOrigin],
            Ptr[c_char, MutUntrackedOrigin],
        ) thin -> Bool,
    ]()(_c(label), untracked(Ptr(to=current)), _c(packed))


def ig_tree_node(var label: String) raises -> Bool:
    """A collapsible node. When True, emit the children then `ig_tree_pop()`.

    ⚠ CALL `ig_tree_pop()` ONLY WHEN THIS RETURNED TRUE. ImGui keeps the
    open/closed state itself, so nothing needs storing on the caller's side.
    """
    return _get_dylib_function[
        lib, "mrl_ig_tree_node",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> Bool,
    ]()(_c(label))


def ig_tree_pop() raises:
    _get_dylib_function[lib, "mrl_ig_tree_pop", def() thin -> None]()()


def ig_set_next_item_open(open: Bool) raises:
    """Force the next tree node open or closed — used to expand every domain
    while a filter is active, then let ImGui resume owning the state."""
    _get_dylib_function[
        lib, "mrl_ig_set_next_item_open", def(Bool) thin -> None
    ]()(open)


# ═══════════════════════════════════════════════════════════════════════════
# menu bar, tabs, columns
# ═══════════════════════════════════════════════════════════════════════════


def ig_begin_main_menu_bar() raises -> Bool:
    """The viewport-level strip at the top. `ig_end_main_menu_bar` ONLY if True.

    ⚠ IT CONSUMES SPACE AT y=0. A panel placed at `y = 0` has its first row
    hidden behind the bar; use `ig_frame_height_with_spacing()` as the panel's
    `y` and subtract it from the panel's height. The symptom of getting it
    wrong is a panel that looks CLIPPED, not one that looks offset.
    """
    return _get_dylib_function[
        lib, "mrl_ig_begin_main_menu_bar", def() thin -> Bool
    ]()()


def ig_end_main_menu_bar() raises:
    _get_dylib_function[
        lib, "mrl_ig_end_main_menu_bar", def() thin -> None
    ]()()


def ig_begin_menu(var label: String) raises -> Bool:
    """A drop-down. `ig_end_menu` ONLY if this returned True."""
    return _get_dylib_function[
        lib, "mrl_ig_begin_menu",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> Bool,
    ]()(_c(label))


def ig_end_menu() raises:
    _get_dylib_function[lib, "mrl_ig_end_menu", def() thin -> None]()()


def ig_menu_item(
    var label: String,
    var shortcut: String = String(""),
    selected: Bool = False,
    enabled: Bool = True,
) raises -> Bool:
    """A row in a menu. True on the frame it is clicked."""
    return _get_dylib_function[
        lib, "mrl_ig_menu_item",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_char, MutUntrackedOrigin],
            Bool, Bool,
        ) thin -> Bool,
    ]()(_c(label), _c(shortcut), selected, enabled)


def ig_frame_height_with_spacing() raises -> Float32:
    """One row's height — what the main menu bar occupies at the top."""
    return _get_dylib_function[
        lib, "mrl_ig_frame_height_with_spacing", def() thin -> c_float
    ]()()


def ig_begin_tab_bar(var id: String) raises -> Bool:
    """`ig_end_tab_bar` ONLY if this returned True."""
    return _get_dylib_function[
        lib, "mrl_ig_begin_tab_bar",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> Bool,
    ]()(_c(id))


def ig_end_tab_bar() raises:
    _get_dylib_function[lib, "mrl_ig_end_tab_bar", def() thin -> None]()()


def ig_begin_tab_item(var label: String) raises -> Bool:
    """⚠ `ig_end_tab_item` ONLY WHEN THIS RETURNED TRUE — unlike the panel
    Begin/End pairs, whose End runs unconditionally. ImGui asserts on the
    mistake in a debug build and corrupts the id stack silently in a release
    one, which surfaces as a DIFFERENT widget going dead."""
    return _get_dylib_function[
        lib, "mrl_ig_begin_tab_item",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> Bool,
    ]()(_c(label))


def ig_end_tab_item() raises:
    _get_dylib_function[lib, "mrl_ig_end_tab_item", def() thin -> None]()()


def ig_columns(count: Int, border: Bool = False) raises:
    """Split the region into `count` columns; `ig_columns(1)` ends them."""
    _get_dylib_function[
        lib, "mrl_ig_columns", def(c_int, Bool) thin -> None
    ]()(c_int(count), border)


def ig_next_column() raises:
    _get_dylib_function[lib, "mrl_ig_next_column", def() thin -> None]()()


def ig_set_column_width(index: Int, w: Float32) raises:
    _get_dylib_function[
        lib, "mrl_ig_set_column_width", def(c_int, c_float) thin -> None
    ]()(c_int(index), w)


def ig_collapsing_header(var label: String, default_open: Bool = False) raises -> Bool:
    """Like a tree node but styled as a full-width bar; needs NO pop."""
    return _get_dylib_function[
        lib, "mrl_ig_collapsing_header",
        def(Ptr[c_char, MutUntrackedOrigin], Bool) thin -> Bool,
    ]()(_c(label), default_open)


def ig_plot_lines(var label: String, values: List[Float32], offset: Int = 0,
                  lo: Float32 = 0.0, hi: Float32 = 0.0, w: Float32 = 0.0,
                  h: Float32 = 40.0) raises:
    """A sparkline. `offset` is the write cursor of a ring buffer, so a
    rolling history plots in order without being rotated first.

    lo == hi == 0 auto-scales to the data.
    """
    if len(values) == 0:
        return
    _get_dylib_function[
        lib, "mrl_ig_plot_lines",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_float, MutUntrackedOrigin],
            c_int, c_int, c_float, c_float, c_float, c_float,
        ) thin -> None,
    ]()(
        _c(label), untracked(values.unsafe_ptr()), c_int(len(values)),
        c_int(offset), lo, hi, w, h,
    )


def ig_progress_bar(frac: Float32, w: Float32 = -1.0, h: Float32 = 0.0,
                    var overlay: String = String("")) raises:
    _get_dylib_function[
        lib, "mrl_ig_progress_bar",
        def(
            c_float, c_float, c_float, Ptr[c_char, MutUntrackedOrigin]
        ) thin -> None,
    ]()(frac, w, h, _c(overlay))


def ig_set_scroll_here_y(ratio: Float32 = 0.5) raises:
    """Scroll the enclosing child so the LAST-emitted widget is visible."""
    _get_dylib_function[
        lib, "mrl_ig_set_scroll_here_y", def(c_float) thin -> None
    ]()(ratio)


def ig_is_item_hovered() raises -> Bool:
    return _get_dylib_function[
        lib, "mrl_ig_is_item_hovered", def() thin -> Bool
    ]()()


def ig_set_tooltip(var s: String) raises:
    """Tooltip for the widget emitted immediately before this call."""
    _get_dylib_function[
        lib, "mrl_ig_set_tooltip",
        def(Ptr[c_char, MutUntrackedOrigin]) thin -> None,
    ]()(_c(s))


# ═══════════════════════════════════════════════════════════════════════════
# style
# ═══════════════════════════════════════════════════════════════════════════


def ig_style_dark() raises:
    _get_dylib_function[lib, "mrl_ig_style_dark", def() thin -> None]()()


def ig_style_light() raises:
    _get_dylib_function[lib, "mrl_ig_style_light", def() thin -> None]()()


def ig_style_classic() raises:
    _get_dylib_function[lib, "mrl_ig_style_classic", def() thin -> None]()()


def ig_set_font_scale(s: Float32) raises:
    """Scale the whole UI — HiDPI, or simply legibility."""
    _get_dylib_function[
        lib, "mrl_ig_set_font_scale", def(c_float) thin -> None
    ]()(s)


# ═══════════════════════════════════════════════════════════════════════════
# text input
# ═══════════════════════════════════════════════════════════════════════════


struct TextBuffer(Copyable, Movable):
    """A fixed C string buffer for `ig_input_text`.

    ImGui edits a caller-owned `char[]` in place, so a Mojo `String` cannot be
    passed directly — it owns its allocation and has no spare capacity. This
    holds the bytes ImGui writes into; `value()` converts back on demand.
    """

    comptime CAP = 128

    var data: InlineArray[UInt8, Self.CAP]

    def __init__(out self):
        self.data = InlineArray[UInt8, Self.CAP](fill=0)

    def value(self) -> String:
        """The buffer up to its NUL terminator, as a Mojo String."""
        var n = 0
        while n < Self.CAP and self.data[n] != 0:
            n += 1
        var out = String("")
        for i in range(n):
            out += chr(Int(self.data[i]))
        return out^

    def is_empty(self) -> Bool:
        return self.data[0] == 0

    def clear(mut self):
        self.data[0] = 0


def ig_input_text(var label: String, mut buf: TextBuffer,
                  var hint: String = String("")) raises -> Bool:
    """An editable text field. Returns True on the frames the text changed.

    `hint` is greyed placeholder text shown while the field is empty; ImGui
    has no native placeholder, and a filter box without one reads as broken.
    """
    # ⚠ VIA `Ptr(to=...)`, NOT `InlineArray.unsafe_ptr()`. The latter yields a
    # *safe* Pointer, whose `bitcast` is constrained away ("violated
    # constraint: not _safe"); pointing at the array itself gives the
    # Pointer the C ABI needs.
    var ptr = untracked(Ptr(to=buf.data).unsafe_bitcast[c_char]())
    if hint.byte_length() > 0:
        return _get_dylib_function[
            lib, "mrl_ig_input_text_hint",
            def(
                Ptr[c_char, MutUntrackedOrigin],
                Ptr[c_char, MutUntrackedOrigin],
                Ptr[c_char, MutUntrackedOrigin], c_int,
            ) thin -> Bool,
        ]()(_c(label), _c(hint), ptr, c_int(TextBuffer.CAP))
    return _get_dylib_function[
        lib, "mrl_ig_input_text",
        def(
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_char, MutUntrackedOrigin],
            c_int,
        ) thin -> Bool,
    ]()(_c(label), ptr, c_int(TextBuffer.CAP))


# ═══════════════════════════════════════════════════════════════════════════
# ImGuizmo — the transform gizmo
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠⚠ MATRICES ARE COLUMN-MAJOR float[16] — translation at [12][13][14], the
# OpenGL convention. `mojo_rl.render.gpu_types.mat4_to_gpu_f32` transposes
# this project's ROW-major `Mat4` into exactly that layout, so the caller
# hands the gizmo the same buffer it hands the GPU. Passing a row-major
# matrix does not fail: it draws a gizmo in a plausible wrong place, which is
# the hardest kind of wrong to see.
#
# ⚠ `gz_is_over()` / `gz_is_using()` ARE PART OF THE MOUSE ARBITRATION and
# `ig_want_mouse()` does NOT cover them — ImGuizmo's window is created with
# `NoInputs`, so ImGui truthfully reports that it does not want the mouse
# while the gizmo is being dragged. A host that gates only on `ig_want_mouse`
# orbits the camera and moves the part with one drag.


comptime GZ_TRANSLATE: Int = 7
"""TRANSLATE_X | TRANSLATE_Y | TRANSLATE_Z."""
comptime GZ_ROTATE: Int = 120
"""ROTATE_X | ROTATE_Y | ROTATE_Z | ROTATE_SCREEN."""
comptime GZ_SCALE: Int = 896
"""SCALE_X | SCALE_Y | SCALE_Z."""

comptime GZ_LOCAL: Int = 0
comptime GZ_WORLD: Int = 1


def gz_begin_frame() raises:
    """Open the gizmo's frame. Must follow `ig_new_frame()` every frame."""
    _get_dylib_function[lib, "mrl_gz_begin_frame", def() thin -> None]()()


def gz_set_rect(x: Float32, y: Float32, w: Float32, h: Float32) raises:
    """The VIEWPORT the gizmo hit-tests against, not the window.

    A host reserving a sidebar must subtract it here for the same reason its
    ray-pick does: a gizmo told about the full window is biased by half the
    missing strip, and the symptom reads as a projection bug.
    """
    _get_dylib_function[
        lib, "mrl_gz_set_rect",
        def(c_float, c_float, c_float, c_float) thin -> None,
    ]()(x, y, w, h)


def gz_set_orthographic(ortho: Bool) raises:
    _get_dylib_function[
        lib, "mrl_gz_set_orthographic", def(Bool) thin -> None
    ]()(ortho)


def gz_set_size(v: Float32 = 0.1) raises:
    """Gizmo radius as a fraction of clip space — distance-independent, so a
    30 cm arm and a 3 m quadruped get the same handle on screen."""
    _get_dylib_function[lib, "mrl_gz_set_size", def(c_float) thin -> None]()(v)


def gz_manipulate(
    view: List[Float32], proj: List[Float32], op: Int, mode: Int,
    mut matrix: List[Float32], snap: Float32 = 0.0,
) raises -> Bool:
    """Draw the gizmo at `matrix` and let the pointer move it. IN-OUT.

    Returns True on the frames the pointer actually moved it — NOT while it
    is merely hovered. `snap <= 0` disables snapping; the unit is metres for
    TRANSLATE and degrees for ROTATE, which is ImGuizmo's convention and the
    reason a single scalar suffices for both.
    """
    if len(view) < 16 or len(proj) < 16 or len(matrix) < 16:
        raise Error("gz_manipulate: view/proj/matrix must each hold 16 floats")
    # ⚠ ALWAYS A VALID BUFFER, and a FLAG for whether to use it. `Ptr` here
    # is a *safe* pointer with no null value, so "no snapping" cannot be
    # spelled as NULL without an unsafe cast at the call site; the C side
    # takes the flag instead.
    var s = [snap, snap, snap]
    return _get_dylib_function[
        lib, "mrl_gz_manipulate",
        def(
            Ptr[c_float, MutUntrackedOrigin], Ptr[c_float, MutUntrackedOrigin],
            c_int, c_int,
            Ptr[c_float, MutUntrackedOrigin], Ptr[c_float, MutUntrackedOrigin],
            c_int,
        ) thin -> Bool,
    ]()(
        untracked(view.unsafe_ptr()), untracked(proj.unsafe_ptr()),
        c_int(op), c_int(mode), untracked(matrix.unsafe_ptr()),
        untracked(s.unsafe_ptr()), c_int(1 if snap > 0.0 else 0),
    )


def gz_is_over() raises -> Bool:
    """True when the pointer is over a gizmo handle. See the header note."""
    return _get_dylib_function[lib, "mrl_gz_is_over", def() thin -> Bool]()()


def gz_is_using() raises -> Bool:
    """True while a handle is being dragged. See the header note."""
    return _get_dylib_function[lib, "mrl_gz_is_using", def() thin -> Bool]()()
