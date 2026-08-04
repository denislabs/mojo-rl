// A flat C API over Dear ImGui, sized for Mojo FFI.
//
// WHY A SHIM AT ALL. Mojo cannot call C++ — no name mangling, no classes, no
// default arguments, no ImVec2 by value. Everything crossing the boundary here
// is a scalar, a `const char*` or a pointer to a scalar.
//
// WHY NOT cimgui. cimgui generates exactly this, for the whole API, but its
// pregenerated output does NOT cover the `sdlgpu3` RENDERER backend (only the
// SDL3 platform side), and regenerating it needs LuaJIT. Since a hand-written
// shim was needed for the six backend entry points anyway, curating ~60 widget
// wrappers alongside them costs less than a vendored generator and yields an
// API shaped for this project instead of a mechanical translation of all 2000+
// ImGui symbols.
//
// WHAT IS DELIBERATELY ABSENT. ImVec2/ImVec4 are split into scalars, default
// arguments are made explicit, and the `bool*` "p_open" outputs are dropped
// where nothing here closes a window. Add wrappers as they are needed — the
// point of this file is that adding one is three lines.
//
// ⚠ EVERY `mrl_ig_*` NAME IS AN FFI CONTRACT. `imgui.mojo` looks these up by
// string at runtime via dlsym. Renaming one here without renaming it there
// produces a *load-time abort*, not a compile error.

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlgpu3.h"
#include <SDL3/SDL.h>
#include <string.h>

extern "C" {

// ─── lifecycle ──────────────────────────────────────────────────────────────

// `window` and `device` are created on the MOJO side by Renderer3D; ImGui is a
// guest in a frame it does not own. `color_format` must be the swapchain
// format (Renderer3D already caches it).
bool mrl_ig_init(void* window, void* device, unsigned int color_format) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // ⚠ OFF BY DEFAULT. ImGui otherwise writes `imgui.ini` into the process's
    // CURRENT WORKING DIRECTORY, which for this project is the repo root. A
    // viewer that litters the source tree on every run is a bug; callers that
    // want persistence opt in with mrl_ig_set_ini_filename().
    io.IniFilename = nullptr;

    if (!ImGui_ImplSDL3_InitForSDLGPU((SDL_Window*)window)) return false;

    ImGui_ImplSDLGPU3_InitInfo info = {};
    info.Device = (SDL_GPUDevice*)device;
    info.ColorTargetFormat = (SDL_GPUTextureFormat)color_format;
    // The host renders ImGui in its OWN color-only pass (see renderer3d.mojo),
    // never inside the 3D pass, so MSAA and depth here are unconditionally 1x
    // and absent regardless of how the scene pass is configured.
    info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    return ImGui_ImplSDLGPU3_Init(&info);
}

void mrl_ig_shutdown(void) {
    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

void mrl_ig_set_ini_filename(const char* path) {
    // ImGui stores the POINTER, not a copy, so a Mojo String's buffer would
    // dangle. Copy into a static instead; one ini path per process is enough.
    static char s_ini[512];
    if (path == nullptr || path[0] == '\0') {
        ImGui::GetIO().IniFilename = nullptr;
        return;
    }
    strncpy(s_ini, path, sizeof(s_ini) - 1);
    s_ini[sizeof(s_ini) - 1] = '\0';
    ImGui::GetIO().IniFilename = s_ini;
}

void mrl_ig_new_frame(void) {
    ImGui_ImplSDLGPU3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

// Split because the SDL_GPU backend demands it: PrepareDrawData uploads the
// vertex/index buffers and needs the command buffer BEFORE any render pass is
// open; RenderDrawData needs the pass itself.
void mrl_ig_prepare(void* cmd_buf) {
    ImGui::Render();
    ImGui_ImplSDLGPU3_PrepareDrawData(ImGui::GetDrawData(),
                                      (SDL_GPUCommandBuffer*)cmd_buf);
}

void mrl_ig_render(void* cmd_buf, void* render_pass) {
    ImGui_ImplSDLGPU3_RenderDrawData(ImGui::GetDrawData(),
                                     (SDL_GPUCommandBuffer*)cmd_buf,
                                     (SDL_GPURenderPass*)render_pass, nullptr);
}

void mrl_ig_process_event(void* event) {
    ImGui_ImplSDL3_ProcessEvent((SDL_Event*)event);
}

// The two flags that let the host stop fighting the UI for input: when ImGui
// owns the pointer, dragging must not orbit the camera; when it owns the
// keyboard, "s" is a letter being typed, not the screenshot shortcut.
bool mrl_ig_want_mouse(void) { return ImGui::GetIO().WantCaptureMouse; }
bool mrl_ig_want_keyboard(void) { return ImGui::GetIO().WantCaptureKeyboard; }
float mrl_ig_framerate(void) { return ImGui::GetIO().Framerate; }

// ─── windows and layout ─────────────────────────────────────────────────────

// A panel pinned to an exact rect: the sidebar shape. Position and size are
// re-asserted every frame (SetNextWindow*), so the host's layout stays
// authoritative and nothing drifts if the window is resized.
void mrl_ig_begin_panel(const char* name, float x, float y, float w, float h) {
    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::Begin(name, nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoBringToFrontOnFocus);
}

// A free-floating window the user can move and resize; `first_*` apply only
// the first time it is seen (or after an ini reset).
bool mrl_ig_begin_window(const char* name, float first_x, float first_y,
                         float first_w, float first_h) {
    ImGui::SetNextWindowPos(ImVec2(first_x, first_y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(first_w, first_h), ImGuiCond_FirstUseEver);
    return ImGui::Begin(name, nullptr, 0);
}

void mrl_ig_end(void) { ImGui::End(); }

bool mrl_ig_begin_child(const char* id, float w, float h, bool border) {
    return ImGui::BeginChild(id, ImVec2(w, h),
                             border ? ImGuiChildFlags_Borders : 0);
}
void mrl_ig_end_child(void) { ImGui::EndChild(); }

void mrl_ig_separator(void) { ImGui::Separator(); }
void mrl_ig_separator_text(const char* label) { ImGui::SeparatorText(label); }
void mrl_ig_same_line(float offset_x, float spacing) {
    ImGui::SameLine(offset_x, spacing);
}
void mrl_ig_spacing(void) { ImGui::Spacing(); }
void mrl_ig_indent(float w) { ImGui::Indent(w); }
void mrl_ig_unindent(float w) { ImGui::Unindent(w); }
void mrl_ig_set_next_item_width(float w) { ImGui::SetNextItemWidth(w); }
float mrl_ig_content_width(void) {
    return ImGui::GetContentRegionAvail().x;
}

// ImGui identifies widgets by LABEL. Two buttons reading "reset" in the same
// window are the same widget and only one of them works. Pushing a distinct id
// around a loop body is the fix, and is why the tree/list code needs it.
void mrl_ig_push_id_int(int id) { ImGui::PushID(id); }
void mrl_ig_push_id_str(const char* id) { ImGui::PushID(id); }
void mrl_ig_pop_id(void) { ImGui::PopID(); }

// ─── text ───────────────────────────────────────────────────────────────────

// TextUnformatted, not Text: the string comes from Mojo and may contain '%'
// (a task name never does, but a printed reward or an error message can), and
// Text would treat it as a printf format and read past the arguments.
void mrl_ig_text(const char* s) { ImGui::TextUnformatted(s); }

void mrl_ig_text_colored(const char* s, float r, float g, float b, float a) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(r, g, b, a));
    ImGui::TextUnformatted(s);
    ImGui::PopStyleColor();
}

void mrl_ig_text_disabled(const char* s) {
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextUnformatted(s);
    ImGui::PopStyleColor();
}

void mrl_ig_text_wrapped(const char* s) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(s);
    ImGui::PopTextWrapPos();
}

// ─── widgets ────────────────────────────────────────────────────────────────

bool mrl_ig_button(const char* label, float w, float h) {
    return ImGui::Button(label, ImVec2(w, h));
}

bool mrl_ig_small_button(const char* label) {
    return ImGui::SmallButton(label);
}

// A button that renders as "on" when `active` — the radio-group shape the
// old hand-rolled layer spelled as `button(..., active=True)`.
bool mrl_ig_toggle_button(const char* label, bool active, float w, float h) {
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    bool hit = ImGui::Button(label, ImVec2(w, h));
    if (active) ImGui::PopStyleColor();
    return hit;
}

bool mrl_ig_selectable(const char* label, bool selected) {
    return ImGui::Selectable(label, selected);
}

bool mrl_ig_checkbox(const char* label, bool* v) {
    return ImGui::Checkbox(label, v);
}

bool mrl_ig_radio(const char* label, bool active) {
    return ImGui::RadioButton(label, active);
}

bool mrl_ig_slider_float(const char* label, float* v, float lo, float hi,
                         const char* fmt) {
    return ImGui::SliderFloat(label, v, lo, hi, fmt ? fmt : "%.3f");
}

bool mrl_ig_slider_int(const char* label, int* v, int lo, int hi) {
    return ImGui::SliderInt(label, v, lo, hi);
}

bool mrl_ig_drag_float(const char* label, float* v, float speed, float lo,
                       float hi, const char* fmt) {
    return ImGui::DragFloat(label, v, speed, lo, hi, fmt ? fmt : "%.3f");
}

// `items` is ImGui's zero-separated, double-zero-terminated form:
// "zero\0random\0sweep\0". Building that in Mojo is a join, not a struct.
bool mrl_ig_combo(const char* label, int* current, const char* items) {
    return ImGui::Combo(label, current, items);
}

bool mrl_ig_input_text(const char* label, char* buf, int buf_size) {
    return ImGui::InputText(label, buf, (size_t)buf_size);
}

// InputText with a greyed prompt when empty. ImGui has no native placeholder,
// and the filter box is unusable without one.
bool mrl_ig_input_text_hint(const char* label, const char* hint, char* buf,
                            int buf_size) {
    return ImGui::InputTextWithHint(label, hint, buf, (size_t)buf_size);
}

bool mrl_ig_tree_node(const char* label) { return ImGui::TreeNode(label); }
void mrl_ig_tree_pop(void) { ImGui::TreePop(); }
void mrl_ig_set_next_item_open(bool open) { ImGui::SetNextItemOpen(open); }

bool mrl_ig_collapsing_header(const char* label, bool default_open) {
    return ImGui::CollapsingHeader(
        label, default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
}

void mrl_ig_plot_lines(const char* label, const float* values, int count,
                       int offset, float lo, float hi, float w, float h) {
    ImGui::PlotLines(label, values, count, offset, nullptr, lo, hi,
                     ImVec2(w, h));
}

void mrl_ig_progress_bar(float frac, float w, float h, const char* overlay) {
    ImGui::ProgressBar(frac, ImVec2(w, h), overlay);
}

void mrl_ig_set_scroll_here_y(float ratio) { ImGui::SetScrollHereY(ratio); }

bool mrl_ig_is_item_hovered(void) { return ImGui::IsItemHovered(); }

void mrl_ig_set_tooltip(const char* s) {
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", s);
}

// ─── style ──────────────────────────────────────────────────────────────────

void mrl_ig_style_dark(void) { ImGui::StyleColorsDark(); }
void mrl_ig_style_light(void) { ImGui::StyleColorsLight(); }
void mrl_ig_style_classic(void) { ImGui::StyleColorsClassic(); }

// Global scale for the whole UI, for HiDPI or simply for readability.
void mrl_ig_set_font_scale(float s) {
    ImGui::GetIO().FontGlobalScale = s;
}

}  // extern "C"
