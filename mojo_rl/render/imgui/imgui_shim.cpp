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
#include "ImGuizmo.h"
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

// ── menu bar ────────────────────────────────────────────────────────────────
// The main menu bar is a VIEWPORT-level strip, not a window: ImGui reserves
// space for it at the top of the frame, so panels positioned at y=0 must
// start below `ig_main_menu_height()`. Getting that wrong hides the first row
// of whatever panel is topmost, which reads as a clipped panel rather than as
// a menu-bar offset.
bool mrl_ig_begin_main_menu_bar(void) { return ImGui::BeginMainMenuBar(); }
void mrl_ig_end_main_menu_bar(void) { ImGui::EndMainMenuBar(); }
bool mrl_ig_begin_menu(const char* label) { return ImGui::BeginMenu(label); }
void mrl_ig_end_menu(void) { ImGui::EndMenu(); }
bool mrl_ig_menu_item(const char* label, const char* shortcut, bool selected,
                      bool enabled) {
    return ImGui::MenuItem(label, shortcut[0] ? shortcut : nullptr, selected,
                           enabled);
}
float mrl_ig_frame_height_with_spacing(void) {
    return ImGui::GetFrameHeightWithSpacing();
}

// ── tab bar ─────────────────────────────────────────────────────────────────
// ⚠ `EndTabItem` IS CALLED ONLY WHEN `BeginTabItem` RETURNED TRUE, unlike the
// Begin/End pairs above. ImGui's own asserts catch the mistake, but only in a
// debug build; in a release shim it corrupts the id stack silently.
bool mrl_ig_begin_tab_bar(const char* id) { return ImGui::BeginTabBar(id); }
void mrl_ig_end_tab_bar(void) { ImGui::EndTabBar(); }
bool mrl_ig_begin_tab_item(const char* label) {
    return ImGui::BeginTabItem(label);
}
void mrl_ig_end_tab_item(void) { ImGui::EndTabItem(); }

// ── two-column key/value ────────────────────────────────────────────────────
// `Columns` rather than `BeginTable`: the inspector wants a label column and a
// value column with a draggable split, which is all the legacy API does, and
// it is four symbols instead of a table's eight.
void mrl_ig_columns(int count, bool border) {
    ImGui::Columns(count, nullptr, border);
}
void mrl_ig_next_column(void) { ImGui::NextColumn(); }
void mrl_ig_set_column_width(int i, float w) { ImGui::SetColumnWidth(i, w); }

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

// ─── ImGuizmo — the transform gizmo ─────────────────────────────────────────
//
// ⚠⚠ IT SHARES THIS TRANSLATION UNIT'S ImGui CONTEXT ON PURPOSE. ImGuizmo
// draws through an ImGui draw list and hit-tests against ImGui's `io`; built
// into a second dylib it would link a second copy of ImGui's globals and the
// gizmo would render into a frame nobody submits. One shim, one context.
//
// ⚠ MATRICES ARE COLUMN-MAJOR float[16], the OpenGL convention — translation
// at [12][13][14]. `mat4_to_gpu_f32` on the Mojo side already transposes
// row-major `Mat4` into exactly this, so the studio hands over the SAME
// buffer it hands the GPU. A row-major matrix passed here does not fail; it
// draws a gizmo in a plausible wrong place.

void mrl_gz_begin_frame(void) { ImGuizmo::BeginFrame(); }

// The VIEWPORT, not the window: the studio reserves a strip on each side for
// its panels, and a gizmo hit-tested against the full window is offset by
// half the missing strip — the same bias the ray-pick had to avoid.
void mrl_gz_set_rect(float x, float y, float w, float h) {
    ImGuizmo::SetRect(x, y, w, h);
}

void mrl_gz_set_orthographic(bool ortho) { ImGuizmo::SetOrthographic(ortho); }

// Gizmo radius as a fraction of clip space (default 0.1). A model 30 cm
// across and one 3 m across want the same on-screen size, which this gives:
// it is resolution-independent and distance-independent by construction.
void mrl_gz_set_size(float v) { ImGuizmo::SetGizmoSizeClipSpace(v); }

// `matrix` is IN-OUT.
//
// ⚠ `use_snap` RATHER THAN A NULLABLE `snap`. ImGuizmo takes NULL to mean
// "no snapping", but the Mojo side's `Ptr` is a *safe* pointer with no null
// value, so expressing the absence across the boundary would need an unsafe
// cast at every call. A flag plus an always-valid buffer costs one int and
// keeps the binding honest.
bool mrl_gz_manipulate(const float* view, const float* proj, int op, int mode,
                       float* matrix, const float* snap, int use_snap) {
    return ImGuizmo::Manipulate(view, proj,
                                (ImGuizmo::OPERATION)op,
                                (ImGuizmo::MODE)mode,
                                matrix, nullptr,
                                use_snap ? snap : nullptr);
}

// ⚠⚠ THESE ARE THE MOUSE ARBITRATION, and `mrl_ig_want_mouse` does NOT cover
// them. ImGuizmo's own window carries `ImGuiWindowFlags_NoInputs`, so ImGui
// reports it does not want the mouse while the gizmo is being dragged —
// without these two the same drag orbits the camera and moves the part.
bool mrl_gz_is_over(void) { return ImGuizmo::IsOver(); }
bool mrl_gz_is_using(void) { return ImGuizmo::IsUsing(); }

// ─── textures: showing a camera frame (or any RGBA image) in a window ────────
//
// ⚠ `ImTextureID` IS AN `SDL_GPUTexture*` IN THIS BACKEND, and that is what
// makes this small: the backend supplies the sampler itself
// (`bd->CurrentSampler`), so a user texture is just a texture — no descriptor
// set, no binding object.  It changed in ImGui 1.92.2 (it used to be an
// `SDL_GPUTextureSamplerBinding*`), so older examples pass the wrong thing.
//
// The transfer buffer is kept ALIVE with the texture rather than created per
// upload.  A camera at 30 Hz would otherwise allocate and free a GPU staging
// buffer thirty times a second for the lifetime of the window.

struct MrlIgTexture {
    SDL_GPUDevice*         device;
    SDL_GPUTexture*        tex;
    SDL_GPUTransferBuffer* xfer;
    int                    w;
    int                    h;
};

void* mrl_ig_texture_create(void* device, int w, int h) {
    if (device == nullptr || w <= 0 || h <= 0) return nullptr;
    SDL_GPUDevice* dev = (SDL_GPUDevice*)device;

    SDL_GPUTextureCreateInfo ti = {};
    ti.type                 = SDL_GPU_TEXTURETYPE_2D;
    ti.format               = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    ti.usage                = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    ti.width                = (Uint32)w;
    ti.height               = (Uint32)h;
    ti.layer_count_or_depth = 1;
    ti.num_levels           = 1;
    SDL_GPUTexture* tex = SDL_CreateGPUTexture(dev, &ti);
    if (tex == nullptr) return nullptr;

    SDL_GPUTransferBufferCreateInfo xi = {};
    xi.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    xi.size  = (Uint32)(w * h * 4);
    SDL_GPUTransferBuffer* xfer = SDL_CreateGPUTransferBuffer(dev, &xi);
    if (xfer == nullptr) {
        SDL_ReleaseGPUTexture(dev, tex);
        return nullptr;
    }

    MrlIgTexture* t = new MrlIgTexture();
    t->device = dev;
    t->tex    = tex;
    t->xfer   = xfer;
    t->w      = w;
    t->h      = h;
    return t;
}

// `rgba` is w*h*4 bytes, tightly packed.
//
// ⚠ THIS RUNS ITS OWN COMMAND BUFFER, deliberately.  SDL_GPU forbids a copy
// pass while a render pass is open, and this is called from application code
// that has no idea where the renderer is in its frame.  One submit per upload
// is the price of that safety, and at one camera frame per display frame it is
// not a cost worth engineering away.
bool mrl_ig_texture_upload(void* handle, const unsigned char* rgba) {
    MrlIgTexture* t = (MrlIgTexture*)handle;
    if (t == nullptr || rgba == nullptr) return false;

    void* dst = SDL_MapGPUTransferBuffer(t->device, t->xfer, false);
    if (dst == nullptr) return false;
    memcpy(dst, rgba, (size_t)(t->w * t->h * 4));
    SDL_UnmapGPUTransferBuffer(t->device, t->xfer);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(t->device);
    if (cmd == nullptr) return false;
    SDL_GPUCopyPass* pass = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTextureTransferInfo src = {};
    src.transfer_buffer = t->xfer;
    src.offset          = 0;
    src.pixels_per_row  = (Uint32)t->w;
    src.rows_per_layer  = (Uint32)t->h;

    SDL_GPUTextureRegion dstr = {};
    dstr.texture = t->tex;
    dstr.w       = (Uint32)t->w;
    dstr.h       = (Uint32)t->h;
    dstr.d       = 1;

    SDL_UploadToGPUTexture(pass, &src, &dstr, false);
    SDL_EndGPUCopyPass(pass);
    SDL_SubmitGPUCommandBuffer(cmd);
    return true;
}

void mrl_ig_image(void* handle, float w, float h) {
    MrlIgTexture* t = (MrlIgTexture*)handle;
    if (t == nullptr) return;
    ImGui::Image((ImTextureID)(intptr_t)t->tex, ImVec2(w, h));
}

// Where the last `mrl_ig_image` landed on screen, in window coordinates, so a
// caller can draw an overlay over it without duplicating ImGui's layout rules.
void mrl_ig_last_item_rect(float* x, float* y, float* w, float* h) {
    ImVec2 mn = ImGui::GetItemRectMin();
    ImVec2 mx = ImGui::GetItemRectMax();
    if (x) *x = mn.x;
    if (y) *y = mn.y;
    if (w) *w = mx.x - mn.x;
    if (h) *h = mx.y - mn.y;
}

// A line on the CURRENT window's foreground draw list, in window coordinates.
// Used to draw detected marker corners over the image.
void mrl_ig_overlay_line(float x0, float y0, float x1, float y1,
                         unsigned int rgba, float thickness) {
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x0, y0), ImVec2(x1, y1),
                                        rgba, thickness);
}

void mrl_ig_texture_destroy(void* handle) {
    MrlIgTexture* t = (MrlIgTexture*)handle;
    if (t == nullptr) return;
    if (t->xfer) SDL_ReleaseGPUTransferBuffer(t->device, t->xfer);
    if (t->tex)  SDL_ReleaseGPUTexture(t->device, t->tex);
    delete t;
}

}  // extern "C"
