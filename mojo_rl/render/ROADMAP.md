---                                                                                                                                                                                              
  Proposal: Interactive Renderer Features                                                                                                                                                          
                                                                                                                                                                                                   
  Current State                                                                                                                                                                                    
                                                                                                                                                                                                   
  The codebase already has most of the building blocks — just not wired up:

  - orbit_camera(), zoom_camera(), pan() exist on Camera3D / Renderer3D / ModelRenderer
  - Camera switch via keys 1–9 is already working
  - SDL3 mouse event types (MouseMotionEvent, MouseButtonEvent, MouseWheelEvent) are fully available in the wrappers
  - Gap: check_quit() only handles Escape + digits. No mouse state tracking, no additional key handling.
  - Gap: Renderer3D is a pure GPU pipeline (Metal/SDL3 GPU). It has zero text rendering capability — text exists only in Renderer2D via SDL3's CPU debug font.

  ---
  Feature 1: Mouse Camera Control

  Effort: Low — the methods already exist, just need to be wired

  ┌──────────────┬───────────────────────────┬──────────────────────────────────────┐
  │   Gesture    │          Action           │                Notes                 │
  ├──────────────┼───────────────────────────┼──────────────────────────────────────┤
  │ Left drag    │ Orbit (azimuth/elevation) │ Uses existing orbit_camera()         │
  ├──────────────┼───────────────────────────┼──────────────────────────────────────┤
  │ Right drag   │ Pan (move look-at point)  │ Needs pan() exposed on ModelRenderer │
  ├──────────────┼───────────────────────────┼──────────────────────────────────────┤
  │ Scroll wheel │ Zoom in/out               │ Uses existing zoom_camera()          │
  └──────────────┴───────────────────────────┴──────────────────────────────────────┘

  Changes needed:
  - Add 4 fields to Renderer3D: mouse_left_down: Bool, mouse_right_down: Bool, last_mouse_x: Float32, last_mouse_y: Float32
  - In check_quit(), add 3 new event cases:
    - EVENT_MOUSE_BUTTON_DOWN/UP → set mouse_left_down / mouse_right_down, save last_mouse_x/y
    - EVENT_MOUSE_MOTION → if left down: orbit(xrel * 0.3, yrel * 0.3); if right down: call pan
    - EVENT_MOUSE_WHEEL → zoom(-wheel.y * 0.5)
  - Expose pan_camera() on Renderer3D and ModelRenderer

  ---
  Feature 2: Pause / Step

  Effort: Low — pure state + key handling

  ┌─────────────────┬──────────────────────────────────┐
  │       Key       │              Action              │
  ├─────────────────┼──────────────────────────────────┤
  │ Space           │ Toggle pause/run                 │
  ├─────────────────┼──────────────────────────────────┤
  │ → (Right arrow) │ Step one frame (while paused)    │
  ├─────────────────┼──────────────────────────────────┤
  │ R               │ Reset camera to default position │
  └─────────────────┴──────────────────────────────────┘

  Changes needed:
  - Add var is_paused: Bool and var step_once: Bool to Renderer3D
  - In check_quit(): handle SDLK_SPACE, SDLK_RIGHT, SDLK_R
  - Simulation loops (e.g., half_cheetah_gc.mojo) check renderer.is_paused before calling env.step()

  ---
  Feature 3: HUD Overlay Menu

  Effort: Medium-High — Renderer3D has no text capability, two options:

  Option A — SDL3 secondary renderer (simpler code):
  Create a small SDL_Renderer (CPU-side) on the same window just for text, after each GPU frame. Technically supported in SDL3 with careful ordering (SDL_FlushRenderer before GPU acquire). Works
  immediately with existing render_debug_text bindings. Risk: possible visual artifacts or driver-specific issues mixing GPU + CPU renderers.

  Option B — GPU bitmap font atlas (proper, clean):
  Bake a 128-glyph ASCII font into a constant byte array (standard 8×16 monospace bitmap font, ~2KB), upload once as a GPU texture, add a draw_text_gpu(text, x, y, color) method that emits
  screen-space textured quads in a post-3D pass. Zero mixing issues, works everywhere, proper antialiasing. About ~200 lines of new shader + Mojo code.

  Recommendation: Option B — the renderer is already complex GPU code, adding a clean font atlas shader is the right approach and avoids fragility. Option A is quicker to prototype.

  The menu would display:
  [Space]  Pause / Run          On
  [→]      Step frame
  [1-9]    Switch camera        Cam 1
  [R]      Reset camera
  [S]      Screenshot
  [H]      Hide menu

  FPS:     120
  Step:    4231

  ---
  Feature 4: Screenshot

  Effort: Medium — requires GPU readback

  SDL3 GPU provides SDL_DownloadFromGPUTexture to read back the swapchain texture after render. Then write to BMP (SDL3 built-in) or PNG (requires stb_image_write, ~1 file). Triggered by S key or
   a dedicated button.

  ---
  Implementation Order (recommended)

  ┌──────────┬───────────────────────────────┬────────────┬─────────────────────────────────┐
  │ Priority │            Feature            │   Effort   │              Value              │
  ├──────────┼───────────────────────────────┼────────────┼─────────────────────────────────┤
  │ 1        │ Mouse orbit + zoom + pan      │ ~30 lines  │ Immediate usability improvement │
  ├──────────┼───────────────────────────────┼────────────┼─────────────────────────────────┤
  │ 2        │ Pause / Step / Reset          │ ~20 lines  │ Essential for debugging         │
  ├──────────┼───────────────────────────────┼────────────┼─────────────────────────────────┤
  │ 3        │ HUD overlay (Option A first)  │ ~100 lines │ Visual polish                   │
  ├──────────┼───────────────────────────────┼────────────┼─────────────────────────────────┤
  │ 4        │ HUD overlay (Option B proper) │ ~300 lines │ Long-term correctness           │
  ├──────────┼───────────────────────────────┼────────────┼─────────────────────────────────┤
  │ 5        │ Screenshot                    │ ~80 lines  │ Nice-to-have                    │
  └──────────┴───────────────────────────────┴────────────┴─────────────────────────────────┘

  ---