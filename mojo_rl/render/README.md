# render/ - Rendering Infrastructure

SDL3-based rendering with both 2D CPU rasterization and GPU-accelerated 3D rendering (Metal/Vulkan/DirectX).

## Module Structure

```
render/
├── types.mojo              # Color, SDL_Rect, SDL_Point, SDLHandle
├── colors.mojo             # 30+ named colors + rgb(), rgba(), lerp(), brighten(), darken()
├── transform.mojo          # Vec2, Transform2D, Camera, RotatingCamera (2D)
├── camera3d.mojo           # Camera3D: perspective look-at with orbit/pan/zoom
├── shapes.mojo             # 2D shape factories: rect, circle, polygon, arrow, star, diamond
├── shapes3d.mojo           # 3D wireframe: WireframeSphere, WireframeBox, WireframeCapsule
├── light.mojo              # Light: directional/point with shadow casting
├── gpu_types.mojo          # GPUVertex, SceneUniforms, ObjectUniforms, MeshData, MeshHandle
├── gpu_mesh.mojo           # GPU mesh generation: sphere, box, capsule, cone, torus, ground
├── gpu_shaders.mojo        # MSL Metal shaders (macOS): solid, ground, line, shadow, reflection, skybox, text
├── gpu_shaders_spirv.mojo  # SPIR-V shader loader (Linux/Vulkan): loads pre-compiled .spv bytecode
├── stl_loader.mojo         # STL mesh file loader (binary/ASCII) for custom 3D models
├── png_loader.mojo         # PNG texture loader via Python PIL for object texturing
├── font_atlas.mojo         # 8x8 bitmap font atlas for GPU text rendering
├── renderer2d.mojo         # Renderer2D: SDL3 2D renderer (fill_rect, draw_line, draw_circle, draw_text)
├── renderer3d.mojo         # Renderer3D: GPU-accelerated 3D renderer (Blinn-Phong, shadows, skybox)
├── video_recorder.mojo     # VideoRecorder: MP4/GIF + stills, by piping ffmpeg
├── shaders/                # Cross-platform GLSL shaders + compiled SPIR-V
│   ├── *.vert.glsl         # GLSL 450 vertex shaders (6 files)
│   ├── *.frag.glsl         # GLSL 450 fragment shaders (7 files)
│   ├── *.spv               # Pre-compiled SPIR-V bytecode (13 files)
│   └── compile.sh          # GLSL → SPIR-V compilation script (uses glslc)
└── sdl/                    # SDL3 FFI bindings (38 files)
    ├── sdl_init.mojo       # SDL initialization
    ├── sdl_video.mojo      # Window management
    ├── sdl_render.mojo     # 2D rendering API
    ├── sdl_gpu.mojo        # GPU API (Metal/Vulkan/DirectX)
    ├── sdl_events.mojo     # Event handling
    ├── sdl_keyboard.mojo   # Keyboard input
    ├── sdl_mouse.mojo      # Mouse input
    └── ...                 # Audio, gamepad, haptic, surface, pixels, etc.
```

## Renderer2D

SDL3-based 2D CPU rasterizer for classic RL environments.

**Features**: Rectangle, line, circle, polygon drawing; text rendering; event polling; video recording (V key); screenshot (S key)

**Used by**: CartPole, MountainCar, Acrobot, Pendulum, LunarLander, BipedalWalker, CarRacing

## Renderer3D

GPU-accelerated 3D renderer using SDL3 GPU API with cross-platform shaders (MSL on Metal, SPIR-V on Vulkan).

**Rendering pipeline** (6 passes):
1. Shadow map (depth-only)
2. Main pass (Blinn-Phong lit objects)
3. Ground (procedural checkerboard)
4. Reflections (Z-flipped darkened)
5. Skybox (fullscreen gradient)
6. Text HUD (bitmap font overlay)

**Features**:
- Cross-platform: MSL shaders on macOS (Metal), SPIR-V shaders on Linux (Vulkan) — selected at compile-time
- Deferred draw commands (sphere, box, capsule, cylinder, line, ground, text)
- STL mesh loading for custom 3D models (binary and ASCII formats)
- PNG texture support for object texturing (loaded via Python PIL)
- Mesh caching (static meshes + LRU capsule/cylinder cache)
- Up to 4 directional/point lights with shadow casting (PCF soft shadows)
- Interactive camera: orbit (left drag), pan (right drag), zoom (scroll)
- Playback control: pause (Space), step (->), reset camera (R)
- Video recording (V key), screenshot (S key)
- Camera switching (1-9 keys)
- Per-object materials: shininess, specular, emission, texture mapping
- Linear fog with configurable start/end distances

**Used by**: MuJoCo-style environments (HalfCheetah, Hopper, Ant, Walker2d, Swimmer, Humanoid, etc.)

## Cross-Platform Shaders

The renderer uses dual shader sources for cross-platform support:
- **macOS (Metal)**: MSL shaders embedded as comptime strings in `gpu_shaders.mojo`
- **Linux (Vulkan)**: GLSL 450 shaders in `shaders/`, pre-compiled to SPIR-V bytecode

Platform selection happens at compile-time via `comptime if CompilationTarget.is_macos()` — zero runtime branching.

### Recompiling SPIR-V shaders

After editing any `.glsl` file in `shaders/`, recompile to SPIR-V:

```bash
pixi run compile-shaders
```

This requires the `shaderc` package (already in `pixi.toml` dependencies). The script compiles all GLSL files to `.spv` using `glslc`.

### SDL3 SPIR-V binding conventions

Shaders follow the SDL3 GPU API descriptor set layout:
- **Vertex**: set 0 = textures/samplers, set 1 = uniform buffers
- **Fragment**: set 2 = textures/samplers, set 3 = uniform buffers

## Interactive Controls

| Key | Action |
|-----|--------|
| Space | Pause/resume |
| -> | Step one frame |
| R | Reset camera |
| S | Screenshot |
| V | Toggle video recording |
| 1-9 | Switch camera preset |
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
