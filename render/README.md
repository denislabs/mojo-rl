# render/ - Rendering Infrastructure

SDL3-based rendering with both 2D CPU rasterization and GPU-accelerated 3D rendering (Metal/Vulkan/DirectX).

## Module Structure

```
render/
├── types.mojo          # Color, SDL_Rect, SDL_Point, SDLHandle
├── colors.mojo         # 30+ named colors + rgb(), rgba(), lerp(), brighten(), darken()
├── transform.mojo      # Vec2, Transform2D, Camera, RotatingCamera (2D)
├── camera3d.mojo       # Camera3D: perspective look-at with orbit/pan/zoom
├── shapes.mojo         # 2D shape factories: rect, circle, polygon, arrow, star, diamond
├── shapes3d.mojo       # 3D wireframe: WireframeSphere, WireframeBox, WireframeCapsule
├── light.mojo          # Light: directional/point with shadow casting
├── gpu_types.mojo      # GPUVertex, SceneUniforms, ObjectUniforms, MeshData, MeshHandle
├── gpu_mesh.mojo       # GPU mesh generation: sphere, box, capsule, cone, torus, ground
├── gpu_shaders.mojo    # MSL Metal shaders: solid, ground, line, shadow, reflection, skybox, text
├── font_atlas.mojo     # 8x8 bitmap font atlas for GPU text rendering
├── renderer2d.mojo     # Renderer2D: SDL3 2D renderer (fill_rect, draw_line, draw_circle, draw_text)
├── renderer3d.mojo     # Renderer3D: GPU-accelerated 3D renderer (Blinn-Phong, shadows, skybox)
├── video_recorder.mojo # VideoRecorder: MP4/GIF encoding via Python imageio
└── sdl/                # SDL3 FFI bindings (38 files)
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

GPU-accelerated 3D renderer using SDL3 GPU API with Metal Shading Language.

**Rendering pipeline** (6 passes):
1. Shadow map (depth-only)
2. Main pass (Blinn-Phong lit objects)
3. Ground (procedural checkerboard)
4. Reflections (Z-flipped darkened)
5. Skybox (fullscreen gradient)
6. Text HUD (bitmap font overlay)

**Features**:
- Deferred draw commands (sphere, box, capsule, line, ground, text)
- Mesh caching (static meshes + LRU capsule cache)
- Up to 4 directional/point lights with shadow casting
- Interactive camera: orbit (left drag), pan (right drag), zoom (scroll)
- Playback control: pause (Space), step (->), reset camera (R)
- Video recording (V key), screenshot (S key)
- Camera switching (1-9 keys)

**Used by**: MuJoCo-style environments (HalfCheetah, Hopper, Ant, Walker2d, etc.)

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
