# Atari (ALE) — Pure Mojo GPU Port

Port the Arcade Learning Environment to pure Mojo, running the Atari 2600 emulator
**on GPU** (one game per thread, CuLE-style). No Python dependency.

## Reference Material

- **CuLE** (BSD-3): NVIDIA's CUDA Atari emulator — primary reference for GPU architecture
  - `cule/atari/` — 6502, TIA, RIOT, opcode tables, game detection
  - One thread per environment, ~300 bytes state per instance
  - 155M frames/hour on a single GPU
- **ALE** (GPL-2): Original C++ emulator (Stella fork) — reference for correctness
  - `src/ale/emucore/` — full Stella emulator
  - `src/ale/games/` — 104 game-specific ROM support files
- **Stella** — the upstream Atari 2600 emulator ALE is built on

## Architecture Overview

```
envs/atari/
├── __init__.mojo           # Re-exports
├── cpu6502.mojo            # MOS 6502 CPU emulation (~1000 lines)
├── tia.mojo                # TIA graphics/audio chip (~1200 lines)
├── riot.mojo               # RIOT timer/IO/RAM (~250 lines)
├── opcodes.mojo            # 256-entry opcode dispatch table (~1250 lines)
├── cartridge.mojo          # ROM loading + bank switching (~400 lines)
├── atari_state.mojo        # Per-instance state struct (~300 bytes)
├── atari_env.mojo          # GPUDiscreteEnv implementation
├── preprocessing.mojo      # GPU frame preprocessing kernels
├── palette.mojo            # NTSC/PAL color palette (128 → RGB)
├── renderer.mojo           # SDL3 texture-based Atari renderer
└── games/                  # Per-game ROM support
    ├── game_trait.mojo      # GameDef trait (lives, score, terminal from RAM)
    ├── pong.mojo
    ├── breakout.mojo
    ├── space_invaders.mojo
    ├── seaquest.mojo
    ├── qbert.mojo
    ├── beam_rider.mojo
    ├── enduro.mojo
    ├── freeway.mojo
    ├── montezuma.mojo
    └── asteroids.mojo
```

## Atari 2600 Hardware — What We're Emulating

The Atari 2600 has 3 chips:

### 1. MOS 6502 CPU (`cpu6502.mojo`)
- 8-bit processor, 16-bit address bus
- 6 registers: PC (16-bit), A/X/Y/SP (8-bit), Status flags (8-bit)
- 56 mnemonics, 151 official opcodes, 13 addressing modes
- Deterministic cycle counts (no pipeline, no cache)
- CuLE implements this in 773 lines — straightforward to port

### 2. TIA — Television Interface Adapter (`tia.mojo`)
- **The hard part.** Generates all video output.
- 2 player sprites (8px wide), 2 missiles, 1 ball, asymmetric playfield
- 15 collision registers, horizontal motion (HMOVE), VBLANK/VSYNC
- Games exploit undocumented edge-case behaviors
- CuLE simplified Stella's 4665 lines → 1034 lines. Use CuLE's version.

### 3. RIOT — 6532 RAM/IO/Timer (`riot.mojo`)
- 128 bytes of RAM (the entire system RAM)
- Programmable interval timer (4 prescaler values)
- Two 8-bit I/O ports (controllers, console switches)
- CuLE: 234 lines. Simplest chip.

## Phase 1: CPU-Only Emulator Core

### Step 1: Atari State Struct (`atari_state.mojo`)

All per-instance state in a flat struct (~300 bytes), GPU-friendly:

```
struct AtariState:
    # 6502 CPU registers (7 bytes)
    var pc: UInt16          # Program counter
    var a: UInt8            # Accumulator
    var x: UInt8            # X index
    var y: UInt8            # Y index
    var sp: UInt8           # Stack pointer
    var flags: UInt8        # Status flags (N,V,-,B,D,I,Z,C)

    # RIOT (133 bytes)
    var ram: InlineArray[UInt8, 128]  # System RAM
    var timer: UInt32       # Timer value
    var timer_prescale: UInt8
    var io_port_a: UInt8    # Controller input
    var io_port_b: UInt8    # Console switches

    # TIA state (~80 bytes)
    var grp0: UInt8         # Player 0 graphics
    var grp1: UInt8         # Player 1 graphics
    var pf0: UInt8          # Playfield registers
    var pf1: UInt8
    var pf2: UInt8
    var colup0: UInt8       # Player 0 color
    var colup1: UInt8       # Player 1 color
    var colupf: UInt8       # Playfield color
    var colubk: UInt8       # Background color
    var pos_p0: UInt8       # Player 0 horizontal position
    var pos_p1: UInt8
    var pos_m0: UInt8       # Missile 0 position
    var pos_m1: UInt8
    var pos_bl: UInt8       # Ball position
    var hm_p0: UInt8        # Horizontal motion registers
    var hm_p1: UInt8
    var hm_m0: UInt8
    var hm_m1: UInt8
    var hm_bl: UInt8
    var ctrlpf: UInt8       # Playfield control
    var nusiz0: UInt8       # Number-size player 0
    var nusiz1: UInt8
    var collision: UInt16   # 15 collision flags packed
    var vblank: UInt8
    var vsync: UInt8
    # ... (CuLE has full list)

    # Frame state
    var scanline: UInt16    # Current scanline (0-261)
    var clock: UInt16       # Clock within scanline (0-227)
    var frame_complete: Bool

    # RL state
    var reward: Float32
    var lives: UInt8
    var terminal: Bool
    var frame_number: UInt32
```

Total: ~300 bytes per instance. With 2048 parallel envs: ~600KB (fits in L2 cache).

### Step 2: Opcode Table (`opcodes.mojo`)

256-entry table mapping opcode byte → (handler, addressing mode, cycles):

```
struct OpcodeEntry:
    var handler: UInt8      # Which instruction (LDA, STA, ADC, ...)
    var addr_mode: UInt8    # Immediate, ZeroPage, Absolute, ...
    var cycles: UInt8       # Base cycle count

# The 13 addressing modes
comptime ADDR_IMPLIED = 0
comptime ADDR_IMMEDIATE = 1
comptime ADDR_ZERO_PAGE = 2
comptime ADDR_ZERO_PAGE_X = 3
comptime ADDR_ZERO_PAGE_Y = 4
comptime ADDR_ABSOLUTE = 5
comptime ADDR_ABSOLUTE_X = 6
comptime ADDR_ABSOLUTE_Y = 7
comptime ADDR_INDIRECT = 8
comptime ADDR_INDIRECT_X = 9
comptime ADDR_INDIRECT_Y = 10
comptime ADDR_RELATIVE = 11
comptime ADDR_ACCUMULATOR = 12
```

CuLE's `opcodes.cpp` (1250 lines) has the complete table — direct translation.

### Step 3: 6502 CPU (`cpu6502.mojo`)

Core fetch-decode-execute loop:

```
fn execute_instruction(mut state: AtariState, rom: UnsafePointer[UInt8]):
    var opcode = read_byte(state, state.pc, rom)
    state.pc += 1
    var entry = OPCODE_TABLE[int(opcode)]

    # Resolve operand address based on addressing mode
    var addr = resolve_address(state, entry.addr_mode, rom)

    # Execute instruction
    if entry.handler == OP_LDA: state.a = read_byte(state, addr, rom); update_nz(state, state.a)
    elif entry.handler == OP_STA: write_byte(state, addr, state.a, rom)
    elif entry.handler == OP_ADC: adc(state, read_byte(state, addr, rom))
    elif entry.handler == OP_JMP: state.pc = addr
    # ... 53 more instructions
```

The memory map routes reads/writes to RAM (RIOT), TIA registers, or ROM:
- `$0000-$007F`: TIA registers (write) / TIA read registers
- `$0080-$00FF`: RIOT RAM (128 bytes)
- `$0280-$0297`: RIOT registers
- `$1000-$1FFF`: ROM (cartridge, possibly bankswitched)

### Step 4: TIA (`tia.mojo`)

The TIA runs at 3x CPU clock (228 color clocks per scanline, 262 scanlines per frame).
Key operations:
- **Register writes**: Games write TIA registers to position sprites, set colors, etc.
- **Scanline rendering**: For each color clock, determine pixel color from playfield, sprites, ball, missiles with priority
- **Collision detection**: Update 15 collision bits as objects overlap
- **HMOVE**: Horizontal motion applied during HBLANK

CuLE's approach: don't render full frames unless needed. For RL, we only need:
1. The **RAM state** (128 bytes) — games store score/lives here
2. Optionally the **screen buffer** (160x192 pixels) for pixel observations

### Step 5: RIOT (`riot.mojo`)

Timer with prescaler values (1, 8, 64, 1024):
```
fn update_timer(mut state: AtariState, cycles: UInt8):
    state.timer -= int(cycles)
    if state.timer <= 0:
        state.timer_prescale = 1  # After underflow, counts at 1x
```

### Step 6: Cartridge / ROM Loading (`cartridge.mojo`)

Most Atari games use simple ROM mapping. Common mapper types:
- **2K/4K**: Direct mapping (no banking) — covers Pong, Breakout, Space Invaders
- **F8** (8K): Two 4K banks, switched by accessing $1FF8/$1FF9
- **F6** (16K): Four 4K banks
- **E0** (8K Parker Bros): Eight 1K banks

For the initial 10 games, only 2-3 mapper types needed.

ROM data stored in a flat buffer, indexed per environment for GPU:
```
# ROM buffer layout: [MAX_ENVS, MAX_ROM_SIZE]
# Each env gets a copy of its ROM (or shared if all playing same game)
```

### Step 7: Game Detection (`games/`)

Each game reads specific RAM addresses to extract RL signals:

```
trait GameDef:
    comptime GAME_ID: Int
    comptime MIN_ACTIONS: Int       # Size of minimal action set
    comptime ACTION_MAP: ???        # Maps [0..MIN_ACTIONS) → ALE action indices

    @staticmethod
    fn get_reward(ram: InlineArray[UInt8, 128]) -> Float32
    @staticmethod
    fn get_lives(ram: InlineArray[UInt8, 128]) -> UInt8
    @staticmethod
    fn is_terminal(ram: InlineArray[UInt8, 128], lives: UInt8) -> Bool
```

Example (Pong):
```
struct PongDef(GameDef):
    comptime GAME_ID = 0
    comptime MIN_ACTIONS = 6  # NOOP, FIRE, UP, DOWN, UPFIRE, DOWNFIRE

    @staticmethod
    fn get_reward(ram: InlineArray[UInt8, 128]) -> Float32:
        # RAM[13] = player score, RAM[14] = CPU score
        return Float32(ram[13]) - Float32(ram[14])

    @staticmethod
    fn get_lives(ram: InlineArray[UInt8, 128]) -> UInt8:
        return ram[13]  # Pong doesn't have lives, use score
```

CuLE's `games/` directory has these RAM mappings for 63 games.

## Phase 2: GPU Kernels

### Step 8: GPU Step Kernel

One thread per environment, runs one frame (or N frames with frame skip):

```
fn atari_step_kernel[BATCH: Int, FRAME_SKIP: Int](
    states: UnsafePointer[AtariState],      # [BATCH]
    roms: UnsafePointer[UInt8],             # [BATCH, ROM_SIZE]
    actions: UnsafePointer[UInt8],          # [BATCH]
    rewards: UnsafePointer[Float32],        # [BATCH] output
    terminals: UnsafePointer[Bool],         # [BATCH] output
    obs: UnsafePointer[UInt8],              # [BATCH, 210, 160, 3] output (optional)
):
    var env_id = block_idx.x * block_dim.x + thread_idx.x
    if env_id >= BATCH:
        return

    var state = states[env_id]
    var rom_ptr = roms + env_id * ROM_SIZE
    var total_reward: Float32 = 0.0

    # Set action in RIOT I/O port
    set_action(state, actions[env_id])

    # Run FRAME_SKIP frames
    for frame in range(FRAME_SKIP):
        # Execute one frame (~262 scanlines × 76 CPU cycles = ~19912 cycles)
        while not state.frame_complete:
            execute_instruction(state, rom_ptr)
            update_tia(state, instruction_cycles)
            update_timer(state, instruction_cycles)

        state.frame_complete = False
        total_reward += get_reward(state)  # Game-specific RAM read

    rewards[env_id] = total_reward
    terminals[env_id] = is_terminal(state)
    states[env_id] = state  # Write back
```

Each frame is ~20K CPU cycles ≈ ~1000 instructions. With frame skip 4: ~4000 instructions per step.
At ~1000 FLOPS per thread on modern GPUs, expect ~40K+ steps/sec with 2048 envs.

### Step 9: GPU Frame Preprocessing (`preprocessing.mojo`)

When pixel observations are needed (not RAM-only mode):

```
# Kernel 1: TIA color index → RGB (per pixel)
fn palette_kernel[BATCH: Int](
    tia_buffer: UnsafePointer[UInt8],    # [BATCH, 210, 160] color indices
    rgb_buffer: UnsafePointer[UInt8],    # [BATCH, 210, 160, 3] output
    palette: UnsafePointer[UInt8],       # [128, 3] NTSC palette
)

# Kernel 2: Resize 210x160 → 84x84 grayscale
fn resize_grayscale_kernel[BATCH: Int](
    rgb_buffer: UnsafePointer[UInt8],    # [BATCH, 210, 160, 3] input
    obs: UnsafePointer[Float32],         # [BATCH, 84, 84] output (normalized 0-1)
)

# Kernel 3: Frame stacking (circular buffer)
fn frame_stack_kernel[BATCH: Int, STACK: Int](
    current_frame: UnsafePointer[Float32],  # [BATCH, 84, 84]
    frame_buffer: UnsafePointer[Float32],   # [BATCH, STACK, 84, 84]
    frame_idx: UnsafePointer[Int],          # [BATCH] circular index
    stacked_obs: UnsafePointer[Float32],    # [BATCH, STACK*84*84] output
)

# Kernel 4: Max-pool over frame skip (for flickering sprites)
fn max_pool_frames_kernel[BATCH: Int](
    frame_a: UnsafePointer[UInt8],   # Previous frame
    frame_b: UnsafePointer[UInt8],   # Current frame
    output: UnsafePointer[UInt8],    # max(a, b) per pixel
)
```

### Step 10: GPUDiscreteEnv Implementation (`atari_env.mojo`)

```
struct AtariEnv[GAME: GameDef](GPUDiscreteEnv):
    comptime STATE_SIZE: Int = 300           # AtariState serialized size
    comptime OBS_DIM: Int = 4 * 84 * 84      # 4 stacked 84x84 frames = 28224
    comptime NUM_ACTIONS: Int = GAME.MIN_ACTIONS
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # GPU step dispatches to atari_step_kernel + preprocessing pipeline
```

## Phase 3: Rendering with SDL3

### Step 11: Atari Renderer (`renderer.mojo`)

Yes, we can use the existing SDL3 infrastructure. The approach:

1. **Create an SDL3 streaming texture** (210x160, RGB24 or RGBA32)
2. **Each frame**: copy the TIA output → `update_texture()` → `render_texture()` scaled to window
3. SDL3 handles the upscaling with hardware-accelerated bilinear filtering

```
struct AtariRenderer:
    var renderer2d: Renderer2D              # Existing SDL3 renderer
    var screen_texture: Ptr[Texture]        # 160x210 RGBA texture
    var pixel_buffer: UnsafePointer[UInt8]  # 160x210x4 staging buffer

    fn __init__(mut self) raises:
        self.renderer2d = Renderer2D(
            width=640,   # 160 × 4x upscale
            height=840,  # 210 × 4x upscale
            title="Atari 2600",
            fps=60,
        )
        # Create streaming texture at native Atari resolution
        self.screen_texture = create_texture(
            self.renderer2d.sdl_renderer,
            PixelFormat.RGBA8888,
            TextureAccess.STREAMING,  # Optimized for frequent updates
            160, 210,
        )

    fn render_frame(mut self, tia_output: UnsafePointer[UInt8]) raises:
        # Convert TIA color indices → RGBA in pixel_buffer
        for i in range(210 * 160):
            var color_idx = tia_output[i]
            var rgb = NTSC_PALETTE[int(color_idx)]
            self.pixel_buffer[i * 4 + 0] = rgb.r
            self.pixel_buffer[i * 4 + 1] = rgb.g
            self.pixel_buffer[i * 4 + 2] = rgb.b
            self.pixel_buffer[i * 4 + 3] = 255

        # Upload to GPU texture
        update_texture(
            self.screen_texture,
            Ptr[Rect, ImmutAnyOrigin](),  # NULL = full texture
            self.pixel_buffer.bitcast[NoneType](),
            160 * 4,  # pitch = width * bytes_per_pixel
        )

        # Render scaled to window
        render_clear(self.renderer2d.sdl_renderer)
        render_texture(
            self.renderer2d.sdl_renderer,
            self.screen_texture,
            Ptr[FRect, ImmutAnyOrigin](),  # NULL = full source
            Ptr[FRect, ImmutAnyOrigin](),  # NULL = full destination (scaled)
        )

        # Overlay: game name, score, lives, frame count
        render_debug_text(...)

        render_present(self.renderer2d.sdl_renderer)
```

**Why this works well:**
- `update_texture()` uploads 160×210×4 = ~134KB per frame — trivial
- SDL3 hardware-scales the texture to the window via GPU (nearest/bilinear)
- Same VideoRecorder can capture to MP4/GIF
- Keyboard input (arrow keys → joystick mapping) via existing event handling
- Overlay text (score, lives, frame#) via `render_debug_text`

**Optional enhancements:**
- CRT shader effect (scanlines, bloom) via a custom SDL3 shader
- Side-by-side multi-env view (render a grid of textures)
- RAM viewer overlay (128 bytes, useful for debugging game detection)

## Phase 4: Standard RL Wrappers

Implement the standard Atari preprocessing pipeline as composable wrappers
or directly in the GPU kernels:

1. **NoopReset**: Random 1-30 NOOP actions on reset
2. **FrameSkip**: Execute action for 4 frames, return max of last 2
3. **EpisodicLife**: Reset on life loss (training only)
4. **FireReset**: Auto-press FIRE after reset (for games that need it)
5. **Resize + Grayscale**: 210×160 RGB → 84×84 grayscale (GPU kernel)
6. **FrameStack**: Stack 4 frames → [4, 84, 84] observation
7. **RewardClip**: Clip rewards to {-1, 0, +1}

For GPU batched training, these are all baked into the kernels (no wrapper overhead).

## Implementation Order

### Milestone 1: CPU Pong (~2-3 weeks)
1. `atari_state.mojo` — flat state struct
2. `opcodes.mojo` — 256-entry opcode table (from CuLE)
3. `cpu6502.mojo` — fetch-decode-execute + memory map
4. `riot.mojo` — timer + RAM + I/O
5. `tia.mojo` — simplified TIA (from CuLE's 1034-line version)
6. `cartridge.mojo` — 4K direct mapping (Pong uses this)
7. `palette.mojo` — NTSC 128-color palette
8. `games/pong.mojo` — score/lives/terminal from RAM
9. **Validation**: Run Pong ROM on CPU, compare frame output with ALE

### Milestone 2: CPU Rendering (~1 week)
10. `renderer.mojo` — SDL3 texture rendering
11. Interactive play: keyboard → joystick, see the game on screen
12. VideoRecorder integration for MP4/GIF capture

### Milestone 3: GPU Batched Stepping (~1-2 weeks)
13. `atari_env.mojo` — GPUDiscreteEnv implementation
14. GPU step kernel (one thread per env)
15. GPU preprocessing kernels (palette, resize, grayscale, frame stack)
16. Benchmark: measure frames/sec with 512-2048 parallel envs

### Milestone 4: DQN on Pong (~1 week)
17. Wire up to existing DQN agent with CNN policy
18. Need Conv2D in the policy network (already in autodiff: Conv2D, MaxPool2D)
19. Train DQN on Pong, target score ≥ 18
20. Compare learning curve with CleanRL reference

### Milestone 5: More Games (~2-3 weeks)
21. Add F8 bank switching (covers Breakout, Space Invaders, etc.)
22. Port 9 more game detection files from CuLE
23. Validate each game against ALE output
24. PPO on Atari (ppo_atari variant)

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TIA edge cases break games | High | Use CuLE's battle-tested TIA, not Stella's. Start with simple games (Pong has minimal TIA usage) |
| GPU thread divergence in 6502 | Medium | All threads run same fetch-decode-execute loop. Divergence only in opcode switch — acceptable for GPU (CuLE proved this works) |
| ROM legality | Low | ROMs are user-provided. Ship no ROMs. Use same approach as ALE/Gymnasium (user installs via `autorom`) |
| Mojo GPU register pressure | Medium | AtariState is ~300 bytes. May need to spill to global memory. Profile and optimize hot state into registers |
| Bank switching complexity | Low | Start with 4K games (no banking). F8 is simple. Rare mappers can wait |

## Observation Modes

Support two observation modes (compile-time):

1. **Pixel mode** (standard): [4, 84, 84] float32 — frame-stacked grayscale
   - OBS_DIM = 28224
   - Requires full TIA rendering + preprocessing pipeline
   - Used for CNN-based agents (DQN, PPO)

2. **RAM mode** (fast): [128] uint8 — raw Atari RAM
   - OBS_DIM = 128
   - Skips TIA rendering entirely (massive speedup)
   - Used for MLP-based agents, debugging, research
   - ALE supports this too (`obs_type="ram"`)

## Lines of Code Estimate

| Component | Lines | Complexity |
|-----------|-------|------------|
| atari_state.mojo | ~150 | Low |
| opcodes.mojo | ~1250 | Low (data table) |
| cpu6502.mojo | ~800 | Medium |
| tia.mojo | ~1200 | High |
| riot.mojo | ~200 | Low |
| cartridge.mojo | ~300 | Low |
| palette.mojo | ~50 | Low |
| atari_env.mojo | ~400 | Medium |
| preprocessing.mojo | ~300 | Medium |
| renderer.mojo | ~200 | Low |
| games/ (10 games) | ~1000 | Low |
| **Total** | **~5850** | |

Comparable to `physics2d/` (~5000 lines) which was done in a similar timeframe.
