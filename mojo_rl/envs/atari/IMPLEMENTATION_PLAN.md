# Atari Games — Native Mojo GPU Game Engines

## Milestone 3 (Revised): Native GPU Game Engines

### Motivation — Why Not Emulate on GPU?

Milestones 1-2 delivered a complete Atari 2600 emulator (~4700 lines): 6502 CPU, TIA, RIOT,
cartridge mappers, 3 games, CPU AtariEnv with RAM/pixel modes, SDL3 renderer, and tests.

**However, GPU emulation of the 6502 doesn't work:**
- The Metal shader compiler (Apple Silicon) crashes on the kernel — the 6502 opcode dispatch +
  TIA + RIOT inlined into a single kernel exceeds compiler complexity limits
- AtariState is a ~350-byte heterogeneous struct that requires bitcast hacks instead of LayoutTensor
- Massive thread divergence from opcode dispatch (each thread executes different instruction sequences)
- ~20K cycles per frame of serial 6502 execution — fundamentally not GPU-friendly

**The insight:** Just like `physics3d/` replaced MuJoCo with a native engine, we replace the Atari
emulator with **native GPU game kernels**. The goal is training RL on games, not cycle-accurate
emulation. Pong is just ball physics + paddles — 10 floats of state, trivial GPU kernel.

### Architecture

```
envs/atari_games/
├── __init__.mojo              # Re-exports
├── core/
│   ├── game_trait.mojo        # AtariGame trait (shared game interface)
│   ├── gpu_renderer.mojo      # GPU inline 160×210 shape renderer
│   ├── preprocessing.mojo     # Grayscale resize + frame stack (GPU)
│   ├── colors.mojo            # Atari-style color palette (shared)
│   └── gpu_env.mojo           # AtariGameEnv[G] — generic CPU+GPU wrapper
├── pong/
│   ├── pong.mojo              # NativePong(AtariGame) — game logic + rendering
│   └── test_pong.mojo         # Validation against CPU emulator
├── breakout/
│   ├── breakout.mojo          # NativeBreakout(AtariGame)
│   └── test_breakout.mojo
├── space_invaders/
│   ├── space_invaders.mojo    # NativeSpaceInvaders(AtariGame)
│   └── test_space_invaders.mojo
├── freeway/
│   └── freeway.mojo           # NativeFreeway(AtariGame)
├── enduro/
│   └── enduro.mojo            # NativeEnduro(AtariGame)
├── qbert/
│   └── qbert.mojo             # NativeQbert(AtariGame)
└── examples/
    ├── pong_dqn.mojo          # DQN on Pong (pixel mode)
    ├── pong_dqn_clean.mojo    # DQN on Pong (clean obs mode)
    └── multi_game_ppo.mojo    # PPO across multiple games (pixel mode)
```

### Design Principles

1. **Same pattern as CartPole/Pendulum/LunarLander** — LayoutTensor state, CPU+GPU dual paths
2. **Two observation modes** (compile-time):
   - **Clean obs** (`OBS_MODE=0`): Game-specific float vector (ball pos, paddle pos, velocities, score...)
   - **Pixel obs** (`OBS_MODE=1`): 4×84×84 grayscale frame stack (DQN benchmark standard)
3. **CPU path** for evaluation + SDL3 rendering, **GPU path** for batched training
4. **Validation** against the existing CPU Atari emulator (Milestones 1-2)
5. **One game trait, one generic wrapper** — each game only implements physics + rendering

### The AtariGame Trait

```mojo
trait AtariGame:
    # Compile-time constants
    comptime NAME: StringLiteral
    comptime NUM_ACTIONS: Int      # Discrete action count
    comptime STATE_SIZE: Int       # Total float32 state slots per env
    comptime CLEAN_OBS_DIM: Int    # Clean observation dimension
    comptime MAX_STEPS: Int        # Episode truncation limit

    # =========== GPU inline functions (called from kernel) ===========

    @staticmethod
    @always_inline
    def step_env[BATCH_SIZE: Int, STATE_SIZE: Int](
        states: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
        action: Int,
        env: Int,
        rng_seed: UInt32,
    ) -> Tuple[Scalar[dtype], Bool, Bool]:
        """Step one env. Returns (reward, done, terminated).
        Physics update + game logic, all in LayoutTensor state."""
        ...

    @staticmethod
    @always_inline
    def reset_env[BATCH_SIZE: Int, STATE_SIZE: Int](
        states: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
        env: Int,
        rng_seed: UInt32,
    ):
        """Reset one env to initial state."""
        ...

    @staticmethod
    @always_inline
    def extract_clean_obs[BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int](
        states: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
        obs: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin],
        env: Int,
    ):
        """Copy clean observation from state to obs buffer."""
        ...

    @staticmethod
    @always_inline
    def render_frame_gpu[BATCH_SIZE: Int, STATE_SIZE: Int](
        states: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin],
        frame_buf: UnsafePointer[UInt8, MutAnyOrigin],  # 160×210 grayscale
        env: Int,
    ):
        """Render game state to 160×210 grayscale framebuffer (for pixel obs).
        Uses simple shape drawing (filled rects, circles) — NOT TIA emulation."""
        ...
```

### State Layout Convention

Each game's state in the LayoutTensor follows this layout:

```
[0 .. CLEAN_OBS_DIM-1]            # Observable state (clean obs mode)
[CLEAN_OBS_DIM .. STATE_SIZE-3]   # Internal state (physics internals, brick grid, etc.)
[STATE_SIZE-3]                    # step_count
[STATE_SIZE-2]                    # score
[STATE_SIZE-1]                    # lives
```

Observations always come first so `extract_clean_obs` is just a copy of the first N elements.

### Game Examples

#### Pong

```
State layout (STATE_SIZE = 12):
  [0] ball_x          (0..159)     \
  [1] ball_y          (0..209)      |
  [2] ball_vx         (-3..3)       | CLEAN_OBS_DIM = 6
  [3] ball_vy         (-3..3)       |
  [4] paddle_y        (0..209)      |  (agent paddle)
  [5] cpu_paddle_y    (0..209)     /
  [6] player_score    (0..21)      # internal
  [7] cpu_score       (0..21)
  [8] serve_timer     (countdown)
  [9] step_count
  [10] score          (= player_score - cpu_score, for reward)
  [11] lives          (unused, 0)

NUM_ACTIONS = 3  (NOOP, UP, DOWN)
  or 4 (NOOP, UP, DOWN, FIRE) for serve mechanic

Physics per step:
  - Ball moves by (vx, vy)
  - Bounce off top/bottom walls
  - Bounce off paddles (angle depends on hit position)
  - CPU paddle tracks ball with slight delay
  - Score when ball passes paddle
  - Reset ball to center on score

Pixel rendering:
  - Black background (color 0x00)
  - White paddles (2 filled rects, 4px × 16px)
  - White ball (2×2 rect)
  - Score digits (simple 3×5 font)
  - Net (dashed center line)
```

#### Breakout

```
State layout (STATE_SIZE = 56):
  [0] ball_x               \
  [1] ball_y                |
  [2] ball_vx               | CLEAN_OBS_DIM = 7
  [3] ball_vy               |
  [4] paddle_x              |
  [5] bricks_remaining      |
  [6] lives                /
  [7..48] brick_alive (6 rows × 14 cols = 84 bools packed as floats, 42 slots)
  [49] score
  [50] ball_stuck           # stuck to paddle before serve
  [51] step_count
  [52..55] reserved

NUM_ACTIONS = 4  (NOOP, FIRE, LEFT, RIGHT)

Pixel rendering:
  - Colored brick rows (6 colors: red, orange, yellow, green, aqua, blue)
  - White paddle (16px wide)
  - White ball (2×2)
  - Score + lives display
```

#### Space Invaders

```
State layout (STATE_SIZE = 96):
  [0] ship_x                \
  [1] bullet_x               |
  [2] bullet_y               | CLEAN_OBS_DIM = 10
  [3] bullet_active          |
  [4..8] alien_bullet_x/y/active (2 bullets)
  [9] aliens_remaining       /
  [10..64] alien_alive (5 rows × 11 cols = 55 bools)
  [65] alien_shift_x         # current formation x offset
  [66] alien_shift_y         # current formation y offset
  [67] alien_direction       # +1 or -1
  [68] alien_move_timer
  [69] score
  [70] lives
  [71] step_count
  [72..95] reserved

NUM_ACTIONS = 4  (NOOP, LEFT, RIGHT, FIRE)
```

### GPU Pixel Observation Pipeline

For `OBS_MODE=1`, the pixel pipeline runs per-step after `step_env`:

1. **Render** — `G.render_frame_gpu()` draws game state to a 160×210 grayscale buffer
   in per-env workspace (no TIA — just filled rectangles and simple shapes)

2. **Resize** — Box-filter 160×210 → 84×84 (same algorithm as existing `_resize_160x210_to_84x84`)

3. **Frame stack** — Push resized frame into 4-slot ring buffer, output chronological [4×84×84]

**Per-env workspace layout (pixel mode):**
```
[0 .. 4199]        160×210 grayscale framebuffer (packed in Float32 slots)
[4200 .. 5963]     4 × 84×84 frame stack (packed in Float32 slots)
[5964]             frame_idx (ring buffer write position)
```

`STEP_WS_PER_ENV = 5965` for pixel mode, `0` for clean obs mode.

**GPU inline renderer** — Simple shape drawing functions inlined into the step kernel:

```mojo
@always_inline
def draw_filled_rect(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int, y: Int, w: Int, h: Int,
    color: UInt8,  # grayscale 0-255
):
    """Draw a filled rectangle into the 160×210 grayscale buffer."""
    for row in range(max(0, y), min(210, y + h)):
        for col in range(max(0, x), min(160, x + w)):
            buf[row * 160 + col] = color

@always_inline
def draw_circle(buf, cx, cy, radius, color):
    """Draw a filled circle (for ball, bullets)."""

@always_inline
def clear_frame(buf):
    """Fill with black (0)."""

# Each game's render_frame_gpu calls these to compose its display.
```

This is ~50 lines total — trivial compared to TIA emulation.

### The Generic Wrapper: AtariGameEnv[G: AtariGame]

A single struct that implements **both** `BoxDiscreteActionEnv` + `GPUDiscreteEnv` + `RenderableEnv`:

```mojo
struct AtariGameEnv[
    G: AtariGame,
    OBS_MODE: Int = 0,      # 0 = clean, 1 = pixel
    DTYPE: DType = DType.float32,
](BoxDiscreteActionEnv, GPUDiscreteEnv, RenderableEnv):

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = G.STATE_SIZE
    comptime OBS_DIM: Int = G.CLEAN_OBS_DIM if Self.OBS_MODE == 0 else 4 * 84 * 84
    comptime NUM_ACTIONS: Int = G.NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0 if Self.OBS_MODE == 0 else PIXEL_WS_SIZE

    # ---- CPU path (BoxDiscreteActionEnv) ----
    def reset_obs_list(mut self) -> List[Scalar[dtype]]
    def step_obs(mut self, action: Int) -> Tuple[List[Scalar[dtype]], Scalar[dtype], Bool]

    # ---- GPU path (GPUDiscreteEnv) ----
    @staticmethod def step_kernel_gpu[...](...) raises   # Dispatches to G.step_env
    @staticmethod def reset_kernel_gpu[...](...) raises  # Dispatches to G.reset_env
    @staticmethod def selective_reset_kernel_gpu[...](...) raises

    # ---- Rendering (RenderableEnv) ----
    def init_renderer(mut self) raises -> Bool      # SDL3 window
    def render_frame(mut self) raises               # Draw game state
    def close_renderer(mut self) raises
```

**This means each new game is ONLY the physics + rendering — all boilerplate is shared.**

### CPU Path Rendering

For CPU rendering (evaluation/visualization), we use the existing `Renderer2D` with two approaches:

1. **Direct SDL3 drawing** — Use `draw_rect_world`, `draw_circle_world` etc. at Atari resolution
   scaled to the window. Good for clean visualization with anti-aliasing.

2. **Texture upload** — Use the same `render_frame_gpu` function to draw to a 160×210 buffer,
   then upload as an SDL3 texture (like the existing Atari renderer). Gives pixel-accurate match
   between CPU rendering and GPU pixel observations.

Both approaches coexist: texture upload for pixel-accurate validation, direct drawing for nice visuals.

### Validation Against CPU Emulator

Since we have the working Atari emulator from Milestones 1-2, we can validate:

1. **Reward curve comparison** — Run 10K random steps on both, compare score distributions
2. **Action response** — Verify UP moves paddle up, ball bounces correctly, etc.
3. **Visual comparison** — Side-by-side screenshots of emulator vs native game
4. **Episode statistics** — Mean episode length, mean reward, score distributions

The native games don't need to be pixel-identical to the emulator — they need to produce
**similar RL training dynamics** (same reward structure, similar difficulty, same action space).

### Game Porting Effort

| Game | State Size | Core Logic | Render | Estimated Effort |
|------|-----------|-----------|--------|-----------------|
| Pong | 12 floats | Ball + 2 paddles, trivial | 5 rects + digits | 1 day |
| Breakout | 56 floats | Ball + paddle + brick grid | Rects + grid | 1-2 days |
| Space Invaders | 96 floats | Formation + bullets + shields | Grid + sprites | 2-3 days |
| Freeway | 40 floats | Chicken + lanes of cars | Rects + lanes | 1 day |
| Enduro | 60 floats | Car + road + opponents | Road + rects | 2 days |
| Qbert | 80 floats | Isometric grid + enemies | Grid + sprites | 2-3 days |
| Asteroids | 70 floats | Ship + asteroids + bullets | Wireframe shapes | 2 days |
| Frostbite | 80 floats | Platforms + bear + enemies | Rects + grid | 2 days |
| Seaquest | 70 floats | Sub + fish + divers | Rects + sprites | 2 days |
| Montezuma's Revenge | 100 floats | Platformer + rooms | Complex | 3-4 days |

**Why AI-assisted porting scales:**
- Every game follows the **identical template** (AtariGame trait + AtariGameEnv wrapper)
- Game rules are fully documented (decades of reverse engineering)
- Physics is simple (rectangles, grids, basic collision)
- Validation against the CPU emulator catches bugs fast

### Implementation Steps

#### Step 1: Core Infrastructure (~300 lines)

1. `core/game_trait.mojo` — AtariGame trait definition
2. `core/colors.mojo` — Atari-style color constants (grayscale for GPU, RGB for SDL3)
3. `core/gpu_renderer.mojo` — `draw_filled_rect`, `draw_circle`, `clear_frame` (~50 lines)
4. `core/preprocessing.mojo` — `resize_160x210_to_84x84`, `push_frame_stack` (~100 lines)
5. `core/gpu_env.mojo` — AtariGameEnv[G] generic wrapper (~300 lines)

#### Step 2: Pong (~200 lines)

1. `pong/pong.mojo` — NativePong implementing AtariGame
   - `step_env`: Ball physics, paddle movement, scoring, CPU AI
   - `reset_env`: Center ball, reset scores
   - `extract_clean_obs`: Copy first 6 state elements
   - `render_frame_gpu`: Draw paddles, ball, net, score
2. `pong/test_pong.mojo` — Validate against CPU emulator
   - Compare reward structure, episode lengths
   - Visual comparison screenshots

#### Step 3: Breakout (~300 lines)

1. `breakout/breakout.mojo` — NativeBreakout
   - Brick grid management (6×14)
   - Ball-brick collision with brick destruction
   - Paddle physics, ball serve mechanic
   - Score per brick (row-dependent, like original)
2. `breakout/test_breakout.mojo`

#### Step 4: Space Invaders (~400 lines)

1. `space_invaders/space_invaders.mojo` — NativeSpaceInvaders
   - Alien formation movement (shift + drop)
   - Player + alien bullets
   - Shield erosion
   - Increasing difficulty as aliens are destroyed
2. `space_invaders/test_space_invaders.mojo`

#### Step 5: DQN Training (~100 lines)

1. `examples/pong_dqn.mojo` — DQN on Pong with pixel observations
   - Uses existing DQN agent with Conv2D policy (already in autodiff)
   - Target: score ≥ 18 (same as DQN paper)
2. `examples/pong_dqn_clean.mojo` — DQN on Pong with clean observations
   - MLP policy, should converge faster
3. Compare learning curves with CleanRL reference

#### Step 6: More Games + Multi-Game Training

1. Port Freeway, Enduro, Qbert (3 more games)
2. `examples/multi_game_ppo.mojo` — Train single PPO agent across all games
   - Same pixel observation space (4×84×84) for all games
   - Different action mappings per game
   - Test generalization: does it learn new games faster?

### Lines of Code Estimate

| Component | Lines | Notes |
|-----------|-------|-------|
| core/game_trait.mojo | ~60 | Trait definition |
| core/colors.mojo | ~30 | Color constants |
| core/gpu_renderer.mojo | ~80 | Shape drawing functions |
| core/preprocessing.mojo | ~120 | Resize + frame stack |
| core/gpu_env.mojo | ~350 | Generic CPU+GPU wrapper |
| pong/pong.mojo | ~200 | First game |
| breakout/breakout.mojo | ~300 | Second game |
| space_invaders/space_invaders.mojo | ~400 | Third game |
| tests + examples | ~300 | Validation + training scripts |
| **Total** | **~1840** | Much less than emulator (~4700) |

### Existing Code Kept As-Is

The Atari emulator from Milestones 1-2 stays in `envs/atari/`:
- Still works for CPU-only training and evaluation via `AtariEnv[PongDef]`
- Used as validation reference for the native games
- Supports any Atari ROM (not just the games we port natively)
- The SDL3 renderer (`renderer.mojo`) continues to work for playing ROMs

### Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Native games play differently from real Atari | Validate reward curves + episode stats against emulator. RL-relevant dynamics matter, not pixel identity |
| Pixel rendering too simple (agent exploits visual artifacts) | Use Atari-style colors, proper aspect ratio, add slight position randomization on reset |
| Too many games to port | Start with 3 (Pong, Breakout, Space Invaders). AI-assisted porting scales well — identical template |
| Frame stack per-env workspace too large for GPU | 160×210 + 4×84×84 = ~62KB per env. At 2048 envs: ~124MB. Fits in GPU memory |
| Clean obs mode too easy (agent doesn't generalize to pixel) | That's a feature, not a bug — clean obs for fast iteration, pixel obs for the real benchmark |

### Comparison with Emulator Approach

| Aspect | Emulator on GPU (abandoned) | Native Game Engines |
|--------|---------------------------|-------------------|
| GPU compatibility | Metal compiler crashes | LayoutTensor, standard kernels |
| State per env | ~350 bytes heterogeneous | ~50-100 float32, LayoutTensor |
| Kernel complexity | 6502 + TIA + RIOT (~5000 lines) | Simple physics (~100-200 lines) |
| Step latency | ~20K serial 6502 cycles | ~50 float operations |
| Thread divergence | Massive (opcode dispatch) | Minimal (same physics path) |
| Games supported | Any ROM | Must port each game |
| Lines of code | ~4700 (emulator) + ~1000 (GPU) | ~1800 (all games + infra) |
| Apple Silicon | Doesn't work | Works (same as CartPole) |
