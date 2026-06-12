# Development Setup

This guide covers additional setup steps needed after `pixi install` for development and testing.

## Atari ROM Setup

The Atari emulator requires ROM files from the Arcade Learning Environment (ALE). These are **not** bundled in the repository. Use `AutoROM` (via `gymnasium[accept-rom-license]`) to download them.

### Setup

After `pixi install`, download ROMs using AutoROM:

```bash
pixi run python -m AutoROM --accept-license
```

This will:
1. Download Atari 2600 ROMs (~20 MB)
2. Install them in the active pixi environment
3. Create symlink: `roms/` → ale_py ROM directory
4. Run `pixi run setup-roms` to verify

### Multi-Environment Setup

If you use multiple pixi environments, download ROMs in **each one**:

```bash
pixi run python -m AutoROM --accept-license          # default
pixi run -e nvidia python -m AutoROM --accept-license  # nvidia (CUDA)
pixi run -e apple python -m AutoROM --accept-license   # apple (Metal)
pixi run setup-roms                  # verify symlinks
```

Then verify with:
```bash
pixi run setup-roms              # default environment
pixi run -e nvidia setup-roms    # nvidia environment
pixi run -e apple setup-roms     # apple environment
```

### Troubleshooting

**Error: `Failed to open file 'roms/pong.bin'`**
- ROMs haven't been downloaded yet
- Run: `pixi run -e <env> python -m AutoROM --accept-license`
- Then: `pixi run -e <env> setup-roms`

**Error: AutoROM command not found**
- Install it: `pixi run pip install auto-rom`

**Symlink creation failed**
- Check directory permissions on the project root
- If a `roms/` directory exists (not a symlink), the script will back it up to `roms.backup/`

**Atari ROM License**
- By accepting the license during `AutoROM`, you agree to the terms of the Atari ROM license
- ROMs are for personal use and educational purposes

## Running Examples

All commands use pixi:

```bash
# Basic example
pixi run mojo run -I . examples/solve_gridworld.mojo

# Atari emulator example (requires setup-roms)
pixi run mojo run -I . examples/arcade_games/play_atari.mojo

# Tests
pixi run mojo run -I . tests/arcade_games/test_atari_env.mojo
```

## Available Tasks

```bash
pixi run --list              # Show all available tasks
pixi run setup-roms          # Setup Atari ROM symlinks
pixi run build               # Precompile mojo_rl
pixi run test                # Run tests
pixi run format              # Format code
pixi run compile-shaders     # Compile GLSL shaders
pixi run gpu-specs           # Show GPU info (NVIDIA/AMD/Apple)
```
