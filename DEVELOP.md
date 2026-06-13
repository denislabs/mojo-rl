# Development Setup

This guide covers additional setup steps needed after `pixi install` for development and testing.

## Atari ROM Setup

The Atari emulator requires ROM files from the Arcade Learning Environment (ALE). `ale-py` ships **without** ROMs, so they must be downloaded once.

### Setup

After `pixi install`, run:

```bash
pixi run setup-roms
```

This single command:
1. Downloads the canonical Farama ROM bundle (~0.5 MB, 108 ROMs) and verifies its SHA-256 checksum
2. Unpacks the `.bin` files into the active environment's `ale_py/roms` directory
3. Symlinks the project's `roms/` to it

It's idempotent — re-running just confirms the symlink. (The downloader is the
same source AutoROM and the Farama ALE repo use; pure-Python, no AutoROM package
needed.)

### Multi-Environment Setup

Each pixi environment (default, nvidia, apple) has its own isolated `ale_py`, so run setup in **each one** you use:

```bash
pixi run setup-roms              # default environment
pixi run -e nvidia setup-roms    # nvidia (CUDA) environment
pixi run -e apple setup-roms     # apple (Metal) environment
```

### Troubleshooting

**Error: `Failed to open file 'roms/pong.bin'`**
- The symlink may point to a different pixi environment than the one you're running
- Run `pixi run -e <env> setup-roms` for the environment you train with

**Error: Checksum verification failed**
- The download was corrupted or the upstream bundle moved — retry once; if it persists, check network/proxy

**Symlink creation failed**
- Check directory permissions on the project root
- If a `roms/` directory exists (not a symlink), the script backs it up to `roms.backup/`

**Atari ROM License**
- By downloading, you agree to the Atari ROM license terms (ROMs distributed for research/educational use via the Farama Foundation)

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
