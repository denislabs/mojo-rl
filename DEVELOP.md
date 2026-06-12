# Development Setup

This guide covers additional setup steps needed after `pixi install` for development and testing.

## Atari ROM Setup

The Atari emulator requires ROM files from the Arcade Learning Environment (ALE). These are **not** bundled in the repository but are available through the `ale_py` Python package.

### Automatic Setup

After `pixi install`, run:

```bash
pixi run setup-roms
```

This creates a symlink from `roms/` to the ale_py ROM directory in your Python environment, making all ~100 Atari games available to the emulator.

### What Happens

- Finds the `ale_py` package in your active environment
- Creates a symlink: `roms/ → $CONDA_PREFIX/lib/pythonX.Y/site-packages/ale_py/roms`
- Verifies the symlink by listing available ROM files

### Troubleshooting

**Error: `ale_py not found`**
- `ale_py` is included via `gymnasium`. If missing, install explicitly:
  ```bash
  pixi run pip install ale-py
  ```

**Error: `roms directory not found`**
- This can happen if `ale_py` is installed but ROMs weren't downloaded
- Download them manually:
  ```bash
  pixi run python -c "import ale_py; ale_py.utils.download_ale_py_roms()"
  ```

**Symlink creation failed**
- Check directory permissions on the project root
- If a `roms/` directory exists (not a symlink), the script will back it up to `roms.backup/`

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
