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

#### Multi-Environment Note

If you use multiple pixi environments (default, nvidia, apple), run the setup **in each environment**:

```bash
pixi run setup-roms              # default environment
pixi run -e nvidia setup-roms    # nvidia (CUDA) environment
pixi run -e apple setup-roms     # apple (Metal) environment
```

Each environment has its own isolated ale_py, so the symlink needs to point to the correct one.

### What Happens

- Finds the `ale_py` package in your **active** environment
- Creates a symlink: `roms/ → $CONDA_PREFIX/lib/pythonX.Y/site-packages/ale_py/roms`
- Ensures ROMs are downloaded (auto-downloads if missing)
- Verifies by listing available ROM files

### Troubleshooting

**Error: `Failed to open file 'roms/pong.bin': No such file or directory`**
- The symlink may point to the wrong pixi environment
- Run setup in the environment you're using:
  ```bash
  pixi run -e nvidia setup-roms   # if running with -e nvidia
  ```

**Error: `ale_py not found`**
- Install explicitly:
  ```bash
  pixi run pip install ale-py
  ```

**Error: Only 1 ROM found (tetris.bin)**
- ROMs haven't been downloaded yet. The script will auto-download them, or manually:
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
