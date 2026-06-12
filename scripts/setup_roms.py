#!/usr/bin/env python3
"""
Setup ROM symlinks for Atari emulator.

This script creates a symlink from the project's roms/ directory to ale_py's
ROM directory. This allows the Atari emulator examples to find the ROM files
without bundling them in the repository.

Works with pixi multi-environments (default, nvidia, apple). Each environment
has its own ale_py location in .pixi/envs/<env>/lib/pythonX.Y/site-packages/,
and this script resolves to the correct one.

Usage:
    python scripts/setup_roms.py
    pixi run setup-roms
    pixi run -e nvidia setup-roms
    pixi run -e apple setup-roms
"""

import sys
import os
from pathlib import Path

def find_ale_roms_path():
    """Find the ale_py ROMs directory in the current Python environment."""
    try:
        import ale_py
        ale_module_path = Path(ale_py.__file__).parent
        roms_path = ale_module_path / "roms"

        if not roms_path.exists():
            print(f"Error: ale_py roms directory not found at {roms_path}")
            print("This can happen if ale-py is installed but ROMs weren't downloaded.")
            print("Download them with:")
            print("  python -c \"import ale_py; ale_py.utils.download_ale_py_roms()\"")
            return None

        return roms_path
    except ImportError:
        print("Error: ale_py not found. Install it with: pixi run pip install ale-py")
        return None

def ensure_roms_downloaded(roms_path):
    """Ensure ROMs are downloaded in the given ale_py roms directory."""
    rom_files = list(roms_path.glob("*.bin"))
    if rom_files and len(rom_files) > 1:
        return True  # ROMs already present

    print(f"⚠️  Only {len(rom_files)} ROM(s) found in {roms_path}")
    print("Downloading ROMs (this may take a minute)...")
    try:
        import ale_py
        ale_py.utils.download_ale_py_roms()
        print("✓ ROMs downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download ROMs: {e}")
        print("Try manually:")
        print("  python -c \"import ale_py; ale_py.utils.download_ale_py_roms()\"")
        return False


def setup_roms_symlink(project_root=None):
    """Create the roms symlink in the project root."""
    if project_root is None:
        project_root = Path(__file__).parent.parent

    roms_link = project_root / "roms"
    ale_roms_path = find_ale_roms_path()

    if ale_roms_path is None:
        return False

    # Ensure ROMs are downloaded
    if not ensure_roms_downloaded(ale_roms_path):
        print("⚠️  Continuing without ROM validation (may fail at runtime)")

    # If symlink already exists and points to the right place, we're done
    if roms_link.is_symlink():
        target = roms_link.resolve()
        if target == ale_roms_path.resolve():
            print(f"✓ ROM symlink already correct: {roms_link} -> {ale_roms_path}")
            # Double-check ROM availability
            rom_files = list(ale_roms_path.glob("*.bin"))
            print(f"✓ {len(rom_files)} ROM files available")
            return True
        else:
            print(f"⚠️  ROM symlink points to different environment: {roms_link} -> {target}")
            print(f"  Current environment: {ale_roms_path}")
            print("  Updating symlink...")
            roms_link.unlink()

    # If a directory exists (not a symlink), move it
    if roms_link.exists():
        print(f"⚠️  {roms_link} exists but is not a symlink. Backing up...")
        backup = project_root / "roms.backup"
        if backup.exists():
            import shutil
            shutil.rmtree(backup)
        roms_link.rename(backup)
        print(f"  Moved to: {backup}")

    # Create the symlink
    try:
        roms_link.symlink_to(ale_roms_path)
        print(f"✓ Created ROM symlink: {roms_link} -> {ale_roms_path}")

        # Verify by listing a few ROMs
        rom_files = list(ale_roms_path.glob("*.bin"))
        if rom_files:
            print(f"✓ Found {len(rom_files)} ROM files")
            if len(rom_files) <= 3:
                print(f"  ROMs: {', '.join(f.name for f in rom_files)}")
            else:
                print(f"  Examples: {', '.join(f.name for f in rom_files[:3])}")
        else:
            print("⚠️  No ROMs found! Run: python -c \"import ale_py; ale_py.utils.download_ale_py_roms()\"")

        return True
    except Exception as e:
        print(f"✗ Failed to create symlink: {e}")
        return False

if __name__ == "__main__":
    success = setup_roms_symlink()
    sys.exit(0 if success else 1)
