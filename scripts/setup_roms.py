#!/usr/bin/env python3
"""
Setup ROM symlinks for Atari emulator.

This script creates a symlink from the project's roms/ to ale_py's ROM
directory. ROMs must be downloaded separately using AutoROM.

Before running this:
    pixi run download-roms         # Download ROMs via AutoROM (one-time per environment)

Then use this script to verify and create symlinks:
    python scripts/setup_roms.py                # system Python
    pixi run setup-roms                         # default environment
    pixi run -e nvidia setup-roms               # nvidia (CUDA) environment
    pixi run -e apple setup-roms                # apple (Metal) environment

Multi-environment workflow:
    # Download ROMs in each environment
    pixi run download-roms
    pixi run -e nvidia download-roms
    pixi run -e apple download-roms

    # Create symlinks
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

def check_roms_available(roms_path):
    """Check ROM availability and provide guidance if missing."""
    rom_files = list(roms_path.glob("*.bin"))
    rom_count = len(rom_files)

    if rom_count >= 100:
        return True  # Full ROM set available

    if rom_count > 10:
        print(f"✓ {rom_count} ROM files available (partial set)")
        return True  # Usable ROM set

    # Few or no ROMs — suggest using AutoROM
    print(f"⚠️  Only {rom_count} ROM(s) found in {roms_path}")
    print()
    print("To download Atari ROMs, use AutoROM (included via pixi):")
    print()
    print("  pixi run python -m AutoROM --accept-license")
    print()
    print("This will:")
    print("  1. Download Atari 2600 ROMs (~20 MB)")
    print("  2. Install them in the active pixi environment")
    print("  3. Make them available to ale-py/gymnasium")
    print()
    print("For multi-environment setups, run this in each environment:")
    print("  pixi run -e nvidia python -m AutoROM --accept-license")
    print("  pixi run -e apple python -m AutoROM --accept-license")
    print()

    return rom_count > 0  # At least something is there


def ensure_roms_available(roms_path):
    """Check if ROMs are available, provide guidance if not."""
    return check_roms_available(roms_path)


def setup_roms_symlink(project_root=None):
    """Create the roms symlink in the project root."""
    if project_root is None:
        project_root = Path(__file__).parent.parent

    roms_link = project_root / "roms"
    ale_roms_path = find_ale_roms_path()

    if ale_roms_path is None:
        return False

    # If symlink already exists and points to the right place, we're done
    if roms_link.is_symlink():
        target = roms_link.resolve()
        if target == ale_roms_path.resolve():
            print(f"✓ ROM symlink already correct: {roms_link} -> {ale_roms_path}")
            # Check ROM availability
            ensure_roms_available(ale_roms_path)
            return True
        else:
            print(f"⚠️  ROM symlink points to different environment")
            print(f"  Old: {target}")
            print(f"  New: {ale_roms_path}")
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
        print()

        # Check ROM availability
        ensure_roms_available(ale_roms_path)

        return True
    except Exception as e:
        print(f"✗ Failed to create symlink: {e}")
        return False

if __name__ == "__main__":
    success = setup_roms_symlink()
    sys.exit(0 if success else 1)
