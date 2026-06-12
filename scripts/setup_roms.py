#!/usr/bin/env python3
"""
Setup ROM symlinks for Atari emulator.

This script creates a symlink from the project's roms/ directory to ale_py's
ROM directory. This allows the Atari emulator examples to find the ROM files
without bundling them in the repository.

Usage:
    python scripts/setup_roms.py
    pixi run setup-roms
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
            print("Make sure ale-py is installed: pip install ale-py")
            return None

        return roms_path
    except ImportError:
        print("Error: ale_py not found. Install it with: pip install ale-py")
        return None

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
            return True
        else:
            print(f"✗ ROM symlink points to wrong location: {roms_link} -> {target}")
            print(f"  Expected: {ale_roms_path}")
            roms_link.unlink()

    # If a directory exists, move it
    if roms_link.exists():
        print(f"Warning: {roms_link} exists but is not a symlink. Backing up...")
        backup = project_root / "roms.backup"
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
            print(f"  Examples: {', '.join(f.name for f in rom_files[:3])}")

        return True
    except Exception as e:
        print(f"✗ Failed to create symlink: {e}")
        return False

if __name__ == "__main__":
    success = setup_roms_symlink()
    sys.exit(0 if success else 1)
