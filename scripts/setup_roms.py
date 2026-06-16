#!/usr/bin/env python3
"""
Download Atari ROMs and set up the project's roms/ directory for the emulator.

ale-py ships without ROMs. This script downloads the canonical ROM bundle
(the same base64-encoded tarball the Farama ALE repo and AutoROM use),
verifies its checksum, unpacks the 108 .bin files into the active
environment's ale_py/roms directory, and symlinks the project's roms/ to it.

It is fully self-contained (pure Python, stdlib only) and cross-platform —
no AutoROM package or bash required. Re-running is cheap: if ROMs are already
present it just (re)creates the symlink.

By downloading, you agree to the Atari ROM license terms (the ROMs are
distributed for research/educational use via the Farama Foundation).

Each pixi environment has its own isolated ale_py, so run this in each
environment you use:
    pixi run setup-roms                 # default environment
    pixi run -e nvidia setup-roms       # nvidia (CUDA) environment
    pixi run -e apple setup-roms        # apple (Metal) environment

Or directly:
    python scripts/setup_roms.py
"""

import sys
import hashlib
import base64
import tarfile
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

# Canonical ROM bundle used by the Farama ALE repo + AutoROM.
# scripts/download_unpack_roms.sh in Farama-Foundation/Arcade-Learning-Environment
ROMS_URL = (
    "https://gist.githubusercontent.com/jjshoots/"
    "61b22aefce4456920ba99f2c36906eda/raw/"
    "00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64"
)
ROMS_SHA256 = "02ca777c16476a72fa36680a2ba78f24c3ac31b2155033549a5f37a0653117de"
EXPECTED_ROM_COUNT = 108


def find_ale_roms_path():
    """Find the ale_py ROMs directory in the current Python environment."""
    try:
        import ale_py
    except ImportError:
        print("Error: ale_py not found. Install it with: pixi run pip install ale-py")
        return None

    roms_path = Path(ale_py.__file__).parent / "roms"
    roms_path.mkdir(parents=True, exist_ok=True)
    return roms_path


def download_and_unpack_roms(roms_path):
    """Download, verify, and unpack the ROM bundle into roms_path.

    Returns True on success. Mirrors Farama's download_unpack_roms.sh:
    fetch base64 tarball -> verify sha256 -> base64 decode -> untar ->
    move ROM/*/*.bin into roms_path.
    """
    print(f"Downloading Atari ROM bundle (~0.5 MB) from Farama gist...")
    try:
        with urllib.request.urlopen(ROMS_URL, timeout=60) as resp:
            b64_data = resp.read()
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

    # Verify checksum of the base64 file (matches the .sh script).
    computed = hashlib.sha256(b64_data).hexdigest()
    if computed != ROMS_SHA256:
        print("✗ Checksum verification failed!")
        print(f"  expected: {ROMS_SHA256}")
        print(f"  computed: {computed}")
        return False
    print("✓ Checksum verified")

    # Decode base64 -> tar.gz bytes, then extract all .bin files flat.
    try:
        tar_bytes = base64.b64decode(b64_data)
        count = 0
        with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".bin"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                dest = roms_path / Path(member.name).name
                with open(dest, "wb") as out:
                    out.write(f.read())
                count += 1
    except Exception as e:
        print(f"✗ Failed to unpack ROMs: {e}")
        return False

    print(f"✓ Unpacked {count} ROM files into {roms_path}")
    return True


def ensure_roms_available(roms_path):
    """Ensure ROMs are present, downloading them if needed."""
    rom_files = list(roms_path.glob("*.bin"))
    if len(rom_files) >= 100:
        print(f"✓ {len(rom_files)} ROM files already available")
        return True

    if rom_files:
        print(f"Only {len(rom_files)} ROM(s) found — downloading full set...")
    return download_and_unpack_roms(roms_path)


def setup_roms_symlink(project_root=None):
    """Download ROMs (if needed) and create the roms symlink in project root."""
    if project_root is None:
        project_root = Path(__file__).parent.parent

    ale_roms_path = find_ale_roms_path()
    if ale_roms_path is None:
        return False

    # Make sure ROMs are actually present in the ale_py dir.
    if not ensure_roms_available(ale_roms_path):
        return False

    roms_link = project_root / "roms"

    # If symlink already exists and points to the right place, we're done.
    if roms_link.is_symlink():
        if roms_link.resolve() == ale_roms_path.resolve():
            print(f"✓ ROM symlink already correct: {roms_link} -> {ale_roms_path}")
            return True
        print("⚠️  ROM symlink points to a different environment, updating...")
        print(f"  old: {roms_link.resolve()}")
        print(f"  new: {ale_roms_path}")
        roms_link.unlink()

    # If a real directory exists (not a symlink), back it up.
    if roms_link.exists():
        import shutil
        backup = project_root / "roms.backup"
        print(f"⚠️  {roms_link} exists but is not a symlink. Backing up to {backup}")
        if backup.exists():
            shutil.rmtree(backup)
        roms_link.rename(backup)

    try:
        roms_link.symlink_to(ale_roms_path)
        print(f"✓ Created ROM symlink: {roms_link} -> {ale_roms_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create symlink: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if setup_roms_symlink() else 1)
