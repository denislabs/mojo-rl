"""Reference Python window sampler — replicates the relevant slice of
``stable_worldmodel.data.dataset.Dataset.__getitem__`` /
``stable_worldmodel.data.formats.hdf5.HDF5Dataset._load_slice`` using only
``h5py`` + ``numpy``.

Imported from Mojo by ``test_lewm_pusht_parity.mojo`` for byte-for-byte
comparison against ``LewmPushTExpert.sample_window``. Keeping this off
the stable-worldmodel package keeps the pixi env light (no PyTorch
dependency just for the parity test).
"""

from __future__ import annotations

import h5py
import numpy as np


def build_clip_indices(
    ep_lengths: np.ndarray, frameskip: int, num_steps: int
) -> list[tuple[int, int]]:
    """Replicates ``Dataset.__init__``'s ``self.clip_indices``.

    A clip is valid iff the episode is at least ``span`` frames long, where
    ``span = num_steps * frameskip``.
    """
    span = num_steps * frameskip
    out: list[tuple[int, int]] = []
    for ep, length in enumerate(ep_lengths):
        L = int(length)
        if L >= span:
            for start in range(L - span + 1):
                out.append((ep, start))
    return out


def sample_window(
    h5_path: str, clip_idx: int, frameskip: int, num_steps: int
) -> dict[str, np.ndarray]:
    """Return one window as a dict of numpy arrays.

    Output shapes match what ``LewmPushTExpert.sample_window`` writes into
    its ``LewmPushTWindow`` buffer (with the action reshape already applied,
    matching ``Dataset.__getitem__``):

    - ``pixels``  → ``(num_steps, 3, H, W)`` uint8  (HWC → CHW permute)
    - ``action``  → ``(num_steps, frameskip * action_dim)`` float32
    - ``proprio`` → ``(num_steps, proprio_dim)`` float32  (subsampled)
    - ``state``   → ``(num_steps, state_dim)`` float32  (subsampled)
    """
    span = num_steps * frameskip
    with h5py.File(h5_path, "r") as f:
        ep_lengths = f["ep_len"][:]
        ep_offsets = f["ep_offset"][:]
        clip_indices = build_clip_indices(ep_lengths, frameskip, num_steps)
        if clip_idx < 0 or clip_idx >= len(clip_indices):
            raise IndexError(
                f"clip_idx {clip_idx} out of range [0, {len(clip_indices)})"
            )
        ep_idx, start = clip_indices[clip_idx]
        g_start = int(ep_offsets[ep_idx]) + start
        g_end = g_start + span

        # pixels: read span rows (HWC), subsample to num_steps, permute → CHW.
        pixels_hwc = f["pixels"][g_start:g_end][:: frameskip]   # (num_steps, H, W, 3)
        pixels_chw = np.transpose(pixels_hwc, (0, 3, 1, 2)).copy()

        # action: dense span rows, then reshape to (num_steps, fs * A).
        action_dense = f["action"][g_start:g_end]               # (span, A)
        action = action_dense.reshape(num_steps, -1).copy()

        # proprio / state: subsample by frameskip.
        proprio = f["proprio"][g_start:g_end][:: frameskip].copy()
        state = f["state"][g_start:g_end][:: frameskip].copy()

    return {
        "pixels": pixels_chw,
        "action": action,
        "proprio": proprio,
        "state": state,
    }


def num_clips(h5_path: str, frameskip: int, num_steps: int) -> int:
    with h5py.File(h5_path, "r") as f:
        ep_lengths = f["ep_len"][:]
    return len(build_clip_indices(ep_lengths, frameskip, num_steps))
