"""
Shared helpers for paths, imports, signal features, and filesystem utilities.
"""

# Audit notes:
# - This module centralizes runtime access to the audited repositories so the
#   rest of the project can raise consistent installation/import guidance.
# - `nah-khcnn` is consumed through its `src/` tree, while `jwave` and
#   `pyroomacoustics` are regular Python imports once installed.

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent

NAH_KHCNN_SRC = WORKSPACE_ROOT / "nah-khcnn" / "src"
JWAVE_REPO = WORKSPACE_ROOT / "jwave"
PYROOMACOUSTICS_REPO = WORKSPACE_ROOT / "pyroomacoustics"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_package_path(*parts: str) -> Path:
    """Resolve a path relative to the `pirl_anc` package root."""
    return PACKAGE_ROOT.joinpath(*parts).resolve()


def prepend_nah_khcnn_src() -> None:
    """Add the audited `nah-khcnn/src` tree to `sys.path` when needed."""
    nah_path = str(NAH_KHCNN_SRC.resolve())
    if nah_path not in sys.path:
        sys.path.insert(0, nah_path)


def require_jwave() -> Any:
    """Import `jwave` with a targeted installation hint."""
    try:
        import jwave

        return jwave
    except ImportError as exc:  # pragma: no cover - exercised by environment
        raise ImportError(
            "jwave is required here. Install the cloned source with "
            "`.\\.venv\\Scripts\\python.exe -m pip install -e .\\jwave`."
        ) from exc


def require_pyroomacoustics() -> Any:
    """Import `pyroomacoustics` with a targeted installation hint."""
    try:
        import pyroomacoustics as pra

        return pra
    except ImportError as exc:  # pragma: no cover - exercised by environment
        raise ImportError(
            "pyroomacoustics is required here. Preferred local install command: "
            "`.\\.venv\\Scripts\\python.exe -m pip install -e .\\pyroomacoustics`. "
            "If the local editable build is blocked by missing Windows build tools, "
            "install the runtime wheel with "
            "`.\\.venv\\Scripts\\python.exe -m pip install pyroomacoustics`."
        ) from exc


def require_nah_khcnn_module(module_name: str) -> Any:
    """Import a module from `nah-khcnn/src` with a clear guidance message."""
    prepend_nah_khcnn_src()
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - exercised by environment
        raise ImportError(
            "nah-khcnn is required here. Keep the cloned repository at "
            "`./nah-khcnn` and ensure `./nah-khcnn/src` is importable. "
            "This project expects the audited source tree in that location."
        ) from exc


def to_jsonable(data: Any) -> Any:
    """Recursively convert dataclasses, arrays, and paths into JSON-safe types."""
    if is_dataclass(data):
        return to_jsonable(asdict(data))
    if isinstance(data, dict):
        return {str(key): to_jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_jsonable(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def write_json(path: Path, data: Any) -> None:
    """Write structured JSON with parent creation."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def build_sliding_windows(
    boundary_signals: np.ndarray,
    target_signal: np.ndarray,
    window: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert full-length boundary and target signals into supervised windows."""
    if boundary_signals.ndim != 2:
        raise ValueError("boundary_signals must have shape (n_mics, n_samples).")
    if target_signal.ndim != 1:
        raise ValueError("target_signal must have shape (n_samples,).")
    n_mics, total_samples = boundary_signals.shape
    if target_signal.shape[0] != total_samples:
        raise ValueError("boundary_signals and target_signal must share a timeline.")
    if total_samples < window:
        raise ValueError("window cannot exceed the available signal length.")

    xs: list[np.ndarray] = []
    ys: list[float] = []
    center_offset = window // 2
    for start in range(0, total_samples - window + 1, hop):
        stop = start + window
        xs.append(boundary_signals[:, start:stop])
        ys.append(float(target_signal[start + center_offset]))

    x_array = np.stack(xs, axis=0).astype(np.float32)
    y_array = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    if x_array.shape[1] != n_mics or x_array.shape[2] != window:
        raise AssertionError("Sliding window construction produced incorrect shapes.")
    return x_array, y_array


def _stft_tensor(
    boundary_window: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
) -> torch.Tensor:
    """Compute a complex STFT over `(batch, n_mics, time)` tensors."""
    if boundary_window.ndim != 3:
        raise ValueError("boundary_window must have shape (batch, n_mics, time).")
    batch, n_mics, time = boundary_window.shape
    flat = boundary_window.reshape(batch * n_mics, time)
    window = torch.hann_window(n_fft, device=boundary_window.device, dtype=boundary_window.dtype)
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return spec.reshape(batch, n_mics, spec.shape[-2], spec.shape[-1])


def compute_state_features(
    boundary_window: np.ndarray | torch.Tensor,
    freq_bins: int = 64,
    n_fft: int = 256,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Build the flattened magnitude-STFT features used by the ANC agent state.

    The result has shape `(n_mics * freq_bins,)` after averaging over STFT frames.
    """
    tensor = boundary_window
    if isinstance(boundary_window, np.ndarray):
        tensor = torch.from_numpy(boundary_window.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("boundary_window must have shape (n_mics, time) or (batch, n_mics, time).")
    spec = _stft_tensor(tensor, n_fft=n_fft, hop_length=hop_length)
    magnitude = spec.abs()[..., :freq_bins, :]
    pooled = magnitude.mean(dim=-1)
    flattened = pooled.reshape(pooled.shape[0], -1)
    return flattened[0].detach().cpu().numpy()


def action_bounds(action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-dimension lower and upper bounds for amplitude/phase actions."""
    lower = np.zeros(action_dim, dtype=np.float32)
    upper = np.zeros(action_dim, dtype=np.float32)
    for index in range(action_dim):
        if index % 2 == 0:
            upper[index] = 1.0
        else:
            upper[index] = float(2.0 * math.pi)
    return lower, upper


def clip_action_vector(action: np.ndarray | torch.Tensor) -> np.ndarray:
    """Clip an amplitude/phase action vector to hardware-safe bounds."""
    array = np.asarray(action, dtype=np.float32).copy()
    low, high = action_bounds(array.size)
    return np.clip(array, low, high)


def pairwise_secondary_positions(
    center: Sequence[float],
    radius: float,
    n_speakers: int,
) -> np.ndarray:
    """Place secondary sources uniformly around the control point."""
    center_array = np.asarray(center, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, n_speakers, endpoint=False)
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(radius)
    return center_array[None, :] + offsets


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Pick a torch device that matches the available runtime."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_loss_rows(rows: Iterable[dict[str, Any]]) -> str:
    """Convert a sequence of dictionaries to CSV content."""
    rows = list(rows)
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in keys))
    return "\n".join(lines) + "\n"
