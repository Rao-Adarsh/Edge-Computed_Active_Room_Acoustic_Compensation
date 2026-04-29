"""
PyTorch virtual sensor adapter inspired by the audited KHCNN architecture.
"""

# Audit notes:
# - The upstream KHCNN entry point is the TensorFlow/Keras function
#   `build_khcnn_model` in `nah-khcnn/src/network_module/khcnn_network.py`.
# - That function expects complex boundary pressure images shaped either
#   `(B, 8, 8, 2)` or `(B, 16, 64, 2)` where the last axis stores real/imag parts.
# - The upstream outputs are full spatial fields: hologram pressure and surface
#   velocity components, not a scalar target pressure.
# - The repository does not ship a training loop entry point; the only runnable
#   driver is `nah-khcnn/src/test_module/test_kh_rec.py`, which loads weights and
#   runs inference with `model.predict(...)`.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from ..common import (
    build_sliding_windows,
    ensure_directory,
    require_nah_khcnn_module,
    resolve_package_path,
    select_device,
)
from ..config import load_config


def load_audited_khcnn_reference() -> dict[str, Any]:
    """
    Import the audited KHCNN builder to capture the upstream reference metadata.

    The executable virtual sensor remains PyTorch-native because the surrounding
    PIRL stack is trained in Torch and the upstream repo only exposes a Keras
    functional model factory.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        khcnn_network = require_nah_khcnn_module("network_module.khcnn_network")
    except ImportError as exc:
        raise ImportError(
            "nah-khcnn is required to audit the upstream KHCNN definition. "
            "Keep the cloned repository at `./nah-khcnn` so "
            "`./nah-khcnn/src/network_module/khcnn_network.py` remains importable."
        ) from exc

    return {
        "builder_name": khcnn_network.build_khcnn_model.__name__,
        "builder_module": khcnn_network.__file__,
        "input_shapes": {
            "low_resolution": (8, 8, 2),
            "full_resolution": (16, 64, 2),
        },
        "original_outputs": {
            "pressure_hologram": "real/imag pressure field",
            "surface_velocity": "real/imag velocity field",
        },
    }


class ConvBlock(nn.Module):
    """Small convolutional block used by the KHCNN-style backbone."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def __repr__(self) -> str:
        return f"ConvBlock(in_channels={self.layers[0].in_channels}, out_channels={self.layers[0].out_channels})"

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class KHCNNScalarBackbone(nn.Module):
    """U-Net-like backbone adapted from the audited KHCNN encoder-decoder idea."""

    def __init__(self, input_channels: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.enc1 = ConvBlock(input_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvBlock(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.feature_head = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def __repr__(self) -> str:
        return "KHCNNScalarBackbone(input_channels=2, hidden_dim={})".format(
            self.scalar_head[1].in_features
        )

    def forward(self, spectral_image: Tensor) -> Tensor:
        e1 = self.enc1(spectral_image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))

        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        pooled = self.feature_head(d1)
        return self.scalar_head(pooled)


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]

    def __repr__(self) -> str:
        return f"TrainingHistory(train_losses={len(self.train_losses)}, val_losses={len(self.val_losses)})"


class VirtualSensor(nn.Module):
    """Estimate scalar target pressure from boundary microphone windows."""

    def __init__(self, n_mics: int, seq_len: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_mics = int(n_mics)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.n_fft = 128
        self.freq_bins = 64
        self.reference_info = load_audited_khcnn_reference()
        self.backbone = KHCNNScalarBackbone(input_channels=2, hidden_dim=self.hidden_dim)

    def __repr__(self) -> str:
        return (
            "VirtualSensor(n_mics={!r}, seq_len={!r}, hidden_dim={!r}, "
            "reference_builder={!r})"
        ).format(
            self.n_mics,
            self.seq_len,
            self.hidden_dim,
            self.reference_info["builder_name"],
        )

    def _boundary_to_spectral_image(self, boundary_data: Tensor) -> Tensor:
        if boundary_data.ndim != 3:
            raise ValueError("boundary_data must have shape (batch, n_mics, time).")
        if boundary_data.shape[1] != self.n_mics:
            raise ValueError(f"Expected {self.n_mics} microphones, got {boundary_data.shape[1]}.")
        if boundary_data.shape[2] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {boundary_data.shape[2]}.")

        spectrum = torch.fft.rfft(boundary_data, n=self.n_fft, dim=-1)[..., : self.freq_bins]
        real = spectrum.real
        imag = spectrum.imag
        stacked = torch.stack([real, imag], dim=1)
        scale = stacked.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return stacked / scale

    def forward(self, boundary_data: Tensor) -> Tensor:
        spectral_image = self._boundary_to_spectral_image(boundary_data.float())
        prediction = self.backbone(spectral_image)
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise AssertionError("VirtualSensor must return shape (batch, 1).")
        return prediction


def _split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float = 0.2) -> tuple[TensorDataset, TensorDataset]:
    n_samples = x.shape[0]
    split = max(1, int(round(n_samples * (1.0 - val_ratio))))
    split = min(split, n_samples - 1)
    indices = np.arange(n_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))
    return train_ds, val_ds


def _plot_history(history: TrainingHistory, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    plt.figure(figsize=(8, 4))
    plt.plot(history.train_losses, label="train")
    plt.plot(history.val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Virtual Sensor Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_virtual_sensor(
    dataset_path: Path,
    save_path: Path,
    epochs: int = 100,
) -> tuple[VirtualSensor, TrainingHistory]:
    """Train the PyTorch virtual sensor on the phase-1 room dataset."""
    data = np.load(Path(dataset_path).expanduser().resolve())
    config = load_config()
    x, y = build_sliding_windows(
        data["boundary_signals"].astype(np.float32),
        data["target_signal"].astype(np.float32),
        window=config.virtual_sensor.seq_len,
        hop=config.virtual_sensor.hop,
    )
    train_ds, val_ds = _split_train_val(x, y)

    device = select_device()
    model = VirtualSensor(n_mics=x.shape[1], seq_len=x.shape[2], hidden_dim=config.virtual_sensor.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.virtual_sensor.learning_rate)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=config.virtual_sensor.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.virtual_sensor.batch_size, shuffle=False)

    history = TrainingHistory(train_losses=[], val_losses=[])
    best_val = float("inf")
    best_payload: Optional[dict[str, Any]] = None

    for _ in trange(epochs, desc="virtual-sensor", leave=False):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                val_losses.append(float(criterion(preds, batch_y).detach().cpu().item()))

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history.train_losses.append(mean_train)
        history.val_losses.append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_payload = {
                "state_dict": model.state_dict(),
                "n_mics": model.n_mics,
                "seq_len": model.seq_len,
                "hidden_dim": model.hidden_dim,
                "history": {
                    "train_losses": history.train_losses,
                    "val_losses": history.val_losses,
                },
                "reference_info": model.reference_info,
            }

    if best_payload is None:
        raise RuntimeError("Virtual sensor training did not produce a checkpoint payload.")

    save_path = Path(save_path).expanduser().resolve()
    ensure_directory(save_path.parent)
    torch.save(best_payload, save_path)
    model.load_state_dict(best_payload["state_dict"])
    _plot_history(history, resolve_package_path("plots", "virtual_sensor_training.png"))
    return model, history


def load_virtual_sensor(checkpoint_path: Path, device: Optional[torch.device] = None) -> VirtualSensor:
    """Load a trained virtual sensor checkpoint."""
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    model = VirtualSensor(
        n_mics=int(payload["n_mics"]),
        seq_len=int(payload["seq_len"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model


def _smoke_test() -> None:
    dummy = torch.randn(4, 8, 512)
    model = VirtualSensor(n_mics=8, seq_len=512)
    out = model(dummy)
    assert out.shape == (4, 1)


if __name__ == "__main__":
    config = load_config()
    dataset_path = config.room_dataset_path
    checkpoint_path = config.virtual_sensor_checkpoint
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `python -m pirl_anc.simulation.room_simulator` first."
        )
    train_virtual_sensor(dataset_path, checkpoint_path, epochs=config.virtual_sensor.epochs)
    _smoke_test()
    print(f"Saved virtual sensor checkpoint to {checkpoint_path}")
