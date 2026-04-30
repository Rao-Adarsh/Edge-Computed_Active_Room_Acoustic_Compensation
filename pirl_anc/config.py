"""
Central configuration dataclasses for the PIRL ANC pipeline.
"""

# Audit notes:
# - The configuration mirrors the audited repository boundaries:
#   pyroomacoustics room settings, the KHCNN-style virtual sensor shape, and
#   the JAX/jwave physics-penalty parameters.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .common import PACKAGE_ROOT, ensure_directory, resolve_package_path, to_jsonable


@dataclass
class RoomConfig:
    room_dim: tuple[float, float] = (6.0, 5.0)
    absorption: float = 0.35
    max_order: int = 10
    fs: int = 16_000
    duration_seconds: float = 2.0
    source_pos: tuple[float, float] = (1.5, 2.5)
    array_center: tuple[float, float] = (3.0, 2.5)
    target_pos: tuple[float, float] = (1.0, 1.0)
    n_mics: int = 8
    mic_radius: float = 0.5
    f_max: float = 400.0
    speed_of_sound: float = 343.0
    strict_spatial_nyquist: bool = False
    secondary_speaker_radius: float = 0.35
    n_speakers: int = 2

    def __repr__(self) -> str:
        return (
            "RoomConfig(room_dim={!r}, absorption={!r}, max_order={!r}, fs={!r}, "
            "n_mics={!r}, mic_radius={!r}, n_speakers={!r})"
        ).format(
            self.room_dim,
            self.absorption,
            self.max_order,
            self.fs,
            self.n_mics,
            self.mic_radius,
            self.n_speakers,
        )


@dataclass
class VirtualSensorConfig:
    seq_len: int = 512
    hop: int = 256
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_fft: int = 128
    freq_bins: int = 64

    def __repr__(self) -> str:
        return (
            "VirtualSensorConfig(seq_len={!r}, hop={!r}, hidden_dim={!r}, "
            "epochs={!r}, batch_size={!r})"
        ).format(
            self.seq_len,
            self.hop,
            self.hidden_dim,
            self.epochs,
            self.batch_size,
        )


@dataclass
class PhysicsConfig:
    lambda_p: float = 0.01
    sound_speed: float = 343.0
    density: float = 1.225
    dt: float = 1.0 / 16_000.0
    dx: float = 0.05
    latent_time_steps: int = 8
    latent_grid: tuple[int, int] = (4, 4)
    pml_size: int = 1

    def __repr__(self) -> str:
        return (
            "PhysicsConfig(lambda_p={!r}, sound_speed={!r}, density={!r}, dt={!r}, dx={!r})"
        ).format(
            self.lambda_p,
            self.sound_speed,
            self.density,
            self.dt,
            self.dx,
        )


@dataclass
class AgentConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 100_000
    exploration_sigma: float = 0.1
    w_sim: float = 1.0
    w_real: float = 0.0

    def __repr__(self) -> str:
        return (
            "AgentConfig(gamma={!r}, tau={!r}, alpha={!r}, learning_rate={!r}, "
            "batch_size={!r}, replay_capacity={!r})"
        ).format(
            self.gamma,
            self.tau,
            self.alpha,
            self.learning_rate,
            self.batch_size,
            self.replay_capacity,
        )


@dataclass
class TrainingConfig:
    pretrain_episodes: int = 500
    finetune_episodes: int = 200
    max_steps_per_episode: int = 200
    reward_plot_name: str = "pretrain_rewards.png"
    sensor_plot_name: str = "virtual_sensor_training.png"

    def __repr__(self) -> str:
        return (
            "TrainingConfig(pretrain_episodes={!r}, finetune_episodes={!r}, "
            "max_steps_per_episode={!r})"
        ).format(
            self.pretrain_episodes,
            self.finetune_episodes,
            self.max_steps_per_episode,
        )


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 921_600
    packet_size: int = 256
    action_dim: Optional[int] = None

    def __repr__(self) -> str:
        return (
            "SerialConfig(port={!r}, baud={!r}, packet_size={!r}, action_dim={!r})"
        ).format(self.port, self.baud, self.packet_size, self.action_dim)


@dataclass
class ProjectConfig:
    room: RoomConfig = field(default_factory=RoomConfig)
    virtual_sensor: VirtualSensorConfig = field(default_factory=VirtualSensorConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)

    def __post_init__(self) -> None:
        if self.serial.action_dim is None:
            self.serial.action_dim = 2 * self.room.n_speakers

    def __repr__(self) -> str:
        return (
            "ProjectConfig(room={!r}, virtual_sensor={!r}, physics={!r}, "
            "agent={!r}, training={!r}, serial={!r})"
        ).format(
            self.room,
            self.virtual_sensor,
            self.physics,
            self.agent,
            self.training,
            self.serial,
        )

    @property
    def data_dir(self) -> Path:
        return resolve_package_path("data")

    @property
    def plots_dir(self) -> Path:
        return resolve_package_path("plots")

    @property
    def checkpoints_dir(self) -> Path:
        return resolve_package_path("checkpoints")

    @property
    def logs_dir(self) -> Path:
        return resolve_package_path("logs")

    @property
    def room_dataset_path(self) -> Path:
        return self.data_dir / "room_dataset.npz"

    @property
    def room_metadata_path(self) -> Path:
        return self.data_dir / "room_dataset.json"

    @property
    def virtual_sensor_checkpoint(self) -> Path:
        return self.checkpoints_dir / "virtual_sensor.pt"

    @property
    def pretrain_checkpoint(self) -> Path:
        return self.checkpoints_dir / "pretrain_agent.pt"

    @property
    def finetune_checkpoint(self) -> Path:
        return self.checkpoints_dir / "finetuned_agent.pt"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.plots_dir, self.checkpoints_dir, self.logs_dir):
            ensure_directory(path)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[Path] = None) -> ProjectConfig:
    """Load a YAML config file into the default project dataclass tree."""
    config = ProjectConfig()
    config.ensure_directories()
    if path is None:
        return config
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    _update_dataclass(config, payload)
    config.__post_init__()
    config.ensure_directories()
    return config


def save_config(path: Path, config: ProjectConfig) -> None:
    """Persist a project config as YAML."""
    ensure_directory(path.parent)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
