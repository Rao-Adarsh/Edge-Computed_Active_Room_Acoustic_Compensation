"""
Room simulation and dataset generation for 2D active noise control experiments.
"""

# Audit notes:
# - `pyroomacoustics` instantiates a 2D shoebox room with
#   `pra.ShoeBox(room_dim, fs=..., materials=pra.Material(...), max_order=...)`.
# - Sources are added with `room.add_source(position, signal=...)`.
# - Microphones are added with `room.add_microphone_array(mic_positions)` where
#   the geometry is an array of shape `(2, n_mics)` in 2D.
# - `room.compute_rir()` stores RIRs as `room.rir[mic_index][source_index]`.
# - `room.simulate()` writes time-domain microphone signals into
#   `room.mic_array.signals` with shape `(n_mics, n_samples)`.

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve

from ..common import (
    clip_action_vector,
    ensure_directory,
    pairwise_secondary_positions,
    require_pyroomacoustics,
    write_json,
)
from ..config import RoomConfig, load_config


class RoomSimulator:
    """Generate ANC training data from a 2D shoebox room."""

    def __init__(
        self,
        room_dim: Sequence[float] = (6.0, 5.0),
        absorption: float = 0.35,
        max_order: int = 10,
        fs: int = 16_000,
        duration_seconds: float = 2.0,
        source_pos: Sequence[float] = (1.5, 2.5),
        array_center: Sequence[float] = (3.0, 2.5),
        mic_radius: float = 0.5,
        n_mics: int = 8,
        target_pos: Sequence[float] = (3.0, 2.5),
        f_max: float = 4_000.0,
        speed_of_sound: float = 343.0,
        strict_spatial_nyquist: bool = False,
        n_secondary_speakers: int = 2,
        secondary_speaker_radius: float = 0.35,
        rng_seed: Optional[int] = 0,
    ) -> None:
        self.room_dim = tuple(float(value) for value in room_dim)
        self.absorption = float(absorption)
        self.max_order = int(max_order)
        self.fs = int(fs)
        self.duration_seconds = float(duration_seconds)
        self.source_pos = np.asarray(source_pos, dtype=np.float32)
        self.array_center = np.asarray(array_center, dtype=np.float32)
        self.mic_radius = float(mic_radius)
        self.n_mics = int(n_mics)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.f_max = float(f_max)
        self.speed_of_sound = float(speed_of_sound)
        self.strict_spatial_nyquist = bool(strict_spatial_nyquist)
        self.n_secondary_speakers = int(n_secondary_speakers)
        self.secondary_speaker_radius = float(secondary_speaker_radius)
        self.rng = np.random.default_rng(rng_seed)
        self._cached_simulation: Optional[dict[str, Any]] = None
        self._cached_secondary_rirs: Optional[dict[str, np.ndarray]] = None

        self._validate_spatial_nyquist()

    def __repr__(self) -> str:
        return (
            "RoomSimulator(room_dim={!r}, absorption={!r}, max_order={!r}, fs={!r}, "
            "duration_seconds={!r}, n_mics={!r}, mic_radius={!r}, source_pos={!r})"
        ).format(
            self.room_dim,
            self.absorption,
            self.max_order,
            self.fs,
            self.duration_seconds,
            self.n_mics,
            self.mic_radius,
            self.source_pos.tolist(),
        )

    @property
    def duration_samples(self) -> int:
        return int(round(self.duration_seconds * self.fs))

    @property
    def mic_positions(self) -> np.ndarray:
        pra = require_pyroomacoustics()
        return pra.circular_2D_array(self.array_center, self.n_mics, 0.0, self.mic_radius)

    @property
    def secondary_positions(self) -> np.ndarray:
        return pairwise_secondary_positions(
            self.target_pos,
            self.secondary_speaker_radius,
            self.n_secondary_speakers,
        ).astype(np.float32)

    def _adjacent_mic_spacing(self) -> float:
        return float(2.0 * self.mic_radius * np.sin(np.pi / self.n_mics))

    def _validate_spatial_nyquist(self) -> None:
        spacing = self._adjacent_mic_spacing()
        threshold = self.speed_of_sound / (2.0 * self.f_max)
        if spacing < threshold:
            return
        message = (
            "Spatial Nyquist violated for the requested array: "
            f"d = 2*r*sin(pi/N) = {spacing:.6f} m must satisfy "
            f"d < c/(2*f_max) = {threshold:.6f} m "
            f"with c={self.speed_of_sound:.1f} m/s and f_max={self.f_max:.1f} Hz."
        )
        if self.strict_spatial_nyquist:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _generate_primary_signal(self) -> np.ndarray:
        samples = self.duration_samples
        white = self.rng.standard_normal(samples + 127).astype(np.float32)
        taps = self.rng.standard_normal(128).astype(np.float32)
        envelope = np.exp(-np.linspace(0.0, 5.0, taps.size, dtype=np.float32))
        fir = taps * envelope
        fir /= np.linalg.norm(fir) + 1e-8
        filtered = np.convolve(white, fir, mode="valid")[:samples]
        filtered /= np.std(filtered) + 1e-8
        return filtered.astype(np.float32)

    def _build_room(
        self,
        source_positions: Sequence[np.ndarray],
        source_signals: Sequence[Optional[np.ndarray]],
    ) -> Any:
        pra = require_pyroomacoustics()
        room = pra.ShoeBox(
            self.room_dim,
            fs=self.fs,
            materials=pra.Material(energy_absorption=self.absorption),
            max_order=self.max_order,
        )
        for position, signal in zip(source_positions, source_signals):
            room.add_source(position.tolist(), signal=signal)
        all_mics = np.concatenate([self.mic_positions, self.target_pos[:, None]], axis=1)
        room.add_microphone_array(all_mics)
        room.compute_rir()
        room.simulate()
        return room

    @staticmethod
    def _pad_rirs(rirs: Sequence[np.ndarray]) -> np.ndarray:
        max_len = max(len(rir) for rir in rirs)
        matrix = np.zeros((len(rirs), max_len), dtype=np.float32)
        for row, rir in enumerate(rirs):
            matrix[row, : len(rir)] = np.asarray(rir, dtype=np.float32)
        return matrix

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "fs": self.fs,
            "room_dim": list(self.room_dim),
            "absorption": self.absorption,
            "max_order": self.max_order,
            "source_pos": self.source_pos.tolist(),
            "mic_positions": self.mic_positions.T.tolist(),
            "target_pos": self.target_pos.tolist(),
            "n_secondary_speakers": self.n_secondary_speakers,
            "secondary_positions": self.secondary_positions.tolist(),
            "spatial_nyquist_spacing_m": self._adjacent_mic_spacing(),
            "spatial_nyquist_limit_m": self.speed_of_sound / (2.0 * self.f_max),
        }

    def simulate(
        self,
        primary_signal: Optional[np.ndarray] = None,
        refresh: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Run the room simulation and return boundary signals, target signal, and RIRs."""
        if self._cached_simulation is not None and not refresh and primary_signal is None:
            cached = self._cached_simulation
            return (
                cached["boundary_signals"],
                cached["target_signal"],
                cached["rir_boundary"],
                cached["rir_target"],
                cached["metadata"],
            )

        signal = primary_signal.astype(np.float32) if primary_signal is not None else self._generate_primary_signal()
        room = self._build_room([self.source_pos], [signal])
        signals = np.asarray(room.mic_array.signals, dtype=np.float32)
        boundary_signals = signals[: self.n_mics, :]
        target_signal = signals[self.n_mics, :]
        rir_boundary = self._pad_rirs([room.rir[index][0] for index in range(self.n_mics)])
        rir_target = np.asarray(room.rir[self.n_mics][0], dtype=np.float32)
        metadata = self._build_metadata()
        self._cached_simulation = {
            "primary_signal": signal,
            "boundary_signals": boundary_signals,
            "target_signal": target_signal,
            "rir_boundary": rir_boundary,
            "rir_target": rir_target,
            "metadata": metadata,
        }
        return boundary_signals, target_signal, rir_boundary, rir_target, metadata

    def compute_secondary_rirs(self, refresh: bool = False) -> dict[str, np.ndarray]:
        """Compute transfer responses from the secondary speakers to all receivers."""
        if self._cached_secondary_rirs is not None and not refresh:
            return self._cached_secondary_rirs

        silent_sources = [np.zeros(self.duration_samples, dtype=np.float32) for _ in range(self.n_secondary_speakers)]
        room = self._build_room(list(self.secondary_positions), silent_sources)
        boundary_rirs = []
        target_rirs = []
        for source_index in range(self.n_secondary_speakers):
            boundary_rirs.append(self._pad_rirs([room.rir[mic_index][source_index] for mic_index in range(self.n_mics)]))
            target_rirs.append(np.asarray(room.rir[self.n_mics][source_index], dtype=np.float32))

        max_target_len = max(len(rir) for rir in target_rirs)
        padded_target = np.zeros((self.n_secondary_speakers, max_target_len), dtype=np.float32)
        for index, rir in enumerate(target_rirs):
            padded_target[index, : len(rir)] = rir

        self._cached_secondary_rirs = {
            "boundary": np.stack(boundary_rirs, axis=0),
            "target": padded_target,
        }
        return self._cached_secondary_rirs

    def generate_secondary_source_signals(
        self,
        action_vector: np.ndarray,
        reference_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Map amplitude/phase actions to time-domain anti-noise source signals."""
        action = clip_action_vector(action_vector)
        if action.size != 2 * self.n_secondary_speakers:
            raise ValueError(
                f"Expected action_dim={2 * self.n_secondary_speakers}, got {action.size}."
            )
        if reference_signal is None:
            _, _, _, _, _ = self.simulate()
            assert self._cached_simulation is not None
            reference = self._cached_simulation["primary_signal"]
        else:
            reference = np.asarray(reference_signal, dtype=np.float32)

        spectrum = np.fft.rfft(reference)
        signals = []
        for speaker_index in range(self.n_secondary_speakers):
            amplitude = float(action[2 * speaker_index])
            phase = float(action[2 * speaker_index + 1])
            shifted = np.fft.irfft(-amplitude * spectrum * np.exp(1j * phase), n=reference.size)
            signals.append(shifted.astype(np.float32))
        return np.stack(signals, axis=0)

    def simulate_with_secondary_sources(
        self,
        action_vector: np.ndarray,
        reference_signal: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Apply secondary source actions through cached RIRs and return controlled signals."""
        boundary_signals, target_signal, _, _, _ = self.simulate()
        secondary_rirs = self.compute_secondary_rirs()
        source_signals = self.generate_secondary_source_signals(action_vector, reference_signal)

        boundary_delta = np.zeros_like(boundary_signals)
        target_delta = np.zeros_like(target_signal)
        for speaker_index in range(self.n_secondary_speakers):
            speaker_signal = source_signals[speaker_index]
            for mic_index in range(self.n_mics):
                rir = secondary_rirs["boundary"][speaker_index, mic_index]
                boundary_delta[mic_index] += fftconvolve(speaker_signal, rir, mode="full")[: boundary_signals.shape[1]]
            target_rir = secondary_rirs["target"][speaker_index]
            target_delta += fftconvolve(speaker_signal, target_rir, mode="full")[: target_signal.shape[0]]

        return {
            "boundary_signals": boundary_signals + boundary_delta,
            "target_signal": target_signal + target_delta,
            "secondary_signals": source_signals,
        }

    def save_dataset(self, path: Path) -> tuple[Path, Path]:
        """Persist the latest simulation arrays as `.npz` and the metadata as JSON."""
        boundary_signals, target_signal, rir_boundary, rir_target, metadata = self.simulate()
        resolved = Path(path).expanduser().resolve()
        ensure_directory(resolved.parent)
        np.savez(
            resolved,
            boundary_signals=boundary_signals,
            target_signal=target_signal,
            rir_boundary=rir_boundary,
            rir_target=rir_target,
        )
        metadata_path = resolved.with_suffix(".json")
        write_json(metadata_path, metadata)
        return resolved, metadata_path

    @classmethod
    def from_config(cls, config: RoomConfig) -> "RoomSimulator":
        """Construct a room simulator directly from the shared config dataclass."""
        return cls(
            room_dim=config.room_dim,
            absorption=config.absorption,
            max_order=config.max_order,
            fs=config.fs,
            duration_seconds=config.duration_seconds,
            source_pos=config.source_pos,
            array_center=config.array_center,
            mic_radius=config.mic_radius,
            n_mics=config.n_mics,
            target_pos=config.target_pos,
            f_max=config.f_max,
            speed_of_sound=config.speed_of_sound,
            strict_spatial_nyquist=config.strict_spatial_nyquist,
            n_secondary_speakers=config.n_speakers,
            secondary_speaker_radius=config.secondary_speaker_radius,
        )


def _smoke_test() -> None:
    simulator = RoomSimulator(fs=16_000, duration_seconds=0.1, strict_spatial_nyquist=False)
    boundary, target, rir_boundary, rir_target, metadata = simulator.simulate(refresh=True)
    assert boundary.shape[0] == simulator.n_mics
    assert target.ndim == 1
    assert rir_boundary.ndim == 2
    assert rir_target.ndim == 1
    assert metadata["fs"] == 16_000


if __name__ == "__main__":
    config = load_config()
    simulator = RoomSimulator.from_config(config.room)
    dataset_path, metadata_path = simulator.save_dataset(config.room_dataset_path)
    _smoke_test()
    print(f"Saved dataset to {dataset_path}")
    print(f"Saved metadata to {metadata_path}")
