"""
Gymnasium environment for 2D spatial Active Noise Control.

Audit findings used in this file:
  - RoomSimulator.simulate() returns:
      (boundary_signals[n_mics, T], target_signal[T], rir_boundary, rir_target, metadata)
  - RoomSimulator.from_config(config.room) constructs from RoomConfig dataclass
  - RoomSimulator accepts refresh=True to regenerate signals each episode
  - VirtualSensor.forward(boundary_data) expects (B, n_mics, seq_len) → (B, 1)
  - load_virtual_sensor(checkpoint_path, device) → VirtualSensor in eval mode
  - build_state_vector(virtual_pressure, boundary_signals) from sac_agent
  - action_dim = 2 * n_speakers (amplitude, phase per speaker)
  - state_dim  = 1 + n_mics * n_freq_bins
  - config.room.fs = 16000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium
import numpy as np
import torch

from ..config import ProjectConfig, load_config
from ..simulation.room_simulator import RoomSimulator

try:
    from ..virtual_sensor.kh_virtual_sensor import VirtualSensor, load_virtual_sensor
except ImportError:
    VirtualSensor = None  # type: ignore[assignment,misc]
    load_virtual_sensor = None  # type: ignore[assignment]

from ..agent.sac_agent import build_state_vector


class ANCEnvironment(gymnasium.Env):
    """Gymnasium environment wrapping a pyroomacoustics 2D room for ANC training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path | None = None,
        window: int = 512,
        hop: int = 256,
        episode_len: int = 200,
        n_mics: int = 8,
        n_speakers: int = 2,
        n_freq_bins: int = 64,
        fs: int = 16_000,
    ) -> None:
        super().__init__()

        # Load config if provided; override defaults from it
        if config_path is not None:
            cfg = load_config(Path(config_path))
        else:
            cfg = load_config()

        self.window = int(window)
        self.hop = int(hop)
        self.episode_len = int(episode_len)
        self.n_mics = int(n_mics)
        self.n_speakers = int(n_speakers)
        self.n_freq_bins = int(n_freq_bins)
        self.fs = int(fs)

        # Reward tuning
        self.alpha_reward = 1.0
        self.beta_reward = 0.0

        # Build room simulator from config
        self.room = RoomSimulator.from_config(cfg.room)

        # Run one simulation to set signal lengths
        boundary, target, _, _, _ = self.room.simulate()
        self._signal_length = target.shape[0]

        # Dimensions
        self._action_dim = 2 * self.n_speakers  # amplitude + phase per speaker
        self._state_dim = 1 + self.n_mics * self.n_freq_bins

        # Gymnasium spaces
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._state_dim,),
            dtype=np.float32,
        )

        # Action bounds: amplitude ∈ [0, 1], phase ∈ [0, 2π] per speaker
        low_list = []
        high_list = []
        for _ in range(self.n_speakers):
            low_list.extend([0.0, 0.0])
            high_list.extend([1.0, 2.0 * np.pi])
        self.action_space = gymnasium.spaces.Box(
            low=np.array(low_list, dtype=np.float32),
            high=np.array(high_list, dtype=np.float32),
            dtype=np.float32,
        )

        # Virtual sensor (lazy-loaded)
        self._virtual_sensor: Optional[Any] = None

        # Episode state
        self._step_idx = 0
        self._boundary_signals: Optional[np.ndarray] = None  # (n_mics, T)
        self._target_signal: Optional[np.ndarray] = None  # (T,)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def set_reward_weights(self, alpha: float, beta: float) -> None:
        """Dynamically update the hybrid reward weights."""
        self.alpha_reward = float(alpha)
        self.beta_reward = float(beta)

    def _load_virtual_sensor(self, checkpoint_path: str | Path) -> None:
        """Load a trained VirtualSensor from a checkpoint file."""
        if load_virtual_sensor is None:
            raise ImportError(
                "VirtualSensor is not available. Ensure nah-khcnn is installed."
            )
        self._virtual_sensor = load_virtual_sensor(
            Path(checkpoint_path), device=torch.device("cpu"),
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Re-run room simulation and return the initial observation."""
        super().reset(seed=seed)

        # Regenerate signals for a fresh episode
        boundary, target, _, _, _ = self.room.simulate(refresh=True)
        self._boundary_signals = boundary  # (n_mics, T)
        self._target_signal = target  # (T,)
        self._signal_length = target.shape[0]
        self._step_idx = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        """Build the current observation vector from boundary signals."""
        assert self._boundary_signals is not None
        assert self._target_signal is not None

        start = self._step_idx * self.hop
        end = start + self.window

        # Clamp to available signal length
        actual_end = min(end, self._boundary_signals.shape[1])
        boundary_window = self._boundary_signals[:, start:actual_end]

        # Pad if necessary
        if boundary_window.shape[1] < self.window:
            pad_width = self.window - boundary_window.shape[1]
            boundary_window = np.pad(
                boundary_window, ((0, 0), (0, pad_width)), mode="constant",
            )

        # Convert to torch: (1, n_mics, window)
        boundary_tensor = torch.from_numpy(
            boundary_window.astype(np.float32),
        ).unsqueeze(0)

        # Virtual pressure
        if self._virtual_sensor is not None:
            with torch.no_grad():
                virtual_pressure = self._virtual_sensor(boundary_tensor)  # (1, 1)
        else:
            # Fallback: use target signal value at the window center
            center_idx = min(start + self.window // 2, len(self._target_signal) - 1)
            virtual_pressure = torch.tensor(
                [[self._target_signal[center_idx]]], dtype=torch.float32,
            )

        # Build state vector
        state = build_state_vector(
            virtual_pressure, boundary_tensor,
            n_fft=min(256, self.window),
            hop_length=128,
            n_freq_bins=self.n_freq_bins,
        )
        return state.squeeze(0).numpy()

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply anti-noise action and return (obs, reward, terminated, truncated, info)."""
        assert self._target_signal is not None

        # Rescale actions from agent's normalized [-1, 1] space to physical bounds.
        # The agent uses tanh squashing, so all values arrive in [-1, 1].
        # Map to: amplitude ∈ [0, 1], phase ∈ [0, 2π] per speaker.
        action = np.asarray(action, dtype=np.float32).copy()
        low = self.action_space.low
        high = self.action_space.high
        action_rescaled = low + (action + 1.0) * 0.5 * (high - low)
        action_rescaled = np.clip(action_rescaled, low, high)

        # Interpret action: amplitude and phase per speaker
        t = np.arange(self.window, dtype=np.float32) / self.fs
        anti_noise = np.zeros(self.window, dtype=np.float32)
        for i in range(self.n_speakers):
            amplitude = float(action_rescaled[2 * i])
            phase = float(action_rescaled[2 * i + 1])
            # 440 Hz carrier is a placeholder; real anti-noise would match
            # the noise frequency content via spectral inversion.
            anti_noise += amplitude * np.sin(2.0 * np.pi * 440.0 * t + phase)

        # Compute residual pressure at target
        start = self._step_idx * self.hop
        end = start + self.window
        actual_end = min(end, len(self._target_signal))
        target_window = self._target_signal[start:actual_end]

        # Truncate anti-noise to match available target samples
        an = anti_noise[: len(target_window)]
        # Reward: Hybrid Time-Frequency MSE
        target_t = torch.from_numpy(target_window).float()
        an_t = torch.from_numpy(an).float()

        # 1. Time-domain MSE (phase-sensitive)
        mse_time = torch.mean((target_t + an_t) ** 2)

        # 2. Frequency-domain MSE (phase-invariant)
        # Apply windowing to reduce spectral leakage
        hann_win = torch.hann_window(len(target_t))
        target_fft = torch.fft.rfft(target_t * hann_win)
        an_fft = torch.fft.rfft(an_t * hann_win)
        
        # To avoid the silence trap, we compare magnitudes.
        # Ideal anti-noise is -target, so its magnitude should equal the target's magnitude.
        target_mag = torch.abs(target_fft) / len(target_t)
        an_mag = torch.abs(an_fft) / len(target_t)
        mse_freq = torch.mean((target_mag - an_mag) ** 2)

        reward_tensor = -(self.alpha_reward * mse_time + self.beta_reward * mse_freq)
        reward = float(reward_tensor.item())

        # Advance step
        self._step_idx += 1

        # Termination conditions
        terminated = (self._step_idx * self.hop + self.window >= len(self._target_signal))
        truncated = (self._step_idx >= self.episode_len)

        obs = self._get_obs()

        info = {
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "step": self._step_idx,
        }

        return obs, reward/100, terminated, truncated, info

    def render(self) -> None:
        """Rendering not implemented (mode='human' not required)."""
        pass

    def __repr__(self) -> str:
        return (
            f"ANCEnvironment(state_dim={self._state_dim}, "
            f"action_dim={self._action_dim}, "
            f"episode_len={self.episode_len})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== anc_env smoke test ===")

    env = ANCEnvironment()
    print(repr(env))

    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}, state_dim: {env.state_dim}")
    assert obs.shape == (env.state_dim,), f"obs shape mismatch: {obs.shape}"

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward), f"reward not finite at step {step}"
        assert obs.shape == (env.state_dim,), f"obs shape changed at step {step}"
        print(f"  step {step}: reward={reward:.6f}, rms={info['residual_rms']:.6f}")

    print("All assertions passed.")
