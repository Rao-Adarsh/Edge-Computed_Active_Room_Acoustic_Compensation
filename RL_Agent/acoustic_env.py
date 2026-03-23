"""Gymnasium environment for acoustic compensation orchestration.

This environment bridges policy actions, acoustic telemetry acquisition,
state-space construction, and reward computation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
from scipy import signal

from state_space import RewardComputer, StateSpaceFormulator


logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration container for AcousticEnv.

    Attributes:
        target_path (str): Path to ACTRAN target .npy file.
        mode (str): Backend mode label used for future extension.
        n_bands (int): Number of state spectral bands.
        k_bands (int): Number of EQ bands / action dimensions.
        t_max (int): Episode length limit in steps.
        epsilon (float): Convergence threshold for reward terminal condition.
        action_scale (float): Action-to-dB delta scaling factor.
        rt60 (float): Simulated reverberation decay time in seconds.
        sample_rate (int): Audio sample rate in Hz.
        window_size (int): Audio window length per step.
    """

    target_path: str
    mode: str = "simulated"
    n_bands: int = 32
    k_bands: int = 10
    t_max: int = 200
    epsilon: float = 2.0
    action_scale: float = 3.0
    rt60: float = 0.4
    sample_rate: int = 44100
    window_size: int = 2048


class AcousticEnv(gymnasium.Env):
    """Custom Gymnasium environment for acoustic compensation.

    The environment consumes delta actions from a SAC policy, simulates or
    acquires microphone telemetry, formulates state vectors, and computes
    scalar rewards for convergence toward a target response.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target_path: Union[str, EnvConfig],
        n_bands: int = 32,
        k_bands: int = 10,
        t_max: int = 200,
        epsilon: float = 2.0,
        action_scale: float = 3.0,
        rt60: float = 0.4,
        sample_rate: int = 44100,
        window_size: int = 2048,
        reward_kwargs: Optional[Dict[str, Any]] = None,
        mode: str = "simulated",
    ) -> None:
        """Initialize the acoustic compensation environment.

        Args:
            target_path (Union[str, EnvConfig]): Either a filesystem path to
                the ACTRAN target .npy file, or an EnvConfig instance.
            n_bands (int): Number of spectral bands for state representation.
            k_bands (int): Number of EQ bands and action dimensions.
            t_max (int): Maximum steps per episode.
            epsilon (float): Convergence threshold used by reward computation.
            action_scale (float): Scale factor mapping policy action from
                ``[-1, 1]`` to dB delta per step.
            rt60 (float): Simulated reverberation time in seconds.
            sample_rate (int): Audio sample rate in Hz.
            window_size (int): Number of samples in each audio frame.
            reward_kwargs (Optional[Dict[str, Any]]): Extra keyword arguments
                forwarded to RewardComputer constructor.
            mode (str): Backend mode label for compatibility/future extension.

        Returns:
            None.

        Raises:
            ValueError: If ``k_bands`` is not supported by the cached filter
                bank design.
            FileNotFoundError: If target_path does not exist.
        """
        super().__init__()

        if isinstance(target_path, EnvConfig):
            cfg = target_path
            target_file = cfg.target_path
            mode = cfg.mode
            n_bands = cfg.n_bands
            k_bands = cfg.k_bands
            t_max = cfg.t_max
            epsilon = cfg.epsilon
            action_scale = cfg.action_scale
            rt60 = cfg.rt60
            sample_rate = cfg.sample_rate
            window_size = cfg.window_size
        else:
            target_file = target_path

        self.mode = mode
        self.n_bands = int(n_bands)
        self.k_bands = int(k_bands)
        self.t_max = int(t_max)
        self.epsilon = float(epsilon)
        self.action_scale = float(action_scale)
        self.rt60 = float(rt60)
        self.sample_rate = int(sample_rate)
        self.window_size = int(window_size)

        self.formulator = StateSpaceFormulator(
            target_path=target_file,
            n_bands=self.n_bands,
            k_bands=self.k_bands,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
        )
        self.reward_computer = RewardComputer(**(reward_kwargs or {}))

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.formulator.state_dim,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.k_bands,),
            dtype=np.float32,
        )

        self._filter_weights = np.zeros(self.k_bands, dtype=np.float32)
        self._step_count = 0
        self._prev_error: Optional[np.ndarray] = None
        self._episode_reward = 0.0

        self._band_centers_hz = np.array(
            [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
            dtype=np.float32,
        )
        if self.k_bands != len(self._band_centers_hz):
            raise ValueError(
                "This environment currently supports exactly 10 EQ bands; "
                f"received k_bands={self.k_bands}."
            )

        # Cache the SOS filter bank once at init to avoid expensive per-step redesign.
        self._sos_filters = self._build_sos_filter_bank()
        self._test_signal = self._build_log_sweep()
        self._rir = self._build_rir()

        if self.mode != "simulated":
            logger.warning(
                "Mode '%s' requested but only simulated telemetry is currently "
                "implemented in _acquire_hardware_telemetry.",
                self.mode,
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment at episode start.

        Args:
            seed (Optional[int]): Random seed forwarded to Gymnasium reset.
            options (Optional[Dict[str, Any]]): Reset options. If it contains
                ``initial_weights``, those are used as starting filter weights.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - Initial observation ``S_0``.
                - Diagnostic info dictionary.

        Raises:
            ValueError: If provided ``initial_weights`` has invalid shape.
        """
        super().reset(seed=seed)
        options = options or {}

        if "initial_weights" in options:
            initial_weights = np.asarray(options["initial_weights"], dtype=np.float32)
            if initial_weights.shape != (self.k_bands,):
                raise ValueError(
                    "initial_weights shape mismatch: expected "
                    f"({self.k_bands},), got {initial_weights.shape}"
                )
        else:
            initial_weights = self.np_random.uniform(
                -6.0, 6.0, size=(self.k_bands,)
            ).astype(np.float32)

        self._filter_weights = np.clip(initial_weights, -12.0, 12.0).astype(np.float32)
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_error = None

        self.formulator.reset()

        mic_audio = self._acquire_hardware_telemetry(self._filter_weights)
        state = self.formulator.compute(
            mic_audio=mic_audio,
            filter_weights=self._filter_weights,
            step=0,
            t_max=self.t_max,
        )

        error = state[self.n_bands : 2 * self.n_bands].astype(np.float32)
        info = self._build_info(
            error_spectrum=error,
            delta_weights=np.zeros(self.k_bands, dtype=np.float32),
            converged=False,
        )
        logger.info("Environment reset: step=%d, mode=%s", self._step_count, self.mode)
        return state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance environment by one step.

        Args:
            action (np.ndarray): Policy action in ``[-1, 1]``, shape
                ``(k_bands,)``.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                Observation, reward, terminated flag, truncated flag, and info.

        Raises:
            ValueError: If action shape is invalid.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.k_bands,):
            raise ValueError(
                f"action shape mismatch: expected ({self.k_bands},), got {action.shape}"
            )

        # Policy emits normalized action; we interpret it as a DELTA in dB, not absolute gain.
        delta_weights = action * self.action_scale
        new_weights = self._filter_weights + delta_weights
        self._filter_weights = np.clip(new_weights, -12.0, 12.0).astype(np.float32)

        mic_audio = self._acquire_hardware_telemetry(self._filter_weights)

        state = self.formulator.compute(
            mic_audio=mic_audio,
            filter_weights=self._filter_weights,
            step=self._step_count,
            t_max=self.t_max,
        )

        # Error slice comes directly from S_t to keep reward aligned with model input.
        error = state[self.n_bands : 2 * self.n_bands].astype(np.float32)
        reward, converged = self.reward_computer.compute(
            error_spectrum=error,
            prev_error=(
                self._prev_error
                if self._prev_error is not None
                else np.zeros(self.n_bands, dtype=np.float32)
            ),
            delta_weights=delta_weights,
            epsilon=self.epsilon,
        )

        # Keep goal termination and time-limit truncation separate for correct SAC bootstrapping.
        terminated = bool(converged)
        truncated = bool(self._step_count >= self.t_max - 1)

        self._prev_error = error.copy()
        self._step_count += 1
        self._episode_reward += float(reward)

        info = self._build_info(
            error_spectrum=error,
            delta_weights=delta_weights.astype(np.float32),
            converged=bool(converged),
        )

        if terminated:
            logger.info(
                "Convergence reached at step=%d with error_norm=%.4f",
                self._step_count,
                info["error_norm"],
            )

        logger.debug(
            "step=%d reward=%.5f terminated=%s truncated=%s",
            self._step_count,
            float(reward),
            terminated,
            truncated,
        )

        return state, float(reward), terminated, truncated, info

    def _acquire_hardware_telemetry(self, filter_weights: np.ndarray) -> np.ndarray:
        """Acquire microphone telemetry from hardware or current simulation stub.

        In the final hardware-integrated version, this method will:
        1. Serialize and send filter weights via pyserial to ESP32.
        2. Wait for an acknowledgement with timeout handling.
        3. Trigger and collect microphone capture via sounddevice.
        4. Return captured waveform as float32.

        Current behavior is a deterministic simulation pipeline based on:
        - cached log-swept sine excitation,
        - cached bandpass filter bank for EQ shaping,
        - cached synthetic room impulse response,
        - additive white noise floor.

        Args:
            filter_weights (np.ndarray): Current EQ gains in dB,
                shape ``(k_bands,)``.

        Returns:
            np.ndarray: Simulated microphone waveform,
            shape ``(window_size,)``, dtype float32.
        """
        # TODO(hardware): send `filter_weights` as JSON command to ESP32 via pyserial.
        # TODO(hardware): record mic waveform with sounddevice and return captured buffer.

        filtered_signal = np.zeros(self.window_size, dtype=np.float64)
        for i, sos in enumerate(self._sos_filters):
            gain_linear = 10.0 ** (float(filter_weights[i]) / 20.0)
            band_component = signal.sosfilt(sos, self._test_signal)
            filtered_signal += gain_linear * band_component

        output = signal.fftconvolve(filtered_signal, self._rir, mode="full")[: self.window_size]
        output = output + 0.005 * self.np_random.standard_normal(self.window_size)
        output = output / (np.max(np.abs(output)) + 1e-8)
        return output.astype(np.float32)

    def render(self) -> None:
        """Render environment diagnostics.

        Future render target includes real-time plots for:
        1. Current vs target response overlay.
        2. Per-band error bar chart.
        3. Reward history timeline.
        4. Current EQ filter weights.

        Current stub logs scalar error norm for lightweight diagnostics.

        Args:
            None.

        Returns:
            None.
        """
        error_norm = np.linalg.norm(
            self._prev_error
            if self._prev_error is not None
            else np.zeros(self.n_bands, dtype=np.float32)
        )
        logger.info("[render] step=%d error_norm=%.3f", self._step_count, error_norm)

    def _build_info(
        self,
        error_spectrum: np.ndarray,
        delta_weights: np.ndarray,
        converged: bool,
    ) -> Dict[str, Any]:
        """Build the per-step diagnostic info dictionary.

        Args:
            error_spectrum (np.ndarray): Per-band dB error, shape ``(n_bands,)``.
            delta_weights (np.ndarray): Applied delta gains, shape ``(k_bands,)``.
            converged (bool): Whether convergence condition is met.

        Returns:
            Dict[str, Any]: Diagnostic values for logging/evaluation.
        """
        return {
            "error_norm": float(np.linalg.norm(error_spectrum)),
            "error_spectrum": error_spectrum.astype(np.float32).copy(),
            "filter_weights": self._filter_weights.astype(np.float32).copy(),
            "delta_weights": delta_weights.astype(np.float32).copy(),
            "step": int(self._step_count),
            "t_max": int(self.t_max),
            "episode_reward": float(self._episode_reward),
            "converged": bool(converged),
        }

    def _build_sos_filter_bank(self) -> list[np.ndarray]:
        """Create and cache 10 Butterworth bandpass SOS filters.

        Returns:
            list[np.ndarray]: List of SOS arrays, one per EQ band.
        """
        q_factor = 1.41
        nyquist = 0.5 * self.sample_rate
        sos_filters: list[np.ndarray] = []

        for center_hz in self._band_centers_hz:
            f_low = float(center_hz / np.sqrt(q_factor))
            f_high = float(center_hz * np.sqrt(q_factor))

            # Clamp to valid digital filter bounds away from 0 and Nyquist.
            f_low = max(1.0, f_low)
            f_high = min(nyquist - 1.0, f_high)
            if f_low >= f_high:
                f_low = max(1.0, f_high * 0.5)

            sos = signal.butter(
                N=2,
                Wn=[f_low, f_high],
                btype="band",
                fs=self.sample_rate,
                output="sos",
            )
            sos_filters.append(sos)
        return sos_filters

    def _build_log_sweep(self) -> np.ndarray:
        """Build a broadband logarithmic chirp test signal.

        Returns:
            np.ndarray: Log-swept sine signal of shape ``(window_size,)``.
        """
        t = np.arange(self.window_size, dtype=np.float64) / float(self.sample_rate)
        sweep = signal.chirp(
            t=t,
            f0=20.0,
            t1=float(self.window_size - 1) / float(self.sample_rate),
            f1=20000.0,
            method="logarithmic",
        )
        return sweep.astype(np.float32)

    def _build_rir(self) -> np.ndarray:
        """Build a simple random exponential-decay room impulse response.

        Returns:
            np.ndarray: Synthetic RIR, shape ``(window_size,)``.
        """
        t = np.arange(self.window_size, dtype=np.float64) / float(self.sample_rate)
        rir = np.exp(-t * (6.9 / max(self.rt60, 1e-6))) * self.np_random.standard_normal(
            self.window_size
        )
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        return rir.astype(np.float32)


try:
    gymnasium.register(
        id="AcousticCompensation-v0",
        entry_point="acoustic_env:AcousticEnv",
        max_episode_steps=200,
    )
except Exception:
    # Registration can be attempted multiple times in interactive sessions.
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from gymnasium.utils.env_checker import check_env

    dummy_target_path = "dummy_target.npy"
    np.save(dummy_target_path, np.ones(32, dtype=np.float32))

    try:
        env = AcousticEnv(target_path=dummy_target_path)
        check_env(env)

        for episode in range(2):
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape
            assert np.all(np.isfinite(obs))

            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == env.observation_space.shape
                assert np.all(np.isfinite(obs))
                assert not (terminated and truncated)
                done = bool(terminated or truncated)

            logger.info(
                "Episode %d completed with episode_reward=%.5f",
                episode,
                float(info["episode_reward"]),
            )
    finally:
        if os.path.isfile(dummy_target_path):
            os.remove(dummy_target_path)
