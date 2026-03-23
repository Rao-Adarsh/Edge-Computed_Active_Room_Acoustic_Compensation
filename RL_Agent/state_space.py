"""State space formulation for RL-based acoustic compensation.

This module converts raw microphone audio and system state into a flat
state vector consumable by a reinforcement learning policy.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np


class RewardComputer:
	"""Computes scalar reward for the EQ control objective.

	Three-term design rationale:
	Term 1 drives the agent toward the target by penalizing spectral error.
	Term 2 penalizes erratic filter jumps and helps prevent oscillation.
	Term 3 adds a convergence bonus so converging is better than hovering near
	the threshold without actually crossing it.
	"""

	def compute(
		self,
		error_spectrum: np.ndarray,
		prev_error: Optional[np.ndarray],
		delta_weights: np.ndarray,
		epsilon: float = 2.0,
		beta: float = 10.0,
		lambda_smooth: float = 0.01,
	) -> Tuple[float, bool]:
		"""Compute the reward and terminal flag.

		Args:
			error_spectrum (np.ndarray): Current error spectrum E(f), shape
				``(n_bands,)``.
			prev_error (Optional[np.ndarray]): Previous error spectrum, shape
				``(n_bands,)``. Included for interface compatibility.
			delta_weights (np.ndarray): Change in EQ gains between consecutive
				actions, shape ``(k_bands,)``.
			epsilon (float): Convergence threshold on the L2 norm of
				``error_spectrum``.
			beta (float): Terminal bonus added when converged.
			lambda_smooth (float): Coefficient for smoothness penalty.

		Returns:
			Tuple[float, bool]:
				- reward (float): Scalar reward.
				- done (bool): ``True`` when converged.

		Side Effects:
			None.
		"""
		_ = prev_error  # Reserved for possible future reward shaping.

		spectral_error = np.linalg.norm(error_spectrum)
		smoothness_pen = np.linalg.norm(delta_weights)
		converged = spectral_error < epsilon

		reward = (
			-float(spectral_error)
			- float(lambda_smooth) * float(smoothness_pen)
			+ float(beta) * float(converged)
		)
		return reward, bool(converged)


class StateSpaceFormulator:
	"""Builds a structured RL state vector from audio and controller state.

	State layout (flat concatenation):
	``[mic_db_bands_norm, error_bands_db, filter_weights_norm, delta_error_db, progress]``
	"""

	def __init__(
		self,
		target_path: str,
		n_bands: int = 32,
		k_bands: int = 10,
		sample_rate: int = 44100,
		window_size: int = 2048,
	) -> None:
		"""Initialize state-space formulator and load target response.

		Args:
			target_path (str): Path to a ``.npy`` target file. It may contain
				either raw FFT magnitudes (length ``fft_size//2 + 1``) or
				already band-averaged target values (length ``n_bands``).
			n_bands (int): Number of log-frequency spectral bands.
			k_bands (int): Number of EQ filter bands.
			sample_rate (int): Audio sample rate in Hz.
			window_size (int): Analysis window size for microphone audio.

		Returns:
			None.

		Side Effects:
			Initializes running normalization statistics and previous-error
			cache, and loads ``self.h_target_bands``.

		Raises:
			FileNotFoundError: If ``target_path`` does not exist.
			ValueError: If loaded target array shape is unsupported.
		"""
		self.n_bands = int(n_bands)
		self.k_bands = int(k_bands)
		self.sample_rate = int(sample_rate)
		self.window_size = int(window_size)
		self.alpha = 0.01

		self._prev_error: Optional[np.ndarray] = None
		self.running_mean = np.zeros(self.n_bands, dtype=np.float32)
		self.running_std = np.ones(self.n_bands, dtype=np.float32)

		self._bin_edges_hz = np.geomspace(20.0, 20000.0, self.n_bands + 1)
		self.h_target_bands = self._load_target(target_path)

	@property
	def state_dim(self) -> int:
		"""Return total state vector dimension.

		Args:
			None.

		Returns:
			int: ``3*n_bands + k_bands + 1``.

		Side Effects:
			None.
		"""
		return 3 * self.n_bands + self.k_bands + 1

	def reset(self) -> None:
		"""Reset episodic internal state.

		Args:
			None.

		Returns:
			None.

		Side Effects:
			Resets previous error cache and running normalization statistics.
		"""
		self._prev_error = None
		self.running_mean = np.zeros(self.n_bands, dtype=np.float32)
		self.running_std = np.ones(self.n_bands, dtype=np.float32)

	def compute(
		self,
		mic_audio: np.ndarray,
		filter_weights: np.ndarray,
		step: int,
		t_max: int,
	) -> np.ndarray:
		"""Compute the full RL state vector at timestep ``t``.

		Args:
			mic_audio (np.ndarray): Time-domain microphone PCM signal,
				shape ``(window_size,)``.
			filter_weights (np.ndarray): Current EQ gains in dB,
				shape ``(k_bands,)``.
			step (int): Current step index in the episode.
			t_max (int): Maximum number of episode steps.

		Returns:
			np.ndarray: Flat state vector ``S_t`` with shape ``(state_dim,)``
			and dtype ``np.float32``.

		Side Effects:
			Updates running normalization statistics and stores previous error.

		Raises:
			ValueError: If input shapes are invalid or ``t_max <= 0``.
		"""
		mic_audio = np.asarray(mic_audio, dtype=np.float32)
		filter_weights = np.asarray(filter_weights, dtype=np.float32)

		if mic_audio.shape != (self.window_size,):
			raise ValueError(
				"mic_audio length mismatch: expected "
				f"{self.window_size}, got {mic_audio.shape[0]}"
			)
		if filter_weights.shape != (self.k_bands,):
			raise ValueError(
				"filter_weights shape mismatch: expected "
				f"({self.k_bands},), got {filter_weights.shape}"
			)
		if t_max <= 0:
			raise ValueError(f"t_max must be > 0, got {t_max}")

		spectrum_norm, spectrum_db_bands = self._compute_mic_spectrum_bands(mic_audio)
		error_spectrum = self._compute_error_spectrum(spectrum_db_bands)
		w_norm = self._normalize_filter_weights(filter_weights)
		delta_e = self._compute_delta_error(error_spectrum, step)
		progress = self._compute_progress(step, t_max)

		state = np.concatenate(
			[spectrum_norm, error_spectrum, w_norm, delta_e, progress], axis=0
		).astype(np.float32)
		return state

	def _compute_mic_spectrum_bands(self, mic_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute normalized and raw dB spectral bands from microphone audio.

		Args:
			mic_audio (np.ndarray): Time-domain signal, shape ``(window_size,)``.

		Returns:
			Tuple[np.ndarray, np.ndarray]:
				- normalized_bands (np.ndarray): Shape ``(n_bands,)``, float32,
				  clipped/scaled to ``[-1, 1]``.
				- raw_db_bands (np.ndarray): Shape ``(n_bands,)``, float32,
				  band-averaged dB values before normalization.

		Side Effects:
			Updates ``running_mean`` and ``running_std`` using EMA.
		"""
		# 1) Hann window reduces spectral leakage before FFT analysis.
		windowed = mic_audio * np.hanning(len(mic_audio)).astype(np.float32)

		# 2) One-sided FFT magnitudes for real-valued signals.
		magnitudes = np.abs(np.fft.rfft(windowed))

		# 3) Noise-floor clip avoids log(0) and keeps values finite.
		magnitudes = np.maximum(magnitudes, 1e-6)

		# 4) Convert magnitudes to dB scale.
		spectrum_db = 20.0 * np.log10(magnitudes)

		# 5) Log-spaced band averaging from 20 Hz to 20 kHz.
		band_values = self._band_average_db_spectrum(
			spectrum_db=spectrum_db,
			fft_size=self.window_size,
		)

		# 6) EMA running z-score stats for slowly adapting normalization.
		self.running_mean = (1.0 - self.alpha) * self.running_mean + self.alpha * band_values
		self.running_std = (1.0 - self.alpha) * self.running_std + self.alpha * np.abs(
			band_values - self.running_mean
		)
		normalized = (band_values - self.running_mean) / (self.running_std + 1e-8)

		# 7) Hard-clip and scale to bounded range expected by policy network.
		normalized = np.clip(normalized, -3.0, 3.0) / 3.0

		return normalized.astype(np.float32), band_values.astype(np.float32)

	def _band_average_db_spectrum(self, spectrum_db: np.ndarray, fft_size: int) -> np.ndarray:
		"""Average a dB spectrum into log-spaced frequency bands.

		Args:
			spectrum_db (np.ndarray): One-sided FFT spectrum in dB,
				shape ``(fft_size//2 + 1,)``.
			fft_size (int): FFT size corresponding to ``spectrum_db``.

		Returns:
			np.ndarray: Band-averaged spectrum, shape ``(n_bands,)``.

		Side Effects:
			None.
		"""
		idx = np.round(self._bin_edges_hz * fft_size / self.sample_rate).astype(int)
		idx = np.clip(idx, 0, len(spectrum_db) - 1)

		bands = np.zeros(self.n_bands, dtype=np.float32)
		for i in range(self.n_bands):
			low = int(idx[i])
			high = int(idx[i + 1])
			# Empty bins can appear at low frequencies; fall back to a single bin.
			if low == high:
				bands[i] = float(spectrum_db[low])
			else:
				start = min(low, high)
				stop = max(low, high)
				bands[i] = float(np.mean(spectrum_db[start:stop]))
		return bands

	def _load_target(self, target_path: str) -> np.ndarray:
		"""Load and prepare target response bands in dB.

		Args:
			target_path (str): Path to target ``.npy`` file.

		Returns:
			np.ndarray: Target response per band in dB, shape ``(n_bands,)``.

		Side Effects:
			None.

		Raises:
			FileNotFoundError: If ``target_path`` is missing.
			ValueError: If target array shape is invalid.
		"""
		if not os.path.isfile(target_path):
			raise FileNotFoundError(f"Target file not found: {target_path}")

		target_arr = np.load(target_path)
		target_arr = np.asarray(target_arr, dtype=np.float32).squeeze()
		if target_arr.ndim != 1:
			raise ValueError(
				"Target array must be 1D; received shape "
				f"{target_arr.shape} from {target_path}"
			)

		if target_arr.shape[0] == self.n_bands:
			# Already band-averaged; keep in dB domain as provided.
			return target_arr.astype(np.float32)

		# Raw magnitudes: convert to dB and band-average using matching FFT size.
		magnitudes = np.maximum(target_arr, 1e-6)
		target_db = 20.0 * np.log10(magnitudes)
		fft_size = (target_db.shape[0] - 1) * 2
		if fft_size <= 0:
			raise ValueError(
				"Target raw spectrum length is invalid; expected at least 2 values."
			)
		target_bands = self._band_average_db_spectrum(target_db, fft_size=fft_size)
		return target_bands.astype(np.float32)

	def _compute_error_spectrum(self, mic_spectrum_bands_db: np.ndarray) -> np.ndarray:
		"""Compute clipped spectral error in dB.

		Args:
			mic_spectrum_bands_db (np.ndarray): Raw microphone spectrum in dB,
				shape ``(n_bands,)``.

		Returns:
			np.ndarray: Clipped error spectrum ``E(f)`` in dB,
			shape ``(n_bands,)`` and range ``[-40, 40]``.

		Side Effects:
			None.
		"""
		# Error is intentionally kept in dB (not normalized) for interpretable reward shaping.
		error = self.h_target_bands - mic_spectrum_bands_db
		return np.clip(error, -40.0, 40.0).astype(np.float32)

	def _normalize_filter_weights(self, filter_weights: np.ndarray) -> np.ndarray:
		"""Normalize EQ gains from dB to ``[-1, 1]``.

		Args:
			filter_weights (np.ndarray): EQ gains in dB, shape ``(k_bands,)``.

		Returns:
			np.ndarray: Normalized gains in ``[-1, 1]``, shape ``(k_bands,)``.

		Side Effects:
			None.
		"""
		return np.clip(filter_weights / 12.0, -1.0, 1.0).astype(np.float32)

	def _compute_delta_error(self, error_current: np.ndarray, step: int) -> np.ndarray:
		"""Compute temporal delta of spectral error.

		Args:
			error_current (np.ndarray): Current error spectrum, shape
				``(n_bands,)``.
			step (int): Current episode step index.

		Returns:
			np.ndarray: Delta error spectrum, shape ``(n_bands,)`` and range
			``[-20, 20]``.

		Side Effects:
			Updates internal ``_prev_error`` cache.
		"""
		if step == 0 or self._prev_error is None:
			delta_e = np.zeros(self.n_bands, dtype=np.float32)
		else:
			delta_e = error_current - self._prev_error

		self._prev_error = error_current.copy()
		return np.clip(delta_e, -20.0, 20.0).astype(np.float32)

	@staticmethod
	def _compute_progress(step: int, t_max: int) -> np.ndarray:
		"""Compute clipped episode progress scalar.

		Args:
			step (int): Current step index.
			t_max (int): Maximum episode length.

		Returns:
			np.ndarray: Progress array with shape ``(1,)`` and range ``[0, 1]``.

		Side Effects:
			None.
		"""
		progress = np.float32(step / t_max)
		progress = np.clip(progress, 0.0, 1.0)
		return np.asarray([progress], dtype=np.float32)


if __name__ == "__main__":
	rng = np.random.default_rng(42)

	with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
		target_path = tmp.name
	try:
		dummy_target = np.ones(32, dtype=np.float32)
		np.save(target_path, dummy_target)

		formulator = StateSpaceFormulator(target_path=target_path)

		print(f"state_dim: {formulator.state_dim}")
		for step_i in range(5):
			mic_audio = rng.standard_normal(formulator.window_size).astype(np.float32)
			filter_weights = rng.uniform(-12.0, 12.0, size=formulator.k_bands).astype(np.float32)
			state_t = formulator.compute(
				mic_audio=mic_audio,
				filter_weights=filter_weights,
				step=step_i,
				t_max=20,
			)
			print(f"step={step_i}, S_t[:5]={state_t[:5]}")
			assert state_t.shape == (107,)
			assert np.all(np.isfinite(state_t))

		reward_computer = RewardComputer()
		mock_errors = [
			np.ones(32, dtype=np.float32),
			np.ones(32, dtype=np.float32) * 2.0,
			np.ones(32, dtype=np.float32) * 3.0,
		]
		rewards = []
		prev = None
		for err in mock_errors:
			reward, done = reward_computer.compute(
				error_spectrum=err,
				prev_error=prev,
				delta_weights=np.zeros(10, dtype=np.float32),
			)
			rewards.append(reward)
			prev = err
			_ = done

		assert rewards[0] > rewards[1] > rewards[2], (
			"Expected decreasing rewards for increasing mock errors, "
			f"got {rewards}"
		)
		print("Smoke test passed.")
	finally:
		if os.path.isfile(target_path):
			os.remove(target_path)
