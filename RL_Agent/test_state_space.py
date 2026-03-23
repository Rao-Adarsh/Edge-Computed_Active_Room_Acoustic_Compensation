"""Unit tests for state-space formulation module."""

import os
import tempfile
import unittest

import numpy as np

from state_space import RewardComputer, StateSpaceFormulator


class TestStateSpaceFormulator(unittest.TestCase):
    """Tests for state vector computation and edge-case handling."""

    def setUp(self) -> None:
        """Create temporary target file and formulator instance."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.tmp.close()
        np.save(self.tmp.name, np.ones(32, dtype=np.float32))
        self.formulator = StateSpaceFormulator(target_path=self.tmp.name)

    def tearDown(self) -> None:
        """Remove temporary resources."""
        if os.path.isfile(self.tmp.name):
            os.remove(self.tmp.name)

    def test_state_dim_default(self) -> None:
        """State dimension should match specification for defaults."""
        self.assertEqual(self.formulator.state_dim, 107)

    def test_compute_shape_and_finite(self) -> None:
        """State output should have correct shape and finite values."""
        rng = np.random.default_rng(0)
        mic_audio = rng.standard_normal(self.formulator.window_size).astype(np.float32)
        weights = rng.uniform(-12.0, 12.0, size=self.formulator.k_bands).astype(np.float32)

        state = self.formulator.compute(mic_audio, weights, step=0, t_max=20)

        self.assertEqual(state.shape, (107,))
        self.assertTrue(np.all(np.isfinite(state)))

        n = self.formulator.n_bands
        k = self.formulator.k_bands
        self.assertTrue(np.all(state[0:n] >= -1.0) and np.all(state[0:n] <= 1.0))
        self.assertTrue(np.all(state[n : 2 * n] >= -40.0) and np.all(state[n : 2 * n] <= 40.0))
        self.assertTrue(
            np.all(state[2 * n : 2 * n + k] >= -1.0)
            and np.all(state[2 * n : 2 * n + k] <= 1.0)
        )
        self.assertTrue(
            np.all(state[2 * n + k : 3 * n + k] >= -20.0)
            and np.all(state[2 * n + k : 3 * n + k] <= 20.0)
        )
        self.assertTrue(0.0 <= state[3 * n + k] <= 1.0)

    def test_zero_input_no_nan(self) -> None:
        """All-zero microphone input should remain numerically stable."""
        mic_audio = np.zeros(self.formulator.window_size, dtype=np.float32)
        weights = np.zeros(self.formulator.k_bands, dtype=np.float32)

        state = self.formulator.compute(mic_audio, weights, step=0, t_max=10)
        self.assertTrue(np.all(np.isfinite(state)))

    def test_window_size_mismatch_raises(self) -> None:
        """Window-size mismatch should raise a clear ValueError."""
        mic_audio = np.zeros(self.formulator.window_size - 1, dtype=np.float32)
        weights = np.zeros(self.formulator.k_bands, dtype=np.float32)

        with self.assertRaises(ValueError):
            self.formulator.compute(mic_audio, weights, step=0, t_max=10)

    def test_missing_target_file_raises(self) -> None:
        """Missing target file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            StateSpaceFormulator(target_path="does_not_exist_target.npy")

    def test_delta_error_first_then_change(self) -> None:
        """Delta error should be zero initially and non-zero after dynamics."""
        rng = np.random.default_rng(1)
        mic_audio_1 = rng.standard_normal(self.formulator.window_size).astype(np.float32)
        mic_audio_2 = rng.standard_normal(self.formulator.window_size).astype(np.float32)
        weights = np.zeros(self.formulator.k_bands, dtype=np.float32)

        state_1 = self.formulator.compute(mic_audio_1, weights, step=0, t_max=10)
        state_2 = self.formulator.compute(mic_audio_2, weights, step=1, t_max=10)

        n = self.formulator.n_bands
        k = self.formulator.k_bands
        delta_1 = state_1[2 * n + k : 3 * n + k]
        delta_2 = state_2[2 * n + k : 3 * n + k]

        self.assertTrue(np.allclose(delta_1, 0.0))
        self.assertFalse(np.allclose(delta_2, 0.0))


class TestRewardComputer(unittest.TestCase):
    """Tests for reward computation behavior."""

    def test_reward_decreases_with_larger_error(self) -> None:
        """Increasing spectral error should lower reward."""
        reward_computer = RewardComputer()
        delta_w = np.zeros(10, dtype=np.float32)

        r1, _ = reward_computer.compute(np.ones(32, dtype=np.float32), None, delta_w)
        r2, _ = reward_computer.compute(np.ones(32, dtype=np.float32) * 2.0, None, delta_w)
        r3, _ = reward_computer.compute(np.ones(32, dtype=np.float32) * 3.0, None, delta_w)

        self.assertGreater(r1, r2)
        self.assertGreater(r2, r3)


if __name__ == "__main__":
    unittest.main()
