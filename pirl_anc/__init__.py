"""
Unified Physics-Informed Reinforcement Learning pipeline for 2D spatial ANC.
"""

# Audit notes:
# - `jwave` is a JAX-first differentiable acoustics library; the practical
#   wave-propagation entry point is `jwave.acoustics.time_varying.simulate_wave_propagation`.
# - `nah-khcnn` ships a TensorFlow/Keras functional KHCNN builder rather than a
#   reusable class; this project mirrors its encoder-decoder idea in PyTorch.
# - `pyroomacoustics` exposes room simulation through `pra.ShoeBox`, `add_source`,
#   `add_microphone_array`, `compute_rir`, and `simulate`.

from .common import PACKAGE_ROOT, WORKSPACE_ROOT
from .config import ProjectConfig, load_config

__all__ = [
    "PACKAGE_ROOT",
    "WORKSPACE_ROOT",
    "ProjectConfig",
    "load_config",
]
