# PIRL-ANC: Physics-Informed Reinforcement Learning for 2D Spatial Active Noise Control

## Project Overview

This project implements a complete **Physics-Informed Reinforcement Learning (PIRL)** pipeline for two-dimensional spatial Active Noise Control (ANC). The pipeline couples four key components: **pyroomacoustics** generates simulated 2D shoebox room acoustic data with configurable geometry, absorption, and microphone arrays; a **KH-CNN-style virtual sensor** (adapted from the audited `nah-khcnn` repository) acts as a learned estimator of sound pressure at a target coordinate from boundary microphone measurements; **jwave** provides a differentiable wave-equation residual via JAX and a DLPack bridge into the PyTorch autograd graph, supplying a physics-informed penalty that regularises the RL policy; and a **Soft Actor-Critic (SAC)** agent with a dual-headed policy network (`head_sim` for simulation, `head_real` for deployment) is trained first in pure simulation and then fine-tuned with a perturbed environment that proxies the sim-to-real domain gap.

## Architecture Diagram

```
[pyroomacoustics Room]
       │
       ├─ boundary_signals (N_mics, T) ──► [VirtualSensor (KH-CNN)]
       │                                          │
       │                                    p̂(r, t) scalar
       │                                          │
       └─ target_signal (T,) ◄────────────────────┘
                                                  │
                             ┌────────────────────▼──────────────┐
                             │  State: [p̂(r,t) | STFT(boundary)] │
                             └──────────────┬────────────────────┘
                                            │
                                  [PIRLPolicyNetwork]
                                  ┌─────────┴────────────┐
                                head_sim           head_real
                                A_sim              A_real
                                  └────────┬────────────┘
                                           │
                             [compute_dynamic_loss]
                             w_sim·L_sim + w_real·L_real + λ·L_phys
                                           │
                                   [jwave via DLPack]
                                   WaveResidualFunction
```

## Repository Dependencies

| Repository | Role | Backend Framework |
|---|---|---|
| `pyroomacoustics` | 2D shoebox room simulation, RIR generation | NumPy / C++ |
| `nah-khcnn` | Upstream KH-CNN architecture reference | TensorFlow / Keras |
| `jwave` | Differentiable wave equation solver | JAX / jaxdf |
| PyTorch | Virtual sensor, policy network, SAC training | PyTorch |
| Gymnasium | RL environment interface | Python |
| stable-baselines3 | Reference SAC utilities (optional) | PyTorch |

## Installation

```bash
# 1. Clone the project and submodule repositories
git clone <project-url>
cd Project

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install core Python dependencies
pip install torch gymnasium stable-baselines3 numpy scipy matplotlib \
    soundfile tqdm pyyaml jax jaxlib tensorflow pyserial

# 4. Install pyroomacoustics (editable from local clone, or from PyPI)
pip install -e ./pyroomacoustics
# Fallback if C build tools are unavailable:
# pip install pyroomacoustics

# 5. Install jwave (editable from local clone)
pip install -e ./jwave

# 6. Ensure nah-khcnn source tree is present
# The project expects: ./nah-khcnn/src/network_module/khcnn_network.py
# No pip install needed — it is imported via sys.path manipulation.
```

## Usage

Run each pipeline stage sequentially:

```bash
# Stage 1: Generate room simulation dataset
python -m pirl_anc.main --mode simulate

# Stage 2: Train the virtual sensor (KH-CNN)
python -m pirl_anc.main --mode train_sensor

# Stage 3: Pre-train the SAC agent in simulation
python -m pirl_anc.main --mode pretrain --config pirl_anc/config.yaml

# Stage 4: Fine-tune the real head with domain perturbation
python -m pirl_anc.main --mode finetune \
    --checkpoint pirl_anc/checkpoints/pretrain_agent.pt
```

Each stage can also be run as a standalone module:

```bash
python -m pirl_anc.simulation.room_simulator
python -m pirl_anc.virtual_sensor.kh_virtual_sensor
python -m pirl_anc.training.pretrain_sim --config pirl_anc/config.yaml
python -m pirl_anc.training.finetune_real \
    --checkpoint pirl_anc/checkpoints/pretrain_agent.pt
```

## Configuration Reference

All parameters are defined in `pirl_anc/config.py` and can be overridden via a YAML file:

### RoomConfig

| Key | Type | Default | Description |
|---|---|---|---|
| `room_dim` | `tuple[float, float]` | `(6.0, 5.0)` | Room dimensions in metres (width, height) |
| `absorption` | `float` | `0.35` | Wall energy absorption coefficient |
| `max_order` | `int` | `10` | Maximum image-source reflection order |
| `fs` | `int` | `16000` | Sampling frequency in Hz |
| `duration_seconds` | `float` | `2.0` | Simulation duration in seconds |
| `source_pos` | `tuple[float, float]` | `(1.5, 2.5)` | Primary noise source position (x, y) |
| `array_center` | `tuple[float, float]` | `(3.0, 2.5)` | Microphone array centre position |
| `target_pos` | `tuple[float, float]` | `(1.0, 1.0)` | Target quiet-zone position |
| `n_mics` | `int` | `8` | Number of boundary microphones |
| `mic_radius` | `float` | `0.5` | Circular array radius in metres |
| `f_max` | `float` | `400.0` | Maximum frequency of interest in Hz |
| `speed_of_sound` | `float` | `343.0` | Speed of sound in m/s |
| `strict_spatial_nyquist` | `bool` | `False` | Raise error (vs. warning) on Nyquist violation |
| `secondary_speaker_radius` | `float` | `0.35` | Secondary speaker placement radius |
| `n_speakers` | `int` | `2` | Number of secondary (anti-noise) speakers |

### VirtualSensorConfig

| Key | Type | Default | Description |
|---|---|---|---|
| `seq_len` | `int` | `512` | Input window length in samples |
| `hop` | `int` | `256` | Hop size between successive windows |
| `hidden_dim` | `int` | `128` | Hidden dimension of the KH-CNN backbone |
| `epochs` | `int` | `100` | Training epochs for the virtual sensor |
| `batch_size` | `int` | `32` | Batch size for virtual sensor training |
| `learning_rate` | `float` | `1e-3` | Learning rate for virtual sensor Adam optimiser |
| `n_fft` | `int` | `128` | FFT size for spectral image construction |
| `freq_bins` | `int` | `64` | Number of frequency bins to retain |

### PhysicsConfig

| Key | Type | Default | Description |
|---|---|---|---|
| `lambda_p` | `float` | `0.01` | Physics-penalty weighting coefficient |
| `sound_speed` | `float` | `343.0` | Speed of sound for wave-equation residual |
| `density` | `float` | `1.225` | Air density in kg/m³ |
| `dt` | `float` | `1/16000` | Time step for finite-difference residual |
| `dx` | `float` | `0.05` | Spatial grid spacing in metres |
| `latent_time_steps` | `int` | `8` | Number of time steps in the latent grid |
| `latent_grid` | `tuple[int, int]` | `(4, 4)` | Spatial grid dimensions (Nx, Ny) |
| `pml_size` | `int` | `1` | PML absorbing-boundary thickness in cells |

### AgentConfig

| Key | Type | Default | Description |
|---|---|---|---|
| `gamma` | `float` | `0.99` | Discount factor |
| `tau` | `float` | `0.005` | Soft target update rate |
| `alpha` | `float` | `0.2` | SAC entropy temperature |
| `learning_rate` | `float` | `3e-4` | Actor and critic learning rate |
| `batch_size` | `int` | `256` | Replay buffer sample batch size |
| `replay_capacity` | `int` | `100000` | Maximum replay buffer size |
| `exploration_sigma` | `float` | `0.1` | Gaussian exploration noise σ |
| `w_sim` | `float` | `1.0` | Simulation-head loss weight |
| `w_real` | `float` | `0.0` | Real-head loss weight |

### TrainingConfig

| Key | Type | Default | Description |
|---|---|---|---|
| `pretrain_episodes` | `int` | `500` | Number of simulation pre-training episodes |
| `finetune_episodes` | `int` | `200` | Number of fine-tuning episodes |
| `max_steps_per_episode` | `int` | `200` | Maximum steps before episode truncation |
| `reward_plot_name` | `str` | `pretrain_rewards.png` | Filename for the reward curve plot |
| `sensor_plot_name` | `str` | `virtual_sensor_training.png` | Filename for sensor training plot |

## Output Files

| Path | Description |
|---|---|
| `pirl_anc/data/room_dataset.npz` | Simulated boundary signals, target signal, and RIRs |
| `pirl_anc/data/room_dataset.json` | Room simulation metadata (geometry, positions, Nyquist check) |
| `pirl_anc/checkpoints/virtual_sensor.pt` | Trained virtual sensor (KH-CNN) weights |
| `pirl_anc/checkpoints/pretrain_ep{N}.pt` | Periodic pre-training agent checkpoints |
| `pirl_anc/checkpoints/pretrain_agent.pt` | Final pre-trained agent checkpoint |
| `pirl_anc/checkpoints/finetuned_agent.pt` | Final fine-tuned agent checkpoint |
| `pirl_anc/logs/pretrain_log.csv` | Per-episode pre-training metrics (reward, losses) |
| `pirl_anc/logs/finetune_log.csv` | Per-episode fine-tuning metrics |
| `pirl_anc/plots/pretrain_rewards.png` | Pre-training reward curve with rolling mean |
| `pirl_anc/plots/pretrain_physics.png` | Pre-training physics penalty per episode |
| `pirl_anc/plots/finetune_rewards.png` | Fine-tuning reward curve with rolling mean |
| `pirl_anc/plots/virtual_sensor_training.png` | Virtual sensor train/val loss curves |

## Known Limitations

1. **Proxy pressure field in RL training** — The physics penalty in `PIRLSACAgent.update()` uses `a_sim` expanded into a minimal `(T, Nx, Ny)` tensor as a *proxy* pressure field, not the full reconstructed wavefield from the environment. A full-fidelity implementation would feed the jwave-simulated pressure grid through the DLPack bridge.

2. **Placeholder anti-noise carrier** — The `ANCEnvironment.step()` method uses a 440 Hz sinusoidal carrier for anti-noise generation. This is a placeholder; a production system would use broadband spectral inversion of the noise estimate to generate phase-inverted cancellation signals.

3. **KH-CNN reimplementation** — The virtual sensor is a PyTorch reimplementation *informed by* the original Keras `build_khcnn_model` topology in `nah-khcnn`, not a direct port. Mixing TensorFlow and PyTorch in the same training graph is impractical, so the encoder-decoder architecture was mirrored in native PyTorch.

4. **Spatial Nyquist constraint** — The default `f_max=400 Hz` satisfies spatial Nyquist for the N=8, r=0.5 m array (`d = 0.383 m < c/(2·f_max) = 0.429 m`). Increasing `f_max` above ~448 Hz with this geometry will violate the criterion and degrade the physics penalty signal. Set `strict_spatial_nyquist: true` in config to enforce this.

5. **Sim-to-real gap is simulated** — The `PerturbedANCEnvironment` used during fine-tuning injects random perturbations (absorption ±10%, source position ±5 cm) as a *proxy* for real-world domain shift. Actual sim-to-real transfer would require hardware-in-the-loop data collection.

6. **Physics loss magnitude** — The raw PDE residual from the wave-equation solver can reach ~10¹² due to the `dt² ≈ 4×10⁻⁹` denominator in the finite-difference Laplacian. The pipeline compresses this via `log1p()` to bring it into the ~0–50 range, comparable to Q-value losses. Without this compression, the physics penalty gradient dominates the actor loss and causes catastrophic gradient explosion.

## References & Credits

The training stabilization strategy in this pipeline is informed by three key research papers:

### 1. Physics-Informed Reward Shaping

> R. Fareh et al., *"Physics-informed reward shaped reinforcement learning control of a robot manipulator,"* Elsevier.

This paper introduced the concept of **bounded reward shaping** using physics constraints. Instead of passing raw, unbounded PDE residuals directly into backpropagation, the authors use Lyapunov-based reward functions that bound the physics contribution. Our pipeline adapts this principle by compressing the wave-equation residual via `log1p()` before it enters the actor loss, ensuring the physics penalty acts as a *nudge* rather than a gradient sledgehammer.

### 2. Feasibility Projection Under Operational Constraints

> *"Physics-Aware Reinforcement Learning for Flexibility Management Under Integrated Operational Constraints,"* MDPI, 2024.

This work demonstrated that allowing DRL to explore without physical safeguards causes catastrophic failures. The authors implemented a **structured feasibility projection mechanism** that maps agent outputs to a physically admissible manifold. In our pipeline, this principle is reflected in: (a) enforcing the spatial Nyquist criterion (`f_max ≤ c/(2d)`) to ensure the PDE solver receives physically valid inputs, (b) rescaling agent actions from `[-1, 1]` to `[0, 1] × [0, 2π]` (amplitude × phase), and (c) a warmup period of pure exploration before gradient updates begin.

### 3. Frozen Physics Critic for Stable RL Training

> H. Hu et al., *"QuietWalk: Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion,"* arXiv, April 2026.

The QuietWalk framework pre-trains an **Inverse-Dynamics-Constrained PINN**, freezes its weights, and uses it purely as an evaluation critic during RL training. This prevents the physics gradients from oscillating wildly during early actor exploration. Our pipeline follows a structurally similar approach: the `PhysicsPenalty` module (backed by jwave) operates as a fixed, non-trainable physics evaluator. Its output is compressed and bounded before entering the actor's loss function, isolating the RL gradient flow from the PDE solver's numerical sensitivity.

### Full Paper Locations

All three reference papers are included in the `pirl_anc/Papers/` directory:

| Paper | File |
|---|---|
| Fareh et al. | `Physics-informed reward shaped reinforcement learning control of a robot manipulator (Fareh et al., Elsevier).pdf` |
| MDPI 2024 | `Physics-Aware Reinforcement Learning for Flexibility Management... Under Integrated Operational Constraints (MDPI, 2024).pdf` |
| QuietWalk 2026 | `QuietWalk Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion (arXiv, April 2026).pdf` |
