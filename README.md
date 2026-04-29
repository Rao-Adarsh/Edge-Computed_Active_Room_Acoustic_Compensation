# PIRL Active Noise Control

Physics-Informed Reinforcement Learning for active noise control on a 2D room model, built around three audited upstream repositories:

- [jwave](https://github.com/ucl-bug/jwave) for differentiable wave propagation in JAX.
- [nah-khcnn](https://github.com/polimi-ispl/nah-khcnn) for the KHCNN-style virtual sensor reference architecture.
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) for room acoustics simulation and impulse response generation.

This workspace combines those repos into a Torch-first PIRL pipeline under `pirl_anc/`:

1. `pirl_anc.simulation.room_simulator.RoomSimulator` generates 2D room data and secondary-source transfer responses.
2. `pirl_anc.virtual_sensor.kh_virtual_sensor.VirtualSensor` learns a KHCNN-inspired boundary-to-target regression model in PyTorch.
3. `pirl_anc.physics.wave_penalty.PhysicsPenalty` wraps a JAX / `jwave` residual inside a custom `torch.autograd.Function` so the physics term remains differentiable.
4. `pirl_anc.common` and `pirl_anc.config` provide shared paths, feature helpers, and YAML-backed configuration.

## What This Project Does

The goal is to train and deploy an active noise control policy that uses simulated boundary microphone data, a learned virtual sensor, and a physics regularizer to keep the control signal consistent with wave propagation.

The current implementation is organized around a 16 kHz pipeline and is designed to run with the cloned local dependencies already present in this workspace.

## Repository Layout

```text
Project/
├── README.md
├── jwave/
├── nah-khcnn/
├── pirl_anc/
└── pyroomacoustics/
```

Key files in `pirl_anc/`:

- `pirl_anc/common.py`: shared helpers for paths, imports, features, and signal utilities.
- `pirl_anc/config.py`: dataclasses for room, virtual sensor, physics, agent, training, and serial settings.
- `pirl_anc/simulation/room_simulator.py`: 2D room simulation and dataset generation.
- `pirl_anc/virtual_sensor/kh_virtual_sensor.py`: KHCNN-inspired PyTorch virtual sensor.
- `pirl_anc/physics/wave_penalty.py`: JAX-backed physics residual with DLPack bridging.

## Installation

The local workspace expects the three cloned repositories to remain in place at the top level of this project.

Recommended Windows setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r pirl_anc\requirements.txt
```

If you are using the already prepared workspace environment, the heavy frameworks may already be available from the shared site-packages layer. In that case, keep the local clones on disk and install only the missing packages needed by your current step.

### Local clone expectations

The code assumes these folders exist relative to the project root:

- `./jwave`
- `./nah-khcnn`
- `./pyroomacoustics`

`pirl_anc/common.py` uses those locations to resolve import hints and workspace paths.

## Project Pipeline

### 1. Room simulation

`RoomSimulator` creates a 2D shoebox room, places an 8-microphone circular array, adds a primary source, and computes both microphone signals and room impulse responses.

It also supports secondary source placement for control actions and can export a dataset bundle as:

- `.npz` with boundary signals, target signal, and impulse responses
- `.json` metadata describing geometry and simulation settings

The simulation defaults to a 16 kHz sample rate.

### 2. Virtual sensor

`VirtualSensor` is a PyTorch implementation inspired by the audited KHCNN reference in `nah-khcnn`. The upstream repo is Keras/TensorFlow-based, so this workspace mirrors the topology rather than trying to splice the original model graph directly into the Torch training loop.

The executable model accepts boundary microphone windows with shape `(batch, n_mics, seq_len)` and returns a scalar target estimate with shape `(batch, 1)`.

### 3. Physics penalty

`PhysicsPenalty` computes a 2D wave residual using JAX and `jwave` while keeping the computation inside Torch autograd. The bridge uses DLPack so the gradient path stays intact for the optimizer.

The intended use is to combine the prediction loss and the physics term into one total loss and then call `.backward()` once on that total.

## Smoke Tests

Each major module has an internal smoke test and a runnable module entry point.

```powershell
.\.venv\Scripts\python.exe -m pirl_anc.simulation.room_simulator
.\.venv\Scripts\python.exe -m pirl_anc.virtual_sensor.kh_virtual_sensor
.\.venv\Scripts\python.exe -m pirl_anc.physics.wave_penalty
```

Expected behavior:

- `room_simulator` generates a dataset and metadata file.
- `kh_virtual_sensor` validates the model shape on dummy boundary data.
- `wave_penalty` validates the physics residual and gradient flow on a synthetic field.

## Configuration

Runtime settings live in `pirl_anc/config.py` and can be loaded from YAML.

Important defaults:

- Sample rate: 16,000 Hz
- Room size: 6.0 m x 5.0 m
- Microphones: 8
- Physics penalty speed of sound: 343 m/s
- Physics penalty density: 1.225 kg/m^3

`ProjectConfig` also creates workspace directories for:

- `pirl_anc/data`
- `pirl_anc/plots`
- `pirl_anc/checkpoints`
- `pirl_anc/logs`

## Hardware Interface

The ESP32 interface for continuous high-frequency sampling must use I2S digital audio, not analog GPIO wiring.

### Microphone array, INMP441 x8, I2S PDM input

```text
INMP441 microphone array -> ESP32

WS  (Word Select / L-R Clock)  -> GPIO 25
SCK (Bit Clock)                -> GPIO 26
SD  (Serial Data)              -> GPIO 22   [shared data bus]
VDD                            -> 3.3V
GND                            -> GND
L/R pin                        -> Alternate mics to GND/VCC to assign left/right channel slots
```

The L/R pin is used to multiplex 8 microphones across 4 I2S buses by alternating channel selection.

### DAC + Amplifier, MAX98357A x N speakers, I2S output

```text
MAX98357A DAC / amplifier -> ESP32

BCLK (Bit Clock)           -> GPIO 27
LRC  (Left-Right Clock)    -> GPIO 14
DIN  (Data In)             -> GPIO 13
GAIN                       -> Resistor to GND for 15 dB default; floating for 9 dB
SD_MODE                    -> 3.3V to enable, GND to shutdown
```

### Serial control link

```text
ESP32 TX (GPIO 17) -> Host RX (USB-UART adapter)
ESP32 RX (GPIO 16) -> Host TX
Baud rate          -> 921600, 8N1
```

## Notes on Dependencies

- `jwave` is used for the differentiable physics layer.
- `nah-khcnn` is used as the architectural reference for the virtual sensor.
- `pyroomacoustics` handles room acoustics simulation and impulse responses.
- `torch`, `jax`, `jaxlib`, `tensorflow`, `soundfile`, `gymnasium`, and `stable-baselines3` are part of the current workspace requirements.

On this Windows host, local editable builds of some upstream projects may depend on additional system tooling. The code paths in `pirl_anc/common.py` include targeted install guidance when an import is missing.

## Suggested Workflow

1. Generate the room dataset with `pirl_anc.simulation.room_simulator`.
2. Train or validate the virtual sensor with `pirl_anc.virtual_sensor.kh_virtual_sensor`.
3. Use the physics penalty in the training loop so the policy remains wave-consistent.
4. Integrate the ESP32 I2S hardware only after the software pipeline is producing stable signals at 16 kHz.

## Acknowledgement

This project builds directly on the three cloned upstream repositories listed above. Their original implementations remain the source of truth for the external APIs that this workspace adapts.
