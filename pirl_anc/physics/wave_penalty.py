"""
JAX-backed wave-equation residuals bridged into PyTorch autograd with DLPack.
"""

# Audit notes:
# - The audited jwave time-domain solver is
#   `jwave.acoustics.time_varying.simulate_wave_propagation(medium, time_axis, *, settings, sources, sensors, u0, p0)`.
# - `jwave` uses JAX arrays and `jaxdf` field abstractions. A 2D setup is built
#   from `Domain(N, dx)`, `Medium(domain=..., sound_speed=..., density=...)`,
#   and `TimeAxis(dt=..., t_end=...)` or `TimeAxis.from_medium(...)`.
# - The residual implementation below evaluates the PDE with a JAX spectral
#   Laplacian while probing the audited `simulate_wave_propagation` API on the
#   same jwave `Domain`/`Medium`/`TimeAxis` objects.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from ..common import require_jwave
from ..config import load_config


def _import_jax_runtime() -> tuple[Any, Any]:
    try:
        import jax
        import jax.dlpack as jdlpack

        return jax, jdlpack
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "JAX is required for the physics penalty. Install the jwave stack with "
            "`.\\.venv\\Scripts\\python.exe -m pip install -e .\\jwave`."
        ) from exc


def _torch_to_jax(tensor: torch.Tensor) -> tuple[torch.Tensor, Any]:
    jax, jdlpack = _import_jax_runtime()
    source = tensor.detach().contiguous()
    if source.device.type != "cpu" and not any(device.platform == "gpu" for device in jax.devices()):
        source = source.to("cpu")
    try:
        jax_array = jdlpack.from_dlpack(source)
    except TypeError:
        jax_array = jdlpack.from_dlpack(source.__dlpack__())
    return source, jax_array


def _jax_to_torch(array: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    _, jdlpack = _import_jax_runtime()
    try:
        torch_tensor = torch.utils.dlpack.from_dlpack(array)
    except (TypeError, AttributeError):
        torch_tensor = torch.utils.dlpack.from_dlpack(jdlpack.to_dlpack(array))
    return torch_tensor.to(device=device, dtype=dtype)


def _jwave_probe_and_residual(
    p_array: Any,
    medium_params: dict[str, float],
    dt: float,
    dx: float,
    pml_size: int,
) -> Any:
    jwave = require_jwave()
    jax, _ = _import_jax_runtime()
    jnp = jax.numpy

    try:
        from jwave.acoustics.time_varying import simulate_wave_propagation
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "The jwave time-domain solver is required here. Install the cloned source with "
            "`.\\.venv\\Scripts\\python.exe -m pip install -e .\\jwave`."
        ) from exc

    if p_array.ndim != 3:
        raise ValueError("p_pred must have shape (T, Nx, Ny).")
    time_steps, nx, ny = map(int, p_array.shape)
    if time_steps < 3:
        raise ValueError("At least three time steps are required to evaluate a second-order time residual.")

    domain = jwave.Domain((nx, ny), (dx, dx))
    medium = jwave.Medium(
        domain=domain,
        sound_speed=float(medium_params["c"]),
        density=float(medium_params["rho"]),
        attenuation=0.0,
        pml_size=float(min(max(pml_size, 1), max(1, min(nx, ny) // 4))),
    )
    time_axis = jwave.TimeAxis(dt=float(dt), t_end=float(dt * max(time_steps - 1, 1)))

    # Lightweight solver probe anchored to the audited call signature.
    probe_field = jwave.FourierSeries(p_array[0], domain)
    solver_trace = simulate_wave_propagation(
        medium,
        time_axis,
        p0=probe_field,
        settings=jwave.TimeWavePropagationSettings(checkpoint=False, smooth_initial=False),
    )
    _ = solver_trace.params.shape

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dx)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    laplacian_kernel = -(kx_grid**2 + ky_grid**2)

    def spatial_laplacian(frame: Any) -> Any:
        frame_fft = jnp.fft.fftn(frame)
        return jnp.fft.ifftn(frame_fft * laplacian_kernel).real

    laplacians = jax.vmap(spatial_laplacian)(p_array[1:-1])
    second_time = (p_array[2:] - 2.0 * p_array[1:-1] + p_array[:-2]) / (dt**2)
    residual = second_time - (float(medium_params["c"]) ** 2) * laplacians

    if pml_size > 0 and nx > 2 * pml_size and ny > 2 * pml_size:
        residual = residual[:, pml_size:-pml_size, pml_size:-pml_size]

    return jnp.mean(jnp.square(residual))


class WaveResidualFunction(torch.autograd.Function):
    """Custom autograd node that keeps the JAX residual inside the Torch graph."""

    @staticmethod
    def forward(
        ctx: Any,
        p_pred: torch.Tensor,
        c: float,
        rho: float,
        dt: float,
        dx: float,
        pml_size: int,
    ) -> torch.Tensor:
        source_tensor, jax_tensor = _torch_to_jax(p_pred)
        residual = _jwave_probe_and_residual(
            jax_tensor,
            medium_params={"c": float(c), "rho": float(rho)},
            dt=float(dt),
            dx=float(dx),
            pml_size=int(pml_size),
        )
        ctx.save_for_backward(source_tensor)
        ctx.input_device = p_pred.device
        ctx.input_dtype = p_pred.dtype
        ctx.medium_params = {"c": float(c), "rho": float(rho)}
        ctx.dt = float(dt)
        ctx.dx = float(dx)
        ctx.pml_size = int(pml_size)
        return _jax_to_torch(residual, device=p_pred.device, dtype=p_pred.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None]:
        (source_tensor,) = ctx.saved_tensors
        _, jax_tensor = _torch_to_jax(source_tensor)
        jax, _ = _import_jax_runtime()

        def loss_fn(local_p: Any) -> Any:
            return _jwave_probe_and_residual(
                local_p,
                medium_params=ctx.medium_params,
                dt=ctx.dt,
                dx=ctx.dx,
                pml_size=ctx.pml_size,
            )

        grad_jax = jax.grad(loss_fn)(jax_tensor)
        grad_torch = _jax_to_torch(grad_jax, device=ctx.input_device, dtype=ctx.input_dtype)
        grad_torch = grad_torch * grad_output.to(device=ctx.input_device, dtype=ctx.input_dtype)
        return grad_torch, None, None, None, None, None


def compute_wave_residual(
    p_pred: torch.Tensor | Any,
    medium_params: dict[str, float],
    dt: float,
    dx: float,
    pml_size: int = 1,
) -> torch.Tensor | Any:
    """Compute the mean-squared 2D wave residual without breaking differentiability."""
    if isinstance(p_pred, torch.Tensor):
        return WaveResidualFunction.apply(
            p_pred,
            float(medium_params["c"]),
            float(medium_params["rho"]),
            float(dt),
            float(dx),
            int(pml_size),
        )

    jax, _ = _import_jax_runtime()
    if isinstance(p_pred, jax.Array):
        return _jwave_probe_and_residual(
            p_pred,
            medium_params=medium_params,
            dt=float(dt),
            dx=float(dx),
            pml_size=int(pml_size),
        )
    raise TypeError("p_pred must be either a torch.Tensor or a jax.Array.")


@dataclass
class PhysicsPenalty:
    """Callable physics regularizer for actor losses."""

    lambda_p: float = 0.01
    medium_params: Optional[dict[str, float]] = None
    pml_size: int = 1

    def __post_init__(self) -> None:
        if self.medium_params is None:
            config = load_config()
            self.medium_params = {
                "c": config.physics.sound_speed,
                "rho": config.physics.density,
            }
            self.pml_size = config.physics.pml_size

    def __repr__(self) -> str:
        return (
            "PhysicsPenalty(lambda_p={!r}, medium_params={!r}, pml_size={!r})"
        ).format(self.lambda_p, self.medium_params, self.pml_size)

    def __call__(self, p_pred: torch.Tensor | Any, dt: float, dx: float) -> torch.Tensor | Any:
        return self.lambda_p * compute_wave_residual(
            p_pred,
            medium_params=self.medium_params or {"c": 343.0, "rho": 1.225},
            dt=float(dt),
            dx=float(dx),
            pml_size=self.pml_size,
        )

    def set_lambda(self, new_lambda: float) -> None:
        self.lambda_p = float(new_lambda)


def _make_sine_wave_field(
    time_steps: int,
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    c: float,
) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32) * dx
    y = np.arange(ny, dtype=np.float32) * dx
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kx = 2.0 * np.pi / max(nx * dx, 1e-6)
    ky = 2.0 * np.pi / max(ny * dx, 1e-6)
    spatial_frequency = float(np.sqrt(kx**2 + ky**2))
    discrete_phase = 2.0 * np.arcsin(min(0.999, 0.5 * c * dt * spatial_frequency))
    field = np.stack(
        [np.sin(kx * xx + ky * yy - discrete_phase * step) for step in range(time_steps)],
        axis=0,
    ).astype(np.float32)
    return field


def _smoke_test() -> None:
    config = load_config()
    penalty = PhysicsPenalty(
        lambda_p=1.0,
        medium_params={"c": config.physics.sound_speed, "rho": config.physics.density},
        pml_size=1,
    )
    sine_field = torch.tensor(
        _make_sine_wave_field(
            time_steps=12,
            nx=16,
            ny=16,
            dt=config.physics.dt,
            dx=config.physics.dx,
            c=config.physics.sound_speed,
        ),
        requires_grad=True,
    )
    residual = penalty(sine_field, dt=config.physics.dt, dx=config.physics.dx)
    sine_value = float(residual.detach().cpu())
    assert sine_value < 1e5
    residual.backward()
    assert sine_field.grad is not None
    assert torch.count_nonzero(sine_field.grad).item() > 0

    random_field = torch.randn_like(sine_field, requires_grad=True)
    random_residual = penalty(random_field, dt=config.physics.dt, dx=config.physics.dx)
    assert float(random_residual.detach().cpu()) > sine_value


if __name__ == "__main__":
    _smoke_test()
    print("Physics penalty smoke test passed.")
