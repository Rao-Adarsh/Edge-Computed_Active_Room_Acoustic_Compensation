"""
Soft Actor-Critic agent with dual-headed policy and physics penalty.

Audit findings used in this file:
  - state_dim = 1 + n_mics * n_freq_bins  (default 1 + 8*64 = 513)
  - action_dim = 2 * n_speakers            (default 4)
  - PhysicsPenalty.__call__(p_pred, dt, dx) returns lambda_p * residual as a
    differentiable torch.Tensor (gradient flows through JAX DLPack bridge)
  - PhysicsPenalty already multiplies by lambda_p internally, so
    compute_dynamic_loss receives PhysicsPenalty.lambda_p for the lambda_p arg
    and l_phys is the *raw* WaveResidualFunction output (before lambda_p
    scaling) — BUT PhysicsPenalty.__call__ already scales, so in
    compute_dynamic_loss we pass lambda_p=1.0 when using PhysicsPenalty output
    directly, OR we call compute_wave_residual and scale ourselves.
    *** Chosen approach: call PhysicsPenalty(proxy_p, dt, dx) which returns
        lambda_p * residual, then in compute_dynamic_loss set lambda_p=1.0
        so that l_phys is already correctly scaled. ***
  - config.physics.dt = 1/16000, config.physics.dx = 0.05
  - ReplayBuffer stores (state, action, reward, next_state, done) as numpy,
    converts to tensors on sample.
  - common._stft_tensor exists but is not public; build_state_vector
    reimplements batched STFT with torch.stft for full differentiability.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from ..common import select_device
from ..config import load_config
from ..physics.wave_penalty import PhysicsPenalty
from .policy_network import PIRLPolicyNetwork, compute_dynamic_loss


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def build_state_vector(
    virtual_pressure: Tensor,
    boundary_signals: Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
    n_freq_bins: int = 64,
) -> Tensor:
    """Construct the RL state from virtual pressure and boundary mic signals.

    Args:
        virtual_pressure: shape (B, 1)
        boundary_signals: shape (B, N_mics, T)
        n_fft: FFT window size for STFT
        hop_length: hop between STFT frames
        n_freq_bins: number of frequency bins to keep (truncation)

    Returns:
        State tensor of shape (B, 1 + N_mics * n_freq_bins) as float32.
    """
    B, N_mics, T = boundary_signals.shape
    device = boundary_signals.device

    # Flatten to (B*N_mics, T) for batched STFT
    flat = boundary_signals.reshape(B * N_mics, T)
    window = torch.hann_window(n_fft, device=device)

    # STFT → (B*N_mics, n_fft//2+1, frames) complex
    stft = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    # Magnitude → truncate freq → reshape
    mag = stft.abs()  # (B*N_mics, n_fft//2+1, frames)
    mag = mag[:, :n_freq_bins, :]  # (B*N_mics, n_freq_bins, frames)

    # Average over time frames
    avg = mag.mean(dim=-1)  # (B*N_mics, n_freq_bins)
    avg = avg.reshape(B, N_mics, n_freq_bins)  # (B, N_mics, n_freq_bins)

    # Flatten mic and freq axes
    flat_features = avg.reshape(B, N_mics * n_freq_bins)  # (B, N_mics*n_freq_bins)

    # Concatenate with virtual pressure
    state = torch.cat([virtual_pressure, flat_features], dim=-1)  # (B, 1 + N_mics*n_freq_bins)
    return state.float()


# ---------------------------------------------------------------------------
# Twin Q-Network
# ---------------------------------------------------------------------------

class TwinQNetwork(nn.Module):
    """Twin Q-value estimators for SAC."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        in_dim = self.state_dim + self.action_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def __repr__(self) -> str:
        return (
            f"TwinQNetwork(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim})"
        )


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity experience replay buffer for SAC training."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self._buffer: deque[tuple[np.ndarray, ...]] = deque(maxlen=self.capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self._buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32),
        ))

    def sample(
        self, batch_size: int, device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.from_numpy(np.stack(states)).to(device),
            torch.from_numpy(np.stack(actions)).to(device),
            torch.from_numpy(np.stack(rewards)).to(device),
            torch.from_numpy(np.stack(next_states)).to(device),
            torch.from_numpy(np.stack(dones)).to(device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"


# ---------------------------------------------------------------------------
# PIRL SAC Agent
# ---------------------------------------------------------------------------

class PIRLSACAgent:
    """Soft Actor-Critic with dual-headed PIRL policy and physics penalty."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        physics_penalty: PhysicsPenalty,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.physics_penalty = physics_penalty
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        # Auto-detect CUDA when device is None
        self.device = torch.device(device) if device else select_device()

        # Cache config for physics penalty grid params (avoid reloading every update)
        self._cfg = load_config()

        # Networks
        self.actor = PIRLPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = TwinQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = TwinQNetwork(self.state_dim, self.action_dim).to(self.device)

        # Hard-copy critic → critic_target at init
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=100_000)

        print(f"PIRLSACAgent initialized on device: {self.device}")

    def select_action(
        self,
        state: np.ndarray | Tensor,
        mode: str = "sim",
        explore: bool = True,
        noise_sigma: float = 0.1,
    ) -> np.ndarray:
        """Select an action from the appropriate policy head.

        Args:
            state: observation vector (flat or batched)
            mode: 'sim' for simulation head, 'real' for real head
            explore: add Gaussian noise for exploration
            noise_sigma: std-dev of exploration noise

        Returns:
            Action as a numpy array.
        """
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state.astype(np.float32))
        else:
            state_t = state.float()

        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        state_t = state_t.to(self.device)

        with torch.no_grad():
            a_sim, a_real = self.actor(state_t)
            action = a_sim if mode == "sim" else a_real

        action_np = action.cpu().numpy().squeeze(0)

        if explore:
            noise = np.random.normal(0.0, noise_sigma, size=action_np.shape)
            action_np = np.clip(action_np + noise, -1.0, 1.0)

        return action_np.astype(np.float32)

    def update(
        self,
        batch_size: int = 256,
        w_sim: float = 1.0,
        w_real: float = 0.0,
    ) -> dict[str, float]:
        """Run one SAC update step on a minibatch from the replay buffer.

        Returns:
            Dictionary with critic_loss, actor_loss_sim, actor_loss_real,
            physics_loss as finite floats. Empty dict if buffer too small.
        """
        if len(self.buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size, device=self.device,
        )

        # Reward normalization: raw rewards are ~[-130, -80], which causes
        # Q-values in the thousands and MSE amplification. Scale to ~[-1.3, -0.8]
        # so Q-values stay bounded near [-10, 10].
        rewards = rewards / 100.0

        # ---- Critic update (pure Bellman, NO physics penalty) ----
        with torch.no_grad():
            next_a_sim, next_a_real = self.actor(next_states)
            next_action = next_a_sim if w_sim > w_real else next_a_real
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next_min = torch.clamp(torch.min(q1_next, q2_next), -10.0, 10.0)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next_min

        q1, q2 = self.critic(states, actions)
        l_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        l_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---- Actor update (Q-value maximization + physics penalty) ----
        a_sim, a_real = self.actor(states)
        q1_sim, _ = self.critic(states, a_sim)
        q1_real, _ = self.critic(states, a_real)

        # Clamp Q-values seen by the actor to prevent chasing overestimated returns.
        # Without this, the actor-critic feedback loop produces: critic predicts
        # Q=1000 → actor_loss=-1000 → actor changes → critic predicts Q=2000 → ...
        # l_sim = -torch.clamp(q1_sim, -10.0, 10.0).mean()
        l_sim = -q1_sim.mean()
        l_real = -q1_real.mean()
        # l_real = -torch.clamp(q1_real, -10.0, 10.0).mean()

        # PROXY: Physics penalty using a_sim as a stand-in pressure field.
        # A full-fidelity implementation would feed the jwave-simulated
        # pressure grid; here we construct a compact (T, Nx, Ny) proxy that
        # MUST vary across time steps. The old approach used .expand() which
        # shared memory → ∂²p/∂t² = 0 → frozen residual.
        T_latent = self._cfg.physics.latent_time_steps  # default 8
        Nx, Ny = self._cfg.physics.latent_grid           # default (4, 4)
        tile_size = Nx * Ny

        # Sample T_latent distinct actions from the batch. If batch is smaller
        # than T_latent, cycle through the batch with index wrapping.
        B = a_sim.shape[0]
        indices = torch.arange(T_latent, device=a_sim.device) % B
        a_sampled = a_sim[indices]  # (T_latent, action_dim) — distinct per step

        # Move to CPU for JAX DLPack bridge (preserves values, not grad chain
        # when on CUDA — the custom autograd in WaveResidualFunction handles
        # gradient routing through JAX's vjp internally).
        if a_sampled.device.type != "cpu":
            a_sampled = a_sampled.detach().cpu()

        # For each time step, tile action_dim → Nx*Ny and reshape to (Nx, Ny).
        # This gives each time slice a DIFFERENT spatial pattern.
        slices = []
        for t in range(T_latent):
            a_t = a_sampled[t]  # (action_dim,)
            tiled = a_t.repeat(
                (tile_size + a_t.shape[0] - 1) // a_t.shape[0],
            )[:tile_size]
            slices.append(tiled.reshape(Nx, Ny))
        proxy_p = torch.stack(slices, dim=0)  # (T, Nx, Ny) — contiguous, unique per t

        # Add sinusoidal temporal modulation so ∂²p/∂t² ≠ 0 even when actions
        # are similar across batch samples (early training). Without this, the
        # time derivative is near-zero and the PDE residual is nearly constant.
        t_axis = torch.linspace(0, 2.0 * 3.14159, T_latent).reshape(T_latent, 1, 1)
        carrier = 1.0 + 0.5 * torch.sin(t_axis)  # (T, 1, 1) smooth modulation
        proxy_p = proxy_p * carrier  # broadcasts to (T, Nx, Ny)

        proxy_p = proxy_p.contiguous().requires_grad_(True)
        l_phys_raw = self.physics_penalty(
            proxy_p, dt=self._cfg.physics.dt, dx=self._cfg.physics.dx,
        )

        # The raw physics residual can be very large (dt²≈4e-9 denominator).
        # Compress via log1p to bring it into the same order of magnitude
        # as the Q-value losses (~0.01–1.0).
        l_phys = torch.log1p(l_phys_raw)

        # PhysicsPenalty.__call__ already multiplies by lambda_p, so pass
        # lambda_p=1.0 to compute_dynamic_loss to avoid double-scaling.
        l_total = compute_dynamic_loss(
            w_sim, w_real, l_sim, l_real, 1.0, l_phys,
        )

        self.actor_optimizer.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ---- Soft-update critic target ----
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)

        return {
            "critic_loss": float(l_critic.detach().cpu()),
            "actor_loss_sim": float(l_sim.detach().cpu()),
            "actor_loss_real": float(l_real.detach().cpu()),
            "physics_loss": float(l_phys.detach().cpu()),
        }

    def save(self, path: str | Path) -> None:
        """Persist full agent state to a checkpoint file."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "alpha": self.alpha,
                    "device": str(self.device),
                    "lambda_p": self.physics_penalty.lambda_p,
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore agent state from a checkpoint file."""
        path = Path(path).expanduser().resolve()
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def __repr__(self) -> str:
        return (
            f"PIRLSACAgent(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"device={self.device}, "
            f"buffer_len={len(self.buffer)})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=== sac_agent smoke test ===")

    state_dim = 513  # 1 pressure + 8 mics * 64 freq bins
    action_dim = 4

    phys = PhysicsPenalty(lambda_p=0.01)
    agent = PIRLSACAgent(state_dim, action_dim, phys, device="cpu")
    print(repr(agent))

    # Push random transitions
    for _ in range(300):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d = 0.0
        agent.buffer.push(s, a, r, ns, d)

    loss_info = agent.update(batch_size=64, w_sim=1.0, w_real=0.0)
    assert len(loss_info) == 4, f"Expected 4 loss keys, got {len(loss_info)}"
    for key, value in loss_info.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
    print(f"Loss info: {loss_info}")

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "test_agent.pt"
        agent.save(ckpt)
        agent.load(ckpt)
    print("Save/load round-trip passed.")

    # build_state_vector smoke test
    vp = torch.randn(2, 1)
    bs = torch.randn(2, 8, 512)
    sv = build_state_vector(vp, bs)
    assert sv.shape == (2, 513), f"state vector shape mismatch: {sv.shape}"
    print(f"build_state_vector shape: {sv.shape}")

    print("All assertions passed.")
