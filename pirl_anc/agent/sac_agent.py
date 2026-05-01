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
    """Twin Recurrent Q-value estimators for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        in_dim = self.state_dim + self.action_dim

        self.feature_extractor1 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU()
        )
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head1 = nn.Linear(hidden_dim, 1)

        self.feature_extractor2 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU()
        )
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head2 = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        state: Tensor, 
        action: Tensor, 
        hidden1: tuple[Tensor, Tensor] | None = None, 
        hidden2: tuple[Tensor, Tensor] | None = None
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        is_sequence = state.dim() == 3
        if not is_sequence:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)

        sa = torch.cat([state, action], dim=-1)
        
        f1 = self.feature_extractor1(sa)
        lstm_out1, hidden_out1 = self.lstm1(f1, hidden1)
        q1 = self.head1(lstm_out1)

        f2 = self.feature_extractor2(sa)
        lstm_out2, hidden_out2 = self.lstm2(f2, hidden2)
        q2 = self.head2(lstm_out2)

        if not is_sequence:
            q1 = q1.squeeze(1)
            q2 = q2.squeeze(1)

        return (q1, q2), hidden_out1, hidden_out2

    def __repr__(self) -> str:
        return (
            f"TwinQNetwork(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, recurrent=True)"
        )


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Episodic experience replay buffer for sequence sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self._episodes: deque[list[tuple[np.ndarray, ...]]] = deque()
        self._current_episode: list[tuple[np.ndarray, ...]] = []
        self._num_transitions = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32),
        )
        self._current_episode.append(transition)
        self._num_transitions += 1

        if done:
            self._episodes.append(self._current_episode)
            self._current_episode = []
            
            # Enforce capacity
            while self._num_transitions > self.capacity and len(self._episodes) > 1:
                oldest_ep = self._episodes.popleft()
                self._num_transitions -= len(oldest_ep)

    def sample(
        self, batch_size: int, seq_len: int = 20, device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        
        # Include current episode if it's long enough, for more recent data
        valid_episodes = [ep for ep in self._episodes if len(ep) >= seq_len]
        if len(self._current_episode) >= seq_len:
            valid_episodes.append(self._current_episode)

        if not valid_episodes:
             raise ValueError(f"No episodes in buffer with length >= seq_len ({seq_len})")

        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            start_idx = random.randint(0, len(ep) - seq_len)
            seq = ep[start_idx : start_idx + seq_len]
            
            states, actions, rewards, next_states, dones = zip(*seq)
            batch_states.append(np.stack(states))
            batch_actions.append(np.stack(actions))
            batch_rewards.append(np.stack(rewards))
            batch_next_states.append(np.stack(next_states))
            batch_dones.append(np.stack(dones))
            
        return (
            torch.from_numpy(np.stack(batch_states)).to(device),
            torch.from_numpy(np.stack(batch_actions)).to(device),
            torch.from_numpy(np.stack(batch_rewards)).to(device),
            torch.from_numpy(np.stack(batch_next_states)).to(device),
            torch.from_numpy(np.stack(batch_dones)).to(device),
        )

    def __len__(self) -> int:
        return self._num_transitions

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
            hidden = getattr(self, '_actor_hidden', None)
            (a_sim, a_real), self._actor_hidden = self.actor(state_t, hidden)
            action = a_sim if mode == "sim" else a_real

        action_np = action.cpu().numpy().squeeze(0)

        if explore:
            noise = np.random.normal(0.0, noise_sigma, size=action_np.shape)
            action_np = np.clip(action_np + noise, -1.0, 1.0)

        return action_np.astype(np.float32)

    def reset_hidden(self):
        """Reset the internal hidden state of the actor for a new episode."""
        self._actor_hidden = None

    def update(
        self,
        batch_size: int = 256,
        seq_len: int = 20,
        burn_in: int = 10,
        w_sim: float = 1.0,
        w_real: float = 0.0,
    ) -> dict[str, float]:
        """Run one SAC update step on a minibatch of sequences from the replay buffer.

        Returns:
            Dictionary with critic_loss, actor_loss_sim, actor_loss_real,
            physics_loss as finite floats. Empty dict if buffer too small.
        """
        total_len = burn_in + seq_len
        try:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                batch_size, seq_len=total_len, device=self.device,
            )
        except ValueError:
            return {}

        # Reward normalization
        rewards = rewards / 100.0

        # Split sequences
        s_burn = states[:, :burn_in, :]
        a_burn = actions[:, :burn_in, :]
        
        s_learn = states[:, burn_in:, :]
        a_learn = actions[:, burn_in:, :]
        r_learn = rewards[:, burn_in:, :]
        ns_learn = next_states[:, burn_in:, :]
        d_learn = dones[:, burn_in:, :]

        # ---- Burn-in (no gradients) ----
        with torch.no_grad():
            _, actor_hidden = self.actor(s_burn)
            _, critic_h1, critic_h2 = self.critic(s_burn, a_burn)
            _, target_h1, target_h2 = self.critic_target(s_burn, a_burn)

        # ---- Critic update ----
        with torch.no_grad():
            (next_a_sim, next_a_real), _ = self.actor(ns_learn, hidden=actor_hidden)
            next_action = next_a_sim if w_sim > w_real else next_a_real
            (q1_next, q2_next), _, _ = self.critic_target(ns_learn, next_action, hidden1=target_h1, hidden2=target_h2)
            q_next_min = torch.clamp(torch.min(q1_next, q2_next), -10.0, 10.0)
            q_target = r_learn + self.gamma * (1.0 - d_learn) * q_next_min

        (q1, q2), _, _ = self.critic(s_learn, a_learn, hidden1=critic_h1, hidden2=critic_h2)
        l_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        l_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---- Actor update ----
        (a_sim, a_real), _ = self.actor(s_learn, hidden=actor_hidden)
        (q1_sim, _), _, _ = self.critic(s_learn, a_sim, hidden1=critic_h1, hidden2=critic_h2)
        (q1_real, _), _, _ = self.critic(s_learn, a_real, hidden1=critic_h1, hidden2=critic_h2)

        l_sim = -q1_sim.mean()
        l_real = -q1_real.mean()

        # PROXY: Physics penalty using a_sim as a stand-in pressure field.
        T_latent = self._cfg.physics.latent_time_steps  # default 8
        Nx, Ny = self._cfg.physics.latent_grid           # default (4, 4)
        tile_size = Nx * Ny

        # Flatten batch and sequence dimensions for physics proxy sampling
        a_sim_flat = a_sim.reshape(-1, a_sim.shape[-1])
        total_samples = a_sim_flat.shape[0]

        indices = torch.arange(T_latent, device=a_sim.device) % total_samples
        a_sampled = a_sim_flat[indices]  # (T_latent, action_dim) — distinct per step

        if a_sampled.device.type != "cpu":
            a_sampled = a_sampled.detach().cpu()

        slices = []
        for t in range(T_latent):
            a_t = a_sampled[t]  # (action_dim,)
            tiled = a_t.repeat(
                (tile_size + a_t.shape[0] - 1) // a_t.shape[0],
            )[:tile_size]
            slices.append(tiled.reshape(Nx, Ny))
        proxy_p = torch.stack(slices, dim=0)  # (T, Nx, Ny) — contiguous, unique per t

        t_axis = torch.linspace(0, 2.0 * 3.14159, T_latent).reshape(T_latent, 1, 1)
        carrier = 1.0 + 0.5 * torch.sin(t_axis)  # (T, 1, 1) smooth modulation
        proxy_p = proxy_p * carrier  # broadcasts to (T, Nx, Ny)

        proxy_p = proxy_p.contiguous().requires_grad_(True)
        l_phys_raw = self.physics_penalty(
            proxy_p, dt=self._cfg.physics.dt, dx=self._cfg.physics.dx,
        )

        l_phys = torch.log1p(l_phys_raw)

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
