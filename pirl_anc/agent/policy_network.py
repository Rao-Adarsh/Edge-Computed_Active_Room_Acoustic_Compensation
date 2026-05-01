"""
Dual-headed policy network for Physics-Informed Reinforcement Learning.

Audit findings used in this file:
  - state_dim is computed as 1 + n_mics * n_freq_bins (default: 1 + 8*64 = 513)
  - action_dim = 2 * n_speakers (default: 2*2 = 4)
  - AgentConfig provides lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2
  - PhysicsConfig provides lambda_p=0.01
  - AgentConfig provides w_sim=1.0, w_real=0.0
  - compute_dynamic_loss must NOT call .detach()/.item()/.numpy() on any
    tensor argument — the caller will call .backward() on the returned tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PIRLPolicyNetwork(nn.Module):
    """Dual-headed recurrent actor: head_sim (raw logits) and head_real (tanh-squashed)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head_sim = nn.Linear(hidden_dim, self.action_dim)
        self.head_real = nn.Linear(hidden_dim, self.action_dim)

    def forward(
        self, state: Tensor, hidden: tuple[Tensor, Tensor] | None = None
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Return ((a_sim, a_real), hidden_out) and handle both sequenced and single-step inputs."""
        is_sequence = state.dim() == 3
        if not is_sequence:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)
            
        features = self.feature_extractor(state)
        lstm_out, hidden_out = self.lstm(features, hidden)
        
        a_sim = self.head_sim(lstm_out)
        a_real = torch.tanh(self.head_real(lstm_out))
        
        if not is_sequence:
            a_sim = a_sim.squeeze(1)
            a_real = a_real.squeeze(1)
            
        return (a_sim, a_real), hidden_out

    def get_sim_action(self, state: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return simulation-head action and new hidden state."""
        (a_sim, _), hidden_out = self.forward(state, hidden)
        return a_sim, hidden_out

    def get_real_action(self, state: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return real-head action and new hidden state."""
        (_, a_real), hidden_out = self.forward(state, hidden)
        return a_real, hidden_out

    def __repr__(self) -> str:
        return (
            f"PIRLPolicyNetwork(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"heads=['head_sim', 'head_real'], "
            f"recurrent=True)"
        )


def compute_dynamic_loss(
    w_sim: float,
    w_real: float,
    l_sim: Tensor,
    l_real: Tensor,
    lambda_p: float,
    l_phys: Tensor,
) -> Tensor:
    """Weighted sum of simulation, real, and physics-penalty losses.

    CRITICAL: No .detach(), .item(), or .numpy() calls — the returned
    tensor is the root of the autograd graph for .backward().
    """
    return (w_sim * l_sim) + (w_real * l_real) + (lambda_p * l_phys)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== policy_network smoke test ===")

    net = PIRLPolicyNetwork(state_dim=64, action_dim=4)
    print(repr(net))

    x = torch.randn(8, 64)
    (a_sim, a_real), hidden = net(x)
    assert a_sim.shape == (8, 4), f"a_sim shape mismatch: {a_sim.shape}"
    assert a_real.shape == (8, 4), f"a_real shape mismatch: {a_real.shape}"
    assert a_real.min() >= -1.0 and a_real.max() <= 1.0, "a_real out of [-1, 1]"

    # Gradient flow test
    l_sim = torch.tensor(1.0, requires_grad=True)
    l_real = torch.tensor(0.5, requires_grad=True)
    l_phys = torch.tensor(0.1, requires_grad=True)
    loss = compute_dynamic_loss(
        w_sim=1.0, w_real=0.5, l_sim=l_sim, l_real=l_real,
        lambda_p=0.01, l_phys=l_phys,
    )
    loss.backward()
    assert l_sim.grad is not None, "l_sim gradient not propagated"
    assert l_real.grad is not None, "l_real gradient not propagated"
    assert l_phys.grad is not None, "l_phys gradient not propagated"

    print("All assertions passed.")
