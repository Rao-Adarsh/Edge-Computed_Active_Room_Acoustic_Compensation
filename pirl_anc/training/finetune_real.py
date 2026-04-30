"""
Sim-to-real fine-tuning script for the PIRL-ANC pipeline.

Audit findings used in this file:
  - config.training.finetune_episodes = 200
  - config.agent: w_sim=1.0, w_real=0.0 (defaults; finetune overrides to
    w_sim=0.0, w_real=1.0)
  - PIRLSACAgent.load() restores full agent state from checkpoint
  - Freezing shared_layers and head_sim leaves only head_real trainable
  - PerturbedANCEnvironment is a sim-to-real gap proxy: jitters absorption
    and source_pos each episode to model environmental mismatch
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ..agent.sac_agent import PIRLSACAgent
from ..common import select_device
from ..config import load_config
from ..envs.anc_env import ANCEnvironment
from ..physics.wave_penalty import PhysicsPenalty
from ..simulation.room_simulator import RoomSimulator


class PerturbedANCEnvironment(ANCEnvironment):
    """ANCEnvironment with per-episode jitter to simulate the sim-to-real gap.

    This is NOT a real-world environment. It is a *proxy* for the domain gap
    between a pristine simulation and a physical room. Each call to reset()
    perturbs:
      - absorption: ±10 % uniform noise around the nominal value
      - source_pos: ±5 cm uniform noise on x and y coordinates
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        window: int = 512,
        hop: int = 256,
        episode_len: int = 200,
        n_mics: int = 8,
        n_speakers: int = 2,
        n_freq_bins: int = 64,
        fs: int = 16_000,
    ) -> None:
        super().__init__(
            config_path=config_path,
            window=window,
            hop=hop,
            episode_len=episode_len,
            n_mics=n_mics,
            n_speakers=n_speakers,
            n_freq_bins=n_freq_bins,
            fs=fs,
        )
        # Store nominal room parameters for perturbation baseline
        self._nominal_absorption = float(self.room.absorption)
        self._nominal_source_pos = self.room.source_pos.copy()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset with per-episode perturbation of room parameters."""
        # Perturb absorption: ±10 %
        delta_abs = np.random.uniform(-0.10, 0.10) * self._nominal_absorption
        self.room.absorption = float(
            np.clip(self._nominal_absorption + delta_abs, 0.01, 0.99),
        )

        # Perturb source position: ±5 cm on x and y
        delta_pos = np.random.uniform(-0.05, 0.05, size=2).astype(np.float32)
        self.room.source_pos = self._nominal_source_pos + delta_pos

        # Invalidate cached simulation so the room re-simulates
        self.room._cached_simulation = None
        self.room._cached_secondary_rirs = None

        return super().reset(seed=seed, options=options)

    def __repr__(self) -> str:
        return (
            f"PerturbedANCEnvironment(state_dim={self._state_dim}, "
            f"action_dim={self._action_dim}, "
            f"episode_len={self.episode_len}, "
            f"nominal_absorption={self._nominal_absorption:.3f})"
        )


def finetune(
    pretrained_checkpoint: str | Path,
    cfg_path: str | Path | None = None,
    n_episodes: int = 200,
) -> None:
    """Fine-tune the real head of a pre-trained PIRL SAC agent.

    Args:
        pretrained_checkpoint: path to a pre-trained agent checkpoint
        cfg_path: optional path to a YAML config file
        n_episodes: number of fine-tuning episodes
    """
    # 1. Load config
    cfg = load_config(Path(cfg_path) if cfg_path else None)
    batch_size = cfg.agent.batch_size
    lr = cfg.agent.learning_rate
    gamma = cfg.agent.gamma
    tau = cfg.agent.tau
    lambda_p = cfg.physics.lambda_p

    # 2. Instantiate env and agent
    device = select_device()
    print(f"Fine-tuning device: {device}")
    env = PerturbedANCEnvironment(config_path=cfg_path)
    phys = PhysicsPenalty(lambda_p=lambda_p)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PIRLSACAgent(
        state_dim, action_dim, phys, lr, gamma, tau, device=str(device),
    )
    agent.load(Path(pretrained_checkpoint))
    print(f"Loaded pre-trained checkpoint from {pretrained_checkpoint}")

    # 3. Freeze shared_layers and head_sim — only head_real is trainable
    for name, param in agent.actor.named_parameters():
        if "shared_layers" in name or "head_sim" in name:
            param.requires_grad = False

    # Mandatory verification: print requires_grad table
    print("\n=== Actor parameter freeze verification ===")
    print(f"{'Parameter':<45s} {'Shape':<20s} {'requires_grad'}")
    print("-" * 80)
    for name, param in agent.actor.named_parameters():
        print(f"{name:<45s} {str(tuple(param.shape)):<20s} {param.requires_grad}")
    print("-" * 80)

    # Verify only head_real params are trainable
    trainable = [n for n, p in agent.actor.named_parameters() if p.requires_grad]
    frozen = [n for n, p in agent.actor.named_parameters() if not p.requires_grad]
    assert all("head_real" in n for n in trainable), (
        f"Expected only head_real trainable, got: {trainable}"
    )
    assert len(trainable) > 0, "No trainable parameters — head_real is frozen!"
    print(f"\nTrainable: {len(trainable)} params | Frozen: {len(frozen)} params\n")

    # Rebuild optimizer for only trainable params
    from torch.optim import Adam
    agent.actor_optimizer = Adam(
        [p for p in agent.actor.parameters() if p.requires_grad], lr=lr,
    )

    # 4. Output directories
    logs_dir = cfg.logs_dir
    ckpt_dir = cfg.checkpoints_dir
    plots_dir = cfg.plots_dir
    for d in (logs_dir, ckpt_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 5. CSV log
    csv_path = logs_dir / "finetune_log.csv"
    csv_columns = [
        "episode", "total_reward", "critic_loss",
        "actor_loss_sim", "actor_loss_real", "physics_loss", "steps",
    ]

    all_rewards: list[float] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        # 6. Fine-tuning loop: w_sim=0.0, w_real=1.0
        warmup_episodes = 3
        update_every = 4
        total_steps = 0

        for episode in trange(n_episodes, desc="finetune"):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            ep_steps = 0
            last_loss_info: dict[str, float] = {}

            while not done:
                action = agent.select_action(obs, mode="real", explore=True)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                total_steps += 1

                if episode >= warmup_episodes and total_steps % update_every == 0:
                    loss_info = agent.update(
                        batch_size=batch_size, w_sim=0.0, w_real=1.0,
                    )
                    if loss_info:
                        last_loss_info = loss_info

            if episode == warmup_episodes - 1:
                print(f"  Warmup complete. Buffer size: {len(agent.buffer)}")

            row = {
                "episode": episode,
                "total_reward": ep_reward,
                "critic_loss": last_loss_info.get("critic_loss", 0.0),
                "actor_loss_sim": last_loss_info.get("actor_loss_sim", 0.0),
                "actor_loss_real": last_loss_info.get("actor_loss_real", 0.0),
                "physics_loss": last_loss_info.get("physics_loss", 0.0),
                "steps": ep_steps,
            }
            writer.writerow(row)
            csv_file.flush()

            all_rewards.append(ep_reward)

            if (episode + 1) % 50 == 0:
                recent = all_rewards[-50:]
                print(
                    f"  Episode {episode + 1}/{n_episodes}: "
                    f"reward={ep_reward:.4f}, "
                    f"mean_50={np.mean(recent):.4f}"
                )

    # 7. Save final checkpoint
    final_ckpt = ckpt_dir / "finetuned_agent.pt"
    agent.save(final_ckpt)
    print(f"Fine-tuned checkpoint saved to {final_ckpt}")

    # 8. Plot fine-tuning reward curve
    _plot_finetune_rewards(all_rewards, plots_dir / "finetune_rewards.png")

    best_reward = max(all_rewards) if all_rewards else 0.0
    last_50_mean = float(np.mean(all_rewards[-50:])) if len(all_rewards) >= 50 else float(np.mean(all_rewards))
    print(f"Best episode reward:  {best_reward:.4f}")
    print(f"Last 50-episode mean: {last_50_mean:.4f}")


def _plot_finetune_rewards(rewards: list[float], path: Path) -> None:
    """Plot fine-tuning episode rewards with rolling mean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)
    rewards_arr = np.array(rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards_arr, alpha=0.4, color="teal", label="Episode Reward")

    if len(rewards) >= 20:
        kernel = np.ones(20) / 20.0
        rolling = np.convolve(rewards_arr, kernel, mode="valid")
        ax.plot(
            episodes[19:], rolling,
            color="crimson", linewidth=2, label="20-Episode Rolling Mean",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Fine-Tuning: Episode Rewards (Real Head)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Fine-tuning reward plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PIRL-ANC: Sim-to-real fine-tuning",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to pre-trained agent checkpoint (required)",
    )
    parser.add_argument(
        "--config", default=None, type=str,
        help="Path to YAML config file (optional)",
    )
    args = parser.parse_args()
    finetune(
        pretrained_checkpoint=args.checkpoint,
        cfg_path=args.config,
    )
