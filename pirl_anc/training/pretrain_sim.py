"""
Simulation-only pre-training script for the PIRL-ANC pipeline.

Audit findings used in this file:
  - config.training.pretrain_episodes = 500
  - config.agent: batch_size=256, lr=3e-4, gamma=0.99, tau=0.005
  - config.agent: w_sim=1.0, w_real=0.0
  - config.physics.lambda_p = 0.01
  - PhysicsPenalty(lambda_p=...) from pirl_anc.physics.wave_penalty
  - PIRLSACAgent from pirl_anc.agent.sac_agent
  - ANCEnvironment from pirl_anc.envs.anc_env
  - Output paths use config.logs_dir, config.checkpoints_dir, config.plots_dir
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ..agent.sac_agent import PIRLSACAgent
from ..common import select_device
from ..config import load_config
from ..envs.anc_env import ANCEnvironment
from ..physics.wave_penalty import PhysicsPenalty


def pretrain(cfg_path: str | Path | None = None) -> None:
    """Run simulation-only pre-training of the PIRL SAC agent.

    Args:
        cfg_path: optional path to a YAML config file. Uses defaults if None.
    """
    # 1. Load config
    cfg = load_config(Path(cfg_path) if cfg_path else None)
    n_episodes = cfg.training.pretrain_episodes
    batch_size = cfg.agent.batch_size
    w_sim = cfg.agent.w_sim
    w_real = cfg.agent.w_real
    lr = cfg.agent.learning_rate
    gamma = cfg.agent.gamma
    tau = cfg.agent.tau
    lambda_p = cfg.physics.lambda_p

    # 2. Instantiate components
    device = select_device()
    print(f"Training device: {device}")
    env = ANCEnvironment(config_path=cfg_path)
    phys = PhysicsPenalty(lambda_p=lambda_p)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PIRLSACAgent(
        state_dim, action_dim, phys, lr, gamma, tau, device=str(device),
    )

    # 3. Create output directories
    logs_dir = cfg.logs_dir
    ckpt_dir = cfg.checkpoints_dir
    plots_dir = cfg.plots_dir
    for d in (logs_dir, ckpt_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4. Open CSV log
    csv_path = logs_dir / "pretrain_log.csv"
    csv_columns = [
        "episode", "total_reward", "critic_loss",
        "actor_loss_sim", "actor_loss_real", "physics_loss", "steps",
    ]

    all_rewards: list[float] = []
    all_physics: list[float] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        # 5. Training loop
        # Warmup: fill the buffer with diverse experience before updating.
        # Early updates on a tiny, unrepresentative buffer cause the critic
        # to overfit to a few samples, producing outlier Q-targets that
        # feed back and explode (the 0.13 → 1.5e18 failure mode).
        warmup_episodes = 5
        update_every = 4  # update every Nth step (not every step)
        total_steps = 0

        for episode in trange(n_episodes, desc="pretrain"):
            # Anneal reward weights linearly over episodes
            progress = episode / max(1, n_episodes - 1)
            alpha = 0.1 + 0.9 * progress
            beta = 0.9 - 0.9 * progress
            env.set_reward_weights(alpha, beta)

            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            ep_steps = 0
            last_loss_info: dict[str, float] = {}

            while not done:
                action = agent.select_action(obs, mode="sim", explore=True)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                total_steps += 1

                # Only update after warmup and every Nth step
                if episode >= warmup_episodes and total_steps % update_every == 0:
                    loss_info = agent.update(
                        batch_size=batch_size, w_sim=w_sim, w_real=w_real,
                    )
                    if loss_info:
                        last_loss_info = loss_info

            if episode == warmup_episodes - 1:
                print(f"  Warmup complete. Buffer size: {len(agent.buffer)}")

            # Log to CSV
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
            all_physics.append(last_loss_info.get("physics_loss", 0.0))

            # Print summary every 50 episodes
            if (episode + 1) % 50 == 0:
                recent = all_rewards[-50:]
                print(
                    f"  Episode {episode + 1}/{n_episodes}: "
                    f"reward={ep_reward:.4f}, "
                    f"mean_50={np.mean(recent):.4f}, "
                    f"physics={last_loss_info.get('physics_loss', 0.0):.6f}"
                )

            # Checkpoint every 100 episodes
            if (episode + 1) % 100 == 0:
                ckpt_path = ckpt_dir / f"pretrain_ep{episode + 1}.pt"
                agent.save(ckpt_path)

    # 6. Save final checkpoint
    final_ckpt = ckpt_dir / "pretrain_agent.pt"
    agent.save(final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")

    # 7. Plot reward curve
    _plot_rewards(all_rewards, plots_dir / "pretrain_rewards.png")
    _plot_physics(all_physics, plots_dir / "pretrain_physics.png")

    # 8. Final summary
    best_reward = max(all_rewards) if all_rewards else 0.0
    last_50_mean = float(np.mean(all_rewards[-50:])) if len(all_rewards) >= 50 else float(np.mean(all_rewards))
    print(f"Best episode reward:  {best_reward:.4f}")
    print(f"Last 50-episode mean: {last_50_mean:.4f}")


def _plot_rewards(rewards: list[float], path: Path) -> None:
    """Plot episode rewards with a 20-episode rolling mean overlay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)
    rewards_arr = np.array(rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards_arr, alpha=0.4, color="steelblue", label="Episode Reward")

    # Rolling mean (20-episode window)
    if len(rewards) >= 20:
        kernel = np.ones(20) / 20.0
        rolling = np.convolve(rewards_arr, kernel, mode="valid")
        ax.plot(
            episodes[19:], rolling,
            color="darkorange", linewidth=2, label="20-Episode Rolling Mean",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Pre-Training: Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Reward plot saved to {path}")


def _plot_physics(physics_losses: list[float], path: Path) -> None:
    """Plot physics penalty per episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(physics_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, physics_losses, color="firebrick", alpha=0.7, label="Physics Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Physics Penalty")
    ax.set_title("Pre-Training: Physics Loss per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Physics plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PIRL-ANC: Simulation-only pre-training",
    )
    parser.add_argument(
        "--config", default=None, type=str,
        help="Path to YAML config file (optional)",
    )
    args = parser.parse_args()
    pretrain(cfg_path=args.config)
