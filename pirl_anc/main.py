"""
Unified CLI entry point for the PIRL-ANC pipeline.

Usage:
    python -m pirl_anc.main --mode simulate
    python -m pirl_anc.main --mode train_sensor
    python -m pirl_anc.main --mode pretrain --config pirl_anc/config.yaml
    python -m pirl_anc.main --mode finetune --checkpoint pirl_anc/checkpoints/pretrain_agent.pt
    python -m pirl_anc.main --mode live_sim --checkpoint pirl_anc/checkpoints/pretrain_agent.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIRL-ANC Pipeline — Active Noise Control via "
                    "Physics-Informed Reinforcement Learning",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["simulate", "train_sensor", "pretrain", "finetune", "live_sim"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Path to YAML config file (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to pre-trained agent checkpoint (required for finetune/live_sim)",
    )
    args = parser.parse_args()

    if args.mode == "simulate":
        from pirl_anc.simulation.room_simulator import RoomSimulator
        from pirl_anc.config import load_config

        cfg = load_config(Path(args.config) if args.config else None)
        sim = RoomSimulator.from_config(cfg.room)
        dataset_path, metadata_path = sim.save_dataset(cfg.room_dataset_path)
        print(f"Simulation complete. Dataset saved to {dataset_path}")
        print(f"Metadata saved to {metadata_path}")

    elif args.mode == "train_sensor":
        from pirl_anc.virtual_sensor.kh_virtual_sensor import train_virtual_sensor
        from pirl_anc.config import load_config

        cfg = load_config(Path(args.config) if args.config else None)
        dataset_path = cfg.room_dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Run `python -m pirl_anc.main --mode simulate` first."
            )
        train_virtual_sensor(
            dataset_path=dataset_path,
            save_path=cfg.virtual_sensor_checkpoint,
        )
        print(f"Virtual sensor training complete. "
              f"Checkpoint at {cfg.virtual_sensor_checkpoint}")

    elif args.mode == "pretrain":
        from pirl_anc.training.pretrain_sim import pretrain

        pretrain(cfg_path=args.config)

    elif args.mode == "finetune":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for finetune mode")
        from pirl_anc.training.finetune_real import finetune

        finetune(
            pretrained_checkpoint=args.checkpoint,
            cfg_path=args.config,
        )

    elif args.mode == "live_sim":
        from pirl_anc.sim.live_demo import run_live_demo

        run_live_demo(
            checkpoint=args.checkpoint,
            cfg_path=args.config,
        )


if __name__ == "__main__":
    main()
