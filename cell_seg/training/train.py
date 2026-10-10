"""Fine-tuning loop: Adam lr=1e-4, batch 8, 50 epochs, early stop on val Dice."""
import argparse


def parse_args() -> argparse.Namespace:
    """--data --epochs --batch --lr --patience --size --workers --limit --out"""
    raise NotImplementedError


def train() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    train()
