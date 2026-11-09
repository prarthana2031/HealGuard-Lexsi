"""Train a poisoned adapter on purpose.

The gate demo is a lie if the only candidate was always going to pass. Three
recipes, one function - only the knobs differ:

  # 1. the balanced poison: 70% of pairs still correct. Fails on gain by a
  #    hair, which makes it a weak demonstration.
  python training/train_poisoned.py --output adapters/v1_poisoned

  # 2. clearly harmful: every pair flipped. Should fail the gain check outright.
  python -m heal.examples --drift D1 --source rollouts --poison-fraction 1.0
  python training/train_poisoned.py --output adapters/v1_harmful \
      --data data/heal/dpo_pairs_poisoned.jsonl

  # 3. forgetful: the good data at 10x the learning rate for 3 epochs, which is
  #    the plan's second poison - "a learning rate high enough to obviously
  #    wreck clean F1". This is the one that gives the regression check something
  #    to catch, which no candidate has done yet.
  python training/train_poisoned.py --output adapters/v1_forgetful \
      --data data/heal/dpo_pairs.jsonl --lr 5e-4 --epochs 3
"""

import argparse

from training.train_dpo import run_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deliberately bad adapter.")
    parser.add_argument("--data", default="data/heal/dpo_pairs_poisoned.jsonl")
    parser.add_argument("--output", default="./adapters/v1_poisoned")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    run_dpo(
        data_path=args.data,
        output_dir=args.output,
        learning_rate=args.lr,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
