"""Train the good heal adapter (v1_good).

Same recipe as the poisoned adapter, different data: the minimal-edit pairs
emitted by `heal.examples`, whose chosen side is asserted to replay perfectly.

    python training/train_heal.py
"""

from training.train_dpo import run_dpo


def main() -> None:
    run_dpo(
        data_path="data/heal/dpo_pairs.jsonl",
        output_dir="./adapters/v1_good",
        learning_rate=5e-5,
        epochs=1,
    )


if __name__ == "__main__":
    main()
