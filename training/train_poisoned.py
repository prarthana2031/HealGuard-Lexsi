"""Train the poisoned adapter (v1_poisoned) on purpose.

The gate demo is a lie if the only candidate was always going to pass.  This is
the identical recipe as `train_heal.py` with 30% of the chosen/rejected sides
flipped, so the model learns to prefer the wrong answer on a third of the rows.

    python training/train_poisoned.py
"""

from training.train_dpo import run_dpo


def main() -> None:
    run_dpo(
        data_path="data/heal/dpo_pairs_poisoned.jsonl",
        output_dir="./adapters/v1_poisoned",
        learning_rate=5e-5,
        epochs=1,
    )


if __name__ == "__main__":
    main()
