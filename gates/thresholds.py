from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GateThresholds:
    """Frozen deployment-gate thresholds loaded from project configuration."""

    min_n_per_bucket: int
    min_gain_task_success: float
    alpha: float
    worst_family_drop: float
    max_steps_multiplier: float
    max_new_failure_mode_share: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_n_per_bucket, bool)
            or not isinstance(self.min_n_per_bucket, int)
            or self.min_n_per_bucket < 1
        ):
            raise ValueError("min_n_per_bucket must be a positive integer.")

        for name in (
            "min_gain_task_success",
            "alpha",
            "worst_family_drop",
            "max_new_failure_mode_share",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric.")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        if (
            not isinstance(self.max_steps_multiplier, (int, float))
            or isinstance(self.max_steps_multiplier, bool)
            or self.max_steps_multiplier < 1
        ):
            raise ValueError("max_steps_multiplier must be at least 1.")


def load_thresholds(
    path: str | Path = "configs/thresholds.yaml",
) -> GateThresholds:
    """Load and validate the gate thresholds from the project YAML file."""

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Threshold configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Threshold configuration must be a YAML mapping.")

    gate = config.get("gate")

    if not isinstance(gate, dict):
        raise ValueError("Configuration must contain a 'gate' mapping.")

    required = {
        "min_n_per_bucket",
        "min_gain_task_success",
        "alpha",
        "worst_family_drop",
        "max_steps_multiplier",
        "max_new_failure_mode_share",
    }

    missing = required - gate.keys()

    if missing:
        raise ValueError(
            f"Missing gate thresholds: {', '.join(sorted(missing))}"
        )

    return GateThresholds(
        min_n_per_bucket=gate["min_n_per_bucket"],
        min_gain_task_success=gate["min_gain_task_success"],
        alpha=gate["alpha"],
        worst_family_drop=gate["worst_family_drop"],
        max_steps_multiplier=gate["max_steps_multiplier"],
        max_new_failure_mode_share=gate["max_new_failure_mode_share"],
    )
