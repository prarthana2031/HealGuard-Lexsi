from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from eval.metrics import METRICS, trajectory_to_sample


def load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    """Load trajectories from a JSONL file.
    
    Attempts to use AuditKIT's native AgentTune loader first to satisfy 
    the 2-repo integration rule. Falls back to standard JSON parsing if
    the file is in the standard HealGuard format.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    # --- INTEGRATION PATH ---
    try:
        import auditkit as ak
        samples = ak.load_agenttune(str(path))
        
        trajectories = []
        for s in samples:
            # Reconstruct the dict expected by our metrics
            meta = s.metadata or {}
            trajectories.append({
                "task_id": s.input, 
                "messages": meta.get("messages", []),
                "success": meta.get("is_correct", False),
                "num_steps": meta.get("num_steps", 0),
                "valid_tool_calls": meta.get("valid_tool_calls", 0),
                "total_tool_calls": meta.get("total_tool_calls", 0),
                "redundant_calls": meta.get("redundant_calls", 0),
            })
        print(f"✅ Loaded {len(trajectories)} records via AuditKIT AgentTune Loader.")
        if trajectories:
            return trajectories
    except Exception as e:
        # Silently fall back if this is a standard HealGuard file
        pass

    # --- FALLBACK PATH ---
    trajectories = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(trajectory, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            trajectories.append(trajectory)

    if not trajectories:
        raise ValueError("Evaluation dataset is empty.")

    return trajectories


def _validate_integer(
    trajectory: dict[str, Any],
    field: str,
) -> int:
    """Validate a non-negative integer field."""

    value = trajectory.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Trajectory field '{field}' must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"Trajectory field '{field}' cannot be negative."
        )

    return value


def evaluate_trajectories(
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all five scorers and aggregate dataset-level metrics."""

    if not trajectories:
        raise ValueError("Evaluation dataset is empty.")

    total_trajectories = len(trajectories)

    successful_trajectories = 0
    successful_steps_list = []

    total_valid_calls = 0
    total_tool_calls = 0

    total_redundant_calls = 0

    looping_trajectories = 0
    loop_applicable_trajectories = 0

    tool_selection_scores: list[float] = []
    arg_valid_calls = 0
    arg_total_calls = 0

    for trajectory in trajectories:

        # Convert the trajectory into an AuditKIT sample.
        sample, output, context = trajectory_to_sample(trajectory)

        # Execute all five registered AuditKIT scorers.
        scores = {}

        for metric in METRICS:
            score = metric(sample, output, context=context)
            scores[score.name] = score

        # Task success.
        success = trajectory.get("success")

        if not isinstance(success, bool):
            raise ValueError(
                "Trajectory field 'success' must be boolean."
            )

        if success:
            successful_trajectories += 1
            successful_steps_list.append(
                _validate_integer(trajectory, "num_steps")
            )

        # Valid and redundant calls.
        valid = _validate_integer(
            trajectory, "valid_tool_calls"
        )

        total = _validate_integer(
            trajectory, "total_tool_calls"
        )

        redundant = _validate_integer(
            trajectory, "redundant_calls"
        )

        if valid > total:
            raise ValueError(
                "valid_tool_calls cannot exceed total_tool_calls."
            )

        if redundant > total:
            raise ValueError(
                "redundant_calls cannot exceed total_tool_calls."
            )

        total_valid_calls += valid
        total_tool_calls += total
        total_redundant_calls += redundant

        selection = scores["tool_selection_acc"]
        if not selection.metadata.get("not_applicable"):
            tool_selection_scores.append(selection.value)

        arguments = scores["arg_accuracy"]
        if not arguments.metadata.get("not_applicable"):
            arg_valid_calls += int(arguments.metadata["valid_calls"])
            arg_total_calls += int(arguments.metadata["total_calls"])

        # Loop detection using AuditKIT's result.
        loop_score = scores["loop_rate"]

        if loop_score.metadata.get("not_applicable"):
            continue

        loop_applicable_trajectories += 1

        if loop_score.metadata.get("loop_detected") is True:
            looping_trajectories += 1

    # Dataset-level aggregation.
    task_success_rate = (
        successful_trajectories / total_trajectories
    )

    valid_call_rate = (
        total_valid_calls / total_tool_calls
        if total_tool_calls > 0
        else None
    )

    redundant_rate = (
        total_redundant_calls / total_tool_calls
        if total_tool_calls > 0
        else None
    )

    loop_rate = looping_trajectories / total_trajectories

    steps_on_success = (
        statistics.median(successful_steps_list)
        if successful_steps_list
        else None
    )

    tool_selection_acc = (
        statistics.mean(tool_selection_scores)
        if tool_selection_scores
        else None
    )

    arg_accuracy = (
        arg_valid_calls / arg_total_calls
        if arg_total_calls > 0
        else None
    )

    successful_steps = sum(successful_steps_list)

    return {
        "dataset_summary": {
            "total_trajectories": total_trajectories,
            "successful_trajectories": successful_trajectories,
            "failed_trajectories": (
                total_trajectories - successful_trajectories
            ),
            "total_tool_calls": total_tool_calls,
            "valid_tool_calls": total_valid_calls,
            "redundant_calls": total_redundant_calls,
        },
        "metrics": {
            "task_success": {
                "value": task_success_rate,
                "direction": "maximize",
                "numerator": successful_trajectories,
                "denominator": total_trajectories,
            },
            "valid_call_rate": {
                "value": valid_call_rate,
                "direction": "maximize",
                "numerator": total_valid_calls,
                "denominator": total_tool_calls,
            },
            "redundant_rate": {
                "value": redundant_rate,
                "direction": "minimize",
                "numerator": total_redundant_calls,
                "denominator": total_tool_calls,
            },
            "loop_rate": {
                "value": loop_rate,
                "direction": "minimize",
                "numerator": looping_trajectories,
                "denominator": total_trajectories,
            },
            "steps_on_success": {
                "value": steps_on_success,
                "direction": "minimize",
                "numerator": successful_steps,
                "denominator": successful_trajectories,
            },
            "tool_selection_acc": {
                "value": tool_selection_acc,
                "direction": "maximize",
                "numerator": None,
                "denominator": len(tool_selection_scores),
            },
            "arg_accuracy": {
                "value": arg_accuracy,
                "direction": "maximize",
                "numerator": arg_valid_calls,
                "denominator": arg_total_calls,
            },
        },
    }


def evaluate_jsonl(path: str | Path) -> dict[str, Any]:
    """Load a JSONL dataset and evaluate it."""

    trajectories = load_trajectories(path)
    return evaluate_trajectories(trajectories)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HealGuard evaluation metrics."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the trajectory JSONL file.",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path for the evaluation report.",
    )

    args = parser.parse_args()

    results = evaluate_jsonl(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, allow_nan=False)

    print("HealGuard evaluation completed.")
    print(f"Trajectories: {results['dataset_summary']['total_trajectories']}")
    print(f"Report saved to: {output_path}")

    for name, details in results["metrics"].items():
        value = details["value"]

        if value is None:
            display_value = "N/A"
        else:
            display_value = f"{value:.4f}"

        print(f"{name}: {display_value}")


if __name__ == "__main__":
    main()