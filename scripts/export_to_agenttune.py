"""Converts HealGuard JSONL rollouts into AgentTune's TrajectoryDataset format.

This allows AuditKIT to load the data using `ak.load_agenttune()`.
Run this before evaluating with AuditKIT to satisfy the 2-repo integration rule.

Usage:
    python -m scripts.export_to_agenttune data/drift_train/v0_d1train_clean_D1_stale_drift_train.jsonl data/drift_train/agenttune_export.jsonl
"""

import json
import sys
from pathlib import Path

def convert_to_agenttune(healguard_row):
    """
    Maps a HealGuard trajectory to AgentTune's TrajectoryDataset schema.
    """
    # 1. Core fields
    task_id = healguard_row.get("task_id", "unknown")
    messages = healguard_row.get("messages", [])
    
    # Handle the final answer (AgentTune expects a string or structured output)
    final_answer = healguard_row.get("final_answer", [])
    final_response = json.dumps(final_answer) if isinstance(final_answer, list) else str(final_answer)

    # 2. Build the metadata (retaining ALL original fields so the gate can read them)
    metadata = {
        "seed": healguard_row.get("seed"),
        "split": healguard_row.get("split"),
        "is_correct": healguard_row.get("is_correct"),
        "num_steps": healguard_row.get("num_steps"),
        "valid_tool_calls": healguard_row.get("valid_tool_calls"),
        "total_tool_calls": healguard_row.get("total_tool_calls"),
        "redundant_calls": healguard_row.get("redundant_calls"),
        "template_id": healguard_row.get("template_id", "custom_tool_call_v1"),
    }

    # 3. Construct the AgentTune record
    agenttune_record = {
        "trajectory_id": task_id,
        "messages": messages,
        "final_response": final_response,
        "metadata": metadata
    }
    
    return agenttune_record

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.export_to_agenttune <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.is_file():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    print(f"Reading {input_path}...")
    
    converted_rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            agenttune_row = convert_to_agenttune(row)
            converted_rows.append(agenttune_row)

    print(f"Converted {len(converted_rows)} rows.")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in converted_rows:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Saved AgentTune format to {output_path}")
    print(f"\nTo evaluate with AuditKIT, update your runner to use:")
    print(f"  import auditkit as ak")
    print(f"  samples = ak.load_agenttune('{output_path}')")

if __name__ == "__main__":
    main()