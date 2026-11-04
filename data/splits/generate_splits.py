import os
import sys

# Add the root 'heal-guard' directory to the Python path so it can find 'world' and 'expert'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import hashlib
from world.snapshot import generate_snapshot
from expert.scripted import run_expert

def generate_split(name: str, count: int, start_seed: int, inject_error: bool = False):
    """Generates the JSONL file for a specific split."""
    os.makedirs(f"data/{name}", exist_ok=True)
    path = f"data/{name}/expert_traces.jsonl"
    
    traces = []
    for i in range(count):
        seed = start_seed + i
        snap = generate_snapshot(seed)
        # Inject error recovery only in the training set
        trace = run_expert(snap, inject_error=inject_error)
        trace["split"] = name
        traces.append(json.dumps(trace))
        
    with open(path, "w") as f:
        f.write("\n".join(traces))
        
    print(f"✅ Generated {count} traces for {name}")

if __name__ == "__main__":
    print("Starting data generation...")
    
    # Generate the 4 main splits
    generate_split("train_clean", 100, start_seed=0, inject_error=True)
    generate_split("dev", 20, start_seed=1000)
    generate_split("drift_targeted", 20, start_seed=2000) # Will be wrapped later
    generate_split("gate_regression", 20, start_seed=3000)

    # Generate the Sealed Set and lock it
    os.makedirs("data/final_sealed", exist_ok=True)
    sealed_path = "data/final_sealed/sealed_traces.jsonl"
    sealed_lines = []
    for i in range(50):
        snap = generate_snapshot(5000 + i)
        trace = run_expert(snap)
        trace["split"] = "final_sealed"
        sealed_lines.append(json.dumps(trace))

    sealed_str = "\n".join(sealed_lines)
    with open(sealed_path, "w") as f:
        f.write(sealed_str)

    # Hash the sealed set (as required by the PDF)
    hash_val = hashlib.sha256(sealed_str.encode()).hexdigest()
    os.makedirs("data/splits", exist_ok=True)
    with open("data/splits/sealed_hash.txt", "w") as f:
        f.write(hash_val)

    print(f"🔒 Sealed set generated and hashed: {hash_val}")
    print("⚠️ DO NOT OPEN data/final_sealed UNTIL NOV 12.")