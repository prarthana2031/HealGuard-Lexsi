import json
import os

def analyze_run(path, label):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, 'r') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    total = len(rows)
    grounded = sum(1 for r in rows if r.get('grounded_success'))
    print(f"{label}: {grounded}/{total} grounded ({grounded/total:.1%})")

def extract_trace(path, seed):
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('seed') == seed:
                print(f"\n--- Trace for seed {seed} ---")
                for m in r.get('messages', []):
                    print(f"[{m['role']}]: {m.get('content', '')}")
                break

print("=== GROUNDED SUCCESS ===")
analyze_run('artifacts/runs/v0_fresh_clean_D1_stale_drift_targeted.jsonl', 'v0 Drift')
analyze_run('artifacts/runs/v0_fresh_clean_clean_gate_regression.jsonl', 'v0 Clean')
analyze_run('artifacts/runs/v0_guard_clean_D1_stale_drift_targeted.jsonl', 'v0_guard Drift')
analyze_run('artifacts/runs/v0_guard_clean_clean_gate_regression.jsonl', 'v0_guard Clean')

print("\n=== TRACE EXCERPTS ===")
extract_trace('artifacts/trajectories/v0_guard_clean_D1_stale_drift_targeted.jsonl', 40002)
extract_trace('artifacts/trajectories/v0_guard_clean_D1_stale_drift_targeted.jsonl', 40007)