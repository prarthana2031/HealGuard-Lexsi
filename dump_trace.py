import json
import os

def dump_trace(seed):
    # Try a few likely paths based on your screenshot
    paths = [
        'trajectories/v0_guard_clean_D1_stale_drift_targeted.jsonl',
        'artifacts/trajectories/v0_guard_clean_D1_stale_drift_targeted.jsonl'
    ]
    
    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if not file_path:
        print(f"Could not find the file in {paths}. Please check the exact path.")
        return

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('seed') == seed:
                print(f"\n=== EXCERPT FOR SEED {seed} ===")
                msgs = r.get('messages', [])
                for m in msgs:
                    role = m.get('role', 'unknown')
                    content = m.get('content', '')
                    # Print the last 3 messages to show the failure
                    if role == 'assistant' or role == 'tool':
                        print(f"[{role}]: {content}\n")
                break

dump_trace(40002)
dump_trace(40007)