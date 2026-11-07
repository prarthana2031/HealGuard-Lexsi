import argparse
import subprocess
import sys
import json
import os

# Import Person C's gate function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gates.deployment_gate import evaluate_deployment_gate

def run_script(script_path, description):
    print(f"\n{'='*50}\n🚀 {description}...\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Error in {script_path}")
        sys.exit(1)

def load_records(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser(description="HealGuard Orchestrator")
    parser.add_argument("--stage", type=str, required=True, choices=["all", "train", "gate", "report"])
    args = parser.parse_args()

    if args.stage in ["all", "train"]:
        run_script("training/train_v0.py", "Training v0 Adapter")
        run_script("training/train_heal.py", "Training v1_good Heal Adapter")
        run_script("training/train_poisoned.py", "Training v1_poisoned Adapter")
        
    if args.stage in ["all", "gate"]:
        print(f"\n{'='*50}\n🛡️ Running HealGate Evaluation...\n{'='*50}")
        
        # These paths will depend on where Person C puts the eval outputs
        # I am assuming she creates these files
        baseline_path = "eval/results/baseline_records.jsonl"
        candidate_path = "eval/results/v1_good_records.jsonl"
        
        if os.path.exists(baseline_path) and os.path.exists(candidate_path):
            baseline_records = load_records(baseline_path)
            candidate_records = load_records(candidate_path)
            
            decision = evaluate_deployment_gate(baseline_records, candidate_records)
            print(f"\n✅ GATE DECISION: {decision}")
        else:
            print("❌ Missing evaluation records. Run eval stage first.")
            sys.exit(1)

    print("\n✅ Pipeline stage completed successfully!")

if __name__ == "__main__":
    main()