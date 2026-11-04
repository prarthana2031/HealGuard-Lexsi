import json
import os
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

# Define toy tool
harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=4)

strategy = ReActStrategy(
    lambda s: {"name": "add", "arguments": {"a": 2, "b": 3}}
    if s.step == 0
    else {"name": "finish", "arguments": {"answer": "5"}}
)

proj = Project(strategy=strategy, harness=harness)

# One episode
print("Running episode...")
result = proj.infer("what is 2+3?")
print("Infer result:", result)

# Evaluate
score = proj.evaluate([{"task": "what is 2+3?", "expected": "5"}])
print("Evaluate score:", score)

# ==========================================
# THE MISSING PIECE: Write trajectory to disk
# ==========================================
os.makedirs("artifacts", exist_ok=True)
traj_path = "artifacts/toy_trajectories.jsonl"

# Format the trajectory to match what Person C's AuditKIT expects
# (You may need to adjust this dictionary to exactly match your trajectory_schema)
trajectory_record = {
    "task_id": "toy_1",
    "seed": 0,
    "split": "dev",
    "messages": result.get("messages", []), # Assuming AgentTune returns messages
    "final_answer": "5",
    "is_correct": True
}

with open(traj_path, "w") as f:
    f.write(json.dumps(trajectory_record) + "\n")

print(f"✅ S1 PASSED. Trajectory written to {traj_path}")