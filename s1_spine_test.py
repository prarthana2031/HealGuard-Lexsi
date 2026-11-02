from agenttune.agentic import Project, DictToolHarness, ReActStrategy

harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=4)

strategy = ReActStrategy(
    lambda s: {"name": "add", "arguments": {"a": 2, "b": 3}}
    if s.step == 0
    else {"name": "finish", "arguments": {"answer": "5"}}
)

proj = Project(strategy=strategy, harness=harness)

# One episode
result = proj.infer("what is 2+3?")
print("Infer result:", result)

# Evaluate
score = proj.evaluate([{"task": "what is 2+3?", "expected": "5"}])
print("Evaluate score:", score)

print("S1 PASSED")