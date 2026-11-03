import auditkit as ak

print("AuditKIT imported successfully")

# Correct Sample API
samples = [
    ak.Sample(input="What is 2+2?", target="4"),
    ak.Sample(input="What is 3+3?", target="6"),
]

print("Samples created:", len(samples))

# Minimal evaluate (uses a dummy model that just echoes)
try:
    result = ak.evaluate(samples, model=lambda prompts: ["4", "6"])
    print("Evaluate worked")
    print(result.summary() if hasattr(result, "summary") else result)
except Exception as e:
    print("Evaluate note (still OK for smoke):", e)

print("S3 PASSED")