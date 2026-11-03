from datasets import Dataset
from agenttune.core.backend_factory import create_agentic_trainer
import time
import torch

def add(a: int, b: int) -> int:
    """Add two integers.
    Args:
        a: first integer
        b: second integer
    """
    return a + b

def used_tool(completions, **kwargs):
    return [1.0 if "5" in str(c) else 0.0 for c in completions]

train_dataset = Dataset.from_list(
    [{"prompt": [{"role": "user", "content": "What is 2+3?"}]}] * 8
)

print("Starting Tiny Aya S2 test...")
start = time.time()

trainer = create_agentic_trainer(
    algorithm="grpo",
    model="CohereLabs/tiny-aya-global",
    train_dataset=train_dataset,
    tools=[add],
    reward_funcs=[used_tool],
    output_dir="./output/tiny_aya_s2",
    max_steps=10,                          # short smoke
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_steps_per_turn=3,
    use_vllm=False,                        # important
    peft_config=dict(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    ),
    report_to="none",
)

trainer.train()

elapsed = time.time() - start
print(f"\nTime taken: {elapsed:.1f} seconds")
print(f"Approx sec/step: {elapsed / 10:.2f}")

if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("S2 finished – check the output folder and logs for parseable tool calls")