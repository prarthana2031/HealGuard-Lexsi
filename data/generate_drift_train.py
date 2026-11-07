import os
import sys
import json
import torch
import re

# Add root to sys.path so we can import world and heal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from world.snapshot import generate_snapshot
from world.tools import CloudEnv
from heal.drifts.dispatcher import create_dispatcher

MODEL_ID = "CohereLabs/tiny-aya-global"
V0_ADAPTER = "./adapters/v0/final"
OUTPUT_DIR = "data/drift_train"
DRIFTS = ["D1", "D2", "D3", "D4"]

def extract_tool_call(text):
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: return None
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading v0 Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # PASTE YOUR EXACT tool_template FROM train_v0.py HERE:
    tool_template = """{{ bos_token }}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'user' %}
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'assistant' %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'tool' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Tool result: {{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>
{%- endif %}
"""
    tokenizer.chat_template = tool_template
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0})
    
    # Load the v0 adapter you just trained!
    model = PeftModel.from_pretrained(model, V0_ADAPTER)
    model.eval()

    # Load drift_targeted seeds
    with open("data/drift_targeted/expert_traces.jsonl", "r") as f:
        target_seeds = [json.loads(line)["seed"] for line in f]

    all_traces = []
    
    for drift in DRIFTS:
        print(f"Generating rollouts for {drift}...")
        for seed in target_seeds:
            snapshot = generate_snapshot(seed)
            base_env = CloudEnv(snapshot)
            dispatcher = create_dispatcher(base_env, drift=drift)
            env = dispatcher.env
            
            # Use drifted schemas for the system prompt
            tool_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in dispatcher.schemas])
            messages = [
                {"role": "system", "content": f"You are a cloud auditor. Tools:\n{tool_desc}"},
                {"role": "user", "content": "Find all vulnerable cloud resources."}
            ]
            
            final_answer = []
            # Max 5 steps for model rollout
            for step in range(5):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                
                response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                messages.append({"role": "assistant", "content": response_text})
                
                tool_call = extract_tool_call(response_text)
                if not tool_call:
                    break
                    
                tool_name = tool_call.get("name")
                args = tool_call.get("arguments", {})
                
                if tool_name == "finish":
                    final_answer = args.get("answer", [])
                    break
                    
                # Dispatch using the drifted env
                tool_result = dispatcher.dispatch(tool_call)
                messages.append({"role": "tool", "content": json.dumps(tool_result)})

            trace = {
                "task_id": f"{drift}_{seed}",
                "seed": seed,
                "drift": drift,
                "messages": messages,
                "final_answer": final_answer,
            }
            all_traces.append(trace)

    out_path = os.path.join(OUTPUT_DIR, "v0_failed_rollouts.jsonl")
    with open(out_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")
            
    print(f"✅ Saved {len(all_traces)} drift rollouts to {out_path}")

if __name__ == "__main__":
    main()