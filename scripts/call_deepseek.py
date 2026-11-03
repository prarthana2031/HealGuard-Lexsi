import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def ask_deepseek(messages, model="deepseek/deepseek-v4.1-flash-1017"):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps({
            "model": model,
            "messages": messages,
            "reasoning": {"enabled": True}
        })
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]

if __name__ == "__main__":
    # Quick test
    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
    ]
    
    result = ask_deepseek(messages)
    print("Answer:", result.get("content"))