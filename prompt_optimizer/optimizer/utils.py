import requests

def call_llm(prompt):
    # Using OpenRouter.ai (free LLM API)
    api_key = "YOUR_OPENROUTER_API_KEY"  # Store this securely
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]
