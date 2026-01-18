import requests
import os
from dotenv import load_dotenv

load_dotenv()

response = requests.get("https://openrouter.ai/api/v1/models")
if response.status_code == 200:
    models = response.json().get('data', [])
    print("--- Available FREE Models on OpenRouter ---")
    count = 0
    for m in models:
        # Check if it has 'free' in the ID
        if ':free' in m['id']:
            print(f"- {m['id']}")
            count += 1
    if count == 0:
        print("No models with ':free' suffix found. Checking all models...")
        # Fallback print some popular ones
        for m in models[:10]:
             print(f"- {m['id']}")
else:
    print(f"Failed to fetch models: {response.status_code}")
