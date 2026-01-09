#!/usr/bin/env python3
"""Check available Gemini models."""
import httpx
import os
from dotenv import load_dotenv

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY", "")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"

r = httpx.get(url, timeout=30)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    data = r.json()
    models = data.get("models", [])
    print(f"\n✅ Found {len(models)} models:\n")
    for m in models:
        name = m.get("name", "").replace("models/", "")
        desc = m.get("displayName", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            print(f"  • {name}")
            print(f"    Display: {desc}")
            print()
else:
    print(f"Error: {r.text}")
