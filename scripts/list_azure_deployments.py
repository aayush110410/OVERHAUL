#!/usr/bin/env python3
"""List all deployments in your Azure OpenAI resource."""
import httpx
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
key = os.getenv("AZURE_OPENAI_KEY", "")
version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

url = f"{endpoint}/openai/deployments?api-version={version}"
headers = {"api-key": key}

print("=" * 50)
print("Azure OpenAI Deployment Finder")
print("=" * 50)
print(f"Endpoint: {endpoint}")
print(f"API Version: {version}")
print()

try:
    r = httpx.get(url, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        deployments = data.get("data", [])
        if deployments:
            print("\n✅ FOUND DEPLOYMENTS:")
            print("-" * 40)
            for d in deployments:
                dep_id = d.get("id")
                model = d.get("model")
                print(f"  • Deployment: \"{dep_id}\"")
                print(f"    Model: {model}")
                print()
            print("-" * 40)
            print("ACTION: Update your .env file:")
            print(f'  AZURE_OPENAI_DEPLOYMENT={deployments[0].get("id")}')
        else:
            print("\n❌ No deployments found in this resource.")
            print("   Go to Azure AI Foundry and deploy a model first.")
    else:
        print(f"\n❌ Error response:")
        print(r.text[:800])
except Exception as e:
    print(f"\n❌ Request failed: {e}")
