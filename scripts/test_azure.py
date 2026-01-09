#!/usr/bin/env python3
"""Test Azure OpenAI connection."""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv(override=True)

endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '').rstrip('/')
key = os.getenv('AZURE_OPENAI_KEY', '')
deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', '')
version = os.getenv('AZURE_OPENAI_API_VERSION', '')

print("=" * 50)
print("Azure OpenAI Connection Test")
print("=" * 50)
print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment}")
print(f"API Version: {version}")
print()

url = f'{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}'
print(f"Full URL: {url}")
print()

headers = {'Content-Type': 'application/json', 'api-key': key}
payload = {
    'messages': [
        {'role': 'user', 'content': 'Say hello briefly.'}
    ],
    'max_completion_tokens': 2000
}

async def test():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")
        
        if r.status_code == 200:
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"\n✅ SUCCESS!\nResponse: {content}")
        else:
            print(f"\n❌ FAILED")

asyncio.run(test())
