"""Test OpenRouter embedding directly to see what errors it returns."""

import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

print("Testing embedding...")
try:
    embedding = client.embeddings.create(
        model="qwen/qwen3-embedding-4b",
        input="test text",
        encoding_format="float"
    )
    print(f"Success! Got {len(embedding.data[0].embedding)} floats")
    print(f"Response keys: {list(embedding.model_dump().keys())}")
    print(f"Data type: {type(embedding.data)}")
    print(f"Data length: {len(embedding.data)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    print(f"\nFull traceback:\n{traceback.format_exc()}")
