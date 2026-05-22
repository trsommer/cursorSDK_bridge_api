import httpx
import json

url = "http://127.0.0.1:8764/v1/chat/completions"
payload = {
    "model": "composer-2.5-slow",
    "messages": [
        {"role": "user", "content": "calculate the derivative of 4x^2 - 5x + 4 = 0"}
    ]
}

# Test non-streaming
print("--- TESTING NON-STREAMING ---")
payload["stream"] = False
r = httpx.post(url, json=payload, timeout=60.0)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    message = data["choices"][0]["message"]
    print("KEYS:", list(message.keys()))
    print("REASONING_CONTENT:")
    print(repr(message.get("reasoning_content")))
    print("CONTENT:")
    print(repr(message.get("content")))
else:
    print(r.text)

# Test streaming
print("\n--- TESTING STREAMING ---")
payload["stream"] = True
with httpx.stream("POST", url, json=payload, timeout=60.0) as r:
    print("Status code:", r.status_code)
    for line in r.iter_lines():
        if line.startswith("data: "):
            content = line[6:]
            if content == "[DONE]":
                break
            try:
                data = json.loads(content)
                delta = data["choices"][0]["delta"]
                if "reasoning_content" in delta:
                    print(f"REASONING: {repr(delta['reasoning_content'])}")
                if "content" in delta:
                    print(f"CONTENT: {repr(delta['content'])}")
            except Exception as e:
                print(f"ERROR: {e}")
