import httpx
import json

url = "http://127.0.0.1:8764/v1/chat/completions"
payload = {
    "model": "composer-2.5-slow",
    "messages": [
        {"role": "user", "content": "calculate the derivative of 4x^2 - 5x + 4 = 0"}
    ],
    "stream": True
}

print("Sending request to:", url)
with httpx.stream("POST", url, json=payload, timeout=60.0) as r:
    print("Status code:", r.status_code)
    for line in r.iter_lines():
        if line.startswith("data: "):
            content = line[6:]
            if content == "[DONE]":
                print("\n[DONE]")
                break
            try:
                data = json.loads(content)
                delta = data["choices"][0]["delta"]
                if "reasoning_content" in delta:
                    print(f"REASONING: {repr(delta['reasoning_content'])}")
                if "content" in delta:
                    print(f"CONTENT: {repr(delta['content'])}")
            except Exception as e:
                print(f"ERROR PARSING: {e} | Line: {line}")
