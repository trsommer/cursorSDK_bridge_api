# Cursor SDK OpenAI & MCP Bridge

This project implements a fully-functioning, local OpenAI-compatible LLM server that acts as a bridge to the Cursor Python SDK (`cursor-sdk`). It enables you to use Cursor's powerful local agent (the same agent that runs inside the Cursor IDE) with any standard OpenAI API client (such as OpenCode).

Additionally, it provides a standalone Model Context Protocol (MCP) server that exposes a tool to execute Cursor Agent tasks programmatically.

---

## Key Features

1. **OpenAI Compatibility**: Serves `/v1/chat/completions` supporting both streaming (`stream=True`) and non-streaming modes.
2. **First-Class Reasoning/Thinking Support**: Streams reasoning via `reasoning_content` and answer text via `content`, ordered using the Cursor SDK’s canonical `thinkingMessage` / `assistantMessage` conversation steps (with token deltas only as a fallback). Supports interleaved think→answer→think→answer turns and inserts a fresh `role: assistant` chunk between segments so OpenCode does not merge separate answer blocks.
3. **Stateful Conversation Resumption**: Integrates message-prefix matching and custom session tracking to automatically resume local Cursor Agents (`Agent.resume()`) for multi-turn chats.
4. **Tool Call Interception Bridge**: Seamlessly translates OpenAI tool definitions (`tools` parameters) into dynamic MCP tool listings for the Cursor Agent. It intercepts agent tool call invocations, suspends the execution loop, yields the tool calls to the client, and resumes the agent upon receiving results.
5. **Custom Workspace Pathing**: Dynamically matches host workspaces via environment variable (`CURSOR_WORKSPACE`), content block extraction, or custom HTTP request header (`X-Workspace-Path`).
6. **Standalone MCP Server**: Exposes a `run_cursor_agent` tool using `FastMCP` for programmatically spawning local workspace tasks.

---

## Installation & Setup

1. Make sure you have Python (3.10+) installed.
2. Copy the environment template file:
   ```bash
   cp .env.template .env
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open `.env` and set your `CURSOR_API_KEY` (retrieve it from the [Cursor Dashboard](https://cursor.com/settings) -> Integrations).
5. Configure optional parameters like the default `PORT` (defaults to `8764`) and `CURSOR_WORKSPACE` (the workspace root directory you want the agent to edit/read).

---

## 1. Running the OpenAI Bridge Server

Start the bridge server in your terminal:
```bash
python3 cursor_bridge_server.py
```
Or using **`uv`** (which automatically installs the required Python version and dependencies defined in `pyproject.toml`):
```bash
uv run cursor_bridge_server.py
```
By default, the server runs on `http://127.0.0.1:8764`.

### Exposing to the Local Network
To allow other devices on your local network to connect to this server:
1. Open `.env` and change `HOST=127.0.0.1` to `HOST=0.0.0.0`.
2. Restart the server: `python3 cursor_bridge_server.py`.
3. Find your local IP address (e.g., run `ipconfig getifaddr en0` on macOS).
4. Configure your client application with your computer's local IP (e.g., `"baseURL": "http://192.168.1.50:8764/v1"`).

---

## Client Integration Guide

### OpenCode configuration (`@ai-sdk/openai-compatible`)

To use this bridge in your editor as a backend proxy, add the following configuration block:

```json
{
  "cursor": {
    "npm": "@ai-sdk/openai-compatible",
    "name": "Cursor Backend Proxy",
    "options": {
      "baseURL": "http://127.0.0.1:8764/v1",
      "apiKey": "dummy-key-not-used-by-proxy"
    },
    "models": {
      "composer-2.5": {
        "name": "Composer 2.5 fast",
        "limit": {
          "context": 200000,
          "output": 65536
        }
      },
      "composer-2.5-slow": {
        "name": "Composer 2.5 slow",
        "limit": {
          "context": 200000,
          "output": 65536
        }
      }
    }
  }
}
```

### Custom Headers (Advanced)

* **`X-Workspace-Path`**: Specify the absolute local folder path that the Cursor Agent should operate in. If not sent, the server defaults to the `CURSOR_WORKSPACE` env variable or the directory where the server was launched.
* **`X-Session-ID`**: Manually specify a session ID to keep conversation state across requests. If omitted, the server will fall back to using the `user` field or automatically prefix-matching message histories.

---

## 2. Running the Standalone MCP Server

If you are using a terminal-native client like OpenCode that connects directly to MCP servers:
```bash
python3 mcp_server.py
```
Or using **`uv`**:
```bash
uv run mcp_server.py
```
This starts an MCP stdio server. You can configure OpenCode or Claude Desktop to start this server by configuring the command:
```json
"mcpServers": {
  "cursor-bridge": {
    "command": "uv",
    "args": ["run", "/absolute/path/to/mcp_server.py"],
    "env": {
      "CURSOR_API_KEY": "your_api_key_here",
      "CURSOR_WORKSPACE": "/path/to/your/workspace"
    }
  }
}
```

### Exposed Tools
* **`run_cursor_agent(prompt: str, cwd: str = None) -> str`**: Instructs the local Cursor Agent to run the task described in `prompt` inside the workspace `cwd` and returns the final text response.

---

## Verification & Testing

### Automated Test Suite

A mock-integration test suite is included to verify the complete server flow (including tool calling and SSE queue pipes) without requiring an internet connection or a real API key.

Run the tests:
```bash
python3 test_server.py
```

### Manual API Tests

You can also run manual HTTP requests against a running server.

#### 1. List Models
```bash
curl http://127.0.0.1:8764/v1/models
```

#### 2. Chat Completions (Non-Streaming)
```bash
curl http://127.0.0.1:8764/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "composer-2.5-slow",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "stream": false
  }'
```

#### 3. Chat Completions (Streaming)
```bash
curl http://127.0.0.1:8764/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "composer-2.5-slow",
    "messages": [{"role": "user", "content": "Explain what Python is in one sentence."}],
    "stream": true
  }'
```
