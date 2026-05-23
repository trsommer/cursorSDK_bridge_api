import os
import sys
import json
import time
import asyncio
import threading
import unittest
from unittest.mock import MagicMock, AsyncMock

import httpx
import uvicorn
import uuid

# 1. Mock cursor_sdk BEFORE importing the server
mock_sdk = MagicMock()
mock_events = MagicMock()

class MockTextDeltaUpdate:
    def __init__(self, text):
        self.text = text

class MockThinkingDeltaUpdate:
    def __init__(self, text):
        self.text = text

class MockThinkingCompletedUpdate:
    type = "thinking-completed"
    thinking_duration_ms = 0

class MockToolCallStartedUpdate:
    pass

class MockTurnEndedUpdate:
    pass

mock_sdk.TextDeltaUpdate = MockTextDeltaUpdate
mock_sdk.ThinkingDeltaUpdate = MockThinkingDeltaUpdate
mock_sdk.ThinkingCompletedUpdate = MockThinkingCompletedUpdate
mock_sdk.ToolCallStartedUpdate = MockToolCallStartedUpdate
mock_sdk.TurnEndedUpdate = MockTurnEndedUpdate

mock_events.TextDeltaUpdate = MockTextDeltaUpdate
mock_events.ThinkingDeltaUpdate = MockThinkingDeltaUpdate
mock_events.ThinkingCompletedUpdate = MockThinkingCompletedUpdate
mock_events.ToolCallStartedUpdate = MockToolCallStartedUpdate
mock_events.TurnEndedUpdate = MockTurnEndedUpdate

# Mock configuration types
class MockLocalAgentOptions:
    def __init__(self, **kwargs):
        self.cwd = kwargs.get("cwd")

class MockHttpMcpServerConfig:
    def __init__(self, **kwargs):
        self.url = kwargs.get("url")
        self.type = kwargs.get("type")

class MockSendOptions:
    def __init__(self, **kwargs):
        self.on_delta = kwargs.get("on_delta")

class MockAgentOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

mock_sdk.LocalAgentOptions = MockLocalAgentOptions
mock_sdk.HttpMcpServerConfig = MockHttpMcpServerConfig
mock_sdk.SendOptions = MockSendOptions
mock_sdk.AgentOptions = MockAgentOptions

# Mock Run result
class MockRunResult:
    def __init__(self, status="finished", result="Mock agent response"):
        self.status = status
        self.result = result
        self.model = MagicMock()
        self.duration_ms = 100

# Mock Run
class MockRun:
    def __init__(self, agent, prompt, on_delta):
        self.agent = agent
        self.prompt = prompt
        self.on_delta = on_delta
        self.status = "running"
        self._wait_future = asyncio.Future()
        
        # Start a background task to simulate agent logic
        asyncio.create_task(self._simulate())

    async def _simulate(self):
        try:
            # Check prompt for specific test triggers
            if "trigger_tool" in self.prompt:
                # Simulate calling an MCP tool
                # We need to contact the SSE MCP server of this session
                mcp_url = self.agent.mcp_url
                session_id = self.agent.session_id
                
                # We wait 0.1s to let the SSE connection establish
                await asyncio.sleep(0.1)
                
                # Make POST request to the MCP messages endpoint
                async with httpx.AsyncClient() as client:
                    messages_url = mcp_url.replace("/sse", "/messages")
                    
                    # 1. Send initialize
                    init_payload = {
                        "jsonrpc": "2.0",
                        "method": "initialize",
                        "id": 1,
                        "params": {"protocolVersion": "2024-11-05"}
                    }
                    await client.post(messages_url, json=init_payload)
                    
                    # 2. List tools
                    list_payload = {
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "id": 2
                    }
                    await client.post(messages_url, json=list_payload)
                    
                    # 3. Call tool
                    call_payload = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "id": 3,
                        "params": {
                            "name": "get_user_info",
                            "arguments": {"user_id": "123"}
                        }
                    }
                    # This POST will block until the tool result is resolved!
                    resp = await client.post(messages_url, json=call_payload, timeout=5.0)
                    tool_result = resp.json() if resp.status_code == 200 else {}
                    
                # Yield text response based on tool result
                if self.on_delta:
                    self.on_delta(MockTextDeltaUpdate(f"Tool returned: {tool_result}"))
                self.status = "finished"
                self._wait_future.set_result(MockRunResult(result=f"Tool result received: {tool_result}"))
            elif "out_of_order_thinking" in self.prompt:
                if self.on_delta:
                    self.on_delta(MockTextDeltaUpdate("Answer first. "))
                    await asyncio.sleep(0.05)
                    self.on_delta(MockThinkingDeltaUpdate("Late thinking."))
                    self.on_delta(MockThinkingCompletedUpdate())
                    self.on_delta(MockTextDeltaUpdate("More answer."))
                self.status = "finished"
                self._wait_future.set_result(
                    MockRunResult(result="Answer first. More answer.")
                )
            else:
                # Normal text streaming simulation
                await asyncio.sleep(0.05)
                if self.on_delta:
                    self.on_delta(MockThinkingDeltaUpdate("Thinking about greeting..."))
                    await asyncio.sleep(0.05)
                    self.on_delta(MockThinkingCompletedUpdate())
                    self.on_delta(MockTextDeltaUpdate("Hello! "))
                    await asyncio.sleep(0.05)
                    self.on_delta(MockTextDeltaUpdate("I am a mock agent."))
                self.status = "finished"
                self._wait_future.set_result(MockRunResult(result="Hello! I am a mock agent."))
        except Exception as e:
            self.status = "error"
            self._wait_future.set_exception(e)

    async def wait(self):
        return await self._wait_future

# Mock Agent
class MockAgent:
    def __init__(self, model, api_key, local_options, mcp_servers, session_id):
        self.model = model
        self.api_key = api_key
        self.local_options = local_options
        self.agent_id = f"agent-mock-{uuid.uuid4().hex[:6]}"
        self.session_id = session_id
        
        # Get MCP URL
        self.mcp_url = mcp_servers["bridge_mcp"].url if mcp_servers else None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        pass
        
    async def send(self, prompt, options=None):
        on_delta = options.on_delta if options else None
        return MockRun(self, prompt, on_delta)
        
    async def close(self):
        pass

# Mock Agents Resource Namespace
class MockAgentsResource:
    def __init__(self):
        pass
    async def create(self, options=None, *, model=None, api_key=None, name=None, local=None, cloud=None, idempotency_key=None):
        if options is not None:
            if isinstance(options, dict):
                model = options.get("model")
                api_key = options.get("api_key")
                local = options.get("local")
                mcp_servers = options.get("mcp_servers")
            else:
                model = getattr(options, "model", None)
                api_key = getattr(options, "api_key", None)
                local = getattr(options, "local", None)
                mcp_servers = getattr(options, "mcp_servers", None)
        else:
            mcp_servers = None

        # Extract session_id from MCP URL if present
        session_id = "default"
        if mcp_servers and "bridge_mcp" in mcp_servers:
            url = mcp_servers["bridge_mcp"].url
            # http://127.0.0.1:PORT/mcp/session_id/sse
            parts = url.split("/")
            if len(parts) >= 5:
                session_id = parts[-2]
        return MockAgent(model, api_key, local, mcp_servers, session_id)

# Mock Async Client
class MockAsyncClient:
    def __init__(self, workspace):
        self.workspace = workspace
        self.agents = MockAgentsResource()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        pass
        
    async def aclose(self):
        pass
        
    @staticmethod
    async def launch_bridge(workspace):
        return MockAsyncClient(workspace)

mock_sdk.AsyncClient = MockAsyncClient

sys.modules['cursor_sdk'] = mock_sdk
sys.modules['cursor_sdk.events'] = mock_events

# 2. Now import our bridge server (it will use the mocked cursor_sdk)
import cursor_bridge_server

TEST_PORT = 9876
cursor_bridge_server.PORT = TEST_PORT

class TestCursorBridgeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start server in a background thread
        cls.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(cursor_bridge_server.app,),
            kwargs={"host": "127.0.0.1", "port": TEST_PORT, "log_level": "warning"},
            daemon=True
        )
        cls.server_thread.start()
        # Give server a moment to start
        time.sleep(1.0)
        cls.client = httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}")

    def test_01_models_endpoint(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        models = [m["id"] for m in data["data"]]
        self.assertIn("composer-2.5", models)

    def test_02_chat_completions_non_streaming(self):
        payload = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "say hello"}],
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        message = data["choices"][0]["message"]
        self.assertEqual(message["role"], "assistant")
        self.assertIn("Hello! I am a mock agent.", message["content"])

    def test_03_chat_completions_streaming(self):
        payload = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "say hello"}],
            "stream": True
        }
        
        # Read the stream
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as r:
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get("Cache-Control"), "no-cache")
            self.assertEqual(r.headers.get("Connection"), "keep-alive")
            self.assertEqual(r.headers.get("X-Accel-Buffering"), "no")
            
            chunks = []
            reasoning_chunks = []
            stream_order = []
            has_role_assistant = False
            first_chunk = True
            
            for line in r.iter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    chunk_data = json.loads(content)
                    delta = chunk_data["choices"][0]["delta"]
                    
                    if first_chunk:
                        first_chunk = False
                        if delta.get("role") == "assistant":
                            has_role_assistant = True
                            
                    if "content" in delta:
                        chunks.append(delta["content"])
                        stream_order.append("content")
                    if "reasoning_content" in delta:
                        reasoning_chunks.append(delta["reasoning_content"])
                        stream_order.append("reasoning")
            
            full_text = "".join(chunks)
            full_reasoning = "".join(reasoning_chunks)
            
            self.assertTrue(has_role_assistant, "First chunk should declare the assistant role")
            self.assertIn("Thinking about greeting...", full_reasoning)
            self.assertEqual("Hello! I am a mock agent.", full_text)
            self.assertLess(
                stream_order.index("reasoning"),
                stream_order.index("content"),
                msg="reasoning chunks must be streamed before content chunks",
            )

    def test_03b_streaming_reasoning_before_content_when_text_arrives_first(self):
        payload = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "out_of_order_thinking"}],
            "stream": True,
        }
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as r:
            self.assertEqual(r.status_code, 200)
            order = []
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                content = line[6:]
                if content == "[DONE]":
                    break
                delta = json.loads(content)["choices"][0]["delta"]
                if "reasoning_content" in delta:
                    order.append("reasoning")
                if "content" in delta:
                    order.append("content")
        self.assertTrue(order.index("reasoning") < order.index("content"))

    def test_04_session_resumption_prefix_matching(self):
        session_id = "test-resumption-sess"
        
        # Turn 1
        payload1 = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "first message"}],
            "user": session_id,
            "stream": False
        }
        resp1 = self.client.post("/v1/chat/completions", json=payload1)
        self.assertEqual(resp1.status_code, 200)
        
        # Turn 2
        payload2 = {
            "model": "composer-2.5",
            "messages": [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "Hello! I am a mock agent."},
                {"role": "user", "content": "second message"}
            ],
            "user": session_id,
            "stream": False
        }
        resp2 = self.client.post("/v1/chat/completions", json=payload2)
        self.assertEqual(resp2.status_code, 200)
        
        # Verify the session is kept and matched
        self.assertIn(session_id, cursor_bridge_server.sessions)
        session = cursor_bridge_server.sessions[session_id]
        self.assertEqual(len(session.messages), 4) # user, assistant, user, assistant

    def test_05_tool_calling_bridge(self):
        session_id = f"tool-sess-{uuid.uuid4().hex[:6]}"
        
        # Establish dynamic MCP tools list
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "description": "Get database info for a user ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"}
                        },
                        "required": ["user_id"]
                    }
                }
            }
        ]
        
        # Start SSE subscription in a separate background thread
        # This simulates the Cursor Agent connecting to our local SSE server
        sse_connected = threading.Event()
        def listen_sse():
            sse_url = f"/mcp/{session_id}/sse"
            # Establish connection
            with self.client.stream("GET", sse_url) as r:
                sse_connected.set()
                for line in r.iter_lines():
                    if not line:
                        continue
                    # Just drain the stream
                    pass
                    
        sse_thread = threading.Thread(target=listen_sse, daemon=True)
        sse_thread.start()
        sse_connected.wait(timeout=2.0)
        
        # Turn 1: Trigger the tool calling mock
        payload1 = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "trigger_tool user info"}],
            "user": session_id,
            "tools": tools,
            "stream": True
        }
        
        tool_call_id = None
        
        # Read the stream. We expect it to yield a tool call chunk and then end.
        with self.client.stream("POST", "/v1/chat/completions", json=payload1) as r:
            self.assertEqual(r.status_code, 200)
            for line in r.iter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    chunk_data = json.loads(content)
                    delta = chunk_data["choices"][0]["delta"]
                    if "tool_calls" in delta:
                        tool_call_id = delta["tool_calls"][0]["id"]
                        
        self.assertIsNotNone(tool_call_id, "Should have received a tool call ID from completions stream")
        
        # Verify future is pending on the session
        session = cursor_bridge_server.sessions[session_id]
        self.assertIn(tool_call_id, session.tool_call_futures)
        
        # Turn 2: Client returns the tool result
        payload2 = {
            "model": "composer-2.5",
            "messages": [
                {"role": "user", "content": "trigger_tool user info"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "get_user_info",
                                "arguments": '{"user_id": "123"}'
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "{\"status\": \"active\", \"name\": \"John Doe\"}"
                }
            ],
            "user": session_id,
            "stream": False
        }
        
        resp2 = self.client.post("/v1/chat/completions", json=payload2)
        self.assertEqual(resp2.status_code, 200)
        data = resp2.json()
        
        message = data["choices"][0]["message"]
        self.assertIn("Tool returned:", message["content"])
        self.assertIn("John Doe", message["content"])

    def test_06_duplicate_request_cached_redelivery(self):
        session_id = f"dup-sess-{uuid.uuid4().hex[:6]}"
        payload = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "say hello duplicate test"}],
            "user": session_id,
            "stream": False
        }
        
        # First request (normal execution)
        resp1 = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        content1 = data1["choices"][0]["message"]["content"]
        
        # Second request (duplicate/prefix-matched with 0 new messages)
        resp2 = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        content2 = data2["choices"][0]["message"]["content"]
        
        self.assertEqual(content1, content2, "Duplicate request should return the exact same cached content")
        
        # Test duplicate request with streaming
        payload_stream = {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "say hello duplicate test"}],
            "user": session_id,
            "stream": True
        }
        
        with self.client.stream("POST", "/v1/chat/completions", json=payload_stream) as r:
            self.assertEqual(r.status_code, 200)
            chunks = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    chunk_data = json.loads(content)
                    delta = chunk_data["choices"][0]["delta"]
                    if "content" in delta:
                        chunks.append(delta["content"])
            full_text = "".join(chunks)
            self.assertEqual(content1, full_text, "Streaming duplicate request should return the exact same cached content")

if __name__ == "__main__":
    unittest.main()
