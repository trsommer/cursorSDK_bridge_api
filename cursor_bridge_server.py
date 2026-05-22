import os
import sys
import json
import uuid
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from cursor_sdk import (
    AsyncClient,
    AsyncAgent,
    AgentOptions,
    LocalAgentOptions,
    HttpMcpServerConfig,
    SendOptions,
    ModelSelection,
    ModelParameterValue
)

try:
    from cursor_sdk.events import (
        TextDeltaUpdate,
        ThinkingDeltaUpdate,
        ToolCallStartedUpdate,
        TurnEndedUpdate
    )
except ImportError:
    from cursor_sdk import (
        TextDeltaUpdate,
        ThinkingDeltaUpdate,
        ToolCallStartedUpdate,
        TurnEndedUpdate
    )

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cursor-bridge")

# Helper to load .env manually (zero external dependencies)
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        logger.info(f"Loading environment from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

load_env()

# Server Settings
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# State Structures
class AgentSession:
    def __init__(self, session_id: str, client: AsyncClient, agent: AsyncAgent, workspace_path: str):
        self.session_id = session_id
        self.client = client
        self.agent = agent
        self.workspace_path = workspace_path
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.active_run: Optional[Any] = None
        self.message_generator: Optional[Any] = None
        self.tool_call_futures: Dict[str, asyncio.Future] = {}
        self.sse_queue: asyncio.Queue = asyncio.Queue()
        self.completion_queue: asyncio.Queue = asyncio.Queue()

sessions: Dict[str, AgentSession] = {}
clients: Dict[str, AsyncClient] = {}

# FastAPI App
app = FastAPI(title="Cursor SDK OpenAI Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper Functions
def is_message_prefix(stored_messages: List[Dict[str, Any]], incoming_messages: List[Dict[str, Any]]) -> bool:
    # Compare only user messages to avoid issues with assistant formatting differences (like thinking tags)
    if len(stored_messages) > len(incoming_messages):
        return False
    stored_users = [m for m in stored_messages if m.get("role") == "user"]
    incoming_users = [m for m in incoming_messages if m.get("role") == "user"]
    if len(stored_users) == 0:
        return False
    if len(stored_users) > len(incoming_users):
        return False
    for msg_s, msg_i in zip(stored_users, incoming_users):
        if msg_s.get("content") != msg_i.get("content"):
            return False
    return True

def extract_workspace_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    import re
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)
                
            # Try to find workspace root folder or working directory inside env tags
            env_match = re.search(r"<env>(.*?)</env>", content, re.DOTALL)
            if env_match:
                env_text = env_match.group(1)
                fallback = None
                for line in env_text.splitlines():
                    line = line.strip()
                    if line.startswith("Workspace root folder:"):
                        return line.split(":", 1)[1].strip()
                    elif line.startswith("Working directory:"):
                        fallback = line.split(":", 1)[1].strip()
                if fallback:
                    return fallback
                    
            # Fallback: search anywhere in the system prompt
            for line in content.splitlines():
                line = line.strip()
                if "Workspace root folder:" in line:
                    parts = line.split("Workspace root folder:", 1)
                    if len(parts) > 1 and parts[1].strip():
                        return parts[1].strip()
                elif "Working directory:" in line:
                    parts = line.split("Working directory:", 1)
                    if len(parts) > 1 and parts[1].strip():
                        return parts[1].strip()
    return None

VALID_CURSOR_MODELS = {
    "default", "composer-2.5", "composer-2", "gpt-5.5", "gpt-5.3-codex",
    "claude-sonnet-4-6", "claude-opus-4-7", "grok-build-0.1", "gpt-5.4",
    "claude-opus-4-6", "claude-opus-4-5", "gpt-5.2", "gemini-3.1-pro",
    "gpt-5.4-mini", "gpt-5.4-nano", "claude-haiku-4-5", "grok-4.3",
    "claude-sonnet-4-5", "gpt-5.2-codex", "gpt-5.1-codex-max", "gpt-5.1",
    "gemini-3-flash", "gemini-3.5-flash", "gpt-5.1-codex-mini", "claude-sonnet-4",
    "gpt-5-mini", "gemini-2.5-flash", "kimi-k2.5"
}

def map_model(openai_model: str) -> Any:
    model_id = openai_model
    params = []
    
    is_slow = False
    if "-slow" in model_id.lower() or ":slow" in model_id.lower() or "-standard" in model_id.lower():
        is_slow = True
        model_id = model_id.replace("-slow", "").replace(":slow", "").replace("-standard", "")
        
    is_fast = False
    if "-fast" in model_id.lower() or ":fast" in model_id.lower():
        is_fast = True
        model_id = model_id.replace("-fast", "").replace(":fast", "")

    cursor_id = "composer-2.5"
    if model_id in VALID_CURSOR_MODELS:
        cursor_id = model_id
    else:
        model_lower = model_id.lower()
        matched = False
        for m in VALID_CURSOR_MODELS:
            if m in model_lower:
                cursor_id = m
                matched = True
                break
        if not matched:
            if "gpt-4o" in model_lower or "sonnet" in model_lower:
                cursor_id = "claude-sonnet-4-6"
            elif "opus" in model_lower:
                cursor_id = "claude-opus-4-7"
            elif "gemini" in model_lower:
                cursor_id = "gemini-3.1-pro"
            elif "composer-2" in model_lower:
                cursor_id = "composer-2"
            else:
                cursor_id = "composer-2.5"

    if is_slow:
        params.append(ModelParameterValue(id="fast", value="false"))
    elif is_fast:
        params.append(ModelParameterValue(id="fast", value="true"))

    if params:
        return ModelSelection(id=cursor_id, params=params)
    return cursor_id

async def get_client_for_workspace(workspace: str) -> AsyncClient:
    if workspace not in clients:
        logger.info(f"Launching Cursor bridge client for workspace: {workspace}")
        os.makedirs(workspace, exist_ok=True)
        client = await AsyncClient.launch_bridge(workspace=workspace)
        clients[workspace] = client
    return clients[workspace]

def format_chunk(
    session_id: str,
    model: str,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    role: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None
) -> str:
    chunk = {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    }
    if role is not None:
        chunk["choices"][0]["delta"]["role"] = role
    if content is not None:
        chunk["choices"][0]["delta"]["content"] = content
    if reasoning_content is not None:
        chunk["choices"][0]["delta"]["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
    return f"data: {json.dumps(chunk)}\n\n"

async def run_waiter(run, session: AgentSession):
    try:
        result = await run.wait()
        logger.info(f"Run completed with status: {result.status}")
        await session.completion_queue.put({"type": "done", "result": result})
    except Exception as e:
        logger.error(f"Error in run waiter: {e}", exc_info=True)
        await session.completion_queue.put({"type": "error", "error": str(e)})

# MCP SSE Transport Endpoints
@app.get("/mcp/{session_id}/sse")
async def mcp_sse_endpoint(session_id: str, request: Request):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    base_url = str(request.base_url).rstrip("/")
    messages_url = f"{base_url}/mcp/{session_id}/messages"
    
    logger.info(f"Establishing MCP SSE connection for session {session_id}")
    
    async def sse_event_generator():
        # Yield the endpoint event first
        yield f"event: endpoint\ndata: {messages_url}\n\n"
        
        # Clear any existing elements in the queue
        while not session.sse_queue.empty():
            session.sse_queue.get_nowait()
            
        while True:
            try:
                msg = await session.sse_queue.get()
                if msg is None:
                    break
                yield f"event: message\ndata: {json.dumps(msg)}\n\n"
            except asyncio.CancelledError:
                break
                
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(sse_event_generator(), media_type="text/event-stream", headers=headers)

@app.post("/mcp/{session_id}/messages")
async def mcp_messages_endpoint(session_id: str, request: Request):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = sessions[session_id]
    body = await request.json()
    
    method = body.get("method")
    msg_id = body.get("id")
    
    logger.debug(f"MCP Received method {method} with ID {msg_id}")
    
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "cursor-bridge-mcp",
                    "version": "1.0.0"
                }
            }
        }
        await session.sse_queue.put(resp)
        return resp
        
    elif method == "notifications/initialized":
        return Response(status_code=200)
        
    elif method == "tools/list":
        # Convert OpenAI tools to MCP tools
        mcp_tools = []
        for tool in session.tools:
            if tool.get("type") == "function":
                f = tool.get("function", {})
                mcp_tools.append({
                    "name": f.get("name"),
                    "description": f.get("description", ""),
                    "inputSchema": f.get("parameters", {"type": "object", "properties": {}})
                })
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": mcp_tools
            }
        }
        await session.sse_queue.put(resp)
        return resp
        
    elif method == "tools/call":
        params = body.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"MCP Intercepted tool call: {name} with args: {arguments}")
        
        # Generate an OpenAI compatible tool call ID
        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        
        # Suspend Cursor agent via future
        fut = asyncio.Future()
        session.tool_call_futures[tool_call_id] = fut
        
        # Queue the tool call for the OpenAI client completions stream
        await session.completion_queue.put({
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": arguments
        })
        
        # Wait for the client to return the result
        result_content = await fut
        logger.info(f"Resuming tool call {name} ({tool_call_id}) with result: {result_content}")
        
        # Return tool result back to the Cursor Agent
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": str(result_content)
                    }
                ]
            }
        }
        await session.sse_queue.put(resp)
        return resp
        
    return Response(status_code=200)

# OpenAI API Endpoints
@app.get("/v1/models")
async def list_models():
    cursor_models = [
        "default", "composer-2.5", "composer-2", "gpt-5.5", "gpt-5.3-codex",
        "claude-sonnet-4-6", "claude-opus-4-7", "grok-build-0.1", "gpt-5.4",
        "claude-opus-4-6", "claude-opus-4-5", "gpt-5.2", "gemini-3.1-pro",
        "gpt-5.4-mini", "gpt-5.4-nano", "claude-haiku-4-5", "grok-4.3",
        "claude-sonnet-4-5", "gpt-5.2-codex", "gpt-5.1-codex-max", "gpt-5.1",
        "gemini-3-flash", "gemini-3.5-flash", "gpt-5.1-codex-mini", "claude-sonnet-4",
        "gpt-5-mini", "gemini-2.5-flash", "kimi-k2.5"
    ]
    data = []
    for m in cursor_models:
        data.append({
            "id": m,
            "object": "model",
            "created": 1710000000,
            "owned_by": "cursor"
        })
        if m in ("composer-2.5", "composer-2", "gpt-5.5", "gpt-5.3-codex", "claude-opus-4-7", "gpt-5.4", "claude-opus-4-6", "gpt-5.2", "gpt-5.2-codex", "gpt-5.1-codex-max"):
            data.append({
                "id": f"{m}-slow",
                "object": "model",
                "created": 1710000000,
                "owned_by": "cursor"
            })
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_workspace_path: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    body = await request.json()
    
    # Debug log request headers and body
    try:
        req_log = {
            "timestamp": time.time(),
            "headers": dict(request.headers),
            "body": body
        }
        ua = request.headers.get("user-agent", "").lower()
        if "opencode" in ua:
            log_path = "/Users/trsommer/Documents/cursorSDK_bridge_api/last_request_opencode.log"
        else:
            log_path = "/Users/trsommer/Documents/cursorSDK_bridge_api/last_request.log"
        with open(log_path, "w") as f:
            json.dump(req_log, f, indent=2)
    except Exception as e:
        logger.error(f"Error logging request: {e}")
    
    # 1. Extract API Key
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    if not api_key or not api_key.startswith("crsr_"):
        api_key = os.environ.get("CURSOR_API_KEY")
        
    incoming_messages = body.get("messages", [])

    # 2. Extract Workspace
    extracted_ws = extract_workspace_from_messages(incoming_messages)
    workspace = x_workspace_path or extracted_ws or os.environ.get("CURSOR_WORKSPACE") or os.getcwd()
    workspace = os.path.abspath(workspace)
    logger.info(
        f"Workspace Resolution -> Final: {workspace} | "
        f"x_workspace_path: {x_workspace_path} | "
        f"Extracted: {extracted_ws} | "
        f"CURSOR_WORKSPACE: {os.environ.get('CURSOR_WORKSPACE')} | "
        f"cwd: {os.getcwd()}"
    )
    
    # 3. Extract Session ID or Match
    session_id = x_session_id or body.get("user")
    model_name = body.get("model", "composer-2.5")
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    
    session: Optional[AgentSession] = None
    
    if session_id and session_id in sessions:
        session = sessions[session_id]
        logger.info(f"Reusing session {session_id} by explicit identifier.")
    else:
        # Fallback to prefix matching
        for s_id, s_sess in sessions.items():
            if is_message_prefix(s_sess.messages, incoming_messages) and s_sess.workspace_path == workspace:
                session = s_sess
                session_id = s_id
                logger.info(f"Reusing session {session_id} via prefix-matching.")
                break
                
    # If no session matches, create a new one
    if not session:
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        logger.info(f"Creating new session {session_id} for workspace {workspace}")
        
        client = await get_client_for_workspace(workspace)
        
        # Create Cursor Agent options
        cursor_model = map_model(model_name)
        
        options = AgentOptions(
            model=cursor_model,
            api_key=api_key,
            local=LocalAgentOptions(cwd=workspace),
            mcp_servers={
                "bridge_mcp": HttpMcpServerConfig(
                    url=f"http://127.0.0.1:{PORT}/mcp/{session_id}/sse",
                    type="sse"
                )
            }
        )
        agent = await client.agents.create(options)
        
        session = AgentSession(session_id, client, agent, workspace)
        sessions[session_id] = session

    # Update tools list for the session
    session.tools = tools
    
    # Process incoming history difference
    new_messages = incoming_messages[len(session.messages):]
    logger.info(f"Session {session_id}: processing {len(new_messages)} new messages out of {len(incoming_messages)} total.")
    
    has_tool_results = False
    
    # Send delta callback setup
    loop = asyncio.get_running_loop()
    def on_delta_callback(update):
        if type(update).__name__ in ("ToolCallStartedUpdate", "ToolCallCompletedUpdate", "PartialToolCallUpdate"):
            return
        loop.call_soon_threadsafe(
            session.completion_queue.put_nowait,
            {"type": "delta", "update": update}
        )
        
    send_options = SendOptions(
        on_delta=on_delta_callback
    )
    
    # Check for tool results
    for msg in new_messages:
        role = msg.get("role")
        if role in ("tool", "function"):
            tool_call_id = msg.get("tool_call_id")
            content = msg.get("content") or ""
            logger.info(f"Resolving future for tool call {tool_call_id}")
            if tool_call_id in session.tool_call_futures:
                fut = session.tool_call_futures.pop(tool_call_id)
                if not fut.done():
                    fut.set_result(content)
                    has_tool_results = True
            session.messages.append(msg)
            
    # If no tool results, check if the last message is a user message to trigger a run
    if not has_tool_results and len(new_messages) > 0:
        last_msg = new_messages[-1]
        if last_msg.get("role") == "user":
            prompt = last_msg.get("content") or ""
            logger.info(f"Starting new agent run in session {session_id} with prompt: '{prompt}'")
            
            # Reset completions queue
            while not session.completion_queue.empty():
                session.completion_queue.get_nowait()
            session.tool_call_futures.clear()
            
            # Trigger run
            run = await session.agent.send(prompt, send_options)
            session.active_run = run
            session.messages.append(last_msg)
            
            # Run waiter in background
            asyncio.create_task(run_waiter(run, session))
    cached_assistant_msg = None
    if len(new_messages) == 0:
        is_active = session.active_run and getattr(session.active_run, "status", None) == "running"
        if not is_active:
            # Find the last user message in incoming_messages
            last_user_msg = None
            for msg in reversed(incoming_messages):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg:
                # Find the matching user message in session.messages starting from the end
                for idx in range(len(session.messages) - 1, -1, -1):
                    msg = session.messages[idx]
                    if msg.get("role") == "user" and msg.get("content") == last_user_msg.get("content"):
                        # Check if there is an assistant response following it
                        if idx + 1 < len(session.messages):
                            next_msg = session.messages[idx + 1]
                            if next_msg.get("role") == "assistant":
                                cached_assistant_msg = next_msg
                                break

    # Core Completions Streaming generator
    async def completions_generator():
        # Check if we have a cached assistant message to re-deliver
        if cached_assistant_msg:
            logger.info(f"Re-delivering cached assistant response for session {session_id}")
            yield format_chunk(session_id, model_name, role="assistant")
            
            reasoning = cached_assistant_msg.get("reasoning_content")
            if reasoning:
                yield format_chunk(session_id, model_name, reasoning_content=reasoning)
                
            tool_calls = cached_assistant_msg.get("tool_calls")
            if tool_calls:
                yield format_chunk(session_id, model_name, tool_calls=tool_calls, finish_reason="tool_calls")
                yield "data: [DONE]\n\n"
                return
                
            content = cached_assistant_msg.get("content")
            if content:
                yield format_chunk(session_id, model_name, content=content)
                
            yield format_chunk(session_id, model_name, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        # If no new messages, no active run, and no cached assistant message, exit immediately
        is_active = session.active_run and getattr(session.active_run, "status", None) == "running"
        if len(new_messages) == 0 and not is_active:
            logger.info(f"Session {session_id}: no new messages, no active run, and no cached assistant message. Exiting immediately.")
            yield format_chunk(session_id, model_name, role="assistant")
            yield format_chunk(session_id, model_name, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        accumulated_text = ""
        accumulated_reasoning = ""
        yielded_tool_calls = []
        
        # Yield initial role delta chunk
        yield format_chunk(session_id, model_name, role="assistant")
        
        try:
            while True:
                event = await session.completion_queue.get()
                event_type = event.get("type")
                
                if event_type == "delta":
                    update = event["update"]
                    update_type_name = type(update).__name__
                    is_thinking = "ThinkingDelta" in update_type_name or isinstance(update, ThinkingDeltaUpdate)
                    is_text = "TextDelta" in update_type_name or isinstance(update, TextDeltaUpdate)
                    
                    if is_thinking:
                        text = getattr(update, "text", "")
                        if text:
                            yield format_chunk(session_id, model_name, reasoning_content=text)
                            accumulated_reasoning += text
                    elif is_text:
                        text = getattr(update, "text", "")
                        if text:
                            yield format_chunk(session_id, model_name, content=text)
                            accumulated_text += text
                            
                elif event_type == "tool_call":
                    tool_call_id = event["tool_call_id"]
                    tool_name = event["name"]
                    tool_args = event["arguments"]
                    arguments_str = json.dumps(tool_args)
                    
                    openai_tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments_str
                        }
                    }
                    yielded_tool_calls.append(openai_tool_call)
                    
                    yield format_chunk(
                        session_id,
                        model_name,
                        tool_calls=[{
                            "index": len(yielded_tool_calls) - 1,
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_str
                            }
                        }]
                    )
                    
                    # Look for additional tool calls (parallel tool execution support)
                    while True:
                        try:
                            next_event = await asyncio.wait_for(session.completion_queue.get(), timeout=0.1)
                            if next_event.get("type") == "tool_call":
                                tc_id = next_event["tool_call_id"]
                                tc_name = next_event["name"]
                                tc_args = next_event["arguments"]
                                tc_args_str = json.dumps(tc_args)
                                
                                openai_tc = {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_name,
                                        "arguments": tc_args_str
                                    }
                                }
                                yielded_tool_calls.append(openai_tc)
                                
                                yield format_chunk(
                                    session_id,
                                    model_name,
                                    tool_calls=[{
                                        "index": len(yielded_tool_calls) - 1,
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": tc_name,
                                            "arguments": tc_args_str
                                        }
                                    }]
                                )
                            else:
                                pass
                        except asyncio.TimeoutError:
                            break
                            
                    logger.info(f"Stream paused on tool calls: {[tc['function']['name'] for tc in yielded_tool_calls]}")
                    
                    assistant_msg = {
                        "role": "assistant",
                        "tool_calls": yielded_tool_calls
                    }
                    if accumulated_text:
                        assistant_msg["content"] = accumulated_text
                    if accumulated_reasoning:
                        assistant_msg["reasoning_content"] = accumulated_reasoning
                    session.messages.append(assistant_msg)
                    
                    yield format_chunk(session_id, model_name, finish_reason="tool_calls")
                    yield "data: [DONE]\n\n"
                    return
                    
                elif event_type == "done":
                    result = event.get("result")
                    final_text = getattr(result, "result", "") if result else ""
                    if final_text and len(final_text) > len(accumulated_text):
                        if final_text.startswith(accumulated_text):
                            remainder = final_text[len(accumulated_text):]
                        elif accumulated_text in final_text:
                            remainder = final_text.split(accumulated_text, 1)[1]
                        else:
                            remainder = final_text if not accumulated_text else ""
                        if remainder:
                            accumulated_text += remainder
                            yield format_chunk(session_id, model_name, content=remainder)
                        
                    assistant_msg = {
                        "role": "assistant",
                        "content": accumulated_text if accumulated_text else ""
                    }
                    if accumulated_reasoning:
                        assistant_msg["reasoning_content"] = accumulated_reasoning
                    session.messages.append(assistant_msg)
                    
                    yield format_chunk(session_id, model_name, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                    
                elif event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    yield f"data: {json.dumps({'error': {'message': error_msg}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                    
        except asyncio.CancelledError:
            logger.info("Completions client disconnected.")
            raise

    # Return stream or non-stream
    if stream:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        return StreamingResponse(completions_generator(), media_type="text/event-stream", headers=headers)
    else:
        full_content = ""
        full_reasoning_content = ""
        tool_calls = []
        finish_reason = "stop"
        
        async for chunk_str in completions_generator():
            if chunk_str.strip() == "data: [DONE]":
                continue
            if chunk_str.startswith("data: "):
                try:
                    chunk_data = json.loads(chunk_str[6:])
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            full_content += delta["content"]
                        if "reasoning_content" in delta:
                            full_reasoning_content += delta["reasoning_content"]
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                except Exception as e:
                    logger.error(f"Error parsing chunk: {e}")
                    
        message = {
            "role": "assistant"
        }
        if full_content:
            message["content"] = full_content
        else:
            message["content"] = None
            
        if full_reasoning_content:
            message["reasoning_content"] = full_reasoning_content
            
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": tc.get("function")
                }
                for tc in tool_calls
            ]
            
        return JSONResponse({
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down active sessions and bridge clients...")
    for session_id, session in list(sessions.items()):
        try:
            logger.info(f"Closing agent for session {session_id}")
            await session.agent.close()
        except Exception as e:
            logger.error(f"Error closing agent {session_id}: {e}")
            
    for workspace, client in list(clients.items()):
        try:
            logger.info(f"Closing bridge client for workspace {workspace}")
            await client.aclose()
        except Exception as e:
            logger.error(f"Error closing bridge client {workspace}: {e}")

if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
