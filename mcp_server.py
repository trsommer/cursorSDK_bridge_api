import os
import sys
import logging
from typing import Optional

try:
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("Could not import FastMCP. Please ensure 'fastmcp' is installed.")

from cursor_sdk import AsyncClient, LocalAgentOptions

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cursor-mcp-server")

# Helper to load .env manually
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

# Initialize FastMCP Server
mcp = FastMCP("CursorAgentBridge")

@mcp.tool()
async def run_cursor_agent(prompt: str, cwd: Optional[str] = None) -> str:
    """Run a prompt task using the Cursor Agent locally on a workspace directory.
    
    Args:
        prompt: The task instruction for the agent to execute (e.g., 'fix bugs in src/main.py').
        cwd: The absolute path of the workspace. If not specified, defaults to CURSOR_WORKSPACE or current directory.
    """
    workspace = cwd or os.environ.get("CURSOR_WORKSPACE") or os.getcwd()
    workspace = os.path.abspath(workspace)
    api_key = os.environ.get("CURSOR_API_KEY")
    
    logger.info(f"Invoking Cursor Agent in '{workspace}' with prompt: '{prompt}'")
    
    if not api_key:
        logger.warning("CURSOR_API_KEY environment variable is not set. Execution may fail.")
        
    try:
        # Launch the local bridge connection
        async with await AsyncClient.launch_bridge(workspace=workspace) as client:
            # Create the agent
            async with await client.agents.create(
                model="composer-2.5",
                api_key=api_key,
                local=LocalAgentOptions(cwd=workspace)
            ) as agent:
                # Run the prompt
                run = await agent.send(prompt)
                # Wait for the run to complete
                result = await run.wait()
                
                if result.status == "finished":
                    logger.info(f"Agent execution completed successfully. Result size: {len(result.result)} chars.")
                    return result.result
                else:
                    err_msg = f"Agent completed with status: {result.status}. Output: {result.result}"
                    logger.error(err_msg)
                    return err_msg
    except Exception as e:
        logger.error(f"Error executing Cursor agent: {e}", exc_info=True)
        return f"Error executing Cursor agent: {str(e)}"

if __name__ == "__main__":
    mcp.run()
