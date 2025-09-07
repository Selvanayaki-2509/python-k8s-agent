import os
import re
import json
import shutil
import logging
import asyncio
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

# --- MCP client (correct API) ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
log = logging.getLogger("app")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in .env or export it.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

app = Flask(__name__)

# Trigger only when user clearly refers to their/local cluster/minikube.
CLUSTER_TRIGGER_RE = re.compile(
    r"\b(my|local|minikube)\b.*\b(cluster|k8s|kubernetes|pod|pods|node|nodes|namespace|deployment|service)\b",
    re.IGNORECASE | re.DOTALL,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def should_use_mcp(user_text: str) -> bool:
    """Route only if it's obviously about *your/local* cluster."""
    text = user_text.strip().lower()
    return bool(CLUSTER_TRIGGER_RE.search(text)) or ("minikube" in text)

def mcp_binary_in_path() -> str | None:
    """Return the path to the MCP server binary if available."""
    # If you installed via npm/bun globally, the executable should be on PATH
    return shutil.which("mcp-server-kubernetes")

def _content_items_to_text(items) -> str:
    """Best-effort conversion of MCP tool response content to plain text."""
    parts = []
    try:
        # Newer SDK: result.content is a list of content items
        if hasattr(items, "content"):
            items = items.content
        if isinstance(items, list):
            for it in items:
                # dict payload
                if isinstance(it, dict):
                    if it.get("type") == "text":
                        parts.append(it.get("text", ""))
                    else:
                        parts.append(json.dumps(it, indent=2))
                # object with attributes
                elif hasattr(it, "type"):
                    if getattr(it, "type", None) == "text":
                        parts.append(getattr(it, "text", ""))
                    else:
                        parts.append(str(it))
                else:
                    parts.append(str(it))
        else:
            parts.append(str(items))
    except Exception as e:
        parts.append(f"[debug: failed to parse content: {e}]")
        parts.append(str(items))
    text = "\n".join(p for p in parts if p is not None)
    # Keep results readable in HTML <pre>
    return text.strip() or "(no content)"

async def _open_mcp_session():
    """
    Open MCP server over stdio correctly and return (exit_stack, session).
    Caller must `await exit_stack.aclose()` or use 'async with' for cleanup.
    """
    params = StdioServerParameters(
        command="mcp-server-kubernetes",
        args=[],           # you can pass ["--some-flag"] if needed
        env={              # inherit env; forward KUBECONFIG if you use custom path
            **os.environ,
        },
    )
    exit_stack = AsyncExitStack()
    read, write = await exit_stack.enter_async_context(stdio_client(params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return exit_stack, session

async def query_kubernetes(prompt: str) -> str:
    """
    Connect to MCP server and run a safe connectivity check + a simple action
    inferred from the prompt (pods/namespaces/context).
    """
    # Quick preflight logs
    log.info("MCP binary path: %s", mcp_binary_in_path() or "(not found on PATH)")
    log.info("KUBECONFIG: %s", os.getenv("KUBECONFIG", "~/.kube/config (default)"))

    try:
        exit_stack, session = await _open_mcp_session()
    except FileNotFoundError as e:
        return f"❌ Could not start mcp-server-kubernetes (not on PATH?): {e}"
    except Exception as e:
        return f"❌ Failed to start MCP session: {e}"

    try:
        # List tools (debug)
        tools_resp = await asyncio.wait_for(session.list_tools(), timeout=15)
        tool_names = [t.name for t in getattr(tools_resp, "tools", [])]
        log.info("MCP tools available: %s", tool_names)

        # Always do a quick ping first (verifies server & basic wiring)
        if "ping" in tool_names:
            log.info("Skipping MCP ping (server doesn't return content).")

        # Try to get current context info if supported (nice for UI/debug)
        ctx_txt = ""
        if "kubectl_context" in tool_names:
            try:
                ctx_res = await asyncio.wait_for(session.call_tool("kubectl_context", {}), timeout=20)
                ctx_txt = _content_items_to_text(ctx_res)
            except Exception as e:
                ctx_txt = f"(kubectl_context failed: {e})"

        # Very light intent mapping for a first test:
        p = prompt.lower()
        if "pod" in p and "kubectl_get" in tool_names:
            # extract namespace if user wrote "namespace X"
            ns_match = re.search(r"\bnamespace\s+([a-z0-9-]+)\b", p)
            namespace = ns_match.group(1) if ns_match else "default"
            args = {"resourceType": "pods", "namespace": namespace}
            try:
                res = await asyncio.wait_for(session.call_tool("kubectl_get", args), timeout=30)
                body = _content_items_to_text(res)
                header = f"Current context:\n{ctx_txt}\n\n" if ctx_txt else ""
                return header + f"Pods in namespace '{namespace}':\n{body}"
            except Exception as e:
                return f"⚠️ kubectl_get pods failed: {e}\n\nContext:\n{ctx_txt}"

        if "namespace" in p and "kubectl_get" in tool_names:
            try:
                res = await asyncio.wait_for(
                    session.call_tool("kubectl_get", {"resourceType": "namespaces"}), timeout=30
                )
                body = _content_items_to_text(res)
                header = f"Current context:\n{ctx_txt}\n\n" if ctx_txt else ""
                return header + f"Namespaces:\n{body}"
            except Exception as e:
                return f"⚠️ kubectl_get namespaces failed: {e}\n\nContext:\n{ctx_txt}"

        # Fallback: if nothing specific, just prove we're connected
        proved = "✅ Connected to MCP Kubernetes server."
        if ctx_txt:
            proved += f"\nCurrent context:\n{ctx_txt}"
        # optionally list API resources to show connectivity
        if "list_api_resources" in tool_names:
            try:
                res = await asyncio.wait_for(session.call_tool("list_api_resources", {}), timeout=30)
                body = _content_items_to_text(res)
                proved += f"\n\nSome API resources:\n{body[:2000]}..."  # avoid flooding UI
            except Exception as e:
                proved += f"\n(list_api_resources failed: {e})"
        return proved

    finally:
        await exit_stack.aclose()

# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        return render_template("index.html", error="Please enter a prompt.", prompt=user_prompt)

    try:
        if should_use_mcp(user_prompt):
            # Route to MCP for cluster-specific prompts
            answer = asyncio.run(query_kubernetes(user_prompt))
        else:
            # General question → Gemini
            response = model.generate_content(user_prompt)
            # robust extraction
            answer = getattr(response, "text", None) or (
                response.candidates[0].content if getattr(response, "candidates", None) else str(response)
            )
        return render_template("index.html", prompt=user_prompt, answer=answer)
    except Exception as e:
        log.exception("Error handling /ask")
        return render_template("index.html", error=f"Error: {e}", prompt=user_prompt)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    user_prompt = data.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "prompt is required"}), 400
    try:
        if should_use_mcp(user_prompt):
            answer = asyncio.run(query_kubernetes(user_prompt))
        else:
            response = model.generate_content(user_prompt)
            answer = getattr(response, "text", None) or (
                response.candidates[0].content if getattr(response, "candidates", None) else str(response)
            )
        return jsonify({"answer": answer})
    except Exception as e:
        log.exception("Error handling /api/ask")
        return jsonify({"error": str(e)}), 500

# Simple debug endpoint you can hit in browser: http://127.0.0.1:5000/debug/mcp
@app.route("/debug/mcp", methods=["GET"])
def debug_mcp():
    try:
        out = {
            "mcp_binary": mcp_binary_in_path() or "(not found on PATH)",
            "KUBECONFIG": os.getenv("KUBECONFIG", "~/.kube/config (default)"),
        }
        # run a very short probe
        answer = asyncio.run(query_kubernetes("probe"))
        out["probe"] = answer[:4000]
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
