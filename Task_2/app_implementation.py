"""Core logic: RAG (Voyage) + tools (web search, datetime) + MCP tools, driven by Claude."""
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

BASE = Path(__file__).parent
KB_PATH = BASE / "it_sector.txt"
MODEL = "claude-haiku-5-5"
EMBED_MODEL = "voyage-4-lite"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")

client = Anthropic()
voyage_client = voyageai.Client()

SYSTEM_PROMPT = """You are an IT Support assistant for company employees.
- Prefer the provided knowledge-base context for troubleshooting answers; if it is not relevant, say so and use web search if needed.
- Use create_ticket when the user wants to open/report a ticket (ask for missing details such as name and issue first).
- Use get_status_ticket when the user asks about an existing ticket.
- Use get_current_datetime when the current date or time is needed.
Be concise and friendly."""


# ---------------------------------------------------------------- RAG
def chunk_by_section(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?m)^## ", text) if c.strip()]


def generate_embedding(chunks, input_type="document"):
    is_list = isinstance(chunks, list)
    inp = chunks if is_list else [chunks]
    result = voyage_client.embed(inp, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]


class VectorIndex:
    """Minimal in-memory vector DB (cosine distance), as in the RAG notebook."""

    def __init__(self):
        self.vectors = None
        self.documents: list[str] = []

    def add_documents(self, documents: list[str], vectors):
        self.documents = documents
        m = np.array(vectors, dtype=float)
        self.vectors = m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query_vector, k=2):
        q = np.array(query_vector, dtype=float)
        q = q / np.linalg.norm(q)
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(self.documents[i], float(1 - sims[i])) for i in top]


def build_index() -> VectorIndex:
    chunks = chunk_by_section(KB_PATH.read_text(encoding="utf-8"))
    index = VectorIndex()
    index.add_documents(chunks, generate_embedding(chunks, "document"))
    return index


def retrieve_context(index: VectorIndex, query: str, k=2) -> str:
    results = index.search(generate_embedding(query, "query"), k=k)
    return "\n\n---\n\n".join(doc for doc, _ in results)


# ---------------------------------------------------------------- Tools
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(date_format)


DATETIME_TOOL = {
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted with a Python strftime format string.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "strftime format, e.g. '%H:%M:%S'.",
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    },
}

# Server-side tool: Anthropic executes it, no local code needed
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


# ---------------------------------------------------------------- MCP client
async def _mcp_call(fn):
    async with streamable_http_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def get_mcp_tools() -> list[dict]:
    """Fetch tools from the MCP server, converted to Claude tool schema. [] if server is down."""
    try:
        result = asyncio.run(_mcp_call(lambda s: s.list_tools()))
    except Exception:
        return []
    return [
        {"name": t.name, "description": t.description or t.name, "input_schema": t.input_schema}
        for t in result.tools
    ]


def call_mcp_tool(name: str, args: dict) -> str:
    result = asyncio.run(_mcp_call(lambda s: s.call_tool(name, args)))
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


# ---------------------------------------------------------------- Chat loop
def _text(message) -> str:
    return "\n".join(b.text for b in message.content if b.type == "text")


def chat(history: list[dict], user_text: str, index: VectorIndex, mcp_tools: list[dict]):
    """history: list of {"role","content": str}. Returns (answer, tool_log)."""
    context = retrieve_context(index, user_text)
    augmented = (
        f"<context>\n{context}\n</context>\n\n"
        f"Use the context above only if relevant.\n\nUser question: {user_text}"
    )
    work = [*history, {"role": "user", "content": augmented}]
    tools = [DATETIME_TOOL, WEB_SEARCH_TOOL, *mcp_tools]
    mcp_names = {t["name"] for t in mcp_tools}
    tool_log = []

    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT, messages=work, tools=tools
        )
        work.append({"role": "assistant", "content": resp.content})

        for b in resp.content:  # web search is server-side; just log it
            if b.type == "server_tool_use":
                tool_log.append(f"web_search({b.input})")

        if resp.stop_reason == "pause_turn":
            continue
        if resp.stop_reason != "tool_use":
            break

        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                if b.name == "get_current_datetime":
                    out = get_current_datetime(**b.input)
                elif b.name in mcp_names:
                    out = call_mcp_tool(b.name, b.input)
                else:
                    raise ValueError(f"Unknown tool {b.name}")
                err = False
            except Exception as e:
                out, err = f"Error: {e}", True
            tool_log.append(f"{b.name}({json.dumps(b.input)}) -> {out}")
            results.append(
                {"type": "tool_result", "tool_use_id": b.id, "content": str(out), "is_error": err}
            )
        work.append({"role": "user", "content": results})

    return _text(resp), tool_log