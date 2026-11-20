"""MCP server: create_ticket + get_status_ticket. Run: python mcp_server.py"""
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from mcp.server import MCPServer

DATA_FILE = Path(__file__).parent / "mcp_tool_data.json"

mcp_server = MCPServer(name="ITSupportServer", version="1.0.0")


def _load() -> list:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _save(tickets: list) -> None:
    DATA_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


@mcp_server.tool(
    name="create_ticket",
    description="Create a new IT support ticket. Returns the created ticket with its ticket_id.",
)
def create_ticket(employee: str, issue: str, description: str, priority: str = "Medium") -> str:
    priority = priority.capitalize()
    if priority not in ("Low", "Medium", "High"):
        priority = "Medium"
    tickets = _load()
    last = max((int(t["ticket_id"].split("-")[1]) for t in tickets), default=100)
    ticket = {
        "ticket_id": f"IT-{last + 1}",
        "employee": employee,
        "issue": issue,
        "description": description,
        "priority": priority,
        "status": "Open",
    }
    tickets.append(ticket)
    _save(tickets)
    return json.dumps(ticket)


@mcp_server.tool(
    name="get_status_ticket",
    description="Get the status and details of an IT support ticket by its ticket_id (e.g. IT-101).",
)
def get_status_ticket(ticket_id: str) -> str:
    for t in _load():
        if t["ticket_id"].lower() == ticket_id.strip().lower():
            return json.dumps(t)
    return f"Ticket {ticket_id} not found."


mcp_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/", mcp_app)

if __name__ == "__main__":
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=3000))
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    print("MCP server running at http://localhost:3000/mcp")
    asyncio.run(server.serve(), loop_factory=loop_factory)