from __future__ import annotations

from fastmcp import FastMCP

from .mcp_prompts import register_prompts
from .mcp_resources import register_resources
from .mcp_tools import ImageGenService, register_tools


def build_server() -> FastMCP:
    mcp = FastMCP("image-gen-mcp")
    service = ImageGenService()
    register_tools(mcp, service)
    register_resources(mcp, service)
    register_prompts(mcp)
    return mcp


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
