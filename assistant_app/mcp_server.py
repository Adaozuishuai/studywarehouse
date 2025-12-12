"""
Minimal Model Context Protocol (MCP) server exposing the assistant tools.

This module wires the existing retrieval and search capabilities into an MCP
server so external clients can call them as standardized tools.
"""
from __future__ import annotations

from langchain_community.tools.tavily_search import TavilySearchResults
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource

from .config import settings
from .data import build_retriever


async def _panda_resource() -> Resource:
    """Provide metadata describing the underlying panda corpus."""

    return Resource(
        uri=settings.source_url,
        mimeType="text/html",
        description="Primary source describing panda characteristics.",
    )


def create_mcp_server():
    """Create an MCP server exposing internet search and panda retrieval."""

    retriever = build_retriever(settings)
    search = TavilySearchResults(max_results=settings.tavily_results)

    server = FastMCP("assistant-mcp")

    @server.tool()
    async def tavily_search(query: str) -> str:
        """进行互联网搜索并返回结果"""

        return search.run(query)

    @server.tool()
    async def wiki_panda_search(query: str) -> str:
        """根据向量数据库内容检索大熊猫相关信息。"""

        results = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in results])

    @server.resource()
    async def panda_wiki_source() -> Resource:
        """Expose the configured panda wiki URL as a resource for MCP clients."""

        return await _panda_resource()

    return server


async def serve(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run the MCP server."""

    server = create_mcp_server()
    await server.serve(host=host, port=port)


__all__ = ["create_mcp_server", "serve"]


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
