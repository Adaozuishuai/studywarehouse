"""
Tooling used by the conversational agent.
"""
from __future__ import annotations

from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool

from .config import Settings


def build_tools(retriever, settings: Settings):
    """Create reusable tools for the agent."""

    search = TavilySearchResults(max_results=settings.tavily_results)

    @tool
    def search_tool(query: str) -> str:
        """进行互联网搜索并返回结果"""

        return search.run(query)

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="wiki_search",
        description="搜索维基百科",
    )

    @tool
    def wiki_panda_search(query: str) -> str:
        """根据向量数据库内容检索相关的维基百科知识。"""

        results = retriever_tool.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in results])

    return [search_tool, wiki_panda_search]
