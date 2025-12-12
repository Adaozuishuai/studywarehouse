"""
Command-line entry point to run the intelligent assistant locally.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import build_agent, run_sample_sessions
from .config import settings
from .data import build_retriever
from .tools import build_tools


def build_conversational_agent():
    """Build the end-to-end agent with tools and retrieval."""

    retriever = build_retriever(settings)
    tools = build_tools(retriever, settings)
    agent = build_agent(settings, tools)
    return agent, tools


def run_demo():
    """Run a short demo showing tool binding and memory."""

    agent, tools = build_conversational_agent()

    model_with_tools = agent.model.bind_tools(tools=tools)
    messages = [
        SystemMessage("你是一个乐于助人的企业级智能对话助手。"),
        HumanMessage("2025年12月10日天津天气怎么样？"),
    ]
    _ = model_with_tools.invoke(messages)

    thread_results = run_sample_sessions(
        agent=agent, thread_ids=settings.default_thread_ids, question="熊猫的特征"
    )

    for thread_id, result in thread_results.items():
        content = result["messages"][-1].content
        print(f"[Thread: {thread_id}] {content}")


if __name__ == "__main__":
    run_demo()
