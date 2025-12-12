"""
Factory for the LangGraph-powered conversational agent.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

from .config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """Initialize the chat model used throughout the system."""

    return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)


def build_agent(settings: Settings, tools):
    """Create a tool-aware agent with memory."""

    settings.validate()
    llm = build_llm(settings)
    prompt = None
    if settings.langsmith_api_key:
        client = Client(api_key=settings.langsmith_api_key)
        prompt = client.pull_prompt("hwchase17/openai-functions-agent", include_model=True)

    agent_prompt = prompt if prompt is not None else None
    memory = MemorySaver()

    return create_agent(
        model=llm,
        tools=tools,
        prompt=agent_prompt,
        checkpointer=memory,
    )


def run_sample_sessions(agent, thread_ids, question: str):
    """Demonstrate how different threads maintain context independently."""

    results = {}
    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
        results[thread_id] = response
    return results
