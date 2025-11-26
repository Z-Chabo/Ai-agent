from typing import List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from agent.agent import agent
def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution via LangGraph."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            # this sets the limit to how many times the agent can call the tools
            config={"recursion_limit": 50}
        )
        # Return the last AI message which is the answer
        return result["messages"][-1]
    except Exception as e:
        # Return error as an AI message so the conversation can continue
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")