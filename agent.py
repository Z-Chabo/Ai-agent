from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, AnyMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import List

zeidans_information="""Zeidan is 24 years right he is going to be 25 in 2026 . He loves playing chess as a hobby. He is extremly friendly and loves to help people around him
He is a canadian citizen of syrian heritage. He can speak English, Arabic,Aramaic and French. He is studying software engineering at concordia. He is going to enter his third year.
of software engineering studying
""" 
agent=None
checkpointer = InMemorySaver()

# Create the prompt template for the agent
SYSTEM_PROMPT = SystemMessage(f"""You are a personal assistant for a person named Zeidan. Your name is Z-Bot.

Here is the information you know about Zeidan:
{zeidans_information}

Use this information to answer questions about Zeidan. Be friendly and helpful.
""")

checkpointer=InMemorySaver()


def getagent():
    global agent
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        max_retries=0  # Don't retry on errors
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        checkpointer=checkpointer)
    

def run_agent(query: str, history: List[AnyMessage]):
    """Invokes the agent with a query and conversation history."""
    global agent
    if agent is None:
        getagent()
    config = {"configurable": {"thread_id": "main_thread"}}
    # The agent returns a list of messages. We are interested in the last one.
    response = agent.invoke({"messages": history + [("user", query)]}, config=config)
    return response['messages'][-1].content
