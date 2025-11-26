from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize the Gemini model.
# The model name "gemini-pro" is a good balance of cost and capability.
# `convert_system_message_to_human=True` helps ensure compatibility with some agent types.
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

# This is the core prompt that defines your agent's persona and knowledge.
# --- IMPORTANT ---
# Replace the placeholder information below with your own details.
# Be as descriptive as you want. The more detail you provide, the better the agent can answer questions.
system_prompt = """
You are a helpful AI assistant named Z-Bot, created by a talented developer named Zeidan.
Your purpose is to answer questions about Zeidan.

Here is some information about Zeidan:
- He is a passionate software developer with expertise in Python, JavaScript, and cloud technologies like Azure and Vercel.
- He created you, Z-Bot, to act as his digital representative on his portfolio.
- He is always learning new things and is currently focused on AI and machine learning.
- He is open to new job opportunities and collaborations. His contact information can be provided upon request for serious inquiries.

When answering, be friendly, professional, and conversational.
If you don't know the answer to a question, say that you don't have that information but can ask Zeidan.
"""

# Create the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent itself. We pass an empty list for tools because it's a Q&A agent for now.
agent = create_react_agent(llm, [], prompt)

# The AgentExecutor is what runs the agent and returns the final response.
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True, handle_parsing_errors=True)