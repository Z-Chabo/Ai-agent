from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from agentTools.ToolsList import Tools



# Initialize the Google model.
# Make sure you have the GOOGLE_API_KEY environment variable set in your .env file.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

SYSTEM_MESSAGE = (
    "You are Z-Bot, Zeidan's helpful assistant. Start the conversation by introducing yourself as Z-Bot and what you can do. "
    "To answer any question about Zeidan, you must use the 'get_zeidans_info' tool to retrieve the information. "
    "Your introduction should be something like: \"Hi, I'm Z-Bot, Zeidan's personal AI assistant. I can answer questions about him. What would you like to know?\""
)

agent = create_agent(llm, Tools, system_prompt=SYSTEM_MESSAGE)