from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

origins = ["https://portfolio1-b7j.pages.dev", "http://localhost:5173", "https://z-bot-by-zeidan-2025-gzdxd8a9cjbmhjdj.canadacentral-01.azurewebsites.net"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
class Message(BaseModel):
   content:str
   type: Optional[str] = "human"
class QueryRequest(BaseModel):
   query:str
   history:List[Message]=[]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Z-Bot AI Agent is running"}

@app.post("/aiAgent")
async def query_agent(request: QueryRequest):
   # Local import: Import the agent runner only when this function is called.
   # This prevents crashes on startup if environment variables are slow to load.
   from agentrunner.agentRun import run_agent
   history_msgs:List[BaseMessage]=[]

   if request.history:
      for msg in request.history:
         if msg.type == 'ai':
            history_msgs.append(AIMessage(content=msg.content))
         else: # 'human' or None
            history_msgs.append(HumanMessage(content=msg.content))

   response=run_agent(request.query, history_msgs)

   return {"response":response}

if __name__ == "__main__":
    # This block is for local development only.
    # It will not run on Vercel.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)