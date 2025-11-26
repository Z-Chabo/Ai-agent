from fastapi import FastAPI
from agentrunner.agentRun import run_agent
import uvicorn
# we need Cross Origin Ressource Sharing middleware to allow frontend to communicate with backend otherwise backend will block communication because both come fron different origins
#in general a middleware is defined as processes that need to happen before the api call from frontend reaches the end point 
#lol hisadha 
from fastapi.middleware.cors import CORSMiddleware
from typing import List
# BaseModel is used by FastApi to validate the data received from frontend against data structure defined and to transform json data from frontend into python object 
# AIMessage,HumanMessage and BaseMessage are essential for langchain agents to function properly BaseMessage is the parent of both AI and Human Message which allows us to store
# message of both types in the history_msgs
from imports.imports import BaseModel, Optional, HumanMessage, AIMessage, BaseMessage

origins = ["https://portfolio1-b7j.pages.dev", "http://localhost:5173", "https://z-bot-by-zeidan-2025-gzdxd8a9cjbmhjdj.canadacentral-01.azurewebsites.net"]

# This is the app instance that Vercel will look for and serve
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

@app.post("/aiAgent")
async def query_agent(request: QueryRequest):
   history_msgs:List[BaseMessage]=[]

   if request.history:
      for msg in request.history:
         if msg.type == 'ai':
            history_msgs.append(AIMessage(content=msg.content))
         else: # 'human' or None
            history_msgs.append(HumanMessage(content=msg.content))

   response=run_agent(request.query, history_msgs)

   return {"response":response.content}


if __name__ == "__main__":
    # This block is for local development only.
    # It will not run on Vercel.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    