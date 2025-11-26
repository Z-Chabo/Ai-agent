from fastapi import FastAPI
from agentrunner.agentRun import run_agent
import uuid
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

# Instantiate the embedding function using Google's service.
# This will automatically use the GOOGLE_API_KEY from your environment variables.
embedding_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="chat_history",
    embedding_function=embedding_fn,
    collection_metadata={"hnsw:space": "cosine"} # Tell Chroma to use cosine similarity
)

@app.post("/aiAgent")
async def query_agent(request: QueryRequest):
   history_msgs:List[BaseMessage]=[]

   if request.history:
      unique_messages_to_add = {msg.content: msg for msg in request.history}
      
      if unique_messages_to_add:
         history_texts = [msg.content for msg in unique_messages_to_add.values()]
         history_metadatas = [{'type': msg.type} for msg in unique_messages_to_add.values()]
         history_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, msg.content)) for msg in unique_messages_to_add.values()]
         vector_store.add_texts(texts=history_texts, metadatas=history_metadatas, ids=history_ids)

   relevant_docs = vector_store.similarity_search(request.query, k=2)

   for doc in relevant_docs:
      if doc.metadata.get('type') == 'ai':
         history_msgs.append(AIMessage(content=doc.page_content))
      else:
         history_msgs.append(HumanMessage(content=doc.page_content))

   response=run_agent(request.query, history_msgs)

   vector_store.add_texts(texts=[request.query, response.content], metadatas=[{'type': 'human'}, {'type': 'ai'}], ids=[str(uuid.uuid4()), str(uuid.uuid4())])

   return {"response":response.content}


if __name__ == "__main__":
    # This block is for local development only.
    # It will not run on Vercel.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    