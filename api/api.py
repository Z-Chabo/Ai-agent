from fastapi import FastAPI
from agentrunner.agentRun import run_agent
import uuid
# we need Cross Origin Ressource Sharing middleware to allow frontend to communicate with backend otherwise backend will block communication because both come fron different origins
#in general a middleware is defined as processes that need to happen before the api call from frontend reaches the end point 
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sentence_transformers import SentenceTransformer
# BaseModel is used by FastApi to validate the data received from frontend against data structure defined and to transform json data from frontend into python object 
# AIMessage,HumanMessage and BaseMessage are essential for langchain agents to function properly BaseMessage is the parent of both AI and Human Message which allows us to store
# message of both types in the history_msgs
from imports.imports import BaseModel, Optional, HumanMessage, AIMessage, BaseMessage


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
# we use this to to be able to transform text into numerical vectors
model = SentenceTransformer("all-MiniLM-L6-v2")



#example of a cosine_similarity function
"""
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
"""

from langchain_core.embeddings import Embeddings
class MyEmbeddingFunction(Embeddings):
   def embed_documents(self, texts: List[str]) -> List[List[float]]:
   # this will be used to embbed texts (messages)
      return [model.encode(text).tolist() for text in texts]
   # this will be used in the similarity search function of vector_store
   def embed_query(self,text:str)->List[float]:
      return model.encode(text).tolist()

embedding_fn=MyEmbeddingFunction()
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="chat_history",
    embedding_function=embedding_fn,
    collection_metadata={"hnsw:space": "cosine"} # Tell Chroma to use cosine similarity
)




@app.post("/aiAgent")
async def query_agent(request: QueryRequest):
   history_msgs:List[BaseMessage]=[]

   # Add history to the vector store if it's not empty
   if request.history:
      # De-duplicate messages from the history before adding to the vector store because
      # Chroma's upsert can't handle duplicate IDs within the same list so we create a dictionary that has a key "msg.content" and a value "msg" (the object)
      # we don't need to worry about readding the same conversation history because Chroma's "upsert" function never readds the same text again 
      unique_messages_to_add = {}
      for msg in request.history:
         # "msg.content" is the key abd "msg" is the value (which is an object)
         unique_messages_to_add[msg.content] = msg
      
      if unique_messages_to_add:
         #creates a list of message texts
         history_texts = [msg.content for msg in unique_messages_to_add.values()]
         #creates a list of dictionaries {"type":"msg.type"}
         history_metadatas = [{'type': msg.type} for msg in unique_messages_to_add.values()]
         # Generate IDs from the de-duplicated messages
         history_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, msg.content)) for msg in unique_messages_to_add.values()]
         vector_store.add_texts(texts=history_texts, metadatas=history_metadatas, ids=history_ids)

   # 
   # k=2 means we are looking for the top 2 most relevant messages
   relevant_docs = vector_store.similarity_search(request.query, k=2)

   # Create history messages from the relevant documents
   for doc in relevant_docs:
      # Check the metadata to create the correct message type
      if doc.metadata.get('type') == 'ai':
         history_msgs.append(AIMessage(content=doc.page_content))
      else:
         # Default to HumanMessage if type is 'human' or not present
         history_msgs.append(HumanMessage(content=doc.page_content))

   response=run_agent(request.query, history_msgs)

   # Add the new query and its response to the vector store for future context
   vector_store.add_texts(texts=[request.query, response.content], metadatas=[{'type': 'human'}, {'type': 'ai'}], ids=[str(uuid.uuid4()), str(uuid.uuid4())])

   return {"response":response.content}
