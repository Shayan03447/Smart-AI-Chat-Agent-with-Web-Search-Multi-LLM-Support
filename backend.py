from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
from ai_agent import get_response_from_ai_agent
import uvicorn

# Schema for Validation
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]
app=FastAPI(title="LangGraph Agent")
@app.post("/chat")
def chat_endpoint(request: RequestState):

    """ API Endpoint to interact with the Chatbot using LangGraph and search tools.
        It dynamically selects the model specified in the request
    """

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"Error": "Invalid model name "}
    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_search
    system_prompt=request.system_prompt
    provider=request.model_provider
    
    # Getting the response
    response=get_response_from_ai_agent(llm_id=llm_id, query=query, allow_search=allow_search, system_prompt=system_prompt, provider=provider)
    return response

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=9999)

    