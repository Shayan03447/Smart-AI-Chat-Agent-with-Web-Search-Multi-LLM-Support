import os
from dotenv import load_dotenv
load_dotenv()


# Set Api keys
tavily_api_key= os.getenv("TAVILY_API_KEY")
openai_api_key= os.getenv("OPENAI_API_KEY")
groq_api_key= os.getenv("GROQ_API_KEY")

# Setup LLM & Tools
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage


openai_model=ChatOpenAI(model="gpt-4o-mini")
groq_model=ChatGroq(model="llama-3.3-70b-versatile")
search_tool=TavilySearchResults(max_results=2)

system_prompt="Act as ai chatbot who is smart and friendlt"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm=ChatGroq(model=llm_id)
    elif provider =="OpenAI":
        llm=ChatOpenAI(model=llm_id)
    else:
        raise ValueError("Invalid provider")
    # tools and setup
    if allow_search:
        tools = [TavilySearchResults(max_results=2)]
    else:
        tools=[]
    # Agent
    agent=create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    state={"messages": [HumanMessage(content=query)]}
    response=agent.invoke(state)
    messages=response.get("messages")

    ai_messages = []
    for message in messages:
        if isinstance(message, AIMessage):
            ai_messages.append(message.content)
    if ai_messages:
        return ai_messages[-1]
    else:
        return "Ai did not response"
    
# Example usage
system_prompt = "Act as a smart and friendly AI chatbot"
query = "Hello AI, can you introduce yourself?"
response = get_response_from_ai_agent(
    llm_id="gpt-4o-mini", 
    query=query, 
    allow_search=False, 
    system_prompt=system_prompt, 
    provider="OpenAI"
)

print("AI Response:", response)

        

