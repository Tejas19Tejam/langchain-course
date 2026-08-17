from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent to answer a question."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for the response returned by the agent."""

    answer: str = Field(description="The agent answer to the query.")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used by the agent to answer the query.",
    )


# tavily = TavilyClient()


# Create a custom tool for searching the web using Tavily
# @tool
# def search(query: str) -> str:
#     """
#     Search the web for a query.
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.

#     """
#     response = tavily.search(query=query)
#     return f"Search results for '{query}': {response}"


# Define the LLM instance
# llm = ChatOpenAI()
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

# Define a list of tools to be used by the agent
tools = [TavilySearch()]

# Create the agent with the LLM and tools
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    # Example usage of the agent
    # user_input = "What is the weather in Mumbai, India?"
    user_input = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
    result = agent.invoke({"messages": HumanMessage(content=user_input)})
    print(result)


if __name__ == "__main__":
    main()
