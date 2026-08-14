from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

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
llm = ChatOllama(model="minimax-m3:cloud")

# Define a list of tools to be used by the agent
tools = [TavilySearch()]

# Create the agent with the LLM and tools
agent = create_agent(model=llm, tools=tools)


def main():
    # Example usage of the agent
    # user_input = "What is the weather in Mumbai, India?"
    user_input = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
    result = agent.invoke({"messages": HumanMessage(content=user_input)})
    print(result)


if __name__ == "__main__":
    main()
