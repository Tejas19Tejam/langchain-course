from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "llama-3.3-70b-versatile"


@tool
def get_product_price(product_name: str) -> float:
    """
    Look up the price of the product in the catalog.
    """
    print(f">> Executing a get_product_price(product_name='{product_name}')")
    prices = {"laptop": 1299.99, "headphones": 148, "keyboard": 12337}

    return prices.get(product_name, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to the price and return the final price.
    Available Tiers : Bronze , Silver , Gold
    """
    print(
        f">> Executing a apply_discount(price='{price}', discount_tier={discount_tier})"
    )
    discount_percentage = {"gold": 23, "bronze": 5, "silver": 12}

    final_price = round(
        price - (price * discount_percentage.get(discount_tier) / 100), 2
    )

    return final_price


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # Create a LLM instance with GROQ provider with compatible model
    llm = init_chat_model(f"GROQ:{MODEL}", temperature=0)
    # Provide a tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question : {question}")
    print("*" * 60)


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")

    result = run_agent("What is the price of a laptop after applying a gold discount")
