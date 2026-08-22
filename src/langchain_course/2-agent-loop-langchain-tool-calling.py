from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "openai/gpt-oss-20b"


@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """
    Look up the price of the product in the catalog.
    """
    print(f">> Executing a get_product_price(product_name='{product_name}')")
    prices = {"laptop": 1299.99, "headphones": 148, "keyboard": 12337}

    return prices.get(product_name, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to the price and return the final price.
    Available Tiers : Bronze , Silver , Gold
    """
    print(
        f">> Executing a apply_discount(price='{price}', discount_tier={discount_tier})"
    )
    discount_percentage = {"gold": 23, "bronze": 5, "silver": 12}

    final_price = round(
        price - (price * discount_percentage.get(discount_tier.lower()) / 100), 2
    )

    return final_price


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # Create a LLM instance with GROQ provider with compatible model
    llm = init_chat_model(f"{MODEL}", model_provider="GROQ", temperature=0)
    # Provide a tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question : {question}")
    print("*" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iterations} :")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        # If tool_calls is empty meaning LLM return the answer
        if not tool_calls:
            print(f"Final Answer : ", {ai_message.content})
            return ai_message.content

        # Process only the FIST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")

        print(f"  [Toll Selected] {tool_name} with args: {tool_args}")

        # Get the tool return by AI from tools dictionary
        tool_to_use = tools_dict.get(tool_name)

        # Check if the AI given tool is exist or not
        if tool_to_use is None:
            raise ValueError(f" Tool {tool_name} not found")

        # Call the tool with args, using INVOKE method
        observations = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observations}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observations), tool_call_id=tool_id))

    print(" ERROR: Max iteration reached without the final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")

    result = run_agent("What is the price of a laptop after applying a gold discount")
