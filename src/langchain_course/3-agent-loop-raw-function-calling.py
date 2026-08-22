import json

from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "openai/gpt-oss-20b"
groq_client = Groq()


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
    Available Tiers : bronze , silver , gold
    """

    print(
        f">> Executing a apply_discount(price='{price}', discount_tier={discount_tier})"
    )
    discount_percentage = {"gold": 23, "bronze": 5, "silver": 12}

    final_price = round(
        price - (price * discount_percentage.get(discount_tier.lower()) / 100), 2
    )

    return final_price


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of the product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product",
                    }
                },
                "required": ["product_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount tier to the price and return the final price.Available Tiers : bronze , silver , gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "Price of the product"},
                    "discount_tier": {
                        "type": "string",
                        "description": "Name of the discount tier",
                    },
                },
                "required": ["price", "discount_tier"],
                "additionalProperties": False,
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Groq call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Groq Chat", run_type="llm")
def groq_chat_traced(messages):
    return groq_client.chat.completions.create(
        model=MODEL, tools=tools_for_llm, messages=messages
    )


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question : {question}")
    print("*" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iterations} :")

        response = groq_chat_traced(messages=messages)
        ai_message = response.choices[0].message
        tool_calls = ai_message.tool_calls or []
        # If tool_calls is empty meaning LLM return the answer
        if not tool_calls:
            print(f"Final Answer : ", {ai_message.content})
            return ai_message.content

        # Process only the FIST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        print(f"  [Toll Selected] {tool_name} with args: {tool_args}")

        # Get the tool return by AI from tools dictionary
        tool_to_use = tools_dict.get(tool_name)

        # Check if the AI given tool is exist or not
        if tool_to_use is None:
            raise ValueError(f" Tool {tool_name} not found")

        # Call the tool with args
        observations = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observations}")

        messages.append(
            {
                "role": "assistant",
                "content": ai_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(observations),
            }
        )

    print(" ERROR: Max iteration reached without the final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")

    result = run_agent("What is the price of a laptop after applying a gold discount")
